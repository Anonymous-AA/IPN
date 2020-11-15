# Isometric Propagation Network
#
import os, re, sys, time, torch, random, argparse, math, json
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from PIL     import ImageFile
from copy    import deepcopy
from pathlib import Path
from collections import defaultdict
# This is used to make dirs in lib visiable to your python program
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path:
  sys.path.insert(0, str(lib_dir))

from sklearn.cluster import KMeans
from config_utils import Logger, time_string, convert_secs2time, AverageMeter
from config_utils import load_configure, obtain_accuracy, count_parameters_in_MB
from datasets     import AwA2_IMG_Rotate_Save 
from models       import distance_func
from networks     import obtain_combine_models_v2
from eval_util    import evaluate_all_dual
from datasets     import DualMetaSampler, AllClassSampler


def get_train_protos(network, attributes, train_classes, unseen_classes, all_class_loader, xargs):
  train_proto_path = '{:}/../train_proto_lists-{:}.pth'.format(xargs.log_dir, xargs.dataset)
  if os.path.exists(train_proto_path):
    train_proto_lists = torch.load(train_proto_path)
  else:
    # get the training protos over all images
    train_proto_lists = dict()
    num_per_class     = defaultdict(lambda: 0)
    data_time, xend   = AverageMeter(), time.time()
    all_class_sampler = all_class_loader.batch_sampler
    for ibatch, (feats, labels) in enumerate(all_class_loader):
      assert len(set(labels.tolist())) == 1
      label = labels[0].item()
      num_per_class[label] += feats.size(0)
      if label not in train_proto_lists:
        train_proto_lists[label] = torch.sum(feats, dim=0) / len(all_class_sampler.label2index[label])
      else:
        train_proto_lists[label]+= torch.sum(feats, dim=0) / len(all_class_sampler.label2index[label])

      data_time.update(time.time() - xend)
      xend = time.time()
      if ibatch % 100 == 0 or ibatch + 1 == len(all_class_loader):
        Tstring = '{:} [{:03d}/{:03d}] AVG=({:.2f}, {:.2f})'.format(time_string(), ibatch, len(all_class_loader), data_time.val, data_time.avg)
        Tstring+= ' :: {:}'.format(convert_secs2time(data_time.avg * (len(all_class_loader)-ibatch), True))
        print('***extract features*** : {:}'.format(Tstring))
    # check numbers
    for key, item in num_per_class.items():
      assert item == len(all_class_sampler.label2index[key]), '[{:}] : {:} vs {:} \n:::{:}'.format(key, item, len(all_class_sampler.label2index[label]), num_per_class)
    torch.save(train_proto_lists, train_proto_path)

  train_protos = [ train_proto_lists[cls] for cls in train_classes ]
  train_protos = torch.stack(train_protos).cuda()
  with torch.no_grad():
    network.eval()
    raw_atts, attention = network.get_attention(attributes.cuda())
    # get seen protos
    #seen_att    = F.softmax(raw_atts[train_classes,:][:,train_classes], dim=1)
    # get unseen protos
    unseen_att = raw_atts[unseen_classes,:][:,train_classes]
  return train_protos, unseen_att


def train_model(loader, semantics, img_protos, adj_distances, network, optimizer, config, logger):
  
  batch_time, Xlosses, CLSFlosses, Rlosses, accs, end = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()
  labelMeter, eps = AverageMeter(), 1e-7
  network.train()

  loader.dataset.set_return_label_mode('new')
  loader.dataset.set_return_img_mode('original_augment')

  logger.print('[TRAIN---{:}], semantics-shape={:}, adj_distances-shape={:}, config={:}'.format(config.epoch_str, semantics.shape, adj_distances.shape, config))

  for batch_idx, (img_feat, targets) in enumerate(loader):

    class_num = semantics.shape[0]
    batch = img_feat.shape[0]

    relations  = network(img_feat.cuda(), img_protos.cuda(), semantics.cuda(), adj_distances)
    raw_att_att, att_att = network.relation_module.get_attention_attribute()
    raw_att_img, att_img = network.relation_module.get_attention_img()
    if config.consistency_type == 'mse':
      consistency_loss = F.mse_loss(raw_att_att, raw_att_img) 
    elif config.consistency_type == 'kla2i':
      consistency_loss = F.kl_div((att_att + eps).log(), att_img + eps, reduction='batchmean')
    elif config.consistency_type == 'kli2a':
      consistency_loss = F.kl_div((att_img + eps).log(), att_att + eps, reduction='batchmean')
    else:
      raise ValueError('Unknown consistency type: {:}'.format(config.consistency_type))

    one_hot_labels = torch.zeros(batch, class_num).scatter_(1, targets.view(-1,1), 1).cuda()
    new_target_idxs = targets.cuda()
    target__labels = targets.cuda()

    if config.loss_type == 'sigmoid-mse':
      prediction = torch.sigmoid(relations)
      cls_loss = F.mse_loss(prediction, one_hot_labels, reduction='mean')
    elif re.match('softmax-*-*', config.loss_type, re.I):
      _, tempreture, epsilon = config.loss_type.split('-')
      tempreture, epsilon = float(tempreture), float(epsilon)
      if epsilon <= 0:
        cls_loss = F.cross_entropy(relations / tempreture, target__labels, weight=None, reduction='mean')
      else:
        log_probs = F.log_softmax(relations / tempreture, dim=1)
        _targets  = torch.zeros_like(log_probs).scatter_(1, target__labels.unsqueeze(1), 1)
        _targets  = (1-epsilon) * _targets + epsilon / relations.size(1)
        cls_loss  = (-_targets * log_probs).sum(dim=1).mean()
    elif config.loss_type == 'softmax':
      cls_loss = F.cross_entropy(relations, target__labels, weight=None, reduction='mean')
    elif config.loss_type == 'mse':
      cls_loss = F.mse_loss(torch.sigmoid(relations), one_hot_labels, reduction='mean')
    elif config.loss_type == 'none':
      positive = -torch.masked_select(relations, one_hot_labels == 1)
      negative = torch.masked_select(relations, one_hot_labels == 0)
      losses   = torch.cat([positive, negative])
      cls_loss = losses.mean()
    else:
      raise ValueError('invalid loss type : {:}'.format(config.loss_type))

    if config.consistency_coef > 0:
      loss = cls_loss + config.consistency_coef * consistency_loss
    else:
      loss = cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    # analysis
    Xlosses.update(loss.item(), batch)
    CLSFlosses.update(cls_loss.item(), batch)
    Rlosses.update(consistency_loss.item(), batch)
    predict_labels = torch.argmax(relations, dim=1)
    with torch.no_grad():
      accuracy = (predict_labels.cpu() == new_target_idxs.cpu()).float().mean().item()
      accs.update(accuracy*100, batch)
      labelMeter.update(class_num, 1)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
  
    if batch_idx % config.log_interval == 0 or batch_idx + 1 == len(loader):
      Tstring = 'TIME[{batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(batch_time=batch_time)
      Sstring = '{:} [{:}] [{:03d}/{:03d}]'.format(time_string(), config.epoch_str, batch_idx, len(loader))
      Astring = 'loss={:.7f} ({:.5f}), cls_loss={:.7f} ({:.5f}), consistency_loss={:.7f} ({:.5f}), acc@1={:.1f} ({:.1f})'.format(Xlosses.val, Xlosses.avg, CLSFlosses.val, CLSFlosses.avg, Rlosses.val, Rlosses.avg, accs.val, accs.avg)
      logger.print('{:} {:} {:} B={:}, L={:} ({:.1f})'.format(Sstring, Tstring, Astring, batch, class_num, labelMeter.avg))
  return Xlosses.avg, accs.avg


def main(xargs):
  # your main function
  # print some necessary informations
  # create logger
  if not os.path.exists(xargs.log_dir):
    os.makedirs(xargs.log_dir)
  logger = Logger(xargs.log_dir, xargs.manual_seed)
  logger.print('args :\n{:}'.format(xargs))
  logger.print('PyTorch: {:}'.format(torch.__version__))

  assert torch.cuda.is_available(), 'You must have at least one GPU'

  # set random seed
  #torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  random.seed(xargs.manual_seed)
  np.random.seed(xargs.manual_seed)
  torch.manual_seed(xargs.manual_seed)
  torch.cuda.manual_seed(xargs.manual_seed)

  logger.print('Start Main with this file : {:}'.format(__file__))
  graph_info = torch.load(Path(xargs.data_root))
  unseen_classes = graph_info['unseen_classes']
  train_classes  = graph_info['train_classes']

  # All labels return original value between 0-49
  train_dataset       = AwA2_IMG_Rotate_Save(graph_info, 'train')
  batch_size          = xargs.class_per_it * xargs.num_shot
  total_episode       = ((len(train_dataset) / batch_size) // 100 + 1) * 100
  #train_sampler       = MetaSampler(train_dataset, total_episode, xargs.class_per_it, xargs.num_shot)
  train_sampler       = DualMetaSampler(train_dataset, total_episode, xargs.class_per_it, xargs.num_shot) 
  # train_loader        = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=xargs.num_workers)
  train_loader        = torch.utils.data.DataLoader(train_dataset,       batch_size=batch_size, shuffle=True , num_workers=xargs.num_workers, drop_last=True)
  test_seen_dataset   = AwA2_IMG_Rotate_Save(graph_info, 'test-seen')
  test_seen_dataset.set_return_img_mode('original')
  test_seen_loader    = torch.utils.data.DataLoader(test_seen_dataset,   batch_size=batch_size, shuffle=False, num_workers=xargs.num_workers)
  test_unseen_dataset = AwA2_IMG_Rotate_Save(graph_info, 'test-unseen')
  test_unseen_dataset.set_return_img_mode('original')
  test_unseen_loader  = torch.utils.data.DataLoader(test_unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=xargs.num_workers)
  all_class_sampler   = AllClassSampler(train_dataset)
  all_class_loader    = torch.utils.data.DataLoader(train_dataset, batch_sampler=all_class_sampler, num_workers=xargs.num_workers, pin_memory=True)
  logger.print('train-dataset       : {:}'.format(train_dataset))
  #logger.print('train_sampler       : {:}'.format(train_sampler))
  logger.print('test-seen-dataset   : {:}'.format(test_seen_dataset))
  logger.print('test-unseen-dataset : {:}'.format(test_unseen_dataset))
  logger.print('all-class-train-sam : {:}'.format(all_class_sampler))

  features       = graph_info['ori_attributes'].float().cuda()
  train_features = features[graph_info['train_classes'], :]
  logger.print('feature-shape={:}, train-feature-shape={:}'.format(list(features.shape), list(train_features.shape)))

  kmeans = KMeans(n_clusters=xargs.clusters, random_state=1337).fit(train_features.cpu().numpy())
  att_centers = torch.tensor(kmeans.cluster_centers_).float().cuda()
  for cls in range(xargs.clusters):
    logger.print('[cluster : {:}] has {:} elements.'.format(cls, (kmeans.labels_ == cls).sum()))
  logger.print('Train-Feature-Shape={:}, use {:} clusters, shape={:}'.format(train_features.shape, xargs.clusters, att_centers.shape))

  # build adjacent matrix
  distances     = distance_func(graph_info['attributes'], graph_info['attributes'], 'euclidean-pow').float().cuda()
  xallx_adj_dis = distances.clone()
  train_adj_dis = distances[graph_info['train_classes'],:][:,graph_info['train_classes']]

  network = obtain_combine_models_v2(xargs.semantic_name, xargs.relation_name, att_centers, 2048)
  network = network.cuda()

  #parameters = [{'params': list(C_Net.parameters()), 'lr': xargs.lr*5, 'weight_decay': xargs.weight_decay*0.1},
  #              {'params': list(R_Net.parameters()), 'lr': xargs.lr  , 'weight_decay': xargs.weight_decay}]
  parameters = network.parameters()
  optimizer  = torch.optim.Adam(parameters, lr=xargs.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=xargs.weight_decay, amsgrad=False)
  #optimizer = torch.optim.SGD(parameters, lr=xargs.lr, momentum=0.9, weight_decay=xargs.weight_decay, nesterov=True)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.1, step_size=xargs.epochs*2//3)
  logger.print('network : {:.2f} MB =>>>\n{:}'.format(count_parameters_in_MB(network), network))
  logger.print('optimizer : {:}'.format(optimizer))
  
  model_lst_path  = logger.checkpoint('ckp-last-{:}.pth'.format(xargs.manual_seed))
  if os.path.isfile(model_lst_path):
    checkpoint  = torch.load(model_lst_path)
    start_epoch = checkpoint['epoch'] + 1
    best_accs   = checkpoint['best_accs']
    network.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger.print('load checkpoint from {:}'.format(model_lst_path))
  else:
    start_epoch, best_accs = 0, {'train': -1, 'xtrain': -1, 'zs': -1, 'gzs-seen': -1, 'gzs-unseen': -1, 'gzs-H':-1, 'best-info': None}
  
  with torch.no_grad():
    train_loader.dataset.set_return_img_mode('original')
    all_class_loader.dataset.set_return_label_mode('original')
    all_class_loader.dataset.set_return_img_mode('original')
    seen_protos, _ = get_train_protos(network, features, train_classes, unseen_classes, all_class_loader, xargs)
  epoch_time, start_time = AverageMeter(), time.time()
  # training
  for iepoch in range(start_epoch, xargs.epochs):
    # set some classes as fake zero-shot classes
    time_str = convert_secs2time(epoch_time.val * (xargs.epochs- iepoch), True) 
    epoch_str= '{:03d}/{:03d}'.format(iepoch, xargs.epochs)
    # last_lr  = lr_scheduler.get_last_lr()
    last_lr  = lr_scheduler.get_lr()
    logger.print('Train the {:}-th epoch, {:}, LR={:1.6f} ~ {:1.6f}'.format(epoch_str, time_str, min(last_lr), max(last_lr)))
  
    config_train = load_configure(None, {'epoch_str': epoch_str, 'log_interval': xargs.log_interval,
                                         'loss_type': xargs.loss_type,
                                         'consistency_coef': xargs.consistency_coef,
                                         'consistency_type': xargs.consistency_type}, None)

    train_cls_loss, train_acc = train_model(train_loader, train_features, seen_protos, train_adj_dis, network, optimizer, config_train, logger)
    
    lr_scheduler.step()
    if train_acc > best_accs['train']: best_accs['train'] = train_acc
    logger.print('Train {:} done, cls-loss={:.3f}, accuracy={:.2f}%, (best={:.2f}).\n'.format(epoch_str, train_cls_loss, train_acc, best_accs['train']))

    if iepoch % xargs.test_interval == 0 or iepoch == xargs.epochs -1:
      with torch.no_grad():
        xinfo = {'train_classes' : graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}
        train_loader.dataset.set_return_img_mode('original')
        all_class_loader.dataset.set_return_label_mode('original')
        all_class_loader.dataset.set_return_img_mode('original')
        seen_protos, unseen_att = get_train_protos(network, features, train_classes, unseen_classes, all_class_loader, xargs)
        for test_topK in range(1, 2):
          logger.print('-----test--init with top-{:} seen protos-------'.format(test_topK))
          topkATT, topkIDX = torch.topk(unseen_att, test_topK, dim=1)
          norm_att      = F.softmax(topkATT, dim=1)
          unseen_protos = norm_att.view(len(unseen_classes), test_topK, 1) * seen_protos[topkIDX]
          unseen_protos = unseen_protos.mean(dim=1)
          protos = []
          for icls in range(features.size(0)):
            if icls in train_classes: protos.append( seen_protos[ train_classes.index(icls) ] )
            else                    : protos.append( unseen_protos[ unseen_classes.index(icls) ] )
          protos = torch.stack(protos)
          train_loader.dataset.set_return_img_mode('original')
          evaluate_all_dual(epoch_str, train_loader, test_unseen_loader, test_seen_loader, features, protos, xallx_adj_dis, network, xinfo, best_accs, logger)

    semantic_lists = network.get_semantic_list(features)
    # save the info
    info = {'epoch'           : iepoch,
            'args'            : deepcopy(xargs),
            'finish'          : iepoch+1==xargs.epochs,
            'best_accs'       : best_accs,
            'semantic_lists'  : semantic_lists,
            'adj_distances'   : xallx_adj_dis,
            'network'         : network.state_dict(),
            'optimizer'       : optimizer.state_dict(),
            'scheduler'       : lr_scheduler.state_dict(),
            }
    try:
      torch.save(info, model_lst_path)
      logger.print('--->>> joint-arch :: save into {:}.\n'.format(model_lst_path))
    except PermmisionError:
      print('unsuccessful write log')

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
  if 'info' in locals() or 'checkpoint' in locals():
    if 'checkpoint' in locals():
      semantic_lists = checkpoint['semantic_lists']
    else:
      semantic_lists = info['semantic_lists']
  '''
  # the final evaluation
  logger.print('final evaluation --->>>')
  with torch.no_grad():
    xinfo = {'train_classes' : graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}
    train_loader.dataset.set_return_img_mode('original')
    evaluate_all('final-eval', train_loader, test_unseen_loader, test_seen_loader, features, xallx_adj_dis, network, xinfo, best_accs, logger)
  logger.print('-'*200)
  '''
  logger.close()

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Isometric Propagation Network (IPN).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--log_dir' ,       type=str,                   help='Save dir.')
  parser.add_argument('--data_root' ,     type=str,                   help='dataset root')
  parser.add_argument('--dataset' ,       type=str,                   help='dataset')
  # Optimization options
  parser.add_argument('--loss_type',      type=str,                   help='The loss type.')
  parser.add_argument('--semantic_name',  type=str,                   help='The .')
  parser.add_argument('--relation_name',  type=str,                   help='The attention type.')
  parser.add_argument('--clusters',       type=int,  default=3,       help='.')
  parser.add_argument('--class_per_it',   type=int,                   help='The number of classes in each episode.')
  parser.add_argument('--num_shot'    ,   type=int,                   help='The number of samples in each class in an episode.')
  parser.add_argument('--epochs',         type=int,                   help='The number of training epochs.')
  parser.add_argument('--manual_seed',    type=int,                   help='The manual seed.')
  parser.add_argument('--lr',             type=float,                 help='The learning rate.')
  parser.add_argument('--weight_decay',   type=float,                 help='The weight decay.')
  parser.add_argument('--consistency_coef', type=float, default=0,    help='The coefficient for the consistency loss.')
  parser.add_argument('--consistency_type', type=str, default=0, choices=['kla2i', 'kli2a', 'mse'], help='The coefficient for the consistency loss.')
  parser.add_argument('--num_workers',    type=int,   default= 8,     help='The number of workers.')
  parser.add_argument('--log_interval',   type=int,   default=10,     help='The log-print interval.')
  parser.add_argument('--test_interval',  type=int,   default=10,     help='The evaluation interval.')
    
  args = parser.parse_args()

  if args.manual_seed is None or args.manual_seed < 0:
    args.manual_seed = random.randint(1, 100000)
  assert args.log_dir is not None, 'The log_dir argument can not be None.'
  main(args)
