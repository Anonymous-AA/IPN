import os, sys, time, torch
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# self-packages
from config_utils import AverageMeter, time_string, obtain_accuracy


def evaluate(loader, features, adj, network):
  losses, all_predictions, all_labels = AverageMeter(), [], []
  network.eval()
  with torch.no_grad():
    #hidden_features = c_net(features.cuda())
    class_num       = features.size(0)
    for batch_idx, (image_features, target) in enumerate(loader):
      batch, feat_dim = image_features.shape
      #relation        = r_net(image_features.cuda(), hidden_features, adj)
      relation = network(image_features.cuda(), features.cuda(), adj.cuda())

      one_hot_labels  = torch.zeros(batch, class_num).scatter_(1, target.view(-1,1), 1).cuda()
      loss = F.mse_loss(torch.sigmoid(relation), one_hot_labels, reduction='mean')
      losses.update(loss.item(), batch)

      predict_labels  = torch.argmax(relation, dim=1).cpu()
      all_predictions.append( predict_labels )
      all_labels.append( target )
  predictions = torch.cat(all_predictions, dim=0)
  labels      = torch.cat(all_labels, dim=0)
  all_classes = sorted( list(set(labels.tolist())) )
  acc_per_classes = []
  for idx, cls in enumerate(all_classes):
    assert idx <= cls, 'invalid all-classes : {:}'.format(all_classes)
    #print ('dataset : {:}'.format(loader.dataset))
    indexes = labels == cls
    xpreds, xlabels = predictions[indexes], labels[indexes]
    acc_per_classes.append( (xpreds==xlabels).float().mean().item() )
  acc_per_class = float( np.mean(acc_per_classes) )
  return losses.avg, ['{:3.1f}'.format(x*100) for x in acc_per_classes], acc_per_class * 100


def evaluate_all(epoch_str, train_loader, test_unseen_loader, \
                         test_seen_loader, features, adj_dis, network, \
                         info, best_accs, logger):
  train_classes, unseen_classes = info['train_classes'], info['unseen_classes']
  logger.print('Evaluate [{:}]'.format(epoch_str))
  # calculate zero shot setting
  test_unseen_loader.dataset.set_return_label_mode('new')
  target_semantics = features[unseen_classes, :]
  target_adj_dis   = adj_dis[unseen_classes,:][:,unseen_classes]
  test_loss, _, test_per_cls_acc = evaluate(test_unseen_loader, target_semantics, target_adj_dis, network)
  if best_accs['zs'] < test_per_cls_acc: best_accs['zs'] = test_per_cls_acc
  logger.print('Test {:} [zero-zero-zero-zero-shot----] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TTEST-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss, test_per_cls_acc, best_accs['zs']))
  # calculate generalized zero-shot setting
  train_loader.dataset.set_return_label_mode('original')
  train_loss, _, train_per_cls_acc = evaluate(train_loader, features, adj_dis, network)
  if best_accs['xtrain'] < train_per_cls_acc: best_accs['xtrain'] = train_per_cls_acc
  logger.print('Test {:} [train-train-train-train-----] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TRAIN-best={:5.2f}%).'.format(time_string(), epoch_str, train_loss, train_per_cls_acc, best_accs['xtrain']))
  test_unseen_loader.dataset.set_return_label_mode('original')
  test_loss_unseen, test_unsn_accs, test_per_cls_acc_unseen = evaluate(test_unseen_loader, features, adj_dis, network)
  if best_accs['gzs-unseen'] < test_per_cls_acc_unseen: best_accs['gzs-unseen'] = test_per_cls_acc_unseen
  logger.print('Test {:} [generalized-zero-shot-unseen] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TUNSN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_unseen, test_per_cls_acc_unseen, best_accs['gzs-unseen']))
  #logger.print('Test {:} [generalized-zero-shot-unseen] {:} ::: {:}.'.format(time_string(), epoch_str, test_unsn_accs))
  # for test data with seen classes
  test_seen_loader.dataset.set_return_label_mode('original')
  test_loss_seen  , test_seen_accs, test_per_cls_acc_seen = evaluate(test_seen_loader, features, adj_dis, network)
  if best_accs['gzs-seen'] < test_per_cls_acc_seen: best_accs['gzs-seen'] = test_per_cls_acc_seen
  logger.print('Test {:} [generalized-zero-shot---seen] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TSEEN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_seen, test_per_cls_acc_seen, best_accs['gzs-seen']))
  #logger.print('Test {:} [generalized-zero-shot---seen] {:} ::: {:}.'.format(time_string(), epoch_str, test_seen_accs))
  harmonic_mean        = (2 * test_per_cls_acc_seen * test_per_cls_acc_unseen) / (test_per_cls_acc_seen + test_per_cls_acc_unseen + 1e-8)
  if best_accs['gzs-H'] < harmonic_mean:  
    best_accs['gzs-H'] = harmonic_mean 
    best_accs['best-info'] = '[{:}] seen={:5.2f}% unseen={:5.2f}%, H={:5.2f}%'.format(epoch_str, test_per_cls_acc_seen, test_per_cls_acc_unseen, harmonic_mean)
  logger.print('Test [generalized-zero-shot-h-mean] {:} H={:.3f}% (HH-best={:.3f}%). ||| Best comes from {:}'.format(epoch_str, harmonic_mean, best_accs['gzs-H'], best_accs['best-info']))


def evaluate_dual(loader, features, img_protos, adj, network):
  losses, all_predictions, all_labels = AverageMeter(), [], []
  network.eval()
  with torch.no_grad():
    #hidden_features = c_net(features.cuda())
    class_num       = features.size(0)
    for batch_idx, (image_features, target) in enumerate(loader):
      batch, feat_dim = image_features.shape
      #relation        = r_net(image_features.cuda(), hidden_features, adj)
      relation = network(image_features.cuda(),img_protos.cuda(), features.cuda(), adj.cuda())
      one_hot_labels  = torch.zeros(batch, class_num).scatter_(1, target.view(-1,1), 1).cuda()
      loss = F.mse_loss(torch.sigmoid(relation), one_hot_labels, reduction='mean')
      losses.update(loss.item(), batch)

      predict_labels  = torch.argmax(relation, dim=1).cpu()
      all_predictions.append( predict_labels )
      all_labels.append( target )
  predictions = torch.cat(all_predictions, dim=0)
  labels      = torch.cat(all_labels, dim=0)
  all_classes = sorted( list(set(labels.tolist())) )
  acc_per_classes = []
  for idx, cls in enumerate(all_classes):
    assert idx <= cls, 'invalid all-classes : {:}'.format(all_classes)
    #print ('dataset : {:}'.format(loader.dataset))
    indexes = labels == cls
    xpreds, xlabels = predictions[indexes], labels[indexes]
    acc_per_classes.append( (xpreds==xlabels).float().mean().item() )
  acc_per_class = float( np.mean(acc_per_classes) )
  return losses.avg, ['{:3.1f}'.format(x*100) for x in acc_per_classes], acc_per_class * 100


def evaluate_all_dual(epoch_str, train_loader, test_unseen_loader, \
                         test_seen_loader, features, img_protos, adj_dis, network, \
                         info, best_accs, logger):
  train_classes, unseen_classes = info['train_classes'], info['unseen_classes']
  logger.print('Evaluate [{:}]'.format(epoch_str))
  # calculate zero shot setting
  test_unseen_loader.dataset.set_return_label_mode('new')
  target_semantics  = features[unseen_classes, :]
  target_img_protos = img_protos[unseen_classes, :]
  target_adj_dis    = adj_dis[unseen_classes,:][:,unseen_classes]
  test_loss, _, test_per_cls_acc = evaluate_dual(test_unseen_loader, target_semantics, target_img_protos, target_adj_dis, network)
  if best_accs['zs'] < test_per_cls_acc: best_accs['zs'] = test_per_cls_acc
  logger.print('Test {:} [zero-zero-zero-zero-shot----] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TTEST-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss, test_per_cls_acc, best_accs['zs']))
  # calculate generalized zero-shot setting
  train_loader.dataset.set_return_label_mode('original')
  train_loss, _, train_per_cls_acc = evaluate_dual(train_loader, features, img_protos, adj_dis, network)
  if best_accs['xtrain'] < train_per_cls_acc: best_accs['xtrain'] = train_per_cls_acc
  logger.print('Test {:} [train-train-train-train-----] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TRAIN-best={:5.2f}%).'.format(time_string(), epoch_str, train_loss, train_per_cls_acc, best_accs['xtrain']))
  test_unseen_loader.dataset.set_return_label_mode('original')
  test_loss_unseen, test_unsn_accs, test_per_cls_acc_unseen = evaluate_dual(test_unseen_loader, features, img_protos, adj_dis, network)
  if best_accs['gzs-unseen'] < test_per_cls_acc_unseen: best_accs['gzs-unseen'] = test_per_cls_acc_unseen
  logger.print('Test {:} [generalized-zero-shot-unseen] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TUNSN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_unseen, test_per_cls_acc_unseen, best_accs['gzs-unseen']))
  #logger.print('Test {:} [generalized-zero-shot-unseen] {:} ::: {:}.'.format(time_string(), epoch_str, test_unsn_accs))
  # for test data with seen classes
  test_seen_loader.dataset.set_return_label_mode('original')
  test_loss_seen  , test_seen_accs, test_per_cls_acc_seen = evaluate_dual(test_seen_loader, features, img_protos, adj_dis, network)
  if best_accs['gzs-seen'] < test_per_cls_acc_seen: best_accs['gzs-seen'] = test_per_cls_acc_seen
  logger.print('Test {:} [generalized-zero-shot---seen] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TSEEN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_seen, test_per_cls_acc_seen, best_accs['gzs-seen']))
  #logger.print('Test {:} [generalized-zero-shot---seen] {:} ::: {:}.'.format(time_string(), epoch_str, test_seen_accs))
  harmonic_mean        = (2 * test_per_cls_acc_seen * test_per_cls_acc_unseen) / (test_per_cls_acc_seen + test_per_cls_acc_unseen + 1e-8)
  if best_accs['gzs-H'] < harmonic_mean:  
    best_accs['gzs-H'] = harmonic_mean 
    best_accs['best-info'] = '[{:}] seen={:5.2f}% unseen={:5.2f}%, H={:5.2f}%'.format(epoch_str, test_per_cls_acc_seen, test_per_cls_acc_unseen, harmonic_mean)
  logger.print('Test [generalized-zero-shot-h-mean] {:} H={:.3f}% (HH-best={:.3f}%). ||| Best comes from {:}'.format(epoch_str, harmonic_mean, best_accs['gzs-H'], best_accs['best-info']))


def evaluate_dual_img(loader, features, img_protos, adj, network, backbone):
  losses, all_predictions, all_labels = AverageMeter(), [], []
  network.eval()
  backbone.eval()
  with torch.no_grad():
    #hidden_features = c_net(features.cuda())
    class_num       = features.size(0)
    for batch_idx, (images, target) in enumerate(loader):
      with torch.no_grad():
        image_features  = backbone(images.cuda())
      batch, feat_dim = image_features.shape
      #relation        = r_net(image_features.cuda(), hidden_features, adj)
      relation = network(image_features.cuda(),img_protos.cuda(), features.cuda(), adj.cuda())
      one_hot_labels  = torch.zeros(batch, class_num).scatter_(1, target.view(-1,1), 1).cuda()
      loss = F.mse_loss(torch.sigmoid(relation), one_hot_labels, reduction='mean')
      losses.update(loss.item(), batch)

      predict_labels  = torch.argmax(relation, dim=1).cpu()
      all_predictions.append( predict_labels )
      all_labels.append( target )
  predictions = torch.cat(all_predictions, dim=0)
  labels      = torch.cat(all_labels, dim=0)
  all_classes = sorted( list(set(labels.tolist())) )
  acc_per_classes = []
  for idx, cls in enumerate(all_classes):
    assert idx <= cls, 'invalid all-classes : {:}'.format(all_classes)
    #print ('dataset : {:}'.format(loader.dataset))
    indexes = labels == cls
    xpreds, xlabels = predictions[indexes], labels[indexes]
    acc_per_classes.append( (xpreds==xlabels).float().mean().item() )
  acc_per_class = float( np.mean(acc_per_classes) )
  return losses.avg, ['{:3.1f}'.format(x*100) for x in acc_per_classes], acc_per_class * 100


def evaluate_all_dual_img(epoch_str, train_loader, test_unseen_loader, \
                         test_seen_loader, features, img_protos, adj_dis, network, backbone, \
                         info, best_accs, logger):
  train_classes, unseen_classes = info['train_classes'], info['unseen_classes']
  logger.print('Evaluate [{:}]'.format(epoch_str))
  # calculate zero shot setting
  test_unseen_loader.dataset.set_return_label_mode('new')
  target_semantics  = features[unseen_classes, :]
  target_img_protos = img_protos[unseen_classes, :]
  target_adj_dis    = adj_dis[unseen_classes,:][:,unseen_classes]
  test_loss, _, test_per_cls_acc = evaluate_dual_img(test_unseen_loader, target_semantics, target_img_protos, target_adj_dis, network, backbone)
  if best_accs['zs'] < test_per_cls_acc: best_accs['zs'] = test_per_cls_acc
  logger.print('Test {:} [zero-zero-zero-zero-shot----] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TTEST-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss, test_per_cls_acc, best_accs['zs']))
  # calculate generalized zero-shot setting
  train_loader.dataset.set_return_label_mode('original')
  train_loss, _, train_per_cls_acc = evaluate_dual_img(train_loader, features, img_protos, adj_dis, network, backbone)
  if best_accs['xtrain'] < train_per_cls_acc: best_accs['xtrain'] = train_per_cls_acc
  logger.print('Test {:} [train-train-train-train-----] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TRAIN-best={:5.2f}%).'.format(time_string(), epoch_str, train_loss, train_per_cls_acc, best_accs['xtrain']))
  test_unseen_loader.dataset.set_return_label_mode('original')
  test_loss_unseen, test_unsn_accs, test_per_cls_acc_unseen = evaluate_dual_img(test_unseen_loader, features, img_protos, adj_dis, network, backbone)
  if best_accs['gzs-unseen'] < test_per_cls_acc_unseen: best_accs['gzs-unseen'] = test_per_cls_acc_unseen
  logger.print('Test {:} [generalized-zero-shot-unseen] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TUNSN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_unseen, test_per_cls_acc_unseen, best_accs['gzs-unseen']))
  #logger.print('Test {:} [generalized-zero-shot-unseen] {:} ::: {:}.'.format(time_string(), epoch_str, test_unsn_accs))
  # for test data with seen classes
  test_seen_loader.dataset.set_return_label_mode('original')
  test_loss_seen  , test_seen_accs, test_per_cls_acc_seen = evaluate_dual_img(test_seen_loader, features, img_protos, adj_dis, network, backbone)
  if best_accs['gzs-seen'] < test_per_cls_acc_seen: best_accs['gzs-seen'] = test_per_cls_acc_seen
  logger.print('Test {:} [generalized-zero-shot---seen] {:} done, loss={:.3f}, per-class-acc={:5.2f}% (TSEEN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_seen, test_per_cls_acc_seen, best_accs['gzs-seen']))
  #logger.print('Test {:} [generalized-zero-shot---seen] {:} ::: {:}.'.format(time_string(), epoch_str, test_seen_accs))
  harmonic_mean        = (2 * test_per_cls_acc_seen * test_per_cls_acc_unseen) / (test_per_cls_acc_seen + test_per_cls_acc_unseen + 1e-8)
  if best_accs['gzs-H'] < harmonic_mean:  
    best_accs['gzs-H'] = harmonic_mean 
    best_accs['best-info'] = '[{:}] seen={:5.2f}% unseen={:5.2f}%, H={:5.2f}%'.format(epoch_str, test_per_cls_acc_seen, test_per_cls_acc_unseen, harmonic_mean)
  logger.print('Test [generalized-zero-shot-h-mean] {:} H={:.3f}% (HH-best={:.3f}%). ||| Best comes from {:}'.format(epoch_str, harmonic_mean, best_accs['gzs-H'], best_accs['best-info']))
