# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys, json
from os import path as osp
from pathlib import Path
from collections import namedtuple
import time, torch
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# self-packages
from config_utils import AverageMeter, time_string, obtain_accuracy
from models import distance_func


support_types = ('str', 'int', 'bool', 'float', 'none')


def convert_param(original_lists):
  assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
  ctype, value = original_lists[0], original_lists[1]
  assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
  is_list = isinstance(value, list)
  if not is_list: value = [value]
  outs = []
  for x in value:
    if ctype == 'int':
      x = int(x)
    elif ctype == 'str':
      x = str(x)
    elif ctype == 'bool':
      x = bool(int(x))
    elif ctype == 'float':
      x = float(x)
    elif ctype == 'none':
      assert x == 'None', 'for none type, the value must be None instead of {:}'.format(x)
      x = None
    else:
      raise TypeError('Does not know this type : {:}'.format(ctype))
    outs.append(x)
  if not is_list: outs = outs[0]
  return outs


def load_configure(path, extra, logger):
  if path is not None:
    path = str(path)
    if hasattr(logger, 'print'): logger.print(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    # Reading data back
    with open(path, 'r') as f:
      data = json.load(f)
    f.close()
    content = { k: convert_param(v) for k,v in data.items()}
  else:
    content = {}
  assert extra is None or isinstance(extra, dict), 'invalid type of extra : {:}'.format(extra)
  if isinstance(extra, dict): content = {**content, **extra}
  Arguments = namedtuple('Configure', ' '.join(content.keys()))
  content   = Arguments(**content)
  if hasattr(logger, 'print'): logger.print('{:}'.format(content))
  return content


def configure2str(config, xpath=None):
  if not isinstance(config, dict):
    config = config._asdict()
  def cstring(x):
    return "\"{:}\"".format(x)
  def gtype(x):
    if isinstance(x, list): x = x[0]
    if isinstance(x, str)  : return 'str'
    elif isinstance(x, bool) : return 'bool'
    elif isinstance(x, int): return 'int'
    elif isinstance(x, float): return 'float'
    elif x is None           : return 'none'
    else: raise ValueError('invalid : {:}'.format(x))
  def cvalue(x, xtype):
    if isinstance(x, list): is_list = True
    else:
      is_list, x = False, [x]
    temps = []
    for temp in x:
      if xtype == 'bool'  : temp = cstring(int(temp))
      elif xtype == 'none': temp = cstring('None')
      else                : temp = cstring(temp)
      temps.append( temp )
    if is_list:
      return "[{:}]".format( ', '.join( temps ) )
    else:
      return temps[0]

  xstrings = []
  for key, value in config.items():
    xtype  = gtype(value)
    string = '  {:20s} : [{:8s}, {:}]'.format(cstring(key), cstring(xtype), cvalue(value, xtype))
    xstrings.append(string)
  Fstring = '{\n' + ',\n'.join(xstrings) + '\n}'
  if xpath is not None:
    parent = Path(xpath).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    if osp.isfile(xpath): os.remove(xpath)
    with open(xpath, "w") as text_file:
      text_file.write('{:}'.format(Fstring))
  return Fstring


def dict2configure(xdict, logger):
  assert isinstance(xdict, dict), 'invalid type : {:}'.format( type(xdict) )
  Arguments = namedtuple('Configure', ' '.join(xdict.keys()))
  content   = Arguments(**xdict)
  if hasattr(logger, 'print'): logger.print('{:}'.format(content))
  return content

def evaluate(models, loader, prototypes, criterion, distance):
  losses, acc1, accuracies = AverageMeter(), AverageMeter(), collections.defaultdict(list)
  #prototypes = F.normalize(prototypes, dim=1)
  cnn_model, model = models
  with torch.no_grad():
    for batch_idx, (images, emb, continous, target) in enumerate(loader):
      batch, target = images.size(0), target.cuda()
      tensors = cnn_model(images)
      vectors = model(tensors)
      # target is 0, 1, 2, ..., test-classes-1
      logits       = - distance_func(vectors, prototypes, distance)
      #logits = torch.nn.functional.cosine_similarity(data.view(-1,1,2048), prototypes.view(1,-1,2048), dim=1)
      cls_loss     = criterion(logits, target)
      losses.update(cls_loss.item(), batch)
      # log 
      [accuracy]   = obtain_accuracy(logits.data, target.data, (1,))
      acc1.update(accuracy.item(), batch)
      corrects     = (logits.argmax(dim=1) == target).cpu().tolist()
      target       = target.cpu().tolist()
      for cls, ok in zip(target, corrects):
        accuracies[cls].append(ok)
  acc_per_class = []
  for cls in accuracies.keys():
    acc_per_class.append( np.mean(accuracies[cls]) )
  acc_per_class = float( np.mean(acc_per_class) )
  return losses.avg, acc1.avg, acc_per_class * 100

def evaluate_all(epoch_str, protos, models, train_loader, test_unseen_loader, \
                         test_seen_loader, cls_loss, best_accs, distance, logger):
  logger.print('Evaluate [{:}] with distance={:}'.format(epoch_str, distance))
  # calculate zero shot setting
  target_wnid_indexes, _, _ = test_unseen_loader.dataset.get_all_wnid_index()
  test_unseen_loader.dataset.set_return_label_mode('new')
  test_loss, test_acc, test_per_cls_acc = evaluate(models, test_unseen_loader, protos[target_wnid_indexes], cls_loss, distance)
  if best_accs['zs'] < test_per_cls_acc: best_accs['zs'] = test_per_cls_acc
  logger.print('Test {:} [zero-zero-zero-zero-shot----] {:} done, loss={:.3f}, accuracy={:5.2f}%, per-class-acc={:5.2f}% (TTEST-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss, test_acc, test_per_cls_acc, best_accs['zs']))
  # calculate generalized zero-shot setting
  _, target_wnid_indexes, _ = train_loader.dataset.get_all_wnid_index()
  target_protos       = protos[target_wnid_indexes]
  train_loader.dataset.set_return_label_mode('original')
  train_loss, train_acc, train_per_cls_acc = evaluate(models, train_loader, target_protos, cls_loss, distance)
  if best_accs['xtrain'] < train_per_cls_acc: best_accs['xtrain'] = train_per_cls_acc
  logger.print('Test {:} [train-train-train-train-----] {:} done, loss={:.3f}, accuracy={:5.2f}%, per-class-acc={:5.2f}% (TRAIN-best={:5.2f}%).'.format(time_string(), epoch_str, train_loss, train_acc, train_per_cls_acc, best_accs['xtrain']))
  _, target_wnid_indexes, _ = test_unseen_loader.dataset.get_all_wnid_index()
  target_protos       = protos[target_wnid_indexes]
  test_unseen_loader.dataset.set_return_label_mode('original')
  test_loss_unseen, test_acc_unseen, test_per_cls_acc_unseen = evaluate(models, test_unseen_loader, target_protos, cls_loss, distance)
  if best_accs['gzs-unseen'] < test_per_cls_acc_unseen: best_accs['gzs-unseen'] = test_per_cls_acc_unseen
  logger.print('Test {:} [generalized-zero-shot-unseen] {:} done, loss={:.3f}, accuracy={:5.2f}%, per-class-acc={:5.2f}% (TUNSN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_unseen, test_acc_unseen, test_per_cls_acc_unseen, best_accs['gzs-unseen']))
  # for test data with seen classes
  test_seen_loader.dataset.set_return_label_mode('original')
  test_loss_seen, test_acc_seen, test_per_cls_acc_seen = evaluate(models, test_seen_loader, target_protos, cls_loss, distance)
  if best_accs['gzs-seen'] < test_per_cls_acc_seen: best_accs['gzs-seen'] = test_per_cls_acc_seen
  logger.print('Test {:} [generalized-zero-shot---seen] {:} done, loss={:.3f}, accuracy={:5.2f}%, per-class-acc={:5.2f}% (TSEEN-best={:5.2f}%).'.format(time_string(), epoch_str, test_loss_seen, test_acc_seen, test_per_cls_acc_seen, best_accs['gzs-seen']))
  harmonic_mean        = (2 * test_per_cls_acc_seen * test_per_cls_acc_unseen) / (test_per_cls_acc_seen + test_per_cls_acc_unseen + 1e-8)
  if best_accs['gzs-H'] < harmonic_mean:  
    best_accs['gzs-H'] = harmonic_mean 
    best_accs['best-info'] = '[{:}] seen={:5.2f}% unseen={:5.2f}%, H={:5.2f}%'.format(epoch_str, test_per_cls_acc_seen, test_per_cls_acc_unseen, harmonic_mean)
  logger.print('Test [generalized-zero-shot-h-mean] {:} H={:.3f}% (HH-best={:.3f}%). ||| Best comes from {:}'.format(epoch_str, harmonic_mean, best_accs['gzs-H'], best_accs['best-info']))

