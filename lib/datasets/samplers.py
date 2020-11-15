import copy, random, torch
import numpy as np
from collections import defaultdict


class DualMetaSampler(object):

  def __init__(self, dataset, iters, class_per_it, num_shot):
    super(DualMetaSampler, self).__init__()
    self.length = len(dataset)
    self.labels = copy.deepcopy( dataset.labels )
    self.label2index = defaultdict(list)
    for index, label in enumerate(self.labels):
      self.label2index[label].append( index )
    self.all_labels  = sorted( list(self.label2index.keys()) )
    self.num_classes = len(self.all_labels)
    self.iters       = int(iters)
    self.sample_per_class = num_shot
    self.classes_per_it   = class_per_it

  def __repr__(self):
    return ('{name}({iters:5d} iters, {sample_per_class:} ways, {classes_per_it:} shot, {num_classes:} classes)'.format(name=self.__class__.__name__, **self.__dict__))

  def __iter__(self):
    # yield a batch of indexes
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iters):
      #batch_size = spc * cpi
      few_shot_train_batch = []
      few_shot_valid_batch = []
      if cpi <= len(self.all_labels):
        batch_few_shot_classes = random.sample(self.all_labels, cpi)
      else:
        batch_few_shot_classes = np.random.choice(self.all_labels, cpi, True).tolist()
      for i, cls in enumerate(batch_few_shot_classes):
        sample_indexes = random.sample(self.label2index[cls], spc*2)
        few_shot_train_batch.extend( sample_indexes[:spc] )
        few_shot_valid_batch.extend( sample_indexes[spc:] )
      yield few_shot_train_batch + few_shot_valid_batch

  def __len__(self):
    return self.iters


class AllClassSampler(object):

  def __init__(self, dataset):
    super(AllClassSampler, self).__init__()
    self.labels = copy.deepcopy( dataset.labels )
    self.label2index = defaultdict(list)
    for index, label in enumerate(self.labels):
      self.label2index[label].append( index )
    self.all_labels  = sorted( list(self.label2index.keys()) )
    self.num_classes = len(self.all_labels)
    self.batch_size  = 30 
    self.all_batchs  = []
    # 
    for lab, idxes in self.label2index.items():
      n_iters = (len(idxes)// self.batch_size) 
      if len(idxes) % self.batch_size != 0: n_iters += 1
      for i in range(n_iters):
        cur_idx = idxes[self.batch_size*(i):min(self.batch_size*(i+1), len(idxes))]
        #print("cur_idx is {}-{}/{}, total-iters:{}".format(idxes.index(cur_idx[0]), idxes.index(cur_idx[-1]), len(idxes), n_iters))
        self.all_batchs.append( copy.deepcopy( cur_idx ) )
    self.length = len(self.all_batchs)

  def __repr__(self):
    return ('{name}({length:5d} iters)'.format(name=self.__class__.__name__, **self.__dict__))

  def __iter__(self):
    # yield a batch of indexes
    for batch in self.all_batchs:
      yield batch

  def __len__(self):
    return self.length
