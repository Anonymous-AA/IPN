import time
import torch
import torch.nn as nn
import numpy as np


def distance_func(x, y, xtype):
  if xtype == 'euclidean':
    return euclidean_dist(x, y, True)
  elif xtype == 'euclidean-pow':
    return euclidean_dist(x, y, False)
  elif xtype == 'cosine':
    return cosine_dist(x, y)
  elif xtype == 'dotprod':
    return dotprod_dist(x, y)
  else:
    raise ValueError('invalid type : {:}'.format(xtype))


def euclidean_dist(x, y, sqrt=False):
  assert x.dim() == 2 and y.dim() == 2
  batchx, dimx = x.size()
  batchy, dimy = y.size()
  x = x.view(batchx, 1, dimx)
  y = y.view(1, batchy, dimy)
  distances = torch.pow(x - y, 2).sum(-1)
  if sqrt:
    distances = distances.clamp(min=1e-12).sqrt()  # for numerical stability
  return distances


def cosine_dist(x, y):
  assert x.dim() == 2 and y.dim() == 2
  batchx, dimx = x.size()
  batchy, dimy = y.size()
  x = x.view(batchx, 1, dimx)
  y = y.view(1, batchy, dimy)
  distances = nn.functional.cosine_similarity(x, y, dim=2, eps=1e-6)
  return distances


def dotprod_dist(x, y):
  assert x.dim() == 2 and y.dim() == 2
  batchx, dimx = x.size()
  batchy, dimy = y.size()
  x = x.view(batchx, 1, dimx)
  y = y.view(1, batchy, dimy)
  distances = torch.sum(x * y, dim=2)
  return distances
