import os, sys, copy, torch, numpy as np
import torch.utils.data as data
from pathlib import Path
from torchvision import transforms
from PIL import Image


def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


class SIMPLE_DATA(data.Dataset):

  def __init__(self, files, num_aug, xtype='imagenet'):
    super(SIMPLE_DATA, self).__init__()
    self.files = files
    if xtype == 'imagenet':
      self.train_transform = transforms.Compose(
              [transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)), \
               transforms.RandomHorizontalFlip(p=0.5)])
      self.valid_transform = transforms.Compose(
              [transforms.Resize(256), transforms.CenterCrop(224)])
    elif xtype == 'tiered-imagenet':
      self.train_transform = transforms.Compose(
              [transforms.RandomHorizontalFlip(), transforms.RandomCrop(84, padding=8)])
      self.valid_transform = transforms.Compose(
              [transforms.CenterCrop(84)])
    else: raise ValueError('invalid tranformation type : {:}'.format(xtype))
    self.totensor = transforms.Compose(
              [transforms.ToTensor(), \
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    self._num_aug = num_aug
    assert self._num_aug >= 0, 'invalid self._num_aug : {:}'.format(self._num_aug)

  def __getitem__(self, index):
    assert 0 <= index < len(self), 'invalid index = {:}'.format(index)
    xfile = self.files[index]
    if xfile.split('.')[-1] == 'npy':
      np_image = np.load(xfile)
      image = Image.fromarray(np_image)
    else:
      image = pil_loader( xfile )
    images = [self.valid_transform(image)]
    for i in range( self._num_aug ):
      images.append( self.train_transform(image) )
    images_090 = [x.rotate( 90) for x in images]
    images_180 = [x.rotate(180) for x in images]
    images_270 = [x.rotate(270) for x in images]
    tensor_000 = [self.totensor(x) for x in images]
    tensor_090 = [self.totensor(x) for x in images_090]
    tensor_180 = [self.totensor(x) for x in images_180]
    tensor_270 = [self.totensor(x) for x in images_270]
    return index, tensor_000, tensor_090, tensor_180, tensor_270

  def __len__(self):
    return len(self.files)
