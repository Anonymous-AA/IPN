###################################################################
# Create the cache dataset file by the following commands:
# [AWA2] python exps/prepare-data-step1.py --name AWA2
# [APY]  python exps/prepare-data-step1.py --name APY
#
###################################################################
# CUB: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# APY: http://vision.cs.uiuc.edu/attributes/
###################################################################
import os, json, time, argparse, torch
from pathlib import Path
import scipy.io as sio
from collections import defaultdict
import numpy as np
from PIL import Image


def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string


def pil_loader(path):
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


dataset_root   = Path(os.environ['HOME']) / 'zshot-data'
save_dir       = dataset_root / 'info-files'
save_dir.mkdir(parents=True, exist_ok=True)


def crop_image(xpath, bbox):
  image = pil_loader(xpath)
  W, H  = bbox[2] - bbox[0], bbox[3] - bbox[1]
  crop_image = image.crop( tuple(bbox) )
  return image, crop_image, min(W, H)


def get_real_path(dataset, xpath, bboxes, index):
  if dataset == 'AWA2':
    image_root = dataset_root / 'Animals_with_Attributes2' / 'JPEGImages'
    all_parts  = xpath.split('/')
    image_file = image_root / all_parts[-2] / all_parts[-1]
    assert image_file.exists(), 'invalid path : {:}'.format(image_file)
  elif dataset == 'CUB' or dataset == 'SUN':
    image_root = dataset_root / dataset
    all_parts  = xpath.split('/')
    image_file = image_root / all_parts[-2] / all_parts[-1]
    if not image_file.exists():
      image_file = image_root / all_parts[-3] / all_parts[-2] / all_parts[-1]
    if not image_file.exists():
      image_file = image_root / all_parts[-4] / all_parts[-3] / all_parts[-2] / all_parts[-1]
    assert image_file.exists(), 'invalid path : {:}'.format(image_file)
  elif dataset == 'APY':
    image_root = dataset_root / dataset
    all_parts  = xpath.split('/')
    old_image_file = image_root / all_parts[-2] / all_parts[-1]
    if not old_image_file.exists():
      old_image_file = image_root / all_parts[-4] / all_parts[-3] / all_parts[-2] / all_parts[-1]
    assert old_image_file.exists(), 'invalid path : {:}'.format(old_image_file)

    new_image_dir = dataset_root / 'APY-CROP'
    new_image_dir.mkdir(parents=True, exist_ok=True)
    new_image_file= new_image_dir / '{:06d}.png'.format(index)
    if not new_image_file.exists():
      try:
        img, crop_I, min_L = crop_image(str(old_image_file), bboxes[index])
        if min_L < 10:
          img.save(str(new_image_file))
        else:
          crop_I.save(str(new_image_file))
      except:
        img.save(str(new_image_file))
    image_file    = new_image_file
    assert image_file.exists(), 'invalid path : {:}'.format(image_file)
  else:
    raise ValueError('invalid dataset name : {:}'.format(dataset))
  return str(image_file)


def load_APY_bounding_box(xfiles):
  lists = [dataset_root / 'APY' / 'attribute_data' / 'apascal_train.txt',
           dataset_root / 'APY' / 'attribute_data' / 'apascal_test.txt' ,
           dataset_root / 'APY' / 'attribute_data' / 'ayahoo_test.txt']
  contents = []
  for filename in lists:
    with open(filename) as f:
      content = f.readlines()
      contents += content
  contents = [x.strip() for x in contents]
  bboxes, attrs = [], []
  for fname, content in zip(xfiles, contents):
    parts = content.split(' ')
    name, cls = parts[0], parts[1]
    bbox  = tuple([int(x) for x in parts[2:6]])
    attrs.append( tuple([int(x) for x in parts[6:]]) )
    assert fname.split('/')[-1] == name, '{:} vs {:}'.format(fname, name)
    bboxes.append( bbox )
  return bboxes


def preprocess_data(dataset_name, save_file):
  # split dataset
  split_dir      = dataset_root / "xlsa17" / "data" / dataset_name
  matcontent     = sio.loadmat(str(split_dir / "res101.mat"))
  img_feature    = matcontent['features'].T
  img_label      = matcontent['labels'].astype(int).squeeze() - 1
  image_files    = matcontent['image_files']
  image_files    = [x[0][0] for x in image_files]
  print('{:} process {:4s} with {:6d} images'.format(time_string(), dataset_name, len(image_files)))
  if dataset_name == 'APY': boxes = load_APY_bounding_box(image_files)
  else                    : boxes = None
  image_files    = [get_real_path(dataset_name, x, boxes, idx) for idx, x in enumerate(image_files)]
  # load info
  matcontent     = sio.loadmat(str(split_dir / "att_splits.mat"))
  allclasses     = [x[0][0] for x in matcontent['allclasses_names']]

  # get specific information
  trainval_loc        = matcontent['trainval_loc'].squeeze() - 1
  trainval_feature    = img_feature[trainval_loc]
  trainval_label      = img_label[trainval_loc]
  trainval_classes    = set(trainval_label.tolist())
  trainval_files      = [image_files[x] for x in trainval_loc]

  test_seen_loc       = matcontent['test_seen_loc'].squeeze() - 1
  test_seen_feature   = img_feature[test_seen_loc]
  test_seen_label     = img_label[test_seen_loc]
  test_seen_files     = [image_files[x] for x in test_seen_loc]

  test_unseen_loc     = matcontent['test_unseen_loc'].squeeze() - 1
  test_unseen_feature = img_feature[test_unseen_loc]
  test_unseen_label   = img_label[test_unseen_loc]
  test_unseen_files   = [image_files[x] for x in test_unseen_loc]

  attributes          = torch.from_numpy(matcontent['att'].T)
  if dataset_name in ['APY', 'SUN']:
    ori_attributes    = torch.from_numpy(matcontent['original_att'].T) * 100
  elif dataset_name in ['AWA1', 'AWA2', 'CUB']:
    ori_attributes    = torch.from_numpy(matcontent['original_att'].T)
  else:
    raise ValueError('invalid dataset-name: {:}'.format(dataset_name))

  train_classes  = sorted( list(set(trainval_label.tolist())) )
  unseen_classes = sorted( list(set(test_unseen_label.tolist())) )

  all_info = {'allclasses'         : allclasses,                         # the list of all classes
              'train_classes'      : train_classes,                      # the list of train classes
              'unseen_classes'     : unseen_classes,                     # the list of unseen classes
              'image_files'        : image_files,
              'image_labels'       : img_label,
              'trainval_feature'   : torch.from_numpy(trainval_feature), # the PyTorch tensor, 23527 * 2048
              'trainval_label'     : trainval_label.tolist(),            # a list of 23527 labels
              'trainval_files'     : trainval_files,
              'test_seen_feature'  : torch.from_numpy(test_seen_feature),# the PyTorch tensor, 5882 * 2048
              'test_seen_label'    : test_seen_label.tolist(),           # a list of 5882 labels
              'test_seen_files'    : test_seen_files,
              'test_unseen_feature': torch.from_numpy(test_unseen_feature), # the PyTorch tensor, 7913 * 2048
              'test_unseen_label'  : test_unseen_label.tolist(),         # a list of 7913 labels
              'test_unseen_files'  : test_unseen_files,
              'attributes'         : attributes,                         # a 50 * 85 PyTorch tensor
              'ori_attributes'     : ori_attributes                      # a 50 * 85 PyTorch tensor
           }
  torch.save(all_info, save_file)
  print('Save all-info into {:}, file size : {:.2f} GB'.format(save_file, os.path.getsize(save_file)/1e9))


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Prepare the dataset file.")
  parser.add_argument('--name', type=str, choices=['APY', 'AWA2', 'CUB', 'SUN'], help='The dataset name.')
  xargs = parser.parse_args()
  save_file = save_dir / 'x-{:}-data-image.pth'.format(xargs.name)
  preprocess_data(xargs.name, save_file)
  """
  names = ['APY', 'AWA2', 'CUB', 'SUN']
  for name in names:
    save_file = save_dir / 'x-{:}-data-image.pth'.format(name)
    preprocess_data(name, save_file)
  """
