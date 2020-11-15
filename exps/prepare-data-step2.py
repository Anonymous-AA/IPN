###################################################################
# Update the cache dataset file by the following commands:
# [AWA2] python exps/prepare-data-step2.py --name AWA2
# [APY]  python exps/prepare-data-step2.py --name APY
#
###################################################################
import torch, sys, os, time, argparse
from pathlib import Path
from torchvision import transforms
from collections import defaultdict
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from models   import obtain_backbone
from datasets import SIMPLE_DATA
from config_utils import time_string, AverageMeter, convert_secs2time

root_dir = Path.home() / 'zshot-data'

def extract_rotate_feats(dataset):
  data_root = '{:}/info-files/x-{:}-data-image.pth'.format(root_dir, dataset)
  xdata     = torch.load(data_root)
  files     = xdata['image_files']
  save_dir  = root_dir / 'rotate-infos' / dataset
  save_dir.mkdir(parents=True, exist_ok=True)
  imagepath2featpath = dict()
  avoid_duplicate    = set()

  cnn_name = 'resnet101'
  backbone = obtain_backbone(cnn_name).cuda()
  backbone = torch.nn.DataParallel(backbone)
  backbone.eval()
  #print("CNN-Backbone ----> \n {:}".format(backbone))

  # 3 is the number of augmentations
  simple_data   = SIMPLE_DATA(files, 3, 'imagenet')
  #simple_loader = torch.utils.data.DataLoader(simple_data, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
  simple_loader = torch.utils.data.DataLoader(simple_data, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

  batch_time, xend = AverageMeter(), time.time()
  for idx, (indexes, tensor_000, tensor_090, tensor_180, tensor_270) in enumerate(simple_loader):
    with torch.no_grad():
      feats_000 = [backbone(x) for x in tensor_000]
      feats_090 = [backbone(x) for x in tensor_090]
      feats_180 = [backbone(x) for x in tensor_180]
      feats_270 = [backbone(x) for x in tensor_270]
      
      for ii, image_idx in enumerate(indexes):
        x_feats_000 = torch.stack([x[ii] for x in feats_000]).cpu()
        x_feats_090 = torch.stack([x[ii] for x in feats_090]).cpu()
        x_feats_180 = torch.stack([x[ii] for x in feats_180]).cpu()
        x_feats_270 = torch.stack([x[ii] for x in feats_270]).cpu()
        ori_file_p  = Path(files[image_idx.item()])
        save_dir_xx = save_dir / ori_file_p.parent.name 
        save_dir_xx.mkdir(parents=True, exist_ok=True)
        save_f_path = save_dir_xx / (ori_file_p.name.split('.')[0] + '.pth')
        torch.save({'feats-000': x_feats_000,
                    'feats-090': x_feats_090,
                    'feats-180': x_feats_180,
                    'feats-270': x_feats_270}, save_f_path)
        imagepath2featpath[ files[image_idx.item()] ] = str(save_f_path)
        assert str(save_f_path) not in avoid_duplicate, 'invalid path : {:}'.format(save_f_path)
        avoid_duplicate.add( str(save_f_path) )
    need_time = convert_secs2time(batch_time.val * (len(simple_loader)-idx), True)
    print ('{:} : {:5d} / {:5d} : {:} : {:}'.format(time_string(), idx, len(simple_loader), need_time, save_f_path))
    batch_time.update(time.time() - xend)
    xend = time.time()
  xdata['image2feat'] = imagepath2featpath
  torch.save(xdata, data_root)
  print('Update all-info in {:}, file size : {:.2f} GB'.format(data_root, os.path.getsize(data_root)/1e9))


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Prepare the dataset file.")
  parser.add_argument('--name', type=str, choices=['APY', 'AWA2', 'CUB', 'SUN'], help='The dataset name.')
  xargs = parser.parse_args()
  extract_rotate_feats(xargs.name)
  """
  datasets   = ['APY', 'SUN', 'AWA2', 'CUB']
  for data_name in datasets:
    print('{:} start creating {:}'.format(time_string(), data_name))
    extract_rotate_feats(data_name)
    print('finish creating {:}\n'.format(data_name))
  """
