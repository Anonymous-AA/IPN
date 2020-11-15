from .distance_utils import distance_func


def obtain_backbone(cnn_name, config={'pretrained':True, 'pretrain_path': None}):
  from .backbone import resnet18, resnet50, resnet101 
  if cnn_name.lower() == 'resnet18':
    model = resnet18(pretrained=config['pretrained'])
  elif cnn_name.lower() == 'resnet50':
    model = resnet50(pretrained=config['pretrained'])
  elif cnn_name.lower() == 'resnet101':
    model = resnet101(pretrained=config['pretrained'])
  else:
    raise ValueError('invalid name : {:}'.format(name))
  if 'pretrain_path' in config and config['pretrain_path'] is not None:
    data = torch.load(config['pretrain_path'], map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in data['net-state-dict'].items():
      name = k[7:] # remove `module.`
      new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
  return model
