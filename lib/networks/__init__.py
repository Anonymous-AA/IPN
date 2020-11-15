#
# Isometric Propagation Network for Generalized Zero-shot Learning
#
# This directory contains the defination for different networks.
#

def obtain_relation_models(name, att_dim, image_dim):
  model_name = name.split('-')[0]
  if model_name == 'IPN':
    from .IPN import IPN
    _, att_C, hidden_C, degree, T = name.split('-')
    return IPN(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  elif model_name == 'IPNX':
    from .IPNX import IPNX
    _, att_C, hidden_C, degree, T, tau = name.split('-')
    return IPNX(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree), int(tau))
  elif model_name == 'Vanilla':
    from .relation_networks import GNNVanillaRelationNet
    _, hidden_C, n_hop, temperature = name.split('-')
    hidden_C, n_hop, temperature = int(hidden_C), int(n_hop), int(temperature)
    return GNNVanillaRelationNet(att_dim, image_dim, hidden_C, n_hop, temperature)
  elif model_name == 'V1':
    from .relation_networks import GNNV1RelationNet
    _, hidden_C = name.split('-')
    return GNNV1RelationNet(att_dim, image_dim, int(hidden_C))
  elif model_name == 'GAT':
    _, hidden_C = name.split('-')
    return GATRelationNet(att_dim, image_dim, int(hidden_C))
  elif model_name == 'PPN':
    _, att_C, hidden_C, degree, T = name.split('-')
    return PPNRelationNet(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  elif model_name == 'PPNV2':
    _, att_C, hidden_C, degree, T, hop = name.split('-')
    return PPNV2RelationNet(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree), int(hop))
  elif model_name == 'PPNV3':
    from .relation_networks import PPNV3RelationNet
    _, hidden_C, degree, T = name.split('-')
    return PPNV3RelationNet(att_dim, image_dim, int(hidden_C), int(T), int(degree))
  elif model_name == 'DualPN':
    from .DualPN import DualPN
    _, att_C, hidden_C, degree, T = name.split('-')
    return DualPN(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  elif model_name == 'DualPN_endecoder':
    from .DualPN import DualPN_endecoder 
    _, att_C, hidden_C, degree, T = name.split('-')
    return DualPN_endecoder(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  elif model_name == 'DualPN_DualAtt':
    from .DualPN import DualPN_DualAtt
    _, att_C, hidden_C, degree, T = name.split('-')
    return DualPN_DualAtt(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  elif model_name == 'DualPN_DualAttDiff':
    from .DualPN import DualPN_DualAttDiff
    _, att_C, hidden_C, degree, T = name.split('-')
    return DualPN_DualAttDiff(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  elif model_name == 'DualPN_Reconstruct':
    from .DualPN import DualPN_Reconstruct
    _, att_C, hidden_C, degree, T = name.split('-')
    return DualPN_Reconstruct(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  elif model_name == 'DualPN_DualClassifier':
    from .DualPN import DualPN_DualClassifier
    _, att_C, hidden_C, degree, T = name.split('-')
    return DualPN_DualClassifier(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))
  else:
    raise ValueError('invalid model_name : {:}'.format(model_name))


def obtain_semantic_models(name, field_centers):
  model_name = name.split('-')[0]
  if model_name == 'Linear':
    from .semantic_networks import LinearEnsemble
    _, out_dim = name.split('-')
    return LinearEnsemble(field_centers, int(out_dim))
  elif model_name == 'identity':
    from .semantic_networks import IdentityMapping 
    _, out_dim = name.split('-')
    return IdentityMapping(int(out_dim))
  elif model_name == 'Scale':
    from .semantic_networks import ScaleEnsemble
    _, out_dim, scale_dim = name.split('-')
    return ScaleEnsemble(field_centers, int(out_dim), int(scale_dim))
  elif 'Norm' in model_name:
    from .semantic_networks import NormEnsemble
    _, out_dim = name.split('-')
    return NormEnsemble(field_centers, int(out_dim), model_name)
  elif model_name == 'ADJ':
    from .semantic_networks import AdjEnsemble
    _, out_dim, temperature = name.split('-')
    return AdjEnsemble(field_centers, int(out_dim), int(temperature))
  else:
    raise ValueError('invalid model_name : {:}'.format(model_name))


def obtain_combine_models_v2(s_name, r_name, field_centers, image_dim):
  s_model = obtain_semantic_models(s_name, field_centers)
  r_model = obtain_relation_models(r_name, s_model.out_dim, image_dim)
  from .combine_networks import CombineNetV2
  return CombineNetV2(s_model, r_model)
