import torch.nn as nn
import torch.nn.functional as F
import torch


class LinearModule(nn.Module):

  def __init__(self, field_center, out_dim):
    super(LinearModule, self).__init__()
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.register_buffer('field_center', field_center.clone())
    self.fc = nn.Linear(field_center.numel(), out_dim)

  def forward(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    response = F.relu(self.fc(input_offsets))
    return response


class LinearEnsemble(nn.Module):

  def __init__(self, field_centers, out_dim):
    super(LinearEnsemble, self).__init__()
    self.individuals = nn.ModuleList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    self.require_adj = False
    for i in range(field_centers.shape[0]):
      layer = LinearModule(field_centers[i], out_dim)
      self.individuals.append( layer )

  def forward(self, semantic_vec):
    responses = [indiv(semantic_vec) for indiv in self.individuals]
    feature_anchor = sum(responses)
    return feature_anchor


class IdentityMapping(nn.Module):

  def __init__(self, out_dim):
    super(IdentityMapping, self).__init__()
    self.require_adj = False
    self.out_dim     = out_dim

  def forward(self, semantic_vec):
    return semantic_vec


class ScaleModule(nn.Module):

  def __init__(self, field_center, xdim):
    super(ScaleModule, self).__init__()
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.register_buffer('field_center', field_center.clone())
    if xdim == 1:
      self.register_parameter('scale', nn.Parameter(torch.ones((1))))
      self.register_parameter('bias' , nn.Parameter(torch.zeros((1))))
    elif xdim == 2:
      self.register_parameter('scale', nn.Parameter(torch.ones((1, field_center.numel()))))
      self.register_parameter('bias' , nn.Parameter(torch.zeros((1, field_center.numel()))))
    else: raise ValueError('invalid xdim = {:}'.format(xdim))

  def __repr__(self):
    return ('{:}(center={:}, scale={:}, bias={:})'.format(self.__class__.__name__, list(self.field_center.shape), list(self.scale.shape), list(self.bias.shape)))

  def forward(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    input_normalz = input_offsets * self.scale + self.bias
    response      = F.leaky_relu(input_normalz, 0.1, True)
    return response


class ScaleEnsemble(nn.Module):

  def __init__(self, field_centers, out_dim, scale_dim):
    super(ScaleEnsemble, self).__init__()
    self.individuals = nn.ModuleList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    for i in range(field_centers.shape[0]):
      layer = ScaleModule(field_centers[i], scale_dim)
      self.individuals.append( layer )
    self.fc = nn.Linear(field_centers.shape[1], out_dim)
    self.relu = nn.ReLU()
    #if has_relu: self.relu = nn.ReLU()
    #else       : self.relu = None

  def forward(self, semantic_vec):
    responses = [indiv(semantic_vec) for indiv in self.individuals]
    features  = sum(responses) / len(responses)
    features  = self.fc(features)
    if self.relu is None: outs = features
    else                : outs = self.relu(features)
    return outs


class NormModule(nn.Module):

  def __init__(self, field_center, xtype):
    super(NormModule, self).__init__()
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.register_buffer('field_center', field_center.clone())
    if xtype.startswith('GroupNorm'):
      _, num_groups = xtype.split('.')
      self.norm = nn.GroupNorm(int(num_groups), field_center.numel())
    elif xtype == 'LayerNorm':
      self.norm = nn.LayerNorm(field_center.numel())
    else: raise ValueError('invalid xdim = {:}'.format(xtype))

  def forward(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    input_normalz = self.norm(input_offsets)
    response      = F.leaky_relu(input_normalz, 0.01, True)
    return response

class NormEnsemble(nn.Module):

  def __init__(self, field_centers, out_dim, xtype):
    super(NormEnsemble, self).__init__()
    self.individuals = nn.ModuleList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    for i in range(field_centers.shape[0]):
      layer = NormModule(field_centers[i], xtype)
      self.individuals.append( layer )
    self.fc = nn.Linear(field_centers.shape[1], out_dim)
    self.relu = nn.ReLU()
    self.require_adj = False
    #if has_relu: self.relu = nn.ReLU()
    #else       : self.relu = None

  def forward(self, semantic_vec):
    responses = [indiv(semantic_vec) for indiv in self.individuals]
    features  = sum(responses) / len(responses)
    features  = self.fc(features)
    if self.relu is None: outs = features
    else                : outs = self.relu(features)
    return outs


class ADJModule(nn.Module):

  def __init__(self, field_center, T):
    super(ADJModule, self).__init__()
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.register_buffer('field_center', field_center.clone())
    self.T = T

  def forward(self, semantic_vec, adj):
    input_offsets = semantic_vec - self.field_center
    adj_norm      = F.softmax(adj * -self.T, dim=1)
    input_normalz = torch.mm(adj_norm, input_offsets)
    response      = F.leaky_relu(input_normalz, 0.01, True)
    return response


class AdjEnsemble(nn.Module):

  def __init__(self, field_centers, out_dim, T):
    super(AdjEnsemble, self).__init__()
    self.individuals = nn.ModuleList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    for i in range(field_centers.shape[0]):
      layer = ADJModule(field_centers[i], T)
      self.individuals.append( layer )
    self.fc = nn.Linear(field_centers.shape[1], out_dim)
    self.relu = nn.ReLU()
    self.require_adj = True

  def forward(self, semantic_vec, adj):
    responses = [indiv(semantic_vec, adj) for indiv in self.individuals]
    features  = sum(responses) / len(responses)
    features  = self.fc(features)
    return self.relu(features)
