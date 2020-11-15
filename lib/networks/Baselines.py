import torch.nn as nn
import torch.nn.functional as F
import torch


class ExpertModule(nn.Module):

  def __init__(self, field_center, out_dim):
    super(ExpertModule, self).__init__()
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.register_buffer('field_center', field_center.clone())
    self.fc = nn.Linear(field_center.numel(), out_dim)

  def forward(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    response = F.relu(self.fc(input_offsets))
    return response


class CooperationModule(nn.Module):

  def __init__(self, field_centers, out_dim):
    super(CooperationModule, self).__init__()
    self.individuals = nn.ModuleList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    for i in range(field_centers.shape[0]):
      layer = ExpertModule(field_centers[i], out_dim)
      self.individuals.append( layer )

  def forward(self, semantic_vec):
    responses = [indiv(semantic_vec) for indiv in self.individuals]
    feature_anchor = sum(responses)
    return feature_anchor


class RelationModule(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, in_C, hidden_C):
    super(RelationModule, self).__init__()
    self.fc1 = nn.Linear(in_C, hidden_C)
    self.fc2 = nn.Linear(hidden_C, 1)

  def forward(self, image_feats, attributes):
    batch, feat_dim = image_feats.shape
    cls_num, at_dim = attributes.shape
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    batch_attFF_ext = attributes.view(1, cls_num, -1).expand(batch, cls_num, feat_dim)
    relation_pairs  = torch.cat((image_feats_ext, batch_attFF_ext), dim=2)

    x = F.relu(self.fc1(relation_pairs))
    outputs = torch.sigmoid(self.fc2(x))
    return outputs.view(batch, cls_num)
