import torch
import torch.nn as nn
import torch.nn.functional as F


class CombineNetV2(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, semantic_module, relation_module):
    super(CombineNetV2, self).__init__()
    self.semantic_module = semantic_module
    self.relation_module = relation_module

  def get_attention(self, semantics):
    if self.semantic_module.require_adj:
      new_semantics = self.semantic_module(semantics, adj)
    else:
      new_semantics = self.semantic_module(semantics)
    return self.relation_module.get_attention(new_semantics)

  def get_semantic_list(self, semantics):
    assert self.semantic_module.require_adj == False
    with torch.no_grad():
      semantics_2 = self.semantic_module(semantics)
      if hasattr(self.relation_module, 'get_new_attribute'):
        semantics_3, _ = self.relation_module.get_new_attribute(semantics_2)
        return [semantics, semantics_2, semantics_3]
      else:
        return [semantics, semantics_2]

  def forward(self, image_feats, img_proto, semantics, adj):
    if self.semantic_module.require_adj:
      new_semantics = self.semantic_module(semantics, adj)
    else:
      new_semantics = self.semantic_module(semantics)
    relations = self.relation_module(image_feats, img_proto, new_semantics, adj)
    return relations

