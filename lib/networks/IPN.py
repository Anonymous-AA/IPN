import torch.nn as nn
import torch.nn.functional as F
import math, torch
from models import distance_func


class IPN(nn.Module):
  """Isometric Propagation Network"""
  def __init__(self, att_dim, image_dim, att_hC, hidden_C, T, degree):
    super(IPN, self).__init__()
    self.att_g_att = nn.Parameter(torch.Tensor(att_dim, att_hC))
    nn.init.kaiming_uniform_(self.att_g_att, a=math.sqrt(5))
    self.att_g_img = nn.Parameter(torch.Tensor(att_dim, att_hC))
    nn.init.kaiming_uniform_(self.att_g_img, a=math.sqrt(5))
    self.T         = T
    assert degree >= 0 and degree < 100, 'invalid degree : {:}'.format(degree)
    self.degree    = degree
    self.thresh    = math.cos(math.pi*degree/180)
    self.att_dim   = att_dim
    self.proto_red_dim = att_dim
    self.proto_slim  = nn.Linear(image_dim, self.proto_red_dim)
    self.img_w       = nn.Parameter(torch.Tensor(image_dim, hidden_C))
    self.proto_w     = nn.Parameter(torch.Tensor(att_dim+self.proto_red_dim, hidden_C))
    self.proto_b     = nn.Parameter(torch.Tensor(1, hidden_C))
    nn.init.kaiming_uniform_(self.img_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.proto_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.proto_b.data, a=math.sqrt(5))
    fan_in, _      = nn.init._calculate_fan_in_and_fan_out(self.img_w)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(self.proto_b, -bound, bound)
    self.fc        = nn.Linear(hidden_C, 1)
    self.raw_attention_attribute = None
    self.raw_attention_img       = None 
    self.attention_attribute = None
    self.attention_img       = None 

  def _set_attention_attribute(self, raw_att_att, attention_attribute):
    self.attention_attribute = attention_attribute
    self.raw_attention_attribute = raw_att_att

  def _set_attention_img(self, raw_att_img, attention_img):
    self.attention_img = attention_img
    self.raw_attention_img = raw_att_img

  def get_attention_attribute(self):
    return self.raw_attention_attribute, self.attention_attribute

  def get_attention_img(self):
    return self.raw_attention_img, self.attention_img

  def extra_repr(self):
    xshape = 'att-att-shape: {:}'.format(list(self.att_g_att.shape)) + 'att-img-shape: {:}'.format(list(self.att_g_img.shape)) + ' img-w-shape : {:}'.format(list(self.img_w.shape)) + ' proto-w-shape : {:}'.format(list(self.proto_w.shape))
    return ('{name}(degree={degree:}, thresh={thresh:.3f}, temperature={T:}, '.format(name=self.__class__.__name__, **self.__dict__) + xshape + ')')

  def get_attention(self, attributes, choice="attribute"):
    if choice == "attribute": att_g = self.att_g_att
    elif choice == "img"    : att_g = self.att_g_img
    elif choice == "image"  : att_g = self.att_g_img
    else                    : raise ValueError("invalid choice {:}".format(choice))
    att_prop_g = torch.mm(attributes, att_g)
    att_prop_h = torch.mm(attributes, att_g)
    distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
    zero_vec   = -9e15 * torch.ones_like(distances)
    raw_attss  = torch.where(distances > self.thresh, distances, zero_vec)
    attention  = F.softmax(raw_attss * self.T, dim=1)
    #return raw_attss, attention
    return distances, attention

  def get_new_attribute(self, attributes):
    raw_attss, attention = self.get_attention(attributes, choice="attribute")
    att_outs   = torch.mm(attention, attributes)
    return att_outs, (raw_attss, attention)

  def get_new_img_proto(self, img_proto):
    raw_attss, attention = self.get_attention(img_proto, choice="img")
    att_outs   = torch.mm(attention, img_proto)
    return att_outs, (raw_attss, attention)

  def _propagate(self, img_proto, attributes):
    att_outs      , (raw_att_att, attention_att) = self.get_new_attribute(attributes)
    img_proto_outs, (raw_att_img, attention_img) = self.get_new_img_proto(img_proto) 
    self._set_attention_attribute(raw_att_att, attention_att) 
    self._set_attention_img(raw_att_img, attention_img) 
    #return att_outs, img_proto_outs, distances > self.thresh
    return att_outs, img_proto_outs, attention_att, attention_img

  def forward(self, image_feats, img_proto, attributes, _):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    img_proto = self.proto_slim(img_proto)
    att_outs, img_proto_outs, _, _ = self._propagate(img_proto, attributes)
    outs = torch.cat([att_outs, img_proto_outs], dim=1)
    batch, feat_dim = image_feats.shape
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    att_feats_ext   = outs.view(1, cls_num, -1).expand(batch, cls_num, outs.shape[1])
    hidden_feats    = F.relu( torch.matmul(image_feats_ext, self.img_w) + torch.matmul(att_feats_ext, self.proto_w) + self.proto_b )
    outputs         = self.fc( hidden_feats )
    return outputs.view(batch, cls_num) 

