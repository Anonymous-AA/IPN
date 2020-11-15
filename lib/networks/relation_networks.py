import torch.nn as nn
import torch.nn.functional as F
import math, torch
from models import distance_func


class GNNVanillaRelationNet(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, hidden_C, n_hop, T=5):
    super(GNNVanillaRelationNet, self).__init__()
    self.fc_img   = nn.Linear(image_dim, hidden_C, bias=False)
    self.prop_att = nn.Linear(att_dim, att_dim)
    self.fc_att   = nn.Linear(att_dim, hidden_C)
    self.fc2      = nn.Linear(hidden_C, 1)
    self.n_hop    = n_hop
    self.T        = T

  def extra_repr(self):
    return ('{name}(n-hop={n_hop:}, temperature={T:})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, image_feats, attributes, adj):
    batch, feat_dim = image_feats.shape
    cls_num, at_dim = attributes.shape
    assert adj.dim() == 2 and adj.size(0) == adj.size(1) == cls_num
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    #batch_attFF_ext = attributes.view(1, cls_num, -1).expand(batch, cls_num, at_dim)
    #relation_pairs  = torch.cat((image_feats_ext, batch_attFF_ext), dim=2)

    #adj_exp      = torch.exp(adj * self.T) * (adj>0).float()
    #adj_norm     = adj_exp / torch.sum(adj_exp, dim=1, keepdim=True)
    adj_norm     = F.softmax(adj * -self.T, dim=1)
    image_hidden = self.fc_img(image_feats_ext)
    attrr_hidden = attributes
    for i in range(self.n_hop):
      attrr_hidden = self.prop_att(attrr_hidden)
      attrr_hidden = torch.mm(adj_norm, attrr_hidden)
      attrr_hidden = F.relu(attrr_hidden)

    attrr_hidden = self.fc_att(attrr_hidden)
    features = F.relu( image_hidden + attrr_hidden.view(1, cls_num, -1) )
    outputs = self.fc2( features )
    return outputs.view(batch, cls_num)


class GNNV1RelationNet(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, hidden_C):
    super(GNNV1RelationNet, self).__init__()
    self.fc1      = nn.Linear(att_dim+image_dim, hidden_C)
    self.fc2      = nn.Linear(hidden_C, 1)

  def forward(self, image_feats, attributes, _):
    batch, feat_dim = image_feats.shape
    cls_num, at_dim = attributes.shape
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    batch_attFF_ext = attributes.view(1, cls_num, -1).expand(batch, cls_num,  at_dim)
    relation_pairs  = torch.cat((image_feats_ext, batch_attFF_ext), dim=2)

    x = F.relu(self.fc1(relation_pairs))
    outputs = self.fc2(x)
    return outputs.view(batch, cls_num)


class GATRelationNet(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, hidden_C):
    super(GATRelationNet, self).__init__()
    self.att_w     = nn.Parameter(torch.Tensor(att_dim, hidden_C))
    self.att_a     = nn.Parameter(torch.Tensor(2*hidden_C, 1))
    nn.init.xavier_uniform_(self.att_w.data, gain=1.414)
    nn.init.xavier_uniform_(self.att_a.data, gain=1.414)
    self.leakyrelu = nn.LeakyReLU(0.2)
    self.dropout   = 0.6
    self.img_w     = nn.Parameter(torch.Tensor(image_dim, hidden_C))
    self.sem_w     = nn.Parameter(torch.Tensor(att_dim, hidden_C))
    self.sem_b     = nn.Parameter(torch.Tensor(1, hidden_C))
    nn.init.xavier_uniform_(self.img_w.data, gain=1.414)
    nn.init.xavier_uniform_(self.sem_w.data, gain=1.414)
    nn.init.xavier_uniform_(self.sem_b.data, gain=1.414)
    self.fc        = nn.Linear(hidden_C, 1)

  def forward(self, image_feats, attributes, _):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    att_h    = torch.mm(attributes, self.att_w)

    att_h_1  = att_h.view(cls_num, 1, -1).repeat(1, cls_num, 1)
    att_h_2  = att_h.view(1, cls_num, -1).repeat(cls_num, 1, 1)
    att_e    = self.leakyrelu(torch.matmul(torch.cat([att_h_1, att_h_2], dim=2), self.att_a)).squeeze(2)
    #zero_vec = -9e15*torch.ones_like(att_e)
    attention = F.softmax(att_e, dim=1)
    attention = F.dropout(attention, self.dropout, training=self.training)
    att_outs  = torch.matmul(attention, att_h)

    batch, feat_dim = image_feats.shape
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    att_feats_ext   = att_outs.view(1, cls_num, -1).expand(batch, cls_num, feat_dim)

    hidden_feats    = F.relu( torch.matmul(image_feats_ext, self.img_w) + torch.matmul(att_feats_ext, self.sem_w) + self.sem_b )
    outputs         = self.fc( hidden_feats )
    return outputs.view(batch, cls_num)


class PPNRelationNet(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, att_hC, hidden_C, T, degree):
    super(PPNRelationNet, self).__init__()
    self.att_g     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    nn.init.kaiming_uniform_(self.att_g, a=math.sqrt(5))
    self.T         = T
    assert degree >= 0 and degree < 100, 'invalid degree : {:}'.format(degree)
    self.degree    = degree
    self.thresh    = math.cos(math.pi*degree/180)
    #self.att_h     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    #init.kaiming_uniform_(self.att_h, a=math.sqrt(5))
    #nn.init.xavier_uniform_(self.att_g.data, gain=1.414)
    #nn.init.xavier_uniform_(self.att_h.data, gain=1.414)
    self.img_w     = nn.Parameter(torch.Tensor(image_dim, hidden_C))
    self.sem_w     = nn.Parameter(torch.Tensor(att_dim, hidden_C))
    self.sem_b     = nn.Parameter(torch.Tensor(1, hidden_C))
    nn.init.kaiming_uniform_(self.img_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_b.data, a=math.sqrt(5))
    fan_in, _      = nn.init._calculate_fan_in_and_fan_out(self.img_w)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(self.sem_b, -bound, bound)
    #nn.init.xavier_uniform_(self.img_w.data, gain=1.414)
    #nn.init.xavier_uniform_(self.sem_w.data, gain=1.414)
    #nn.init.xavier_uniform_(self.sem_b.data, gain=1.414)
    self.fc        = nn.Linear(hidden_C, 1)

  def extra_repr(self):
    xshape = 'att-shape: {:}'.format(list(self.att_g.shape)) + ' img-w-shape : {:}'.format(list(self.img_w.shape)) + ' sem-w-shape : {:}'.format(list(self.sem_w.shape))
    return ('{name}(degree={degree:}, thresh={thresh:.3f}, temperature={T:}, '.format(name=self.__class__.__name__, **self.__dict__) + xshape + ')')

  def get_new_attribute(self, attributes):
    att_prop_g = torch.mm(attributes, self.att_g)
    att_prop_h = torch.mm(attributes, self.att_g)
    distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
    zero_vec   = -9e15 * torch.ones_like(distances)
    raw_attss  = torch.where(distances > self.thresh, distances, zero_vec)
    attention  = F.softmax(raw_attss * self.T, dim=1)
    att_outs   = torch.mm(attention, attributes)
    return att_outs, distances > self.thresh

  def forward(self, image_feats, attributes, _):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    att_outs, _ = self.get_new_attribute(attributes)
    #att_outs = attributes

    batch, feat_dim = image_feats.shape
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    att_feats_ext   = att_outs.view(1, cls_num, -1).expand(batch, cls_num, feat_dim)

    hidden_feats    = F.relu( torch.matmul(image_feats_ext, self.img_w) + torch.matmul(att_feats_ext, self.sem_w) + self.sem_b )
    outputs         = self.fc( hidden_feats )
    return outputs.view(batch, cls_num)


class PPNV2RelationNet(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, att_hC, hidden_C, T, degree, n_hop):
    super(PPNV2RelationNet, self).__init__()
    self.att_g     = nn.Parameter(torch.Tensor(att_dim, att_hC))
    self.T         = T
    assert degree >= 0 and degree < 100, 'invalid degree : {:}'.format(degree)
    self.degree    = degree
    self.thresh    = math.cos(math.pi*degree/180)
    assert n_hop >= 0, 'invalid n_hop : {:}'.format(n_hop)
    self.n_hop     = n_hop
    self.img_w     = nn.Parameter(torch.Tensor(image_dim, hidden_C))
    self.sem_w     = nn.Parameter(torch.Tensor(att_dim, hidden_C))
    self.sem_b     = nn.Parameter(torch.Tensor(1, hidden_C))
    self.fc        = nn.Linear(hidden_C, 1)
    self.initialize()

  def initialize(self):
    nn.init.kaiming_uniform_(self.att_g, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.img_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_b.data, a=math.sqrt(5))

  def extra_repr(self):
    xshape = 'att-shape: {:}'.format(list(self.att_g.shape)) + ' img-w-shape : {:}'.format(list(self.img_w.shape)) + ' sem-w-shape : {:}'.format(list(self.sem_w.shape))
    return ('{name}(degree={degree:}, thresh={thresh:.3f}, temperature={T:}, hop={n_hop:}'.format(name=self.__class__.__name__, **self.__dict__) + xshape + ')')

  def get_new_attribute(self, attributes):
    if self.n_hop == 0: return attributes
    for ihop in range(self.n_hop):
      att_prop_g = torch.mm(attributes, self.att_g)
      att_prop_h = torch.mm(attributes, self.att_g)
      distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
      zero_vec   = -9e15 * torch.ones_like(distances)
      raw_attss  = torch.where(distances > self.thresh, distances, zero_vec)
      attention  = F.softmax(raw_attss * self.T, dim=1)
      att_outs   = torch.mm(attention, attributes)
      # update attributes
      attributes = att_outs
    return att_outs, distances > self.thresh

  def forward(self, image_feats, attributes, _):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    if self.n_hop == 0:
      att_outs    = self.get_new_attribute(attributes)
    else:
      att_outs, _ = self.get_new_attribute(attributes)
    #att_outs = attributes

    batch, feat_dim = image_feats.shape
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    att_feats_ext   = att_outs.view(1, cls_num, -1).expand(batch, cls_num, feat_dim)

    hidden_feats    = F.relu( torch.matmul(image_feats_ext, self.img_w) + torch.matmul(att_feats_ext, self.sem_w) + self.sem_b )
    outputs         = self.fc( hidden_feats )
    return outputs.view(batch, cls_num)


class PPNV3RelationNet(nn.Module):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, hidden_C, T, degree):
    super(PPNV3RelationNet, self).__init__()
    self.T         = T
    assert degree >= 0 and degree < 100, 'invalid degree : {:}'.format(degree)
    self.degree    = degree
    self.thresh    = math.cos(math.pi*degree/180)
    self.img_w     = nn.Parameter(torch.Tensor(image_dim, hidden_C))
    self.sem_w     = nn.Parameter(torch.Tensor(att_dim, hidden_C))
    self.sem_b     = nn.Parameter(torch.Tensor(1, hidden_C))
    self.fc        = nn.Linear(hidden_C, 1)
    self.initialize()

  def initialize(self):
    nn.init.kaiming_uniform_(self.img_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_w.data, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.sem_b.data, a=math.sqrt(5))

  def extra_repr(self):
    xshape = ' img-w-shape : {:}'.format(list(self.img_w.shape)) + ' sem-w-shape : {:}'.format(list(self.sem_w.shape))
    return ('{name}(degree={degree:}, thresh={thresh:.3f}, temperature={T:}, '.format(name=self.__class__.__name__, **self.__dict__) + xshape + ')')

  def x_get_new_attribute(self, attributes, adj_distances):
    # normalize into 0 ~ 1
    distances  = 1 / (adj_distances.float() + 1.0)
    zero_vec   = -9e15 * torch.ones_like(distances)
    raw_attss  = torch.where(distances > self.thresh, distances, zero_vec)
    attention  = F.softmax(raw_attss * self.T, dim=1)
    att_outs   = torch.mm(attention, attributes)
    return att_outs, distances > self.thresh

  def forward(self, image_feats, attributes, adj_distances):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    att_outs, _ = self.x_get_new_attribute(attributes, adj_distances)
    #att_outs = attributes

    batch, feat_dim = image_feats.shape
    image_feats_ext = image_feats.view(batch, 1, -1).expand(batch, cls_num, feat_dim)
    att_feats_ext   = att_outs.view(1, cls_num, -1).expand(batch, cls_num, feat_dim)

    hidden_feats    = F.relu( torch.matmul(image_feats_ext, self.img_w) + torch.matmul(att_feats_ext, self.sem_w) + self.sem_b )
    outputs         = self.fc( hidden_feats )
    return outputs.view(batch, cls_num)
