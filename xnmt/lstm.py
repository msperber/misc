import dynet as dy
from residual import PseudoState
from batch_norm import BatchNorm
import numpy as np

class PythonLSTMBuilder:
  """
  This is a Python version of the vanilla LSTM.
  In contrast to the C++ version, this one does not support multiple layers or dropout.
  """
  def __init__(self, layers, input_dim, hidden_dim, model):
    if layers!=1: raise RuntimeError("PythonLSTMBuilder supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
  
    # [i; f; o; g]
    self.p_x2i = model.add_parameters(dim=(hidden_dim*4, input_dim)) # Ws
    self.p_h2i = model.add_parameters(dim=(hidden_dim*4, hidden_dim)) # Wfs
    self.p_bi  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0)) # Ufs
    
  def whoami(self): return "PythonLSTMBuilder"
  
  def set_dropout(self, p):
    if p>0.0: raise RuntimeError("PythonLSTMBuilder does not support dropout")
  def disable_dropout(self):
    pass
  def initial_state(self):
    return self
  def add_inputs(self):
    pass
  def transduce(self, xs):
    i_x2i = dy.parameter(self.p_x2i)
    i_h2i = dy.parameter(self.p_h2i)
    i_bi = dy.parameter(self.p_bi)
    h = []
    c = []
    for i, x_t in enumerate(xs):
      if i==0:
        tmp = dy.affine_transform([i_bi, i_x2i, x_t])
      else:
        tmp = dy.affine_transform([i_bi, i_x2i, x_t, i_h2i, h[-1]])
      i_ait = dy.pick_range(tmp, 0, self.hidden_dim)
      i_aft = dy.pick_range(tmp, self.hidden_dim, self.hidden_dim*2)
      i_aot = dy.pick_range(tmp, self.hidden_dim*2, self.hidden_dim*3)
      i_agt = dy.pick_range(tmp, self.hidden_dim*3, self.hidden_dim*4)
      i_it = dy.logistic(i_ait)
      i_ft = dy.logistic(i_aft + 1.0)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if i==0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        c.append(dy.cmult(i_ft, c[-1]) + dy.cmult(i_it, i_gt))
      h.append(dy.cmult(i_ot, dy.tanh(c[-1])))
    return h
  
class ConvLSTMBuilder:
  """
  This is a ConvLSTM implementation using a single bidirectional layer.
  """
  def __init__(self, layers, input_dim, model, chn_dim=3, num_filters=32):
    if layers!=1: raise RuntimeError("ConvLSTMBuilder supports only exactly one layer")
    if input_dim%chn_dim!=0: raise RuntimeError("input_dim must be divisible by chn_dim")
    self.input_dim = input_dim

    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 1
    self.filter_size_freq = 3
    normalInit=dy.NormalInitializer(0, 0.1)

    self.params = {}
    for direction in ["fwd","bwd"]:
      self.params["x2all_" + direction] = \
          model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, 
                                    self.chn_dim, self.num_filters * 4),
                               init=normalInit)
      self.params["h2all_" + direction] = \
          model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, 
                                    self.chn_dim, self.num_filters * 4),
                               init=normalInit)
      self.params["b_" + direction] = \
          model.add_parameters(dim=(self.num_filters * 4,), init=normalInit)
    
  def whoami(self): return "ConvLSTMBuilder"
  
  def set_dropout(self, p):
    if p>0.0: raise RuntimeError("ConvLSTMBuilder does not support dropout")
  def disable_dropout(self):
    pass
  def initial_state(self):
    return self
  def add_inputs(self):
    pass
  def transduce(self, es, train=False):
    es_expr = es.as_tensor()
    sent_len = es_expr.dim()[0][0]
    batch_size=es_expr.dim()[1]
    
    if es_expr.dim() == ((sent_len, self.freq_dim, self.chn_dim), batch_size):
      es_chn = es_expr
    else:
      es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size) # ((276, 80, 3), 1)

    h_out = {}
    for direction in ["fwd", "bwd"]:
      # input convolutions
      x_filtered_all = dy.conv2d_bias(es_chn, dy.parameter(self.params["x2all_" + direction]), dy.parameter(self.params["b_" + direction]), stride=(1,1), is_valid=False)
      x_filtered_i = dy.pick_range(x_filtered_all, 0, self.num_filters, 2)
      x_filtered_f = dy.pick_range(x_filtered_all, self.num_filters, self.num_filters*2, 2)
      x_filtered_o = dy.pick_range(x_filtered_all, self.num_filters*2, self.num_filters*3, 2)
      x_filtered_g = dy.pick_range(x_filtered_all, self.num_filters*3, self.num_filters*4, 2)

      # convert tensor into list
      xs_i = [dy.pick_range(x_filtered_i, i, i+1) for i in range(x_filtered_i.dim()[0][0])]
      xs_f = [dy.pick_range(x_filtered_f, i, i+1) for i in range(x_filtered_f.dim()[0][0])]
      xs_o = [dy.pick_range(x_filtered_o, i, i+1) for i in range(x_filtered_o.dim()[0][0])]
      xs_g = [dy.pick_range(x_filtered_g, i, i+1) for i in range(x_filtered_g.dim()[0][0])]

      h = []
      c = []
      for input_pos in range(len(xs_i)):
        directional_pos = input_pos if direction=="fwd" else len(xs_i)-input_pos-1
        i_ait = xs_i[directional_pos]
        i_aft = xs_f[directional_pos]
        i_aot = xs_o[directional_pos]
        i_agt = xs_g[directional_pos]
        if input_pos>0:
          # recurrent convolutions
          wh_all = dy.conv2d(h[-1], dy.parameter(self.params["h2all_" + direction]), stride=(1,1), is_valid=False)
          wh_i = dy.pick_range(wh_all, 0, self.num_filters, 2)
          wh_f = dy.pick_range(wh_all, self.num_filters, self.num_filters*2, 2)
          wh_o = dy.pick_range(wh_all, self.num_filters*2, self.num_filters*3, 2)
          wh_g = dy.pick_range(wh_all, self.num_filters*3, self.num_filters*4, 2)
          i_ait += wh_i
          i_aft += wh_f
          i_aot += wh_o
          i_agt += wh_g
        
        # standard LSTM logic
        i_it = dy.logistic(i_ait)
        i_ft = dy.logistic(i_aft + 1.0)
        i_ot = dy.logistic(i_aot)
        i_gt = dy.tanh(i_agt)
        if input_pos==0:
          c.append(dy.cmult(i_it, i_gt))
        else:
          c.append(dy.cmult(i_ft, c[-1]) + dy.cmult(i_it, i_gt))
        h_t = dy.cmult(i_ot, dy.tanh(c[-1]))
        h.append(h_t)
      h_out[direction] = h
    return [dy.concatenate([dy.reshape(state_fwd, (state_fwd.dim()[0][1] * state_fwd.dim()[0][2],), batch_size=batch_size),
                            dy.reshape(state_bwd, (state_bwd.dim()[0][1] * state_bwd.dim()[0][2],), batch_size=batch_size)]) \
            for (state_fwd,state_bwd) in zip(h_out["fwd"], reversed(h_out["bwd"]))]
  

class NetworkInNetworkBiRNNBuilder(object):
  """
  Builder for NiN-interleaved RNNs that delegates to regular RNNs and wires them together.
  See http://iamaaditya.github.io/2016/03/one-by-one-convolution/
  and https://arxiv.org/pdf/1610.03022.pdf
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory,
               batch_norm=False, stride=1, num_projections=1):
    """
    :param num_layers: depth of the network
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. VanillaLSTMBuilder
    :param batch_norm: uses batch norm between projection and non-linearity
    :param stride: in (first) projection layer, concatenate n frames and use the projection for subsampling
    :param num_projections: number of projections (only the first projection does any subsampling)
    """
    assert num_layers > 0
    assert hidden_dim % 2 == 0
    self.builder_layers = []
    self.hidden_dim = hidden_dim
    self.stride=stride
    self.num_projections = num_projections
    f = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    b = rnn_builder_factory(1, input_dim, hidden_dim / 2, model)
    self.use_bn = batch_norm
    bn = BatchNorm(model, hidden_dim, 2)
    self.builder_layers.append((f, b, bn))
    for _ in xrange(num_layers - 1):
      f = rnn_builder_factory(1, hidden_dim, hidden_dim / 2, model)
      b = rnn_builder_factory(1, hidden_dim, hidden_dim / 2, model)
      bn = BatchNorm(model, hidden_dim, 2) if batch_norm else None
      self.builder_layers.append((f, b, bn))
    self.lintransf_layers = []
    for _ in xrange(num_layers):
      proj_params = []
      for proj_i in range(num_projections):
        if proj_i==0:
          proj_param = model.add_parameters(dim=(hidden_dim, hidden_dim*stride))
        else:
          proj_param = model.add_parameters(dim=(hidden_dim, hidden_dim))
        proj_params.append(proj_param)
      self.lintransf_layers.append(proj_params)
    self.train = True

  def whoami(self): return "NetworkInNetworkBiRNNBuilder"

  def set_dropout(self, p):
    for (fb, bb, bn) in self.builder_layers:
      fb.set_dropout(p)
      bb.set_dropout(p)
  def disable_dropout(self):
    for (fb, bb, bn) in self.builder_layers:
      fb.disable_dropout()
      bb.disable_dropout()

  def transduce(self, es):
    """
    returns the list of output Expressions obtained by adding the given inputs
    to the current state, one by one, to both the forward and backward RNNs, 
    and concatenating.
        
    :param es: a list of Expression

    """
    for layer_i, (fb, bb, bn) in enumerate(self.builder_layers):
      fs = fb.initial_state().transduce(es)
      bs = bb.initial_state().transduce(reversed(es))
      interleaved = []
      for pos in range(len(fs)):
        interleaved.append(fs[pos])
        interleaved.append(bs[-pos-1])
      es = self.apply_nin_projections(self.lintransf_layers[layer_i], interleaved, bn, stride=self.stride*2)
    return es
  def apply_nin_projections(self, lintransf_params, es, bn, stride):
    for proj_i in range(len(lintransf_params)):
      es = self.apply_one_nin(es, bn, stride if proj_i==0 else 1, lintransf_params[proj_i])
    return es
  def apply_one_nin(self, es, bn, stride, lintransf):
    batch_size = es[0].dim()[1]
    if len(es)%stride!=0:
      zero_pad = dy.inputTensor(np.zeros(es[0].dim()[0]+(es[0].dim()[1],)), batched=True)
      es.extend([zero_pad] * (stride-len(es)%stride))
    projections = []
    lintransf_param = dy.parameter(lintransf)
    for pos in range(0, len(es), stride):
      concat = dy.concatenate(es[pos:pos+stride])
      proj = lintransf_param * concat
      projections.append(proj)
    if self.use_bn:
      bn_layer = bn.bn_expr(dy.concatenate([dy.reshape(x, (1,self.hidden_dim), batch_size=batch_size) for x in projections], 
                                0), 
                 train=self.train)
      nonlin = dy.rectify(bn_layer)
      es = [dy.pick(nonlin, i) for i in range(nonlin.dim()[0][0])]
    else:
      es = []
      for proj in projections:
        nonlin = dy.rectify(proj)
        es.append(nonlin)
    return es
    
  def initial_state(self):
    return PseudoState(self)
