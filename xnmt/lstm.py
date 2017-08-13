import dynet as dy
from residual import PseudoState
from batch_norm import BatchNorm
import numpy as np


def builder_for_spec(spec):
  if spec=="vanilla":
    return dy.VanillaLSTMBuilder
  elif spec=="compact":
    return dy.CompactVanillaLSTMBuilder
  else:
    raise Exception("unknown LSTM spec %s" % spec)
  

class LSTMState(object):
    def __init__(self, builder, h_t=None, c_t=None, state_idx=-1, prev_state=None):
      self.builder = builder
      self.state_idx=state_idx
      self.prev_state = prev_state
      self.h_t = h_t
      self.c_t = c_t

    def add_input(self, x_t):
      h_t, c_t = self.builder.add_input(x_t, self.prev_state)
      return LSTMState(self.builder, h_t, c_t, self.state_idx+1, prev_state=self)
      
    def transduce(self, xs):
        return self.builder.transduce(xs)

    def output(self): return self.h_t

    def prev(self): return self.prev_state
    def b(self): return self.builder
    def get_state_idx(self): return self.state_idx


class CustomCompactLSTMBuilder(object):
  def __init__(self, layers, input_dim, hidden_dim, model):
    if layers!=1: raise RuntimeError("CustomCompactLSTMBuilder supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
  
    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))
    
    self.dropout_rate = 0.0
    self.weightnoise_std = 0.0

  def whoami(self): return "CustomCompactLSTMBuilder"
  
  def set_dropout(self, p):
    if not 0.0 <= p <= 1.0:
      raise Exception("dropout rate must be a probability (>=0 and <=1)")
    self.dropout_rate = p
  def disable_dropout(self):
    self.dropout_rate = 0.0
    
  def set_weightnoise(self, std):
    if not 0.0 <= std:
      raise Exception("weight noise must have standard deviation >=0")
    self.weightnoise_std = std
  def disable_weightnoise(self):
    self.weightnoise_std = 0.0
    
  def initial_state(self, vecs=None):
    self.Wx = dy.parameter(self.p_Wx)
    self.Wh = dy.parameter(self.p_Wh)
    self.b = dy.parameter(self.p_b)
    self.set_dropout_masks() # TODO: check the paper if we should do different mask for each batch?
    if vecs is not None:
      assert len(vecs)==2
      return LSTMState(self, h_t=vecs[0], c_t=vecs[1])
    else:
      return LSTMState(self)
  def set_dropout_masks(self, batch_size=1):
    if self.dropout_rate > 0.0:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)
      self.dropout_mask_h = dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size)
    
  def add_input(self, x_t, prev_state):
    if prev_state is None or prev_state.h_t is None:
      h_tm1 = dy.zeroes(dim=(self.hidden_dim,), batch_size=x_t.dim()[1])
    else:
      h_tm1 = prev_state.h_t
    if prev_state is None or prev_state.c_t is None:
      c_tm1 = dy.zeroes(dim=(self.hidden_dim,), batch_size=x_t.dim()[1])
    else:
      c_tm1 = prev_state.c_t
    if self.dropout_rate > 0.0:
      # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
      gates_t = dy.vanilla_lstm_gates_dropout(x_t, h_tm1, self.Wx, self.Wh, self.b, self.dropout_mask_x, self.dropout_mask_h, self.weightnoise_std)
    else:
      gates_t = dy.vanilla_lstm_gates(x_t, h_tm1, self.Wx, self.Wh, self.b, self.weightnoise_std)
    try:
      c_t = dy.vanilla_lstm_c(c_tm1, gates_t)
    except ValueError:
      c_t = dy.vanilla_lstm_c(c_tm1, gates_t)
    h_t = dy.vanilla_lstm_h(c_t, gates_t)
    return h_t, c_t
    
  def transduce(self, xs):
    xs = list(xs)
    h = [dy.zeroes(dim=(self.hidden_dim,), batch_size=xs[0].dim()[1])]
    c = [dy.zeroes(dim=(self.hidden_dim,), batch_size=xs[0].dim()[1])]
    for x_t in xs:
      if self.dropout_rate > 0.0:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        gates_t = dy.vanilla_lstm_gates_dropout([x_t], h[-1], self.Wx, self.Wh, self.b, self.dropout_mask_x, self.dropout_mask_h, self.weightnoise_std)
      else:
        gates_t = dy.vanilla_lstm_gates([x_t], h[-1], self.Wx, self.Wh, self.b, self.weightnoise_std)
      c_t = dy.vanilla_lstm_c(c[-1], gates_t)
      c.append(c_t)
      h.append(dy.vanilla_lstm_h(c_t, gates_t))
    return h
  


class CustomLSTMBuilder(object):
  """
  This is a Python version of the vanilla LSTM.
  In contrast to the C++ version, this one does currently not support multiple layers or dropout.
  """
  def __init__(self, layers, input_dim, hidden_dim, model):
    if layers!=1: raise RuntimeError("CustomLSTMBuilder supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
  
    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))
    
  def whoami(self): return "CustomLSTMBuilder"
  
  def set_dropout(self, p):
    if p>0.0: raise RuntimeError("CustomLSTMBuilder does not support dropout")
  def disable_dropout(self):
    pass
  def transduce(self, xs):
    Wx = dy.parameter(self.p_Wx)
    Wh = dy.parameter(self.p_Wh)
    b = dy.parameter(self.p_b)
    h = []
    c = []
    for i, x_t in enumerate(xs):
      if i==0:
        tmp = dy.affine_transform([b, Wx, x_t])
      else:
        tmp = dy.affine_transform([b, Wx, x_t, Wh, h[-1]])
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
  def __init__(self, input_dim, model, chn_dim=3, num_filters=32, residual=True):
    if input_dim%chn_dim!=0: raise RuntimeError("input_dim must be divisible by chn_dim")
    self.input_dim = input_dim

    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 1
    self.filter_size_freq = 3
    self.residual = residual
    if residual and chn_dim!=num_filters: raise RuntimeError("Residual connections required chn_dim==num_filters, but found %s != %s" % (chn_dim, num_filters))
    normalInit=dy.NormalInitializer(0, 0.1)

    self.params = {}
    for direction in ["fwd","bwd"]:
      self.params["x2all_" + direction] = \
          model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, 
                                    self.chn_dim, self.num_filters * 4),
                               init=normalInit)
      self.params["h2all_" + direction] = \
          model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, 
                                    self.num_filters, self.num_filters * 4),
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
      es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)

    h_out = {}
    for direction in ["fwd", "bwd"]:
      # input convolutions
      gates_xt_bias = dy.conv2d_bias(es_chn, dy.parameter(self.params["x2all_" + direction]), dy.parameter(self.params["b_" + direction]), stride=(1,1), is_valid=False)
      gates_xt_bias_list = [dy.pick_range(gates_xt_bias, i, i+1) for i in range(gates_xt_bias.dim()[0][0])]

      h = []
      c = []
      for input_pos in range(len(gates_xt_bias_list)):
        directional_pos = input_pos if direction=="fwd" else len(gates_xt_bias_list)-input_pos-1
        gates_t = gates_xt_bias_list[directional_pos]
        if input_pos>0:
          # recurrent convolutions
          gates_h_t = dy.conv2d(h[-1], dy.parameter(self.params["h2all_" + direction]), stride=(1,1), is_valid=False)
          gates_t += gates_h_t
        
        # standard LSTM logic
        if len(c)==0:
          c_tm1 = dy.zeros((self.freq_dim * self.num_filters,), batch_size=batch_size)
        else:
          c_tm1 = c[-1]
        # TODO: to save memory, could extend vanilla_lstm_c, vanilla_lstm_h to allow arbitrary tensors instead of just vectors; then we can avoid the reshapes below
        gates_t_reshaped = dy.reshape(gates_t, (4 * self.freq_dim * self.num_filters,), batch_size=batch_size)
        c_t = dy.reshape(dy.vanilla_lstm_c(c_tm1, gates_t_reshaped), (self.freq_dim * self.num_filters,), batch_size=batch_size) 
        c.append(c_t)
        h_t = dy.vanilla_lstm_h(c_t, gates_t_reshaped)
        h.append(dy.reshape(h_t, (1, self.freq_dim, self.num_filters, ), batch_size=batch_size))
      h_out[direction] = h
    ret_expr = []
    for state_i in range(len(h_out["fwd"])):
      state_fwd = h_out["fwd"][state_i]
      state_bwd = h_out["bwd"][-1-state_i]
      output_dim = (state_fwd.dim()[0][1] * state_fwd.dim()[0][2],)
      if self.residual:
        fwd_reshape = dy.reshape(state_fwd + dy.pick_range(es_chn, state_i, state_i+1), output_dim, batch_size=batch_size)
        bwd_reshape = dy.reshape(state_bwd + dy.pick_range(es_chn, state_i, state_i+1), output_dim, batch_size=batch_size)
      else:
        fwd_reshape = dy.reshape(state_fwd, output_dim, batch_size=batch_size)
        bwd_reshape = dy.reshape(state_bwd, output_dim, batch_size=batch_size)
      ret_expr.append(dy.concatenate([fwd_reshape, bwd_reshape]))
    return ret_expr
  

class NetworkInNetworkBiRNNBuilder(object):
  """
  Builder for NiN-interleaved RNNs that delegates to regular RNNs and wires them together.
  See http://iamaaditya.github.io/2016/03/one-by-one-convolution/
  and https://arxiv.org/pdf/1610.03022.pdf
  """
  def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory,
               batch_norm=False, stride=1, num_projections=1, projection_enabled=True,
               nonlinearity="relu"):
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
    assert num_projections > 0
    self.builder_layers = []
    self.hidden_dim = hidden_dim
    self.stride=stride
    self.num_projections = num_projections
    self.projection_enabled = projection_enabled
    self.nonlinearity = nonlinearity
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
  def set_weight_noise(self, p):
    for (fb, bb, bn) in self.builder_layers:
      fb.set_weight_noise(p)
      bb.set_weight_noise(p)
  def disable_weight_noise(self):
    for (fb, bb, bn) in self.builder_layers:
      fb.disable_weight_noise()
      bb.disable_weight_noise()

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
      if self.projection_enabled:
        proj = lintransf_param * concat
      else: proj = concat
      projections.append(proj)
    if self.use_bn:
      bn_layer = bn.bn_expr(dy.concatenate([dy.reshape(x, (1,self.hidden_dim), batch_size=batch_size) for x in projections], 
                                0), 
                 train=self.train)
      nonlin = self.apply_nonlinearity(bn_layer)
      es = [dy.pick(nonlin, i) for i in range(nonlin.dim()[0][0])]
    else:
      es = []
      for proj in projections:
        nonlin = self.apply_nonlinearity(proj)
        es.append(nonlin)
    return es
  def apply_nonlinearity(self, expr):
    if self.nonlinearity is None:
      return expr
    elif self.nonlinearity.lower()=="relu":
      return dy.rectify(expr)
    else:
      raise RuntimeError("unknown nonlinearity %s" % self.nonlinearity)
  def initial_state(self):
    return PseudoState(self)
