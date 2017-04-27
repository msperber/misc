import dynet as dy

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
      i_ait = dy.pickrange(tmp, 0, self.hidden_dim)
      i_aft = dy.pickrange(tmp, self.hidden_dim, self.hidden_dim*2)
      i_aot = dy.pickrange(tmp, self.hidden_dim*2, self.hidden_dim*3)
      i_agt = dy.pickrange(tmp, self.hidden_dim*3, self.hidden_dim*4)
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
  This is a ConvLSTM implementation for a single layer & direction.
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

    # [i; f; o; g]
    self.p_x2i = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters * 4),
                                         init=normalInit)
    self.p_h2i = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters * 4),
                                         init=normalInit)
    self.p_bi  = model.add_parameters(dim=(self.num_filters*4,), init=dy.ConstInitializer(0.0))


    
  def whoami(self): return "ConvLSTMBuilder"
  
  def set_dropout(self, p):
    if p>0.0: raise RuntimeError("ConvLSTMBuilder does not support dropout")
  def disable_dropout(self):
    pass
  def initial_state(self):
    return self
  def add_inputs(self):
    pass
  def transduce(self, es):
    es_expr = es.as_tensor()
    sent_len = es_expr.dim()[0][0]
    batch_size=es_expr.dim()[1]
    
    es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size) # ((276, 80, 3), 1)

    i_x2i = dy.parameter(self.p_x2i)
    i_h2i = dy.parameter(self.p_h2i)
    i_bi = dy.parameter(self.p_bi)
    
    x_filtered = dy.conv2d_bias(es_chn, dy.parameter(self.p_x2i), dy.parameter(self.p_bi), stride=(1,1), is_valid=False)
    xs = [dy.pick(x_filtered, i) for i in range(x_filtered.dim()[0][0])]
    h = []
    c = []
    for i, x_t in enumerate(xs):
      tmp = xs[i]
      if i>0:
        wh = dy.conv2d(dy.reshape(h[-1], (1, h[-1].dim()[0][0], h[-1].dim()[0][1]), batch_size=batch_size), dy.parameter(self.p_h2i), stride=(1,1), is_valid=False)
        tmp += dy.reshape(wh, (wh.dim()[0][1], wh.dim()[0][2]), batch_size=batch_size)

      i_ait = dy.select_cols(tmp, range(self.num_filters))
      i_aft = dy.select_cols(tmp, range(self.num_filters,2*self.num_filters))
      i_aot = dy.select_cols(tmp, range(2*self.num_filters,3*self.num_filters))
      i_agt = dy.select_cols(tmp, range(3*self.num_filters,4*self.num_filters))
      i_it = dy.logistic(i_ait)
      i_ft = dy.logistic(i_aft + 1.0)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if i==0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        c.append(dy.cmult(i_ft, c[-1]) + dy.cmult(i_it, i_gt))
      h_t = dy.cmult(i_ot, dy.tanh(c[-1]))
      h.append(h_t)
    return [dy.reshape(state, (state.dim()[0][0] * state.dim()[0][1],), batch_size=batch_size) for state in h]
  