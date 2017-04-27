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
  def __init__(self, layers, input_dim, hidden_dim, model, chn_dim=3):
    if layers!=1: raise RuntimeError("ConfLSTMBuilder supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
  
    # [i; f; o; g]
    self.p_x2i = model.add_parameters(dim=(hidden_dim*4, input_dim)) # Ws
    self.p_h2i = model.add_parameters(dim=(hidden_dim*4, hidden_dim)) # Wfs
    self.p_bi  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0)) # Ufs

    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = 32
    self.filter_size_time = 1
    self.filter_size_freq = 3
    normalInit=dy.NormalInitializer(0, 0.1)
    self.filters_vertical = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.filters_horizontal = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    
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
#    cnn_layer1 = dy.conv2d(es_chn, dy.parameter(self.filters1), stride=self.stride, is_valid=True) # ((137, 39, 32), 1)
#    cnn_layer2 = dy.conv2d(cnn_layer1, dy.parameter(self.filters2), stride=self.stride, is_valid=True) # ((68, 19, 32), 1)
#    cnn_out = dy.reshape(cnn_layer2, (cnn_layer2.dim()[0][0], cnn_layer2.dim()[0][1]*cnn_layer2.dim()[0][2]), batch_size=batch_size) # ((68, 608), 1)
#    es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]

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