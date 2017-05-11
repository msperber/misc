import dynet as dy
import embedder

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
  def transduce(self, es):
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
  
