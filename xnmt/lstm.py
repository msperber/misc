import dynet as dy

def builder_for_spec(spec):
  if spec=="vanilla":
    return dy.VanillaLSTMBuilder
  elif spec=="compact":
    return dy.CompactVanillaLSTMBuilder
  

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


class LowMemLSTMBuilder(object):
  """
  This is a test of the new dynet LSTM node collection.
  Currently, it does not support multiple layers or dropout.
  """
  def __init__(self, layers, input_dim, hidden_dim, model):
    if layers!=1: raise RuntimeError("LowMemLSTMBuilder supports only exactly one layer")
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
  
    # [i; f; o; g]
    self.p_Wx = model.add_parameters(dim=(hidden_dim*4, input_dim))
    self.p_Wh = model.add_parameters(dim=(hidden_dim*4, hidden_dim))
    self.p_b  = model.add_parameters(dim=(hidden_dim*4,), init=dy.ConstInitializer(0.0))
    
  def whoami(self): return "LowMemLSTMBuilder"
  
  def set_dropout(self, p):
    if p>0.0: raise RuntimeError("LowMemLSTMBuilder does not support dropout")
  def disable_dropout(self):
    pass
  def initial_state(self, vecs=None):
    self.Wx = dy.parameter(self.p_Wx)
    self.Wh = dy.parameter(self.p_Wh)
    self.b = dy.parameter(self.p_b)
    if vecs is not None:
      assert len(vecs)==2
      return LSTMState(self, h_t=vecs[0], c_t=vecs[1])
    else:
      return LSTMState(self)
  def add_input(self, x_t, prev_state):
    if prev_state is None or prev_state.h_t is None:
      h_tm1 = dy.zeroes(dim=(self.hidden_dim,), batch_size=x_t.dim()[1])
    else:
      h_tm1 = prev_state.h_t
    if prev_state is None or prev_state.c_t is None:
      c_tm1 = dy.zeroes(dim=(self.hidden_dim,), batch_size=x_t.dim()[1])
    else:
      c_tm1 = prev_state.c_t
    gates_t = dy.vanilla_lstm_gates(x_t, h_tm1, self.Wx, self.Wh, self.b)
    try:
      c_t = dy.vanilla_lstm_c(c_tm1, gates_t)
    except ValueError:
      c_t = dy.vanilla_lstm_c(c_tm1, gates_t)
    h_t = dy.vanilla_lstm_h(c_t, gates_t)
    return h_t, c_t
    
  def transduce(self, xs):
    xs = list(xs)
    Wx = dy.parameter(self.p_Wx)
    Wh = dy.parameter(self.p_Wh)
    b = dy.parameter(self.p_b)
    h = [dy.zeroes(dim=(self.hidden_dim,), batch_size=xs[0].dim()[1])]
    c = [dy.zeroes(dim=(self.hidden_dim,), batch_size=xs[0].dim()[1])]
    for i, x_t in enumerate(xs):
      gates_t = dy.vanilla_lstm_gates(x_t, h[-1], Wx, Wh, b)
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
