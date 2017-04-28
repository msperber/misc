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

    self.p_x2i = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_x2f = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_x2o = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_x2g = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_h2i = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_h2f = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_h2o = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_h2g = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.p_bi  = model.add_parameters(dim=(self.num_filters,), init=normalInit)
    self.p_bf  = model.add_parameters(dim=(self.num_filters,), init=dy.ConstInitializer(0.0))
    self.p_bo  = model.add_parameters(dim=(self.num_filters,), init=normalInit)
    self.p_bg  = model.add_parameters(dim=(self.num_filters,), init=normalInit)


    
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

    x_filtered_i = dy.conv2d_bias(es_chn, dy.parameter(self.p_x2i), dy.parameter(self.p_bi), stride=(1,1), is_valid=False)
    x_filtered_f = dy.conv2d_bias(es_chn, dy.parameter(self.p_x2f), dy.parameter(self.p_bf), stride=(1,1), is_valid=False)
    x_filtered_o = dy.conv2d_bias(es_chn, dy.parameter(self.p_x2o), dy.parameter(self.p_bo), stride=(1,1), is_valid=False)
    x_filtered_g = dy.conv2d_bias(es_chn, dy.parameter(self.p_x2g), dy.parameter(self.p_bg), stride=(1,1), is_valid=False)
    xs_i = [dy.pick(x_filtered_i, i) for i in range(x_filtered_i.dim()[0][0])]
    xs_f = [dy.pick(x_filtered_f, i) for i in range(x_filtered_f.dim()[0][0])]
    xs_o = [dy.pick(x_filtered_o, i) for i in range(x_filtered_o.dim()[0][0])]
    xs_g = [dy.pick(x_filtered_g, i) for i in range(x_filtered_g.dim()[0][0])]
    h = []
    c = []
    for i in range(len(xs_i)):
      i_ait = xs_i[i]
      i_aft = xs_f[i]
      i_aot = xs_o[i]
      i_agt = xs_g[i]
      if i>0:
        wh_i = dy.conv2d(dy.reshape(h[-1], (1, h[-1].dim()[0][0], h[-1].dim()[0][1]), batch_size=batch_size), dy.parameter(self.p_h2i), stride=(1,1), is_valid=False)
        wh_f = dy.conv2d(dy.reshape(h[-1], (1, h[-1].dim()[0][0], h[-1].dim()[0][1]), batch_size=batch_size), dy.parameter(self.p_h2f), stride=(1,1), is_valid=False)
        wh_o = dy.conv2d(dy.reshape(h[-1], (1, h[-1].dim()[0][0], h[-1].dim()[0][1]), batch_size=batch_size), dy.parameter(self.p_h2o), stride=(1,1), is_valid=False)
        wh_g = dy.conv2d(dy.reshape(h[-1], (1, h[-1].dim()[0][0], h[-1].dim()[0][1]), batch_size=batch_size), dy.parameter(self.p_h2g), stride=(1,1), is_valid=False)
        i_ait += dy.reshape(wh_i, (wh_i.dim()[0][1], wh_i.dim()[0][2]), batch_size=batch_size)
        i_aft += dy.reshape(wh_f, (wh_i.dim()[0][1], wh_f.dim()[0][2]), batch_size=batch_size)
        i_aot += dy.reshape(wh_o, (wh_i.dim()[0][1], wh_o.dim()[0][2]), batch_size=batch_size)
        i_agt += dy.reshape(wh_g, (wh_i.dim()[0][1], wh_g.dim()[0][2]), batch_size=batch_size)

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
  
class PreInitBiRNNBuilder(object):
    """
    Builder for BiRNNs that delegates to regular RNNs and wires them together.  
    
        builder = BiRNNBuilder(1, 128, 100, model, LSTMBuilder)
        [o1,o2,o3] = builder.transduce([i1,i2,i3])
    """
    def __init__(self, model, rnn_builders_fwd, rnn_builders_bwd):
        """
        :param model
        :param rnn_builders_fwd: already initialized builders
        :param rnn_builders_bwd: already initialized builders
        """
        assert len(rnn_builders_fwd) == len(rnn_builders_bwd)
        self.builder_layers = []
        for f,b in zip(rnn_builders_fwd, rnn_builders_bwd):
          self.builder_layers.append((f,b))

    def whoami(self): return "PreInitBiRNNBuilder"

    def set_dropout(self, p):
      for (fb,bb) in self.builder_layers:
        fb.set_dropout(p)
        bb.set_dropout(p)
    def disable_dropout(self):
      for (fb,bb) in self.builder_layers:
        fb.disable_dropout()
        bb.disable_dropout()

    def add_inputs(self, es):
        """
        returns the list of state pairs (stateF, stateB) obtained by adding 
        inputs to both forward (stateF) and backward (stateB) RNNs.  

        @param es: a list of Expression

        see also transduce(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
             transduce is much more memory efficient than add_inputs. 
        """
        for (fb,bb) in self.builder_layers[:-1]:
            fs = fb.initial_state().transduce(es)
            bs = bb.initial_state().transduce(reversed(es))
            es = [dy.concatenate([f,b]) for f,b in zip(fs, reversed(bs))]
        (fb,bb) = self.builder_layers[-1]
        fs = fb.initial_state().add_inputs(es)
        bs = bb.initial_state().add_inputs(reversed(es))
        return [(f,b) for f,b in zip(fs, reversed(bs))]

    def transduce(self, es):
        """
        returns the list of output Expressions obtained by adding the given inputs
        to the current state, one by one, to both the forward and backward RNNs, 
        and concatenating.
        
        @param es: a list of Expression

        see also add_inputs(xs)

        .transduce(xs) is different from .add_inputs(xs) in the following way:

            .add_inputs(xs) returns a list of RNNState pairs. RNNState objects can be
             queried in various ways. In particular, they allow access to the previous
             state, as well as to the state-vectors (h() and s() )

            .transduce(xs) returns a list of Expression. These are just the output
             expressions. For many cases, this suffices. 
             transduce is much more memory efficient than add_inputs. 
        """
        for (fb,bb) in self.builder_layers:
            fs = fb.initial_state().transduce(es)
            bs = bb.initial_state().transduce(embedder.ExpressionSequence(expr_list=reversed(es)))
            es = [dy.concatenate([f,b]) for f,b in zip(fs, reversed(bs))]
        return es
