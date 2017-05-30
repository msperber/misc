import math
import dynet as dy
from residual import PseudoState
from embedder import ExpressionSequence
from batch_norm import ConvolutionalBatchNorm

class ConvBiRNNBuilder(object):
  """
  Inputs are first put through 2 CNN layers, each with stride (2,2), so dimensionality
  is reduced by 4 in both directions.
  Then, we add a configurable number of bidirectional RNN layers on top.
  """
  
  def __init__(self, num_layers, input_dim, hidden_dim, model, rnn_builder_factory, chn_dim=3):
    """
    :param num_layers: depth of the RNN
    :param input_dim: size of the inputs
    :param hidden_dim: size of the outputs (and intermediate RNN layer representations)
    :param model
    :param rnn_builder_factory: RNNBuilder subclass, e.g. LSTMBuilder
    """
    assert num_layers > 0
    assert hidden_dim % 2 == 0
    assert input_dim % chn_dim == 0
      
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = 32
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.stride = (2,2)
    
    normalInit=dy.NormalInitializer(0, 0.1)
    self.filters1 = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.chn_dim, self.num_filters),
                                         init=normalInit)
    self.filters2 = model.add_parameters(dim=(self.filter_size_time, self.filter_size_freq, self.num_filters, self.num_filters),
                                         init=normalInit)
    conv_dim_l1 = math.ceil(float(self.freq_dim - self.filter_size_freq + 1) / float(self.stride[1]))
    conv_dim_l2 = int(math.ceil(float(conv_dim_l1 - self.filter_size_freq + 1) / float(self.stride[1])))
    conv_dim_out = conv_dim_l2 * self.num_filters

    self.builder_layers = []
    f = rnn_builder_factory(1, conv_dim_out, hidden_dim / 2, model)
    b = rnn_builder_factory(1, conv_dim_out, hidden_dim / 2, model)
    self.builder_layers.append((f, b))
    for _ in xrange(num_layers - 1):
      f = rnn_builder_factory(1, hidden_dim, hidden_dim / 2, model)
      b = rnn_builder_factory(1, hidden_dim, hidden_dim / 2, model)
      self.builder_layers.append((f, b))

  def whoami(self): return "ConvBiRNNBuilder"

  def set_dropout(self, p):
    for (fb, bb) in self.builder_layers:
      fb.set_dropout(p)
      bb.set_dropout(p)
  def disable_dropout(self):
    for (fb, bb) in self.builder_layers:
      fb.disable_dropout()
      bb.disable_dropout()

  def transduce(self, es):
    es_expr = es.as_tensor()

    # e.g. es_expr.dim() ==((276, 240), 1)
    sent_len = es_expr.dim()[0][0]
    batch_size=es_expr.dim()[1]
    
    # convolutions won't work if sent length is too short; pad if necessary
    pad_size = 0
    while math.ceil(float(sent_len + pad_size - self.filter_size_time + 1) / float(self.stride[0])) < self.filter_size_time:
      pad_size += 1
    if pad_size>0:
      es_expr = dy.concatenate([es_expr, dy.zeroes((pad_size, self.freq_dim * self.chn_dim), batch_size=es_expr.dim()[1])])
      sent_len += pad_size

    # convolution layers    
    es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size) # ((276, 80, 3), 1)
    cnn_layer1 = dy.conv2d(es_chn, dy.parameter(self.filters1), stride=self.stride, is_valid=True) # ((137, 39, 32), 1)
    cnn_layer2 = dy.conv2d(cnn_layer1, dy.parameter(self.filters2), stride=self.stride, is_valid=True) # ((68, 19, 32), 1)
    cnn_out = dy.reshape(cnn_layer2, (cnn_layer2.dim()[0][0], cnn_layer2.dim()[0][1]*cnn_layer2.dim()[0][2]), batch_size=batch_size) # ((68, 608), 1)
    es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]
    
    # RNN layers
    for (fb, bb) in self.builder_layers:
      fs = fb.initial_state().transduce(es_list)
      bs = bb.initial_state().transduce(reversed(es_list))
      es_list = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
    return es_list

  def initial_state(self):
    return PseudoState(self)


class StridedConvEncBuilder(object):
  """
  Implements several CNN layers, each with stride (2,2), resulting in downsampling by 2**(num_layers-1) in both dimensions.
  """
  
  def __init__(self, num_layers, input_dim, model, chn_dim=3, num_filters=32, output_tensor=False):
    """
    :param num_layers: encoder depth
    :param input_dim: size of the inputs, before factoring out the channels.
                      We will end up with a convolutional layer of size num_steps X input_dim/chn_dim X chn_dim 
    :param model
    :param chn_dim: channel dimension
    :param num_filters
    :param output_tensor: if set, the output is directly given as a 3d-tensor, rather than converted to a list of vector expressions
    """
    assert num_layers > 0
    assert input_dim % chn_dim == 0
    
    self.num_layers = num_layers
    self.chn_dim = chn_dim
    self.freq_dim = input_dim / chn_dim
    self.num_filters = num_filters
    self.filter_size_time = 3
    self.filter_size_freq = 3
    self.stride = (2,2)
    self.output_tensor = output_tensor
    
    self.use_bn = True
    self.train = True
    
    normalInit=dy.NormalInitializer(0, 0.1)
    self.filters_layers = []
    self.bn_layers = []
    self.filters_layers = []
    for layer_i in range(num_layers):
      filters = model.add_parameters(dim=(self.filter_size_time,
                                          self.filter_size_freq,
                                          self.chn_dim if layer_i==0 else self.num_filters,
                                          self.num_filters),
                                     init=normalInit)
      if self.use_bn:
        self.bn_layers.append(ConvolutionalBatchNorm(model, self.num_filters))
      self.filters_layers.append(filters)
  
  def get_output_dim(self):
    conv_dim = self.freq_dim
    for _ in range(self.num_layers):
      conv_dim = int(math.ceil(float(conv_dim - self.filter_size_freq + 1) / float(self.stride[1])))
    return conv_dim * self.num_filters
  
  def get_output_len(self, input_len):
    conv_dim = input_len
    for _ in range(self.num_layers):
      conv_dim = int(math.ceil(float(conv_dim - self.filter_size_time + 1) / float(self.stride[0])))
    return conv_dim

  def whoami(self): return "StridedConvEncBuilder"

  def set_dropout(self, p):
    if p>0.0: raise NotImplementedError("StridedConvEncBuilder does not support dropout")
  def disable_dropout(self):
    pass

  def transduce(self, es):
    es_expr = es.as_tensor()

    sent_len = es_expr.dim()[0][0]
    batch_size=es_expr.dim()[1]
    
    # convolutions won't work if sentence length is too short; pad if necessary
    pad_size = 0
    while self.get_output_len(sent_len + pad_size) < self.filter_size_time:
      pad_size += 1
    if pad_size>0:
      es_expr = dy.concatenate([es_expr, dy.zeroes((pad_size, self.freq_dim * self.chn_dim), batch_size=es_expr.dim()[1])])
      sent_len += pad_size

    # loop over layers    
    es_chn = dy.reshape(es_expr, (sent_len, self.freq_dim, self.chn_dim), batch_size=batch_size)
    cnn_layer = es_chn
    for layer_i in range(len(self.filters_layers)):
      filters = self.filters_layers[layer_i]
      cnn_layer = dy.conv2d(cnn_layer, dy.parameter(filters), stride=self.stride, is_valid=True)
      if self.use_bn:
        cnn_layer = self.bn_layers[layer_i].bn_expr(cnn_layer, train=self.train)
      cnn_layer = dy.rectify(cnn_layer) # TODO: might do maxout (see https://arxiv.org/abs/1701.02720 )
    if self.output_tensor:
      return cnn_layer
    else:
      cnn_out = dy.reshape(cnn_layer, (cnn_layer.dim()[0][0], cnn_layer.dim()[0][1]*cnn_layer.dim()[0][2]), batch_size=batch_size)
      es_list = [cnn_out[i] for i in range(cnn_out.dim()[0][0])]
      return es_list
