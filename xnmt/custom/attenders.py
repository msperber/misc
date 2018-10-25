import numbers

import dynet as dy

import expression_seqs
import param_collections
import param_initializers
from modelparts import attenders
from persistence import Serializable, serializable_init, Ref, bare


class MlpLocationAttender(attenders.Attender, Serializable):
  """
  Implements the attention model of Chorowski et. al (2015): Attention-Based Models for Speech Recognition.

  This adds a convolutional filter over the previous timestep's attention scores.

  Note: currently untested.

  Args:
    input_dim: input dimension
    state_dim: dimension of state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """

  yaml_tag = '!MlpLocationAttender'


  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               state_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init: param_initializers.ParamInitializer = Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)))\
          -> None:
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = param_collections.ParamManager.my_params(self)
    self.pW = param_collection.add_parameters((hidden_dim, input_dim), init=param_init.initializer((hidden_dim, input_dim)))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim), init=param_init.initializer((hidden_dim, state_dim)))
    self.pb = param_collection.add_parameters((hidden_dim,), init=bias_init.initializer((hidden_dim,)))
    self.pU = param_collection.add_parameters((1, hidden_dim), init=param_init.initializer((1, hidden_dim)))
    self.pL = param_collection.add_parameters((100, 1, 1, hidden_dim), init=param_init.initializer((100, 1, 1, hidden_dim)))
    self.curr_sent = None

  def init_sent(self, sent: expression_seqs.ExpressionSequence):
    self.attention_vecs = []
    self.curr_sent = sent
    I = self.curr_sent.as_tensor()
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)
    self.WI = dy.affine_transform([b, W, I])
    wi_dim = self.WI.dim()
    # if the input size is "1" then the last dimension will be dropped.
    if len(wi_dim[0]) == 1:
      self.WI = dy.reshape(self.WI, (wi_dim[0][0], 1), batch_size=wi_dim[1])

  def calc_attention(self, state):
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    WI = self.WI
    curr_sent_mask = self.curr_sent.mask
    if self.attention_vecs:
      conv_feats = dy.conv2d(self.attention_vecs[-1],
                             self.pL,
                             stride=[1, 1],
                             is_valid=False)
      conv_feats = dy.transpose(dy.reshape(conv_feats,
                                           (conv_feats.dim()[0][0],self.hidden_dim),
                                           batch_size=conv_feats.dim()[1]))
      h = dy.tanh(dy.colwise_add(WI + conv_feats, V * state))
    else:
      h = dy.tanh(dy.colwise_add(WI, V * state))
    scores = dy.transpose(U * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    return I * attention