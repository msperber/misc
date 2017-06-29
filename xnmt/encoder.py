import dynet as dy
from batcher import *
import residual
import pyramidal
import conv_encoder
from embedder import ExpressionSequence
import lstm
from translator import TrainTestInterface
import inspect

class Encoder(TrainTestInterface):
  """
  A parent class representing all classes that encode inputs.
  """
  def __init__(self, model, global_train_params, input_dim):
    """
    Every encoder constructor needs to accept at least these 3 parameters 
    """
    raise NotImplementedError('__init__ must be implemented in Encoder subclasses')

  def transduce(self, sent):
    """Encode inputs into outputs.

    :param sent: The input to be encoded. This is duck-typed, so it is the appropriate input for this particular type of encoder. Frequently it will be a list of word embeddings, but it can be anything else.
    :returns: The encoded output. Frequently this will be a list of expressions representing the encoded vectors for each word.
    """
    raise NotImplementedError('transduce must be implemented in Encoder subclasses')
  
  @staticmethod
  def from_spec(encoder_spec, global_train_params, model):
    """Create an encoder from a specification.

    :param encoder_spec: Encoder-specific settings (encoders must consume all provided settings)
    :param global_train_params: dictionary with global params such as dropout and default_layer_dim, which the encoders are free to make use of.
    :param model: The model that we should add the parameters to
    """
    encoder_spec = dict(encoder_spec)
    encoder_type = encoder_spec.pop("type")
    encoder_spec["model"] = model
    encoder_spec["global_train_params"] = global_train_params
    known_encoders = [key for (key,val) in globals().items() if inspect.isclass(val) and issubclass(val, Encoder) and key not in ["BuilderEncoder","Encoder"]]
    if encoder_type not in known_encoders and encoder_type+"Encoder" not in known_encoders:
      raise RuntimeError("specified encoder %s is unknown, choices are: %s" 
                         % (encoder_type,", ".join([key for (key,val) in globals().items() if inspect.isclass(val) and issubclass(val, Encoder)])))
    encoder_class = globals().get(encoder_type, globals().get(encoder_type+"Encoder"))
    return encoder_class(**encoder_spec)

class BuilderEncoder(Encoder):
  def transduce(self, sent):
    return self.builder.transduce(sent)

class BiLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim=512, layers=1, hidden_dim=None, dropout=None, weight_noise=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    if weight_noise is None: weight_noise = global_train_params.get("weight_noise", 0.0)
    self.weight_noise = weight_noise
    self.builder = dy.BiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)
    self.builder.set_weight_noise(self.weight_noise if val else 0.0)

class ResidualLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    self.builder = residual.ResidualRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, residual_to_output, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class ResidualBiLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim=512, layers=1, hidden_dim=None, residual_to_output=False, dropout=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    self.builder = residual.ResidualBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, residual_to_output, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)

class PyramidalLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim=512, layers=1, hidden_dim=None, downsampling_method="skip", dropout=None, weight_noise=None, reduce_factor=2):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    if weight_noise is None: weight_noise = global_train_params.get("weight_noise", 0.0)
    self.weight_noise = weight_noise
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, downsampling_method, reduce_factor)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, downsampling_method, reduce_factor, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)
    self.builder.set_weight_noise(self.weight_noise if val else 0.0)

class ConvLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim, chn_dim=32, num_filters=32, residual=True):
    self.builder = lstm.ConvLSTMBuilder(input_dim, model, chn_dim, num_filters, residual)
    self.serialize_params = [model, global_train_params, input_dim, chn_dim, num_filters, residual]
    
class StridedConvEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim, layers=1, chn_dim=3, num_filters=32, 
               output_tensor=False, batch_norm=True, stride=(2,2), nonlinearity="relu", init_gauss_var=0.1):
    self.builder = conv_encoder.StridedConvEncBuilder(layers, input_dim, model, chn_dim, 
                                            num_filters, output_tensor, batch_norm,
                                            stride, nonlinearity, init_gauss_var)
    self.serialize_params = [model, global_train_params, input_dim, layers, chn_dim, num_filters, output_tensor, batch_norm, stride, nonlinearity, init_gauss_var]
  def set_train(self, val):
    self.builder.train = val

class PoolingConvEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim, pooling=[None, (1,1)], chn_dim=3, num_filters=32, 
               output_tensor=False, nonlinearity="relu", init_gauss_var=0.1):
    self.builder = conv_encoder.PoolingConvEncBuilder(input_dim, model, pooling, chn_dim, num_filters, output_tensor, nonlinearity, init_gauss_var)
    self.serialize_params = [model, global_train_params, input_dim, pooling, chn_dim, num_filters, output_tensor, nonlinearity, init_gauss_var]

class NetworkInNetworkBiLSTMEncoder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim, layers=1, hidden_dim=None, batch_norm=True, stride=1, num_projections=1, projection_enabled=True, nonlinearity="relu", dropout=None, weight_noise=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    if weight_noise is None: weight_noise = global_train_params.get("weight_noise", 0.0)
    self.weight_noise = weight_noise
    self.builder = lstm.NetworkInNetworkBiRNNBuilder(layers, input_dim, hidden_dim, model, 
                                            dy.VanillaLSTMBuilder, batch_norm, stride,
                                            num_projections, projection_enabled,
                                            nonlinearity, dropout)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, batch_norm, stride, num_projections, projection_enabled, nonlinearity, dropout]
  def set_train(self, val):
    self.builder.set_dropout(self.dropout if val else 0.0)
    self.builder.set_weight_noise(self.weight_noise if val else 0.0)

class ConvBiRNNBuilder(BuilderEncoder):
  def __init__(self, model, global_train_params, input_dim, layers, hidden_dim=None, chn_dim=3, num_filters=32, filter_size_time=3, filter_size_freq=3, stride=(2,2), dropout=None):
    if hidden_dim is None: hidden_dim = global_train_params.get("default_layer_dim", 512)
    if dropout is None: dropout = global_train_params.get("dropout", 0.0)
    self.dropout = dropout
    self.builder = conv_encoder.ConvBiRNNBuilder(layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder,
                                            chn_dim, num_filters, filter_size_time, filter_size_freq, stride)
    self.serialize_params = [model, global_train_params, input_dim, layers, hidden_dim, chn_dim, num_filters, filter_size_time, filter_size_freq, stride, dropout]
  def set_train(self, val):
    self.builder.train = val
    self.builder.set_dropout(self.dropout if val else 0.0)
  
class ModularEncoder(Encoder):
  def __init__(self, model, global_train_params, input_dim, modules):
    self.modules = []
    if input_dim != modules[0].get("input_dim"):
      raise RuntimeError("Mismatching input dimensions of first module: %s != %s".format(input_dim, modules[0].get("input_dim")))
    for module_spec in modules:
      self.modules.append(Encoder.from_spec(module_spec, global_train_params, model))
    self.serialize_params = [model, global_train_params, input_dim, modules]

  def transduce(self, sent, train=False):
    for i, module in enumerate(self.modules):
      sent = module.transduce(sent)
      if i<len(self.modules)-1:
        if type(sent)==dy.Expression:
          sent = ExpressionSequence(expr_tensor=sent)
        else:
          sent = ExpressionSequence(expr_list=sent)
    return sent

  def get_train_test_components(self):
    return self.modules
