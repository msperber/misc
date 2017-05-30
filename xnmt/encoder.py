import dynet as dy
from batcher import *
import residual
import pyramidal
import conv_encoder
from embedder import ExpressionSequence
import lstm

class Encoder:
  """
  A parent class representing all classes that encode inputs.
  """

  def transduce(self, sent):
    """Encode inputs into outputs.

    :param sent: The input to be encoded. This is duck-typed, so it is the appropriate input for this particular type of encoder. Frequently it will be a list of word embeddings, but it can be anything else.
    :returns: The encoded output. Frequently this will be a list of expressions representing the encoded vectors for each word.
    """
    raise NotImplementedError('transduce must be implemented in Encoder subclasses')
  
  @staticmethod
  def from_spec(spec, layers, input_dim, output_dim, model, residual_to_output):
    """Create an encoder from a specification.

    :param spec: Options include bilstm, residuallstm, residualbylstm, pyramidalbilstm, convbilstm, modular.
    :param layers: Number of layers
    :param input_dim: Input dimension
    :param output_dim: Output dimension
    :param model: The model that we should add the parameters to
    :param residual_to_output: For residual encoders, whether to add a residual to the output layer.
    """
    spec_lower = spec.lower()
    if spec_lower == "bilstm":
      return BiLSTMEncoder(layers, input_dim, output_dim, model)
    elif spec_lower == "residuallstm":
      return ResidualLSTMEncoder(layers, input_dim, output_dim, model, residual_to_output)
    elif spec_lower == "residualbilstm":
      return ResidualBiLSTMEncoder(layers, input_dim, output_dim, model, residual_to_output)
    elif spec_lower == "pyramidalbilstm":
      return PyramidalLSTMEncoder(layers, input_dim, output_dim, model)
    elif spec_lower == "convbilstm":
      return ConvBiRNNBuilder(layers, input_dim, output_dim, model)
    elif spec_lower == "modular":
      stridedConv = StridedConvEncoder(layers, input_dim, model) 
      return ModularEncoder(model,
                             stridedConv,
                             ConvLSTMEncoder(layers=1, input_dim=stridedConv.builder.get_output_dim(), model=model, chn_dim=32),
                             NetworkInNetworkBiLSTMEncoder(layers, stridedConv.builder.get_output_dim()*2, output_dim, model),
#                             pyramidal.PyramidalRNNBuilder(layers, output_dim, output_dim, model, dy.VanillaLSTMBuilder),
#                             dy.BiRNNBuilder(layers, output_dim, output_dim, model, dy.VanillaLSTMBuilder)
                            
                            )
    else:
      raise RuntimeError("Unknown encoder type {}".format(spec_lower))

class BuilderEncoder(Encoder):
  def transduce(self, sent):
    return self.builder.transduce(sent)

class BiLSTMEncoder(BuilderEncoder):
  def __init__(self, layers, input_dim, output_dim, model):
    self.builder = dy.BiRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder)
    self.serialize_params = [layers, input_dim, output_dim, model]

class ResidualLSTMEncoder(BuilderEncoder):
  def __init__(self, layers, input_dim, output_dim, model, residual_to_output):
    self.builder = residual.ResidualRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    self.serialize_params = [layers, input_dim, output_dim, model, residual_to_output]

class ResidualBiLSTMEncoder(BuilderEncoder):
  def __init__(self, layers, input_dim, output_dim, model, residual_to_output):
    self.builder = residual.ResidualBiRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    self.serialize_params = [layers, input_dim, output_dim, model, residual_to_output]

class PyramidalLSTMEncoder(BuilderEncoder):
  def __init__(self, layers, input_dim, output_dim, model):
    self.builder = pyramidal.PyramidalRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder)
    self.serialize_params = [layers, input_dim, output_dim, model]

class ConvLSTMEncoder(BuilderEncoder):
  def __init__(self, layers, input_dim, model, chn_dim):
    self.builder = lstm.ConvLSTMBuilder(layers, input_dim, model, chn_dim=chn_dim)
    self.serialize_params = [layers, input_dim, model, chn_dim]
    
class StridedConvEncoder(BuilderEncoder):
  def __init__(self, layers, input_dim, model):
    self.builder = conv_encoder.StridedConvEncBuilder(layers, input_dim, model, output_tensor=True)
    self.serialize_params = [layers, input_dim, model]

class NetworkInNetworkBiLSTMEncoder(BuilderEncoder):
  def __init__(self, layers, input_dim, output_dim, model):
    self.builder = lstm.NetworkInNetworkBiRNNBuilder(layers, input_dim, output_dim, model, dy.VanillaLSTMBuilder)
    self.serialize_params = [layers, input_dim, output_dim, model]
  
class ModularEncoder(Encoder):
  def __init__(self, model, *module_list):
    self.module_list = module_list
    self.serialize_params = [model] + list(module_list)

  def transduce(self, sent, train=False):
    for i, module in enumerate(self.module_list):
      sent = module.transduce(sent)
      if i<len(self.module_list)-1:
        if type(sent)==dy.Expression:
          sent = ExpressionSequence(expr_tensor=sent)
        else:
          sent = ExpressionSequence(expr_list=sent)
    return sent
