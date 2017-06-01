import dynet as dy
from batcher import *
import residual
import pyramidal
import conv_encoder
from embedder import ExpressionSequence
import lstm
from translator import TrainTestInterface

class Encoder(TrainTestInterface):
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
  def from_spec(encoder_spec, model):
    """Create an encoder from a specification.

    :param encoder_spec: Encoder-specific settings (encoders must consume all provided settings)
    :param model: The model that we should add the parameters to
    """
#      stridedConv = StridedConvEncoder(layers, input_dim, model) 
#      return ModularEncoder(model,
#                             stridedConv,
#                             ConvLSTMEncoder(layers=1, input_dim=stridedConv.builder.get_output_dim(), model=model, chn_dim=32),
#                             NetworkInNetworkBiLSTMEncoder(layers, stridedConv.builder.get_output_dim()*2, output_dim, model),
#                             BiLSTMEncoder(layers, output_dim, output_dim, model, dropout=0.0),
                            
    registered_encoders = {
                         "bilstm" : BiLSTMEncoder,
                         "residuallstm" : ResidualLSTMEncoder,
                         "residualbilstm" : ResidualBiLSTMEncoder,
                         "pyramidalbilstm" : PyramidalLSTMEncoder,
                         "convbilstm" : ConvLSTMEncoder,
                         "stridedconv" : StridedConvEncoder,
                         "convlstm" : ConvLSTMEncoder,
                         "ninbilstm" : NetworkInNetworkBiLSTMEncoder,
                         "modular" : ModularEncoder
                         }

    encoder_type = encoder_spec["type"].lower()
    if encoder_type not in registered_encoders:
      raise RuntimeError("Unknown encoder type %s, choices are: %s" % (encoder_type, registered_encoders.keys()))
    return registered_encoders[encoder_type](encoder_spec, model)

class BuilderEncoder(Encoder):
  def __init__(self, encoder_spec, model):
    self.serialize_params = [encoder_spec, model]
    self.init_builder(encoder_spec, model)
  def transduce(self, sent):
    return self.builder.transduce(sent)
  def init_builder(self, encoder_spec, model):
    raise NotImplementedError("init_builder() must be implemented by BuilderEncoder subclasses")
  def use_params(self, encoder_spec, params):
    """
    Slightly hacky first approach toward formalized documentation / logging.
    """
    ret = []
    print("> encoder %s:" % (encoder_spec["type"]))
    for param in params:
      if type(param)==str:
        if param not in encoder_spec:
          raise RuntimeError("Missing encoder param %s in encoder %s" % (param, encoder_spec["type"]))
        ret.append(encoder_spec[param])
        print("  %s: %s" % (param, ret[-1]))
      else:
        ret.append(param)
    return ret

class BiLSTMEncoder(BuilderEncoder):
  def init_builder(self, encoder_spec, model):
    params = self.use_params(encoder_spec, ["layers", "input_dim", "output_dim", model, dy.VanillaLSTMBuilder])
    self.builder = dy.BiRNNBuilder(*params)

class ResidualLSTMEncoder(BuilderEncoder):
  def init_builder(self, encoder_spec, model):
    params = self.use_params(encoder_spec, ["layers", "input_dim", "output_dim", model, dy.VanillaLSTMBuilder, "residual_to_output"])
    self.builder = residual.ResidualRNNBuilder(*params)

class ResidualBiLSTMEncoder(BuilderEncoder):
  def init_builder(self, encoder_spec, model):
    params = self.use_params(encoder_spec, ["layers", "input_dim", "output_dim", model, dy.VanillaLSTMBuilder, "residual_to_output"])
    self.builder = residual.ResidualBiRNNBuilder(*params)

class PyramidalLSTMEncoder(BuilderEncoder):
  def init_builder(self, encoder_spec, model):
    params = self.use_params(encoder_spec, ["layers", "input_dim", "output_dim", model, dy.VanillaLSTMBuilder])
    self.builder = pyramidal.PyramidalRNNBuilder(*params)

class ConvLSTMEncoder(BuilderEncoder):
  def init_builder(self, encoder_spec, model):
    params = self.use_params(encoder_spec, ["layers", "input_dim", model, "chn_dim"])
    self.builder = lstm.ConvLSTMBuilder(*params)
    
class StridedConvEncoder(BuilderEncoder):
  def init_builder(self, encoder_spec, model):
    params = self.use_params(encoder_spec, ["layers", "input_dim", model, "chn_dim", 
                                            "num_filters", "output_tensor", "batch_norm"])
    self.builder = conv_encoder.StridedConvEncBuilder(*params)
  def set_train(self, val):
    self.builder.train = val

class NetworkInNetworkBiLSTMEncoder(BuilderEncoder):
  def init_builder(self, encoder_spec, model):
    params = self.use_params(encoder_spec, ["layers", "input_dim", "output_dim", model, 
                                            dy.VanillaLSTMBuilder, "batch_norm"])
    self.builder = lstm.NetworkInNetworkBiRNNBuilder(*params)
  def set_train(self, val):
    self.builder.train = val
  
class ModularEncoder(Encoder):
  def __init__(self, encoder_spec, model):
    self.modules = []
    if encoder_spec.get("input_dim", None) != encoder_spec["modules"][0].get("input_dim"):
      raise RuntimeError("Mismatching input dimensions of first module: %s != %s".format(encoder_spec.get("input_dim", None), encoder_spec["modules"][0].get("input_dim")))
    if encoder_spec.get("ouput_dim", None) != encoder_spec["modules"][-1].get("ouput_dim"):
      raise RuntimeError("Mismatching ouput dimensions of last module: %s != %s".format(encoder_spec.get("output_dim", None), encoder_spec["modules"][-1].get("output_dim")))
    for module_spec in encoder_spec["modules"]:
      self.modules.append(Encoder.from_spec(module_spec, model))
    self.serialize_params = [encoder_spec, model]

  def transduce(self, sent, train=False):
    for i, module in enumerate(self.modules):
      sent = module.transduce(sent)
      if i<len(self.modules)-1:
        if type(sent)==dy.Expression:
          sent = ExpressionSequence(expr_tensor=sent)
        else:
          sent = ExpressionSequence(expr_list=sent)
    return sent
