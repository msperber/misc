import dynet as dy
from mlp import MLP
import inspect
from batcher import *
from translator import TrainTestInterface

class Decoder(TrainTestInterface):
  '''
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  '''

  '''
  Document me
  '''

  def calc_loss(self, x, ref_action):
    raise NotImplementedError('calc_loss must be implemented in Decoder subclasses')

class RnnDecoder(Decoder):
  @staticmethod
  def rnn_from_spec(spec, num_layers, input_dim, hidden_dim, model, residual_to_output):
    decoder_type = spec.lower()
    if decoder_type == "lstm":
      return dy.VanillaLSTMBuilder(num_layers, input_dim, hidden_dim, model)
    elif decoder_type == "residuallstm":
      return residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim, model, dy.VanillaLSTMBuilder, residual_to_output)
    else:
      raise RuntimeError("Unknown decoder type {}".format(spec))


class MlpSoftmaxDecoder(RnnDecoder):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.
  def __init__(self, rnn_spec, layers, input_dim, lstm_dim, mlp_hidden_dim, vocab_size, model, trg_embed_dim, dropout,
               residual_to_output):
    self.input_dim = input_dim
    self.dropout = dropout
    self.fwd_lstm = RnnDecoder.rnn_from_spec(rnn_spec, layers, trg_embed_dim, lstm_dim, model, residual_to_output)
    self.mlp = MLP(input_dim + lstm_dim, mlp_hidden_dim, vocab_size, model)
    self.state = None
    self.serialize_params = [rnn_spec, layers, input_dim, lstm_dim, mlp_hidden_dim, vocab_size, model, trg_embed_dim, dropout, residual_to_output]

  def initialize(self):
    self.state = self.fwd_lstm.initial_state()

  def add_input(self, trg_embedding):
    self.state = self.state.add_input(trg_embedding)

  def get_scores(self, context):
    mlp_input = dy.concatenate([context, self.state.output()])
    mlp_input = dy.reshape(mlp_input, (mlp_input.dim()[0][0],))
    scores = self.mlp(mlp_input)
    return scores

  def calc_loss(self, context, ref_action):
    scores = self.get_scores(context)
    # single mode
    if not Batcher.is_batch_word(ref_action):
      return dy.pickneglogsoftmax(scores, ref_action)
    # minibatch mode
    else:
      return dy.pickneglogsoftmax_batch(scores, ref_action)

  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)
