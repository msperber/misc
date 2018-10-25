import dynet as dy

from xnmt.models import base as models
from xnmt import event_trigger, losses, loss_calculators
from xnmt.persistence import Serializable, serializable_init

class DualEncoderSimilarity(models.ConditionedModel, Serializable):
  """
  Trains two encoders to produce the same outputs using a similarity-based loss.

  The two encoders are refered to as src and trg encoders below.

  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the source language
    src_encoder: An encoder to generate encoded source inputs
    trg_embedder: A word embedder for the target language
    trg_encoder: An encoder to generate encoded target inputs
  """

  yaml_tag = "!DualEncoderSimilarity"

  @serializable_init
  def __init__(self, src_reader, trg_reader, src_embedder, src_encoder, trg_embedder, trg_encoder):
    super().__init__(src_reader, trg_reader)
    self.src_embedder = src_embedder
    self.src_encoder = src_encoder
    self.trg_embedder = trg_embedder
    self.trg_encoder = trg_encoder

  def calc_loss(self, src, trg, loss_calculator):

    event_trigger.start_sent(src)

    src_embeddings = self.src_embedder.embed_sent(src)
    src_encodings = self.src_encoder(src_embeddings)

    trg_embeddings = self.trg_embedder.embed_sent(trg)
    trg_encodings = self.trg_encoder(trg_embeddings)

    model_loss = losses.FactoredLossExpr()
    model_loss.add_loss("dist", loss_calculator(src_encodings, trg_encodings))

    return model_loss


class DistLoss(Serializable, loss_calculators.LossCalculator):
  """
  A loss for the similarity of the two sequences by taking each sequence's average vector and computing e.g. MSE.

  Args:
     dist_op: a DyNet operation that computes a loss based on the difference of the two average vectors.
  """

  yaml_tag = '!DistLoss'

  @serializable_init
  def __init__(self, dist_op: str = "squared_norm") -> None:
    if callable(dist_op):
      self.dist_op = dist_op
    else:
      self.dist_op = getattr(dy, dist_op)

  def __call__(self, src_encodings, trg_encodings):
    src_avg = dy.sum_dim(src_encodings.as_tensor(), [1])/(src_encodings.as_tensor().dim()[0][1])
    trg_avg = dy.sum_dim(trg_encodings.as_tensor(), [1])/(trg_encodings.as_tensor().dim()[0][1])
    return self.dist_op(src_avg - trg_avg)
