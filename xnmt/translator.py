from __future__ import division, generators

import dynet as dy
from batcher import *
from search_strategy import *
from vocab import Vocab
from serializer import Serializable, DependentInitParam
from train_test_interface import TrainTestInterface
from embedder import SimpleWordEmbedder
from decoder import MlpSoftmaxDecoder
from output import TextOutput
from batcher import Batcher

class Translator(TrainTestInterface):
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''

  def calc_loss(self, src, trg):
    '''Calculate loss based on input-output pairs.

    :param src: The source, a sentence or a batch of sentences.
    :param trg: The target, a sentence or a batch of sentences.
    :returns: An expression representing the loss.
    '''
    raise NotImplementedError('calc_loss must be implemented for Translator subclasses')

  def translate(self, src):
    '''Translate a particular sentence.

    :param src: The source, a sentence or a batch of sentences.
    :returns: A translated expression.
    '''
    raise NotImplementedError('translate must be implemented for Translator subclasses')

  def set_train(self, val):
    for component in self.get_train_test_components():
      Translator.set_train_recursive(component, val)
  @staticmethod
  def set_train_recursive(component, val):
    component.set_train(val)
    for sub_component in component.get_train_test_components():
      Translator.set_train_recursive(sub_component, val)

  @staticmethod
  def translator_from_spec(input_embedder, encoder, attender, output_embedder, decoder, 
                           label_smoothing_weights):
    if label_smoothing_weights is None or len(label_smoothing_weights)==0:
      return DefaultTranslator(input_embedder, encoder, attender, output_embedder, decoder)
    else:
      return NeighborLabelSmoothingTranslator(input_embedder, encoder, attender, 
                                              output_embedder, decoder,
                                              label_smoothing_weights)
    

class DefaultTranslator(Translator, Serializable):
  '''
  A default translator based on attentional sequence-to-sequence models.
  '''

  yaml_tag = u'!DefaultTranslator'


  def __init__(self, src_embedder, encoder, attender, trg_embedder, decoder):
    '''Constructor.

    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    '''
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder

  def shared_params(self):
    return [
            set(["src_embedder.emb_dim", "encoder.input_dim"]),
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]), # TODO: encoder.hidden_dim may not always exist (e.g. for CNN encoders), need to deal with that case
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"]),
            ]
  def dependent_init_params(self):
    return [
            DependentInitParam(param_descr="src_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].src_reader.vocab_size()),
            DependentInitParam(param_descr="decoder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size()),
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.context["corpus_parser"].trg_reader.vocab_size()),
            ]

  def get_train_test_components(self):
    return [self.encoder, self.decoder]

  def calc_loss(self, src, trg, info=None):
    embeddings = self.src_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    self.attender.start_sent(encodings)
    self.decoder.initialize()
    if Batcher.is_batched(src):
      batch_size = len(src)
      self.decoder.add_input(self.trg_embedder.embed(Batcher.mark_as_batch([0] * batch_size)))  # XXX: HACK, need to initialize decoder better
    else:
      self.decoder.add_input(self.trg_embedder.embed(0))  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not Batcher.is_batched(src):
      for ref_word in trg:
        context = self.attender.calc_context(self.decoder.state.output())
        word_loss = self.decoder.calc_loss(context, ref_word)
        losses.append(word_loss)
        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    # minibatch mode
    else:
      max_len = max([len(single_trg) for single_trg in trg])

      for i in range(max_len):
        ref_word = Batcher.mark_as_batch([single_trg[i] if i < len(single_trg) else Vocab.ES for single_trg in trg])
        context = self.attender.calc_context(self.decoder.state.output())

        word_loss = self.decoder.calc_loss(context, ref_word)
        mask_exp = dy.inputVector([1 if i < len(single_trg) else 0 for single_trg in trg])
        mask_exp = dy.reshape(mask_exp, (1,), len(trg))
        word_loss = dy.sum_batches(word_loss * mask_exp)
        losses.append(word_loss)

        self.decoder.add_input(self.trg_embedder.embed(ref_word))

    return dy.esum(losses)

  def translate(self, src, trg_vocab, search_strategy=None, report=None):
    # Not including this as a default argument is a hack to get our documentation pipeline working
    if search_strategy == None:
      search_strategy = BeamSearch(1, len_norm=NoNormalization())
    if not Batcher.is_batched(src):
      src = Batcher.mark_as_batch([src])
    outputs = []
    for sents in src:
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder.transduce(embeddings)
      self.attender.start_sent(encodings)
      self.decoder.initialize()
      output_actions = search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder, src_length=len(sents))
      if report != None:
        report.trg_words = [trg_vocab[x] for x in output_actions[1:]] # The first token is the start token
        report.attentions = self.attender.attention_vecs
      outputs.append(TextOutput(output_actions, trg_vocab))
    return outputs

class NeighborLabelSmoothingTranslator(DefaultTranslator):
  """
  Label smoothing, according to https://arxiv.org/pdf/1612.02695.pdf
  """
  def __init__(self, input_embedder, encoder, attender, output_embedder, decoder,
               smoothing_weights=[1.0]):
    '''Constructor.

    :param input_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param output_embedder: A word embedder for the output language
    :param decoder: A decoder
    :param smoothing_weights: list of floats; w[0] is for the current label, w[1] for the immediate adjacent neighbors, w[2] for their neighbors, etc.
    '''
    self.input_embedder = input_embedder
    self.encoder = encoder
    self.attender = attender
    self.output_embedder = output_embedder
    self.decoder = decoder
    self.smoothing_weights = smoothing_weights
    weight_sum = sum(smoothing_weights) + sum(smoothing_weights[1:])
    if weight_sum != 1.0: raise RuntimeError("smoothing weights must add to 1.0 after duplicating, but adds to %s" % weight_sum)
    self.serialize_params = [input_embedder, encoder, attender, output_embedder, decoder, smoothing_weights]

  
  def calc_loss(self, src, trg):
    embeddings = self.input_embedder.embed_sent(src)
    encodings = self.encoder.transduce(embeddings)
    self.attender.start_sent(encodings)
    self.decoder.initialize()
    self.decoder.add_input(self.output_embedder.embed(0))  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not Batcher.is_batch_sent(src):
      for ref_word_i in range(len(trg)):
        context = self.attender.calc_context(self.decoder.state.output())
        for label_i in range(len(self.smoothing_weights)):
          word_loss = self.decoder.calc_loss(context, trg[ref_word_i+label_i] if ref_word_i+label_i < len(trg) else Vocab.ES)
          losses.append(word_loss * self.smoothing_weights[label_i])
          if label_i > 0:
            word_loss = self.decoder.calc_loss(context, trg[ref_word_i-label_i] if ref_word_i-label_i >=0 else Vocab.ES)
            losses.append(word_loss * self.smoothing_weights[label_i])
        self.decoder.add_input(self.output_embedder.embed(trg[ref_word_i]))

    # minibatch mode
    else:
      max_len = max([len(single_trg) for single_trg in trg])

      for ref_word_i in range(max_len):
        context = self.attender.calc_context(self.decoder.state.output())
        for label_i in range(len(self.smoothing_weights)):
          if ref_word_i+label_i < max_len:
            ref_word = Batcher.mark_as_batch([single_trg[ref_word_i+label_i] if ref_word_i+label_i < len(single_trg) else Vocab.ES for single_trg in trg])
            word_loss = self.decoder.calc_loss(context, ref_word)
            mask_exp = dy.inputVector([1 if ref_word_i+label_i < len(single_trg) else 0 for single_trg in trg])
            mask_exp = dy.reshape(mask_exp, (1,), len(trg))
            word_loss = dy.sum_batches(word_loss * mask_exp)
            losses.append(word_loss * self.smoothing_weights[label_i])
          if label_i > 0 and ref_word_i-label_i>=0:
            ref_word = Batcher.mark_as_batch([single_trg[ref_word_i-label_i] if 0 <= ref_word_i-label_i < len(single_trg) else Vocab.ES for single_trg in trg])
            word_loss = self.decoder.calc_loss(context, ref_word)
            mask_exp = dy.inputVector([1 if 0 <= ref_word_i-label_i < len(single_trg) else 0 for single_trg in trg])
            mask_exp = dy.reshape(mask_exp, (1,), len(trg))
            word_loss = dy.sum_batches(word_loss * mask_exp)
            losses.append(word_loss * self.smoothing_weights[label_i])

        self.decoder.add_input(self.output_embedder.embed(ref_word))

    return dy.esum(losses)
