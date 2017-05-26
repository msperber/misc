from __future__ import division, generators

import dynet as dy
from batcher import *
from search_strategy import *
from vocab import Vocab

class Translator:
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''

  '''
  Calculate the loss of the input and output.
  '''

  def loss(self, x, y):
    raise NotImplementedError('loss must be implemented for Translator subclasses')

  '''
  Calculate the loss for a batch. By default, just iterate. Overload for better efficiency.
  '''

  def batch_loss(self, xs, ys):
    return dy.esum([self.loss(x, y) for x, y in zip(xs, ys)])

class DefaultTranslator(Translator):
  def __init__(self, input_embedder, encoder, attender, output_embedder, decoder):
    self.input_embedder = input_embedder
    self.encoder = encoder
    self.attender = attender
    self.output_embedder = output_embedder
    self.decoder = decoder

  def calc_loss(self, source, target, train=False):
    embeddings = self.input_embedder.embed_sentence(source)
    encodings = self.encoder.transduce(embeddings, train=train)
    self.attender.start_sentence(encodings)
    self.decoder.initialize()
    self.decoder.add_input(self.output_embedder.embed(0))  # XXX: HACK, need to initialize decoder better
    losses = []

    # single mode
    if not Batcher.is_batch_sentence(source):
      for ref_word in target:
        context = self.attender.calc_context(self.decoder.state.output())
        word_loss = self.decoder.calc_loss(context, ref_word)
        losses.append(word_loss)
        self.decoder.add_input(self.output_embedder.embed(ref_word))

    # minibatch mode
    else:
      max_len = max([len(single_target) for single_target in target])

      for i in range(max_len):
        ref_word = Batcher.mark_as_batch([single_target[i] if i < len(single_target) else Vocab.ES for single_target in target])
        context = self.attender.calc_context(self.decoder.state.output())

        word_loss = self.decoder.calc_loss(context, ref_word)
        mask_exp = dy.inputVector([1 if i < len(single_target) else 0 for single_target in target])
        mask_exp = dy.reshape(mask_exp, (1,), len(target))
        word_loss = dy.sum_batches(word_loss * mask_exp)
        losses.append(word_loss)

        self.decoder.add_input(self.output_embedder.embed(ref_word))

    return dy.esum(losses)

  def translate(self, source, search_strategy=BeamSearch(1, len_norm=NoNormalization())):
    output = []
    if not Batcher.is_batch_sentence(source):
      source = Batcher.mark_as_batch([source])
    for sentences in source:
      embeddings = self.input_embedder.embed_sentence(source)
      encodings = self.encoder.transduce(embeddings, train=False)
      self.attender.start_sentence(encodings)
      self.decoder.initialize()
      output.append(search_strategy.generate_output(self.decoder, self.attender, self.output_embedder, source_length=len(sentences)))
    return output
