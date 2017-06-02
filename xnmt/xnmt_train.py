# coding: utf-8
from __future__ import division

import argparse
import math
import sys
import dynet as dy
from embedder import *
from attender import *
from input import *
from encoder import *
from decoder import *
from translator import *
from model_params import *
from loss_tracker import *
from serializer import *
from options import Option, OptionParser, general_options

'''
This will be the main class to perform training.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("eval_every", int, default_value=1000, force_flag=True),
  Option("batch_size", int, default_value=32, force_flag=True),
  Option("batch_strategy", default_value="src"),
  Option("train_src"),
  Option("train_trg"),
  Option("dev_src"),
  Option("dev_trg"),
  Option("model_file"),
  Option("pretrained_model_file", default_value="", help_str="Path of pre-trained model file"),
  Option("input_vocab", default_value="", help_str="Path of fixed input vocab file"),
  Option("output_vocab", default_value="", help_str="Path of fixed output vocab file"),
  Option("input_format", default_value="text", help_str="Format of input data: text/contvec"),
  Option("default_layer_dim", int, default_value=512, help_str="Default size to use for layers if not otherwise overridden"),
  Option("input_word_embed_dim", int, required=False),
  Option("output_word_embed_dim", int, required=False),
  Option("output_state_dim", int, required=False),
  Option("output_mlp_hidden_dim", int, required=False),
  Option("attender_hidden_dim", int, required=False),
  Option("trainer", default_value="sgd"),
  Option("learning_rate", float, default_value=0.1),
  Option("lr_decay", float, default_value=1.0),
  Option("lr_threshold", float, default_value=1e-5),
  Option("eval_metrics", default_value="bleu"),
  Option("encoder", dict, default_value={}),  
  Option("encoder.layers", int, default_value=1),
  Option("encoder.type", default_value="BiLSTM"),
  Option("encoder.output_dim", int, required=False),
  Option("encoder.input_dim", int, required=False),
  Option("decoder_type", default_value="LSTM"),
  Option("decoder_layers", int, default_value=2),
  Option("residual_to_output", bool, default_value=True,
         help_str="If using residual networks in the decoder, whether to add a residual connection to the output layer"),
]

class XnmtTrainer:
  def __init__(self, args):
    dy.renew_cg()

    self.args = args  # save for later
    self.model = dy.Model()

    if args.trainer.lower() == "sgd":
      self.trainer = dy.SimpleSGDTrainer(self.model, e0 = args.learning_rate)
    elif args.trainer.lower() == "adam":
      self.trainer = dy.AdamTrainer(self.model, alpha = args.learning_rate)
    else:
      raise RuntimeError("Unknown trainer {}".format(args.trainer))
    
    if args.lr_decay > 1.0 or args.lr_decay <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.learning_scale = 1.0
    self.early_stopping_reached = False


    # Create the model serializer
    self.create_model()
    # single mode
    if args.batch_size is None or args.batch_size == 1 or args.batch_strategy.lower() == 'none':
      print('Start training in non-minibatch mode...')
      self.logger = NonBatchLossTracker(args.eval_every, self.total_train_sent)

    # minibatch mode
    else:
      print('Start training in minibatch mode...')
      self.batcher = Batcher.select_batcher(args.batch_strategy)(args.batch_size)
      if args.input_format == "contvec":
        assert self.train_src[0].nparr.shape[1] == self.input_embedder.emb_dim, "input embed dim is different size than expected"
        self.batcher.pad_token = np.zeros(self.input_embedder.emb_dim)
      self.train_src, self.train_trg = self.batcher.pack(self.train_src, self.train_trg)
      self.dev_src, self.dev_trg = self.batcher.pack(self.dev_src, self.dev_trg)
      self.logger = BatchLossTracker(args.eval_every, self.total_train_sent)

  def create_model(self):
    if self.args.pretrained_model_file:
      self.model_serializer = JSONSerializer()
      self.model_params = self.model_serializer.load_from_file(self.args.pretrained_model_file, self.model)
      src_vocab = Vocab(self.model_params.src_vocab)
      trg_vocab = Vocab(self.model_params.trg_vocab)
      self.encoder = self.model_params.encoder
      self.attender = self.model_params.attender
      self.decoder = self.model_params.decoder
      self.input_embedder = self.model_params.input_embedder
      self.output_embedder = self.model_params.output_embedder
      self.translator = DefaultTranslator(self.input_embedder, self.encoder, self.attender, 
                                          self.output_embedder, self.decoder)
      self.input_reader = InputReader.create_input_reader(self.args.input_format, src_vocab)
      self.output_reader = InputReader.create_input_reader("text", trg_vocab)
      self.input_reader.freeze()
      self.output_reader.freeze()
      self.read_data()
      return

    self.model_serializer = JSONSerializer()

    # Read in training and dev corpora
    input_vocab, output_vocab = None, None
    if self.args.input_vocab:
      input_vocab = Vocab(vocab_file=self.args.input_vocab)
    if self.args.output_vocab:
      output_vocab = Vocab(vocab_file=self.args.output_vocab)
    self.input_reader = InputReader.create_input_reader(self.args.input_format, input_vocab)
    self.output_reader = InputReader.create_input_reader("text", output_vocab)
    if self.args.input_vocab:
      self.input_reader.freeze()
    if self.args.output_vocab:
      self.output_reader.freeze()
    self.read_data()
    
    # Get layer sizes: replace by default if not specified
    for opt in ["input_word_embed_dim", "output_word_embed_dim", "output_state_dim",
                "output_mlp_hidden_dim", "attender_hidden_dim", ]:
      if getattr(self.args, opt) is None:
        setattr(self.args, opt, self.args.default_layer_dim)
    if getattr(self.args, "encoder") is None:
      self.encoder = {"input_dim":self.args.input_word_embed_dim, "output_dim" : self.args.default_layer_dim}
    else:
      if self.args.encoder.get("output_dim", None) is None: self.args.encoder["output_dim"] = self.args.default_layer_dim
      if self.args.encoder.get("input_dim", None) is None: self.args.encoder["input_dim"] = self.args.input_word_embed_dim

    self.input_word_emb_dim = self.args.input_word_embed_dim
    self.output_word_emb_dim = self.args.output_word_embed_dim
    self.output_state_dim = self.args.output_state_dim
    self.attender_hidden_dim = self.args.attender_hidden_dim
    self.output_mlp_hidden_dim = self.args.output_mlp_hidden_dim
    self.encoder_hidden_dim = self.args.encoder["output_dim"]

    self.input_embedder = Embedder.from_spec(self.args.input_format, len(self.input_reader.vocab),
                                             self.input_word_emb_dim, self.model)

    self.output_embedder = SimpleWordEmbedder(len(self.output_reader.vocab), self.output_word_emb_dim, self.model)

    self.encoder = Encoder.from_spec(self.args.encoder, self.model)

    self.attender = StandardAttender(self.encoder_hidden_dim, self.output_state_dim, self.attender_hidden_dim,
                                     self.model)

    decoder_rnn = Decoder.rnn_from_spec(self.args.decoder_type, self.args.decoder_layers, self.output_word_emb_dim,
                                        self.output_state_dim, self.model, self.args.residual_to_output)
    self.decoder = MlpSoftmaxDecoder(self.args.decoder_layers, self.encoder_hidden_dim, self.output_state_dim,
                                     self.output_mlp_hidden_dim, len(self.output_reader.vocab),
                                     self.model, self.output_word_emb_dim, decoder_rnn)

    self.translator = DefaultTranslator(self.input_embedder, self.encoder, self.attender, self.output_embedder, self.decoder)
    self.model_params = ModelParams(self.encoder, self.attender, self.decoder, self.input_reader.vocab.i2w,
                                    self.output_reader.vocab.i2w, self.input_embedder, self.output_embedder)



  def read_data(self):
    self.train_src = self.input_reader.read_file(self.args.train_src)
    self.train_trg = self.output_reader.read_file(self.args.train_trg)
    assert len(self.train_src) == len(self.train_trg)
    self.total_train_sent = len(self.train_src)
    if self.args.eval_every == None:
      self.args.eval_every = self.total_train_sent

    self.input_reader.freeze()
    self.output_reader.freeze()

    self.dev_src = self.input_reader.read_file(self.args.dev_src)
    self.dev_trg = self.output_reader.read_file(self.args.dev_trg)
    assert len(self.dev_src) == len(self.dev_trg)


  def run_epoch(self):
    self.logger.new_epoch()

    self.translator.set_train(True)
    for batch_num, (src, trg) in enumerate(zip(self.train_src, self.train_trg)):

      # Loss calculation
      dy.renew_cg()
      loss = self.translator.calc_loss(src, trg)
      self.logger.update_epoch_loss(src, trg, loss.value())

      loss.backward()
      self.trainer.update(self.learning_scale)
      

      # Devel reporting
      if self.logger.report_train_process():
        self.translator.set_train(False)
        self.logger.new_dev()
        for src, trg in zip(self.dev_src, self.dev_trg):
          dy.renew_cg()
          loss = self.translator.calc_loss(src, trg).value()
          self.logger.update_dev_loss(trg, loss)

        # Write out the model if it's the best one
        if self.logger.report_dev_and_check_model(self.args.model_file):
          self.model_serializer.save_to_file(self.args.model_file, self.model_params, self.model)
        else:
          # otherwise: learning rate decay / early stopping
          if self.args.lr_decay < 1.0:
            self.learning_scale *= self.args.lr_decay
            print('new learning rate: %s' % (self.learning_scale * self.args.learning_rate))
          if self.learning_scale * self.args.learning_rate < self.args.lr_threshold:
            print('Early stopping')
            self.early_stopping_reached = True
            
        self.trainer.update_epoch()
        self.translator.set_train(True)

    return math.exp(self.logger.epoch_loss / self.logger.epoch_words), \
           math.exp(self.logger.dev_loss / self.logger.dev_words)


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_task("train", general_options + options)
  args = parser.args_from_command_line("train", sys.argv[1:])
  print("Starting xnmt-train:\nArguments: %r" % (args))

  xnmt_trainer = XnmtTrainer(args)

  while not xnmt_trainer.early_stopping_reached:
    xnmt_trainer.run_epoch()
