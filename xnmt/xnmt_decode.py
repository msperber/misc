# coding: utf-8

from output import *
from serializer import *
import codecs
import sys
from options import OptionParser, Option
from io import open

'''
This will be the main class to perform decoding.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("model_file", force_flag=True, required=True, help_str="pretrained (saved) model path"),
  Option("src_file", help_str="path of input src file to be translated"),
  Option("trg_file", help_str="path of file where expected trg translatons will be written"),
  Option("max_src_len", int, required=False, help_str="Remove sentences from data to decode that are longer than this on the source side"),
  Option("max_num_sents", int, required=False, help_str="Consider only first n sentences"),
  Option("input_format", default_value="text", help_str="format of input data: text/contvec"),
  Option("post_process", default_value="none", help_str="post-processing of translation outputs: none/join-char/join-bpe"),
  Option("beam", int, default_value=1),
  Option("max_len", int, default_value=100),
]

NO_DECODING_ATTEMPTED = u"@@NO_DECODING_ATTEMPTED@@"

def xnmt_decode(args, model_elements=None):
  """
  :param model_elements: If None, the model will be loaded from args.model_file. If set, should
  equal (src_vocab, trg_vocab, translator).
  """
  if model_elements is None:
    model = dy.Model()
    model_serializer = JSONSerializer()
    model_params = model_serializer.load_from_file(args.model_file, model)

    src_vocab = Vocab(model_params.src_vocab)
    trg_vocab = Vocab(model_params.trg_vocab)

    translator = DefaultTranslator(model_params.input_embedder, model_params.encoder, model_params.attender, model_params.output_embedder, model_params.decoder)

  else:
    src_vocab, trg_vocab, translator = model_elements

  input_reader = InputReader.create_input_reader(args.input_format, src_vocab)
  input_reader.freeze()

  if args.post_process=="none":
    output_generator = PlainTextOutput()
  elif args.post_process=="join-char":
    output_generator = JoinedCharTextOutput()
  elif args.post_process=="join-bpe":
    output_generator = JoinedBPETextOutput()
  else:
    raise RuntimeError("Unkonwn postprocessing argument {}".format(args.postprocess)) 
  output_generator.load_vocab(trg_vocab)

  src_corpus = input_reader.read_file(args.src_file)
  
  search_strategy=BeamSearch(b=args.beam, max_len=args.max_len, len_norm=PolynomialNormalization(apply_during_search=True))

  # Perform decoding

  translator.set_train(False)
  with open(args.trg_file, 'wb') as fp:  # Saving the translated output to a trg file
    for src_i, src in enumerate(src_corpus):
      if args.max_src_len is not None and len(src) > args.max_src_len:
        trg_sent = NO_DECODING_ATTEMPTED
      elif args.max_num_sents is not None and src_i >= args.max_num_sents:
        trg_sent = NO_DECODING_ATTEMPTED
      else:
        dy.renew_cg()
        token_string = translator.translate(src, search_strategy)
        trg_sent = output_generator.process(token_string)[0]

      if isinstance(trg_sent, unicode):
        trg_sent = trg_sent.encode('utf-8', errors='ignore')

      else:  # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        #trg_sent = unicode(trg_sent, 'utf-8', errors='ignore').encode('utf-8', errors='ignore')
        trg_sent = trg_sent.decode('utf-8', errors='ignore').encode('utf-8', errors='ignore')

      fp.write(trg_sent + '\n')


if __name__ == "__main__":
  # Parse arguments
  parser = OptionParser()
  parser.add_task("decode", options)
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)

