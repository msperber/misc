# coding: utf-8

from output import *
from serializer import *
import codecs
import sys
from options import OptionParser, Option

reload(sys)
sys.setdefaultencoding('utf8')

'''
This will be the main class to perform decoding.
'''

options = [
  Option("model_file", force_flag=True, required=True, help="pretrained (saved) model path"),
  Option("source_file", help="path of input source file to be translated"),
  Option("target_file", help="path of file where expected target translatons will be written"),
  Option("input_format", default_value="text", help="format of input data: text/contvec"),
  Option("post_process", default_value="none", help="post-processing of translation outputs: none/join-char/join-bpe"),
]


def xnmt_decode(args, search_strategy=BeamSearch(1, len_norm=NoNormalization()), model_elements=None):
  """
  :param model_elements: If None, the model will be loaded from args.model_file. If set, should
  equal (source_vocab, target_vocab, translator).
  """
  if model_elements is None:
    model = dy.Model()
    model_serializer = JSONSerializer()
    model_params = model_serializer.load_from_file(args.model_file, model)

    source_vocab = Vocab(model_params.source_vocab)
    target_vocab = Vocab(model_params.target_vocab)

    translator = DefaultTranslator(model_params.encoder, model_params.attender, model_params.decoder)

  else:
    source_vocab, target_vocab, translator = model_elements

  input_reader = InputReader.create_input_reader(args.input_format, source_vocab)
  input_reader.freeze()

  if args.post_process=="none":
    output_generator = PlainTextOutput()
  elif args.post_process=="join-char":
    output_generator = JoinedCharTextOutput()
  elif args.post_process=="join-bpe":
    output_generator = JoinedBPETextOutput()
  else:
    raise RuntimeError("Unkonwn postprocessing argument {}".format(args.postprocess)) 
  output_generator.load_vocab(target_vocab)

  source_corpus = input_reader.read_file(args.source_file)

  # Perform decoding
  with codecs.open(args.target_file, 'w', 'utf-8') as fp:  # Saving the translated output to a target file
    for src in source_corpus:
      dy.renew_cg()
      token_string = translator.translate(src, search_strategy)
      target_sentence = output_generator.process(token_string)[0]

      if isinstance(target_sentence, unicode):
        target_sentence = target_sentence.encode('utf8', errors='ignore')

      else:  # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        target_sentence = unicode(target_sentence, 'utf8', errors='ignore').encode('utf8')

      fp.write(target_sentence + u'\n')


if __name__ == "__main__":
  # Parse arguments
  parser = OptionParser()
  parser.add_task("decode", options)
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)

