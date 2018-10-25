from xnmt import input_readers
from xnmt.models import translators
from xnmt.persistence import Serializable, serializable_init

class EnsembleInputReader(input_readers.InputReader, Serializable):
  """
  An input reader to use for ensembling to models that have different inputs (e.g. speech and text)

  Args:
    input_reader1: input reader for first model
    input_reader2: input reader for secondmodel
  """
  yaml_tag = "!EnsembleInputReader"
  @serializable_init
  def __init__(self, input_reader1, input_reader2):
    # TODO: generalize to n input readers
    self.input_reader1 = input_reader1
    self.input_reader2 = input_reader2
    self.vocab = self.input_reader1.vocab if hasattr(self.input_reader1, "vocab") else  self.input_reader2.vocab # TODO: hack to make EnsembleTranslator happy

  def read_sents(self, filename, filter_ids=None):
    for s in zip(self.input_reader1.read_sents(filename[0]), self.input_reader2.read_sents(filename[1])):
      yield translators.EnsembleListDelegate([s[0],s[1]])

  def count_sents(self, filename):
    cnt = self.input_reader1.count_sents()
    assert cnt == self.input_reader2.count_sents()
    return cnt
