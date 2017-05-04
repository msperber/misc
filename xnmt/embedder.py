from __future__ import division, generators

from batcher import *

class Embedder:
  '''
  A template class to embed a word or token.
  '''

  '''
  Takes a string or word ID and returns its embedding.
  '''

  def embed(self, x):
    raise NotImplementedError('embed must be implemented in Embedder subclasses')

  @staticmethod
  def from_spec(input_format, vocab_size, emb_dim, model):
    input_format_lower = input_format.lower()
    if input_format_lower == "text":
      return SimpleWordEmbedder(vocab_size, emb_dim, model)
    elif input_format_lower == "contvec":
      return NoopEmbedder(emb_dim, model)
    else:
      raise RuntimeError("Unknown input type {}".format(input_format))
  
  def get_embed_dim(self):
    return self.emb_dim

class ExpressionSequence():
  """
  A class to represent a sequence of expressions.
  
  Internal representation is either a list of expressions or a single tensor or both.
  If necessary, both forms of representation are created from the other on demand.
  """
  def __init__(self, **kwargs):
    """
    :param expr_list: a python list of expressions
    :param expr_tensor: a tensor where highest dimension are the sequence items
    :raises valueError: raises an exception if neither expr_list nor expr_tensor are given,
                        or if both have inconsistent length
    """
    self.expr_list = kwargs.pop('expr_list', None)
    self.expr_tensor = kwargs.pop('expr_tensor', None)
    if not (self.expr_list or self.expr_tensor):
      raise ValueError("must provide expr_list or expr_tensor")
    if self.expr_list and self.expr_tensor:
      if len(self.expr_list) != self.expr_tensor.dim()[0][0]:
        raise ValueError("expr_list and expr_tensor must be of same length")

  def __len__(self):
    """
    :returns: length of sequence
    """
    if self.expr_list: return len(self.expr_list)
    else: return self.expr_tensor.dim()[0][0]

  def __iter__(self):
    """
    :returns: iterator over the sequence; results in explicit conversion to list
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return iter(self.expr_list)

  def __getitem__(self, key):
    """
    :returns: sequence item (expression); does not result in explicit conversion to list
    """
    if self.expr_list: return self.expr_list[key]
    else: return dy.pick(self.expr_tensor, key)

  def as_tensor(self):
    """
    :returns: the whole sequence as a tensor expression. 
    """
    if self.expr_tensor is None:
      self.expr_tensor = dy.concatenate(list(map(lambda x:dy.transpose(x), self)))
    return self.expr_tensor
      
class SimpleWordEmbedder(Embedder):
  """
  Simple word embeddings via lookup.
  """

  def __init__(self, vocab_size, emb_dim, model):
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.embeddings = model.add_lookup_parameters((vocab_size, emb_dim))
    self.serialize_params = [vocab_size, emb_dim, model]

  def embed(self, x):
    # single mode
    if not Batcher.is_batch_word(x):
      return self.embeddings[x]
    # minibatch mode
    else:
      return self.embeddings.batch(x)

  def embed_sentence(self, sentence):
    # single mode
    if not Batcher.is_batch_sentence(sentence):
      embeddings = [self.embed(word) for word in sentence]
    # minibatch mode
    else:
      embeddings = []
      for word_i in range(len(sentence[0])):
        embeddings.append(self.embed(Batcher.mark_as_batch([single_sentence[word_i] for single_sentence in sentence])))

    return ExpressionSequence(expr_list=embeddings)

class NoopEmbedder(Embedder):
  """
  This embedder performs no lookups but only passes through the inputs.
  
  Normally, then input is an Input object, which is converted to an expression.
  
  We can also input an ExpressionSequence, which is simply returned as-is.
  This is useful e.g. to stack several encoders, where the second encoder performs no
  lookups.
  """
  def __init__(self, emb_dim, model):
    self.emb_dim = emb_dim
    self.serialize_params = [emb_dim, model]

  def embed(self, x):
    if isinstance(x, dy.Expression): return x
    # single mode
    if not Batcher.is_batch_word(x):
      return dy.inputVector(x)
    # minibatch mode
    else:
      return dy.inputTensor(x, batched=True)

  def embed_sentence(self, sentence):
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    if isinstance(sentence, ExpressionSequence):
      return sentence
    
    batched = Batcher.is_batch_sentence(sentence)
    first_sent = sentence[0] if batched else sentence
    if hasattr(first_sent, "get_array"):
      if not batched:
        return ExpressionSequence(expr_tensor=dy.inputTensor(sentence.get_array(), batched=False))
      else:
        return ExpressionSequence(expr_tensor=dy.inputTensor(map(lambda s: s.get_array(), sentence), batched=True))
    else:
      if not batched:
        embeddings = [self.embed(word) for word in sentence]
      else:
        embeddings = []
        for word_i in range(len(first_sent)):
          embeddings.append(self.embed(Batcher.mark_as_batch([single_sentence[word_i] for single_sentence in sentence])))
      return ExpressionSequence(expr_list=embeddings)

