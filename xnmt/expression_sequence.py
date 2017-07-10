import dynet as dy
from batcher import Batcher

class ExpressionSequence(object):
  """A class to represent a sequence of expressions.
  
  Internal representation is either a list of expressions or a single tensor or both.
  If necessary, both forms of representation are created from the other on demand.
  """
  def __init__(self, **kwargs):
    """Constructor.

    :param expr_list: a python list of expressions
    :param expr_tensor: a tensor where highest dimension are the sequence items
    :raises valueError:
      raises an exception if neither expr_list nor expr_tensor are given,
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
    """Return length.

    :returns: length of sequence
    """
    if self.expr_list: return len(self.expr_list)
    else: return self.expr_tensor.dim()[0][0]

  def __iter__(self):
    """Return iterator.

    :returns: iterator over the sequence; results in explicit conversion to list
    """
    if self.expr_list is None:
      self.expr_list = [self[i] for i in range(len(self))]
    return iter(self.expr_list)

  def __getitem__(self, key):
    """Get a single item.

    :returns: sequence item (expression); does not result in explicit conversion to list
    """
    if self.expr_list: return self.expr_list[key]
    else: return dy.pick(self.expr_tensor, key)

  def as_tensor(self):
    """Get a tensor.
    :returns: the whole sequence as a tensor expression where each column is one of the embeddings.
    """
    if self.expr_tensor is None:
      self.expr_tensor = dy.concatenate_cols(self.expr_list)
    return self.expr_tensor

class LazyNumpyExpressionSequence(ExpressionSequence):
  """
  This is initialized via numpy arrays, and dynet expressions are only created
  once a consumer requests representation as list or tensor.
  """
  def __init__(self, lazy_data):
    """
    :param lazy_data: numpy array, or Batcher.Batch of numpy arrays
    """
    self.lazy_data = lazy_data
    self.expr_list, self.expr_tensor = None, None 
  def __len__(self):
    if self.expr_list or self.expr_tensor:
      return super(LazyNumpyExpressionSequence, self).__len__()
    else:
      if Batcher.is_batch_sent(self.lazy_data):
        return self.lazy_data[0].shape[0] 
      else: return self.lazy_data.shape[0]
  def __iter__(self):
    if not (self.expr_list or self.expr_tensor):
      self.expr_list = [self[i] for i in range(len(self))]
    return super(LazyNumpyExpressionSequence, self).__iter__()
  def __getitem__(self, key):
    if self.expr_list or self.expr_tensor:
      return super(LazyNumpyExpressionSequence, self).__getitem__()
    else:
      if Batcher.is_batch_sent(self.lazy_data):
        return dy.inputTensor([self.lazy_data[batch][key] for batch in range(len(self.lazy_data))], batched=True)
      else:
        return dy.inputTensor(self.lazy_data[key], batched=False)
  def as_tensor(self):
    if not (self.expr_list or self.expr_tensor):
      self.expr_tensor = dy.inputTensor(self.lazy_data, batched=self.batched)
    return super(LazyNumpyExpressionSequence, self).as_tensor()
