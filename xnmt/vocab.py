class Vocab:
  '''
  Converts between strings and integer ids
  '''

  SS = 0
  ES = 1

  def __init__(self, i2w=None, vocab_file=None):
    """
    :param i2w: list of words, including <s> and </s>
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    i2w and vocab_file are mutually exclusive
    """
    assert i2w is None or vocab_file is None
    if vocab_file:
      i2w = Vocab.i2w_from_vocab_file(vocab_file)
    if (i2w is not None):
      self.i2w = i2w
      self.w2i = {word: id for (id, word) in enumerate(self.i2w)}
      self.frozen = False
      return
    self.w2i = {}
    self.i2w = []
    self.frozen = False
    self.unk_token = None
    self.w2i[u'<s>'] = self.SS
    self.w2i[u'</s>'] = self.ES
    self.i2w.append(u'<s>')
    self.i2w.append(u'</s>')

  @staticmethod
  def i2w_from_vocab_file(vocab_file):
    """
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    """
    vocab = [u"<s>", u"</s>"]
    with open(vocab_file) as f:
      for line in f:
        word = line.decode('utf-8').strip()
        if word in [u'<s>', u'</s>', u'<unk>']:
          raise RuntimeError("Vocab file {} contains a reserved word: {}" % (vocab_file, word))
        vocab.append(word)
    return vocab

  def convert(self, w):
    assert isinstance(w, unicode)
    if w not in self.w2i:
      if self.frozen:
        assert self.unk_token != None, 'Attempt to convert an OOV in a frozen vocabulary with no UNK token set'
        return self.unk_token
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    return self.w2i[w]

  def __getitem__(self, i):
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

  def freeze(self):
    self.frozen = True

  def set_unk(self, w):
    assert self.frozen, 'Attempt to call set_unk on a non-frozen dict'
    if w not in self.w2i:
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    self.unk_token = self.w2i[w]
