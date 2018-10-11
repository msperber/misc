from typing import Optional, Sequence
import numbers
import ast

import dynet as dy

from xnmt import batchers, events, expression_seqs, input_readers, param_collections, param_initializers, sent, vocabs
from xnmt.modelparts import embedders
from xnmt.transducers import base as transducers
from xnmt.persistence import bare, Ref, Serializable, serializable_init



class LatticeReader(input_readers.BaseTextReader, Serializable):
  """
  Reads lattices from a text file.

  The expected lattice file format is as follows:
  * 1 line per lattice
  * lines are serialized python lists / tuples
  * 2 lists per lattice:
    - list of nodes, with every node a 4-tuple: (lexicon_entry, fwd_log_prob, marginal_log_prob, bwd_log_prob)
    - list of arcs, each arc a tuple: (node_id_start, node_id_end)
            - node_id references the nodes and is 0-indexed
            - node_id_start < node_id_end
  * All paths must share a common start and end node, i.e. <s> and </s> need to be contained in the lattice

  A simple example lattice:
    [('<s>', 0.0, 0.0, 0.0), ('buenas', 0, 0.0, 0.0), ('tardes', 0, 0.0, 0.0), ('</s>', 0.0, 0.0, 0.0)],[(0, 1), (1, 2), (2, 3)]

  Args:
    vocab: Vocabulary to convert string tokens to integer ids. If not given, plain text will be assumed to contain
           space-separated integer ids.
  """
  yaml_tag = '!LatticeReader'

  @serializable_init
  def __init__(self, vocab: vocabs.Vocab):
    self.vocab = vocab

  def read_sent(self, line, idx):
    node_list, arc_list = ast.literal_eval(line)
    nodes = [sent.LatticeNode(nodes_prev=[], nodes_next=[],
                              value=self.vocab.convert(item[0]),
                              fwd_log_prob=item[1], marginal_log_prob=item[2], bwd_log_prob=item[2])
             for item in node_list]
    for from_index, to_index in arc_list:
      nodes[from_index].nodes_next.append(to_index)
      nodes[to_index].nodes_prev.append(from_index)

    assert nodes[0].value == self.vocab.SS
    assert nodes[-1].value == self.vocab.ES

    return sent.Lattice(idx=idx, nodes=nodes, vocab=self.vocab)

  def vocab_size(self):
    return len(self.vocab)


class LatticeEmbedder(embedders.SimpleWordEmbedder, Serializable):
  """
  Simple word embeddings via lookup.

  Args:
    vocab_size:
    emb_dim:
    word_dropout: drop out word types with a certain probability, sampling word types on a per-sentence level,
                  see https://arxiv.org/abs/1512.05287
  """

  yaml_tag = '!LatticeEmbedder'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               vocab=None,
               vocab_size=None,
               emb_dim=Ref("exp_global.default_layer_dim"),
               word_dropout=0.0,
               arc_dropout=0.0,
               yaml_path=None,
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(
                 param_initializers.GlorotInitializer)),
               src_reader=Ref("model.src_reader", default=None),
               trg_reader=Ref("model.trg_reader", default=None)):
    # TODO: refactor by taking a base embedder, and only adding the lattice structure on top of its output?
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    self.emb_dim = emb_dim
    self.word_dropout = word_dropout
    param_collection = param_collections.ParamManager.my_params(self)
    self.embeddings = param_collection.add_lookup_parameters((self.vocab_size, self.emb_dim),
                                                             init=param_init.initializer(
                                                               (self.vocab_size, self.emb_dim), is_lookup=True))
    self.word_id_mask = None
    self.weight_noise = 0.0
    self.fix_norm = None
    self.arc_dropout = arc_dropout

  def embed_sent(self, s):
    if batchers.is_batched(s):
      assert len(s) == 1, "LatticeEmbedder requires batch size of 1"
      assert s.mask is None
      s = s[0]
    embedded_nodes = [word.new_node_with_val(self.embed(word.value)) for word in s]
    return sent.Lattice(idx=s.idx, nodes=embedded_nodes, vocab=s.vocab)

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val


class LatticeLSTMTransducer(transducers.SeqTransducer, Serializable):
  """
  A lattice LSTM.

  This is the unidirectional single-layer lattice LSTM.

  Args:
    input_dim: size of inputs
    hidden_dim: number of hidden units
    dropout: dropout rate for variational dropout, or 0.0 to disable dropout
  """

  yaml_tag = "!LatticeLSTMTransducer"

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0)) -> None:
    self.dropout_rate = dropout
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    model = param_collections.ParamManager.my_params(self)

    # [i; o; g]
    self.p_Wx_iog = model.add_parameters(dim=(hidden_dim * 3, input_dim))
    self.p_Wh_iog = model.add_parameters(dim=(hidden_dim * 3, hidden_dim))
    self.p_b_iog = model.add_parameters(dim=(hidden_dim * 3,), init=dy.ConstInitializer(0.0))
    self.p_Wx_f = model.add_parameters(dim=(hidden_dim, input_dim))
    self.p_Wh_f = model.add_parameters(dim=(hidden_dim, hidden_dim))
    self.p_b_f = model.add_parameters(dim=(hidden_dim,), init=dy.ConstInitializer(1.0))

    self.dropout_mask_x = None
    self.dropout_mask_h = None

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None
    self.dropout_mask_x = None
    self.dropout_mask_h = None

  def get_final_states(self):
    return self._final_states

  def set_dropout_masks(self, batch_size=1):
    if self.dropout_rate > 0.0 and self.train:
      retention_rate = 1.0 - self.dropout_rate
      scale = 1.0 / retention_rate
      self.dropout_mask_x = dy.random_bernoulli((self.input_dim,), retention_rate, scale, batch_size=batch_size)
      self.dropout_mask_h = dy.random_bernoulli((self.hidden_dim,), retention_rate, scale, batch_size=batch_size)

  def transduce(self, lattice):
    Wx_iog = dy.parameter(self.p_Wx_iog)
    Wh_iog = dy.parameter(self.p_Wh_iog)
    b_iog = dy.parameter(self.p_b_iog)
    Wx_f = dy.parameter(self.p_Wx_f)
    Wh_f = dy.parameter(self.p_Wh_f)
    b_f = dy.parameter(self.p_b_f)
    h = []
    c = []

    batch_size = lattice[0].value.dim()[1]
    if self.dropout_rate > 0.0 and self.train:
      self.set_dropout_masks(batch_size=batch_size)

    for x_t in lattice:
      val = x_t.value
      if self.dropout_rate > 0.0 and self.train:
        val = dy.cmult(val, self.dropout_mask_x)
      i_ft_list = []
      if len(x_t.nodes_prev) == 0:
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val])
      else:
        h_tilde = sum(h[pred] for pred in x_t.nodes_prev)
        tmp_iog = dy.affine_transform([b_iog, Wx_iog, val, Wh_iog, h_tilde])
        for pred in x_t.nodes_prev:
          i_ft_list.append(dy.logistic(dy.affine_transform([b_f, Wx_f, val, Wh_f, h[pred]])))
      i_ait = dy.pick_range(tmp_iog, 0, self.hidden_dim)
      i_aot = dy.pick_range(tmp_iog, self.hidden_dim, self.hidden_dim * 2)
      i_agt = dy.pick_range(tmp_iog, self.hidden_dim * 2, self.hidden_dim * 3)

      i_it = dy.logistic(i_ait)
      i_ot = dy.logistic(i_aot)
      i_gt = dy.tanh(i_agt)
      if len(x_t.nodes_prev) == 0:
        c.append(dy.cmult(i_it, i_gt))
      else:
        fc = dy.cmult(i_ft_list[0], c[x_t.nodes_prev[0]])
        for i in range(1, len(x_t.nodes_prev)):
          fc += dy.cmult(i_ft_list[i], c[x_t.nodes_prev[i]])
        c.append(fc + dy.cmult(i_it, i_gt))
      h_t = dy.cmult(i_ot, dy.tanh(c[-1]))
      if self.dropout_rate > 0.0 and self.train:
        h_t = dy.cmult(h_t, self.dropout_mask_h)
      h.append(h_t)
    self._final_states = [transducers.FinalTransducerState(h[-1], c[-1])]
    return sent.Lattice(idx=lattice.idx,
                        nodes=[node_t.new_node_with_val(h_t) for node_t, h_t in zip(lattice.nodes, h)],
                        vocab=lattice.vocab)


class BiLatticeLSTMTransducer(transducers.SeqTransducer, Serializable):
  """
  A multi-layered bidirectional lattice LSTM.

  Makes use of several LatticeLSTMTransducer instances and combines them appropriately.

  Args:
    layers: number of layers
    input_dim: size of inputs
    hidden_dim: number of hidden units
    dropout: dropout rate for variational dropout, or 0.0 to disable dropout
    forward_layers: determined automatically
    backward_layers: determined automatically
  """

  yaml_tag = '!BiLatticeLSTMTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               layers: numbers.Integral = 1,
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               dropout: numbers.Real = Ref("exp_global.dropout", default=0.0),
               forward_layers: Optional[Sequence[LatticeLSTMTransducer]] = None,
               backward_layers: Optional[Sequence[LatticeLSTMTransducer]] = None) -> None:
    self.num_layers = layers
    input_dim = input_dim
    hidden_dim = hidden_dim
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers",
                                                          forward_layers,
                                                          lambda: self._make_dir_layers(input_dim=input_dim,
                                                                                        hidden_dim=hidden_dim,
                                                                                        dropout=dropout,
                                                                                        layers=layers))
    self.backward_layers = self.add_serializable_component("backward_layers",
                                                           backward_layers,
                                                           lambda: self._make_dir_layers(input_dim=input_dim,
                                                                                         hidden_dim=hidden_dim,
                                                                                         dropout=dropout,
                                                                                         layers=layers))

  def _make_dir_layers(self, input_dim, hidden_dim, dropout, layers):
    dir_layers = [LatticeLSTMTransducer(input_dim=input_dim, hidden_dim=hidden_dim / 2, dropout=dropout)]
    dir_layers += [LatticeLSTMTransducer(input_dim=hidden_dim, hidden_dim=hidden_dim / 2, dropout=dropout) for
                            _ in range(layers - 1)]
    return dir_layers

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self._final_states = None

  def get_final_states(self):
    return self._final_states

  def transduce(self, lattice):
    if isinstance(lattice, expression_seqs.ExpressionSequence):
      lattice = sent.Lattice(idx=lattice.idx,
                             nodes=[sent.LatticeNode([i - 1] if i > 0 else [],
                                                     [i + 1] if i < len(lattice) - 1 else [],
                                                     value)
                                    for (i, value) in enumerate(lattice)])

    # first layer
    forward_es = self.forward_layers[0].transduce(lattice)
    rev_backward_es = self.backward_layers[0].transduce(lattice.reversed())

    for layer_i in range(1, len(self.forward_layers)):
      concat_fwd = sent.Lattice(idx=lattice.idx,
                                nodes=[node_fwd.new_node_with_val(dy.concatenate([node_fwd.value, node_bwd.value]))
                                       for node_fwd, node_bwd in zip(forward_es, reversed(rev_backward_es.nodes))],
                                vocab=lattice.vocab)
      concat_bwd = sent.Lattice(idx=lattice.idx,
                                nodes=[node_bwd.new_node_with_val(dy.concatenate([node_fwd.value, node_bwd.value]))
                                       for node_fwd, node_bwd in zip(reversed(forward_es.nodes), rev_backward_es)],
                                vocab=lattice.vocab)
      new_forward_es = self.forward_layers[layer_i].transduce(concat_fwd)
      rev_backward_es = self.backward_layers[layer_i].transduce(concat_bwd)
      forward_es = new_forward_es

    self._final_states = [
      transducers.FinalTransducerState(dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].main_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].main_expr()]),
                                       dy.concatenate([self.forward_layers[layer_i].get_final_states()[0].cell_expr(),
                                                       self.backward_layers[layer_i].get_final_states()[
                                                         0].cell_expr()])) \
      for layer_i in range(len(self.forward_layers))]
    return sent.Lattice(idx=lattice.idx,
                        nodes=[lattice.nodes[i].new_node_with_val(dy.concatenate([forward_es[i].value,
                                                                                  rev_backward_es[-i - 1].value]))
                               for i in range(forward_es.sent_len())],
                        vocab=lattice.vocab)
