import math
import numbers
from typing import List

import numpy as np
import scipy.sparse, scipy.sparse.csgraph
import dynet as dy

from xnmt import expression_seqs, events, param_collections, param_initializers
from xnmt.transducers import base as transducers
from xnmt.persistence import bare, Ref, Serializable, serializable_init

class LatticePositionalSeqTransducer(transducers.SeqTransducer, Serializable):
  yaml_tag = '!LatticePositionalSeqTransducer'

  @serializable_init
  def __init__(self,
               max_pos: numbers.Integral,
               op: str = 'sum',
               emb_type: str = 'param',
               input_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: param_initializers.ParamInitializer = Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer))):
    """
    max_pos: largest embedded position
    op: how to combine positional encodings with the original encodings, can be "sum" or "concat"
    type: what type of embddings to use, "param"=parameterized (others, such as the trigonometric embeddings are todo)
    input_dim: embedding size
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.input_dim = input_dim
    self.op = op
    self.emb_type = emb_type
    param_init = param_init
    dim = (self.input_dim, max_pos)
    param_collection = param_collections.ParamManager.my_params(self)
    self.embedder = param_collection.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  def transduce(self, src: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    num_nodes = src.sent_len()

    # TODO: should we cache these?
    adj_matrix = np.full((num_nodes, num_nodes), -np.inf)
    for node_i, node in enumerate(src.nodes):
      for next_node in node.nodes_next:
        adj_matrix[node_i,next_node] = -1
    # computing longest paths
    dist_from_start = scipy.sparse.csgraph.dijkstra(csgraph=adj_matrix,
                                                    indices=[0],
                                                    unweighted=True)

    embeddings = dy.select_cols(dy.parameter(self.embedder), [int(-d) for d in dist_from_start[0]])

    if self.op == 'sum':
      output = embeddings + src.as_tensor()
    elif self.op == 'concat':
      output = dy.concatenate([embeddings, src.as_tensor()])
    else:
      raise ValueError(f'Illegal op {op} in PositionalTransducer (options are "sum"/"concat")')
    output_seq = expression_seqs.ExpressionSequence(expr_tensor=output, mask=src.mask)
    self._final_states = [transducers.FinalTransducerState(output_seq[-1])]
    return output_seq


class MultiHeadAttentionLatticeTransducer(transducers.SeqTransducer, Serializable):
  """
  A lattice transducer for lattice inputs.

  """
  yaml_tag = '!MultiHeadAttentionLatticeTransducer'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               input_dim=Ref("exp_global.default_layer_dim"),
               param_init=Ref("exp_global.param_init", default=bare(param_initializers.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(param_initializers.ZeroInitializer)),
               num_heads=8):
    assert (input_dim % num_heads == 0)

    param_collection = param_collections.ParamManager.my_params(self)

    self.input_dim = input_dim
    self.num_heads = num_heads
    self.head_dim = input_dim // num_heads

    self.pWq, self.pWk, self.pWv, self.pWo = [
      param_collection.add_parameters(dim=(input_dim, input_dim), init=param_init.initializer((input_dim, input_dim)))
      for _ in range(4)]
    self.pbq, self.pbk, self.pbv, self.pbo = [
      param_collection.add_parameters(dim=(1, input_dim), init=bias_init.initializer((1, input_dim,))) for _ in
      range(4)]

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self.cur_src = src[0]
    self._final_states = None

  def get_final_states(self) -> List[transducers.FinalTransducerState]:
    return self._final_states

  @events.handle_xnmt_event
  def on_set_train(self, val):
    self.train = val

  def transduce(self, expr_seq: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:
    """
    transduce the sequence

    Args:
      expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    Returns:
      expression sequence
    """

    Wq, Wk, Wv, Wo = [dy.parameter(x) for x in (self.pWq, self.pWk, self.pWv, self.pWo)]
    bq, bk, bv, bo = [dy.parameter(x) for x in (self.pbq, self.pbk, self.pbv, self.pbo)]

    # Start with a [(length, model_size) x batch] tensor
    x = expr_seq.as_transposed_tensor()
    x_len = x.dim()[0][0]
    x_batch = x.dim()[1]
    # Get the query key and value vectors
    q = bq + x * Wq
    k = bk + x * Wk
    v = bv + x * Wv

    # Split to batches [(length, head_dim) x batch * num_heads] tensor
    q, k, v = [dy.reshape(x, (x_len, self.head_dim), batch_size=x_batch * self.num_heads) for x in (q, k, v)]

    # TODO: should we cache these?
    # compute conditionals as shortest paths over negative log probs
    # TODO: this gets prob of the most likely path, but we should really sum over all paths
    num_nodes = self.cur_src.sent_len()
    log_of_zero = -100.0
    adj_matrix = np.full((num_nodes, num_nodes), -log_of_zero)
    for node_i, node in enumerate(self.cur_src.nodes):
      for next_node in node.nodes_next:
        adj_matrix[node_i, next_node] = -self.cur_src.nodes[next_node].fwd_log_prob
    fwd_pairwise_cond = scipy.sparse.csgraph.bellman_ford(csgraph=adj_matrix)
    adj_matrix = np.full((num_nodes, num_nodes), -log_of_zero)
    for node_i, node in enumerate(self.cur_src.nodes):
      for prev_node in node.nodes_prev:
        adj_matrix[node_i, prev_node] = -self.cur_src.nodes[prev_node].bwd_log_prob
    bwd_pairwise_cond = scipy.sparse.csgraph.bellman_ford(csgraph=adj_matrix)
    pairwise_cond = -np.maximum(fwd_pairwise_cond, bwd_pairwise_cond)

    # Do scaled dot product [(length, length) x batch * num_heads], rows are queries, columns are keys
    attn_score = q * dy.transpose(k) / math.sqrt(self.head_dim)
    attn_score += dy.inputTensor(pairwise_cond) # TODO: rows/cols correct
    if expr_seq.mask is not None:
      mask = dy.inputTensor(np.repeat(expr_seq.mask.np_arr, self.num_heads, axis=0).transpose(), batched=True) * -1e10
      attn_score = attn_score + mask
    attn_prob = dy.softmax(attn_score, d=1)
    # Reduce using attention and resize to match [(length, model_size) x batch]
    o = dy.reshape(attn_prob * v, (x_len, self.input_dim), batch_size=x_batch)
    # Final transformation
    o = bo + o * Wo

    expr_seq = expression_seqs.ExpressionSequence(expr_transposed_tensor=o, mask=expr_seq.mask)

    self._final_states = [transducers.FinalTransducerState(expr_seq[-1], None)]

    return expr_seq
