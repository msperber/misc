import functools
import math
import numbers
from typing import List

import numpy as np
import scipy.sparse, scipy.sparse.csgraph
import dynet as dy

from xnmt import expression_seqs, events, param_collections, param_initializers, sent
from xnmt.transducers import base as transducers
from xnmt.persistence import bare, Ref, Serializable, serializable_init

class LatticePositionalSeqTransducer(transducers.SeqTransducer, Serializable):
  yaml_tag = '!LatticePositionalSeqTransducer'

  @serializable_init
  @events.register_xnmt_handler
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

  @events.handle_xnmt_event
  def on_start_sent(self, src):
    self.cur_src = src[0] # TODO: support minibatches

  def transduce(self, src: expression_seqs.ExpressionSequence) -> expression_seqs.ExpressionSequence:

    dist_from_start = self.longest_distances(self.cur_src)

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

  @functools.lru_cache(maxsize=None)
  def longest_distances(self, lattice):
    num_nodes = lattice.sent_len()
    adj_matrix = np.full((num_nodes, num_nodes), -np.inf)
    for node_i, node in enumerate(lattice.nodes):
      for next_node in node.nodes_next:
        adj_matrix[node_i, next_node] = -1
    # computing longest paths
    dist_from_start = scipy.sparse.csgraph.dijkstra(csgraph=adj_matrix,
                                                    indices=[0])
    return dist_from_start


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
    self.cur_src = src[0] # TODO: support minibatches
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

    pairwise_cond = self.compute_pairwise_log_conditionals(self.cur_src)

    # Do scaled dot product [(length, length) x batch * num_heads], rows are queries, columns are keys
    attn_score = q * dy.transpose(k) / math.sqrt(self.head_dim)
    attn_score += dy.inputTensor(pairwise_cond)
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

  @functools.lru_cache(maxsize=None)
  def compute_pairwise_log_conditionals(self, lattice: sent.Lattice, annotate=0) -> np.ndarray:
    """
    Compute pairwise log conditionals.

    For row i and column j, the result is log(Pr(j in path | i in path))

    Runs in O(|V|^3).

    Args:
      lattice: The input lattice.

    Returns:
      A numpy array
    """
    pairwise = []
    for node_i in range(lattice.sent_len()):
      pairwise.append(self.compute_log_conditionals_one(lattice, node_i))
    pairwise_fwd = np.asarray(pairwise)

    pairwise = []
    for node_i in range(lattice.sent_len())[::-1]:
      pairwise.append(list(reversed(self.compute_log_conditionals_one(lattice.reversed(), node_i))))
    pairwise_bwd = np.asarray(pairwise)

    ret = np.maximum(pairwise_fwd, pairwise_bwd)

    for node_i in range(lattice.sent_len()):
      lattice.nodes[node_i].cond_log_prob = ret[annotate][node_i]

    return ret


  def compute_log_conditionals_one(self, lattice: sent.Lattice, condition_on: numbers.Integral,
                                   log_zero: numbers.Integral = -100.0) -> List[numbers.Real]:
    """
    Compute conditional log probabilities for every node being visited after a given node has been visited.

    Note that this is directional: If V1 comes before V2 in a path, then the conditional will be zero.

    Runs in O(|V|+|E|) = O(|V|^2) for a lattice with nodes V and edges E

    Args:
      lattice: The lattice
      condition_on: index of node that must be traversed
      log_zero: negative number that approximates the log of zero (-inf cannot be passed to DyNet)

    Returns:
      List of log conditionals with same node ordering as for input lattice.
    """
    cond_log_probs = [log_zero] * lattice.sent_len()
    cond_log_probs[condition_on] = 0.0
    for node_i in range(lattice.sent_len()): # nodes are in topological order so we can simply loop in order
      node = lattice.nodes[node_i]
      for next_node in node.nodes_next:
        next_log_prob = lattice.nodes[next_node].fwd_log_prob
        next_cond_prob = math.exp(cond_log_probs[next_node]) + math.exp(next_log_prob) * math.exp(cond_log_probs[node_i])
        cond_log_probs[next_node] = math.log(next_cond_prob) if next_cond_prob>0.0 else log_zero
    return cond_log_probs
