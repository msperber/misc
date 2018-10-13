import numbers
from typing import List

import numpy as np
import scipy.sparse, scipy.sparse.csgraph
import dynet as dy

from xnmt import expression_seqs, param_collections, param_initializers
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

    adj_matrix = np.full((num_nodes, num_nodes), np.inf)
    for node_i, node in enumerate(src.nodes):
      for next_node in node.nodes_next:
        adj_matrix[node_i,next_node] = 1
    dist_from_start = scipy.sparse.csgraph.dijkstra(csgraph=adj_matrix,
                                                    indices=[0],
                                                    unweighted=True)

    embeddings = dy.select_cols(dy.parameter(self.embedder), [int(d) for d in dist_from_start[0]])

    if self.op == 'sum':
      output = embeddings + src.as_tensor()
    elif self.op == 'concat':
      output = dy.concatenate([embeddings, src.as_tensor()])
    else:
      raise ValueError(f'Illegal op {op} in PositionalTransducer (options are "sum"/"concat")')
    output_seq = expression_seqs.ExpressionSequence(expr_tensor=output, mask=src.mask)
    self._final_states = [transducers.FinalTransducerState(output_seq[-1])]
    return output_seq
