from __future__ import division, generators

import dynet as dy
from batcher import *
from search_strategy import *
from vocab import Vocab
from serializer import Serializable, DependentInitParam
from train_test_interface import TrainTestInterface

##### A class for retrieval databases

# This file contains databases used for retrieval.
# At the moment it includes only a standard database that keeps all of the things
# to be retrieved in a list.

class StandardRetrievalDatabase(Serializable):
  """This is a database to be used for retrieval. Its database member"""

  yaml_tag = u"!StandardRetrievalDatabase"

  def __init__(self, reader, database_file):
    self.reader = reader
    self.database_file = database_file
    self.database = reader.read_sents(database_file)

##### The actual retriever class

class Retriever(TrainTestInterface):
  '''
  A template class implementing a retrieval model.
  '''

  def calc_loss(self, src, db_idx):
    '''Calculate loss based on a database index.

    :param src: The source input.
    :param db_idx: The correct index in the database to be retrieved.
    :returns: An expression representing the loss.
    '''
    raise NotImplementedError('calc_loss must be implemented for Retriever subclasses')

  def index_database(self):
    '''A function that can be called before actually performing retrieval.

    This will perform any necessary pre-processing to make retrieval more efficient.
    If the model is updated, assume that the indexing result is stale and no longer applicable.
    '''
    pass
    

  def retrieve(self, src):
    '''Perform retrieval, trying to get the sentence that most closely matches in the database.

    :param src: The source.
    :returns: The ID of the example that most closely matches in the database.
    '''
    raise NotImplementedError('retrieve must be implemented for Retriever subclasses')

  def set_train(self, val):
    for component in self.get_train_test_components():
      Retriever.set_train_recursive(component, val)
  @staticmethod
  def set_train_recursive(component, val):
    component.set_train(val)
    for sub_component in component.get_train_test_components():
      Retriever.set_train_recursive(sub_component, val)


class DotProductRetriever(Retriever, Serializable):
  '''
  A retriever trains using max-margin methods.
  '''
  
  yaml_tag = u'!DotProductRetriever'


  def __init__(self, src_embedder, src_encoder, trg_embedder, trg_encoder, database):
    '''Constructor.

    :param src_embedder: A word embedder for the source language
    :param src_encoder: An encoder for the source language
    :param trg_embedder: A word embedder for the target language
    :param trg_encoder: An encoder for the target language
    :param database: A database of things to retrieve
    '''
    self.src_embedder = src_embedder
    self.src_encoder = src_encoder
    self.trg_embedder = trg_embedder
    self.trg_encoder = trg_encoder
    self.database = database
  
  def get_train_test_components(self):
    return [self.src_encoder, self.trg_encoder]

  def calc_loss(self, src, db_idx):
    # raise NotImplementedError("calc_loss needs to calculate the max-margin objective")
    return dy.scalarInput(0)

  def index_database(self):
    # raise NotImplementedError("index_database needs to calculate the vectors for all the elements in the database and find the closest")
    pass

  def retrieve(self, src):
    # raise NotImplementedError("retrieve needs find the example index with the largest dot product")
    return 0

