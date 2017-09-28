#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Given reference sentences, sample corrupted sentences by first drawing a targeted
distance from a Poisson distribution, then for each edit sampling either an insertion,
deletion, or substitution. By default, (#sub,#ins,#del) is sampled uniformly from
a simplex such that trg_dist = #sub + #ins + #del. Optionally, integer-valued weights
can be specified for the 3 operations, in which case the we sample from a
higher-dimensional simplex and multiple times for each operation depending on the weight.
E.g. for weights (3,1,2), we have trg_dist = #sub1 + #sub2 + #sub3 + #ins + #del1 + #del2.

If <out.txt> is given, the results are written into out.txt.tmp, and when that is finished
moved to out.txt

Usage:
  raml_sample.py [options] <tau> <in.txt> [<out.txt>]

Options:
    -c --char-tokens           assume inputs/outputs are char-tokenized (with __ as whitespace), but sampling is word-based
    -e --empty-line-repl s     replace empty lines by string s
    -o --op-weights s          integer-valued weights (>=0) for sub, ins, del operations (defaults to "1 1 1")
    -v --vocab f               set vocab (otherwise, inferred from in.txt)
    -w --weighted-vocab f      weighted vocab for unigram sampling (expects text lines: "<weight> <word>")
    -i --ignore-vocab f        ignore (don't delete or subsitute for) tokens specified in this vocab file
    (as always: -h for help)
"""


__author__      = "Matthias Sperber"
__date__        = "March 10, 2017"

import sys, os
import io
import docopt
import math
import numpy as np
from scipy.special import binom
from scipy.misc import logsumexp

class Usage(Exception):
  def __init__(self, msg):
    self.msg = msg
class ModuleTest(Exception):
  def __init__(self, msg):
    self.msg = msg


class Aligner:
  # gap penalty:
  gapPenalty = -1.0
  gapSymbol = None

  # similarity function:
  def sim(self, word1, word2):
    if word1==word2: return 0
    else: return -1

  # performs edit distance alignment between two lists
  # outputs c, x, y, s:
  # c = score
  # x = l1 with gapSymbol inserted to indicate insertions
  # y = l2 with gapSymbol inserted to indicate deletions
  # s = list of same length as x and y, specifying correct words, substitutions,
  #      deletions, insertions as 'c', 's', 'd', 'i'
  def align(self, l1, l2):
    # compute matrix
    F = [[0] * (len(l2)+1) for i in xrange((len(l1)+1))]
    for i in range(len(l1)+1):
      F[i][0] = i * self.gapPenalty
    for j in range(len(l2)+1):
      F[0][j] = j * self.gapPenalty
    for i in range(0, len(l1)):
      for j in range(0, len(l2)):
        match = F[i][j] + self.sim(l1[i], l2[j])
        delete = F[i][j+1] + self.gapPenalty
        insert = F[i+1][j] + self.gapPenalty
        F[i+1][j+1] = max(match, delete, insert)
    c = F[len(l1)][len(l2)]
    x = []
    y = []
    i = len(l1)-1
    j = len(l2)-1
    while i>=0 and j>=0:
      score = F[i+1][j+1]
      scoreDiag = F[i][j]
      scoreUp = F[i+1][j]
      scoreLeft = F[i][j+1]
      if score == scoreLeft + self.gapPenalty:
        x = [l1[i]] + x
        y = [self.gapSymbol] + y
        i -= 1
      elif score == scoreUp + self.gapPenalty:
        x = [self.gapSymbol] + x
        y = [l2[j]] + y
        j -= 1
      else:
        assert score == scoreDiag + self.sim(l1[i], l2[j])
        x = [l1[i]] + x
        y = [l2[j]] + y
        i -= 1
        j -= 1
    while i>=0:
      x = [l1[i]] + x
      y = [self.gapSymbol] + y
      i -= 1
    while j>=0:
      x = [self.gapSymbol] + x
      y = [l2[j]] + y
      j -= 1
    s = []
    assert len(x) == len(y)
    for i in range(len(x)):
      if x[i] is self.gapSymbol and y[i] is not self.gapSymbol:
        s.append('i')
      elif x[i] is not self.gapSymbol and y[i] is self.gapSymbol:
        s.append('d')
      elif self.sim(x[i], y[i]) >= 0:
        s.append('c')
      else:
        s.append('s')
    return c, x, y, s


def sample_corrupted(words, tau, vocab, vocab_weights=None, op_weights=(1,1,1), ignoreVocab=[], use_word_sim=False, prefer_shorter=False):
  sent_len = len([w for w in words if not w in ignoreVocab])
  distance = sample_edit_distance(tau, sent_len)
  sub_del_candidate_positions = [i for i in range(len(words)) if not words[i] in ignoreVocab]
  num_sub, num_ins, num_del = sample_num_operations(distance, op_weights)
  sub_positions = sample_sub_positions(words, sub_del_candidate_positions, num_sub, prefer_shorter)
  del_positions = sample_sub_positions(words, [p for p in sub_del_candidate_positions if not p in sub_positions], num_del, prefer_shorter)
  ins_positions = sample_ins_positions(range(sent_len+1), num_ins)
  ret_words = \
      corrupt_positions(words, vocab, sub_positions, ins_positions, del_positions, 
                        vocab_weights=vocab_weights, use_word_sim=use_word_sim)
  return ret_words, num_sub, num_ins, num_del

def sample_edit_distance(tau, sent_len):
  lam = tau * sent_len
  sampled = None
  while sampled is None or sampled > sent_len: # TODO: would be nicer to compute truncated distribution explicitly, but this will do for now
    sampled = np.random.poisson(lam=lam)
  return sampled

def sample_num_operations(distance, op_weights):
  num_buckets = sum(op_weights)
  sorted_samples = sorted(np.random.choice(range(1,distance+num_buckets), size=num_buckets-1, replace=False))
  sorted_samples = [0] + sorted_samples + [distance+num_buckets]
  num_sub = sum([sorted_samples[i+1] - sorted_samples[i] - 1 for i in range(0,op_weights[0])])
  num_ins = sum([sorted_samples[i+1] - sorted_samples[i] - 1 for i in range(op_weights[0],op_weights[0]+op_weights[1])])
  num_del = sum([sorted_samples[i+1] - sorted_samples[i] - 1 for i in range(op_weights[0]+op_weights[1],op_weights[0]+op_weights[1]+op_weights[2])])
  assert distance == num_sub + num_ins + num_del
  assert min(num_sub, num_ins, num_del) >= 0
  return num_sub, num_ins, num_del

def sample_sub_positions(words, pos_choices, num_sub, prefer_shorter=False):
  if num_sub==0: return []
  p = None
  if prefer_shorter:
    p = np.exp(-1.0 * np.asarray([len(words[pos]) for pos in pos_choices])) / np.sum(np.exp(-1.0 * np.asarray([len(words[pos]) for pos in pos_choices])))
  sub_positions = np.random.choice(pos_choices, size=num_sub, replace=False, p=p)
  return sub_positions

def sample_ins_positions(pos_choices, num_ins):
  if num_ins==0: return []
  ins_positions = np.random.choice(a=pos_choices, size=num_ins, replace=True)
  return ins_positions

def corrupt_positions(words, vocab, sub_positions, ins_positions, del_positions,
                    vocab_weights=None, use_word_sim=False):
  if use_word_sim: assert vocab_weights is None
    
  ret_words = list(words)
  for pos in sub_positions:
    word = words[pos]
    if use_word_sim:
      vocab_weights = get_similarities(word, vocab)
    while(word==words[pos]):
      word = np.random.choice(vocab, p=vocab_weights)
    ret_words[pos] = word
  for pos in del_positions:
    ret_words[pos] = None
  for pos in reversed(sorted(ins_positions)):
    if use_word_sim:
      vocab_weights = get_similarities("", vocab)
    word = np.random.choice(vocab, p=vocab_weights)
    ret_words.insert(pos, word)
  ret_words = [w for w in ret_words if w is not None]
  return ret_words

aligner = Aligner()
cache = {}
def get_similarities(word, vocab):
  if word in cache:
    return cache[word]
  else:
    dist = [aligner.align(word, w2)[0] for w2 in vocab]
    similarities=np.exp(dist)/sum(np.exp(dist))
    if len(cache) < 1000:
      cache[word] = similarities
    return similarities

def main(argv=None):
  arguments = docopt.docopt(__doc__, options_first=True, argv=argv)
  print arguments
    
  tau = float(arguments["<tau>"])
  inputFileName = arguments["<in.txt>"]
  outputFileName = arguments["<out.txt>"]
  if outputFileName:
    outF = io.open(outputFileName + ".tmp", "w")
    
  emptyLineRepl = arguments["--empty-line-repl"]
  if emptyLineRepl is None: emptyLineRepl = ""
    
  op_weights = arguments["--op-weights"]
  if op_weights is None: op_weights = (1,1,1)
  else: op_weights = tuple([int(w) for w in op_weights.split()])
    
  vocabFileName = arguments["--vocab"]
  weightedVocabFileName = arguments["--weighted-vocab"]

  ignoreVocabFileName = arguments["--ignore-vocab"]
   
  assumeCharTokens = arguments["--char-tokens"]
    
  vocab = []
  vocab_weights = None
  if vocabFileName:
    for v in io.open(vocabFileName).readlines():
      vocab.append(v.strip())
  elif weightedVocabFileName:
    vocab_weights = []
    for line in io.open(weightedVocabFileName).readlines():
      vocab.append(line.split()[1])
      vocab_weights.append(float(line.split()[0]))
    weightSum = sum(vocab_weights)
    for i in range(len(vocab_weights)):
      vocab_weights[i] /= weightSum
  else:
    vocabSet = set()
    for line in io.open(inputFileName):
      if assumeCharTokens:
        line = "".join([c if c!=u"__" else u" " for c in line.split()])
      for v in line.strip().split():
        vocabSet.add(v)
    vocab = list(vocabSet)
  ignoreVocab = []
  if ignoreVocabFileName:
    for v in io.open(ignoreVocabFileName).readlines():
      ignoreVocab.append(v.strip())


  ###########################
  ## MAIN PROGRAM ###########
  ###########################

  total_sub, total_ins, total_del, total_ref_len = 0, 0, 0, 0
  for line in io.open(inputFileName):
    if assumeCharTokens:
      line = u"".join([c if c!=u"__" else u" " for c in line.split()])
    words = line.strip().split()
    ret_words, num_sub, num_ins, num_del = \
                    sample_corrupted(words=words, 
                                     tau=tau, vocab=vocab,
                                     vocab_weights=vocab_weights,
                                     op_weights=op_weights,
                                     ignoreVocab=ignoreVocab)
    total_sub += num_sub
    total_ins += num_ins
    total_del += num_del
    total_ref_len += len(words)
    outLine = u" ".join(ret_words)
        
    if assumeCharTokens:
      outLine = u" ".join([s if s!=u" " else u"__" for s in list(outLine)])
       
    if len(ret_words)==0:
      outLine = emptyLineRepl

    if outputFileName is not None:
      outF.write((outLine + u"\n"))
    else:
      print outLine
  
  print >> sys.stderr, "=========="
  print >> sys.stderr, "average edit rate:", \
          float(total_sub+total_ins+total_del)/float(total_ref_len)*100.0, \
          "% (S:", total_sub, ", I:", total_ins, ", D:", total_del, ")"
  if outputFileName is not None:
    outF.close()
    os.rename(outputFileName + ".tmp", outputFileName)


  ###########################
  ###########################

if __name__ == "__main__":
  sys.exit(main())
    
