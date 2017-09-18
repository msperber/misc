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
    (as always: -h for help)
"""


__author__      = "Matthias Sperber"
__date__        = "March 10, 2017"

import sys, os
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

def sample_corrupted(words, tau, vocab, vocabWeights=None, op_weights=(1,1,1)):
    sent_len = len(words)
    distance = sample_edit_distance(tau, sent_len)
    num_sub, num_ins, num_del = sample_num_operations(distance, op_weights)
    sub_positions = sample_sub_positions(range(sent_len), num_sub)
    del_positions = sample_sub_positions([p for p in range(sent_len) if not p in sub_positions], num_del)
    ins_positions = sample_ins_positions(range(sent_len+1), num_ins)
    ret_words = \
        corrupt_positions(words, vocab, sub_positions, ins_positions, del_positions, 
                          vocabWeights=vocabWeights)
    return ret_words, num_sub, num_ins, num_del

def sample_edit_distance(tau, sent_len):
    lam = tau * sent_len
    sampled = None
    #return min(np.random.poisson(lam=lam), sent_len)
    while sampled is None or sampled > sent_len:
      sampled = np.random.poisson(lam=lam)
    return sampled

def sample_num_operations(distance, op_weights):
    num_buckets = sum(op_weights)
    num_sub, num_ins, num_del = 0,0,0
    # due to rounding issues, we might need several attempts:
    while distance != num_sub + num_ins + num_del:
        # we do the *multiplicator thing to handle cases where distance < sum(op_weights)
        sorted_samples = sorted(np.random.choice(range(distance*num_buckets), size=num_buckets-1, replace=False))
        sorted_samples = [0] + sorted_samples + [distance*num_buckets]
        num_sub = sum([sorted_samples[i+1] - sorted_samples[i] for i in range(0,op_weights[0])])
        num_ins = sum([sorted_samples[i+1] - sorted_samples[i] for i in range(op_weights[0],op_weights[0]+op_weights[1])])
        num_del = sum([sorted_samples[i+1] - sorted_samples[i] for i in range(op_weights[0]+op_weights[1],op_weights[0]+op_weights[1]+op_weights[2])])
        num_sub = int(round(num_sub/float(num_buckets)))
        num_ins = int(round(num_ins/float(num_buckets)))
        num_del = int(round(num_del/float(num_buckets)))
    assert distance == num_sub + num_ins + num_del
    assert min(num_sub, num_ins, num_del) >= 0
    return num_sub, num_ins, num_del

def sample_sub_positions(pos_choices, num_sub):
    if num_sub==0: return []
    sub_positions = np.random.choice(pos_choices, size=num_sub, replace=False)
    return sub_positions

def sample_ins_positions(pos_choices, num_ins):
    if num_ins==0: return []
    ins_positions = np.random.choice(a=pos_choices, size=num_ins, replace=True)
    return ins_positions

def corrupt_positions(words, vocab, sub_positions, ins_positions, del_positions,
                      vocabWeights=None):
    ret_words = list(words)
    for pos in sub_positions:
        word = words[pos]
        while(word==words[pos]):
            word = np.random.choice(vocab, p=vocabWeights)
        ret_words[pos] = word
    for pos in del_positions:
        ret_words[pos] = None
    for pos in reversed(sorted(ins_positions)):
        word = np.random.choice(vocab)
        ret_words.insert(pos, word)
    ret_words = [w for w in ret_words if w is not None]
    return ret_words

def main(argv=None):
    arguments = docopt.docopt(__doc__, options_first=True, argv=argv)
    print arguments
    
    tau = float(arguments["<tau>"])
    inputFileName = arguments["<in.txt>"]
    outputFileName = arguments["<out.txt>"]
    if outputFileName:
        outF = open(outputFileName + ".tmp", "w")
    
    emptyLineRepl = arguments["--empty-line-repl"]
    if emptyLineRepl is None: emptyLineRepl = ""
    
    op_weights = arguments["--op-weights"]
    if op_weights is None: op_weights = (1,1,1)
    else: op_weights = tuple([int(w) for w in op_weights.split()])
    
    vocabFileName = arguments["--vocab"]
    weightedVocabFileName = arguments["--weighted-vocab"]
#    delRate = arguments['--del-rate']
#    if delRate is not None: delRate = float(delRate)
#    insBoost = arguments["--ins-boost"]
#    if insBoost is not None: insBoost = float(insBoost)
    
    assumeCharTokens = arguments["--char-tokens"]
    
    vocab = []
    vocabWeights = None
    if vocabFileName:
        for v in open(vocabFileName).readlines():
            vocab.append(v.decode("utf-8").strip())
    elif weightedVocabFileName:
        vocabWeights = []
        for line in open(weightedVocabFileName).readlines():
            vocab.append(line.decode("utf-8").split()[1])
            vocabWeights.append(float(line.decode("utf-8").split()[0]))
        weightSum = sum(vocabWeights)
        for i in range(len(vocabWeights)):
            vocabWeights[i] /= weightSum
    else:
        vocabSet = set()
        for line in open(inputFileName):
            if assumeCharTokens:
                line = "".join([c if c!="__" else " " for c in line.split()])
            for v in line.decode("utf-8").strip().split():
                vocabSet.add(v)
        vocab = list(vocabSet)


    ###########################
    ## MAIN PROGRAM ###########
    ###########################

    total_sub, total_ins, total_del, total_ref_len = 0, 0, 0, 0
    for line in open(inputFileName):
        line = line.decode("utf-8")
        if assumeCharTokens:
            line = "".join([c if c!="__" else " " for c in line.split()])
        words = line.strip().split()
        ret_words, num_sub, num_ins, num_del = \
                        sample_corrupted(
                                        words=words, 
                                        tau=tau, vocab=vocab,
                                        vocabWeights=vocabWeights,
                                        op_weights=op_weights)
        total_sub += num_sub
        total_ins += num_ins
        total_del += num_del
        total_ref_len += len(words)
        outLine = u" ".join(ret_words)
        
        if assumeCharTokens:
            outLine = " ".join([s if s!=" " else "__" for s in list(outLine)])
        
        if len(ret_words)==0:
            outLine = emptyLineRepl

        if outputFileName is not None:
            outF.write((outLine + "\n").encode("utf-8"))
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
    
