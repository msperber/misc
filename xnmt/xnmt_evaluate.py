import argparse
import sys
from evaluator import BLEUEvaluator, WEREvaluator, CEREvaluator
from options import Option, OptionParser

options = [
    Option("ref_file", help="path of the reference file"),
    Option("hyp_file", help="path of the hypothesis target file"),
    Option("evaluator", default_value="bleu", help="Evaluation metrics (bleu/wer/cer)")
]



def read_data(loc_):
    """Reads the lines in the file specified in loc_ and return the list after inserting the tokens
    """
    data = list()
    with open(loc_) as fp:
        for line in fp:
            t = line.split()
            data.append(t)
    return data


def xnmt_evaluate(args):
    """"Returns the eval score (e.g. BLEU) of the hyp sentences using reference target sentences
    """

    if args.evaluator == "bleu":
        evaluator = BLEUEvaluator(ngram=4)
    elif args.evaluator == "wer":
        evaluator = WEREvaluator()
    elif args.evaluator == "cer":
        evaluator = CEREvaluator()
    else:
        raise RuntimeError("Unkonwn evaluation metric {}".format(args.evaluator))

    ref_corpus = read_data(args.ref_file)
    hyp_corpus = read_data(args.hyp_file)

    eval_score = evaluator.evaluate(ref_corpus, hyp_corpus)

    return eval_score


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_task("evaluate", options)
    args = parser.args_from_command_line("evaluate", sys.argv[1:])

    score = xnmt_evaluate(args)
    print("{} Score = {}".format(args.evaluator, score))
