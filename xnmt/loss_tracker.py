from __future__ import division, generators

import sys
import math
import time

class LossTracker:
    '''
    A template class to generate report for training.
    '''

    REPORT_TEMPLATE = 'Epoch %.4f: {}_ppl=%.4f (words=%d, words/sec=%.2f, time=%s)'

    def __init__(self, eval_every, total_train_sent):
        self.eval_every = eval_every
        self.total_train_sent = total_train_sent

        self.epoch_num = 0

        self.epoch_loss = 0.0
        self.epoch_words = 0
        self.sent_num = 0
        self.sent_num_not_report = 0
        self.fractional_epoch = 0

        self.dev_loss = 0.0
        self.best_dev_loss = sys.float_info.max
        self.dev_words = 0

        self.start_time = time.time()

    def new_epoch(self):
        self.epoch_loss = 0.0
        self.epoch_words = 0
        self.epoch_num += 1
        self.sent_num = 0
        self.sent_num_not_report = 0

    def update_epoch_loss(self, src, tgt, loss):
        batch_sent_num = self.count_sent_num(src)
        self.sent_num += batch_sent_num
        self.sent_num_not_report += batch_sent_num
        self.epoch_words += self.count_tgt_words(tgt)
        self.epoch_loss += loss
    
    def format_time(self, seconds):
        return "{}-{}".format(int(seconds) // 86400, 
                              time.strftime("%H:%M:%S", time.gmtime(seconds)),
                             )

    def report_train_process(self):
        print_report = (self.sent_num_not_report >= self.eval_every) or (self.sent_num == self.total_train_sent)
        if print_report:
            while self.sent_num_not_report >= self.eval_every:
                self.sent_num_not_report -= self.eval_every
            self.fractional_epoch = (self.epoch_num - 1) + self.sent_num / self.total_train_sent
            elapsed_time = time.time() - self.start_time
            print(LossTracker.REPORT_TEMPLATE.format('train') % (
                self.fractional_epoch, math.exp(self.epoch_loss / self.epoch_words),
                self.epoch_words, self.epoch_words / elapsed_time, self.format_time(elapsed_time)))
        return print_report

    def new_dev(self):
        self.dev_loss = 0.0
        self.dev_words = 0

    def update_dev_loss(self, tgt, loss):
        self.dev_loss += loss
        self.dev_words += self.count_tgt_words(tgt)

    def report_dev_and_check_model(self, model_file):
        elapsed_time = time.time() - self.start_time
        print(LossTracker.REPORT_TEMPLATE.format('test') % (
            self.fractional_epoch, math.exp(self.dev_loss / self.dev_words),
            self.dev_words, self.epoch_words / elapsed_time, self.format_time(elapsed_time)))
        save_model = self.dev_loss < self.best_dev_loss
        if save_model:
            self.best_dev_loss = self.dev_loss
            print('Epoch %.4f: best dev loss, writing model to %s' % (self.fractional_epoch, model_file))
        return save_model

    def count_tgt_words(self, tgt_words):
        raise NotImplementedError('count_tgt_words must be implemented in LossTracker subclasses')

    def count_sent_num(self, obj):
        raise NotImplementedError('count_tgt_words must be implemented in LossTracker subclasses')

    def clear_counters(self):
        self.sent_num = 0
        self.sent_num_not_report = 0

    def report_ppl(self):
        pass


class BatchLossTracker(LossTracker):

    def count_tgt_words(self, tgt_words):
        return sum(len(x) for x in tgt_words)

    def count_sent_num(self, obj):
        return len(obj)


class NonBatchLossTracker(LossTracker):

    def count_tgt_words(self, tgt_words):
        return len(tgt_words)

    def count_sent_num(self, obj):
        return 1
