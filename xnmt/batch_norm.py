import dynet as dy
import numpy as np

class ConvolutionalBatchNorm(object):

  bn_eps = 0.1
  bn_momentum = 0.1

  def __init__(self, model, num_filters):

    self.bn_gamma = model.add_parameters(dim=(1,1,num_filters, ), init=dy.ConstInitializer(1.0))
    self.bn_beta = model.add_parameters(dim=(1,1,num_filters, ), init=dy.ConstInitializer(0.0))
    self.bn_population_running_mean = np.zeros((num_filters, ))
    self.bn_population_running_std = np.ones((num_filters, ))
  
  def bn_expr(self, input_expr, train):
    param_bn_gamma = dy.parameter(self.bn_gamma)
    param_bn_beta = dy.parameter(self.bn_beta)
    if train:
      bn_mean = dy.moment_dim(input_expr, [0,1], 1, True) # mean over batches, time and freq dimensions
      neg_bn_mean_reshaped = -dy.reshape(-bn_mean, (1, 1, bn_mean.dim()[0][0]))
      self.bn_population_running_mean += -self.bn_momentum*self.bn_population_running_mean + self.bn_momentum * bn_mean.npvalue()
#          bn_std = dy.std_dim(cnn_layer, [0,1], True) # currently unusably slow, but would be less wasteful memory-wise
      bn_std = dy.sqrt(dy.moment_dim(dy.cadd(input_expr, neg_bn_mean_reshaped), [0,1], 2, True))
      self.bn_population_running_std += -self.bn_momentum*self.bn_population_running_std + self.bn_momentum * bn_std.npvalue()
    else:
      neg_bn_mean_reshaped = -dy.reshape(dy.inputVector(self.bn_population_running_mean), (1, 1, self.bn_population_running_mean.shape[0]))
      bn_std = dy.inputVector(self.bn_population_running_std)
    bn_numerator = dy.cadd(input_expr, neg_bn_mean_reshaped)
    bn_xhat = dy.cdiv(bn_numerator, dy.reshape(bn_std, (1, 1, bn_std.dim()[0][0])) + self.bn_eps)
    bn_y = dy.cadd(dy.cmult(param_bn_gamma, bn_xhat), param_bn_beta) # y = gamma * xhat + beta
    return bn_y
    