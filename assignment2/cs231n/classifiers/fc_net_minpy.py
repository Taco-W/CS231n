"""
This file implements fullyconnected network in minpy.

All the array created in this file belongs to minpy.array Type.
Types of input values to loss() function, i.e. training/testing data & targets, should also be minpy.array.
"""

import numpy as np

from cs231n.layers_hack import *
from cs231n.layer_utils_hack import *

#call minpy package
configfile = '~/minpy/python/minpy'
sys.path.append(os.path.dirname(os.path.expanduser(configfile)))

import minpy
import minpy.core

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    # TODO: params should be created as minpy.array
    self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale 
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale 
    self.params['b2'] = np.zeros(num_classes)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    # Note: types of X, y are mxnet.ndarray

    def train_loss(X, y, W1, W2, b1, b2):
      l1, l1_cache = affine_relu_forward(X, W1, b1)
      l2, l2_cache = affine_forward(l1, W2, b2)
      scores = l2 

      # TODO: Deal with Test Mode 
      if y is None:
        return scores
    
      loss, d_scores = softmax_loss(scores, y)
      loss += np.sum(W1 ** 2) * 0.5 * self.reg
      loss += np.sum(W2 ** 2) * 0.5 * self.reg
      return loss

    grad_function = core.grad_and_loss(train_loss)

    # TODO: support input/output self.params as a whole?
    grads = {}
    loss, grads['W1'], grads['b1'], grads['W2'], grads['b2'] 
      = grad_function(X, y, 
                      self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2'])

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def GetWeightName(self, kth):
    return 'W' + str(kth)

  def GetBiasName(self, kth):
    return 'B' + str(kth)

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    for l in xrange(self.num_layers):
      if l == 0:
        input_d = input_dim
      else:
        input_d = hidden_dims[l-1]

      if l < self.num_layers - 1:
        out_d = hidden_dims[l]
      else:
        out_d = num_classes

      self.params[self.GetWeightName(l)] = np.random.randn(input_d, out_d) * weight_scale
      self.params[self.GetBiasName(l)] = np.zeros(out_d)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   

    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    last_layer_output = X
    cache = {}

    for l in xrange(self.num_layers):
      if l < (self.num_layers - 1):
        if (not self.use_batchnorm) and (not self.use_dropout):
          last_layer_output, cache[l] = affine_relu_forward(last_layer_output, 
            self.params[self.GetWeightName(l)], self.params[self.GetBiasName(l)]) 
        else:
          assert False
      else:
        last_layer_output, cache[l] = affine_forward(last_layer_output, 
          self.params[self.GetWeightName(l)], self.params[self.GetBiasName(l)])

    scores = last_layer_output 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    # If test mode return early
    if mode == 'test':
      return scores

    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    grads = {}
    loss, d_scores = softmax_loss(scores, y)
    delta = d_scores
    for l in xrange(self.num_layers - 1, -1, -1):
      if l == self.num_layers - 1:
        delta, grads[self.GetWeightName(l)], grads[self.GetBiasName(l)] = affine_backward(delta, cache[l])
      else:
        delta, grads[self.GetWeightName(l)], grads[self.GetBiasName(l)] = affine_relu_backward(delta, cache[l])

        loss += np.sum(self.params[self.GetWeightName(l)] **2) * 0.5 * self.reg 
        grads[self.GetWeightName(l)] += self.params[self.GetWeightName(l)] * self.reg


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
