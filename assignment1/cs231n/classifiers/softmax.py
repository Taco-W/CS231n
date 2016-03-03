import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  for i in range(X.shape[0]):
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    pred = np.dot(X[i, :], W)
    tmpX = np.exp(pred - pred.max())
    tmpY = tmpX / np.sum(tmpX)
    loss += -np.log(tmpY[y[i]])
    
    del_pred = tmpY
    del_pred[y[i]] -= 1

    dW += np.dot(np.expand_dims(X[i, :], axis = 1), np.expand_dims(del_pred, axis = 0))
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= X.shape[0]
  dW /= X.shape[0]

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  pred = np.dot(X, W)
  exp = np.exp(pred)
  prob = exp / np.expand_dims(np.sum(exp, axis = 1), axis=1)
  loss += -np.log(prob[np.arange(X.shape[0]), y]).sum() / X.shape[0]
  loss +=  0.5 * reg * np.sum(W * W)

  d_pred = prob
  d_pred[np.arange(X.shape[0]), y] -= 1
  
  dW += np.dot(np.transpose(X), d_pred) / X.shape[0]

  dW += reg * W

  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

