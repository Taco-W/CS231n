import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dClass = np.zeros((1, num_classes))
  loss = 0.0

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    dClass[0, y[i]] = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dClass[0, j] = 1
        dClass[0, y[i]] -= 1
      else:
        dClass[0, j] = 0
    dW += np.expand_dims(X[i].transpose(), axis=1).dot(dClass)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dClass = np.zeros(num_classes)
  loss = 0.0
  scores = X.dot(W)  # D_N * D_Ans 

  correct_class_scores = scores[np.asarray(range(num_train)), y] # D_N

  margin = scores - np.expand_dims(correct_class_scores, axis = 1) + 1
  margin[np.asarray(range(num_train)), y] = 0

  margin_gt0 = np.maximum(margin, 0)

  loss = np.sum(margin_gt0)/ num_train
  loss += 0.5 * reg * np.sum(W * W)

  d_margin_gt0 = np.ones(margin_gt0.shape) / num_train
  d_margin = d_margin_gt0
  d_margin[margin < 0] = 0
  d_margin[np.asarray(range(num_train)), y] = 0

  d_correct_class_scores = - np.sum(d_margin, axis = 1) 
  d_scores = d_margin;
  d_scores[np.asarray(range(num_train)), y] += d_correct_class_scores
  dW += X.transpose().dot(d_scores)

  dW += reg * W
  '''
  greater_than_zero_pos_x,  greater_than_zero_pos_y = np.where(margin > 0)

  print 'greater_than_zero_pos_x.shape:\n', greater_than_zero_pos_x.shape
  print 'greater_than_zero_pos_y.shape:\n', greater_than_zero_pos_y.shape

  count_num = np.bincount(greater_than_zero_pos_y)
  print 'count shape: ', count_num.shape
  dClass[0:count_num.shape[0]:1] = count_num

  count_gd_num = np.bincount(y)
  print 'count gd shape: ', count_gd_num.shape
  dGdClass = np.zeros(num_classes)
  dGdClass[0:count_gd_num.shape[0]:1] = count_gd_num
  dClass -= dGdClass

  dW += X[i].transpose().dot(dClass)

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    dClass[0, y[i]] = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dClass[0, j] = 1
        dClass[0, y[i]] -= 1
      else:
        dClass[0, j] = 0
    dW += np.expand_dims(X[i].transpose(), axis=1).dot(dClass)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  '''

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
