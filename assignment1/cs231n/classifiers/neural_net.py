import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = np.sqrt(2.0/input_size) * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)

    print 'input_size', np.sqrt(2.0/input_size)
    print 'hidden_size', np.sqrt(1.0/hidden_size)
    self.params['W2'] = np.sqrt(1.0/hidden_size) * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    WList = [W1, W2]
    data_num, D = X.shape

    # Compute the forward pass
    scores = None

    fc1 = np.dot(X, W1) + b1
    fc1_relu = np.maximum(fc1, 0)
    fc2 = np.dot(fc1_relu, W2) + b2 
    scores = fc2

    if y is None:
      return scores

    # Issue: np.amax. 
    #pred_exp = np.exp(fc2 - np.expand_dims(np.amax(fc2, axis = 1), axis = 1))
    pred_exp = np.exp(fc2)

    pred_conf = pred_exp / np.expand_dims(np.sum(pred_exp, axis = 1), axis = 1)

    loss = np.sum( - np.log( pred_conf[np.arange(data_num), y] ) ) / data_num
    loss_with_reg = loss +  0.5 * reg * np.sum(W1**2) + 0.5 * reg * np.sum(W2**2)
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    delta_fc2 = pred_conf # [data_num, target_dim]
    delta_fc2[np.arange(data_num), y] -= 1
    grads['b2'] = np.sum(delta_fc2, axis = 0) / data_num
    grads['W2'] = np.dot(np.transpose(fc1_relu), delta_fc2) / data_num
    delta_fc1_relu = np.dot(delta_fc2, np.transpose(W2))
    delta_fc1 = delta_fc1_relu
    delta_fc1[fc1 <= 0] = 0
    grads['b1'] = np.sum(delta_fc1, axis = 0) / data_num
    grads['W1'] = np.dot(np.transpose(X), delta_fc1) / data_num

    # add reg loss
    grads['W2'] += W2 * reg
    grads['W1'] += W1 * reg
    
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss_with_reg, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    batch_num = num_train / batch_size
    print 'batch_num', batch_size
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      if(it % batch_num == 0):
        batch_indices = np.random.choice(batch_num, batch_num, replace = False)

      batch_k = batch_indices[it % batch_num]
      X_batch = X[batch_k*batch_size:(batch_k+1)*batch_size] 
      y_batch = y[batch_k*batch_size:(batch_k+1)*batch_size]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################

      self.params['W1'] -= grads['W1'] * learning_rate 
      self.params['b1'] -= grads['b1'] * learning_rate 
      self.params['W2'] -= grads['W2'] * learning_rate 
      self.params['b2'] -= grads['b2'] * learning_rate 

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    data_num, D = X.shape

    fc1 = np.dot(X, W1) + b1
    fc1_relu = np.maximum(fc1, 0)
    fc2 = np.dot(fc1_relu, W2) + b2 
    y_pred = np.argmax(fc2, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


