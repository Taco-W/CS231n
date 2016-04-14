import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  xshape = x.shape
  x = x.reshape(xshape[0], -1)
  out = x.dot(w) + b
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  xshape = x.shape
  x = x.reshape(xshape[0],-1)
  dx = dout.dot(w.T)
  dx.shape = xshape
  dw = x.T.dot(dout)
  db = dout.sum(axis = 0)
  return dx, dw, db


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  xshape = x.shape
  x = x.reshape(xshape[0],-1)
  dx = dout.dot(w.T)
  dx.shape = xshape
  dw = x.T.dot(dout)
  db = dout.sum(axis = 0)
  return dx, dw, db



def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)
  # TODO: Wx, Wh, b only stored once
  cache = (x, prev_h, Wx, Wh, b, next_h)
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  # flatten cache
  x, prev_h, Wx, Wh, b, next_h = cache
  # bp tanh
  dnext_h = dnext_h * (1 - next_h * next_h)
  # this is db
  db = dnext_h.sum(axis = 0)
  # back to dx
  dx = np.dot(dnext_h, Wx.T)
  # back to prev_h
  dprev_h = np.dot(dnext_h, Wh.T)
  # grad of Wx
  dWx = x.T.dot(dnext_h)
  # grad of Wh
  dWh = prev_h.T.dot(dnext_h)
  return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  # get shapes
  N,T,D = x.shape
  _,H = h0.shape
  # init result
  h = np.zeros([N,0,H])
  cache = []
  for t in range(T):
    if t == 0:
      h_prev = h0
    else:
      h_prev = h[:,t-1,:].reshape(N,H)
    x_t = x[:,t,:].reshape(N,D)
    h_t, cache_t = rnn_step_forward(x_t, h_prev, Wx, Wh, b) 
    h = np.append(h, h_t.reshape([N,1,H]), axis = 1)
    cache.append(cache_t)
  return h, cache

def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  # Get shapes
  (x, prev_h, Wx, Wh, b, next_h) = cache[0]
  T = len(cache)
  N,D = x.shape
  _,H = next_h.shape
  # Start loop
  dprev_h_t = np.zeros([N,H])
  dx = np.zeros([N,T,D])
  for t in reversed(range(T)):
    dnext_h_t = dprev_h_t + dh[:,t,:].reshape(N,H)
    dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h_t, cache[t])
    dx[:,t:t+1,:] = dx_t.reshape([N,1,D])
    if t == T - 1:
      dWx = dWx_t
      dWh = dWh_t
      db = db_t
    else:
      dWx += dWx_t
      dWh += dWh_t
      db += db_t
  dh0 = dprev_h_t
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  # get shape
  N,T = x.shape
  V,D = W.shape
  x_sparse = np.zeros([N, T,V])
  for i in range(N):
    for j in range(T):
      x_sparse[i,j,x[i,j]] = 1
  out = x_sparse.dot(W)
  cache = (x_sparse)
  return out, cache

def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  N,T,D = dout.shape
  _,_,V = cache.shape
  dW = cache.reshape(N*T, V).T.dot(dout.reshape(N*T,D)) 
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  N, D = np.shape(x)
  _, H = np.shape(prev_h)
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  # split a into 4 parts
  a_i = a[:, 0:H]
  a_f = a[:, H:2*H]
  a_o = a[:, 2*H:3*H]
  a_g = a[:, 3*H:4*H]
  # gate function
  i = sigmoid(a_i)
  f = sigmoid(a_f)
  o = sigmoid(a_o)
  g = np.tanh(a_g)
  next_c = f * prev_c + i * g
  next_h = o * np.tanh(next_c)
  cache = (x, prev_h, prev_c, Wx, Wh, b, next_h, next_c, i, f, o, g, a_i, a_f, a_o, a_g)
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None

  x, prev_h, prev_c, Wx, Wh, b, next_h, next_c, i, f, o, g, a_i, a_f, a_o, a_g = cache
  
  # dnext will also be produced by h
  dnext_c = dnext_c + dnext_h * o * (1 - np.tanh(next_c) * np.tanh(next_c))
  
  d_o = dnext_h * np.tanh(next_c)
  d_f = dnext_c * prev_c
  dprev_c = dnext_c * f
  d_i = dnext_c * g
  d_g = dnext_c * i
  d_ai = d_i * i * (1 - i)
  d_af = d_f * f * (1 - f)
  d_ao = d_o * o * (1 - o)
  d_ag = d_g * (1 - g * g)
  # concatenate d_a_sub
  d_a = np.append(d_ai, d_af, axis = 1)
  d_a = np.append(d_a, d_ao, axis = 1)
  d_a = np.append(d_a, d_ag, axis = 1)

  # db 
  db = d_a.sum(axis = 0)
  # back to dx
  dx = np.dot(d_a, Wx.T)
  # back to prev_h
  dprev_h = np.dot(d_a, Wh.T)
  # grad of Wx
  dWx = x.T.dot(d_a)
  # grad of Wh
  dWh = prev_h.T.dot(d_a)

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  # get shapes
  N,T,D = x.shape
  _,H = h0.shape
  # init result
  h = np.zeros([N,0,H])
  c = np.zeros([N,0,H])
  cache = []
  for t in range(T):
    if t == 0:
      h_prev = h0
      c_prev = np.zeros([N,H])
    else:
      h_prev = h[:,t-1,:].reshape(N,H)
      c_prev = c[:,t-1,:].reshape(N,H)
    x_t = x[:,t,:].reshape(N,D)
    h_t, c_t, cache_t = lstm_step_forward(x_t, h_prev, c_prev, Wx, Wh, b) 
    h = np.append(h, h_t.reshape([N,1,H]), axis = 1)
    c = np.append(c, c_t.reshape([N,1,H]), axis = 1)
    cache.append(cache_t)
  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  # Get shapes
  x, prev_h, prev_c, Wx, Wh, b, next_h, next_c, i, f, o, g, a_i, a_f, a_o, a_g = cache[0]
  T = len(cache)
  N,D = x.shape
  _,H = next_h.shape
  # Start loop
  dprev_h_t = np.zeros([N,H])
  dprev_c_t = np.zeros([N,H])
  dx = np.zeros([N,T,D])
  for t in reversed(range(T)):
    dnext_h_t = dprev_h_t + dh[:,t,:].reshape(N,H)
    dnext_c_t = dprev_c_t
    dx_t, dprev_h_t, dprev_c_t, dWx_t, dWh_t, db_t = lstm_step_backward(dnext_h_t, dnext_c_t, cache[t])
    dx[:,t:t+1,:] = dx_t.reshape([N,1,D])
    if t == T - 1:
      dWx = dWx_t
      dWh = dWh_t
      db = db_t
    else:
      dWx += dWx_t
      dWh += dWh_t
      db += db_t
  dh0 = dprev_h_t
  return dx, dh0, dWx, dWh, db

def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

