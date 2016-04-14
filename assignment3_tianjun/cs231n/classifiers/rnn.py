import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
  """
  A CaptioningRNN produces captions from image features using a recurrent
  neural network.

  The RNN receives input vectors of size D, has a vocab size of V, works on
  sequences of length T, has an RNN hidden dimension of H, uses word vectors
  of dimension W, and operates on minibatches of size N.

  Note that we don't use any regularization for the CaptioningRNN.
  """
  
  def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn', dtype=np.float32):
    """
    Construct a new CaptioningRNN instance.

    Inputs:
    - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
      and maps each string to a unique integer in the range [0, V).
    - input_dim: Dimension D of input image feature vectors.
    - wordvec_dim: Dimension W of word vectors.
    - hidden_dim: Dimension H for the hidden state of the RNN.
    - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
    - dtype: numpy datatype to use; use float32 for training and float64 for
      numeric gradient checking.
    """
    
    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)
    
    self.cell_type = cell_type
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    self.params = {}
    

    self._null = word_to_idx['<NULL>']
    
    '''
    mxidx = -1
    for each_key in word_to_idx.keys():
      if word_to_idx[each_key] > mxidx:
        mxidx = word_to_idx[each_key]
    word_to_idx['<START>'] = mxidx + 1
    word_to_idx['<END>'] = mxidx + 2
    self._start = word_to_idx.get('<START>')
    self._end = word_to_idx.get('<END>')
    '''
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)
    vocab_size = len(word_to_idx)
    
    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
    self.params['W_embed'] /= 100
    
    # Initialize CNN -> hidden state projection parameters
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)
    
    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)
      
    # Cast parameters to correct dtype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)


  def loss(self, features, captions):
    """
    Compute training-time loss for the RNN. We input image features and
    ground-truth captions for those images, and use an RNN (or LSTM) to compute
    loss and gradients on all parameters.
    
    Inputs:
    - features: Input image features, of shape (N, D)
    - captions: Ground-truth captions; an integer array of shape (N, T) where
      each element is in the range 0 <= y[i, t] < V
      
    Returns a tuple of:
    - loss: Scalar loss
    - grads: Dictionary of gradients parallel to self.params
    """
    # Cut captions into two pieces: captions_in has everything but the last word
    # and will be input to the RNN; captions_out has everything but the first
    # word and this is what we will expect the RNN to generate. These are offset
    # by one relative to each other because the RNN should produce word (t+1)
    # after receiving word t. The first element of captions_in will be the START
    # token, and the first element of captions_out will be the first word.
    N,D = features.shape
    _,T = captions.shape
    '''
    starts = self._start * np.ones([N,1], dtype=np.int)
    captions_in = captions[:, :-1]
    captions_in = np.append(starts, captions_in, axis = 1)
    ends = self._end * np.ones([N,1], dtype=np.int)
    captions_out = captions[:, 1:]
    captions_out = np.append(captions_out, ends, axis = 1)
    '''
    # You'll need this 
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    mask = (captions_out != self._null)

    # Weight and bias for the affine transform from image features to initial
    # hidden state
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    
    # Word embedding matrix
    W_embed = self.params['W_embed']

    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the hidden-to-vocab transformation.
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    loss, grads = 0.0, {}
    # (1) Use an affine transformation to compute the initial hidden state     #
    #     from the image features. This should produce an array of shape (N, H)#
    h0, cache_imgproj = affine_forward(features, W_proj, b_proj)
    
    # (2) Use a word embedding layer to transform the words in captions_in     #
    #     from indices to vectors, giving an array of shape (N, T, W).         #
    embed, cache_embed = word_embedding_forward(captions_in, W_embed)
    
    # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
    #     process the sequence of input word vectors and produce hidden state  #
    #     vectors for all timesteps, producing an array of shape (N, T, H).    #
    if self.cell_type == 'rnn':
      rnn_out, cache_rnn = rnn_forward(embed, h0, Wx, Wh, b) 
    else:
      rnn_out, cache_rnn = lstm_forward(embed, h0, Wx, Wh, b)

    # (4) Use a (temporal) affine transformation to compute scores over the    #
    #     vocabulary at every timestep using the hidden states, giving an      #
    #     array of shape (N, T, V).                                            #
    affine_out, cache_affine = temporal_affine_forward(rnn_out, W_vocab, b_vocab) 
    
    # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
    #     the points where the output word is <NULL> using the mask above.     #
    loss, dsoftmax = temporal_softmax_loss(affine_out, captions_out, mask) 
    
    #                                                                          #
    # In the backward pass you will need to compute the gradient of the loss   #
    # with respect to all model parameters. Use the loss and grads variables   #
    # defined above to store loss and gradients; grads[k] should give the      #
    # gradients for self.params[k].                                            #
    daffine_out, dW_vocab, db_vocab = temporal_affine_backward(dsoftmax, cache_affine)
    if self.cell_type == "rnn":
      drnn_out, drnn_h0, dWx, dWh, db = rnn_backward(daffine_out, cache_rnn)
    else:
      drnn_out, drnn_h0, dWx, dWh, db = lstm_backward(daffine_out, cache_rnn)
    dW_embed = word_embedding_backward(drnn_out, cache_embed)
    dx, dW_proj, db_proj = affine_backward(drnn_h0, cache_imgproj)
    grads['W_proj'] = dW_proj
    grads['b_proj'] = db_proj
    grads['W_embed'] = dW_embed
    grads['Wx'] = dWx
    grads['Wh'] = dWh
    grads['b'] = db
    grads['W_vocab'] = dW_vocab
    grads['b_vocab'] = db_vocab
    return loss, grads


  def sample(self, features, max_length=30):
    """
    Run a test-time forward pass for the model, sampling captions for input
    feature vectors.

    At each timestep, we embed the current word, pass it and the previous hidden
    state to the RNN to get the next hidden state, use the hidden state to get
    scores for all vocab words, and choose the word with the highest score as
    the next word. The initial hidden state is computed by applying an affine
    transform to the input image features, and the initial word is the <START>
    token.

    For LSTMs you will also have to keep track of the cell state; in that case
    the initial cell state should be zero.

    Inputs:
    - features: Array of input image features of shape (N, D).
    - max_length: Maximum length T of generated captions.

    Returns:
    - captions: Array of shape (N, max_length) giving sampled captions,
      where each element is an integer in the range [0, V). The first element
      of captions should be the first sampled word, not the <START> token.
    """
    N = features.shape[0]
    captions = self._null * np.ones((N, max_length), dtype=np.int32)

    # Unpack parameters
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    W_embed = self.params['W_embed']
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    h0, cache_imgproj = affine_forward(features, W_proj, b_proj)
    _, H = h0.shape
    # prepare for the starts
    starts = self._start * np.ones([N,1], dtype=np.int)
     
    for t in xrange(max_length):
        if t == 0:
            h_prev = h0
            c_prev = np.zeros([N,H])
            input_word = starts
        embed, cache_embed = word_embedding_forward(input_word, W_embed)
        embed = embed.reshape([N,embed.shape[2]])
        if self.cell_type == "rnn":
            h_t, cache_t = rnn_step_forward(embed, h_prev, Wx, Wh, b) 
        else:
            h_t, c_t, cache_t = lstm_step_forward(embed, h_prev, c_prev, Wx, Wh, b)
            c_prev = c_t
        
        h_prev = h_t

        # get output, used as an input
        h_t_affin = h_t.reshape(N,1,H)
        affout_t, cache_affine = temporal_affine_forward(h_t_affin, W_vocab, b_vocab) 
        input_word = np.argmax(affout_t, axis=2)
        captions[:, t] = input_word.reshape([1, N])

    return captions
