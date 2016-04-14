# As usual, a bit of setup

import time, os, json
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

from cs231n.classifiers.pretrained_cnn import PretrainedCNN
from cs231n.data_utils import load_tiny_imagenet
from cs231n.image_utils import blur_image, deprocess_image, preprocess_image

#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True)
model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')


def create_class_visualization(target_y, model, **kwargs):
  """
  Perform optimization over the image to generate class visualizations.
        
  Inputs:
  - target_y: Integer in the range [0, 100) giving the target class
  - model: A PretrainedCNN that will be used for generation
                
  Keyword arguments:
  - learning_rate: Floating point number giving the learning rate
  - blur_every: An integer; how often to blur the image as a regularizer
  - l2_reg: Floating point number giving L2 regularization strength on the image; this is lambda in the equation above.
  - max_jitter: How much random jitter to add to the image as regularization
  - num_iterations: How many iterations to run for
  - show_every: How often to show the image
  """
  learning_rate = kwargs.pop('learning_rate', 10000)
  blur_every = kwargs.pop('blur_every', 1)
  l2_reg = kwargs.pop('l2_reg', 1e-6)
  max_jitter = kwargs.pop('max_jitter', 4)
  num_iterations = kwargs.pop('num_iterations', 100)
  show_every = kwargs.pop('show_every', 25)

  X = np.random.randn(1, 3, 64, 64)
  mode = 'test'
  for t in xrange(num_iterations):
    # As a regularizer, add random jitter to the image
    ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
    X = np.roll(np.roll(X, ox, -1), oy, -2)
    scores, cache = model.forward(X, mode=mode)
    class_mask = np.zeros(scores.shape)
    class_mask[0,target_y] = 1
    scores = scores * class_mask
    dX, grads = model.backward(scores, cache)
    dX = dX - l2_reg * X
    X = X + learning_rate * dX
    # Undo the jitter
    X = np.roll(np.roll(X, -ox, -1), -oy, -2)
    # As a regularizer, clip the image
    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
    # As a regularizer, periodically blur the image
    if t % blur_every == 0:
      X = blur_image(X)
    # Periodically show the image
    if t % show_every == 0:
      plt.imshow(deprocess_image(X, data['mean_image']))
      plt.gcf().set_size_inches(3, 3)
      plt.axis('off')
      img_path = 'images/class_%d_%d.jpg' % (target_y, t)
      plt.savefig(img_path)
  return X

'''
# visualize class
target_y = 50
print data['class_names'][target_y]
X = create_class_visualization(target_y, model, l2_reg=2e-5,num_iterations=300, show_every=100)
'''

# Deep Dream
def deepdream(X, layer, model, **kwargs):
  """
  Generate a DeepDream image.
        
  Inputs:
  - X: Starting image, of shape (1, 3, H, W)
  - layer: Index of layer at which to dream
  - model: A PretrainedCNN object
                    
  Keyword arguments:
  - learning_rate: How much to update the image at each iteration
  - max_jitter: Maximum number of pixels for jitter regularization
  - num_iterations: How many iterations to run for
  - show_every: How often to show the generated image
  """

  X = X.copy()
    
  learning_rate = kwargs.pop('learning_rate', 5.0)
  max_jitter = kwargs.pop('max_jitter', 16)
  num_iterations = kwargs.pop('num_iterations', 100)
  show_every = kwargs.pop('show_every', 25)

  for t in xrange(num_iterations):
  # As a regularizer, add random jitter to the image
    ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
    X = np.roll(np.roll(X, ox, -1), oy, -2)
    activation, cache = model.forward(X, mode='test', start=0, end=layer)
    dX, grads = model.backward(activation, cache)
    X = X + learning_rate * dX
    # Undo the jitter
    X = np.roll(np.roll(X, -ox, -1), -oy, -2)
    # As a regularizer, clip the image
    mean_pixel = data['mean_image'].mean(axis=(1, 2), keepdims=True)
    X = np.clip(X, -mean_pixel, 255.0 - mean_pixel)
    # Periodically show the image
    if t == 0 or (t + 1) % show_every == 0:
      img = deprocess_image(X, data['mean_image'], mean='pixel')
      plt.imshow(img)
      plt.title('t = %d' % (t + 1))
      plt.gcf().set_size_inches(8, 8)
      plt.axis('off')
      filename = 'images/deepdream_%d.jpg' % (t+1)
      plt.savefig(filename)
  return X

def read_image(filename, max_size):
    """
    Read an image from disk and resize it so its larger side is max_size
    """
    img = imread(filename)
    H, W, _ = img.shape
    if H >= W:
      img = imresize(img, (max_size, int(W * float(max_size) / H)))
    elif H < W:
      img = imresize(img, (int(H * float(max_size) / W), max_size))
    return img

filename = 'kitten.jpg'
max_size = 256
img = read_image(filename, max_size)
plt.imshow(img)
plt.axis('off')

# Preprocess the image by converting to float, transposing,
# and performing mean subtraction.
img_pre = preprocess_image(img, data['mean_image'], mean='pixel')
out = deepdream(img_pre, 7, model, learning_rate=2000)


