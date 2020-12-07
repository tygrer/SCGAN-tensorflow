import tensorflow as tf

## Layers: follow the naming convention used in the original paper

### Generator layers
def _l2normalize(v, eps=1e-12):
  """l2 normize the input vector."""
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(weights, num_iters=1, update_collection=None,
                           with_sigma=False):
  """Performs Spectral Normalization on a weight tensor.

  Specifically it divides the weight tensor by its largest singular value. This
  is intended to stabilize GAN training, by making the discriminator satisfy a
  local 1-Lipschitz constraint.
  Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
  [sn-gan] https://openreview.net/pdf?id=B1QRgziT-

  Args:
    weights: The weight tensor which requires spectral normalization
    num_iters: Number of SN iterations.
    update_collection: The update collection for assigning persisted variable u.
                       If None, the function will update u during the forward
                       pass. Else if the update_collection equals 'NO_OPS', the
                       function will not update the u during the forward. This
                       is useful for the discriminator, since it does not update
                       u in the second pass.
                       Else, it will put the assignment in a collection
                       defined by the user. Then the user need to run the
                       assignment explicitly.
    with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
  Returns:
    w_bar: The normalized weight tensor
    sigma: The estimated singular value for the weight tensor.
  """
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
  u = tf.get_variable('u', [1, w_shape[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
  u_ = u
  for _ in range(num_iters):
    v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
    u_ = _l2normalize(tf.matmul(v_, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
  w_mat /= sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(u_)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_))
  if with_sigma:
    return w_bar, sigma
  else:
    return w_bar

def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'c7sk-32'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[7, 7, input.get_shape()[3], k])

    padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output

def dk(input, k, reuse=False, norm='instance', is_training=True, name=None):
  """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
    name: string, e.g. 'd64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(conv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output

def Rk(input, k,  reuse=False, norm='instance', is_training=True, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  Args:
    input: 4D Tensor
    k: integer, number of filters (output depth)
    reuse: boolean
    name: string
  Returns:
    4D tensor (same shape as input)
  """
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      weights1 = _weights("weights1",
        shape=[3, 3, input.get_shape()[3], k])
      padded1 = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = tf.nn.conv2d(padded1, weights1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = _norm(conv1, is_training, norm)
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
      weights2 = _weights("weights2",
        shape=[3, 3, relu1.get_shape()[3], k])

      padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = tf.nn.conv2d(padded2, weights2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = _norm(conv2, is_training, norm)
    output = input+normalized2
    return output

def n_res_blocks(input, reuse, norm='instance', is_training=True, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
    input = output
  return output

def cat_two_channel(input1, input2, k, reuse=False, is_training=False, norm="",name="cat"):
    with tf.variable_scope(name, reuse=reuse) as scope:
        n, h, w, c = input1.shape.as_list()
        x = tf.concat([tf.slice(input1,[0,0,0,0],[1,h,w,1]),tf.slice(input2,[0,0,0,0],[1,h,w,1])],-1)
        for i in list(range(c)[1:]):
          x = tf.concat([x,tf.slice(input1,[0,0,0,i],[1,h,w,1]),tf.slice(input2,[0,0,0,i],[1,h,w,1])],-1)
        '''
       print(x.shape.as_list())
        weights1 = _weights("weights",
                            shape=[3, 3, x.get_shape()[3], k])
        padded1 = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        conv1 = tf.nn.conv2d(padded1, weights1,
                             strides=[1, 1, 1, 1], padding='VALID')
        normalized1 = _norm(conv1, is_training, norm)
        relu1 = tf.nn.relu(normalized1)
        #x = tf.concat([input1, input2], axis=0)
        '''
        # x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
        # x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        # output = tf.reshape(x_transposed, [-1, h, w, c]
    return x

def uk(input, k, reuse=False, norm='instance', is_training=True, name=None, output_size=None):
  """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
      with k filters, stride 1/2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'c7sk-32'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()

    weights = _weights("weights",
      shape=[3, 3, k, input_shape[3]])

    if not output_size:
      output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, k]
    fsconv = tf.nn.conv2d_transpose(input, weights,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(fsconv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output

### Discriminator layers
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None, update_collection=None):
  """ A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    slope: LeakyReLU's slope
    stride: integer
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'C64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], k], update_collection=update_collection)

    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    normalized = _norm(conv, is_training, norm)
    output = _leaky_relu(normalized, slope)
    return output

def last_conv(input, reuse=False, use_sigmoid=False, name=None, is_training=True):
  """ Last convolutional layer of discriminator network
      (1 filter with size 4x4, stride 1)
  Args:
    input: 4D tensor
    reuse: boolean
    use_sigmoid: boolean (False if use lsgan)
    name: string, e.g. 'C64'
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], 1])
    biases = _biases("biases", [1])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    if use_sigmoid:
      output = tf.sigmoid(output)
    return output

### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02,update_collection=None):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  #w_bar = spectral_normed_weight(var, num_iters=1, update_collection=update_collection)
  return var


def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _norm(input, is_training, norm='instance'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input,is_training)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  else:
    return input

def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope("batch_norm"):
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)

def _instance_norm(input,is_training):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)

def dark_channel(input):
  rgb_min = tf.reduce_min(input, -1, keep_dims=True)
  dark_map=-tf.nn.max_pool(-rgb_min,[1,5,5,1],[1, 1, 1 ,1],"SAME")
  return dark_map