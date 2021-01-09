import tensorflow as tf
import utils,non_local,ops

class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name,reuse=self.reuse):
      # conv layers
      in_darkmap = ops.dark_channel(input)
      channel = self.ngf

      x = ops.conv(input, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
      x = ops._instance_norm(x,name='ins_norm')
      x = ops.relu(x)

      # Down-Sampling
      for i in range(2):
        x = ops.conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
        x = ops._instance_norm(x, name='ins_norm_' + str(i))
        x = ops.relu(x)

        channel = channel * 2

      # Down-Sampling Bottleneck
      for i in range(4):
        x = ops.resblock(x, channel, scope='resblock_' + str(i))

      # Class Activation Map
      cam_x = ops.global_avg_pooling(x)
      cam_gap_logit, cam_x_weight = ops.fully_connected_with_w(cam_x, scope='CAM_logit')
      x_gap = tf.multiply(x, cam_x_weight)

      cam_x = ops.global_max_pooling(x)
      cam_gmp_logit, cam_x_weight = ops.fully_connected_with_w(cam_x, reuse=True, scope='CAM_logit')
      x_gmp = tf.multiply(x, cam_x_weight)

      cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
      x = tf.concat([x_gap, x_gmp], axis=-1)

      x = ops.conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
      x = ops.relu(x)

      heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

      # Gamma, Beta block
      gamma, beta = self.MLP(x, reuse=self.reuse)

      # Up-Sampling Bottleneck
      for i in range(4):
        x = ops.adaptive_ins_layer_resblock(x, channel, gamma, beta, smoothing=True,
                                        scope='adaptive_resblock' + str(i))

      # Up-Sampling
      for i in range(2):
        x = ops.up_sample(x, scale_factor=2)
        x = ops.conv(x, channel // 2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='up_conv_' + str(i))
        x = ops.layer_instance_norm(x, scope='layer_ins_norm_' + str(i))
        x = ops.relu(x)

        channel = channel // 2

      x = ops.conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
      x = ops.tanh(x)
      out_darkmap = ops.dark_channel(x)
      self.reuse = True
      self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
      max_val = tf.reduce_max(in_darkmap, axis=-1, keep_dims=True)
      cond = tf.equal(in_darkmap, max_val)
      res = tf.where(cond)
      res_1d = tf.slice(res, [0, 0], [1, 4])
      res_1d = tf.squeeze(res_1d)
      # res_1d = tf.expand_dims(tf.expand_dims(res_1d,0),-1)
      A = tf.slice(input, res_1d, [-1, 1, 1, 3])

      return x, in_darkmap, out_darkmap, A, cam_logit, heatmap  # tf.ones_like(out_darkmap)*0.85

  def MLP(self, x, use_bias=True, reuse=False, scope='MLP'):
    channel = self.ngf * 4

    with tf.variable_scope(scope, reuse=reuse):
      for i in range(2):
        x = ops.fully_connected(x, channel, use_bias, scope='linear_' + str(i))
        x = ops.relu(x)

      gamma = ops.fully_connected(x, channel, use_bias, scope='gamma')
      beta = ops.fully_connected(x, channel, use_bias, scope='beta')

      gamma = tf.reshape(gamma, shape=[1, 1, 1, channel])
      beta = tf.reshape(beta, shape=[1, 1, 1, channel])

      return gamma, beta




  def encode_image(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image

  def decode_image(self, input):
  #   #image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.decode_jpeg(input)
    return image



