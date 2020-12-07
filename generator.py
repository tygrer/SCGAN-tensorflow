import tensorflow as tf
import ops
import utils,non_local

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
    with tf.variable_scope(self.name):
      # conv layers
      in_darkmap = ops.dark_channel(input)
      #input_n = tf.concat([input, in_darkmap], axis=-1)
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_32')
      # (?, w, h, 32)
      d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d64')
     # ars = ops.n_res_blocks(d64, reuse=self.reuse, n=2, is_training=self.is_training)

      #art = ops.uk(ars, 3, is_training=self.is_training, norm=self.norm,
      #    reuse=self.reuse, name='a_u3')  # (?, w, h, 3)
      #aestimate,_ = non_local.sn_non_local_block_sim_attention(art, in_darkmap, None, reuse=self.reuse, name='at_non_local')
      d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

      #res_output,_ = non_local.sn_non_local_block_sim_self(d128, None, reuse=self.reuse, name='g_non_local')

      if self.image_size <= 128 or self.is_training == False:
        # use 6 residual blocks for 128x128 images
        '''
        output1 = ops.Rk(res_output, 4*self.ngf, reuse=self.reuse, norm=self.norm, is_training=self.is_training, name='R{}'.format(1))
        res_output_mix1 = ops.cat_two_channel(output1, res_output, 4*self.ngf, reuse=self.reuse, name='res_cat1')
        output2 = ops.Rk(res_output_mix1, 4 * self.ngf, reuse=self.reuse, norm=self.norm, is_training=self.is_training,
                        name='R{}'.format(2))
        res_output_mix2 = ops.cat_two_channel(res_output_mix1, res_output, 4*self.ngf, reuse=self.reuse, name='res_cat21',norm=self.norm, is_training=self.is_training)
        res_output_mix2 = ops.cat_two_channel(output2, res_output_mix2, 4*self.ngf, reuse=self.reuse, name='res_cat22',norm=self.norm, is_training=self.is_training)
        output3 = ops.Rk(res_output_mix2, 4 * self.ngf, reuse=self.reuse, norm=self.norm, is_training=self.is_training,
                        name='R{}'.format(3))
        res_output_mix3 = ops.cat_two_channel(res_output_mix2, res_output_mix1, 4*self.ngf, reuse=self.reuse, name='res_cat31',norm=self.norm, is_training=self.is_training)
        res_output_mix3 = ops.cat_two_channel(output3, res_output_mix3, 4*self.ngf, reuse=self.reuse, name='res_cat32',norm=self.norm, is_training=self.is_training)
        output4 = ops.Rk(res_output_mix3, 4 * self.ngf, reuse=self.reuse,  norm=self.norm, is_training=self.is_training,
                        name='R{}'.format(4))
        res_output_mix4 = ops.cat_two_channel(res_output_mix3, res_output_mix2, 4*self.ngf, reuse=self.reuse, name='res_cat41',norm=self.norm, is_training=self.is_training)
        res_output_mix4 = ops.cat_two_channel(output4, res_output_mix4, 4*self.ngf, reuse=self.reuse, name='res_cat42',norm=self.norm, is_training=self.is_training)
        output5 = ops.Rk(res_output_mix4, 4 * self.ngf, reuse=self.reuse, norm=self.norm, is_training=self.is_training,
                        name='R{}'.format(5))
        res_output_mix5 = ops.cat_two_channel(res_output_mix4, res_output_mix3, 4*self.ngf, reuse=self.reuse, name='res_cat51',norm=self.norm, is_training=self.is_training)
        res_output_mix5 = ops.cat_two_channel(output5, res_output_mix5, 4*self.ngf, reuse=self.reuse, name='res_cat52',norm=self.norm, is_training=self.is_training)
        res_output = ops.Rk(res_output_mix5, 4 * self.ngf, reuse=self.reuse, norm=self.norm, is_training=self.is_training,
                        name='R{}'.format(6))
        '''
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6, is_training=self.is_training)      # (?, w/4, h/4, 128)
      else:
        # 9 blocks for higher resolution
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9, is_training=self.is_training)  # (?, w/4, h/4, 128)


      # fractional-strided convolution
      #res_output_mix = ops.cat_two_channel(res_output, d128, 4 * self.ngf, reuse=self.reuse, name='res_cat',norm=self.norm, is_training=self.is_training)
      u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u64')         # (?, w/2, h/2, 64)
      #u64_mix = ops.cat_two_channel(u64, d64, 2*self.ngf, reuse=self.reuse, name='u64_cat',norm=self.norm, is_training=self.is_training)
      u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)
      #u32_mix = ops.cat_two_channel(u32, c7s1_32, reuse=self.reuse,name='u32_cat')
      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here
      output = ops.c7s1_k(u32, 3, norm=None,
          activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
      out_darkmap = ops.dark_channel(output)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output, in_darkmap, out_darkmap, tf.ones_like(out_darkmap)

  def encode_image(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image

  def decode_image(self, input):
  #   #image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.decode_jpeg(input)
    return image



