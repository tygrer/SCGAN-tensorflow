import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
from guided_filter import guided_filter

REAL_LABEL = 0.9

class CycleGAN:
  def __init__(self,
               X_train_file='',
               Y_train_file='',
               X_pair_train_file='',
               Y_pair_train_file='',
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               lambda1=10,
               lambda2=10,
               learning_rate=2e-4,
               is_training=True,
               beta1=0.5,
               ngf=64
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.X_train_file = X_train_file
    self.Y_train_file = Y_train_file
    self.X_pair_train_file = X_pair_train_file
    self.Y_pair_train_file = Y_pair_train_file
    self.is_training = is_training
    with tf.device('/device:GPU:2'):
      self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    with tf.device('/device:GPU:3'):
      self.D_Y = Discriminator('D_Y',self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    with tf.device('/device:GPU:4'):
      self.F = Generator('F', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    with tf.device('/device:GPU:5'):
      self.D_X = Discriminator('D_X',
      self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

  def get_vars(self):
  #   """Get variables."""
    t_vars = tf.trainable_variables()
  #   # TODO(olganw): scoping or collections for this instead of name hack
  #
    self.sigma_ratio_vars = [var for var in t_vars if 'sigma_ratio' in var.name]

  def model(self):
    X_reader = Reader(self.X_train_file, name='X',
        image_size=self.image_size, batch_size=self.batch_size)
    Y_reader = Reader(self.Y_train_file, name='Y',
        image_size=self.image_size, batch_size=self.batch_size)
    X_pair_reader = Reader(self.X_pair_train_file, name='Xpair',
        image_size=self.image_size, batch_size=self.batch_size)
    Y_pair_reader = Reader(self.Y_pair_train_file, name='Ypair',
        image_size=self.image_size, batch_size=self.batch_size)

    x = X_reader.feed()
    y = Y_reader.feed()
    x_pair = X_pair_reader.feed()
    y_pair = Y_pair_reader.feed()

    # X -> Y
    fake_y, real_x_dm, fake_y_dm, x_aestimate = self.G(x)
    fake_y_pair, real_x_pair_dm, fake_y_pair_dm, x_pair_aestimate = self.G(x_pair)
    G_l1_loss = self.pair_l1_loss(fake_y_pair,y_pair)
    G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
    t1, t2, g_atmospheric_loss = self.atmospheric_refine_loss_g(real_x_dm, fake_y_dm, x, fake_y, x_aestimate)
    restructx, _, restructx_dm, restructy_aestimate = self.F(fake_y)

    D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)

    # Y -> X
    fake_x, real_y_dm, fake_x_dm, y_aestimate = self.F(y)
    restructy, _, restructy_dm, restructx_aestimate = self.G(fake_x)
    fake_x_pair, real_y_pair_dm, fake_x_pair_dm, y_pair_aestimate = self.F(y_pair)
    F_l1_loss = self.pair_l1_loss(fake_x_pair, x_pair)
    F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
    t1_f, t2_f, f_atmospheric_loss = self.atmospheric_refine_loss_f(real_y_dm, fake_x_dm, y, fake_x, y_aestimate)
    D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)
    cycle_loss = self.cycle_consistency_loss(restructx, restructy, x, y)
    cycle_guided_loss = self.guided_filter_consistency_loss(restructx, restructy, x, y)
    dark_channel_loss = self.dark_channel_loss(fake_y_dm, fake_x_dm, real_x_dm, real_y_dm)
    G_loss =  G_gan_loss + cycle_loss + cycle_guided_loss + G_l1_loss + g_atmospheric_loss + dark_channel_loss
    F_loss = F_gan_loss + cycle_loss + cycle_guided_loss + F_l1_loss + f_atmospheric_loss + dark_channel_loss


    # summary
    #tf.summary.histogram('D_Y/true', self.D_Y(y,None))
    #tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x),"NO_OPS"))
    #tf.summary.histogram('D_X/true', self.D_X(x,None))
    #tf.summary.histogram('D_X/fake', self.D_X(self.F(y),"NO_OPS"))

    tf.summary.scalar('loss/G', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/G_L', G_l1_loss)
    tf.summary.scalar('loss/F', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/F_L', F_l1_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)
    tf.summary.scalar('loss/cycle_guided', cycle_guided_loss)
    tf.summary.scalar('loss/g_atmospheric_loss', g_atmospheric_loss)
    tf.summary.scalar('loss/f_atmospheric_loss', f_atmospheric_loss)
    tf.summary.scalar('loss/dark_channel_loss', dark_channel_loss)

    tf.summary.image('X/darkmap', utils.batch_convert2int(real_x_dm))
    tf.summary.image('Y/gen_darkmap', utils.batch_convert2int(fake_y_dm))
    tf.summary.image('X/generated', utils.batch_convert2int(fake_y))
    tf.summary.image('X/restru_darkmap', utils.batch_convert2int(restructx_dm))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(restructx))
    tf.summary.image('X/fake_refine_x', utils.batch_convert2int(self.fake_refinex))
    tf.summary.image('X/t1', utils.batch_convert2int((t1)))
    tf.summary.image('X/t2', utils.batch_convert2int((t2)))
    tf.summary.image('X/A', utils.batch_convert2int((x_aestimate)))
    tf.summary.image('X/pair', utils.batch_convert2int(x_pair))
    tf.summary.image('Y/pair', utils.batch_convert2int(y_pair))
    tf.summary.image('Y/darkmap', utils.batch_convert2int(real_y_dm))
    tf.summary.image('X/gen_darkmap', utils.batch_convert2int(fake_x_dm))
    tf.summary.image('Y/generated', utils.batch_convert2int(fake_x))
    tf.summary.image('Y/restru_darkmap', utils.batch_convert2int(restructy_dm))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(restructy))
    tf.summary.image('Y/fake_refine_Y', utils.batch_convert2int(self.fake_refiney))
    tf.summary.image('Y/t1', utils.batch_convert2int((t1_f)))
    tf.summary.image('Y/t2', utils.batch_convert2int((t2_f)))
    tf.summary.image('Y/A', utils.batch_convert2int((y_aestimate)))
    self.get_vars()
    print('sigma_ratio_vars', self.sigma_ratio_vars)
    for var in self.sigma_ratio_vars:
       tf.summary.scalar(var.name, var)
    return G_loss, D_Y_loss, F_loss, D_X_loss, g_atmospheric_loss, f_atmospheric_loss, dark_channel_loss,\
           cycle_guided_loss,cycle_loss,G_gan_loss,F_gan_loss,G_l1_loss,F_l1_loss,fake_x,fake_y




  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 2000
      decay_steps = 5000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  # tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                  #                           decay_steps, end_learning_rate,
                  #                           power=1.0),
            tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, 0.97),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    with tf.device('/cpu:0'):

      G_optimizer1 = make_optimizer(G_loss, self.G.variables, name='Adam_G')
      D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    #G_optimizer2 = make_optimizer(G_loss, self.G.variables, name='Adam_G')
      F_optimizer1 =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
      D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')
    #F_optimizer2 = make_optimizer(F_loss, self.F.variables, name='Adam_F')

    #with tf.control_dependencies([G_optimizer1, D_Y_optimizer, G_optimizer2, F_optimizer1, D_X_optimizer,F_optimizer2]):
      with tf.control_dependencies(
            [G_optimizer1, D_Y_optimizer, F_optimizer1, D_X_optimizer]):
        return tf.no_op(name='optimizers')


  def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """

    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y,None), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y,"NO_OPS")))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y,None)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y,"NO_OPS")))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y,"NO_OPS"), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y,"NO_OPS"))) / 2
    return loss

  def cycle_consistency_loss(self, rx, ry, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(rx-x))
    backward_loss = tf.reduce_mean(tf.abs(ry-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def guided_filter_consistency_loss(self, rx, ry, x, y):
    """ cycle consistency loss (L1 norm)
    """
    r = 4
    eps = 1e-6
    nhwc = True
    self.fake_refinex = tf.clip_by_value(guided_filter(x, rx, r, eps, nhwc),0,1)
    self.fake_refiney = tf.clip_by_value(guided_filter(y, ry, r, eps, nhwc),0,1)

    # self.refinex = guided_filter(x, x, r, eps, nhwc)
    # self.refiney = guided_filter(y, y, r, eps, nhwc)
    forward_loss_edge_x = tf.reduce_mean(tf.abs(self.fake_refinex - rx))
    backward_loss_edge_y = tf.reduce_mean(tf.abs(self.fake_refiney - ry))
    forward_loss_color_x = tf.reduce_mean(tf.abs(self.fake_refinex - x))
    backward_loss_color_y = tf.reduce_mean(tf.abs(self.fake_refiney - y))
    forward_loss = 2*forward_loss_color_x+ 2*forward_loss_edge_x
    backward_loss = 2*backward_loss_color_y + 2*backward_loss_edge_y
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def pair_l1_loss(self, fake, gt):
    l1 = tf.reduce_mean(tf.abs(fake-gt))*self.lambda1
    return l1

  def atmospheric_refine_loss_g(self, real_dm, fake_dm, I, J, a):
    r = 4
    eps = 1e-6
    nhwc = True
    fake_dm = tf.concat([fake_dm, fake_dm, fake_dm], axis=-1)
    real_dm = tf.concat([real_dm, real_dm, real_dm],axis=-1)
    # print("I shape", I.get_shape())
    # print("real_dm shape", real_dm.get_shape())
    ti = guided_filter(I, real_dm, r, eps, nhwc)
    tj = guided_filter(J, fake_dm, r, eps, nhwc)
    t = tf.div((ti-a),(tj-a))
    t = tf.clip_by_value(t, 0.1, 1)
    restruct_I = tf.multiply(J,t)+tf.multiply((1-t),a)
    atmospheric_loss = tf.reduce_mean(tf.abs(restruct_I - I))
    return ti, tj, atmospheric_loss

  def atmospheric_refine_loss_f(self, real_dm, fake_dm, J, I, a):
    r = 4
    eps = 1e-6
    nhwc = True
    fake_dm = tf.concat([fake_dm, fake_dm, fake_dm], axis=-1)
    real_dm = tf.concat([real_dm, real_dm, real_dm],axis=-1)
    ti = guided_filter(I, fake_dm, r, eps, nhwc)
    tj = guided_filter(J, real_dm, r, eps, nhwc)
    t = tf.div((ti-a),(tj-a))
    restruct_J = tf.div(I-a,tf.clip_by_value(t,0.1,1)) + a
    atmospheric_loss = tf.reduce_mean(tf.abs(restruct_J - J))
    return ti, tj, atmospheric_loss

  def dark_channel_loss(self, real_dm, fake_dm, real, fake):
    r = 4
    eps = 1e-6
    nhwc = True
    real_dp = guided_filter(real, real_dm, r, eps, nhwc)
    fake_dp = guided_filter(fake, fake_dm, r, eps, nhwc)
    forward_dark_loss = tf.reduce_mean(tf.abs(real_dp - fake_dp))
    backward_dark_loss = tf.reduce_mean(tf.abs(real_dp - fake_dp))
    dark_loss = 10*forward_dark_loss + 10*backward_dark_loss
    return dark_loss