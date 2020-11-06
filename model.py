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
    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

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
    cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)
    cycle_guided_loss = self.guided_filter_consistency_loss(self.G, self.F, x, y)
    # X -> Y
    fake_y = self.G(x)
    fake_y_pair = self.G(x_pair)
    G_l1_loss = self.pair_l1_loss(fake_y_pair,y_pair)
    G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
    G_loss =  G_gan_loss + cycle_loss + cycle_guided_loss + G_l1_loss
    D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)

    # Y -> X
    fake_x = self.F(y)
    fake_x_pair = self.F(y_pair)
    F_l1_loss = self.pair_l1_loss(fake_x_pair, x_pair)
    F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
    F_loss = F_gan_loss + cycle_loss + cycle_guided_loss + F_l1_loss
    D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)

    # summary
    tf.summary.histogram('D_Y/true', self.D_Y(y))
    tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
    tf.summary.histogram('D_X/true', self.D_X(x))
    tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

    tf.summary.scalar('loss/G', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/G_L', G_l1_loss)
    tf.summary.scalar('loss/F', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/F_L', F_l1_loss)
    tf.summary.scalar('loss/cycle', cycle_loss)
    tf.summary.scalar('loss/cycle_guided', cycle_guided_loss)

    tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(x))))
    tf.summary.image('X/fake_refine_x', utils.batch_convert2int(self.fake_refinex))
    tf.summary.image('X/refine_x', utils.batch_convert2int(self.refinex))
    tf.summary.image('X/pair', utils.batch_convert2int(x_pair))
    tf.summary.image('Y/pair', utils.batch_convert2int(y_pair))
    tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(y))))
    tf.summary.image('Y/fake_refine_Y', utils.batch_convert2int(self.fake_refiney))
    tf.summary.image('Y/refine_Y', utils.batch_convert2int(self.refiney))

    return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 10000
      decay_steps = 50000
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
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss

  def cycle_consistency_loss(self, G, F, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def guided_filter_consistency_loss(self, G, F, x, y):
    """ cycle consistency loss (L1 norm)
    """
    r = 4
    eps = 1e-6
    nhwc = True
    self.fake_refinex = guided_filter(x, F(G(x)), r, eps, nhwc)
    self.fake_refiney = guided_filter(y, G(F(y)), r, eps, nhwc)
    self.refinex = guided_filter(x, x, r, eps, nhwc)
    self.refiney = guided_filter(y, y, r, eps, nhwc)
    forward_loss_color_x = tf.reduce_mean(tf.abs(self.fake_refinex - F(G(x))))
    backward_loss_color_y = tf.reduce_mean(tf.abs(self.fake_refiney - G(F(y))))
    forward_loss_edge_x = tf.reduce_mean(tf.abs(self.fake_refinex - x))
    backward_loss_edge_y = tf.reduce_mean(tf.abs(self.fake_refiney - y))
    forward_loss = forward_loss_color_x+ forward_loss_edge_x
    backward_loss = backward_loss_color_y + backward_loss_edge_y
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def pair_l1_loss(self, fake, gt):
    l1 = tf.reduce_mean(tf.abs(fake-gt))*self.lambda1
    return l1