import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
from guided_filter import guided_filter

REAL_LABEL = 1

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
    use_sigmoid = use_lsgan
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
    t1_g, t2_g, g_atmospheric_loss = self.atmospheric_refine_loss_g(x, fake_y, x_aestimate)
    restructx, _, restructx_dm, restructy_aestimate = self.F(fake_y)
    #1_f, t2_f, f_atmospheric_loss_back = self.atmospheric_refine_loss_f(fake_y, restructx, restructy_aestimate)
    D_Y_loss = self.discriminator_loss(self.D_Y, y, fake_y, use_lsgan=self.use_lsgan)
    foreground_gt, foreground_restruction_gI,foreground_g_loss = self.only_limited_foreground_loss_g(x, fake_y, x_aestimate)
    #foreground_g_loss_back = self.only_limited_foreground_loss_f(fake_y, x, restructy_aestimate)
    only_limit_foreground_g_loss = foreground_g_loss #+ foreground_g_loss_back
    restruct_loss, tt, aa = self.restruct_phscial_model(x, fake_y)
    # Y -> X
    fake_x, real_y_dm, fake_x_dm, y_aestimate = self.F(y)
    restructy, _, restructy_dm, restructx_aestimate = self.G(fake_x)
    fake_x_pair, real_y_pair_dm, fake_x_pair_dm, y_pair_aestimate = self.F(y_pair)
    F_l1_loss = self.pair_l1_loss(fake_x_pair, x_pair)
    F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
    #1_f_1, t2_f_1, f_atmospheric_loss = self.atmospheric_refine_loss_f(y, fake_x, restructx_aestimate)
    #1_g_1, t2_g_1, g_atmospheric_loss_back = self.atmospheric_refine_loss_g(fake_x, restructy, restructx_aestimate)
    D_X_loss = self.discriminator_loss(self.D_X, x, fake_x, use_lsgan=self.use_lsgan)
    foreground_ft, foreground_restruction_fI, foreground_f_loss = self.only_limited_foreground_loss_f(y, fake_x, y_aestimate)
    #foreground_f_loss_back = self.only_limited_foreground_loss_f(fake_x, y, restructx_aestimate)
    only_limit_foreground_f_loss = foreground_f_loss #+ foreground_f_loss_back
    cycle_loss = self.cycle_consistency_loss(restructx, restructy, x, y)
    cycle_guided_loss = self.guided_filter_consistency_loss(restructx, restructy, x, y)
    dark_channel_loss = self.dark_channel_loss(restructx_dm, restructy_dm, real_x_dm, real_y_dm)

    G_loss =  G_gan_loss + cycle_loss + cycle_guided_loss #+ G_l1_loss  #+ dark_channel_loss#+restruct_loss + only_limit_foreground_g_loss #+ g_atmospheric_loss + f_atmospheric_loss_back
    F_loss = F_gan_loss + cycle_loss + cycle_guided_loss #+ F_l1_loss #+ dark_channel_loss #+ f_atmospheric_loss  + g_atmospheric_loss_back

    atmospheric_loss_g = g_atmospheric_loss + restruct_loss + only_limit_foreground_g_loss  #+ f_atmospheric_loss_back
    atmospheric_loss_f = only_limit_foreground_f_loss #+ g_atmospheric_loss_back
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
    #tf.summary.scalar('loss/f_atmospheric_loss', f_atmospheric_loss)
    tf.summary.scalar('loss/dark_channel_loss', dark_channel_loss)
    tf.summary.scalar('loss/only_limit_foreground_g_loss', only_limit_foreground_g_loss)
    tf.summary.scalar('loss/only_limit_foreground_f_loss', only_limit_foreground_f_loss)
    tf.summary.scalar('loss/restruct_loss', restruct_loss)

    tf.summary.image('X/darkmap', utils.batch_convert2int(real_x_dm))
    tf.summary.image('Y/gen_darkmap', utils.batch_convert2int(fake_x_dm))
    tf.summary.image('X/generated', utils.batch_convert2int(fake_y))
    tf.summary.image('X/restru_darkmap', utils.batch_convert2int(restructx_dm))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(restructx))
    tf.summary.image('X/fake_refine_x', utils.batch_convert2int(self.fake_refinex))
    tf.summary.image('X/t1', utils.batch_convert2int((t1_g)))
    tf.summary.image('X/t2', utils.batch_convert2int((t2_g)))
    tf.summary.image('X/A', utils.batch_convert2int((x_aestimate)))
    tf.summary.image('X/restruct_t',utils.batch_convert2int((tt)))
    tf.summary.image('X/restruct_A',utils.batch_convert2int(aa))
    tf.summary.image('X/pair', utils.batch_convert2int(x_pair))
    tf.summary.image('X/forgroundgt', utils.batch_convert2int(foreground_gt))
    tf.summary.image('X/forgroundg_RI_t', utils.batch_convert2int(foreground_restruction_gI))
    tf.summary.image('Y/pair', utils.batch_convert2int(y_pair))
    tf.summary.image('Y/darkmap', utils.batch_convert2int(real_y_dm))
    tf.summary.image('X/gen_darkmap', utils.batch_convert2int(fake_y_dm))
    tf.summary.image('Y/generated', utils.batch_convert2int(fake_x))
    tf.summary.image('Y/restru_darkmap', utils.batch_convert2int(restructy_dm))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(restructy))
    tf.summary.image('Y/fake_refine_Y', utils.batch_convert2int(self.fake_refiney))
    #tf.summary.image('Y/t1', utils.batch_convert2int((t1_f_1)))
    #tf.summary.image('Y/t2', utils.batch_convert2int((t2_f_1)))
    tf.summary.image('Y/A', utils.batch_convert2int((y_aestimate)))
    tf.summary.image('Y/forgroundft', utils.batch_convert2int(foreground_ft))
    tf.summary.image('Y/forgroundg_RI_t', utils.batch_convert2int(foreground_restruction_fI))
    self.get_vars()
    print('sigma_ratio_vars', self.sigma_ratio_vars)
    for var in self.sigma_ratio_vars:
       tf.summary.scalar(var.name, var)
    return G_loss, D_Y_loss, F_loss, D_X_loss, g_atmospheric_loss, dark_channel_loss,\
           cycle_guided_loss,cycle_loss,G_gan_loss,F_gan_loss,G_l1_loss,F_l1_loss,fake_x,fake_y, \
           atmospheric_loss_g,only_limit_foreground_g_loss,only_limit_foreground_f_loss


  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss, atmospheric_loss_g):
    def make_optimizer_G(loss, variables, atmospheric_loss=None, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 10000
      decay_steps = 30000
      start_atmospheric_step =90000
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
      '''
      learning_step = (
        tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
          .minimize(loss, global_step=global_step, var_list=variables)
      )
      learning_rate_at = (
          tf.where(
                  tf.greater_equal(global_step, start_atmospheric_step),
            learning_rate,
            0.0
          )
      )
      '''
      if atmospheric_loss is not None:
        learning_step = (tf.cond(global_step>start_atmospheric_step,
          lambda:tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name + 'at')
            .minimize(loss+atmospheric_loss, global_step=global_step, var_list=variables),
          lambda:tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name + 'at')
            .minimize(loss, global_step=global_step, var_list=variables),
          )
        )
      else:
        learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
            .minimize(loss, global_step=global_step, var_list=variables)
        )
      return learning_step
    def make_optimizer_D(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (c~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      start_decay_step = 10000
      decay_steps = 30000
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
      learning_step = (
        tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
          .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer  = make_optimizer_G(G_loss, self.G.variables, None, name='Adam_G')
    F_optimizer  =  make_optimizer_G(F_loss, self.F.variables, None, name='Adam_F')
    D_Y_optimizer = make_optimizer_D(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    D_X_optimizer = make_optimizer_D(D_X_loss, self.D_X.variables, name='Adam_D_X')
    #G_optimizer = make_optimizer_D(G_loss, self.G.variables, name='Adam_G_Y')
    #F_optimizer = make_optimizer_D(F_loss, self.F.variables,name='Adam_F_X')
  #G_optimizer2 = make_optimizer(G_loss, self.G.variables, name='Adam_G')


  #F_optimizer2 = make_optimizer(F_loss, self.F.variables, name='Adam_F')

    #with tf.control_dependencies([G_optimizer1, D_Y_optimizer, F_optimizer1, D_X_optimizer, G_optimizer2, F_optimizer2]):
    with tf.control_dependencies(
         [G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
    #with tf.control_dependencies([D_Y_optimizer, D_X_optimizer, G_optimizer, F_optimizer]):

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
    forward_loss = tf.reduce_mean(tf.abs(x-rx))
    backward_loss = tf.reduce_mean(tf.abs(y-ry))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def guided_filter_consistency_loss(self, rx, ry, x, y):
    """ cycle consistency loss (L1 norm)
    """
    r = 4
    eps = 1e-6
    nhwc = True
    self.fake_refinex = guided_filter(x, rx, r, eps, nhwc)
    self.fake_refiney = guided_filter(y, ry, r, eps, nhwc)

    # self.refinex = guided_filter(x, x, r, eps, nhwc)
    # self.refiney = guided_filter(y, y, r, eps, nhwc)
    forward_loss_edge_x = tf.reduce_mean(tf.abs(self.fake_refinex - rx))
    backward_loss_edge_y = tf.reduce_mean(tf.abs(self.fake_refiney - ry))
    forward_loss_color_x = tf.reduce_mean(tf.abs(x - self.fake_refinex))
    backward_loss_color_y = tf.reduce_mean(tf.abs(y - self.fake_refiney))
    forward_loss = forward_loss_color_x+ forward_loss_edge_x
    backward_loss = backward_loss_color_y + backward_loss_edge_y
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def pair_l1_loss(self, fake, gt):
    l1 = tf.reduce_mean(tf.abs(gt-fake))
    return l1

  def atmospheric_refine_loss_g(self, I, J, a):

    t = self.refine_dark_channel_transmisstion(I, J, a)
    #t = tf.clip_by_value(t, 0.1, 1)
    restruct_I = 2*(tf.multiply(0.5*(J-a), t)+0.5*a+0.5)-1 #(tf.multiply(J - a, t) + a - J)/2

    #restruct_I = tf.clip_by_value(restruct_I, -1, 1)
    atmospheric_loss = tf.reduce_mean(tf.abs(I-restruct_I))

    return tf.clip_by_value(t, -1, 1), tf.clip_by_value(restruct_I, -1, 1), atmospheric_loss

  def atmospheric_refine_loss_f(self, J, I, a):

    t = self.refine_dark_channel_transmisstion(I, J, a)
    #t = tf.clip_by_value(t, 0.1, 1)
    #omg = tf.get_variable(
    #  'omgf', [], initializer=tf.constant_initializer(0.95))
    #t = -(ti - tj)

    restruct_J = 2*tf.divide(0.5*I+0.5-tf.multiply(1-t,0.5*a+0.5),t)-1#tf.div(I-a,self.protect_value(t)) + a#tf.div(2 * I + a * (t - 1),self.protect_value(t + 1))
    #restruct_J = tf.clip_by_value(restruct_J, -1, 1)
    atmospheric_loss = tf.reduce_mean(tf.abs(J-restruct_J))
    return  tf.clip_by_value(t, -1, 1), tf.clip_by_value(restruct_J, -1, 1), atmospheric_loss

  def only_limited_foreground_loss_g(self, I, J, a):
    t = self.dark_channel_foreground_transmission(I, a)
    t = tf.clip_by_value(t, 0.1, 1)
    restruction_I = 2 * (tf.multiply(0.5*J+0.5,t) + tf.multiply(1-t, 0.5*a+0.5))-1
    # min1=min(min(T));
    # max1=max(max(T));
    # a=20/(max1-min1);
    # b=-10-a*min1;
    # alpht=1./(1+exp(-a.*T-b));
    confident_map = 0.8*t
    dark_channel_loss = tf.reduce_mean(tf.multiply(confident_map,tf.squared_difference(restruction_I,I)))
    return t, restruction_I, dark_channel_loss

  def only_limited_foreground_loss_f(self, J, I, a):
    t = self.dark_channel_foreground_transmission(I, a)
    t = tf.clip_by_value(t, 0.1, 1)
    restruction_J = 2*tf.div(0.5*I+0.5-tf.multiply(1-t,0.5*a+0.5),t)-1
    # min1=min(min(T));
    # max1=max(max(T));
    # a=20/(max1-min1);
    # b=-10-a*min1;
    # alpht=1./(1+exp(-a.*T-b));
    confident_map = 0.8*t
    dark_channel_loss = tf.reduce_mean(tf.multiply(confident_map,tf.squared_difference(restruction_J,I)))
    return t, restruction_J, dark_channel_loss

  def refine_dark_channel_transmisstion(self, I, J, a):
    r = 4
    eps = 1e-6
    nhwc = True
    fake_dm, real_dm = self.refine_dark_channel_transmission_with_a(I, J, a)
    fake_dm = tf.concat([fake_dm, fake_dm, fake_dm], axis=-1)
    real_dm = tf.concat([real_dm, real_dm, real_dm],axis=-1)
    ti = guided_filter(I, fake_dm, r, eps, nhwc)
    tj = guided_filter(J, real_dm, r, eps, nhwc)

    ti_clip = tf.expand_dims(tf.reduce_mean(ti, axis=-1), axis=-1)
    ti = tf.concat([ti_clip, ti_clip, ti_clip], axis=-1)
    tj_clip = tf.expand_dims(tf.reduce_mean(tj, axis=-1), axis=-1)
    tj = tf.concat([tj_clip, tj_clip, tj_clip], axis=-1)
    t = tf.divide(ti - 1, self.protect_value(tj - 1))
    #tf.div((2 * ti - tj - 1),self.protect_value(tj - 1))
    #t = tf.clip_by_value(t, -1, 1)
    #t = tf.nn.softmax(t, axis=-1) + 0.01
    #t = self.protect_value(t)
    return t


  def refine_dark_channel_transmission_with_a(self, I, J, a):
    a = tf.clip_by_value(self.protect_value(a), -1, 1)
    J = tf.clip_by_value(self.protect_value(J), -1, 1)
    I = tf.clip_by_value(self.protect_value(I), -1, 1)
    I_dm= ops.dark_channel(tf.divide(0.5*I+0.5, 0.5*a+0.5))
    J_dm = ops.dark_channel(tf.divide(0.5*J+0.5, 0.5*a+0.5))
    return I_dm, J_dm

  def dark_channel_foreground_transmission(self, I, a):
    #a = tf.clip_by_value(self.protect_value(a), -1, 1)
    I_dm= ops.dark_channel(tf.divide(0.5*I+0.5, 0.5*a+0.5))
    t = 1 - 0.95*I_dm
    fake_dm = tf.concat([t, t, t], axis=-1)
    r = 4
    eps = 1e-6
    nhwc = True
    t = guided_filter(I, fake_dm, r, eps, nhwc)
    return t

  def dark_channel_loss(self, res_x, res_y, real_x, real_y):
    forward_dark_loss = tf.reduce_mean(tf.abs(real_x - res_x))
    backward_dark_loss = tf.reduce_mean(tf.abs(real_y - res_y))
    dark_loss = 10 * forward_dark_loss + 10 * backward_dark_loss
    return dark_loss

  def protect_value(self, t):
    signt = tf.sign(t)
    abst = tf.abs(t)
    #clipt = tf.clip_by_value(abst, 0.01, 1)
    t = tf.where(signt != 0, abst * signt, abst + 0.01)
    return t

  def estimated_t(self,I,J):
    Ib, Ig, Ir = tf.split(I, [1, 1, 1], -1)
    Jb, Jg, Jr = tf.split(J, [1, 1, 1], -1)
    t = tf.divide(0.5*(Ir-Ig)+0.5,0.5*self.protect_value(Jr-Jg)+0.5)
    t = tf.clip_by_value(t, 0.1, 1)
    return t

  def estimated_a(self, I,J,t):
    A = tf.divide(0.5*(I-J)+0.5,self.protect_value(1-t))-(0.5*J+0.5)
    A = tf.clip_by_value(A, -1, 1)
    return A

  def restruct_phscial_model(self, I, J):
    t = self.estimated_t(I, J)
    a = self.estimated_a(I, J, t)
    loss = tf.reduce_mean(tf.abs(tf.multiply((0.5*J+0.5), t)+tf.multiply((1-t), (0.5*a+0.5)) - (0.5*I+0.5)))
    return loss, t, a

