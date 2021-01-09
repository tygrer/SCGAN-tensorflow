import tensorflow as tf

import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator
from guided_filter import guided_filter
from ops import *
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
               ngf=64,
               gan_type='lsgan'
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
    self.gan_type = gan_type
    with tf.device('/device:GPU:2'):
      self.G = Generator('Generator_X', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    with tf.device('/device:GPU:3'):
      self.D_Y = Discriminator('Discriminator_Y',self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    with tf.device('/device:GPU:4'):
      self.F = Generator('Generator_Y', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    with tf.device('/device:GPU:5'):
      self.D_X = Discriminator('Discriminator_X',
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
    x_ab, in_dark_ab, out_dark_ab, Aab, cam_ab, heatmap_ab = self.G(x)  # real a
    x_ba, in_dark_ba, out_dark_ba, Aba, cam_ba, heatmap_ba = self.F(y)  # real b
    x_ab_pair, in_dark_ab_pair, out_dark_ab_pair, Aab_pair, cam_ab_pair, heatmap_ab_pair = self.G(x_pair)  # real a
    x_aba, in_dark_aba, out_dark_aba, Aaba, cam_aba, heatmap_aba = self.F(x_ab)  # real b
    x_bab, in_dark_bab, out_dark_bab, Abab, cam_bab, heatmap_bab = self.G(x_ba)  # real a
    x_ba_pair, in_dark_ba_pair, out_dark_ba_pair, Aba_pair, cam_ba_pair, heatmap_ba_pair = self.F(y_pair)# real b

    x_aa, in_dark_aa, out_dark_aa, Aaa, cam_aa, heatmap_aa = self.F(x)  # fake b
    x_bb, in_dark_bb, out_dark_bb, Abb, cam_bb, heatmap_bb = self.G(y)   # fake a

    real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit = self.discriminate_real(x,
                                                                                            y)
    fake_A_logit, fake_A_cam_logit, fake_B_logit, fake_B_cam_logit = self.discriminate_fake(x_ba, x_ab)

    """ Define Loss """
    if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
      GP_A, GP_CAM_A = self.gradient_panalty(real=x, fake=x_ba, scope="discriminator_A")
      GP_B, GP_CAM_B = self.gradient_panalty(real=y, fake=x_ab, scope="discriminator_B")
    else:
      GP_A, GP_CAM_A = 0, 0
      GP_B, GP_CAM_B = 0, 0

    G_ad_loss_A = (generator_loss(self.gan_type, fake_A_logit) + generator_loss(self.gan_type, fake_A_cam_logit))
    G_ad_loss_B = (generator_loss(self.gan_type, fake_B_logit) + generator_loss(self.gan_type, fake_B_cam_logit))

    D_ad_loss_A = (discriminator_loss(self.gan_type, real_A_logit, fake_A_logit) + discriminator_loss(self.gan_type,
                                                                                                      real_A_cam_logit,
                                                                                                      fake_A_cam_logit) + GP_A + GP_CAM_A)
    D_ad_loss_B = (discriminator_loss(self.gan_type, real_B_logit, fake_B_logit) + discriminator_loss(self.gan_type,
                                                                                                      real_B_cam_logit,
                                                                                                      fake_B_cam_logit) + GP_B + GP_CAM_B)
    l1_A = L1_loss(x_ab_pair, y_pair)
    l1_B = L1_loss(x_ba_pair, x_pair)
    reconstruction_A = L1_loss(x_aba, x)  # reconstruction
    reconstruction_B = L1_loss(x_bab, y)  # reconstruction

    identity_A = L1_loss(x_aa, x)
    identity_B = L1_loss(x_bb, y)

    cam_A = cam_loss(source=cam_ba, non_source=cam_aa)
    cam_B = cam_loss(source=cam_ab, non_source=cam_bb)

    Generator_A_gan = G_ad_loss_A
    Generator_A_cycle = 10 * reconstruction_B
    Generator_A_identity = 10 * identity_A
    Generator_A_cam = 10 * cam_A

    Generator_B_gan =  G_ad_loss_B
    Generator_B_cycle = 10 * reconstruction_A
    Generator_B_identity = 10 * identity_B
    Generator_B_cam = 10 * cam_B
    cycle_guided_loss_A, cycle_guided_loss_B = self.guided_filter_consistency_loss(x_aba, x_bab, x, y)
    dark_channel_loss_A, dark_channel_loss_B = self.dark_channel_loss(out_dark_aba, out_dark_bab, in_dark_ab, in_dark_ba)

    self.Generator_A_loss = Generator_A_gan + Generator_A_cycle + Generator_A_identity + Generator_A_cam + \
                       10*cycle_guided_loss_A + 10*dark_channel_loss_A + l1_A
    self.Generator_B_loss = Generator_B_gan + Generator_B_cycle + Generator_B_identity + Generator_B_cam + \
                       10*cycle_guided_loss_B + 10*dark_channel_loss_B + l1_B

    self.Discriminator_A_loss = D_ad_loss_A
    self.Discriminator_B_loss =  D_ad_loss_B


    self.Generator_loss = self.Generator_A_loss + self.Generator_B_loss# + regularization_loss('Generator_')
    self.Discriminator_loss = self.Discriminator_A_loss + self.Discriminator_B_loss #+ regularization_loss('Discriminator_')

    """ Result Image """


    t1_g, t2_g, g_atmospheric_loss = self.atmospheric_refine_loss_g(x, x_ab, Aab)
    #t1_f, t2_f, f_atmospheric_loss_back = self.atmospheric_refine_loss_f(fake_y, restructx, restructy_aestimate)
    foreground_gt, foreground_restruction_gI,foreground_g_loss = self.only_limited_foreground_loss_g(x, x_ab, Aab)
    #foreground_g_loss_back = self.only_limited_foreground_loss_f(fake_y, x, restructy_aestimate)
    only_limit_foreground_g_loss = foreground_g_loss #+ foreground_g_loss_back
    restruct_loss, tt, aa = self.restruct_phscial_model(x, x_aba)
    # Y -> X

    #1_f_1, t2_f_1, f_atmospheric_loss = self.atmospheric_refine_loss_f(y, fake_x, restructx_aestimate)
    #1_g_1, t2_g_1, g_atmospheric_loss_back = self.atmospheric_refine_loss_g(fake_x, restructy, restructx_aestimate)

    foreground_ft, foreground_restruction_fI, foreground_f_loss = self.only_limited_foreground_loss_f(y, x_ba, Aba)
    #foreground_f_loss_back = self.only_limited_foreground_loss_f(fake_x, y, restructx_aestimate)
    only_limit_foreground_f_loss = foreground_f_loss #+ foreground_f_loss_back


    # G_loss =  G_gan_loss + cycle_loss + cycle_guided_loss + G_l1_loss  + dark_channel_loss#+restruct_loss + only_limit_foreground_g_loss #+ g_atmospheric_loss + f_atmospheric_loss_back
    # F_loss = F_gan_loss + cycle_loss + cycle_guided_loss + F_l1_loss + dark_channel_loss #+ f_atmospheric_loss  + g_atmospheric_loss_back

    atmospheric_loss_g = g_atmospheric_loss + restruct_loss + only_limit_foreground_g_loss  #+ f_atmospheric_loss_back
    #atmospheric_loss_f = only_limit_foreground_f_loss #+ g_atmospheric_loss_back
    # summary
    #tf.summary.histogram('D_Y/true', self.D_Y(y,None))
    #tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x),"NO_OPS"))
    #tf.summary.histogram('D_X/true', self.D_X(x,None))
    #tf.summary.histogram('D_X/fake', self.D_X(self.F(y),"NO_OPS"))

    tf.summary.scalar('loss/G', Generator_A_gan)
    tf.summary.scalar('loss/D_Y', self.Discriminator_A_loss)
    tf.summary.scalar('loss/G_L', l1_A)
    tf.summary.scalar('loss/G_identity', Generator_A_identity)
    tf.summary.scalar('loss/G_cam', Generator_A_cam)
    tf.summary.scalar('loss/F', Generator_B_gan)
    tf.summary.scalar('loss/D_X', self.Discriminator_B_loss)
    tf.summary.scalar('loss/F_L', l1_B)
    tf.summary.scalar('loss/F_identity', Generator_B_identity)
    tf.summary.scalar('loss/F_cam', Generator_B_cam)
    tf.summary.scalar('loss/cycle_A', Generator_A_cycle)
    tf.summary.scalar('loss/cycle_B', Generator_B_cycle)
    tf.summary.scalar('loss/cycle_guided_A', cycle_guided_loss_A)
    tf.summary.scalar('loss/cycle_guided_B', cycle_guided_loss_B)
    tf.summary.scalar('loss/g_atmospheric_loss', g_atmospheric_loss)
    #tf.summary.scalar('loss/f_atmospheric_loss', f_atmospheric_loss)
    tf.summary.scalar('loss/dark_channel_loss_A', dark_channel_loss_A)
    tf.summary.scalar('loss/dark_channel_loss_B', dark_channel_loss_B)
    tf.summary.scalar('loss/only_limit_foreground_g_loss', only_limit_foreground_g_loss)
    tf.summary.scalar('loss/only_limit_foreground_f_loss', only_limit_foreground_f_loss)
    tf.summary.scalar('loss/restruct_loss', restruct_loss)

    tf.summary.image('X/darkmap', utils.batch_convert2int(in_dark_ab))
    tf.summary.image('Y/gen_darkmap', utils.batch_convert2int(out_dark_ab))
    tf.summary.image('X/generated', utils.batch_convert2int(x_ab))
    tf.summary.image('X/restru_darkmap', utils.batch_convert2int(in_dark_aba))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(x_aba))
    tf.summary.image('X/fake_refine_x', utils.batch_convert2int(self.fake_refinex))
    tf.summary.image('X/t1', utils.batch_convert2int((t1_g)))
    tf.summary.image('X/t2', utils.batch_convert2int((t2_g)))
    tf.summary.image('X/A', utils.batch_convert2int((Aab)))
    tf.summary.image('X/restruct_t',utils.batch_convert2int((tt)))
    tf.summary.image('X/restruct_A',utils.batch_convert2int(aa))
    tf.summary.image('X/pair', utils.batch_convert2int(x_pair))
    tf.summary.image('X/forgroundgt', utils.batch_convert2int(foreground_gt))
    tf.summary.image('X/forgroundg_RI_t', utils.batch_convert2int(foreground_restruction_gI))
    tf.summary.image('Y/pair', utils.batch_convert2int(y_pair))
    tf.summary.image('Y/darkmap', utils.batch_convert2int(in_dark_ba))
    tf.summary.image('X/gen_darkmap', utils.batch_convert2int(out_dark_ba))
    tf.summary.image('Y/generated', utils.batch_convert2int(x_ba))
    tf.summary.image('Y/restru_darkmap', utils.batch_convert2int(out_dark_bab))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(x_bab))
    tf.summary.image('Y/fake_refine_Y', utils.batch_convert2int(self.fake_refiney))
    #tf.summary.image('Y/t1', utils.batch_convert2int((t1_f_1)))
    #tf.summary.image('Y/t2', utils.batch_convert2int((t2_f_1)))
    tf.summary.image('Y/A', utils.batch_convert2int((Aba)))
    tf.summary.image('Y/forgroundft', utils.batch_convert2int(foreground_ft))
    tf.summary.image('Y/forgroundg_RI_t', utils.batch_convert2int(foreground_restruction_fI))
    self.get_vars()
    print('sigma_ratio_vars', self.sigma_ratio_vars)
    for var in self.sigma_ratio_vars:
       tf.summary.scalar(var.name, var)
    return self.Generator_loss,  self.Discriminator_loss ,x_ab,x_ba, \
           atmospheric_loss_g,only_limit_foreground_g_loss,only_limit_foreground_f_loss


  def optimize(self, G_loss, D_loss, atmospheric_loss_g):
    #t_vars = tf.trainable_variables()
    #G_vars = [var for var in t_vars if 'Generator_' in var.name]
    #D_vars = [var for var in t_vars if 'Discriminator_' in var.name]
    def make_optimizer_G(loss, variables, atmospheric_loss=None, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 1000
      decay_steps = 5000
      start_atmospheric_step = 90000000000
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
        learning_step = (tf.cond(tf.greater_equal(global_step, start_atmospheric_step),
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
      start_decay_step = 1000
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
      learning_step = (
        tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
          .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    D_x_optimizer  = make_optimizer_D(self.Discriminator_A_loss, self.D_X.variables,  name='Adam_D_X')
    D_y_optimizer = make_optimizer_D(self.Discriminator_B_loss, self.D_Y.variables, name='Adam_D_Y')
    G_optimizer = make_optimizer_G(self.Generator_A_loss, self.G.variables, None, name='Adam_G_Y')
    F_optimizer = make_optimizer_G(self.Generator_B_loss, self.F.variables, None, name='Adam_F_X')


  #F_optimizer2 = make_optimizer(F_loss, self.F.variables, name='Adam_F')

    #with tf.control_dependencies([G_optimizer1, D_Y_optimizer, F_optimizer1, D_X_optimizer, G_optimizer2, F_optimizer2]):
    with tf.control_dependencies(
         [G_optimizer, F_optimizer, D_x_optimizer, D_y_optimizer]):
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
      error_real = -tf.reduce_mean(safe_log(D(y,None)))
      error_fake = -tf.reduce_mean(safe_log(1-D(fake_y,"NO_OPS")))
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
      loss = -tf.reduce_mean(safe_log(D(fake_y,"NO_OPS"))) / 2
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
    return forward_loss, backward_loss

  def pair_l1_loss(self, fake, gt):
    l1 = tf.reduce_mean(tf.abs(gt-fake))*5
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
    I_dm= dark_channel(tf.divide(0.5*I+0.5, 0.5*a+0.5))
    J_dm = dark_channel(tf.divide(0.5*J+0.5, 0.5*a+0.5))
    return I_dm, J_dm

  def dark_channel_foreground_transmission(self, I, a):
    #a = tf.clip_by_value(self.protect_value(a), -1, 1)
    I_dm= dark_channel(tf.divide(0.5*I+0.5, 0.5*a+0.5))
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
    #dark_loss = 10 * forward_dark_loss + 10 * backward_dark_loss
    return forward_dark_loss, backward_dark_loss

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

  def gradient_panalty(self, real, fake, scope="discriminator_A"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit, cam_logit, _, _ = self.discriminator(interpolated, reuse=True, scope=scope)


        GP = []
        cam_GP = []

        for i in range(2) :
            grad = tf.gradients(logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))

        for i in range(2) :
            grad = tf.gradients(cam_logit[i], interpolated)[0] # gradient of D(interpolated)
            grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

            # WGAN - LP
            if self.gan_type == 'wgan-lp' :
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.))))

            elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
                cam_GP.append(self.ld * tf.reduce_mean(tf.square(grad_norm - 1.)))


        return sum(GP), sum(cam_GP)

  def discriminate_real(self, x_A, x_B):
    real_A_logit, real_A_cam_logit, _, _ = self.D_X(x_A)
    real_B_logit, real_B_cam_logit, _, _ = self.D_Y(x_B)

    return real_A_logit, real_A_cam_logit, real_B_logit, real_B_cam_logit

  def discriminate_fake(self, x_ba, x_ab):
    fake_A_logit, fake_A_cam_logit, _, _ = self.D_X(x_ba)
    fake_B_logit, fake_B_cam_logit, _, _ = self.D_Y(x_ab)

    return fake_A_logit, fake_A_cam_logit, fake_B_logit, fake_B_cam_logit