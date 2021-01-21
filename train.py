import tensorflow as tf
from model import CycleGAN
from reader import Reader
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 128, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-3,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.8,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', 'data/haze2000.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', 'data/clear2000.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('X_pair', 'align/haze.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y_pair', 'align/clear.tfrecords',
                       'Y tfrecords file for training, default: data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', None,#"20191112-2150",
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')#20191111-2035


def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        X_train_file=FLAGS.X,
        Y_train_file=FLAGS.Y,
        X_pair_train_file=FLAGS.X_pair,
        Y_pair_train_file=FLAGS.Y_pair,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf
    )
    G_loss, D_Y_loss, F_loss, D_X_loss, dark_channel_loss, cycle_guided_loss,\
    cycle_loss, G_gan_loss, F_gan_loss, G_l1_loss, F_l1_loss, fake_x, fake_y = cycle_gan.model()
    optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      print(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop():
        # get previously generated images
        fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

        # train
        _,G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, \
        dark_channel_loss_val, cycle_guided_loss_val,\
    cycle_loss_val, G_gan_loss_val, F_gan_loss_val, G_l1_loss_val, F_l1_loss_val, summary = (
              sess.run(
                  [optimizers,G_loss, D_Y_loss, F_loss, D_X_loss,
                   dark_channel_loss, cycle_guided_loss,
    cycle_loss, G_gan_loss, F_gan_loss, G_l1_loss, F_l1_loss, summary_op],
                  feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                             cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
              )
        )



        if step % 100 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  G_loss   : {}'.format(G_loss_val))
          logging.info('  D_G_loss : {}'.format(D_Y_loss_val))
          logging.info('  F_loss   : {}'.format(F_loss_val))
          logging.info('  D_F_loss : {}'.format(D_X_loss_val))
          logging.info('  dark_channel_loss : {}'.format(dark_channel_loss_val))
          logging.info('  cycle_guided_loss : {}'.format(cycle_guided_loss_val))
          logging.info('  cycle_loss : {}'.format(cycle_loss_val))
          logging.info('  G_gan_loss : {}'.format(G_gan_loss_val))
          logging.info('  F_gan_loss : {}'.format(F_gan_loss_val))
          logging.info('  G_l1_loss : {}'.format(G_l1_loss_val))
          logging.info('  F_l1_loss : {}'.format(F_l1_loss_val))
        if step % 1000 == 0:
            train_writer.add_summary(summary, step)
            train_writer.flush()
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)

        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

# def test():
#     coord = tf.train.Coordinator()
#     gentor = Generator('G', False, 64, 'instance', FLAGS.image_size)
#     saver.restore(sess, ckpt_path)
#     # Set up tf session and initialize variables.
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     init = tf.global_variables_initializer()
#
#     sess.run(init)
#     sess.run(tf.local_variables_initializer())
#     checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
#     print(checkpoints_dir)
#     meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
#     restore = tf.train.import_meta_graph(meta_graph_path)
#     restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
#     threads = tf.train.start_queue_runners(coord=coord, sess=sess)
#     # Iterate over training steps.
#     feed = {:}
#     preds, _ = sess.run([pred, update_op],feed_dict=feed)
#     coord.request_stop()
#     coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
