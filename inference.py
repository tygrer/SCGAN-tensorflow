"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --checkpoint 2021298434 \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size_inference 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
import build_data
import export_graph
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint', '', '')
tf.flags.DEFINE_string('input_dir', '/raid/tanggaoyang/test_images/', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', '', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size_inference', '256', 'image size, default: 256')

def inference():
  test_record_name = "./test_image.tfrecords"
  model_name = os.path.basename(FLAGS.checkpoint) + ".pb"
  export_graph.export_graph(model_name,XtoY=True)
  test_file_paths = build_data.data_reader(FLAGS.input_dir, test_record_name)
  model_path = "./pretrained/" + model_name
  build_data.data_writer(FLAGS.input_dir, test_record_name)
  graph = tf.Graph()

  with graph.as_default():
    reader = tf.TFRecordReader()
    file_queue = tf.train.string_input_producer([test_record_name])
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
        'image/file_name': tf.FixedLenFeature([], tf.string),
        'image/encoded_image': tf.FixedLenFeature([], tf.string),
      })

    image_buffer = features['image/encoded_image']
    name_buffer = features['image/file_name']
    input_image = tf.image.decode_jpeg(image_buffer, channels=3)
    input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
    input_image = utils.convert2float(input_image)
    input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
    # input_images = tf.train.batch(
    #   [input_image], batch_size=1, num_threads=8,
    #   capacity=1000 + 3 * 1
    # )

    with tf.gfile.FastGFile(model_path, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_images] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='output')
  with tf.Session(graph=graph) as sess:
    generated = output_images.eval()
  if not os.path.exists(
          os.path.join(os.path.expanduser('.'), os.path.basename(FLAGS.checkpoint),'output')):
    os.makedirs(os.path.join(os.path.expanduser('.'), os.path.basename(FLAGS.checkpoint),'output'))
  if len(test_file_paths) == output_images.shape[0]:
    for test_i in test_file_paths:
      print(test_i)
      #imo = os.path.join(os.path.expanduser('.'), 'output', os.path.basename(test_i))
      with open(test_i, 'wb') as f:
        f.write(generated[test_i,:,:,:])

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
