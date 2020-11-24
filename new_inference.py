import tensorflow as tf
import os
from model import CycleGAN
import utils
import build_data
import datetime
import cv2
from PIL import Image
import numpy as np
import time
import export_graph
import glob
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint', '', '')
tf.flags.DEFINE_string('input_dir', '/raid/tanggaoyang/test_images/', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', '', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size_inference', '128', 'image size, default: 256')


# In[20]:
test_record_name = "./test_image.tfrecords"
model_name = os.path.basename(FLAGS.checkpoint) + ".pb"
export_graph.export_graph(model_name, "./checkpoints/"+FLAGS.checkpoint, XtoY=True)
print("success to export graph")
test_file_paths = build_data.data_reader(FLAGS.input_dir, test_record_name)
model_path = "./pretrained/" + model_name
# build_data.data_writer(FLAGS.input_dir, test_record_name)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:  # 加载模型
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# In[21]:

test_img_base_path = FLAGS.input_dir
imgs_files = os.path.join(test_img_base_path, "*")  # 测试验证图片的路径
imgs_list = glob.glob(imgs_files)
num_imgs = len(imgs_list)
print("Images num:" + str(num_imgs))
inference_path = "./inference_result"
new_files = []
if not os.path.exists(inference_path):
    os.mkdir(inference_path)
total_time = 0

# In[22]:

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('input_image:0')  # 获取输入图片的tensor
        prediction = detection_graph.get_tensor_by_name('output_image:0')  # 输出prediction的tensor
        start_time = datetime.datetime.now()
        print("STARTING ...")
        for image_path in imgs_list:
            print(image_path)
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_np = cv2.resize(image_np,(FLAGS.image_size_inference ,FLAGS.image_size_inference))
            image_np = np.array(image_np).astype('float32')
            print(len(image_np))
            #image_np_expanded = np.expand_dims(image_np, axis=0)
            # 图片的处理，每次预测一张图，batch_size=1，当然也可以一次预测多张图片
            # Definite input and output Tensors for detection_graph
            out_name = os.path.join(inference_path, image_path.split("/")[-1])
            time1 = time.time()
            prediction_out = sess.run(
                prediction, feed_dict={image_tensor: image_np})  # 运行一次模型
            time2 = time.time()
            total_time += float(time2 - time1)

            print(len(prediction_out))

            result = Image.fromarray(np.array(prediction_out).astype(np.uint8))
            # 由于本例是图像分割模型，输出也是图片，将prediction直接转为array格式保存即可
            result.save(out_name, quality=100)
        end_time = datetime.datetime.now()

        print("START TIME :" + str(start_time))
        print("END TIME :" + str(end_time))
        print("THE TOTAL TIME COST IS:" + str(total_time))
        print("THE average TIME COST IS:" + str(float(total_time) / float(num_imgs)))