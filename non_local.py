# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import ops

def conv1x1(input_, output_dim,
            init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  k_h = 1
  k_w = 1
  d_h = 1
  d_w = 1
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    #w_bar = ops.spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_non_local_block_sim_self(x, update_collection, name, reuse=False, init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name,reuse=reuse):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta_self')
    theta = tf.reshape(
        theta, [batch_size, location_num, num_channels // 8])

    # phi path
    phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi_self')
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
    phi = tf.reshape(
        phi, [batch_size, downsampled_num, num_channels // 8])


    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    print(tf.reduce_sum(attn, axis=-1))

    # g path
    g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g_self')
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = tf.reshape(
      g, [batch_size, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn_self')
    return x + sigma * attn_g, attn_g

def sn_non_local_block_sim_attention(x, y, update_collection, name, reuse=False, init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name, reuse=reuse):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num
    '''
    theta = sn_conv1x1(x, num_channels , update_collection, init, 'sn_conv_theta')
    theta = tf.reshape(
      theta, [batch_size, location_num, num_channels])

    # phi path
    phi = sn_conv1x1(y, num_channels, update_collection, init, 'sn_conv_phi')
    phi = tf.reshape(
      phi, [batch_size, downsampled_num, num_channels])

    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    print(tf.reduce_sum(attn, axis=-1))
    
    # g path
    g = sn_conv1x1(x, num_channels, update_collection, init, 'sn_conv_g')
    #g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
    g = tf.reshape(
      g, [batch_size, downsampled_num, num_channels])
    
    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels])
    '''
    #y = tf.concat([y,y,y], axis=-1)
    attn = tf.multiply(x,y)
    #sigma = tf.get_variable(
    #  'sigma_ratio_a', [], initializer=tf.constant_initializer(1.0))
    #attn = ops._instance_norm(attn, is_training=True)
    mean, variance = tf.nn.moments(attn, axes=[1, 2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    attn = (attn - mean) * inv
    #attn_g = sn_conv1x1(attn, num_channels, update_collection, init, 'sn_conv_attn')
    #attn = tf.nn.softmax(sigma * attn,axis=-1)+0.01
    #x = tf.clip_by_value(sigma * attn, -1, 1)
    return attn , attn