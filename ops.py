# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Tensorflow ops used by GAN.

"""

import tensorflow as tf
import numpy as np
import logging

def lrelu(x, leak=0.3):
    return tf.maximum(x, leak * x)

def batch_norm(opts, _input, is_train, reuse, scope, scale=True):
    """Batch normalization based on tf.contrib.layers.

    """
    return tf.contrib.layers.batch_norm(
        _input, center=True, scale=scale,
        epsilon=opts['batch_norm_eps'], decay=opts['batch_norm_decay'],
        is_training=is_train, reuse=reuse, updates_collections=None,
        scope=scope, fused=False)

def upsample_nn(input_, new_size, scope=None, reuse=None):
    """NN up-sampling
    """

    with tf.variable_scope(scope or "upsample_nn", reuse=reuse):
        result = tf.image.resize_nearest_neighbor(input_, new_size)

    return result

def downsample(input_, d_h=2, d_w=2, conv_filters_dim=None, scope=None, reuse=None):
    """NN up-sampling
    """

    with tf.variable_scope(scope or "downsample", reuse=reuse):
        result = tf.nn.max_pool(input_, ksize=[1, d_h, d_w, 1], strides=[1, d_h, d_w, 1], padding='SAME')

    return result

def linear(opts, input_, output_dim, scope=None, reuse=None):

    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()

    assert len(shape) > 0
    in_shape = shape[1]
    if len(shape) > 2:

        input_ = tf.reshape(input_, [-1, np.prod(shape[1:])])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope or "lin", reuse=reuse):

        matrix = tf.get_variable(
                "W", [in_shape, output_dim], tf.float32,
                tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable(
                "b", [output_dim],
                initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_, matrix) + bias

def auxilinear(opts, input_, output_dim, auxi_weights, scope=None, reuse=None):

    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    auxi_shape = auxi_weights.get_shape().as_list()
    auxi_dim = auxi_shape[1]

    assert len(shape) > 0
    in_shape = shape[1]
    if len(shape) > 2:

        input_ = tf.reshape(input_, [-1, np.prod(shape[1:])])
        in_shape = np.prod(shape[1:])

    with tf.variable_scope(scope or "lin", reuse=reuse):
        if in_shape - auxi_dim:
            matrix = tf.get_variable(
                    "W", [in_shape - auxi_dim, output_dim], tf.float32,
                    tf.truncated_normal_initializer(stddev=stddev))
            bias = tf.get_variable(
                    "b", [output_dim],
                    initializer=tf.constant_initializer(bias_start))
            input_main = input_[:,:in_shape - auxi_dim]
            input_auxi = tf.expand_dims(input_[:,in_shape - auxi_dim:], 1)
            output = tf.matmul(input_main, matrix) + tf.squeeze(tf.matmul(input_auxi, auxi_weights), [1]) + bias
        else:
            bias = tf.get_variable(
                    "b", [output_dim],
                    initializer=tf.constant_initializer(bias_start))
            input_auxi = tf.expand_dims(input_[:,in_shape - auxi_dim:], 1)
            output = tf.squeeze(tf.matmul(input_auxi, auxi_weights), [1]) + bias

    return output

def t_network(opts, input_, auxi_weights_info, layer_num, keep_prob, reuse):
    
    t_info = opts['t_info']
    num_of_T = int(auxi_weights_info / (10 * opts['main_info'][layer_num]))
    with tf.variable_scope('t_network/t_network_%d' % layer_num):
        for t in range(num_of_T):
            h_prev = input_
            for l in range(len(t_info)):
                h_prev = linear(opts, h_prev, t_info[l], scope='t_%d_%d_%d' % (layer_num, t+1, l), reuse=reuse)
                h_prev = tf.nn.dropout(h_prev, keep_prob)
                h_prev = tf.nn.tanh(h_prev)
            h_prev = linear(opts, h_prev, 10 * opts['main_info'][layer_num], scope='t_%d_%d_%d' % (layer_num, t+1, len(t_info)), reuse=reuse)
            h_prev = tf.nn.tanh(h_prev)
            
            if t == 0:
                auxi_weights = h_prev
            else:
                auxi_weights = tf.concat([auxi_weights, h_prev], 0)
        
        return auxi_weights
    
def conv2d(opts, input_, output_dim, d_h=2, d_w=2, scope=None,
           conv_filters_dim=None, padding='SAME', l2_norm=False):
    """Convolutional layer.

    Args:
        input_: should be a 4d tensor with [num_points, dim1, dim2, dim3].

    """

    stddev = opts['init_std']
    bias_start = opts['init_bias']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d works only with 4d tensors.'

    with tf.variable_scope(scope or 'conv2d'):
        w = tf.get_variable(
            'filter', [k_h, k_w, shape[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if l2_norm:
            w = tf.nn.l2_normalize(w, 2)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_dim],
            initializer=tf.constant_initializer(bias_start))
        conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(opts, input_, output_shape, d_h=2, d_w=2, scope=None, conv_filters_dim=None, padding='SAME'):
    """Transposed convolution (fractional stride convolution) layer.

    """

    stddev = opts['init_std']
    shape = input_.get_shape().as_list()
    if conv_filters_dim is None:
        conv_filters_dim = opts['conv_filters_dim']
    k_h = conv_filters_dim
    k_w = k_h

    assert len(shape) == 4, 'Conv2d_transpose works only with 4d tensors.'
    assert len(output_shape) == 4, 'outut_shape should be 4dimensional'

    with tf.variable_scope(scope or "deconv2d"):
        w = tf.get_variable(
            'filter', [k_h, k_w, output_shape[-1], shape[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape,
            strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'b', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)


    return deconv


def log_sum_exp(logits):
    l_max = tf.reduce_max(logits, axis=1, keep_dims=True)
    return tf.add(l_max,
                  tf.reduce_sum(
                    tf.exp(tf.subtract(
                        logits,
                        tf.tile(l_max, tf.stack([1, logits.get_shape()[1]])))),
                    axis=1))
