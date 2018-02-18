import tensorflow as tf
import numpy as np
import math, random

class MaskedLayer(object):
    placeholder_ws = None
    placeholder_pinv_ws = None
    placeholder_avg_project = None

    def update_mask(self, sess, alpha=0.99):
        if self.placeholder_ws is None: return
        weight = sess.run(self.W)
        indices = np.random.randint(0, weight.shape[1], self.s)
        ws = weight[:, indices]
        self.ws = ws
        self.pinv_ws = np.linalg.pinv(ws)
        if random.random() < 0.2:
            self.avg_project = alpha * self.avg_project + (1 - alpha) * np.matmul(self.ws, self.pinv_ws)

    def fill_dict(self, dictionary, is_training):
        if self.ws is None: return
        if is_training:
            dictionary[self.placeholder_ws] = self.ws
            dictionary[self.placeholder_pinv_ws] = self.pinv_ws
        else:
            dictionary[self.placeholder_avg_project] = self.avg_project

class MaskedConv2d(MaskedLayer):
    def __init__(self, name, image, fout, patch_size, strides, s, padding, is_training):
        self.s = s
        self.placeholder_ws = None
        self.ws = None
        if self.s > 0:
            fin = image.get_shape().as_list()[-1]
            dim = patch_size * patch_size * fin
            self.avg_project = np.zeros([dim, dim])
            with tf.name_scope(name):
                self.placeholder_ws = tf.placeholder_with_default(tf.zeros([dim, s]), shape=[dim, s], name='matrix_ws')
                self.placeholder_pinv_ws = tf.placeholder_with_default(tf.zeros([s, dim]), shape=[s, dim], name='matrix_pinv_ws')
                self.placeholder_avg_project = tf.placeholder_with_default(tf.zeros([dim, dim]), shape=[dim, dim], name='matrix_avg_project')
                self.W = weight_variable([dim, fout])
                self.b = bias_variable([fout])

            def train_func():
                W4 = tf.reshape(self.W, [patch_size, patch_size, fin, fout])
                ws4 = tf.reshape(self.placeholder_ws, [patch_size, patch_size, fin, s])
                projection = conv2d_with_weight(image, ws4, strides, padding)
                projection = tf.tensordot(projection, tf.matmul(self.placeholder_pinv_ws, self.W), axes=[[3], [0]])
                return conv2d_with_weight(image, W4, strides, padding) - projection + self.b

            def test_func():
                W4 = tf.reshape(self.W - tf.matmul(self.placeholder_avg_project, self.W), [patch_size, patch_size, fin, fout])
                return conv2d_with_weight(image, W4, strides, padding) + self.b

            self.output = tf.cond(is_training, train_func, test_func)
        else:
            with tf.name_scope(name):
                self.output = conv2d(image, patch_size=patch_size, fout=fout, strides=strides, padding=padding)

class MaskedFC(MaskedLayer):
    def __init__(self, name, image, fout, s, is_training):
        self.s = s
        self.placeholder_ws = None
        self.ws = None
        shape = image.get_shape().as_list()
        fin = 1
        for i in range(1, len(shape)):
            fin *= shape[i]
        image = tf.reshape(image, [-1, fin])
        if self.s > 0:
            self.avg_project = np.zeros([fin, fin])
            with tf.name_scope(name):
                self.placeholder_ws = tf.placeholder_with_default(tf.zeros([fin, s]), shape=[fin, s], name='matrix_ws')
                self.placeholder_pinv_ws = tf.placeholder_with_default(tf.zeros([s, fin]), shape=[s, fin], name='matrix_pinv_ws')
                self.placeholder_avg_project = tf.placeholder_with_default(tf.zeros([fin, fin]), shape=[fin, fin], name='matrix_avg_project')
                self.W = weight_variable([fin, fout])
                self.b = bias_variable([fout])

            def train_func():
                projection = tf.matmul(tf.matmul(tf.matmul(image, self.placeholder_ws), self.placeholder_pinv_ws), self.W)
                return tf.matmul(image, self.W) - projection + self.b

            def test_func():
                return tf.matmul(image - tf.matmul(image, self.placeholder_avg_project), self.W) + self.b

            self.output = tf.cond(is_training, train_func, test_func)
        else:
            with tf.name_scope(name):
                W = weight_variable([fin, fout])
                b = bias_variable([fout])
                self.output = tf.matmul(image, W) + b

def weight_variable(shape, scale=2.0):
    fin = 1
    for i in range(0, len(shape) - 1):
        fin *= shape[i]
    initial = tf.truncated_normal(shape, stddev=math.sqrt(scale / fin), name='weight')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape, name='bias')
    return tf.Variable(initial)

def conv2d(x, patch_size, fout, strides, padding):
    fin = x.get_shape().as_list()[-1]
    W = weight_variable([patch_size, patch_size, fin, fout])
    b = bias_variable([fout])
    return tf.nn.conv2d(x, W, strides=strides, padding=padding) + b

def conv2d_with_weight(x, W, strides, padding):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

