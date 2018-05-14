import tensorflow as tf
from nets import inception, resnet_v1, resnet_v2, resnet_utils, vgg, mobilenet_v1
from preprocessing.vgg_preprocessing import preprocess_image as vgg_preprocess
from preprocessing.inception_preprocessing import preprocess_image as inception_preprocess
import numpy as np
import tf_jpeg_utils
import denoisers

slim = tf.contrib.slim

class ResizeAndPadding(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'ResizeAndPadding'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                x_input = denoisers.resize_and_padding_layer(x_input)
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)
            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class BitDepth(object):
    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'BitDepth'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                x_input = denoisers.bit_depth_reduction(x_input, step_num=4, alpha=1e4)
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)
            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class TVM(object):
    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'TVM'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                x_input = denoisers.tvm_layer(x_input, tv_weight=0.2, stepsize=0.05, iternum=20, smooth=False)
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)
            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits


class JPEG(object):
    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'JPEG'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                x_input = tf_jpeg_utils.jpeg_compress_decompress(x_input, soft_rounding=False)
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)
            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class Gaussian(object):
    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'Gaussian'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                x_input = x_input + 0.25 * tf.random_normal(tf.shape(x_input))
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)
            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class RandMix(object):
    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'RandMix'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                x_input = denoisers.iterative_clustering_layer(source=x_input, n_clusters=20, sigma=10, alpha=10, noise_level_1=0.125, noise_level_2=0.125)
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)
            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

class RandDisc(object):
    def __init__(self, num_classes, batch_size=None):
        self.num_classes = num_classes
        self.built = False
        self.logits = None
        self.ckpt = 'ens_adv_inception_resnet_v2.ckpt'
        self.name = 'RandDisc'

    def __call__(self, x_input, batch_size=None, is_training=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            with tf.variable_scope(self.name):
                x_input = denoisers.iterative_clustering_layer(source=x_input, n_clusters=20, sigma=10, alpha=100, noise_level_1=0.125, noise_level_2=0.125)
                logits, end_points = inception.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=is_training,
                    reuse=reuse)
            preds = tf.argmax(logits, axis=1)
        self.built = True
        self.logits = logits
        self.preds = preds
        return logits

