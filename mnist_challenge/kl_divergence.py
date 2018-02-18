from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm
import tensorflow as tf

def kl_divergence(p1, p2):
    return p1 * np.log(p1 / p2) + (1.0 - p1) * np.log((1.0 - p1) / (1.0 - p2))

def total_variation(p1, p2):
    return 2*np.abs(p1-p2)

def hellinger_distance(p1, p2):
    return 0.5 * (np.square(np.sqrt(p1) - np.sqrt(p2)) + np.square(np.sqrt(1.0 - p1) - np.sqrt(1.0 - p2)))

def func(x):
    return np.maximum(0.0+1e-8, np.minimum(1.0-1e-8, norm.cdf(x/0.15)))

def down_sampling(x, ksize, stride):
    x_input = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x_input, [-1, 28, 28, 1])
    x_output = tf.nn.max_pool(x_image, [1,ksize,ksize,1], [1,stride,stride,1], 'VALID')
    x_output = tf.reshape(x_output, [-1, x_output.get_shape().as_list()[1] * x_output.get_shape().as_list()[2]])

    sess = tf.Session()
    return sess.run(x_output, feed_dict={x_input: x})

def compute_accuracy(x, margin, eps):
    mid = 0.55
    prob0 = func(x - mid)
    prob1 = func(x + eps - mid)
    prob2 = func(x - eps - mid)
    kl_dvg = np.sum(np.maximum(kl_divergence(prob0, prob1), kl_divergence(prob0, prob2)), axis=1)

    # kl_dvg = 784 * (eps / 0.15) ** 2 / 2
    tv_distance = np.sqrt(kl_dvg)

    # pixel_hellinger = np.maximum(hellinger_distance(prob0, prob1), hellinger_distance(prob0, prob2))
    # image_hellinger = 1.0 - np.exp(np.sum(np.log(1.0 - pixel_hellinger), axis=1))
    # tv_distance = 2.0 * np.sqrt(image_hellinger * (2.0 - image_hellinger))

    accuracy = np.sum((margin > tv_distance).astype(np.float32)) / margin.shape[0]
    return accuracy

if __name__ == '__main__':
    import json

    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
    parameters = [(1,1)]

    with open('config.json') as config_file:
        config = json.load(config_file)

    x = np.load('clean.npy')

    cropped_x = []
    margin = []
    for (ksize, stride) in parameters:
        cropped_x.append(down_sampling(x, ksize, stride))
        margin.append(np.load('rand_disc/margin_%d%d.npy' % (ksize, stride)))

    for eps in eps_list:
        m = len(parameters)
        acc = np.zeros(m)

        for i in range(m):
            acc[i] = compute_accuracy(cropped_x[i], margin[i], eps)

        opt_i = np.argmax(acc)
        print('%g\t%g\t%s' % (eps, acc[opt_i], parameters[opt_i]))


