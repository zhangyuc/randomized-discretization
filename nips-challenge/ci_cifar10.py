from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data

import tensorflow as tf
import numpy as np
import time, sys, os, math
import tf_nn_utils as tf_utils
import tf_cifar10_utils
from timeit import default_timer as timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=4, suppress=True, threshold=1000, linewidth=500)


class Attacker(object):
    def __init__(self, model, max_epsilon, learning_rate, batchsize):
        self.model = model
        self.batchsize = batchsize
        self.max_epsilon = max_epsilon
        self.processed_batch_num = 0
        self.time_per_iter = 0
        self.overall_error = 0.0

        # placeholders
        self.input = model.input
        self.label = tf.placeholder(tf.int32, shape=[batchsize])
        self.adv_image = tf.get_variable('adv_image', shape=[batchsize, 32, 32, 3])
        self.initialization_step = tf.assign(self.adv_image, self.input)

        # label_mask = tf.one_hot(self.label, 10, dtype=tf.float32)
        correct_prediction = tf.equal(tf.argmax(model.logits, axis=1), tf.cast(self.label, tf.int64))
        self.error = 1 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # loss and gradient
        self.loss = - tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.logits, labels=self.label))
        self.grad = tf.gradients(self.loss, self.input)[0]

        # define optimization step
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * max_epsilon)
        self.gradient_step = opt.apply_gradients([(tf.sign(self.grad), self.adv_image)])

    def run(self, sess, images, labels, iternum, cl):
        start1 = start2 = timer()
        sess.run(self.initialization_step, feed_dict={self.input: images})
        adv_images = images
        confidence_level = cl * np.ones([self.batchsize, 32, 32, 3])

        for i in range(iternum):
            if i == 1: start2 = timer()
            sess.run(self.gradient_step,
                     feed_dict={self.input: adv_images, self.label: labels, self.model.ci: confidence_level})
            adv_images = sess.run(self.adv_image)
            adv_images = np.maximum(images - self.max_epsilon, np.minimum(images + self.max_epsilon, adv_images))
        end = timer()

        adv_images, error = sess.run([self.adv_image, self.error],
                                     feed_dict={self.input: adv_images, self.label: labels,
                                                self.model.ci: confidence_level})
        error = round(error, 3)

        print('Attacker -- iternum: %d, time: %g sec, error: %g' % (iternum, end - start1, error))
        sys.stdout.flush()

        self.processed_batch_num += 1
        self.time_per_iter = (end - start2) / (iternum - 1) if iternum > 1 else end - start1
        self.overall_error += error
        return adv_images - images


def suppress(x, ci):
    def f(t, r): return tf.minimum(tf.maximum(0.01 * t, t - r), t + r)
    x_min = f(x - ci, ci * 2)
    x_max = f(x + ci, ci * 2)
    return (x_max + x_min) / 2, (x_max - x_min) / 2


def relu(x, ci):
    x_max = tf.nn.relu(x + ci)
    x_min = tf.nn.relu(x - ci)
    return (x_max + x_min) / 2, (x_max - x_min) / 2


def sigmoid(x, ci):
    x_max = tf.nn.sigmoid(x + ci)
    x_min = tf.nn.sigmoid(x - ci)
    return (x_max + x_min) / 2, (x_max - x_min) / 2


def pool(x, ci):
    x = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    ci = tf.nn.avg_pool(ci, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    return x, ci


def conv2d(x, ci, fout, patch_size):
    fin = x.get_shape().as_list()[-1]
    W = tf_utils.weight_variable([patch_size, patch_size, fin, fout], scale=0.1)
    b = tf_utils.bias_variable([fout])
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
    ci = tf.nn.conv2d(ci, tf.abs(W), strides=[1, 1, 1, 1], padding='SAME')
    return x, ci


def fc(x, ci, fin, fout):
    W = tf_utils.weight_variable([fin, fout])
    b = tf_utils.bias_variable([fout])
    x = tf.matmul(x, W) + b
    ci = tf.matmul(ci, tf.abs(W))
    return x, ci


class DropNet:
    def __init__(self, params):
        self.hidden_nodes = params['hidden_nodes']

    def __call__(self, x, ci):
        self.input = x
        self.ci = ci

        # Convonlutional layer 1
        x, ci = conv2d(x, ci, fout=self.hidden_nodes[0], patch_size=6)
        x, ci = relu(x, ci)
        x, ci = suppress(x, ci)
        x, ci = pool(x, ci)

        # Convonlutional layer 2
        x, ci = conv2d(x, ci, fout=self.hidden_nodes[1], patch_size=6)
        x, ci = relu(x, ci)
        x, ci = suppress(x, ci)
        x, ci = pool(x, ci)

        # Convonlutional layer 3
        x, ci = conv2d(x, ci, fout=self.hidden_nodes[2], patch_size=6)
        x, ci = relu(x, ci)
        x, ci = suppress(x, ci)
        x, ci = pool(x, ci)

        shape = x.get_shape().as_list()
        n_hidden = shape[1] * shape[2] * shape[3]
        x = tf.reshape(x, [-1, n_hidden])
        ci = tf.reshape(ci, [-1, n_hidden])

        # # fully connected layer 1
        # x, ci = fc(x, ci, n_hidden, self.hidden_nodes[3])
        # x, ci = relu(x, ci)
        # x, ci = suppress(x, ci)
        #
        # # fully connected layer 2
        # x, ci = fc(x, ci, self.hidden_nodes[3], self.hidden_nodes[4])
        # x, ci = relu(x, ci)
        # x, ci = suppress(x, ci)

        # output layer
        x, ci = fc(x, ci, n_hidden, 10)
        x, ci = suppress(x, ci)
        self.logits = tf.nn.softmax(tf.clip_by_value(x, -20, 20))
        return self.logits


def train(params):
    data = tf_cifar10_utils.read_cifar()
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    ci = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    truth = tf.placeholder(tf.int32, shape=[None])
    sess = tf.Session()

    # define model outputs
    model = DropNet(params)
    y = model(x, ci)

    # define attacker
    attacker = Attacker(model=model, max_epsilon=0.03, learning_rate=1.0, batchsize=100)

    # training
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.logits, labels=truth))

    if params['optimizer'] == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=0.9)
    elif params['optimizer'] == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
    grads_and_vars = opt.compute_gradients(cross_entropy)
    train_step = opt.apply_gradients(grads_and_vars)

    prediction = tf.argmax(y, 1)
    sess.run(tf.global_variables_initializer())
    time_elapsed = 0
    sys.stdout.flush()

    def evaluate(dataset, batchsize, max_n, cl):
        index, n = 0, min(max_n, len(dataset.images))
        error = 0
        while index < n:
            next_index = min(n, index + batchsize)
            confidence_level = cl * np.ones([next_index - index, 32, 32, 3])
            t = dataset.labels[index:next_index]
            p = sess.run(prediction, feed_dict={x: dataset.images[index:next_index], ci: confidence_level})
            error += (1.0 - np.mean(p == t)) * (next_index - index)
            index = next_index
        return error / n

    confidence_level = params['cl'] * np.ones([params['batchsize'], 32, 32, 3])
    for i in range(params['iternum']):
        if (i + 1) % 1000 == 0:
            error_train = evaluate(data.train, 1000, 10000, params['cl'])
            error_test = evaluate(data.test, 1000, 10000, params['cl'])
            print("%f\titer=%d\ttrain=%g\ttest=%g" % (time_elapsed, i + 1, error_train, error_test))

            # run attacker
            batch = data.test.next_batch(100)
            attacker.run(sess=sess, images=batch[0], labels=batch[1], iternum=1, cl=params['cl'])

        sys.stdout.flush()
        start = time.time()

        batch = data.train.next_batch(params['batchsize'])
        feed_dict = {x: batch[0], truth: batch[1], ci: confidence_level}
        train_step.run(session=sess, feed_dict=feed_dict)
        end = time.time()
        time_elapsed += end - start


def main(args):
    index = 1
    params = {'optimizer': 'rmsprop', 'hidden_nodes': [32, 64, 64, 128, 128], 'learning_rate': 1e-3,
              'iternum': 100000, 'thread': 0, 'batchsize': 50, 'cl': 0.05}
    while (index < len(args)):
        if args[index] in {'--optimizer'}:
            params[args[index][2:]] = args[index + 1]
        elif args[index] in {'--hidden_nodes'}:
            params[args[index][2:]] = [int(x) for x in args[index + 1][1:-1].split(',')]
        elif args[index] in {'--dropout_rate'}:
            params[args[index][2:]] = [float(x) for x in args[index + 1][1:-1].split(',')]
        elif args[index] in {'--learning_rate'}:
            params[args[index][2:]] = float(args[index + 1])
        elif args[index] in {'--iternum', '--thread', '--batchsize'}:
            params[args[index][2:]] = int(args[index + 1])
        else:
            print('unknown option: %s' % args[index])
            return
        index += 2

    print('params: ' + ' '.join(('--%s %s' % item) for item in params.items()))
    train(params)


if __name__ == "__main__":
    main(sys.argv)
