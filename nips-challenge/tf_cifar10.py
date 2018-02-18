from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data

import tensorflow as tf
import numpy as np
import time, sys, os, math
import tf_nn_utils as tf_utils
import tf_cifar10_utils
import denoisers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=4, suppress=True, threshold=1000, linewidth=500)

def pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class DropNet:
    def __init__(self, params):
        self.noise_level = tf.placeholder(tf.float32, shape=())
        self.hidden_nodes = params['hidden_nodes']
        self.dropout_rate = params['dropout_rate']

    def __call__(self, x, is_training):
        x = x + tf.random_normal(tf.shape(x)) * self.noise_level
        # x = denoisers.iterative_clustering_layer(source=x, n_clusters=5, sigma=1.0, alphas=[100], noise_level=0.0)

        # Convonlutional layer 1
        x = tf_cifar10_utils.conv2d(x, fout=self.hidden_nodes[0], patch_size=5)
        x = pool_2x2(tf.nn.relu(x))
        # x = tf.nn.l2_normalize(x, dim=3)

        # Convonlutional layer 2
        x = tf_cifar10_utils.conv2d(x, fout=self.hidden_nodes[1], patch_size=5)
        x = pool_2x2(tf.nn.relu(x))
        # x = tf.nn.l2_normalize(x, dim=3)

        # Convonlutional layer 3
        x = tf_cifar10_utils.conv2d(x, fout=self.hidden_nodes[2], patch_size=5)
        x = pool_2x2(tf.nn.relu(x))
        # x = tf.nn.l2_normalize(x, dim=3)

        shape = x.get_shape().as_list()
        n_hidden = shape[1] * shape[2] * shape[3]
        x = tf.reshape(x, [-1, n_hidden])

        # fully connected layer 1
        x = tf_cifar10_utils.fc(x, fout=self.hidden_nodes[3])
        x = tf.nn.relu(x)
        if self.dropout_rate[3] > 0:
            x = tf.layers.dropout(x, rate=self.dropout_rate[3], training=is_training)

        # fully connected layer 2
        x = tf_cifar10_utils.fc(x, fout=self.hidden_nodes[4])
        x = tf.nn.relu(x)
        if self.dropout_rate[4] > 0:
            x = tf.layers.dropout(x, rate=self.dropout_rate[4], training=is_training)

        # output layer
        x = tf_cifar10_utils.fc(x, fout=10)
        output = tf.nn.softmax(tf.clip_by_value(x, -20, 20))
        return output

def train(params):
    data = tf_cifar10_utils.read_cifar()
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    truth = tf.placeholder(tf.int32, shape=[None])
    is_training = tf.placeholder(tf.bool, shape=())

    if params['thread'] > 0:
        config = tf.ConfigProto(intra_op_parallelism_threads=params['thread'],
                                inter_op_parallelism_threads=params['thread'])
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    # define model outputs
    model = DropNet(params)
    y = model(x, is_training)

    # training
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=truth))

    if params['optimizer'] == 'momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=0.9)
    elif params['optimizer'] == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
    grads_and_vars = opt.compute_gradients(cross_entropy)
    train_step = opt.apply_gradients(grads_and_vars)

    prediction = tf.one_hot(tf.argmax(y, axis=1), 10)
    sess.run(tf.global_variables_initializer())
    time_elapsed = 0
    sys.stdout.flush()

    def evaluate(dataset, batchsize, max_n, noise_level):
        index, n = 0, min(max_n, len(dataset.images))
        error = 0
        while index < n:
            next_index = min(n, index + batchsize)
            t = dataset.labels[index:next_index]
            feed_dict = {x: dataset.images[index:next_index], truth: t, is_training: False, model.noise_level:noise_level}
            predictions = np.zeros([next_index-index, 10])
            for _ in range(1):
                p = sess.run(prediction, feed_dict=feed_dict)
                predictions += p
            error += (1.0 - np.mean(np.argmax(predictions, axis=1) == t)) * (next_index - index)
            index = next_index
        return error / n

    noise_level = 0.2
    for i in range(params['iternum']):
        batch = data.train.next_batch(params['batchsize'])
        if (i + 1) % 1000 == 0:
            error_train = evaluate(data.train, 1000, 5000, noise_level)
            error_test = evaluate(data.test, 1000, 5000, noise_level)
            print("%f\titer=%d\ttrain=%g\ttest=%g" % (time_elapsed, i + 1, error_train, error_test))

        sys.stdout.flush()
        start = time.time()
        feed_dict = {x: batch[0], truth: batch[1], is_training: True, model.noise_level:noise_level}
        train_step.run(session=sess, feed_dict=feed_dict)
        end = time.time()
        time_elapsed += end - start



def main(args):
    index = 1
    params = {'optimizer': 'rmsprop', 'hidden_nodes': [96, 128, 192, 512, 512],
              'dropout_rate': [0, 0, 0, 0, 0], 'learning_rate': 5e-4, 'iternum': 100000, 'thread': 0, 'batchsize': 50}
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
