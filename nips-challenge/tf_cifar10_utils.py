import tensorflow as tf
import numpy as np
from PIL import Image
import tf_nn_utils

class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_cifar():
    class DataSets(object):
        pass

    data_sets = DataSets()

    X = np.load(open("/scr/zhangyuc/data/cifar10/cifar10.whitened_image.npy", "rb"))
    X = X.transpose((1, 2, 0)) / 255.0 * 2 - 1.0
    print(X.shape)

    y = np.load(open("/scr/zhangyuc/data/cifar10/cifar10.label.npy", "rb"))

    train_size = 50000
    test_size = 10000

    train_images = X[0:train_size].reshape((train_size, 32, 32, 3))
    train_labels = y[0:train_size]
    test_images = X[train_size:train_size + test_size].reshape((test_size, 32, 32, 3))
    test_labels = y[train_size:train_size + test_size]
    all_images = np.copy(X[:train_size + test_size].reshape((train_size + test_size, 32, 32, 3)))
    all_labels = np.copy(y[:train_size + test_size])

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    data_sets.all = DataSet(all_images, all_labels)
    return data_sets

def save_images(image, filename):
    image = np.maximum(0.0, np.minimum(1.0, image))
    with tf.gfile.Open((filename), 'w') as f:
        img = ((image + 1.0) / 2.0 * 255.0).astype(np.uint8)
        Image.fromarray(img).save(f, format='PNG')

def conv2d(x, fout, patch_size):
    fin = x.get_shape().as_list()[-1]
    W = tf_nn_utils.weight_variable([patch_size, patch_size, fin, fout], scale=0.1)
    b = tf_nn_utils.bias_variable([fout])
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b

def fc(x, fout):
    fin = x.get_shape().as_list()[-1]
    W = tf_nn_utils.weight_variable([fin, fout])
    b = tf_nn_utils.bias_variable([fout])
    return tf.matmul(x, W) + b