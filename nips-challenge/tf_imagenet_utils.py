import tensorflow as tf
import numpy as np
from PIL import Image
import os

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

def read():
    class DataSets(object):
        pass
    data_sets = DataSets()

    files = tf.gfile.Glob(os.path.join('../datasets/images', '*.png'))
    images = np.zeros([len(files), 299, 299, 3])
    idx = 0
    for filepath in files:
        with tf.gfile.Open(filepath) as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
            images[idx] = image * 2.0 - 1
        idx += 1

    data_sets.train = DataSet(images, np.zeros([len(files)]))
    return data_sets

def save_images(image, filename):
    with tf.gfile.Open(filename, 'w') as f:
        img = (((image + 1.0) * 0.5) * 255.0).astype(np.uint8)
        Image.fromarray(img).save(f, format='PNG')