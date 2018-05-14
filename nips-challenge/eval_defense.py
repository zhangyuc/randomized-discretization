"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil, math
import numpy as np
from PIL import Image
import StringIO
import tensorflow as tf
from timeit import default_timer as timer
import baseline_models as models

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', './model_ckpts', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', './datasets/images', 'Input directory with images.')

tf.flags.DEFINE_string(
    'label_dir', './datasets/labels.csv', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'testnum', 1, 'How many tests per image.')

tf.flags.DEFINE_string(
    'test', '0', 'models for testing.')

FLAGS = tf.flags.FLAGS

# flog = open('log.txt', 'w')

def string_to_list(s):
    return [int(x) for x in filter(None, s.split(','))]


def compress_by_jpeg(images):
    jpeg_images = np.zeros(images.shape)
    for i in range(images.shape[0]):
        buffer = StringIO.StringIO()
        raw = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        Image.fromarray(raw).save(buffer, "JPEG", quality=15)
        jpeg_images[i, :, :, :] = np.array(Image.open(buffer).convert('RGB')).astype(np.float) / 255.0
    jpeg_images = jpeg_images * 2.0 - 1.0
    return jpeg_images

def load_labels():
    filename = FLAGS.label_dir
    labels = {}
    for line in open(filename).readlines():
        items = filter(None, line.split(','))
        labels[items[0]] = int(items[1])
    return labels

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath) as f:
            # original images
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0

        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

class Evaluator(object):
    def __init__(self, name, models, image, image_input, true_label, test):
        self.predictions = []
        for i in test:
            self.predictions.append(tf.one_hot(tf.argmax(models[i].logits, axis=1), 1001))

        self.name = name
        self.model_num = len(self.predictions)
        self.processed_batch_num = 0
        self.overall_accuracy = np.zeros(self.model_num)
        self.label = true_label
        self.image_input = image_input
        self.assign_image = tf.assign(image, image_input)

    def run(self, sess, image_input, y, iternum=1):
        predictions = []
        for i in range(self.model_num):
            predictions.append(np.zeros([image_input.shape[0], 1001]))

        sess.run(self.assign_image, feed_dict={self.image_input: image_input})
        for _ in range(iternum):
            pval = sess.run(self.predictions)
            for i in range(self.model_num):
                predictions[i] += pval[i] / float(iternum)

        accuracies = []
        n = len(y)
        y_onehot = np.zeros([n, 1001])
        y_onehot[np.arange(n), y] = 1

        for i in range(self.model_num):
            final_prediction = np.argmax(predictions[i], axis=1)
            correct_prediction = np.equal(final_prediction, y)
            accuracy = np.mean(correct_prediction.astype(np.float32))
            accuracies.append(accuracy)

        accuracies = [round(x,3) for x in accuracies]
        print('%s evaluation accuracies: %s' % (self.name, accuracies))

        self.processed_batch_num += 1
        self.overall_accuracy += accuracies
        if self.processed_batch_num % 10 == 0:
            print('%s overall evaluation accuracies: %s' % (self.name, self.overall_accuracy / self.processed_batch_num))

def main(_):
    full_start = timer()
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        image_input = tf.placeholder(tf.float32, shape=batch_shape)
        image = tf.get_variable('adversarial_image', shape=batch_shape)
        label = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        sess = tf.Session()


        initialized_vars = set()
        savers = []

        # list of models in our ensemble
        # model 0-5
        all_models = [models.BitDepth, models.JPEG, models.TVM, models.Gaussian, models.RandDisc, models.RandMix]
        test = string_to_list(FLAGS.test)
        indices_to_load = test

        labels = load_labels()

        # build all the models and specify the saver
        for i, model in enumerate(all_models):
            all_models[i] = model(num_classes)
            if i in indices_to_load:
                print('creating model %d: %s' % (i, all_models[i].name))
                all_models[i](image, FLAGS.batch_size)
                all_vars = slim.get_model_variables()
                uninitialized_vars = set(all_vars) - initialized_vars
                saver_dict = {v.op.name[len(all_models[i].name) + 1:]: v for v in uninitialized_vars}
                savers.append(tf.train.Saver(saver_dict))
                initialized_vars = set(all_vars)
            else:
                savers.append(None)

        original_eval = Evaluator(name='original', models=all_models, image_input=image_input, image=image,
                                  true_label=label, test=test)

        sess.run(tf.global_variables_initializer())
        for i in indices_to_load:
            savers[i].restore(sess, FLAGS.checkpoint_path + '/' + all_models[i].ckpt)

        init_time = timer() - full_start
        print("Initialization done after {} sec".format(init_time))

        if FLAGS.output_dir != '':
            shutil.rmtree(FLAGS.output_dir)
            os.mkdir(FLAGS.output_dir)

        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            y = np.asarray([labels[fn[:-4]] for fn in filenames])

            # evaluation
            if len(test) > 0:
                original_eval.run(sess, images, y, iternum=FLAGS.testnum)


        sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run()
