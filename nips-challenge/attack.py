"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
import numpy as np
from PIL import Image
import StringIO
import tensorflow as tf
from timeit import default_timer as timer

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', './model_ckpts', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', 'datasets/images', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_string(
    'label_dir', './datasets/labels.csv', 'Input directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 4.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_integer(
    'iternum', 10, 'How many iterations does the attacker runs.')

tf.flags.DEFINE_integer(
    'gradnum', 1, 'How many gradients in each iteration.')

tf.flags.DEFINE_integer(
    'testnum', 1, 'How many tests per image.')

tf.flags.DEFINE_float(
    'learning_rate', 0.2, 'The learning rate of attacker.')

tf.flags.DEFINE_float(
    'margin', 0.01, 'margin parameter in the loss function.')

tf.flags.DEFINE_string(
    'attack', '1', 'models for whitebox attacking.')

tf.flags.DEFINE_string(
    'test', '1,2', 'models for testing.')

FLAGS = tf.flags.FLAGS

# flog = open('log.txt', 'w')

def string_to_list(s):
    return [int(x) for x in filter(None, s.split(','))]

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

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')

class Evaluator(object):
    def __init__(self, name, models, image, image_input, true_label, test):
        self.predictions = []
        for i in test:
            self.predictions.append(tf.one_hot(tf.argmax(models[i].logits, axis=1), 1001))

        self.name = name
        self.model_num = len(self.predictions)
        self.processed_batch_num = 0
        self.overall_errors = np.zeros(self.model_num)
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

        errors = []
        n = len(y)
        y_onehot = np.zeros([n, 1001])
        y_onehot[np.arange(n), y] = 1

        for i in range(self.model_num):
            final_prediction = np.argmax(predictions[i], axis=1)
            correct_prediction = np.equal(final_prediction, y)
            error = 1 - np.mean(correct_prediction.astype(np.float32))
            errors.append(error)

        errors = [round(x,3) for x in errors]
        print('%s evaluation errors: %s' % (self.name, errors))

        self.processed_batch_num += 1
        self.overall_errors += errors
        if self.processed_batch_num % 10 == 0:
            print('%s overall evaluation errors: %s' % (self.name, self.overall_errors / self.processed_batch_num))

class Attacker(object):
    def __init__(self, name, models, image_input, image, true_label, max_epsilon, attack, test,
                 optimizer, loss_type, margin, learning_rate):
        self.name = name
        self.models = models
        self.max_epsilon = max_epsilon
        self.processed_batch_num = 0
        self.time_per_iter = 0
        self.overall_attack_errors = np.zeros(len(attack))
        self.overall_test_errors = np.zeros(len(test))
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.attack = attack
        if len(self.attack) == 0:
            return

        # placeholders
        self.label = true_label
        self.image_input = image_input
        self.image = image
        self.assign_image = tf.assign(image, image_input)
        self.assign_add_image = tf.assign_add(image, image_input)

        label_mask = tf.one_hot(true_label, 1001, dtype=tf.float32)

        def define_errors(model_indices):
            errors = []
            for i in model_indices:
                correct_prediction = tf.equal(tf.argmax(models[i].logits, axis=1), tf.cast(true_label, tf.int64))
                error = 1 - tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                errors.append(error)
            return errors

        self.attack_errors = define_errors(attack)
        self.test_errors = define_errors(test)

        # define gradient
        grad = None
        if loss_type == 'cross-entropy':
            softmax_prob_sum = 0
            for i in attack:
                softmax_prob_sum += tf.reduce_sum(tf.nn.softmax(models[i].logits) * label_mask, axis=1)
            self.mixture_loss = tf.reduce_mean(tf.log(margin + softmax_prob_sum))
            grad = tf.gradients(self.mixture_loss, image)[0]

        self.grad = grad

        # define optimization step
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * max_epsilon)
        self.all_model_gradient_step = opt.apply_gradients([(tf.sign(grad), image)])
        self.apply_null = opt.apply_gradients([(tf.zeros(image.get_shape().as_list(), dtype=tf.float32), image)])

        # define clipping step
        clipped_image = tf.clip_by_value(image, image_input - max_epsilon, image_input + max_epsilon)
        clipped_image = tf.clip_by_value(clipped_image, -1, 1)
        self.clipping_step = tf.assign(image, clipped_image)

        # define custom gradient step
        self.custom_gradient = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3])
        opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * max_epsilon)
        self.custom_gradient_step = opt.apply_gradients([(tf.sign(self.custom_gradient), image)])

    def run(self, sess, x_batch, iternum, y, gradnum=1):
        start1 = start2 = timer()

        sess.run(self.assign_image, feed_dict={self.image_input: x_batch})
        if self.optimizer == 'rmsprop':
            for _ in range(200):
                sess.run(self.apply_null)

        for i in range(iternum):
            if i == 1: start2 = timer()
            sample_mean = np.zeros(list(x_batch.shape))
            for gi in range(gradnum):
                grad = sess.run(self.grad, feed_dict={self.label: y, self.image_input: x_batch})
                sample_mean += grad
            sample_mean = sample_mean / gradnum

            sess.run(self.custom_gradient_step, feed_dict={self.custom_gradient: sample_mean})
            sess.run(self.clipping_step, feed_dict={self.image_input: x_batch})
        end = timer()

        x_adv, attack_errors_after_attack, test_errors_after_attack = sess.run(
            [self.image, self.attack_errors, self.test_errors], feed_dict={self.label: y})
        attack_errors_after_attack = [round(x, 3) for x in attack_errors_after_attack]
        test_errors_after_attack = [round(x, 3) for x in test_errors_after_attack]

        print('%s -- iternum: %d, time: %g sec, attack_errors: %s, test_errors: %s' % (
            self.name, iternum, end - start1, attack_errors_after_attack, test_errors_after_attack))
        sys.stdout.flush()

        self.processed_batch_num += 1
        self.time_per_iter = (end - start2) / (iternum - 1) if iternum > 1 else end - start1
        self.overall_attack_errors += attack_errors_after_attack
        self.overall_test_errors += test_errors_after_attack
        return x_adv - x_batch

def main(_):
    full_start = timer()
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        image_input = tf.placeholder(tf.float32, shape=batch_shape)
        image = tf.get_variable('adversarial_image', shape=batch_shape)
        label = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        sess = tf.Session()

        import models

        initialized_vars = set()
        savers = []

        # list of models in our ensemble
        # model 0-5
        all_models = [models.BasicModel, models.Gaussian, models.RandMix, models.RandDisc,
                      models.BitDepth, models.ResizeAndPadding, models.JPEG, models.TVM]
        whitebox_models = string_to_list(FLAGS.attack)
        test = string_to_list(FLAGS.test)
        indices_to_load = [index for index in range(len(all_models)) if
                           index in [0] + whitebox_models + test]

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

        with tf.variable_scope('whitebox-attacker'):
            whitebox_attacker = Attacker(name='whitebox-attacker', models=all_models, image_input=image_input,
                                         image=image, true_label=label,
                                         max_epsilon=eps, attack=whitebox_models,
                                         test=[], optimizer='sgd',
                                         loss_type='cross-entropy', margin=FLAGS.margin,
                                         learning_rate=FLAGS.learning_rate)

        if len(test) > 0:
            with tf.variable_scope('raw_evaluator'):
                original_eval = Evaluator(name='original', models=all_models, image_input=image_input, image=image,
                                          true_label=label, test=test)

        labels = load_labels()
        # Run computation
        postprocess_time = 0.0
        processed = 0.0

        sess.run(tf.global_variables_initializer())
        for i in indices_to_load:
            savers[i].restore(sess, FLAGS.checkpoint_path + '/' + all_models[i].ckpt)

        init_time = timer() - full_start
        print("Initialization done after {} sec".format(init_time))

        if FLAGS.output_dir != '':
            shutil.rmtree(FLAGS.output_dir)
            os.mkdir(FLAGS.output_dir)

        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
            original = np.copy(images)
            y = np.asarray([labels[fn[:-4]] for fn in filenames])

            # starting processing this batch
            print('batch %d' % (whitebox_attacker.processed_batch_num + 1))
            whitebox_perturb = whitebox_attacker.run(sess, images, FLAGS.iternum, y=y, gradnum=FLAGS.gradnum)
            perturb = whitebox_perturb
            images += perturb

            # evaluation
            start = timer()
            if len(test) > 0:
                original_eval.run(sess, images, y, iternum=FLAGS.testnum)

            if FLAGS.output_dir != '':
                save_images(original, [s[:-4] + '.original.png' for s in filenames], FLAGS.output_dir)
            #     save_images(perturb, [s[:-4] + '.final_attack.png' for s in filenames], FLAGS.output_dir)
            #     save_images(images, [s[:-4] + '.final_image.png' for s in filenames], FLAGS.output_dir)

            end = timer()
            postprocess_time += end - start
            processed += FLAGS.batch_size
            print('time elapsed: %g' % (timer() - full_start))

        print("DONE: Processed {} images in {} sec".format(processed, timer() - full_start))
        sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run()
