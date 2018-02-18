import tensorflow as tf
import numpy as np
import sys, os, math, shutil
import tf_nn_utils
import tf_imagenet_utils

# class Attacker:
#     def __init__(self, autoencoder, max_epsilon, iternum):
#         self.max_epsilon = max_epsilon
#         self.input_images = autoencoder.images
#         self.original_output = tf.placeholder(tf.float32, shape=[None, 32 * 32 * 3])
#         self.adv_images = tf.get_variable('adv_image', shape=autoencoder.images.get_shape.as_list())
#         self.initialize = tf.assign(self.adv_images, autoencoder.images + max_epsilon * tf.random_normal(tf.shape(autoencoder.images)))
#
#         # define attack step
#         loss = tf.reduce_mean(tf.square(self.original_output - ))
#         grad = tf.gradients(loss, autoencoder.images)[0]
#         opt = tf.train.GradientDescentOptimizer(learning_rate=1.25 * max_epsilon / iternum)
#         self.attack_step = opt.apply_gradients([(tf.sign(grad), self.adv_images)])

class Autoencoder:
    def __init__(self, params):
        self.images = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        self.noise_level = tf.placeholder(tf.float32, shape=())

        reshaped_images = tf.reshape(self.images, [-1, 299, 299, 3])
        x = reshaped_images + tf.random_normal(tf.shape(reshaped_images)) * self.noise_level

        # convolutional layers
        n = len(params['fout'])
        layers = []
        for i in range(n):
            layers.append(x)
            x = tf_nn_utils.conv2d(x, patch_size=5, fout=params['fout'][i], strides=[1, 1, 1, 1], padding='SAME')
            if i > 1 and i % 2 == 1:
                x += layers[i-2]
            x = tf.nn.relu(x)

        x = tf_nn_utils.conv2d(x, patch_size=5, fout=3, strides=[1, 1, 1, 1], padding='SAME')
        self.output = x
        self.loss = tf.reduce_mean(tf.square(reshaped_images-self.output))

def main(args):
    params = {}
    params['fout'] = [32] * 5
    params['learning_rate'] = 1e-3
    params['iternum'] = 1000
    params['batchsize'] = 50
    params['noise_level'] = 0.5

    data = tf_imagenet_utils.read()
    autoencoder = Autoencoder(params)

    sess = tf.Session()
    opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    grads_and_vars = opt.compute_gradients(autoencoder.loss)
    train_step = opt.apply_gradients(grads_and_vars)

    sess.run(tf.global_variables_initializer())
    for i in range(params['iternum']):
        batch = data.train.next_batch(params['batchsize'])
        if (i + 1) % 50 == 0:
            test_batch = data.train.next_batch(100)
            loss = sess.run(autoencoder.loss, feed_dict={autoencoder.images: test_batch[0], autoencoder.noise_level: params['noise_level']})
            print("iter=%d\tloss=%g" % (i + 1, loss))

        sess.run(train_step, feed_dict={autoencoder.images: batch[0], autoencoder.noise_level: params['noise_level']})

    # output images
    n = 20
    test_batch = data.train.next_batch(n)
    output_images = sess.run(autoencoder.output, feed_dict={autoencoder.images: test_batch[0], autoencoder.noise_level: 0.5})

    shutil.rmtree('autoencoder_output')
    os.mkdir('autoencoder_output')
    for i in range(n):
        tf_imagenet_utils.save_images(test_batch[0][i].reshape([299, 299, 3]), 'autoencoder_output/%s.before.png' % i)
        tf_imagenet_utils.save_images(output_images[i], 'autoencoder_output/%s.after.png' % i)


if __name__ == "__main__":
    main(sys.argv)
