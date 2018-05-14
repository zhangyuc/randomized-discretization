import tensorflow as tf
import numpy as np
from PIL import Image
import itertools, os, shutil
import color_spaces
from sklearn.decomposition import PCA
import tf_jpeg_utils

BASIC_COLORS = np.asarray(list(itertools.product([0,255],[0,255],[0,255])))
DIRS = [[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]]

from sklearn.cluster import KMeans

def load_images(filepath):
    with tf.gfile.Open(filepath) as f:
        image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    image = image * 2.0 - 1.0
    return image

def save_images(image, filename):
    image = np.maximum(-1.0, np.minimum(1.0, image))
    with tf.gfile.Open((filename), 'w') as f:
        img = (((image + 1.0) * 0.5) * 255.0).astype(np.uint8)
        Image.fromarray(img).save(f, format='PNG')

def load_cifar10_images():
    X = np.load(open("/scr/zhangyuc/data/cifar10/cifar10.image.npy", "rb"))
    X = X.transpose((1, 2, 0)).reshape([-1, 32, 32, 3]) / 256.0 * 2 - 1.0
    return X

def cs_transform(images):
    # return color_spaces.rgb_to_hsv(images)
    return images

def sample_centroid(images, n_clusters, sigma):
    batchsize, width, height, channel = images.get_shape().as_list()
    images = tf.reshape(images, [-1, width*height, channel])
    shape = [batchsize, width * height]
    covered = tf.zeros(shape)
    centroids = []

    for _ in range(n_clusters):
        indices = tf.random_uniform(shape=[1], minval=0, maxval=width*height, dtype=tf.int32)
        validness = (tf.reshape(tf.gather(covered, indices, axis=1), [batchsize]) < 0.5)
        selected_points = tf.where(validness, tf.gather(params=images, indices=indices, axis=1), 1e4 * tf.ones([batchsize, 1, channel]))
        centroids.append(selected_points)

        # update covered points
        distances = tf.reduce_sum(tf.square(cs_transform(images) - cs_transform(selected_points)), axis=2)
        covered = tf.maximum(covered, tf.where(distances < sigma ** 2, tf.ones(shape), tf.zeros(shape)))

    return tf.concat(centroids, axis=1)

def sample_centroid_with_kpp(images, n_samples, n_clusters, sigma):
    batchsize, width, height, channel = images.get_shape().as_list()
    images = tf.reshape(images, [-1, width*height, channel])
    samples = []

    for _ in range(n_samples):
        indices = tf.random_uniform(shape=[1], minval=0, maxval=width*height, dtype=tf.int32)
        selected_points = tf.gather(params=images, indices=indices, axis=1)
        samples.append(selected_points)

    samples = tf.concat(samples, axis=1)
    distances = 1e4 * tf.ones([batchsize, n_samples])
    centroids = []
    for _ in range(n_clusters):
        indices = tf.reshape(tf.multinomial( sigma * distances, 1), [batchsize])
        weights = tf.expand_dims(tf.one_hot(indices, depth=n_samples), 2)
        selected_points = tf.expand_dims(tf.reduce_sum(weights * samples, axis=1), 1)
        centroids.append(selected_points)

        # update minimal distances
        distances = tf.minimum(distances, tf.reduce_sum(tf.square(samples - selected_points), axis=2))

    return tf.concat(centroids, axis=1)

def rgb_clustering(images, centroids, alpha, noise_level):
    batchsize, width, height, channel = images.get_shape().as_list()

    # Gaussian mixture clustering
    cluster_num = centroids.get_shape().as_list()[1]
    reshaped_images = tf.reshape(images, [-1, width, height, 1, channel])
    reshaped_centroids = tf.reshape(centroids, [-1, 1, 1, cluster_num, channel])
    distances = tf.reduce_sum(tf.square(cs_transform(reshaped_centroids) - cs_transform(reshaped_images)), axis=4)
    logits = tf.clip_by_value(- alpha*distances, -200, 200)
    probs = tf.expand_dims(tf.nn.softmax(logits), 4)
    new_images = tf.reduce_sum(reshaped_centroids * probs, axis=3)

    # update cluster centers
    new_centroids = tf.reduce_sum(reshaped_images * probs, axis=[1, 2]) / (tf.reduce_sum(probs, axis=[1, 2]) + 1e-16)
    return new_images, new_centroids

def iterative_clustering_layer(source, n_clusters, sigma, alpha, noise_level_1, noise_level_2, adaptive_centers=True):
    source1 = source + tf.random_normal(tf.shape(source)) * noise_level_1
    source2 = source + tf.random_normal(tf.shape(source)) * noise_level_2
    if adaptive_centers:
        centroids = sample_centroid_with_kpp(source1, 100, n_clusters, sigma)
    else:
        centroids = tf.tile(tf.expand_dims(tf.constant(DIRS, dtype=tf.float32), axis=0), [source.get_shape().as_list()[0], 1, 1]) * 0.5
    image, _ = rgb_clustering(source2, centroids, alpha, 0.0)
    return image

class ColorClusteringDenoiser:
    def __init__(self, input_shape, n_clusters, sigma, alpha, noise_level_1, noise_level_2, sess=None):
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess

        self.source = tf.placeholder(tf.float32, shape=input_shape)
        self.denoised = iterative_clustering_layer(self.source, n_clusters, sigma, alpha, noise_level_1, noise_level_2)

    def denoise(self, images):
        return self.sess.run(self.denoised, feed_dict={self.source: images})

    def grad_norm(self, images):
        gn = tf.reduce_sum(tf.abs(tf.gradients(self.denoised, self.source)))
        return self.sess.run(gn, feed_dict={self.source: images})

def padding_layer_iyswim(inputs, shape, name=None):
    h_start = shape[0]
    w_start = shape[1]
    output_short = shape[2]
    input_shape = tf.shape(inputs)
    input_short = tf.reduce_min(input_shape[1:3])
    input_long = tf.reduce_max(input_shape[1:3])
    output_long = tf.to_int32(tf.ceil(
        1. * tf.to_float(output_short) * tf.to_float(input_long) / tf.to_float(input_short)))
    output_height = tf.to_int32(input_shape[1] >= input_shape[2]) * output_long +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_short
    output_width = tf.to_int32(input_shape[1] >= input_shape[2]) * output_short +\
        tf.to_int32(input_shape[1] < input_shape[2]) * output_long
    return tf.pad(inputs, tf.to_int32(tf.stack([[0, 0], [h_start, output_height - h_start - input_shape[1]], [w_start, output_width - w_start - input_shape[2]], [0, 0]])), name=name)

def tf_randint(minval, maxval):
    return tf.random_uniform((), minval=minval, maxval=maxval, dtype=tf.int32)

def resize_and_padding_layer(source):
    img_resize_factor = tf_randint(310, 331)
    img_resize_tensor = tf.stack([img_resize_factor, img_resize_factor])
    resized_images = tf.image.resize_images(source, img_resize_tensor, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    shape_tensor = tf.stack([tf_randint(0, 331 - img_resize_factor), tf_randint(0, 331 - img_resize_factor), 331])
    padded_images = padding_layer_iyswim(resized_images, shape_tensor)
    return tf.reshape(padded_images, [-1, 331, 331, 3])

def shift_image(images, tx, ty):
    transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
    return tf.contrib.image.transform(images, transforms)

def tvm_layer(inputs, tv_weight, stepsize, iternum, smooth):
    images = inputs
    for _ in range(iternum):
        delta0 = inputs - images
        if smooth:
            func = lambda x: tf.sigmoid(25 * x) * 2 - 1
            delta1 = func(shift_image(images, 1, 1) - images)
            delta2 = func(shift_image(images, 1, -1) - images)
            delta3 = func(shift_image(images, -1, 1) - images)
            delta4 = func(shift_image(images, -1, -1) - images)
        else:
            delta1 = tf.sign(shift_image(images, 1, 1) - images)
            delta2 = tf.sign(shift_image(images, 1, -1) - images)
            delta3 = tf.sign(shift_image(images, -1, 1) - images)
            delta4 = tf.sign(shift_image(images, -1, -1) - images)
        images = images + stepsize * (delta0 + tv_weight * (delta1 + delta2 + delta3 + delta4))
    return images

def bit_depth_reduction(inputs, step_num, alpha, min_val=-1.0, max_val=1.0):
    steps = min_val + np.arange(1, step_num, dtype=np.float32) / step_num * (max_val - min_val)
    steps = steps.reshape([1,1,1,1,step_num-1])
    tf_steps = tf.constant(steps, dtype=tf.float32)

    inputs = tf.expand_dims(inputs, 4)
    quantized_inputs = min_val + tf.reduce_sum(tf.sigmoid(alpha * (inputs - tf_steps)), axis=4) / (step_num-1) * (max_val - min_val)
    return quantized_inputs

class Denoiser:
    def __init__(self, input_shape):
        self.sess = tf.Session()
        self.source = tf.placeholder(tf.float32, shape=input_shape)
        # self.denoised = tvm_layer(self.source, tv_weight, stepsize, iternum, False)
        # self.denoised = resize_and_padding_layer(self.source)
        # self.denoised = tf_jpeg_utils.jpeg_compress_decompress(self.source)
        self.denoised = bit_depth_reduction(self.source, step_num=8, alpha=1000)

    def denoise(self, images):
        return self.sess.run(self.denoised, feed_dict={self.source: images})


def main():
    np.random.seed(0)
    files = tf.gfile.Glob(os.path.join('./samples', '*.original.png'))[0:10]
    shape = [len(files), 299, 299, 3]
    images = np.zeros(shape)
    for i in range(len(files)):
        images[i] = load_images(files[i])

    # denoiser = ColorClusteringDenoiser(input_shape=[len(files), 299, 299, 3], n_clusters=5, sigma=10, alpha=100, noise_level_1=0.0, noise_level_2=0.125)
    denoiser = Denoiser(input_shape=[len(files), 299, 299, 3])
    denoised_images = denoiser.denoise(images)

    shutil.rmtree('denoiser_output')
    os.mkdir('denoiser_output')
    for i in range(len(files)):
        save_images(images[i], 'denoiser_output/%s.before.png' % i)
        save_images(denoised_images[i], 'denoiser_output/%s.after.png' % i)

if __name__ == '__main__':
    main()
