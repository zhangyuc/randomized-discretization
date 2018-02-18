import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np
from PIL import Image
import itertools, os, shutil, math
import denoisers
import color_spaces
import scipy

DIRS = [[0,0,0], [-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1], [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]]

def load_images(filepath):
    with tf.gfile.Open(filepath) as f:
        image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    image = image * 2.0 - 1.0
    return image

def save_image(image, filename):
    image = np.maximum(-1.0, np.minimum(1.0, image))
    with tf.gfile.Open((filename), 'w') as f:
        img = (((image + 1.0) * 0.5) * 255.0).astype(np.uint8)
        Image.fromarray(img).save(f, format='PNG')

def initc(source, n_clusters):
    # source += 0.125 * tf.random_normal(tf.shape(source))
    # return denoisers.sample_centroid_with_kpp(source, n_samples=100, n_clusters=n_clusters, sigma=10)
    return tf.tile(tf.expand_dims(tf.constant(denoisers.DIRS, dtype=tf.float32), axis=0), [source.get_shape().as_list()[0], 1, 1]) * 0.5

def multinomial_kl_divergence(p,q):
    return np.sum(p * np.log(p/q), axis=-1)

def sample(pixels, centroids, noise_level):
    batchsize = pixels.get_shape().as_list()[0]
    pixel_num = pixels.get_shape().as_list()[1]
    cluster_num = centroids.get_shape().as_list()[1]

    # Gaussian mixture clustering
    random_pixels = tf.reshape(pixels, [batchsize, pixel_num, 1, 3]) + noise_level * tf.random_normal([batchsize, pixel_num, 1, 3])
    reshaped_centroids = tf.reshape(centroids, [batchsize, 1, cluster_num, 3])
    # distances = tf.reduce_sum(tf.square(color_spaces.rgb_to_hsv(reshaped_centroids) - color_spaces.rgb_to_hsv(random_pixels)), axis=3)
    distances = tf.reduce_sum(tf.square(reshaped_centroids - random_pixels), axis=3)
    return tf.argmin(distances, axis=2)

def descretize(images, centroids, noise_level):
    batchsize = images.get_shape().as_list()[0]
    cluster_num = centroids.get_shape().as_list()[1]

    pixels = tf.reshape(images, [batchsize, -1, 3])
    indices = tf.reshape(sample(pixels, centroids, noise_level), [batchsize, -1])

    weights = tf.expand_dims(tf.one_hot(indices, depth=cluster_num), 3)
    reshaped_centroids = tf.reshape(centroids, [batchsize, 1, cluster_num, 3])
    discretized_pixels = tf.reduce_sum(weights * reshaped_centroids, axis=2)
    return tf.reshape(discretized_pixels, tf.shape(images))

def compute_probs(pixels, centroids, noise_level, copies, eps):
    batchsize, pixel_num, _ = pixels.get_shape().as_list()
    cluster_num = centroids.get_shape().as_list()[1]

    noisy_pixels = tf.reshape(pixels, [batchsize, 1, pixel_num, 1, 3]) + noise_level * tf.random_normal([batchsize, 1, pixel_num, copies, 3])
    perturb = eps * tf.reshape(tf.constant(np.asarray(DIRS, dtype=np.float32)), [1, 9, 1, 1, 3])
    perturbed_pixels = noisy_pixels + perturb
    perturbed_pixels = tf.reshape(perturbed_pixels, [batchsize, 9*pixel_num*copies, 3])

    indices = tf.reshape(sample(perturbed_pixels, centroids, noise_level=0), [batchsize, -1])
    probs = tf.reduce_sum(tf.reshape(tf.one_hot(indices, depth=cluster_num), [batchsize, 9, pixel_num, copies, cluster_num]), axis=3) / copies
    return tf.unstack(probs, axis=1)

def compute_rand_disc_kl(output_image=True):
    image_size = 299
    eps = 1.0 * (2.0 / 256)
    n_clusters = 8
    noise_level = 0.25
    batchsize = 100

    # parameters for approximating kl divergence
    pixel_num = 10
    trails = 10
    copies = 100000
    buffer_size = 1000000 / pixel_num / batchsize

    files = tf.gfile.Glob(os.path.join('../datasets/images', '*.png'))[0:batchsize]
    filenames = [s[len('../datasets/images'):] for s in files]

    batchsize = len(files)
    batch_shape = [batchsize, image_size, image_size, 3]
    images = np.zeros(batch_shape)
    for i in range(batchsize):
        images[i] = load_images(files[i])

    images_input = tf.placeholder(tf.float32, shape=batch_shape)
    pixels_input = tf.placeholder(tf.float32, shape=[batchsize, pixel_num, 3])
    centroids_input = tf.placeholder(tf.float32, shape=[batchsize, n_clusters, 3])
    discretized_images = descretize(images_input, centroids_input, noise_level=noise_level)
    probs = compute_probs(pixels_input, centroids_input, noise_level=noise_level, copies=buffer_size, eps=eps)

    # global initialization
    sess = tf.Session()

    total_kl = np.zeros([batchsize])
    for _ in range(trails):
        init_centroids = sess.run(initc(images_input, n_clusters=n_clusters), feed_dict={images_input: images})
        dimg = sess.run(discretized_images, feed_dict={images_input: images, centroids_input: init_centroids})

        indices = np.random.randint(0, image_size*image_size, [pixel_num])
        pixels = images.reshape([batchsize, image_size*image_size, 3])[:, indices]
        probabilities = [0] * 9
        m = int(copies/buffer_size)
        for _ in range(m):
            tmp = sess.run(probs, feed_dict={pixels_input: pixels, centroids_input: init_centroids})
            for i in range(9):
                probabilities[i] += tmp[i] / m

        kl = np.zeros([8, batchsize, pixel_num])
        for i in range(1, 9):
            kl[i-1] = multinomial_kl_divergence(probabilities[i] + 1e-5, probabilities[0] + 1e-5)
        total_kl += np.sum(np.amax(kl, axis=0), axis=1) / trails / pixel_num * 299 * 299

    if output_image:
        for i in range(len(files)):
            save_image(dimg[i], 'denoiser_output/%s.discrete.png' % filenames[i][:-4])

    return total_kl + (1.0 / (0.125 / 2.0 * 256)) ** 2 * 100 * 3.0 / 2.0

def compute_gaussian_kl(output_image=True):
    image_size = 299
    batchsize = 100
    noise_level = 0.25

    files = tf.gfile.Glob(os.path.join('../datasets/images', '*.png'))[0:batchsize]
    filenames = [s[len('../datasets/images'):] for s in files]

    batchsize = len(files)
    batch_shape = [batchsize, image_size, image_size, 3]
    images = np.zeros(batch_shape)
    for i in range(batchsize):
        images[i] = load_images(files[i])

    if output_image:
        for i in range(len(files)):
            save_image(images[i] + noise_level * np.random.randn(image_size, image_size, 3), 'denoiser_output/%s.gaussian.png' % filenames[i][:-4])

    return np.ones([batchsize]) * (1.0 / (noise_level / 2.0 * 256) * 299) ** 2 * 3.0 / 2.0

if __name__ == '__main__':
    mode = 'rand'

    if mode == 'gaussian':
        kl_dvg = compute_gaussian_kl()
        margin = np.loadtxt('result_output/margin_gaussian.txt', delimiter=',')
    else:
        kl_dvg = compute_rand_disc_kl()
        margin = np.loadtxt('result_output/margin_rand_disc.fixed_center.txt', delimiter=',')

    eps_list = np.asarray(range(50)) / 200.0
    for eps in eps_list:
        tv_distance = np.sqrt(2.0 * kl_dvg) * eps
        accuracy = np.sum((margin > tv_distance).astype(np.float32)) / margin.shape[0]
        print(accuracy)

    print kl_dvg
    print margin