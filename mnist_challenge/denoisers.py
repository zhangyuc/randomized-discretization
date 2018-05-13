import tensorflow as tf

def sample_centroid_with_kpp(images, n_samples, n_clusters, sigma):
    batchsize, width, height, channel = images.get_shape().as_list()
    images = tf.reshape(images, [-1, width*height, channel])
    samples = []

    for _ in range(n_samples):
        indices = tf.random_uniform(shape=[1], minval=0, maxval=width*height, dtype=tf.int32)
        selected_points = tf.gather(params=images, indices=indices, axis=1)
        samples.append(selected_points)

    samples = tf.concat(samples, axis=1)
    distances_shape = tf.concat([tf.slice(tf.shape(images), [0], [1]), [n_samples]], 0)
    distances = 1e4 * tf.ones(distances_shape)
    centroids = []
    for _ in range(n_clusters):
        indices = tf.reshape(tf.multinomial( sigma * distances, 1), [-1])
        weights = tf.expand_dims(tf.one_hot(indices, depth=n_samples), 2)
        selected_points = tf.expand_dims(tf.reduce_sum(weights * samples, axis=1), 1)
        centroids.append(selected_points)

        # update minimal distances
        distances = tf.minimum(distances, tf.reduce_sum(tf.square(samples - selected_points), axis=2))

    return tf.concat(centroids, axis=1)

def rgb_clustering(images, centroids, alpha):
    batchsize, width, height, channel = images.get_shape().as_list()

    # Gaussian mixture clustering
    cluster_num = centroids.get_shape().as_list()[1]
    reshaped_images = tf.reshape(images, [-1, width, height, 1, channel])
    reshaped_centroids = tf.reshape(centroids, [-1, 1, 1, cluster_num, channel])
    distances = tf.reduce_sum(tf.square(reshaped_centroids - reshaped_images), axis=4)
    logits = tf.clip_by_value(- alpha*distances, -200, 200)
    probs = tf.expand_dims(tf.nn.softmax(logits), 4)
    new_images = tf.reduce_sum(reshaped_centroids * probs, axis=3)

    # update cluster centers
    new_centroids = tf.reduce_sum(reshaped_images * probs, axis=[1, 2]) / (tf.reduce_sum(probs, axis=[1, 2]) + 1e-16)
    return new_images, new_centroids

def iterative_clustering_layer(source, n_clusters, sigma, alpha, noise_level_1, noise_level_2):
    source1 = source + tf.random_normal(tf.shape(source)) * noise_level_1
    source2 = source + tf.random_normal(tf.shape(source)) * noise_level_2
    centroids = sample_centroid_with_kpp(source1, 100, n_clusters, sigma)
    image, _ = rgb_clustering(source2, centroids, alpha)
    return image
