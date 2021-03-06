import tensorflow as tf


def rgb_to_lab(srgb):
    srgb = srgb / 2.0 + 0.5
    with tf.name_scope("rgb_to_lab"):
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((
                                                                 srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                                                                                                  xyz_normalized_pixels ** (
                                                                                                  1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def rgb_to_hsv(images):
    images = images / 2.0 + 0.5
    pixels = tf.reshape(images, [-1, 3])
    r, g, b = tf.unstack(pixels, axis=1)

    maxv = tf.maximum(r, tf.maximum(g,b))
    minv = tf.minimum(r, tf.minimum(g,b))
    df = maxv - minv

    maxv_nonzero = maxv + 1e-8
    df_nonzero = df + 1e-8

    h = tf.where(r > maxv - 1e-8, (1.0 + (g - b) / df_nonzero) / 6, tf.where(g > maxv - 1e-8, (3.0 + (b - r) / df_nonzero) / 6, (5.0 + (r - g) / df_nonzero) / 6))
    s = df / maxv_nonzero
    v = maxv

    pixels_hsv = tf.stack([h, s, v], axis=1)
    return tf.reshape(pixels_hsv, tf.shape(images)) * 2.0 - 1.0

