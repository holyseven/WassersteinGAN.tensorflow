__author__ = 'shekkizh'
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
from tqdm import trange
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def save_image(image, image_size, save_dir, name=""):
    """
    Save image by unprocessing assuming mean 127.5
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    image += 1
    image *= 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.reshape(image, (image_size, image_size, -1))
    misc.imsave(os.path.join(save_dir, name + "pred_image.png"), image)


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def weight_variable_xavier_initialized(shape, constant=1, name=None):
    stddev = constant * np.sqrt(2.0 / (shape[2] + shape[3]))
    return weight_variable(shape, stddev=stddev, name=name)


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def leaky_relu(x, alpha=0.2, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(name, inputs, trainable, data_format, mode,
               use_gamma=True, use_beta=True, bn_epsilon=1e-5, bn_ema=0.9, float_type=tf.float32):
    # This is a rapid version of batch normalization but it does not suit well for multiple gpus.
    # When trainable and not training mode, statistics will be frozen and parameters can be trained.

    def get_bn_variables(n_out, use_scale, use_bias, trainable, float_type):
        # TODO: not sure what to do.
        float_type = tf.float32

        if use_bias:
            beta = tf.get_variable('beta', [n_out],
                                   initializer=tf.constant_initializer(), trainable=trainable, dtype=float_type)
        else:
            beta = tf.zeros([n_out], name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [n_out],
                                    initializer=tf.constant_initializer(1.0), trainable=trainable, dtype=float_type)
        else:
            gamma = tf.ones([n_out], name='gamma')
        # x * gamma + beta

        moving_mean = tf.get_variable('moving_mean', [n_out],
                                      initializer=tf.constant_initializer(), trainable=False, dtype=float_type)
        moving_var = tf.get_variable('moving_variance', [n_out],
                                     initializer=tf.constant_initializer(1), trainable=False, dtype=float_type)
        return beta, gamma, moving_mean, moving_var

    def update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, decay):
        from tensorflow.contrib.framework import add_model_variable
        from tensorflow.python.training import moving_averages
        # TODO is there a way to use zero_debias in multi-GPU?
        update_op1 = moving_averages.assign_moving_average(
            moving_mean, batch_mean, decay, zero_debias=False,
            name='mean_ema_op')
        update_op2 = moving_averages.assign_moving_average(
            moving_var, batch_var, decay, zero_debias=False,
            name='var_ema_op')
        add_model_variable(moving_mean)
        add_model_variable(moving_var)

        # seems faster than delayed update, but might behave otherwise in distributed settings.
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')

    # ======================== Checking valid values =========================
    if data_format not in ['NHWC', 'NCHW']:
        raise TypeError(
            "Only two data formats are supported at this moment: 'NHWC' or 'NCHW', "
            "%s is an unknown data format." % data_format)
    assert inputs.get_shape().ndims == 4, 'inputs should have rank 4.'
    assert inputs.dtype == float_type, 'inputs data type is different from %s' % float_type
    if mode not in ['train', 'training', 'val', 'validation', 'test', 'eval']:
        raise TypeError("Unknown mode.")

    # ======================== Setting default values =========================
    shape = inputs.get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]
    if data_format == 'NCHW':
        n_out = shape[1]
    if mode is 'training' or mode is 'train':
        mode = 'train'
    else:
        mode = 'test'

    # ======================== Main operations =============================
    with tf.variable_scope(name):
        beta, gamma, moving_mean, moving_var = get_bn_variables(n_out, use_gamma, use_beta, trainable, float_type)

        if mode == 'train' and trainable:
            xn, batch_mean, batch_var = tf.nn.fused_batch_norm(
                inputs, gamma, beta, epsilon=bn_epsilon,
                is_training=True, data_format=data_format)
            if tf.get_variable_scope().reuse:
                return xn
            else:
                return update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, bn_ema)
        else:
            xn = tf.nn.batch_normalization(
                inputs, moving_mean, moving_var, beta, gamma, bn_epsilon)
            return xn


def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm


def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)

def save_imshow_grid(images, logs_dir, filename, shape):
    """
    Plot images in a grid of a given shape.
    """
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in trange(size, desc="Saving images"):
        grid[i].axis('off')
        grid[i].imshow(images[i])

    plt.savefig(os.path.join(logs_dir, filename))
