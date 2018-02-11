from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import time
import functools
import sys
import os
import numpy as np
import scipy.misc
from six.moves import xrange
import tensorflow as tf

sys.path.insert(0, os.path.abspath('./source_dir'))
# from source_dir.cnn import *
from Tensorflow.tfgan.source_dir.cnn import *

tfgan = tf.contrib.gan
slim = tf.contrib.slim
layers = tf.contrib.layers
ds = tf.contrib.distributions
DATA_DIR = '../../datas/styles_images'
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)


def visualize_training_generator(train_step_num, data_np):
    """Visualize generator outputs during training.
    
    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        data: Data to plot. A numpy array, most likely from an evaluated TensorFlow
            tensor.
    """
    print('Training step: %i' % train_step_num)
    plt.axis('off')
    plt.imshow(np.squeeze(data_np))
    plt.show()


def visualize_image(tensor_to_visualize):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    # images_np = np.add(images_np, 1)
    # print(images_np)
    plt.imshow(np.squeeze(images_np))
    plt.show()


def generator_fn(noise, weight_decay=2.5e-5):
    with slim.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 256)
        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 3, 4, normalizer_fn=None,
                            activation_fn=tf.tanh)
        
        return net


def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5):
    """Discriminator network on MNIST digits.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
    
    Returns:
        Logits for the probability that the image is real.
    """
    with slim.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)


noise, images = train_input_fn(DATA_DIR, None)

gan_model = tfgan.gan_model(
    generator_fn,
    discriminator_fn,
    real_data=images,
    generator_inputs=tf.random_normal([1, 64]))

generated_data_to_visualize = gan_model.generated_data

# We can use the minimax loss from the original paper.
vanilla_gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.minimax_generator_loss,
    discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)

generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    vanilla_gan_loss,
    generator_optimizer,
    discriminator_optimizer)

train_step_fn = tfgan.get_sequential_train_steps()
global_step = tf.train.get_or_create_global_step()

print(generated_data_to_visualize)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):
        start_time = time.time()
        for i in range(801):
            cur_loss, _ = train_step_fn(
                sess, gan_train_ops, global_step, train_step_kwargs={})
            if i % 100 == 0:
                print(cur_loss)
                visualize_training_generator(i, sess.run(generated_data_to_visualize))

