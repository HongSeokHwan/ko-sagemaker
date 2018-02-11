from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from tensorflow.contrib.metrics import confusion_matrix
import os
import tensorflow as tf
from util import *

INPUT_TENSOR_NAME = 'inputs'
LEARNING_RATE = 0.001
SIGNATURE_NAME = 'predictions'
BATCH_SIZE = 1


def serving_input_fn(params):
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 28, 28, 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, params):
    return _generate_input_fn(training_dir, 'styles')


def eval_input_fn(training_dir, params):
    return _generate_input_fn(training_dir, 'validation_styles')


def predict_input_fn():
    noise = tf.random_normal([batch_size, noise_dims])
    return noise


def _parse_function(filename, label):
    input_string = tf.read_file(filename)
    resized_image = add_jpeg_decoding(input_string)
    return resized_image, label


def process_fn(data_dir, mode_dir):
    path = os.path.join(data_dir, mode_dir)
    filenames, labels = get_filenames_and_labels(path)
    data_set = tf.data.Dataset.from_tensor_slices((filenames, labels))
    data_set = data_set.repeat()
    data_set = data_set.map(_parse_function)
    data_set = data_set.shuffle(buffer_size=500)
    batched_dataset = data_set.batch(BATCH_SIZE)
    iterator = batched_dataset.make_one_shot_iterator()
    return iterator


def _generate_input_fn(data_dir, mode_dir):
    iterator = process_fn(data_dir, mode_dir)
    images, labels = iterator.get_next()
    noise = tf.random_normal([BATCH_SIZE, 64], dtype=tf.float32)
    return noise, images
