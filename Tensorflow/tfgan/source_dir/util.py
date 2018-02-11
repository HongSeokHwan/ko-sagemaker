import os.path
import re
from tensorflow.python.platform import gfile
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

IMAGE_SIZE = 28


def get_file_list(sub_dir):
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    sub_dir_file_list = []
    
    for extension in extensions:
        file_glob = os.path.join(sub_dir, '*.' + extension)
        sub_dir_file_list.extend(gfile.Glob(file_glob))
    
    return sub_dir_file_list


def create_image_lists(image_dir):
    """
    Builds a list of training images from the file system.
    original file was retrain.py in https://github.com/tensorflow/models/
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)][1:]
    
    for sub_dir in sub_dirs:
        file_list = get_file_list(sub_dir)
        
        if not file_list:
            continue
        label_name = \
            re.sub(r'[^a-z0-9]+', ' ', os.path.basename(sub_dir).lower())
        result[label_name] = file_list
    
    return result


def get_filenames_and_labels(image_dir):
    image_list = create_image_lists(image_dir)
    image_list_keys = list(image_list.keys())
    one_hot_depth = len(image_list_keys)
    file_lists = []
    labels = []
    label_number = 0
    for key in image_list:
        file_list = image_list[key]
        file_lists.extend(file_list)
        
        n = len(file_list)
        # label_one_hot = tf.one_hot(image_list_keys.index(key), one_hot_depth)
        labels.extend([label_number for _ in range(n)])
        label_number += 1
    
    return file_lists, labels


def add_jpeg_decoding(image_string, input_width=IMAGE_SIZE,
                      input_height=IMAGE_SIZE,
                      input_depth=3, input_mean=128, input_std=128):
    decoded_image = tf.image.decode_jpeg(
        image_string, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    mul_image = tf.reshape(mul_image, [input_height, input_width, input_depth])
    return mul_image


def shallow_cnn(input_layer, mode):
    net = slim.conv2d(input_layer, 32, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    
    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    
    flat = slim.flatten(net, scope='flatten')
    dense = slim.fully_connected(flat, 1024, scope='fc/1')
    dropout = slim.dropout(dense, is_training=(mode == Modes.TRAIN))
    logits = slim.fully_connected(dropout, 10, scope='logits')
    
    return logits


def vgg_16(input_layer, mode):
    vgg = nets.vgg
    logits, _ = vgg.vgg_16(input_layer, num_classes=10,
                           is_training=(mode == Modes.TRAIN),
                           spatial_squeeze=True)
    return logits


def inception_v1(input_layer, mode):
    inception = nets.inception
    logits, _ = inception.inception_v1(input_layer, num_classes=10,
                           is_training=(mode == Modes.TRAIN),
                           spatial_squeeze=True)
    return logits
