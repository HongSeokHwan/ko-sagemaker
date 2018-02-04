import numpy as np
import os
import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'


def estimator_fn(run_config, params):
    feature_columns = [
        tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[2])]
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=2,
                                      config=run_config)


def serving_input_fn(params):
    feature_spec = {
        INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[2])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)()


def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    return _generate_input_fn(training_dir, 'xor_train.csv')


def eval_input_fn(training_dir, params):
    """Returns input function that would feed the model during evaluation"""
    return _generate_input_fn(training_dir, 'xor_test.csv')


def _generate_input_fn(training_dir, training_filename):
    data_file = os.path.join(training_dir, training_filename)
    train_set = np.loadtxt(fname=data_file, delimiter=',')
    
    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: train_set[:, 0:-1]},
        y=np.array(train_set[:, [-1]]),
        num_epochs=None,
        shuffle=True)()
