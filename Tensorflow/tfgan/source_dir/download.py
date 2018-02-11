from urllib.request import urlretrieve
from aitslim.cnn.parameter import CHECKPOINT_LIST
import sys
import time
import os
import tarfile


def _get_checkpoint_tar_name(model_name):
    for checkpoint_name in CHECKPOINT_LIST:
        if checkpoint_name.find(model_name) != -1:
            return checkpoint_name
    raise KeyError(
        "{} does not have pre-trained checkpoint file".format(model_name))


def may_be_download(model_name):
    tmp_dir = '/tmp'
    tmp_check_point_path = os.path.join(tmp_dir, model_name)
    checkpoint_tar_name = _get_checkpoint_tar_name(model_name)
    checkpoint_file_name = checkpoint_tar_name.split('.').pop(0) + '.ckpt'
    checkpoint_file_path = os.path.join(tmp_check_point_path,
                                        checkpoint_file_name)
    if os.path.exists(tmp_check_point_path):
        return checkpoint_file_path
    
    local_path = os.path.join(tmp_dir, checkpoint_tar_name)
    url = 'http://download.tensorflow.org/models/{}'.format(
        checkpoint_tar_name)
    urlretrieve(url, local_path, reporthook)
    tar = tarfile.open(local_path)
    tar.extractall(tmp_check_point_path)
    return checkpoint_file_path


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(
        "\r model downloading...%d%%, %d MB, %d KB/s, %d seconds passed" %
        (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
