---
layout: post
title:  "SageMaker Tutorial"
author: huhuta
categories: SageMaker
comments: true
---

# Sagemaker Xor tutorial
---

## prerequisite
Tensorflow Estimator에 대해서 처음 접하신다면 아래 페이지를 먼저 참고하시는 것을 추천 드립니다.  
[Estimator xor tutorial link](http://blog.sagemaker.io/tensorflow/2018/02/02/xor.html)

예제 코드는  [여기](https://github.com/HongSeokHwan/ko-sagemaker/tree/master/Sagemaker/xor)에 있습니다. <br>
혹은 아래 script를 terminal에서 실행하면 됩니다.
```bash
git clone https://github.com/HongSeokHwan/ko-sagemaker.git
cd ko-sagemaker/Sagemaker/xor
jupyter notebook
```

### directory 구조
---
directory내에 Tensorflow Estimator tutorial 과 다르게 추가된 파일은 xor_classifier.py  
입니다. 이 파일에 대해서는 아래에서 설명하도록 하겠습니다.


```python
!tree
```

    .
    ├── __init__.py
    ├── data
    │   ├── xor_test.csv
    │   └── xor_train.csv
    ├── xor-sagemaekr-train-evaluate.ipynb
    ├── xor_all.ipynb
    └── xor_classifier.py
    
    1 directory, 6 files


## Sagemaker에서 Tensorflow estimator를 사용할때의 차이점
---
1. input function을 작성할시 parameter로 s3 bucket의 위치를 전달받는다.
2. entry point 파일에 input function, estimator가 정의되어 있어야 한다.
3. input function에는 train, evaluation, serving을 위한 세가지 input function이 정의되 있어야 한다.
4. predefined estimator를 사용할 경우 estimator_fn을 작성하고 custom estimator를 사용할 경우 model_fn을 작성한다.
5. estimator config는 sagemaker에서 정의 하므로 run_config를 인자로 받는다. (**cluster spec**)


## S3에 Data Upload 하기
위 1번 항목에 서술한 것처럼 S3 bucket내에 학습, 평가에 사용할  
Data가 미리 upload 되어 있어야 합니다.  
Python Sagemaker SDK를 활용해서 csv 파일들을 Upload하겠습니다.
  


```python
import sagemaker
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()

# path = local data의 위치
inputs = sagemaker_session.upload_data(path='data',
                                       bucket='tutorial-dudaji',
                                       key_prefix='xor/data')
print(inputs)
```

    s3://tutorial-dudaji/xor/data


inputs 에 저장된 s3 location은 나중에 train, evaluate과정에서 전달됩니다.  

### Sagemaker Tensorflow Entry point 파일 해부
---
   
```python
import numpy as np
import os
import tensorflow as tf


# run_config를 인자로 받아 Sagemaker에서 정의한 configure에 따라서 학습
# default config와의 가장큰 차이는 cluster spec 차이
def estimator_fn(run_config, params):
    feature_columns = [
        tf.feature_column.numeric_column('x', shape=[2])]
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=2,
                                      config=run_config)


# Sagemaker에서 host할 때 필요한 input function 
def serving_input_fn(params):
    feature_spec = {
        'x': tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)()


#학습 할때 input function, training_dir에는 s3 bucket내 data의 위치가 전달된다.
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
        x={"x": train_set[:, 0:-1]},
        y=np.array(train_set[:, [-1]]),
        num_epochs=None,
        shuffle=True)()
```

### Sagemaker로 Estimator Train, Evaluate 시키기
---


```python
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role

role = get_execution_role()

xor_classifier = TensorFlow(entry_point='xor_classifier.py',
                            role=role,
                            train_instance_count=1,
                            train_instance_type='ml.c4.xlarge',
                            training_steps=1000,
                            evaluation_steps=100)
```

### 학습 시작
---
학습시 fit을 실행시키면 되며 이때 인자로 data가 있는 s3 의 위치를 넘겨줍니다.
이 위치가 input_fn에 전달됩니다.

![image.png](attachment:image.png)


```python
xor_classifier.fit(inputs, run_tensorboard_locally=True)
```

    INFO:sagemaker:TensorBoard 0.1.7 at http://localhost:6006
    INFO:sagemaker:Created S3 bucket: sagemaker-us-east-1-728064587231
    INFO:sagemaker:Creating training-job with name: sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918


    .........................................................
    [31mexecuting startup script (first run)[0m
    [31m2018-01-24 08:35:23,957 INFO - root - running container entrypoint[0m
    [31m2018-01-24 08:35:23,957 INFO - root - starting train task[0m
    [31m2018-01-24 08:35:25,558 INFO - botocore.vendored.requests.packages.urllib3.connectionpool - Starting new HTTP connection (1): 169.254.170.2[0m
    [31m2018-01-24 08:35:26,444 INFO - botocore.vendored.requests.packages.urllib3.connectionpool - Starting new HTTPS connection (1): s3.amazonaws.com[0m
    [31m2018-01-24 08:35:26,615 INFO - botocore.vendored.requests.packages.urllib3.connectionpool - Starting new HTTPS connection (1): s3.amazonaws.com[0m
    [31mINFO:tensorflow:----------------------TF_CONFIG--------------------------[0m
    [31mINFO:tensorflow:{"environment": "cloud", "cluster": {"master": ["algo-1:2222"]}, "task": {"index": 0, "type": "master"}}[0m
    [31mINFO:tensorflow:---------------------------------------------------------[0m
    [31mINFO:tensorflow:going to training[0m
    [31m2018-01-24 08:35:26,680 INFO - root - creating RunConfig:[0m
    [31m2018-01-24 08:35:26,680 INFO - root - {'save_checkpoints_secs': 300}[0m
    [31m2018-01-24 08:35:26,680 INFO - root - invoking estimator_fn[0m
    [31mINFO:tensorflow:Using config: {'_model_dir': u's3://sagemaker-us-east-1-728064587231/sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918/checkpoints', '_save_checkpoints_secs': 300, '_num_ps_replicas': 0, '_keep_checkpoint_max': 5, '_session_config': None, '_tf_random_seed': None, '_task_type': u'master', '_environment': u'cloud', '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fac2db01750>, '_tf_config': gpu_options {
      per_process_gpu_memory_fraction: 1.0[0m
    [31m}[0m
    [31m, '_num_worker_replicas': 1, '_task_id': 0, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_evaluation_master': '', '_keep_checkpoint_every_n_hours': 10000, '_master': '', '_log_step_count_steps': 100}[0m
    [31m2018-01-24 08:35:26,681 INFO - root - creating Experiment:[0m
    [31m2018-01-24 08:35:26,681 INFO - root - {'min_eval_frequency': 1000}[0m
    [31mWARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/monitors.py:267: __init__ (from tensorflow.contrib.learn.python.learn.monitors) is deprecated and will be removed after 2016-12-05.[0m
    [31mInstructions for updating:[0m
    [31mMonitors are deprecated. Please use tf.train.SessionRunHook.[0m
    [31mINFO:tensorflow:Create CheckpointSaverHook.[0m
    [31m2018-01-24 08:35:28.554590: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA[0m
    [31mINFO:tensorflow:Saving checkpoints for 1 into s3://sagemaker-us-east-1-728064587231/sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918/checkpoints/model.ckpt.[0m
    [31mINFO:tensorflow:loss = 91.227, step = 1[0m
    [31mINFO:tensorflow:global_step/sec: 708.672[0m
    [31mINFO:tensorflow:loss = 0.276273, step = 101 (0.141 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 830.462[0m
    [31mINFO:tensorflow:loss = 0.0866725, step = 201 (0.120 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 917.962[0m
    [31mINFO:tensorflow:loss = 0.0548231, step = 301 (0.109 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 907.086[0m
    [31mINFO:tensorflow:loss = 0.0328132, step = 401 (0.110 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 888.044[0m
    [31mINFO:tensorflow:loss = 0.0196804, step = 501 (0.113 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 879.918[0m
    [31mINFO:tensorflow:loss = 0.0187322, step = 601 (0.114 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 874.256[0m
    [31mINFO:tensorflow:loss = 0.016088, step = 701 (0.114 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 893.057[0m
    [31mINFO:tensorflow:loss = 0.0120865, step = 801 (0.112 sec)[0m
    [31mINFO:tensorflow:global_step/sec: 899.507[0m
    [31mINFO:tensorflow:loss = 0.011387, step = 901 (0.111 sec)[0m
    [31mWARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.[0m
    [31mWARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.[0m
    [31mINFO:tensorflow:Starting evaluation at 2018-01-24-08:35:33[0m
    [31mINFO:tensorflow:Restoring parameters from s3://sagemaker-us-east-1-728064587231/sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918/checkpoints/model.ckpt-1[0m
    [31mINFO:tensorflow:Evaluation [1/100][0m
    [31mINFO:tensorflow:Evaluation [2/100][0m
    [31mINFO:tensorflow:Evaluation [3/100][0m
    [31mINFO:tensorflow:Evaluation [4/100][0m
    [31mINFO:tensorflow:Evaluation [5/100][0m
    [31mINFO:tensorflow:Evaluation [6/100][0m
    [31mINFO:tensorflow:Evaluation [7/100][0m
    [31mINFO:tensorflow:Evaluation [8/100][0m
    [31mINFO:tensorflow:Evaluation [9/100][0m
    [31mINFO:tensorflow:Evaluation [10/100][0m
    [31mINFO:tensorflow:Evaluation [11/100][0m
    [31mINFO:tensorflow:Evaluation [12/100][0m
    [31mINFO:tensorflow:Evaluation [13/100][0m
    [31mINFO:tensorflow:Evaluation [14/100][0m
    [31mINFO:tensorflow:Evaluation [15/100][0m
    [31mINFO:tensorflow:Evaluation [16/100][0m
    [31mINFO:tensorflow:Evaluation [17/100][0m
    [31mINFO:tensorflow:Evaluation [18/100][0m
    [31mINFO:tensorflow:Evaluation [19/100][0m
    [31mINFO:tensorflow:Evaluation [20/100][0m
    [31mINFO:tensorflow:Evaluation [21/100][0m
    [31mINFO:tensorflow:Evaluation [22/100][0m
    [31mINFO:tensorflow:Evaluation [23/100][0m
    [31mINFO:tensorflow:Evaluation [24/100][0m
    [31mINFO:tensorflow:Evaluation [25/100][0m
    [31mINFO:tensorflow:Evaluation [26/100][0m
    [31mINFO:tensorflow:Evaluation [27/100][0m
    [31mINFO:tensorflow:Evaluation [28/100][0m
    [31mINFO:tensorflow:Evaluation [29/100][0m
    [31mINFO:tensorflow:Evaluation [30/100][0m
    [31mINFO:tensorflow:Evaluation [31/100][0m
    [31mINFO:tensorflow:Evaluation [32/100][0m
    [31mINFO:tensorflow:Evaluation [33/100][0m
    [31mINFO:tensorflow:Evaluation [34/100][0m
    [31mINFO:tensorflow:Evaluation [35/100][0m
    [31mINFO:tensorflow:Evaluation [36/100][0m
    [31mINFO:tensorflow:Evaluation [37/100][0m
    [31mINFO:tensorflow:Evaluation [38/100][0m
    [31mINFO:tensorflow:Evaluation [39/100][0m
    [31mINFO:tensorflow:Evaluation [40/100][0m
    [31mINFO:tensorflow:Evaluation [41/100][0m
    [31mINFO:tensorflow:Evaluation [42/100][0m
    [31mINFO:tensorflow:Evaluation [43/100][0m
    [31mINFO:tensorflow:Evaluation [44/100][0m
    [31mINFO:tensorflow:Evaluation [45/100][0m
    [31mINFO:tensorflow:Evaluation [46/100][0m
    [31mINFO:tensorflow:Evaluation [47/100][0m
    [31mINFO:tensorflow:Evaluation [48/100][0m
    [31mINFO:tensorflow:Evaluation [49/100][0m
    [31mINFO:tensorflow:Evaluation [50/100][0m
    [31mINFO:tensorflow:Evaluation [51/100][0m
    [31mINFO:tensorflow:Evaluation [52/100][0m
    [31mINFO:tensorflow:Evaluation [53/100][0m
    [31mINFO:tensorflow:Evaluation [54/100][0m
    [31mINFO:tensorflow:Evaluation [55/100][0m
    [31mINFO:tensorflow:Evaluation [56/100][0m
    [31mINFO:tensorflow:Evaluation [57/100][0m
    [31mINFO:tensorflow:Evaluation [58/100][0m
    [31mINFO:tensorflow:Evaluation [59/100][0m
    [31mINFO:tensorflow:Evaluation [60/100][0m
    [31mINFO:tensorflow:Evaluation [61/100][0m
    [31mINFO:tensorflow:Evaluation [62/100][0m
    [31mINFO:tensorflow:Evaluation [63/100][0m
    [31mINFO:tensorflow:Evaluation [64/100][0m
    [31mINFO:tensorflow:Evaluation [65/100][0m
    [31mINFO:tensorflow:Evaluation [66/100][0m
    [31mINFO:tensorflow:Evaluation [67/100][0m
    [31mINFO:tensorflow:Evaluation [68/100][0m
    [31mINFO:tensorflow:Evaluation [69/100][0m
    [31mINFO:tensorflow:Evaluation [70/100][0m
    [31mINFO:tensorflow:Evaluation [71/100][0m
    [31mINFO:tensorflow:Evaluation [72/100][0m
    [31mINFO:tensorflow:Evaluation [73/100][0m
    [31mINFO:tensorflow:Evaluation [74/100][0m
    [31mINFO:tensorflow:Evaluation [75/100][0m
    [31mINFO:tensorflow:Evaluation [76/100][0m
    [31mINFO:tensorflow:Evaluation [77/100][0m
    [31mINFO:tensorflow:Evaluation [78/100][0m
    [31mINFO:tensorflow:Evaluation [79/100][0m
    [31mINFO:tensorflow:Evaluation [80/100][0m
    [31mINFO:tensorflow:Evaluation [81/100][0m
    [31mINFO:tensorflow:Evaluation [82/100][0m
    [31mINFO:tensorflow:Evaluation [83/100][0m
    [31mINFO:tensorflow:Evaluation [84/100][0m
    [31mINFO:tensorflow:Evaluation [85/100][0m
    [31mINFO:tensorflow:Evaluation [86/100][0m
    [31mINFO:tensorflow:Evaluation [87/100][0m
    [31mINFO:tensorflow:Evaluation [88/100][0m
    [31mINFO:tensorflow:Evaluation [89/100][0m
    [31mINFO:tensorflow:Evaluation [90/100][0m
    [31mINFO:tensorflow:Evaluation [91/100][0m
    [31mINFO:tensorflow:Evaluation [92/100][0m
    [31mINFO:tensorflow:Evaluation [93/100][0m
    [31mINFO:tensorflow:Evaluation [94/100][0m
    [31mINFO:tensorflow:Evaluation [95/100][0m
    [31mINFO:tensorflow:Evaluation [96/100][0m
    [31mINFO:tensorflow:Evaluation [97/100][0m
    [31mINFO:tensorflow:Evaluation [98/100][0m
    [31mINFO:tensorflow:Evaluation [99/100][0m
    [31mINFO:tensorflow:Evaluation [100/100][0m
    [31mINFO:tensorflow:Finished evaluation at 2018-01-24-08:35:34[0m
    [31mINFO:tensorflow:Saving dict for global step 1: accuracy = 0.499609, accuracy_baseline = 0.500391, auc = 1.0, auc_precision_recall = 1.0, average_loss = 0.674477, global_step = 1, label/mean = 0.499609, loss = 86.3331, prediction/mean = 0.522813[0m
    [31mINFO:tensorflow:Validation (step 1000): loss = 86.3331, accuracy_baseline = 0.500391, global_step = 1, auc = 1.0, prediction/mean = 0.522813, label/mean = 0.499609, average_loss = 0.674477, auc_precision_recall = 1.0, accuracy = 0.499609[0m
    [31mINFO:tensorflow:Saving checkpoints for 1000 into s3://sagemaker-us-east-1-728064587231/sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918/checkpoints/model.ckpt.[0m
    [31mINFO:tensorflow:Loss for final step: 0.0106414.[0m
    [31mWARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.[0m
    [31mWARNING:tensorflow:Casting <dtype: 'float32'> labels to bool.[0m
    [31mINFO:tensorflow:Starting evaluation at 2018-01-24-08:35:39[0m
    [31mINFO:tensorflow:Restoring parameters from s3://sagemaker-us-east-1-728064587231/sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918/checkpoints/model.ckpt-1000[0m
    [31mINFO:tensorflow:Evaluation [1/100][0m
    [31mINFO:tensorflow:Evaluation [2/100][0m
    [31mINFO:tensorflow:Evaluation [3/100][0m
    [31mINFO:tensorflow:Evaluation [4/100][0m
    [31mINFO:tensorflow:Evaluation [5/100][0m
    [31mINFO:tensorflow:Evaluation [6/100][0m
    [31mINFO:tensorflow:Evaluation [7/100][0m
    [31mINFO:tensorflow:Evaluation [8/100][0m
    [31mINFO:tensorflow:Evaluation [9/100][0m
    [31mINFO:tensorflow:Evaluation [10/100][0m
    [31mINFO:tensorflow:Evaluation [11/100][0m
    [31mINFO:tensorflow:Evaluation [12/100][0m
    [31mINFO:tensorflow:Evaluation [13/100][0m
    [31mINFO:tensorflow:Evaluation [14/100][0m
    [31mINFO:tensorflow:Evaluation [15/100][0m
    [31mINFO:tensorflow:Evaluation [16/100][0m
    [31mINFO:tensorflow:Evaluation [17/100][0m
    [31mINFO:tensorflow:Evaluation [18/100][0m
    [31mINFO:tensorflow:Evaluation [19/100][0m
    [31mINFO:tensorflow:Evaluation [20/100][0m
    [31mINFO:tensorflow:Evaluation [21/100][0m
    [31mINFO:tensorflow:Evaluation [22/100][0m
    [31mINFO:tensorflow:Evaluation [23/100][0m
    [31mINFO:tensorflow:Evaluation [24/100][0m
    [31mINFO:tensorflow:Evaluation [25/100][0m
    [31mINFO:tensorflow:Evaluation [26/100][0m
    [31mINFO:tensorflow:Evaluation [27/100][0m
    [31mINFO:tensorflow:Evaluation [28/100][0m
    [31mINFO:tensorflow:Evaluation [29/100][0m
    [31mINFO:tensorflow:Evaluation [30/100][0m
    [31mINFO:tensorflow:Evaluation [31/100][0m
    [31mINFO:tensorflow:Evaluation [32/100][0m
    [31mINFO:tensorflow:Evaluation [33/100][0m
    [31mINFO:tensorflow:Evaluation [34/100][0m
    [31mINFO:tensorflow:Evaluation [35/100][0m
    [31mINFO:tensorflow:Evaluation [36/100][0m
    [31mINFO:tensorflow:Evaluation [37/100][0m
    [31mINFO:tensorflow:Evaluation [38/100][0m
    [31mINFO:tensorflow:Evaluation [39/100][0m
    [31mINFO:tensorflow:Evaluation [40/100][0m
    [31mINFO:tensorflow:Evaluation [41/100][0m
    [31mINFO:tensorflow:Evaluation [42/100][0m
    [31mINFO:tensorflow:Evaluation [43/100][0m
    [31mINFO:tensorflow:Evaluation [44/100][0m
    [31mINFO:tensorflow:Evaluation [45/100][0m
    [31mINFO:tensorflow:Evaluation [46/100][0m
    [31mINFO:tensorflow:Evaluation [47/100][0m
    [31mINFO:tensorflow:Evaluation [48/100][0m
    [31mINFO:tensorflow:Evaluation [49/100][0m
    [31mINFO:tensorflow:Evaluation [50/100][0m
    [31mINFO:tensorflow:Evaluation [51/100][0m
    [31mINFO:tensorflow:Evaluation [52/100][0m
    [31mINFO:tensorflow:Evaluation [53/100][0m
    [31mINFO:tensorflow:Evaluation [54/100][0m
    [31mINFO:tensorflow:Evaluation [55/100][0m
    [31mINFO:tensorflow:Evaluation [56/100][0m
    [31mINFO:tensorflow:Evaluation [57/100][0m
    [31mINFO:tensorflow:Evaluation [58/100][0m
    [31mINFO:tensorflow:Evaluation [59/100][0m
    [31mINFO:tensorflow:Evaluation [60/100][0m
    [31mINFO:tensorflow:Evaluation [61/100][0m
    [31mINFO:tensorflow:Evaluation [62/100][0m
    [31mINFO:tensorflow:Evaluation [63/100][0m
    [31mINFO:tensorflow:Evaluation [64/100][0m
    [31mINFO:tensorflow:Evaluation [65/100][0m
    [31mINFO:tensorflow:Evaluation [66/100][0m
    [31mINFO:tensorflow:Evaluation [67/100][0m
    [31mINFO:tensorflow:Evaluation [68/100][0m
    [31mINFO:tensorflow:Evaluation [69/100][0m
    [31mINFO:tensorflow:Evaluation [70/100][0m
    [31mINFO:tensorflow:Evaluation [71/100][0m
    [31mINFO:tensorflow:Evaluation [72/100][0m
    [31mINFO:tensorflow:Evaluation [73/100][0m
    [31mINFO:tensorflow:Evaluation [74/100][0m
    [31mINFO:tensorflow:Evaluation [75/100][0m
    [31mINFO:tensorflow:Evaluation [76/100][0m
    [31mINFO:tensorflow:Evaluation [77/100][0m
    [31mINFO:tensorflow:Evaluation [78/100][0m
    [31mINFO:tensorflow:Evaluation [79/100][0m
    [31mINFO:tensorflow:Evaluation [80/100][0m
    [31mINFO:tensorflow:Evaluation [81/100][0m
    [31mINFO:tensorflow:Evaluation [82/100][0m
    [31mINFO:tensorflow:Evaluation [83/100][0m
    [31mINFO:tensorflow:Evaluation [84/100][0m
    [31mINFO:tensorflow:Evaluation [85/100][0m
    [31mINFO:tensorflow:Evaluation [86/100][0m
    [31mINFO:tensorflow:Evaluation [87/100][0m
    [31mINFO:tensorflow:Evaluation [88/100][0m
    [31mINFO:tensorflow:Evaluation [89/100][0m
    [31mINFO:tensorflow:Evaluation [90/100][0m
    [31mINFO:tensorflow:Evaluation [91/100][0m
    [31mINFO:tensorflow:Evaluation [92/100][0m
    [31mINFO:tensorflow:Evaluation [93/100][0m
    [31mINFO:tensorflow:Evaluation [94/100][0m
    [31mINFO:tensorflow:Evaluation [95/100][0m
    [31mINFO:tensorflow:Evaluation [96/100][0m
    [31mINFO:tensorflow:Evaluation [97/100][0m
    [31mINFO:tensorflow:Evaluation [98/100][0m
    [31mINFO:tensorflow:Evaluation [99/100][0m
    [31mINFO:tensorflow:Evaluation [100/100][0m
    [31mINFO:tensorflow:Finished evaluation at 2018-01-24-08:35:42[0m
    [31mINFO:tensorflow:Saving dict for global step 1000: accuracy = 1.0, accuracy_baseline = 0.500703, auc = 1.0, auc_precision_recall = 1.0, average_loss = 7.68955e-05, global_step = 1000, label/mean = 0.499297, loss = 0.00984262, prediction/mean = 0.49933[0m
    [31mINFO:tensorflow:Restoring parameters from s3://sagemaker-us-east-1-728064587231/sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918/checkpoints/model.ckpt-1000[0m
    [31mINFO:tensorflow:Assets added to graph.[0m
    [31mINFO:tensorflow:No assets to write.[0m
    [31mINFO:tensorflow:SavedModel written to: s3://sagemaker-us-east-1-728064587231/sagemaker-tensorflow-py2-cpu-2018-01-24-08-29-48-918/checkpoints/export/Servo/temp-1516782942/saved_model.pb[0m
    [31mINFO:tensorflow:writing success training[0m
    [31m2018-01-24 08:35:46,866 INFO - botocore.vendored.requests.packages.urllib3.connectionpool - Starting new HTTPS connection (1): s3.amazonaws.com[0m
    [31m2018-01-24 08:35:46,963 INFO - tf_container.serve - Downloaded saved model at /opt/ml/model/export/Servo/1516782942/saved_model.pb[0m
    ===== Job Complete =====

