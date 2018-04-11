---
layout: post
title:  "py-faster-rcnn demo.py 분석"
author: huhuta
categories: Machine_learning
comments: true
---

# py-faster-rcnn demo.py 분석

## Result
[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)에 따라서 caffe 설치후 demo.py 를 실행하면 볼 수 있는 사진 입니다.
![demo_picture](https://turbosnu.files.wordpress.com/2017/05/faster_rcnn_tensorflow.png?w=656)
[사진 출처](https://turbosnu.files.wordpress.com/2017/05/faster_rcnn_tensorflow.png?w=656)

### model loading

먼저 demo를 실행하기전 caffemodel과 prototxt의 위치를 지정해주어야 합니다.
```python
prototxt = 'example.prototxt'
caffemodel = 'example.caffemodel'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
```

prototxt 에는 network가 define 되어 있고.  
caffemodel에는 network에 필요한 값들이 저장되어 있습니다. 

예를 들어서
prototxt에 Y = Wx + b 라는 식이 에 들어 있고,  
W = 1, b = 2 라는 값이 caffemodel에 저장되어 있다면,  
input으로 x = 2를 건네서 Y를 계산하면 4라는 output이 나옵니다.  

## Input Layer
```
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 224
  dim: 224
}

```
object detection에는 image가 사용되므로 input layer (feature)는  
(1, 224, 224, 3) 1개의 224 x 224 size의 3channel (RGB) 이미지를 input으로 받습니다.

## pre-processing

pre-processing 과정에서 configure file에 설정된 대로  
mean substraction과 image size scale등을 해줍니다.
```python
im_orig -= cfg.PIXEL_MEANS

im_shape = im_orig.shape
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])

processed_ims = []
im_scale_factors = []

for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)
```

## net fowarding
그리고 fowarding을 시켜 score 와 box들을 얻습니다.

```python
blobs_out = net.forward(**forward_kwargs)

assert len(im_scales) == 1, "Only single-image batch implemented"
rois = net.blobs['rois'].data.copy()
# unscale back to raw image space
boxes = rois[:, 1:5] / im_scales[0]

# use softmax estimated probabilities
scores = blobs_out['cls_prob']

# Apply bounding-box regression deltas
box_deltas = blobs_out['bbox_pred']
pred_boxes = bbox_transform_inv(boxes, box_deltas)
pred_boxes = clip_boxes(pred_boxes, im.shape)

return scores, pred_boxes
```

return된 scores,와 boxes의 아웃풋을 살펴보면 
```python
scores, boxes = im_detect(net, im)
print(scores.shape, boxes.shape)
```

(300, 21), (300, 84) 인 것을 볼 수 있습니다.
```
Loaded network ZF_faster_rcnn_final.caffemodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/images.jpeg
((300, 21), (300, 84))
Detection took 1.606s for 300 object proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demo for data/demo/000456.jpg
((300, 21), (300, 84))
Detection took 1.192s for 300 object proposals
```

300개의 proposal 마다 각각 20 + 1 개의 클래스에 대한 score, 그리고 각 클래스의 box 위치가
담겨 있습니다.

>ex.  
>bicycle (score=0.88, box=[12, 21, 300, 322])  
>bird (score=0.22 box=[14, 21, 300, 322])  
>...
```
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
```

## post-processing
300개의 proposal에는 score가 담겨 있고 이 score는 낮을 수도 높을 수도 있습니다.  
score중 80% 이상의 confidence를 보인 것만 아래 코드에서 추립니다.
```python
CONF_THRESH = 0.8
NMS_THRESH = 0.3
for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    vis_detections(im, cls, dets, thresh=CONF_THRESH)
```

plot으로 이미지를 보여주기 위해서 score와 box를 합쳐서 length가 5인 list들로 묶어줍니다.  
  
코드 중에 nms라는 것이 있습니다. nms는 Non maximum proposal의 약자입니다.  
nms가 어떤 역할을 하는지는 아래 사진을 보면 이해하기 쉬울 거라고 생각합니다.  
![nms](https://www.mpi-inf.mpg.de/fileadmin/_processed_/7/f/csm_teaser_2017_cvpr_gnet_df007dc3b7.png)  
[사진 출처](https://www.mpi-inf.mpg.de/fileadmin/_processed_/7/f/csm_teaser_2017_cvpr_gnet_df007dc3b7.png)

이렇게 output 중 의미 있는 detection만 골라낸 후   
```python
def vis_detections
```
에서 detection된 부분을 박스표시해서 이미지를 보여주는 것으로 끝입니다.
