#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from shutil import copyfile


CLASSES = ('__background__', # always index 0
         'adidas', 'apple', 'bmw', 'chimay', 'corona', 'erdinger',
         'fedex', 'ford', 'google', 'heineken', 'milka', 'paulaner',
         'rittersport', 'singha', 'stellaartois', 'tsingtao', 'aldi',
         'becks', 'carlsberg', 'cocacola', 'dhl', 'esso', 'ferrari',
         'fosters', 'guiness', 'HP', 'nvidia', 'pepsi', 'shell',
         'starbucks', 'texaco', 'ups')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'vgg_m': ('fl',
                  'vgg_cnn_m_1024_rpn_stage1_iter_80000.caffemodel'),
        'fl': ('FL',
                  'vgg16_faster_rcnn_iter_80000.caffemodel') }

bboxArray = []
scoreArray = []
classArray = []
def vis_detections(class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bboxArray.append(dets[i, :4])
        scoreArray.append(dets[i, -1])
        classArray.append(class_name)

def write_bboxes(im, thresh=0.5):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(bboxArray)):
        bbox = bboxArray[i]
        score = scoreArray[i]
        class_name = classArray[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('Detections with '
                  'p(obj | box) >= {:.1f}').format(thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.3
    global bboxArray
    global scoreArray
    global classArray
    bboxArray = []
    scoreArray = []
    classArray = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(cls, dets, thresh=CONF_THRESH)
        
    print len(bboxArray)
    write_bboxes(im, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    prototxt = os.path.join('/home/andras/github/logoretrieval/py_faster_rcnn/models/fl/VGG16/faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['145074777.jpg', '2636557914.jpg', '3426961300.jpg', '3807811747.jpg', '4425073529.jpg', '4761260517.jpg', '1468736300.jpg', '276515560.jpg', '3438303866.jpg', '3948482004.jpg', 
'4499338915.jpg', '4763207899.jpg',
        '2450743885.jpg', '2970796187.jpg', '3441398196.jpg', '4061674634.jpg', 
'451265524.jpg', '4763210295.jpg',
        '2534155497.jpg', '3294282629.jpg', '3541292073.jpg', '4359633049.jpg', 
'4605630935.jpg', '4955394412.jpg',
        '2540056504.jpg', '3385248586.jpg', '3703822708.jpg', '4406847336.jpg', 
'4745120286.jpg', '79242964.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
        plt.savefig('/home/andras/github/logoretrieval/py_faster_rcnn/result/' + im_name)


    #plt.savefig('/home/andras/github/py-faster-rcnn/tools/1.png')
    plt.show()

