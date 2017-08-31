#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import os
import shutil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)
    
    cfg.TEST.HAS_RPN = True
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.IMS_PER_BATCH = 1
    cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
    cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    cfg.TRAIN.RPN_BATCHSIZE = 256
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.BG_THRESH_LO = 0.0

    if os.path.exists('data/cache'):
        shutil.rmtree('data/cache')



    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    roidb_det = None
    #imdb_det, roidb_det = combined_roidb('srf_ice_good_logo+srf_ice_good_occlusion_logo')
    #imdb, roidb = combined_roidb('synmetu_ta_train_all')
    #imdb, roidb = combined_roidb('flbl_detection_train_all+bl_detection_train')
    imdb, roidb = combined_roidb('fl27_detection_train+bl_detection_train+toplogo_detection_train+ownlogos_det')
    #imdb_det, roidb_det = combined_roidb('fl27_detection_train+bl_detection_train+toplogo_detection_train+logos32plus_detection+flbl_detection_train_all')
    #imdb, roidb = combined_roidb('srf_football_logo+srf_ice_good_logo+srf_ice_good_occlusion_logo+srf_ice_bad_logo+srf_ice_bad_occlusion_logo+srf_ski_good_logo')

   
    #imdb, roidb = combined_roidb(args.imdb_name)
    #imdb, roidb = combined_roidb('srf_ice_good+srf_ice_good_occlusion+srf_ice_bad+srf_ice_bad_occlusion+srf_ski+srf_football+fl_train+fl_val_logo+fl_test_logo+fl27_train+bl_train+bl_test+toplogo_train+logos32plus')
    #imdb, roidb = combined_roidb('srf_ice_good+srf_ice_good_occlusion+srf_ice_bad+srf_ice_bad_occlusion+srf_ski+srf_football+fl_train+fl_val_logo+fl27_train+bl_train+bl_test+toplogo_train+logos32plus')

    #imdb, roidb = combined_roidb('fl_train+fl_val_logo')
    #imdb, roidb = combined_roidb('fl_train+fl_val_logo+fl27_train+bl_train+toplogo_train+logos32plus')
    #imdb, roidb = combined_roidb('synmetu_ta_train_all')
    #imdb, roidb = combined_roidb('srf_ice_good+srf_ice_good_occlusion')
    #imdb, roidb = combined_roidb('flbl_detection_train_all')
    #imdb, roidb = combined_roidb('fl_train+fl_val_logo')


    output_dir = os.path.expanduser('~/github/logoretrieval/py_faster_rcnn/output/final/publicNonFlickr_ownlogo_detection_vgg16')
    #output_dir =  os.path.expanduser('~/github/logoretrieval/py_faster_rcnn/output/final/allnet_detector_resnet50_bn_scale_merged')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print 'Output will be saved to `{:s}`'.format(output_dir)

    print cfg.USE_GPU_NMS
    print args.pretrained_model
    train_net(args.solver, roidb=roidb, output_dir=output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters, roidb_det=roidb_det)
