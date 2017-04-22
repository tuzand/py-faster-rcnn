# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.fl import fl
from datasets.srf_ice import srf_ice
from datasets.srf_ski import srf_ski
from datasets.logo import logo
from datasets.logo_detection import logo_detection
from datasets.metu import metu
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

srf_ice_good_path = '/home/andras/data/datasets/srf_ice/good'
for split in ['srf_ice_good']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: srf_ice(split, srf_ice_good_path))

srf_ice_good_occlusion_path = '/home/andras/data/datasets/srf_ice/good_occlusion'
for split in ['srf_ice_good_occlusion']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: srf_ice(split, srf_ice_good_occlusion_path))

srf_ice_good_logo_path = '/home/andras/data/datasets/srf_ice_logo/good'
for split in ['srf_ice_good_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ice_good_logo_path))

srf_ice_good_occlusion_logo_path = '/home/andras/data/datasets/srf_ice_logo/good_occlusion'
for split in ['srf_ice_good_occlusion_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ice_good_occlusion_logo_path))

srf_ski_path = '/home/andras/data/datasets/srf_ski/good'
for split in ['srf_ski_good']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: srf_ski(split, srf_ski_path))

srf_ski_logo_path = '/home/andras/data/datasets/srf_ski_logo/good'
for split in ['srf_ski_logo_good']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ski_logo_path))

fl_path = '/home/andras/data/datasets/FL32/FlickrLogos-v2/fl/fl'
for split in ['fl_train', 'fl_test', 'fl_test_logo', 'fl_trainval', 'fl_val_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo(split, fl_path))

flbl_path = '/home/andras/data/datasets/FLBL/FLBL'
for split in ['flbl_train_all', 'flbl_train', 'flbl_test']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo(split, flbl_path))


for split in ['fl_test_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: fl(split, fl_path))

fl27_path = '/home/andras/data/datasets/FL27/FL27'
for split in ['fl27_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo(split, fl27_path))

toplogo_path = '/home/andras/data/datasets/toplogo/toplogo'
for split in ['toplogo_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo(split, toplogo_path))

bl_path = '/home/andras/data/datasets/BL/BL'
for split in ['bl_train_all', 'bl_train', 'bl_test']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo(split, bl_path))

logos32plus_path = '/home/andras/data/datasets/L32P/classification'
for split in ['logos32plus']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo(split, logos32plus_path))

fl_detection_path = '/home/andras/data/datasets/FL32/FlickrLogos-v2/fl/fl_detection'
for split in ['fl_detection_train', 'fl_detection_test', 'fl_detection_test_logo', 'fl_detection_trainval', 'fl_detection_val_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, fl_detection_path))

fl27_detection_path = '/home/andras/data/datasets/FL27/FL27_detection'
for split in ['fl27_detection_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, fl27_detection_path))

toplogo_detection_path = '/home/andras/data/datasets/toplogo/toplogo_detection'
for split in ['toplogo_detection_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, toplogo_detection_path))

bl_detection_path = '/home/andras/data/datasets/BL/BL_detection'
for split in ['bl_detection_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, bl_detection_path))

logos32plus_detection_path = '/home/andras/data/datasets/L32P/detection'
for split in ['logos32plus_detection']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, logos32plus_detection_path))

metu_path = '/home/andras/data/datasets/METU/metu'
for split in ['metu_train', 'metu_sample']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: metu(split, metu_path))

synmetumir_path = '/home/andras/data/datasets/SYNMETUMIR'
for split in ['synmetumir_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, synmetumir_path))

synmetuta_path = '/home/andras/data/datasets/SYNMETUTA'
for split in ['synmetu_ta_train_all']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, synmetuta_path))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
