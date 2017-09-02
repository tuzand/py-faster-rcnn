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
from datasets.schalke import schalke
from datasets.alllogo import alllogo
from datasets.publiclogo import publiclogo
import numpy as np
import os



ownlogos_det_path = os.path.expanduser('~/data/datasets/logodata_det')
for split in ['ownlogos_det']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, ownlogos_det_path))




# Set up voc_<year>_<split> using selective search "fast" mode
srf_ice_good_path = os.path.expanduser('~/data/datasets/srf_ice/good')
for split in ['srf_ice_good']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: alllogo(split, srf_ice_good_path))

srf_ice_good_occlusion_path = os.path.expanduser('~/data/datasets/srf_ice/good_occlusion')
for split in ['srf_ice_good_occlusion']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: alllogo(split, srf_ice_good_occlusion_path))

srf_ice_bad_path = os.path.expanduser('~/data/datasets/srf_ice/bad')
for split in ['srf_ice_bad']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: alllogo(split, srf_ice_bad_path))

srf_ice_bad_occlusion_path = os.path.expanduser('~/data/datasets/srf_ice/bad_occlusion')
for split in ['srf_ice_bad_occlusion']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: alllogo(split, srf_ice_bad_occlusion_path))

srf_ski_path = os.path.expanduser('~/data/datasets/srf_ski/good')
for split in ['srf_ski']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: alllogo(split, srf_ski_path))

srf_football_path = os.path.expanduser('~/data/datasets/srf_football/')
for split in ['srf_football']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: alllogo(split, srf_football_path))

fl_path = os.path.expanduser('~/data/datasets/FL32/FlickrLogos-v2/fl/fl')
for split in ['fl_train', 'fl_test', 'fl_test_logo', 'fl_trainval', 'fl_val_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: fl(split, fl_path))

fl27_path = os.path.expanduser('~/data/datasets/FL27/FL27')
for split in ['fl27_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: publiclogo(split, fl27_path))

toplogo_path = os.path.expanduser('~/data/datasets/toplogo/toplogo')
for split in ['toplogo_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: publiclogo(split, toplogo_path))

bl_path = os.path.expanduser('~/data/datasets/BL/BL')
for split in ['bl_all', 'bl_train', 'bl_test']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: publiclogo(split, bl_path))

logos32plus_path = os.path.expanduser('~/data/datasets/L32P/classification')
for split in ['logos32plus']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: alllogo(split, logos32plus_path))

fl_detection_path = os.path.expanduser('~/data/datasets/FL32/FlickrLogos-v2/fl/fl_detection')
for split in ['fl_detection_train', 'fl_detection_test', 'fl_detection_test_logo', 'fl_detection_trainval', 'fl_detection_val_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, fl_detection_path))

fl27_detection_path = os.path.expanduser('~/data/datasets/FL27/FL27_detection')
for split in ['fl27_detection_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, fl27_detection_path))

toplogo_detection_path = os.path.expanduser('~/data/datasets/toplogo/toplogo_detection')
for split in ['toplogo_detection_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, toplogo_detection_path))

bl_detection_path = os.path.expanduser('~/data/datasets/BL/BL_detection')
for split in ['bl_detection_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, bl_detection_path))

logos32plus_detection_path = os.path.expanduser('~/data/datasets/L32P/detection')
for split in ['logos32plus_detection']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, logos32plus_detection_path))

srf_ice_good_logo_path = os.path.expanduser('~/data/datasets/srf_ice_logo/good')
for split in ['srf_ice_good_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ice_good_logo_path))

srf_ice_good_occlusion_logo_path = os.path.expanduser('~/data/datasets/srf_ice_logo/good_occlusion')
for split in ['srf_ice_good_occlusion_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ice_good_occlusion_logo_path))

srf_ice_bad_logo_path = os.path.expanduser('~/data/datasets/srf_ice_logo/bad')
for split in ['srf_ice_bad_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ice_bad_logo_path))

srf_ice_bad_occlusion_logo_path = os.path.expanduser('~/data/datasets/srf_ice_logo/bad_occlusion')
for split in ['srf_ice_bad_occlusion_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ice_bad_occlusion_logo_path))

srf_ski_logo_path = os.path.expanduser('~/data/datasets/srf_ski_logo/good')
for split in ['srf_ski_good_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_ski_logo_path))

srf_football_logo_path = os.path.expanduser('~/data/datasets/srf_football_logo/')
for split in ['srf_football_logo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, srf_football_logo_path))

flbl_detection_path = os.path.expanduser('~/data/datasets/FLBL/FLBL_detection')
for split in ['flbl_detection_train_all']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, flbl_detection_path))

metu_path = os.path.expanduser('~/data/datasets/METU/metu')
for split in ['metu_train', 'metu_sample']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: metu(split, metu_path))

synmetumir_path = os.path.expanduser('~/data/datasets/SYNMETUMIR')
for split in ['synmetumir_train']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, synmetumir_path))

synlogo_path = os.path.expanduser('~/data/datasets/SYNLOGO')
for split in ['synlogo']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, synlogo_path))

synmetuta_path = os.path.expanduser('~/data/datasets/SYNMETUTA')
for split in ['synmetu_ta_train_all']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, synmetuta_path))

schalke_path = os.path.expanduser('~/data/datasets/schalke')
for split in ['schalke']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: schalke(split, schalke_path))

schalke_det_path = os.path.expanduser('~/data/datasets/schalke_det')
for split in ['schalke_det']:
    name = '{}'.format(split)
    __sets[name] = (lambda split=split: logo_detection(split, schalke_det_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

