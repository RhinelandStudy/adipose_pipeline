#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:18:10 2017

@author: shahidm
"""

from os.path import realpath, join, abspath, dirname


# defaults
SCRIPT_PATH = dirname(realpath(__file__))


MODELS_DIR = abspath(join(SCRIPT_PATH, 'models'))

multiviewModel = join(MODELS_DIR, 'Segmentation')
singleViewModels = join(MODELS_DIR, 'Segmentation')
localizationModels = join(MODELS_DIR, 'Localization')
control_images = True
imgSize = [72, 224, 256]
run_localization = True
compartments = 20
increase_threshold = 0.4
sat_to_vat_threshold = 2.0

