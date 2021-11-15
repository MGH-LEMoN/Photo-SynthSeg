import os
"""
This file contains all the static parameters needed for the project
"""

PROJ_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJ_DIR, 'data')
RESULTS_DIR = os.path.join(PROJ_DIR, 'results')
MODELS_DIR = os.path.join(PROJ_DIR, 'models')

LABEL_MAPS_DIR = os.path.join(
    DATA_DIR, 'SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem')
PARAM_FILES_DIR = os.path.join(
    DATA_DIR,
    'SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem')

GENERATION_LABELS = os.path.join(PARAM_FILES_DIR,
                                 'generation_charm_choroid_lesions.npy')
GENERATION_CLASSES = os.path.join(
    PARAM_FILES_DIR, 'generation_classes_charm_choroid_lesions_gm.npy')
SEGMENTATION_LABELS = os.path.join(
    PARAM_FILES_DIR, 'segmentation_new_charm_choroid_lesions.npy')
