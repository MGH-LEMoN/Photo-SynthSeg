"""This script enables to launch predictions with SynthSeg from the terminal."""

# print information
print('\n')
print('SynthSeg prediction')
print('\n')

# python imports
import os
import sys
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# add main folder to python path and import ./SynthSeg/predict.py
synthseg_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
sys.path.append(synthseg_home)
from SynthSeg.predict import predict


# parse arguments
parser = ArgumentParser()
parser.add_argument("path_images", type=str, help="images to segment. Can be the path to a single image or to a folder")
parser.add_argument("path_segmentations", type=str, help="path where to save the segmentations. Must be the same type "
                                                         "as path_images (path to a single image or to a folder)")
parser.add_argument("--post", type=str, default=None, dest="path_posteriors",
                    help="path where to save the posteriors. "
                         "Must be the same type as path_images (path to a single image or to a folder)")
parser.add_argument("--vol", type=str, default=None, dest="path_volumes",
                    help="path to a csv file where to save the volumes of all ROIs for all patients")
parser.add_argument("--resampled", type=str, default=None, dest="path_resampled",
                    help="path where to save the resampled images (at 1mm isotropic resolution). "
                         "Must be the same type as path_images (path to a single image or to a folder)")
parser.add_argument("--cpu", action="store_true", help="enforce running with CPU rather than GPU.")
parser.add_argument("--crop", type=int, default=None, dest="cropping",
                    help="Size of 3D image patches that will be extracted from the input for analysis.")
parser.add_argument("--threads", type=int, default=1, dest="threads",
                    help="number of threads to be used by tensorflow when running on CPU.")
args = vars(parser.parse_args())

# enforce CPU processing if necessary
if args['cpu']:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
del args['cpu']

# limit the number of threads to be used if running on CPU
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(args['threads'])
del args['threads']

# default parameters
path_label_list = os.path.join(synthseg_home, 'data/labels_classes_priors/segmentation_labels.npy')
path_names_list = os.path.join(synthseg_home, 'data/labels_classes_priors/segmentation_names.npy')
path_topology_classes = os.path.join(synthseg_home, 'data/labels_classes_priors/topological_classes.npy')
path_model = os.path.join(synthseg_home, 'models/SynthSeg.h5')
args['segmentation_label_list'] = path_label_list
args['segmentation_names_list'] = path_names_list
args['topology_classes'] = path_topology_classes
args['path_model'] = path_model

# call predict
predict(**args)
