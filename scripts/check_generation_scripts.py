import numpy as np
from check_generation import check_generation

# general parameters
examples = 10
spacing = 4
# result_folder = '/home/benjamin/data/SynthSeg/generated_examples'
result_folder = f"/space/calico/1/users/Harsha/SynthSeg/generated_examples_stack_D{spacing:02d}"

# new outputs
# tensor_names = [
#     'labels_in',
#     'deform',
#     'gmm',
#     'bias',
#     'image_out',
#     'labels_out',
# ]
tensor_names = [
    "labels_input",
    "random_spatial_deformation_1",
    "sample_conditional_gmm_1",
    "bias_field_corruption_1",
    # # "sample_resolution_1",
    "intensity_augmentation_1",
    "gaussian_blur_1",
    "mimic_acquisition_1",
    "image_out",
    "labels_out",
]
filenames = [
    "labels_in_save",
    "deform_save",
    "gmm_save",
    "bias_save",
    # "sample_resolution_save",
    "intensity_augmentation_save",
    "blur_save",
    "mimic_acquisition_save",
    "image_out_save",
    "labels_out_save",
]

# inputs
# labels_folder = '/home/benjamin/data/Buckner40/labels/training/merged_choroid_extra_cerebral/subject15_seg_extra_cerebral.nii.gz'
labels_folder = "/space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_label_maps_manual_auto_photos_noCerebellumOrBrainstem/HCP_228434_seg_cerebral.nii.gz"
# generation_labels = '/home/benjamin/data/SynthSeg/labels_classes_stats/generation_charm_choroid_lesions_csf.npy'
generation_labels = "/space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/generation_charm_choroid_lesions.npy"
# n_neutral_labels = 19
n_neutral_labels = 5
# segmentation_labels = '/home/benjamin/data/SynthSeg/labels_classes_stats/segmentation_charm_choroid_lesions_csf.npy'
segmentation_labels = "/space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/segmentation_new_charm_choroid_lesions.npy"
subjects_prob = None  # np.array([1]*79 + [10000])
patch_dir = None  # '/home/benjamin/data/noisy_patches'

# general parameters
batchsize = 1
channels = 1
target_resolution = None
output_shape = None
output_divisible_by_n = None

# GMM sampling
# generation_classes = '/home/benjamin/data/SynthSeg/labels_classes_stats/generation_classes_charm_choroid_lesions_gm_csf.npy'
generation_classes = "/space/calico/1/users/Harsha/SynthSeg/data/SynthSeg_param_files_manual_auto_photos_noCerebellumOrBrainstem/generation_classes_charm_choroid_lesions_gm.npy"
# prior_distribution = 'normal'
prior_distribution = "uniform"
# prior_means = '/home/benjamin/Pictures/images/thesis/generation_chapter3/prior_means.npy'
# prior_stds = '/home/benjamin/Pictures/images/thesis/generation_chapter3/prior_stds.npy'
prior_means = None
prior_stds = None
specific_stats_for_channel = False
mix_prior_and_random = False

# spatial transformation
flip = False
scaling_bounds = np.array(
    [[1.01, 1, 1], [1.011, 1, 1]]
)  # scaling in axial plane
rotation_bounds = np.array(
    [[-1, -1, -8], [-0.99, -0.99, -7.9]]
)  # rotation for visualisation in axial plane
shearing_bounds = 0.00001  # [0.0012, 0.0013]
translation_bounds = False
# nonlin_std = 2.5
nonlin_std = (4, 0, 4)
nonlin_shape_factor = (0.0625, 1 / spacing, 0.0625)

# blurring/ downsample
randomise_res = False
max_res_iso = 4.0
max_res_aniso = 8.0
data_res = (1, spacing, 1)  # np.array([1., 1., 5.])
thickness = None  # '/home/benjamin/data/SynthSeg/PAMI/labels_classes_stats/data_res/thickness_411.npy'
thickness = (
    1,
    0.01,
    1,
)  # '/home/benjamin/data/SynthSeg/PAMI/labels_classes_stats/data_res/thickness_411.npy'
downsample = True
# blur_range = 1.01
blur_range = 1.03

# bias field
bias_field_std = 0.5

bias_shape_factor = (0.025, 1 / spacing, 0.025)
return_gradients = False

check_generation(
    labels_folder,
    examples,
    tensor_names,
    filenames,
    result_folder,
    generation_labels=generation_labels,
    n_neutral_labels=n_neutral_labels,
    segmentation_labels=segmentation_labels,
    #  subjects_prob=subjects_prob,
    #  patch_dir=patch_dir,
    batchsize=batchsize,
    n_channels=channels,
    target_res=target_resolution,
    output_shape=output_shape,
    output_div_by_n=output_divisible_by_n,
    generation_classes=generation_classes,
    prior_distributions=prior_distribution,
    prior_means=prior_means,
    prior_stds=prior_stds,
    use_specific_stats_for_channel=specific_stats_for_channel,
    mix_prior_and_random=mix_prior_and_random,
    flipping=flip,
    scaling_bounds=scaling_bounds,
    rotation_bounds=rotation_bounds,
    shearing_bounds=shearing_bounds,
    translation_bounds=translation_bounds,
    nonlin_std=nonlin_std,
    nonlin_shape_factor=nonlin_shape_factor,
    randomise_res=randomise_res,
    max_res_iso=max_res_iso,
    max_res_aniso=max_res_aniso,
    data_res=data_res,
    thickness=thickness,
    downsample=downsample,
    blur_range=blur_range,
    bias_field_std=bias_field_std,
    bias_shape_factor=bias_shape_factor,
    return_gradients=return_gradients,
)
