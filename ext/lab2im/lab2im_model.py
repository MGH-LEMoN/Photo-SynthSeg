# python imports
import keras.layers as KL
import numpy as np
from keras.models import Model

# project imports
from . import layers, utils
from .edit_tensors import blurring_sigma_for_downsampling, resample_tensor


def lab2im_model(labels_shape,
                 n_channels,
                 generation_labels,
                 output_labels,
                 atlas_res,
                 target_res,
                 output_shape=None,
                 output_div_by_n=None,
                 blur_range=1.15):
    """
    This function builds a keras/tensorflow model to generate images from provided label maps.
    The images are generated by sampling a Gaussian Mixture Model (of given parameters), conditionned on the label map.
    The model will take as inputs:
        -a label map
        -a vector containing the means of the Gaussian Mixture Model for each label,
        -a vector containing the standard deviations of the Gaussian Mixture Model for each label,
        -an array of size batch*(n_dims+1)*(n_dims+1) representing a linear transformation
    The model returns:
        -the generated image normalised between 0 and 1.
        -the corresponding label map, with only the labels present in output_labels (the other are reset to zero).
    :param labels_shape: shape of the input label maps. Can be a sequence or a 1d numpy array.
    :param n_channels: number of channels to be synthetised.
    :param generation_labels: list of all possible label values in the input label maps.
    Can be a sequence or a 1d numpy array.
    :param output_labels: list of the same length as generation_labels to indicate which values to use in the label maps
    returned by this model, i.e. all occurences of generation_labels[i] in the input label maps will be converted to
    output_labels[i] in the returned label maps. Examples:
    Set output_labels[i] to zero if you wish to erase the value generation_labels[i] from the returned label maps.
    Set output_labels[i]=generation_labels[i] if you wish to keep the value generation_labels[i] in the returned maps.
    Can be a list or a 1d numpy array. By default output_labels is equal to generation_labels.
    :param atlas_res: resolution of the input label maps.
    Can be a number (isotropic resolution), a sequence, or a 1d numpy array.
    :param target_res: target resolution of the generated images and corresponding label maps.
    Can be a number (isotropic resolution), a sequence, or a 1d numpy array.
    :param output_shape: (optional) desired shape of the output images.
    If the atlas and target resolutions are the same, the output will be cropped to output_shape, and if the two
    resolutions are different, the output will be resized with trilinear interpolation to output_shape.
    Can be an integer (same size in all dimensions), a sequence, or a 1d numpy array.
    :param output_div_by_n: (optional) forces the output shape to be divisible by this value. It overwrites output_shape
    if necessary. Can be an integer (same size in all dimensions), a sequence, or a 1d numpy array.
    :param blur_range: (optional) Randomise the standard deviation of the blurring kernels, (whether data_res is given
    or not). At each mini_batch, the standard deviation of the blurring kernels are multiplied by a coefficient sampled
    from a uniform distribution with bounds [1/blur_range, blur_range]. If None, no randomisation. Default is 1.15.
    """

    # reformat resolutions
    labels_shape = utils.reformat_to_list(labels_shape)
    n_dims, _ = utils.get_dims(labels_shape)
    atlas_res = utils.reformat_to_n_channels_array(atlas_res, n_dims=n_dims)[0]
    target_res = atlas_res if (
        target_res is None) else utils.reformat_to_n_channels_array(
            target_res, n_dims)[0]

    # get shapes
    crop_shape, output_shape = get_shapes(labels_shape, output_shape,
                                          atlas_res, target_res,
                                          output_div_by_n)

    # define model inputs
    labels_input = KL.Input(shape=labels_shape + [1],
                            name='labels_input',
                            dtype='int32')
    means_input = KL.Input(shape=list(generation_labels.shape) + [n_channels],
                           name='means_input')
    stds_input = KL.Input(shape=list(generation_labels.shape) + [n_channels],
                          name='stds_input')

    # deform labels
    labels = layers.RandomSpatialDeformation(
        inter_method='nearest')(labels_input)

    # cropping
    if crop_shape != labels_shape:
        labels._keras_shape = tuple(labels.get_shape().as_list())
        labels = layers.RandomCrop(crop_shape)(labels)

    # build synthetic image
    labels._keras_shape = tuple(labels.get_shape().as_list())
    image = layers.SampleConditionalGMM(generation_labels)(
        [labels, means_input, stds_input])

    # apply bias field
    image._keras_shape = tuple(image.get_shape().as_list())
    image = layers.BiasFieldCorruption(.3,
                                       .025,
                                       same_bias_for_all_channels=False)(image)

    # intensity augmentation
    image._keras_shape = tuple(image.get_shape().as_list())
    image = layers.IntensityAugmentation(clip=300,
                                         normalise=True,
                                         gamma_std=.2)(image)

    # blur image
    sigma = blurring_sigma_for_downsampling(atlas_res, target_res)
    image._keras_shape = tuple(image.get_shape().as_list())
    image = layers.GaussianBlur(sigma=sigma,
                                random_blur_range=blur_range)(image)

    # resample to target res
    if crop_shape != output_shape:
        image = resample_tensor(image, output_shape, interp_method='linear')
        labels = resample_tensor(labels, output_shape, interp_method='nearest')

    # reset unwanted labels to zero
    labels = layers.ConvertLabels(generation_labels, dest_values=output_labels, name='labels_out')(labels)

    # build model (dummy layer enables to keep the labels when plugging this model to other models)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, labels])
    brain_model = Model(inputs=[labels_input, means_input, stds_input],
                        outputs=[image, labels])

    return brain_model


def get_shapes(labels_shape, output_shape, atlas_res, target_res,
               output_div_by_n):

    n_dims = len(atlas_res)

    # get resampling factor
    if atlas_res.tolist() != target_res.tolist():
        resample_factor = [
            atlas_res[i] / float(target_res[i]) for i in range(n_dims)
        ]
    else:
        resample_factor = None

    # output shape specified, need to get cropping shape, and resample shape if necessary
    if output_shape is not None:
        output_shape = utils.reformat_to_list(output_shape,
                                              length=n_dims,
                                              dtype='int')

        # make sure that output shape is smaller or equal to label shape
        if resample_factor is not None:
            output_shape = [
                min(int(labels_shape[i] * resample_factor[i]), output_shape[i])
                for i in range(n_dims)
            ]
        else:
            output_shape = [
                min(labels_shape[i], output_shape[i]) for i in range(n_dims)
            ]

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n)
                         for s in output_shape]
            if output_shape != tmp_shape:
                print('output shape {0} not divisible by {1}, changed to {2}'.
                      format(output_shape, output_div_by_n, tmp_shape))
                output_shape = tmp_shape

        # get cropping and resample shape
        if resample_factor is not None:
            cropping_shape = [
                int(np.around(output_shape[i] / resample_factor[i], 0))
                for i in range(n_dims)
            ]
        else:
            cropping_shape = output_shape

    # no output shape specified, so no cropping unless label_shape is not divisible by output_div_by_n
    else:
        cropping_shape = labels_shape
        if resample_factor is not None:
            output_shape = [
                int(np.around(cropping_shape[i] * resample_factor[i], 0))
                for i in range(n_dims)
            ]
        else:
            output_shape = cropping_shape
        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, answer_type='closer')
                            for s in output_shape]

    return cropping_shape, output_shape
