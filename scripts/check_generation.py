# python imports
import os
import copy
import time
import numpy as np
from keras.models import Model

# project imports
from SynthSeg.brain_generator import BrainGenerator

# third-party imports
from ext.lab2im import utils


def check_generation(
    labels_dir,
    n_examples,
    names_output_tensors,
    filenames_output_tensors,
    result_dir,
    generation_labels=None,
    n_neutral_labels=None,
    segmentation_labels=None,
    #  subjects_prob=None,
    batchsize=1,
    n_channels=1,
    target_res=None,
    output_shape=None,
    output_div_by_n=None,
    generation_classes=None,
    prior_distributions="uniform",
    prior_means=None,
    prior_stds=None,
    use_specific_stats_for_channel=False,
    mix_prior_and_random=False,
    flipping=True,
    scaling_bounds=0.2,
    rotation_bounds=15,
    shearing_bounds=0.012,
    translation_bounds=False,
    nonlin_std=4.0,
    nonlin_shape_factor=0.04,
    randomise_res=False,
    max_res_iso=4.0,
    max_res_aniso=8.0,
    data_res=None,
    thickness=None,
    downsample=False,
    blur_range=1.03,
    bias_field_std=0.5,
    bias_shape_factor=0.025,
    return_gradients=False,
):

    # get labels
    generation_labels, _ = utils.get_list_labels(
        label_list=generation_labels, labels_dir=labels_dir
    )
    if segmentation_labels is not None:
        segmentation_labels, _ = utils.get_list_labels(
            label_list=segmentation_labels
        )
    else:
        segmentation_labels = generation_labels

    # prepare model folder
    utils.mkdir(result_dir)

    # instantiate BrainGenerator object
    brain_generator = BrainGenerator(
        labels_dir=labels_dir,
        generation_labels=generation_labels,
        n_neutral_labels=n_neutral_labels,
        output_labels=segmentation_labels,
        #  subjects_prob=subjects_prob,
        batchsize=batchsize,
        n_channels=n_channels,
        target_res=target_res,
        output_shape=output_shape,
        output_div_by_n=output_div_by_n,
        generation_classes=generation_classes,
        prior_distributions=prior_distributions,
        prior_means=prior_means,
        prior_stds=prior_stds,
        use_specific_stats_for_channel=use_specific_stats_for_channel,
        mix_prior_and_random=mix_prior_and_random,
        flipping=flipping,
        scaling_bounds=scaling_bounds,
        rotation_bounds=rotation_bounds,
        shearing_bounds=shearing_bounds,
        translation_bounds=translation_bounds,
        nonlin_shape_factor=nonlin_shape_factor,
        nonlin_std=nonlin_std,
        randomise_res=randomise_res,
        max_res_iso=max_res_iso,
        max_res_aniso=max_res_aniso,
        data_res=data_res,
        thickness=thickness,
        downsample=downsample,
        blur_range=blur_range,
        bias_field_std=bias_field_std,
        bias_shape_factor=bias_shape_factor,
        #  return_gradients=return_gradients
    )

    # get generative model
    labels_to_image_model = brain_generator.labels_to_image_model
    train_example_gen = brain_generator.model_inputs_generator

    # redefine model to include all the layers to check
    list_output_tensors = []
    for idx, name in enumerate(names_output_tensors):
        if "mimic" in name:
            if len(labels_to_image_model.get_layer(name).output) == 2:
                list_output_tensors.extend(
                    labels_to_image_model.get_layer(name).output
                )

                filenames_output_tensors.insert(
                    idx + 1, "mimic_acquisition_stack_downres_save"
                )
                continue
        list_output_tensors.append(labels_to_image_model.get_layer(name).output)
    model_to_check = Model(
        inputs=labels_to_image_model.inputs, outputs=list_output_tensors
    )

    # predict
    n = len(str(n_examples))
    for i in range(1, n_examples + 1):

        start = time.time()
        outputs = model_to_check.predict(next(train_example_gen))
        print(
            "\nprediction {0:d} took {1:.01f}s".format(i, time.time() - start)
        )

        for output, name in zip(outputs, filenames_output_tensors):
            for b in range(batchsize):
                tmp_name = copy.deepcopy(name)
                tmp_output = np.squeeze(output[b, ...])
                if "_argmax" in tmp_name:
                    tmp_output = tmp_output.argmax(-1)
                    tmp_name = tmp_name.replace("_argmax", "")
                if "_convert" in tmp_name:
                    tmp_output = generation_labels[tmp_output]
                    tmp_name = tmp_name.replace("_convert", "")
                if "_save" in tmp_name:
                    result_dir_example = os.path.join(result_dir, f"{i:02d}")
                    os.makedirs(result_dir_example, exist_ok=True)
                    path = os.path.join(
                        result_dir_example,
                        tmp_name.replace("_save", "")
                        + "_%.{}d".format(n) % i
                        + ".nii.gz",
                    )
                    if batchsize > 1:
                        path = path.replace(".nii.gz", "_%s.nii.gz" % (b + 1))
                    if "_downres" in tmp_name:
                        path = path.replace("_downres", "")
                        utils.save_volume(
                            tmp_output,
                            np.eye(4),
                            brain_generator.header,
                            path,
                            res=data_res,
                        )
                    else:
                        # utils.save_volume(output, np.eye(4), brain_generator.header, path)
                        utils.save_volume(tmp_output, np.eye(4), None, path)
                else:
                    print("{0} : {1}".format(tmp_name, tmp_output))
