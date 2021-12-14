import json
import os

import numpy as np
from ext.lab2im import utils
from SynthSeg.evaluate import fast_dice

from dice_utils import return_common_subjects


def calculate_dice(config, folder1, folder2, file_name, slice=False, merge=0):
    folder1_list, folder2_list = utils.list_images_in_folder(
        folder1), utils.list_images_in_folder(folder2)

    folder1_list, folder2_list = return_common_subjects(
        folder1_list, folder2_list)

    final_dice_scores = dict()
    for file1, file2 in zip(folder1_list, folder2_list):
        subject_id = os.path.split(file1)[-1][:7]

        x = utils.load_volume(file1)
        y = utils.load_volume(file2)

        assert x.shape == y.shape, "Shape Mismatch"

        if slice:
            slice_idx = np.argmax((x > 1).sum(0).sum(0))

            x = x[:, :, slice_idx].astype("int")
            y = y[:, :, slice_idx].astype("int")

        required_labels = config.required_labels
        if merge:
            x, y, required_labels = merge_labels_in_image(config, x, y)

        dice_coeff = fast_dice(x, y, required_labels)
        required_labels = required_labels.astype("int").tolist()
        final_dice_scores[subject_id] = dict(zip(required_labels, dice_coeff))

    merge_tag = "merge" if merge else "no-merge"

    with open(
            os.path.join(config.SYNTHSEG_RESULTS,
                         f"{file_name}_{merge_tag}.json"),
            "w",
            encoding="utf-8",
    ) as fp:
        json.dump(final_dice_scores, fp, sort_keys=True, indent=4)


def merge_labels_in_image(config, x, y):
    merge_required_labels = []
    for (id1, id2) in config.LABEL_PAIRS:
        x[x == id2] = id1
        y[y == id2] = id1

        merge_required_labels.append(id1)

    merge_required_labels = np.array(merge_required_labels)

    return x, y, merge_required_labels
