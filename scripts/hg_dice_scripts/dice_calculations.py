import json
import os

import numpy as np
from dice_utils import return_common_subjects
from ext.lab2im import utils
from SynthSeg.evaluate import fast_dice


def print_dice_to_file(config, dice_scores, **kwargs):
    """Print dice scores to file

    Args:
        config (object): Project configuration object.
        dice_scores ([type]): [description]
    """
    merge_tag = "merge" if kwargs["merge"] else "no-merge"

    with open(
            os.path.join(config.dice_dir,
                         f"{kwargs['output_name']}_{merge_tag}.json"),
            "w",
            encoding="utf-8",
    ) as fp:
        json.dump(dice_scores, fp, sort_keys=True, indent=4)


#FIXME: Maybe a dataframe is preferable instead of a dict object
# (use tabulate on the dataframe to nicely print the values)
def calculate_and_print_dice(config, **kwargs):
    """[summary]

    Args:
        config ([type]): [description]
    """
    source_list = utils.list_images_in_folder(getattr(config,
                                                      kwargs["source"]))
    target_list = utils.list_images_in_folder(getattr(config,
                                                      kwargs["target"]))

    source_list, target_list = return_common_subjects(source_list, target_list)

    final_dice_scores = dict()
    for file1, file2 in zip(source_list, target_list):
        subject_id = os.path.split(file1)[-1][:7]

        x = utils.load_volume(file1)
        y = utils.load_volume(file2)

        # assert x.shape[:-1] == y.shape[:-1], "Shape Mismatch"
        if x.shape[:-1] != y.shape[:-1]:
            print(f'{subject_id}, {x.shape[:-1]}, {y.shape[:-1]}')
            continue

        if slice:
            x_slice_idx = np.argmax((x > 1).sum(0).sum(0))

            if y.shape[-1] < x.shape[-1]:
                x_range = np.arange(x_slice_idx % 2, x.shape[-1], 2)
                y_slice_idx = np.where(x_range == x_slice_idx)
                y_slice_idx = y_slice_idx[0][0]
            else:
                y_slice_idx = x_slice_idx

            x = x[:, :, x_slice_idx].astype("int")
            y = y[:, :, y_slice_idx].astype("int")

        required_labels = config.required_labels
        if kwargs["merge"]:
            x, y, required_labels = merge_labels_in_image(config, x, y)

        dice_coeff = fast_dice(x, y, required_labels)
        final_dice_scores[subject_id] = dict(zip(required_labels, dice_coeff))

        #FIXME: Should I pass just the relevant keys or is **kwargs fine?
        print_dice_to_file(config, final_dice_scores, **kwargs)


def merge_labels_in_image(config, x, y):
    for (id1, id2) in config.LABEL_PAIRS:
        x[x == id2] = id1
        y[y == id2] = id1

    merge_required_labels = [int(item[0]) for item in config.LABEL_PAIRS]

    return x, y, merge_required_labels


def verify_dice_dict(config, item):
    """Check if dictionary is legitimate. i.e. If the values for keys
    "source", "target", "output_name", "slice" and "merge" are valud

    Args:
        config (object): Project Configuration object
        item (dict): dictionary object to generate dice scores

    Returns:
        [type]: [description]
    """
    keys = ["source", "target", "output_name", "merge", "slice"]
    if not all(key in item.keys() for key in keys):
        print("Missing required keys")

    if not getattr(config, item["source"], None):
        print(f'Source folder {item["source"]} does not exist')
        return
    if not getattr(config, item["target"], None):
        print(f'Target folder {item["target"]} does not exist')
        return
    if (not isinstance(item["output_name"], str)) or (not item["output_name"]):
        print("Output File Name Cannot be Empty")
        return
    if not isinstance(item["slice"], (bool, int)):
        return
    if not isinstance(item["merge"], (bool, int)):
        return
    if not len(os.listdir(getattr(config, item["source"], None))):
        print(f"Source Directory {getattr(config, item['source'], None)} is empty")
        return
    if not len(os.listdir(getattr(config, item["target"], None))):
        print(f"Target Directory {getattr(config, item['target'], None)} is empty")
        return
    return 1


def calculate_dice_for_dict(config, item_list):
    """Calculate dice between source and target

    Args:
        config (object): Project Configuration object
        item_list (list): list of dictionary objects

        Contents of the dictionary:
        source (string)
        target (string)
        output_name (string)
        slice (bool/int): 0 - 3D dice; 1 - 2D slice
        merge (bool/int): Merge (1) or not (0) segments as specified in
                    config.MERGE_LABEL_PAIRS
    """
    for item in item_list:
        print(item['message'])
        if not verify_dice_dict(config, item):
            print('Skipping...')
            continue
        calculate_and_print_dice(config, **item)
