import glob
import os

from dice_config import *


def files_at_path(path_str):
    return sorted(glob.glob(os.path.join(path_str, '*')))


def id_check(config, *args):
    fn_list = set([os.path.split(item)[-1][:7] for item in args])

    assert len(fn_list) == 1, 'File MisMatch'

    if config.IGNORE_SUBJECTS:
        if not fn_list.intersection(set(config.IGNORE_SUBJECTS)):
            return 0
        else:
            raise Exception('Something is off')
    else:
        print(list(fn_list)[0])

    return 1


def return_common_subjects(*args):
    if len(set([len(item) for item in args])) > 1:

        args = [{
            os.path.split(input_file)[-1][:7]: input_file
            for input_file in file_list
        } for file_list in args]

        lst = [set(lst.keys()) for lst in args]

        # One-Liner to intersect a list of sets
        common_names = sorted(lst[0].intersection(*lst))

        args = [[lst[key] for key in common_names] for lst in args]

    return args
