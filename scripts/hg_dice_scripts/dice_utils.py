import glob
import os


def files_at_path(path_str):
    return sorted(glob.glob(os.path.join(path_str, "*")))


def run_make_target(config, flag):
    os.system(f"make -C {config.SYNTHSEG_PRJCT} predict-{flag}")


def id_check(*args):
    fn_list = set([os.path.split(item)[-1][:7] for item in args])
    return len(fn_list) == 1


def return_common_subjects(*args):
    args = [{
        os.path.split(input_file)[-1][:7]: input_file
        for input_file in file_list
    } for file_list in args]

    lst = [set(lst.keys()) for lst in args]

    # One-Liner to intersect a list of sets
    common_names = sorted(lst[0].intersection(*lst))

    args = [[lst[key] for key in common_names] for lst in args]

    return args
