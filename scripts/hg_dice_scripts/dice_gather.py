import glob
import os
import re
from shutil import copyfile


def list_files(source, expr):
    if len(expr) == 1:
        file_list = glob.glob(os.path.join(source, *expr))
        if not file_list:
            file_list = glob.glob(os.path.join(source, "*", expr[-1]))
    else:
        file_list = glob.glob(os.path.join(source, "*", *expr[:-1], expr[-1]))

    return sorted(file_list)


# def list_files(source, expr):
#     file_list = sorted(
#         glob.glob(os.path.join(source, "**", *expr), recursive=True))
#     return file_list


def create_destination_name(source):
    file_name = os.path.basename(source)
    subject_id = re.findall(r"\d+(?:-|_)\d+", file_name)

    if not subject_id:
        subject_id = re.findall(r"\d+(?:-|_)\d+", source)[0]
        file_name = "-".join([subject_id, file_name])

    file_name = file_name.strip("NP").replace("_", "-")
    print(file_name)

    return file_name


def copy_files_from_source(source, destination, expr):
    """[summary]

    Args:
        source ([type]): [description]
        destination ([type]): [description]
        expr ([type]): [description]

    Raises:
        Exception: [description]
    """
    os.makedirs(destination, exist_ok=True)
    file_list = list_files(source, expr)

    print(f"Copying {len(file_list)} files from source...")

    for file in file_list:
        dest_fn = create_destination_name(file)
        dst_scan_file = os.path.join(destination, dest_fn)
        copyfile(file, dst_scan_file)

    return


def copy_relevant_files(config, some_dict):

    for key in some_dict.keys():
        if some_dict[key]["expr"]:
            print(f'Copying {some_dict[key]["message"]}')
            src_name = some_dict[key]["source"]
            src_path = getattr(config, src_name, None)

            if not src_path:
                src_path = os.path.join(config.SYNTHSEG_RESULTS, src_name)
                if os.path.isdir(src_path):
                    setattr(config, src_name, src_path)
                else:
                    raise Exception("Source Directory Does not Exist")

            destinations = some_dict[key]["destination"]
            if isinstance(destinations, str):
                destinations = [destinations]

            for destination in destinations:
                dest = getattr(config, destination, None)
                copy_files_from_source(src_path, dest, some_dict[key]["expr"])
