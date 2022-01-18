import os

from ext.lab2im import utils
from nipype.interfaces.freesurfer import MRIConvert

from dice_config import *
from dice_utils import files_at_path, id_check, return_common_subjects


def run_mri_convert(in_file, ref_file, out_file):
    """This is NiPype Python Interface for mri_convert on Freesurfer

    Args:
        in_file (FileString): input file to convert
        ref_file (FileString): reference file
        out_file (FileString): name of the output file
    """
    mc = MRIConvert()
    mc.terminal_output = "none"
    mc.inputs.in_file = in_file
    mc.inputs.out_file = out_file
    if ref_file:
        mc.inputs.reslice_like = ref_file
    mc.inputs.out_type = "mgz"
    mc.inputs.out_datatype = "float"
    mc.inputs.resample_type = "nearest"

    mc.run()


def perform_overlay(config):
    mri_scans_reg = utils.list_images_in_folder(config.HARD_REF_WARPED)
    mri_resampled_segs = utils.list_images_in_folder(
        config.MRI_SCANS_SYNTHSEG_RESAMPLED)

    mri_scans_reg, mri_resampled_segs = return_common_subjects(
        mri_scans_reg, mri_resampled_segs)

    os.makedirs(config.MRI_SCANS_SYNTHSEG_REG_RES, exist_ok=True)

    print("Creating...")
    count = 0
    for scan_reg, mri_resampled_seg in zip(mri_scans_reg, mri_resampled_segs):
        if not id_check(scan_reg, mri_resampled_seg):
            raise Exception("ID Check Failed")

        _, scan_reg_aff, scan_reg_head = utils.load_volume(scan_reg,
                                                           im_only=False)
        mrs_im = utils.load_volume(mri_resampled_seg)

        _, file_name = os.path.split(mri_resampled_seg)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + ".reg" + file_ext
        print(out_file)
        out_file = os.path.join(config.MRI_SCANS_SYNTHSEG_REG_RES, out_file)

        # We can now combine the segmentation voxels with the registered header.
        utils.save_volume(mrs_im, scan_reg_aff, scan_reg_head, out_file)

        count += 1
    print(f"Overlayed {count} files")
    # this new file should overlay with the 3D photo reconstruction


def convert_to_single_channel(config, folder_str):
    """Convert RGB volume to Grayscale volume

    Args:
        config ([type]): [description]
        folder_str ([type]): [description]
    """
    file_list = utils.list_images_in_folder(
        config.__getattribute__(folder_str))
    for file in file_list:
        im, aff, hdr = utils.load_volume(file, im_only=False)

        if im.ndim == 4 and im.shape[-1] == 3:
            im = np.mean(im, axis=-1)

        utils.save_volume(im, aff, hdr, file)
    pass


def perform_registration(config, input_path, reference_path, output_path):
    input_files = utils.list_images_in_folder(input_path)
    reference_files = utils.list_images_in_folder(reference_path)

    input_files, reference_files = return_common_subjects(
        input_files, reference_files)

    os.makedirs(output_path, exist_ok=True)

    print("Creating...")
    count = 0
    for input_file, reference_file in zip(input_files, reference_files):
        if not id_check(input_file, reference_file):
            raise Exception("ID Check Failed")

        _, file_name = os.path.split(input_file)
        file_name, file_ext = os.path.splitext(file_name)

        out_file = file_name + ".res" + file_ext
        print(out_file)
        out_file = os.path.join(output_path, out_file)

        run_mri_convert(input_file, reference_file, out_file)
        count += 1
    print(f"Registered {count} files")


def move_volumes_into_target_spaces(config, item_list):
    for item in item_list:
        source = getattr(config, item["source"], None)
        reference = getattr(config, item["reference"], None)
        target = getattr(config, item["target"], None)

        if not source:
            raise Exception(f'Source folder {item["source"]} does not exist')

        if not reference:
            raise Exception(
                f'Reference folder {item["reference"]} does not exist')

        if not target:
            print(f"Target folder DNE: But adding now")
            target_folder_name = item["target"].lower().replace("_", ".")
            setattr(
                config,
                item["target"],
                f"{config.SYNTHSEG_RESULTS}/{target_folder_name}",
            )
            target = getattr(config, item["target"], None)

        print(f'Putting {" ".join(item["target"].split("_"))}')
        perform_registration(config, source, reference, target)
