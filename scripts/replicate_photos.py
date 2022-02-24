import glob
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from ext.hg_utils import zoom
from ext.lab2im import utils
from scipy.ndimage import affine_transform
from tqdm import tqdm

PRJCT_DIR = '/space/calico/1/users/Harsha/SynthSeg'
DATA_DIR = os.path.join(PRJCT_DIR, 'data')
IN_DIR = os.path.join(DATA_DIR, '4harshaHCP')
OUT_DIR = os.path.join(DATA_DIR, '4harshaHCP_extracts')


def process_t1(t1_file, t1_name):

    # 1. Sample 3 rotations about the 3 axes, e.g., between -30 and 30 degrees.
    rotation = np.random.randint(-30, 31, 3)

    # 2. Sample 3 translations along the 3 axes, e.g., between 20 and 20 mm
    translation = np.random.randint(-20, 21, 3)

    # 3. Build a rigid 3D rotation + translation (4x4) matrix using the rotations and shifts
    t1_rigid_mat = utils.create_affine_transformation_matrix(
        3, None, rotation, None, translation)

    t1_rigid_out = os.path.join(OUT_DIR, t1_name, f'{t1_name}.rigid.npy')
    np.save(t1_rigid_out, t1_rigid_mat)

    # 4. Open the T1, and premultiply the affine matrix of the header (“vox2ras”) by the matrix from 3.
    volume, aff, hdr = utils.load_volume(t1_file, im_only=False)
    new_aff = np.matmul(t1_rigid_mat, aff)
    hdr.set_sform(new_aff)

    # 5. Binarize the T1 volume by thresholding at 0 and save it with the new header, and call it “mri.mask.mgz”
    t1_out_path = os.path.join(OUT_DIR, t1_name, f'{t1_name}.mri.mask.mgz')
    utils.save_volume(volume > 0, new_aff, hdr, t1_out_path)


def process_t2(t2_file, t2_name):

    AFFINE_DIR = os.path.join(OUT_DIR, t2_name, 'slice_affines')
    PHOTO_DIR = os.path.join(OUT_DIR, t2_name, 'photo_dir')

    # 6. Create a directory “photo_dir"
    os.makedirs(AFFINE_DIR, exist_ok=True)
    os.makedirs(PHOTO_DIR, exist_ok=True)

    # 7. Open the T2
    t2_vol = utils.load_volume(t2_file)

    Nslices_of_T2 = t2_vol.shape[-1]
    c = 0
    for z in range(Nslices_of_T2):
        curr_slice = t2_vol[..., z]

        if np.sum(curr_slice) and not z % 6:
            c += 1

            # Sample a rotation eg between -30 and 30 degrees
            rotation = np.random.randint(-30, 31, 1)

            # Sample 2 translations along the 2 axes, eg, between -10 and 10 pixels
            translation = np.random.randint(-10, 11, 2)

            # Sample 2 small shears about the 2 axes (eg between -0.1 and 0.1)
            shearing = np.random.uniform(-0.1, 0.1, 2)

            # Build a 2D (3x3) matrix with the rotation, translations, and shears
            slice_aff_mat = utils.create_affine_transformation_matrix(
                2, None, rotation, shearing, translation)

            # Save this matrix somewhere for evaluation later on eg as a numpy array
            slice_aff_out = os.path.join(AFFINE_DIR,
                                         f'{t2_name}.slice.{c:03d}.npy')
            np.save(slice_aff_out, slice_aff_mat)

            # Use this matrix to deform the slice
            deformed_slice = affine_transform(curr_slice,
                                              slice_aff_mat,
                                              mode='constant',
                                              order=1)

            # Threshold the deformed slice at zero to get a mask (1 inside, 0 outside)
            mask = curr_slice > 0
            # write it as photo_dir/image.[c].npy
            # (format the number c with 2 digits so they are in order when listed)
            out_file_name = os.path.join(PHOTO_DIR,
                                         f'{t2_name}.image.{c:03d}.npy')
            np.save(out_file_name, mask)

            # add illumination field to the slice
            # Sample a random zero-mean gaussian tensor of size (5x5) and multiply by a small standard deviation (eg 0.1)
            small_vol = 0.1 * np.random.normal(size=(5, 5))

            # Upscale the tensor to the size of the slice
            # edit_volumes.resample_volume(small)
            factors = np.divide(mask.shape, small_vol.shape)

            # Take the pixel-wise exponential of the upsampled tensor to get an illumination field
            bias_result = zoom.scipy_zoom(small_vol, factors, mask.shape)

            # Multiply the deformed slice by the illumination field
            corrupted_image = np.multiply(deformed_slice, bias_result)

            # Write the corrupted image to photo_dir/image.[c].tif
            img_out = os.path.join(PHOTO_DIR, f'{t2_name}.image.{c:03d}.tiff')
            plt.imsave(img_out, corrupted_image)

            # fig, ax = plt.subplots(1, 2)
            # l0 = ax[0].imshow(curr_slice)
            # l1 = ax[1].imshow(aff_slice)
            # plt.colorbar(l0, ax=ax[0])
            # plt.colorbar(l1, ax=ax[1])


def main():
    # list all T1 and T2 files
    t1_files = sorted(glob.glob(os.path.join(IN_DIR, '*T1.nii.gz')))
    t2_files = sorted(glob.glob(os.path.join(IN_DIR, '*T2.nii.gz')))

    assert len(t1_files) == len(t2_files), 'Subject Mismatch'

    for i in tqdm(range(len(t1_files[:50]))):
        t1_file = t1_files[i]
        t2_file = t2_files[i]

        # get file name
        t1_fname = os.path.split(t1_file)[1]
        t2_fname = os.path.split(t2_file)[1]

        # get subject ID
        t1_subject_name = t1_fname.split('.')[0]
        t2_subject_name = t2_fname.split('.')[0]

        assert t1_subject_name == t2_subject_name, "Incorrect Subject Name"

        # make output directory for subject
        out_subject_dir = os.path.join(OUT_DIR, t1_subject_name)
        os.makedirs(out_subject_dir, exist_ok=True)

        # create symlinks to source files (T1, T2)
        t1_dst = os.path.join(out_subject_dir, t1_fname)
        t2_dst = os.path.join(out_subject_dir, t2_fname)

        if not os.path.exists(t1_dst):
            os.symlink(t1_file, t1_dst)

        if not os.path.exists(t2_dst):
            os.symlink(t2_file, t2_dst)

        # work on T1 and T2 volumes
        process_t1(t1_file, t1_subject_name)
        process_t2(t2_file, t2_subject_name)


if __name__ == '__main__':
    main()
