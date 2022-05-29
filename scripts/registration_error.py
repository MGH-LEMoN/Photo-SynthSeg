import glob
import os
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from ext.lab2im import utils

rcParams.update({"figure.autolayout": True})

sns.set(
    style="whitegrid",
    rc={
        "text.usetex": True,
        "font.family": "serif",
    },
)


# CUSTOM = 'subject_133'
PRJCT_DIR = "/space/calico/1/users/Harsha/SynthSeg"


def get_error(skip, jitter, sub_id=None):
    PAD = 3  # for reconstruction

    results_dir, subjects = subject_selector(skip, jitter)

    subject_id = subjects[sub_id]
    subject_dir = os.path.join(results_dir, subject_id)
    # print(sub_id)

    # rigid transform from T1
    rigid_transform = np.load(os.path.join(subject_dir, f"{subject_id}.rigid.npy"))
    # print(rigid_transform.shape)

    # photo affine from T2
    photo_affines = sorted(
        glob.glob(os.path.join(subject_dir, "slice_affines", "*.npy"))
    )
    photo_affine_matrix = np.stack([np.load(item) for item in photo_affines], axis=2)
    # print(photo_affine_matrix.shape)
    # print()

    recon_M = np.load(
        os.path.join(subject_dir, f"ref_mask_skip_{skip:02d}", "slice_matrix_M.npy")
    )
    recon_M = recon_M[:, :, PAD:-PAD]  # removing matrices corresponding to padding
    recon_M = recon_M[:, [0, 1, 3], :]
    recon_M = recon_M[[0, 1, 3], :, :]
    # print(recon_M.shape)

    all_paddings = np.load(
        os.path.join(subject_dir, f"ref_mask_skip_{skip:02d}", "all_paddings.npy")
    )

    assert photo_affine_matrix.shape[-1] == recon_M.shape[-1], "Slice count mismatch"

    t1_path = os.path.join(subject_dir, f"{subject_id}.T1.nii.gz")
    t2_path = os.path.join(subject_dir, f"{subject_id}.T2.nii.gz")

    t1_vol, t1_aff, t1_hdr = utils.load_volume(t1_path, im_only=False)
    t2_vol, t2_aff, t2_hdr = utils.load_volume(t2_path, im_only=False)

    Nslices_of_T2 = t2_vol.shape[-1]
    errors_norm_slices = []
    for z in range(Nslices_of_T2):
        curr_slice = t2_vol[..., z]
        if np.sum(curr_slice) and not z % skip:
            harshas_z = z
            break
    skip = skip

    for z in range(photo_affine_matrix.shape[-1]):
        curr_slice = t2_vol[..., harshas_z + skip * z]
        # print(curr_slice.shape)
        curr_slice = np.pad(np.rot90(curr_slice), 25)
        # curr_slice[i-1:i+1, j-1:j+1] = 2 * np.max(curr_slice)
        # plt.imshow(curr_slice)

        i, j = np.where(curr_slice > 0)
        v = np.stack([i, j, np.ones(i.shape)])
        v2 = np.matmul(np.linalg.inv(photo_affine_matrix[:, :, z]), v)

        recon = os.path.join(
            subject_dir, f"ref_mask_skip_{skip:02d}", "photo_recon.mgz"
        )
        recon_vol, recon_aff, recon_hdr = utils.load_volume(recon, im_only=False)
        recon_vol = recon_vol[
            :, :, PAD:-PAD, :
        ]  # removing slices corresponding to padding
        # print(recon_vol.shape)

        zp = len(all_paddings) - z - 1

        P = np.eye(3)
        P[:-1, -1] = all_paddings[zp]
        Pinv = np.eye(3)
        Pinv[:-1, -1] = -all_paddings[zp]
        M = recon_M[:, :, zp]

        v3 = np.matmul(P, v2)
        v4 = np.matmul(np.linalg.inv(M), v3)

        v5 = np.matmul(Pinv, v4)

        v4_3d = np.stack([v4[0, :], v4[1, :], (zp + PAD) * v4[-1, :], v4[-1, :]])

        # v4_3d = np.array([i4, j4, zp+3, 1])
        ras_new = np.matmul(recon_aff, v4_3d)
        # print(v4_3d)
        # print(ras_new)

        # Undo padding / rotation
        ii = j - 25
        jj = curr_slice.shape[0] - i - 1 - 25

        v_3d = np.stack(
            [ii, jj, (harshas_z + skip * z) * np.ones(ii.shape), np.ones(ii.shape)]
        )
        # v_3d = np.array([ii, jj, harshas_z + skip * z, 1])

        ras_gt = np.matmul(rigid_transform, np.matmul(t1_aff, v_3d))

        # errors_slice = np.linalg.norm(ras_new-ras_gt)
        errors_slice = ras_new[:-1] - ras_gt[:-1]
        error_norms_slice = np.sqrt(np.sum(errors_slice**2, axis=0))

        errors_norm_slices.append(error_norms_slice)

    return (
        errors_norm_slices,
        np.nanmean(np.concatenate(errors_norm_slices)),
        np.nanstd(np.concatenate(errors_norm_slices)),
    )


def save_errors(skip, jitter, corr):
    all_errors = [item[0] for item in corr]
    all_means = [item[1] for item in corr]
    all_stds = [item[2] for item in corr]

    head_dir = get_results_dir(skip, jitter)
    os.makedirs(head_dir, exist_ok=True)

    np.save(os.path.join(head_dir, f"hcp-skip-{skip:02d}-errors"), all_errors)
    np.save(os.path.join(head_dir, f"hcp-skip-{skip:02d}-means"), all_means)
    np.save(os.path.join(head_dir, f"hcp-skip-{skip:02d}-stds"), all_stds)


def get_results_dir(skip, jitter):
    return (
        f"{PRJCT_DIR}/results/hcp-results-20220528/4harshaHCP-skip-{skip:02d}-r{jitter}"
    )


def subject_selector(skip, jitter):
    results_dir = get_results_dir(skip, jitter)
    subjects = sorted(os.listdir(results_dir))
    # subjects = [item for item in subjects if CUSTOM in item]
    return results_dir, subjects


def get_error_wrapper(sub_id, skip, jitter):
    _, subjects = subject_selector(skip, jitter)
    try:
        return get_error(skip, jitter, sub_id)
    except:
        print(f"Failed: {subjects[sub_id]}")
        return None, None, None


def main_mp(skip, jitter):
    _, subjects = subject_selector(skip, jitter)

    with Pool() as pool:
        corr = pool.map(
            partial(get_error_wrapper, skip=skip, jitter=jitter), range(len(subjects))
        )

    save_errors(skip, jitter, corr)


def get_means_and_stds(jitter):
    r3_means = []
    r3_stds = []
    skip_range = list(range(2, 3, 2))

    for skip in skip_range:
        means_file = os.path.join(
            get_results_dir(skip, jitter), f"hcp-skip-{skip:02d}-means.npy"
        )
        stds_file = os.path.join(
            get_results_dir(skip, jitter), f"hcp-skip-{skip:02d}-stds.npy"
        )

        if not os.path.exists(means_file):
            skip_range.remove(skip)
            continue

        means = np.load(means_file, allow_pickle=True)
        stds = np.load(stds_file, allow_pickle=True)

        r3_means.append(means)
        r3_stds.append(stds)

    r3_mean_df = pd.DataFrame(r3_means).T
    r3_mean_df.columns = np.round(np.array(skip_range) * 0.7, 1)

    r3_mean_df = r3_mean_df.stack().reset_index().drop(labels=["level_0"], axis=1)
    r3_mean_df.columns = ["Skip", "Mean Error"]

    r3_mean_df["Jitter"] = jitter
    return r3_mean_df


def boxplot_jitter():
    df_list = []
    for jitter in range(0, 4):
        df_list.append(get_means_and_stds(jitter))

    final_df = pd.concat(df_list, axis=0)

    ax = sns.boxplot(
        x="Jitter", y="Mean Error", hue="Skip", data=final_df, palette="Greys_r"
    )
    fig = ax.get_figure()
    ax.legend(ncol=5, edgecolor="white", framealpha=0.25)
    fig.savefig("output_all_jitters.png")


if __name__ == "__main__":
    for skip in range(2, 3, 2):
        for jitter in range(0, 4):
            print(f"SKIP: {skip:02d}-r{jitter}")
            main_mp(skip, jitter)

    # boxplot_jitter()
