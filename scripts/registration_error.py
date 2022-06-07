import glob
import json
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

SOME_SUFFIX = ""
# {curtailed | identity |}


def get_nonzero_slice_ids(t2_vol):
    """find all non-zero slices"""
    slice_sum = np.sum(t2_vol, axis=(0, 1))
    return np.where(slice_sum > 0)[0]


def slice_ids_method1(skip, t2_vol):
    """current method of selecting slices
    Example:
    slice_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    slices:    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1,  1,  0,  0,  0]
    selected:  [0, 0, 0, 0, 1, 0, 1, 0, 1, 0,  1,  0,  0,  0] (skip = 2)
    selected:  [0, 0, 0, 1, 0, 0, 1, 0, 0, 1,  0,  0,  0,  0] (skip = 3)
    """
    non_zero_slice_ids = get_nonzero_slice_ids(t2_vol)

    first_nz_slice = non_zero_slice_ids[0]
    slice_ids_of_interest = np.where(non_zero_slice_ids % skip == 0)
    slice_ids_of_interest = slice_ids_of_interest[0] + first_nz_slice

    return slice_ids_of_interest


def get_middle_elements(test_list, K):
    # using list slicing
    return test_list[
        int(len(test_list) / 2) - int(K / 2) : int(len(test_list) / 2) + int(K / 2) + 1
    ]


def get_error_optimized(results_dir, sub_id=None):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_
        sub_id (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    PAD = 3  # for reconstruction

    skip = get_skip(results_dir)

    rigid_transform = np.load(utils.list_files(results_dir, True, "rigid.npy")[0])
    photo_affines = sorted(
        utils.list_files(os.path.join(results_dir, "slice_affines"), True, "*.npy")
    )
    photo_affine_matrix = np.stack([np.load(item) for item in photo_affines], axis=2)

    recon_matrix = np.load(
        os.path.join(results_dir, f"ref_mask_skip_{skip:02d}", "slice_matrix_M.npy")
    )
    # # assuming i renamed the ref_mask folders in all subjects
    # recon_matrix = np.load(
    #     utils.list_files(
    #         os.path.join(results_dir, "ref_mask"), True, "slice_matrix_M.npy"
    #     )[0]
    # )

    recon_matrix = recon_matrix[
        :, :, PAD:-PAD
    ]  # removing matrices corresponding to padding
    recon_matrix = recon_matrix[:, [0, 1, 3], :]
    recon_matrix = recon_matrix[[0, 1, 3], :, :]

    # all_paddings = np.load(
    #     os.path.join(subject_dir, f"ref_mask_skip_{skip:02d}", "all_paddings.npy")
    # )
    all_paddings = np.load(
        utils.list_files(
            os.path.join(results_dir, "ref_mask"), True, "all_paddings.npy"
        )[0]
    )

    assert (
        photo_affine_matrix.shape[-1] == recon_matrix.shape[-1]
    ), "Slice count mismatch"

    t1_path = utils.list_files(results_dir, True, "T1.nii.gz")[0]
    t2_path = utils.list_files(results_dir, True, "T2.nii.gz")[0]

    _, t1_aff, _ = utils.load_volume(t1_path, im_only=False)
    t2_vol, _, _ = utils.load_volume(t2_path, im_only=False)

    num_slices_t2 = t2_vol.shape[-1]
    errors_norm_slices = []
    for z in range(num_slices_t2):
        curr_slice = t2_vol[..., z]
        if np.sum(curr_slice) and not z % skip:
            harshas_z = z
            break

    recon = os.path.join(results_dir, f"ref_mask_skip_{skip:02d}", "photo_recon.mgz")
    recon_vol, recon_aff, _ = utils.load_volume(recon, im_only=False)
    recon_vol = recon_vol[:, :, PAD:-PAD, :]  # removing slices corresponding to padding

    for z in range(photo_affine_matrix.shape[-1]):
        curr_slice = t2_vol[..., harshas_z + skip * z]
        curr_slice = np.pad(np.rot90(curr_slice), 25)
        # curr_slice[i-1:i+1, j-1:j+1] = 2 * np.max(curr_slice)
        # plt.imshow(curr_slice)

        i, j = np.where(curr_slice > 0)
        v = np.stack([i, j, np.ones(i.shape)])
        v2 = np.matmul(np.linalg.inv(photo_affine_matrix[:, :, z]), v)

        zp = len(all_paddings) - z - 1

        P = np.eye(3)
        P[:-1, -1] = all_paddings[zp]
        Pinv = np.eye(3)
        Pinv[:-1, -1] = -all_paddings[zp]
        M = recon_matrix[:, :, zp]

        v3 = np.matmul(P, v2)
        v4 = np.matmul(np.linalg.inv(M), v3)

        # v5 = np.matmul(Pinv, v4)

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

    # putting errors in a volume

    return sub_id, (
        errors_norm_slices,
        np.nanmean(np.concatenate(errors_norm_slices)),
        np.nanstd(np.concatenate(errors_norm_slices)),
        np.nanmedian(np.concatenate(errors_norm_slices)),
    )


def get_error(results_dir, sub_id=None):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_
        sub_id (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    pad = 3  # for reconstruction

    skip = get_skip(results_dir)

    subject_id = utils.list_subfolders(results_dir, False)[sub_id]
    subject_dir = os.path.join(results_dir, subject_id)

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

    recon_matrix = np.load(
        os.path.join(subject_dir, f"ref_mask_skip_{skip:02d}", "slice_matrix_M.npy")
    )
    recon_matrix = recon_matrix[
        :, :, pad:-pad
    ]  # removing matrices corresponding to padding
    recon_matrix = recon_matrix[:, [0, 1, 3], :]
    recon_matrix = recon_matrix[[0, 1, 3], :, :]
    # print(recon_matrix.shape)

    # num_repeats = recon_matrix.shape[-1]
    # recon_matrix = np.concatenate([np.eye(3)[..., None]] * num_repeats, axis=-1)

    all_paddings = np.load(
        os.path.join(subject_dir, f"ref_mask_skip_{skip:02d}", "all_paddings.npy")
    )

    assert (
        photo_affine_matrix.shape[-1] == recon_matrix.shape[-1]
    ), "Slice count mismatch"

    t1_path = os.path.join(subject_dir, f"{subject_id}.T1.nii.gz")
    t2_path = os.path.join(subject_dir, f"{subject_id}.T2.nii.gz")

    _, t1_aff, _ = utils.load_volume(t1_path, im_only=False)
    t2_vol, _, _ = utils.load_volume(t2_path, im_only=False)

    recon = os.path.join(subject_dir, f"ref_mask_skip_{skip:02d}", "photo_recon.mgz")
    recon_vol, recon_aff, _ = utils.load_volume(recon, im_only=False)
    # recon_vol = recon_vol[
    #     :, :, pad:-pad, :
    # ]  # removing slices corresponding to padding
    # # print(recon_vol.shape)

    # num_slices_t2 = t2_vol.shape[-1]
    errors_norm_slices = []
    # for z in range(num_slices_t2):
    #     curr_slice = t2_vol[..., z]
    #     if np.sum(curr_slice) and not z % skip:
    #         harshas_z = z
    #         break

    slice_ids_of_interest = slice_ids_method1(skip, t2_vol)

    # get the number of slices in it's highest spacing counterpart
    # (for equal comparison with the reconstruction)
    # high_subject_dir = subject_dir.replace(f"skip-{skip:02d}", "skip-16")
    # num_slices = len(
    #     utils.list_files(os.path.join(high_subject_dir, "slice_affines"), False, ".npy")
    # )
    # keep_slice_list = get_middle_elements(slice_ids_of_interest, num_slices)

    errors_vol = np.zeros(
        (*t2_vol.shape[:-1], len(slice_ids_of_interest)), dtype=np.float32
    )
    for z, slice_id in enumerate(slice_ids_of_interest):
        # if slice_id not in keep_slice_list:
        #     continue

        curr_slice = t2_vol[..., slice_id]
        curr_slice = np.pad(np.rot90(curr_slice), 25)
        # curr_slice[i-1:i+1, j-1:j+1] = 2 * np.max(curr_slice)
        # plt.imshow(curr_slice)

        i, j = np.where(curr_slice > 0)
        v = np.stack([i, j, np.ones(i.shape)])
        v2 = np.matmul(np.linalg.inv(photo_affine_matrix[:, :, z]), v)

        # zp = len(all_paddings) - z - 1
        zp = len(all_paddings) - z - 1

        P = np.eye(3)
        P[:-1, -1] = all_paddings[zp]
        Pinv = np.eye(3)
        Pinv[:-1, -1] = -all_paddings[zp]
        M = recon_matrix[:, :, zp]

        v3 = np.matmul(P, v2)
        v4 = np.matmul(np.linalg.inv(M), v3)

        # v5 = np.matmul(Pinv, v4)

        v4_3d = np.stack([v4[0, :], v4[1, :], (zp + pad) * v4[-1, :], v4[-1, :]])

        # v4_3d = np.array([i4, j4, zp+3, 1])
        ras_new = np.matmul(recon_aff, v4_3d)
        # print(v4_3d)
        # print(ras_new)

        # Undo padding / rotation
        ii = j - 25
        jj = curr_slice.shape[0] - i - 1 - 25

        v_3d = np.stack([ii, jj, slice_id * np.ones(ii.shape), np.ones(ii.shape)])
        # v_3d = np.array([ii, jj, harshas_z + skip * z, 1])

        ras_gt = np.matmul(rigid_transform, np.matmul(t1_aff, v_3d))

        # errors_slice = np.linalg.norm(ras_new-ras_gt)
        errors_slice = ras_new[:-1] - ras_gt[:-1]
        error_norms_slice = np.sqrt(np.sum(errors_slice**2, axis=0))

        errors_norm_slices.append(error_norms_slice)

        # putting errors in a volume
        # errors_vol[ii, jj, z] = errors_slice

    return sub_id, (
        errors_norm_slices,
        np.nanmean(np.concatenate(errors_norm_slices)),
        np.nanstd(np.concatenate(errors_norm_slices)),
        np.nanmedian(np.concatenate(errors_norm_slices)),
    )


def save_errors(results_dir, corr):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_
        corr (_type_): _description_
    """
    # FIXME: clean this function
    subject_ids = [int(item[0]) for item in corr]
    all_errors = [item[1][0] for item in corr]
    all_means = [item[1][1] for item in corr]
    all_stds = [item[1][2] for item in corr]
    all_medians = [item[1][3] for item in corr]

    all_errors = dict(zip(subject_ids, all_errors))
    all_means = dict(zip(subject_ids, all_means))
    all_stds = dict(zip(subject_ids, all_stds))
    all_medians = dict(zip(subject_ids, all_medians))

    all_mean_of_means = dict()
    for k, v in all_errors.items():
        try:
            all_mean_of_means[k] = np.mean([np.mean(slice) for slice in v])
        except:
            all_mean_of_means[k] = None

    jitter_val = get_jitter(results_dir)
    skip_val = get_skip(results_dir)

    file_suffix = f"skip-{skip_val:02d}-r{jitter_val}"

    head_dir = os.path.join(
        os.path.dirname(results_dir),
        "-".join(["hcp-errors", SOME_SUFFIX].strip("-")),
    )
    os.makedirs(head_dir, exist_ok=True)

    np.save(os.path.join(head_dir, f"hcp-errors-{file_suffix}"), all_errors)
    # np.save(os.path.join(head_dir, f"hcp-means-{file_suffix}"), all_means)
    # np.save(os.path.join(head_dir, f"hcp-stds-{file_suffix}"), all_stds)
    # np.save(os.path.join(head_dir, f"hcp-medians-{file_suffix}"), all_medians)

    # with open(os.path.join(head_dir, f"hcp-errors-{file_suffix}"), "w") as write_file:
    #     json.dump(all_errors, write_file, indent=4)

    with open(os.path.join(head_dir, f"hcp-means-{file_suffix}"), "w") as write_file:
        json.dump(all_means, write_file, indent=4)

    with open(os.path.join(head_dir, f"hcp-stds-{file_suffix}"), "w") as write_file:
        json.dump(all_stds, write_file, indent=4)

    with open(os.path.join(head_dir, f"hcp-medians-{file_suffix}"), "w") as write_file:
        json.dump(all_medians, write_file, indent=4)

    with open(
        os.path.join(head_dir, f"hcp-mean-of-means-{file_suffix}"), "w"
    ) as write_file:
        json.dump(all_mean_of_means, write_file, indent=4)


def subject_selector(results_dir, n_size=None):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_count = len(utils.list_subfolders(results_dir))

    if file_count == 100:
        subject_ids = range(file_count)
        return subject_ids

    if file_count != 897:  # This is the total number of subjects for HCP
        print("Not all subjects are present in here")
        file_count = 897
    # subjects = [item for item in subjects if CUSTOM in item]

    if n_size is not None:
        subject_ids = np.random.choice(range(file_count), n_size, replace=False)
    else:
        subject_ids = range(file_count)

    return subject_ids


def get_error_wrapper(sub_id, results_dir):
    """_summary_

    Args:
        sub_id (_type_): _description_
        skip (_type_): _description_
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    subjects = sorted(os.listdir(results_dir))

    try:
        return get_error(results_dir, sub_id)
    except:
        print(f"Failed: {subjects[sub_id]}")
        return sub_id, (None, None, None, None)


def get_error_wrapper_new(sub_id, results_dir):
    """_summary_

    Args:
        sub_id (_type_): _description_
        skip (_type_): _description_
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    subjects = sorted(os.listdir(results_dir))

    subject_id = utils.list_subfolders(results_dir, False)[sub_id]
    subject_dir = os.path.join(results_dir, subject_id)

    try:
        return get_error_optimized(subject_dir)
    except:
        print(f"Failed: {subjects[sub_id]}")
        return sub_id, (None, None, None, None)


def main_mp(results_dir, sample_size=None):
    """_summary_

    Args:
        skip (_type_): _description_
        jitter (_type_): _description_
    """
    subjects = subject_selector(results_dir, sample_size)

    with Pool() as pool:
        corr = pool.map(partial(get_error_wrapper, results_dir=results_dir), subjects)

    save_errors(results_dir, corr)


def get_jitter(dir_name):
    """_summary_

    Args:
        dir_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    return int(dir_name[-1])


def get_skip(file_path):
    """_summary_

    Args:
        file_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(file_path, str):
        return int(file_path.split("-")[-2])
    else:
        return sorted(list({int(item.split("-")[-2]) for item in file_path}))

    # def get_means_and_stds_old(dir_path):
    """_summary_

    Args:
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_list = utils.list_files(dir_path, True, "means.npy")

    r3_means = [np.load(file, allow_pickle=True) for file in file_list]
    skip_vals = get_skip(file_list)

    r3_mean_df = pd.DataFrame(r3_means).T
    r3_mean_df.columns = np.round(np.array(skip_vals) * 0.7, 1)

    r3_mean_df = r3_mean_df.stack().reset_index().drop(labels=["level_0"], axis=1)
    r3_mean_df.columns = ["Skip", "Mean Error"]

    r3_mean_df["Jitter"] = get_jitter(dir_path)

    return r3_mean_df


def get_just_means_for_filtering(dir_path):
    file_list = utils.list_files(dir_path, True, "hcp-means-skip")

    r3_means = [pd.read_json(file, orient="index") for file in file_list]

    col_item1 = lambda x: np.round(np.array(get_skip(x)) * 0.7, 1)
    columns = [str(col_item1(file)) + "_" + str(get_jitter(file)) for file in file_list]

    r3_mean_df = pd.concat(r3_means, axis=1)
    # print(f"Before removing NaN rows: {r3_mean_df.shape[0]}")

    r3_mean_df = r3_mean_df.dropna(axis=0, how="any")
    # print(f"After removing NaN rows: {r3_mean_df.shape[0]}")

    if r3_mean_df.empty:
        return None

    r3_mean_df.columns = columns

    return r3_mean_df["1.4_0"] < 0.8


def get_means_and_stds(dir_path, error_type="mean"):
    """_summary_

    Args:
        jitter (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("hcp-" + error_type + "-skip")
    file_list = utils.list_files(dir_path, True, "hcp-" + error_type + "-skip")

    r3_means = [pd.read_json(file, orient="index") for file in file_list]

    col_item1 = lambda x: np.round(np.array(get_skip(x)) * 0.7, 1)
    columns = [str(col_item1(file)) + "_" + str(get_jitter(file)) for file in file_list]

    r3_mean_df = pd.concat(r3_means, axis=1)
    print(f"Before removing NaN rows: {r3_mean_df.shape[0]}")

    r3_mean_df = r3_mean_df.dropna(axis=0, how="any")
    print(f"After removing NaN rows: {r3_mean_df.shape[0]}")

    if r3_mean_df.empty:
        return None

    r3_mean_df.columns = columns

    # filter subjects on the two clusters found in means
    r3_mean_df = r3_mean_df[get_just_means_for_filtering(dir_path)]

    r3_mean_df = r3_mean_df.stack().reset_index().drop(labels=["level_0"], axis=1)
    r3_mean_df.columns = ["Skip", f"{error_type.capitalize()} Error"]

    r3_mean_df[["Spacing", "Jitter"]] = pd.DataFrame(
        r3_mean_df["Skip"].str.split("_").tolist()
    )
    r3_mean_df = r3_mean_df.drop(labels=["Skip"], axis=1)

    return r3_mean_df


def make_error_boxplot(data_frame, out_file, x_col, hue_col, type_str):
    """_summary_

    Args:
        df (_type_): _description_
        out_file (_type_): _description_
        x_col (_type_): _description_
        hue_col (_type_): _description_
    """
    bp_ax = sns.violinplot(
        x=x_col,
        y=f"{type_str.capitalize()} Error",
        hue=hue_col,
        data=data_frame,
        palette="Greys_r",
        linewidth=0.5,
    )
    fig = bp_ax.get_figure()
    # bp_ax.legend(ncol=10, edgecolor="white", framealpha=0.25)
    fig.savefig(out_file, dpi=1200, bbox_inches="tight")
    plt.clf()


def plot_file_name(results_dir, plot_idx, type):
    """_summary_

    Args:
        results_dir (_type_): _description_
        plot_idx (_type_): _description_
    """
    out_file = f"l8{SOME_SUFFIX}-{type}_{plot_idx:02d}.png".strip("-")
    out_file = os.path.join(results_dir, out_file)
    return out_file


def plot_registration_error(results_dir, error_str="mean"):
    """_summary_

    Args:
        jitters (_type_): _description_
    """
    print(error_str)

    errors_dir = os.path.join(
        results_dir, "-".join(["hcp-errors", SOME_SUFFIX]).strip("-")
    )
    final_df = get_means_and_stds(errors_dir, error_str)

    if final_df is not None:
        out_file = plot_file_name(results_dir, 1, error_str)
        make_error_boxplot(final_df, out_file, "Spacing", "Jitter", error_str)

        out_file = plot_file_name(results_dir, 2, error_str)
        make_error_boxplot(final_df, out_file, "Jitter", "Spacing", error_str)
    else:
        print(f"Empty DataFrame for {error_str}")


def calculate_registration_error(results_dir, n_subjects=None):
    """_summary_"""
    recon_folders = utils.list_subfolders(results_dir, True, "4harshaHCP-skip-")

    for recon_folder in recon_folders:
        if "ex" in os.path.basename(recon_folder):
            continue
        np.random.seed(0)
        print(os.path.basename(recon_folder))
        main_mp(recon_folder, n_subjects)


if __name__ == "__main__":
    PRJCT_DIR = "/space/calico/1/users/Harsha/SynthSeg/results"

    FOLDER = "hcp-results-20220601"
    # {options: hcp-results | hcp-results-2020527 | hcp-results-2020528}

    # set this to a high value if you want to run all subjects
    # there are nearly 897 subjects in the dataset
    M = 100

    full_results_path = os.path.join(PRJCT_DIR, FOLDER)

    if not os.path.exists(full_results_path):
        raise Exception("Folder does not exist")

    # calculate_registration_error(full_results_path, M)
    for stat_key in ["means", "stds", "medians", "mean-of-means"]:
        plot_registration_error(full_results_path, stat_key)
