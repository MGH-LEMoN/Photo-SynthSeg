import os
import subprocess
from time import time

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tabulate import tabulate

LABELS_SHAPE = [256, 256, 256]

# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_branch() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("ascii")
        .strip()
    )

    # return (
    #     subprocess.check_output(["git", "branch", "--show-current"])
    #     .decode("ascii")
    #     .strip()
    # )


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def print_shapes(input, factors):
    in_shape = input.shape
    out_shape = np.multiply(input.shape, factors)
    print(f" Going from {in_shape} --> {out_shape}")


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(
            f"\tFunction {func.__name__!r:20}: {(t2-t1):.4f}s     Sum: {np.sum(result)}"
        )
        return result

    return wrap_func


def generate_small_bias(spacing):
    bias_shape_factor = [0.025, 1 / spacing, 0.025]
    bias_field_std = 0.5

    small_bias_size = np.ceil(np.multiply(LABELS_SHAPE, bias_shape_factor)).astype(int)

    small_bias = (
        bias_field_std
        * np.random.uniform(size=[1])
        * np.random.normal(size=small_bias_size)
    )

    factors = np.divide(1.0, bias_shape_factor).tolist()

    return small_bias, factors


# @timer_func
def scipy_zoom(small_vol, factors, target_shape=None, grid_mode=True, flag="bias"):
    if flag == "def":
        factors = factors + [1]

    try:
        out_vol = zoom(small_vol, factors, order=1, grid_mode=grid_mode)
    except:
        out_vol = zoom(small_vol, factors, order=1)

    if flag == "bias":
        return np.exp(out_vol)

    return out_vol


# @timer_func
def bias_jei_zoom(small_bias, factors, target_shape=None, grid_mode=True):
    # return bias_field
    Sx, Sy, Sz = small_bias.shape
    Bx, By, Bz = (
        target_shape
        if target_shape
        else np.multiply(small_bias.shape, factors).astype(int)
    )

    I = small_bias
    A = np.zeros((Bx, Sy, Sz))
    Xs = np.arange(Bx)

    interp_factor = (1 - factors[0]) / (2 * factors[0]) if grid_mode else 0
    Xs_prime = interp_factor + Xs / factors[0]

    f = np.maximum(0, np.floor(Xs_prime).astype(int))
    c = np.minimum(f + 1, Sx - 1)

    w2 = Xs_prime - f
    w1 = 1 - w2

    for i in range(Bx):
        k = w1[i] * I[f[i], :, :] + w2[i] * I[c[i], :, :]
        A[i, :, :] = k

    I = A
    A = np.zeros((Bx, By, Sz))
    Ys = np.arange(By)
    interp_factor = (1 - factors[1]) / (2 * factors[1]) if grid_mode else 0
    Ys_prime = interp_factor + Ys / factors[1]
    f = np.maximum(0, np.floor(Ys_prime).astype(int))
    c = np.minimum(f + 1, Sy - 1)

    w2 = Ys_prime - f
    w1 = 1 - w2

    for i in range(By):
        k = w1[i] * I[:, f[i], :] + w2[i] * I[:, c[i], :]
        A[:, i, :] = k

    I = A
    A = np.zeros((Bx, By, Bz))
    Zs = np.arange(Bz)

    interp_factor = (1 - factors[2]) / (2 * factors[2]) if grid_mode else 0
    Zs_prime = interp_factor + Zs / factors[2]

    f = np.maximum(0, np.floor(Zs_prime).astype(int))
    c = np.minimum(f + 1, Sz - 1)

    w2 = Zs_prime - f
    w1 = 1 - w2

    for i in range(Bz):
        k = w1[i] * I[:, :, f[i]] + w2[i] * I[:, :, c[i]]
        A[:, :, i] = k

    return np.exp(A)


# @timer_func
def bias_einsum_zoom(small_bias, factors, target_shape=None, grid_mode=True):
    Sx, Sy, Sz = small_bias.shape
    Bx, By, Bz = (
        target_shape
        if target_shape
        else np.multiply(small_bias.shape, factors).astype(int)
    )

    I = small_bias
    letter = {0: "i", 1: "j", 2: "k"}
    for idx, (big, small) in enumerate(zip((Bx, By, Bz), (Sx, Sy, Sz))):
        Xs = np.arange(big)
        interp_factor = (1 - factors[idx]) / (2 * factors[idx]) if grid_mode else 0
        Xs_prime = interp_factor + Xs / factors[idx]

        f = np.maximum(0, np.floor(Xs_prime).astype(int))
        c = np.minimum(f + 1, small - 1)

        w2 = Xs_prime - f
        w1 = 1 - w2

        summand1 = np.einsum(f"{letter[idx]}, ijk->ijk", w1, np.take(I, f, idx))
        summand2 = np.einsum(f"{letter[idx]}, ijk->ijk", w2, np.take(I, c, idx))
        I = summand1 + summand2

    return np.exp(I)


# @timer_func
def einsum_zoom(small_bias, factors, target_shape=None, grid_mode=True, flag="bias"):
    small_shape = small_bias.shape

    if len(small_shape) == 4:
        small_shape = small_shape[:-1]

    Bx, By, Bz = target_shape if target_shape else np.multiply(small_shape, factors)

    I = small_bias
    letter = {0: "i", 1: "j", 2: "k"}
    for idx, (big, small) in enumerate(zip((Bx, By, Bz), small_shape)):
        Xs = np.arange(big)

        interp_factor = (1 - factors[idx]) / (2 * factors[idx]) if grid_mode else 0
        Xs_prime = interp_factor + Xs / factors[idx]

        f = np.maximum(0, np.floor(Xs_prime).astype(int))
        c = np.minimum(f + 1, small - 1)

        w2 = Xs_prime - f
        w1 = 1 - w2

        if flag == "bias":
            ein_str = f"{letter[idx]}, ijk->ijk"
        else:
            ein_str = f"{letter[idx]}, ijkl->ijkl"

        summand1 = np.einsum(ein_str, w1, np.take(I, f, idx))
        summand2 = np.einsum(ein_str, w2, np.take(I, c, idx))
        I = summand1 + summand2

    if flag == "def":
        return I
    else:
        return np.exp(I)


# @timer_func
def prod_zoom(small_bias, factors, target_shape=None, grid_mode=True, flag="bias"):
    small_shape = small_bias.shape

    if len(small_shape) == 4:
        small_shape = small_shape[:-1]
        range_k = 4
    else:
        range_k = 3

    Bx, By, Bz = target_shape if target_shape else np.multiply(small_shape, factors)

    I = small_bias
    for idx, (big, small) in enumerate(zip((Bx, By, Bz), small_shape)):
        some_factor = small_shape[idx]
        Xs = np.arange(big)

        interp_factor = (1 - factors[idx]) / (2 * factors[idx]) if grid_mode else 0
        Xs_prime = interp_factor + Xs / factors[idx]

        f = np.maximum(0, np.floor(Xs_prime).astype(int))
        c = np.minimum(f + 1, small - 1)

        w2 = Xs_prime - f
        w1 = 1 - w2

        my_list = list(range(range_k))
        my_list.remove(idx)
        summand1 = np.expand_dims(w1, my_list) * np.take(I, f, idx)
        summand2 = np.expand_dims(w2, my_list) * np.take(I, c, idx)
        I = summand1 + summand2

    if flag == "def":
        return I
    else:
        return np.exp(I)


# @timer_func
def def_prod_zoom(small_bias, factors, target_shape=None, grid_mode=True, flag=None):
    Sx, Sy, Sz = small_bias.shape[:-1]
    Bx, By, Bz = (
        target_shape if target_shape else np.multiply(small_bias.shape[:-1], factors)
    )

    I = small_bias
    for idx, (big, small) in enumerate(zip((Bx, By, Bz), (Sx, Sy, Sz))):
        Xs = np.arange(big)

        interp_factor = (1 - factors[idx]) / (2 * factors[idx]) if grid_mode else 0
        Xs_prime = interp_factor + Xs / factors[idx]

        f = np.maximum(0, np.floor(Xs_prime).astype(int))
        c = np.minimum(f + 1, small - 1)

        w2 = Xs_prime - f
        w1 = 1 - w2

        my_list = list(range(4))
        my_list.remove(idx)
        summand1 = np.expand_dims(w1, my_list) * np.take(I, f, idx)
        summand2 = np.expand_dims(w2, my_list) * np.take(I, c, idx)
        I = summand1 + summand2

    return I


# @timer_func
def def_jei_zoom(small_def, factors, target_shape, grid_mode=True, flag="def"):
    # return bias_field
    Sx, Sy, Sz, dims = small_def.shape
    # dims will always be 3... if not, maybe throw an error?
    Bx, By, Bz = (
        target_shape
        if target_shape
        else np.multiply(small_bias.shape[:-1], factors).astype(int)
    )

    I = small_def
    A = np.zeros((Bx, Sy, Sz, dims))
    Xs = np.arange(Bx)

    interp_factor = (1 - factors[0]) / (2 * factors[0]) if grid_mode else 0
    Xs_prime = interp_factor + Xs / factors[0]

    f = np.maximum(0, np.floor(Xs_prime).astype(int))
    c = np.minimum(f + 1, Sx - 1)

    # w1 = c - Xs_prime
    w2 = Xs_prime - f
    w1 = 1 - w2

    for i in range(Bx):
        k = w1[i] * I[f[i], :, :, :] + w2[i] * I[c[i], :, :, :]
        A[i, :, :, :] = k

    I = A
    A = np.zeros((Bx, By, Sz, 3))
    Ys = np.arange(By)

    interp_factor = (1 - factors[1]) / (2 * factors[1]) if grid_mode else 0
    Ys_prime = interp_factor + Ys / factors[1]

    f = np.maximum(0, np.floor(Ys_prime).astype(int))
    c = np.minimum(f + 1, Sy - 1)

    # w1 = c - Xs_prime
    w2 = Ys_prime - f
    w1 = 1 - w2

    for i in range(By):
        k = w1[i] * I[:, f[i], :, :] + w2[i] * I[:, c[i], :, :]
        A[:, i, :, :] = k

    I = A
    A = np.zeros((Bx, By, Bz, 3))
    Zs = np.arange(Bz)

    interp_factor = (1 - factors[2]) / (2 * factors[2]) if grid_mode else 0
    Zs_prime = interp_factor + Zs / factors[2]

    f = np.maximum(0, np.floor(Zs_prime).astype(int))
    c = np.minimum(f + 1, Sz - 1)

    # w1 = c - Xs_prime
    w2 = Zs_prime - f
    w1 = 1 - w2

    for i in range(Bz):
        k = w1[i] * I[:, :, f[i], :] + w2[i] * I[:, :, c[i], :]
        A[:, :, i, :] = k

    return A


def print_absolute_diff_table(some_list, header):
    len_list = len(some_list)
    def_diff = np.zeros((len_list, len_list))

    for i, elem1 in enumerate(some_list):
        for j, elem2 in enumerate(some_list):
            def_diff[i, j] = np.sum(np.abs(elem1 - elem2))

    def_diff = zip(header, def_diff.tolist())
    def_diff = [[i, *j] for (i, j) in def_diff]

    print()
    print(tabulate(def_diff, headers=[" "] + header))
    print()


def save_images(some_list, idx, header, spacing, flag=None):
    img = nib.load("/usr/local/freesurfer/dev/subjects/bert/mri/nu.mgz")
    for item, hdr_tag in zip(some_list, header):
        img1 = nib.Nifti1Image(item, img.affine, img.header)
        file_name = f"{hdr_tag}_S{spacing:02}-R{idx:02}.nii.gz"
        nib.save(img1, os.path.join(os.getcwd(), "images", flag, file_name))


if __name__ == "__main__":
    header = ["scipy", "for_loop", "multiply", "einsum"]
    os.makedirs(os.path.join(os.getcwd(), "images", "bias"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "images", "deformation"), exist_ok=True)

    print(f"Labels Shape is: {LABELS_SHAPE}\n")

    for spacing in [2, 6, 8, 10, 12]:
        for test in range(1):
            print(f"Test Run: {test}")
            print(f"Spacing = {spacing}", end=" ")

            small_bias, bias_factors = generate_small_bias(spacing)

            # if test == 0:
            #     print_shapes(small_bias, bias_factors)

            bias_result_list = []
            print()
            for func in [scipy_zoom, bias_jei_zoom, einsum_zoom, prod_zoom]:
                result = func(small_bias, bias_factors, target_shape=LABELS_SHAPE)
                bias_result_list.append(result)

            if test == 0:
                print_absolute_diff_table(bias_result_list[1:], header)

            small_def = np.stack([small_bias, small_bias, small_bias], axis=-1)

            def_result_list = []
            print()
            for func in [
                scipy_zoom,
                def_jei_zoom,
                einsum_zoom,
                def_prod_zoom,
                prod_zoom,
            ]:
                result = func(
                    small_def, bias_factors, target_shape=LABELS_SHAPE, flag="def"
                )
                def_result_list.append(result)

            if test == 0:
                print_absolute_diff_table(def_result_list[1:], header)

            # save_images(bias_result_list, test, header, spacing, 'bias')
            # save_images(def_list, test, header, spacing, 'deformation')
