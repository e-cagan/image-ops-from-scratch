"""
Utility functions for filters
"""

import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_gaussian_kernel_1d(k_size, sigma):
    """
    Creates a 1D kernel for convolution for gaussian blur.

    k_size -> Kernel size (Should be a positive and odd number.)

    sigma -> Gaussian coefficient (should be a positive number and greater than zero.)
    """

    # Check the kernel size is odd or even
    if k_size % 2 == 0:
        print("Please choose an odd number for kernel size.")
        sys.exit()
    
    # Check the sigma is greater than zero
    if sigma <= 0:
        print("Please choose a number greater than zero for sigma.")
        sys.exit()

    # Create coordinates
    m = k_size // 2 # -> Filling the kernel based on k_size modulus
    coords_l = [0]

    # Add values to coordinates
    for i in range(k_size):
        if i == 0:
            continue

        if np.abs(i) <= m:
            coords_l.append(i)
            coords_l.append(-i)
    
    # Sort the coords list
    coords_l = sorted(coords_l)

    # Convert it to a numpy array
    coords = np.array(coords_l)

    # DEBUG
    print(f"M: {m}")
    print(f"Coordinates: {coords}")

    # Calculate gaussian weights then normalize them
    g_x = np.exp(-((coords**2) / (2 * sigma**2)))
    g_x = g_x / np.sum(g_x)

    # Cast g_x's type to float
    g_x = g_x.astype(np.float32)

    # DEBUG
    print(f"G_x: {g_x}")

    return g_x


def convolve1d_axis(img, kernel_1d, axis, pad_mode="reflect"):
    """
    Applies convolution on 1D to image.

    img -> Image

    kernel_1d -> 1D kernel

    axis -> A parameter for deciding the axis for padding (0 for height, 1 for width)

    pad_mode -> Choosing the padding mode. (only reflect for now.)
    """

    # Work in float for convolution
    img_f = img.astype(np.float32)

    # Take the modulus of kernel length
    m = len(kernel_1d) // 2

    # Initialize the variables before using
    img_pad = None
    windows = None
    convolved = None

    # Apply convolution (1 for width, 0 for height)
    if axis == 1:
        img_pad = np.pad(img_f, pad_width=((0, 0), (m, m)), mode=pad_mode)
        windows = sliding_window_view(img_pad, window_shape=len(kernel_1d), axis=axis)
        convolved = np.sum(windows * kernel_1d, axis=-1)
    elif axis == 0:
        img_pad = np.pad(img_f, pad_width=((m, m), (0, 0)), mode=pad_mode)
        windows = sliding_window_view(img_pad, window_shape=len(kernel_1d), axis=axis)
        convolved = np.sum(windows * kernel_1d, axis=-1)
    else:
        return "Unsupported axis type!"

    return convolved
