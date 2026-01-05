"""
Module for applying gaussian blur to image
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

from grayscale import to_grayscale


def gaussian_blur(img, k_size, sigma):
    """
    Function that applies gaussian blur to an image

    img -> An array of pixels

    k_size -> kernel size

    sigma -> blur coefficient

    Using seperable gaussian blur approach which is multiplying 1D gaussian kernels for horizontal and vertical
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

    # DEBUG
    print(f"G_x: {g_x}")

    # Work in float for convolution
    img_f = img.astype(np.float32)
    g_x = g_x.astype(np.float32)

    # Horizontal blur: pad only width (left-right)
    img_hpad = np.pad(img_f, pad_width=((0, 0), (m, m)), mode="reflect")
    windows_h = sliding_window_view(img_hpad, window_shape=k_size, axis=1)  # (H, W, k)
    weighted_h = np.sum(windows_h * g_x, axis=-1)  # (H, W)

    # Vertical blur: pad only height (top-bottom)
    img_vpad = np.pad(weighted_h, pad_width=((m, m), (0, 0)), mode="reflect")
    windows_v = sliding_window_view(img_vpad, window_shape=k_size, axis=0)  # (H, W, k)
    blurred = np.sum(windows_v * g_x, axis=-1)  # (H, W)

    # Debug shapes
    print(f"IMG_HPAD SHAPE: {img_hpad.shape}")       # (H, W+2m)
    print(f"WINDOWS_H SHAPE: {windows_h.shape}")     # (H, W, k)
    print(f"WEIGHTED_H SHAPE: {weighted_h.shape}")   # (H, W)
    print(f"IMG_VPAD SHAPE: {img_vpad.shape}")       # (H+2m, W)
    print(f"WINDOWS_V SHAPE: {windows_v.shape}")     # (H, W, k)
    print(f"BLURRED SHAPE: {blurred.shape}")         # (H, W)

    # Convert back to uint8 for display/save
    blurred = np.clip(blurred, 0, 255)
    blurred = np.round(blurred).astype(np.uint8)

    return blurred


def main():
    """
    Function for testing the gaussian blur function.
    """

    # Defining the filename
    FILENAME = "front.jpeg"

    # Read the image and convert it to np array
    img = Image.open(f"data/inputs/{FILENAME}")
    img = np.array(img)

    # Apply the implemented grayscaling function
    gray = to_grayscale(img)

    # Then apply gaussian blur to the grayscaled image
    blurred = gaussian_blur(gray, 5, 1)

    # Show and save the image
    plt.title('Blurred Image')
    plt.imshow(blurred, cmap="gray")
    plt.imsave(f"data/outputs/blurred_{FILENAME}", blurred, cmap="gray")
    print("Image saved successfully to output dir.")
    

if __name__ == '__main__':
    main()