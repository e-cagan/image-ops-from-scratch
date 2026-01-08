"""
Module for applying gaussian blur to image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from grayscale import to_grayscale
from utils import make_gaussian_kernel_1d, convolve1d_axis


def gaussian_blur(img, k_size, sigma):
    """
    Function that applies gaussian blur to an image

    img -> An array of pixels

    k_size -> kernel size

    sigma -> blur coefficient

    Using seperable gaussian blur approach which is multiplying 1D gaussian kernels for horizontal and vertical
    """

    # Use predefined util function for creating gaussian kernel
    kernel = make_gaussian_kernel_1d(k_size, sigma)

    # Apply convolution on height
    conv_height = convolve1d_axis(img=img, kernel_1d=kernel, axis=0, pad_mode="reflect")

    # Apply convolution on width
    blurred = convolve1d_axis(img=conv_height, kernel_1d=kernel, axis=1, pad_mode="reflect")

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