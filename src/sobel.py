"""
Module for applying Sobel filter to images. Implemented with NumPy from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from grayscale import to_grayscale
from gaussian_blur import gaussian_blur
from utils import convolve1d_axis


def sobel(img):
    """
    Function that applies sobel filter to an image.

    img -> Image

    Note: Kernel size is constant which is 3x3

    Also it should return magnitude for minimum. Other params are for debugging.
    """

    # Create constant vectors (kernels) to operate
    DERIVATIVE_VEC = np.array([-1, 0, 1])
    SMOOTHING_VEC = np.array([1, 2, 1])

    # Cast to float type for image
    img = img.astype(np.float32)

    # First, smooth on height and derive on width (Gx) 
    smooth_h = convolve1d_axis(img=img, kernel_1d=SMOOTHING_VEC, axis=0, pad_mode="reflect")
    Gx = convolve1d_axis(img=smooth_h, kernel_1d=DERIVATIVE_VEC, axis=1, pad_mode="reflect")

    # Second, smooth on width and derive on height (Gy) 
    smooth_w = convolve1d_axis(img=img, kernel_1d=SMOOTHING_VEC, axis=1, pad_mode="reflect")
    Gy = convolve1d_axis(img=smooth_w, kernel_1d=DERIVATIVE_VEC, axis=0, pad_mode="reflect")

    # Calculate and normalize magnitude for edge output
    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = magnitude / magnitude.max()

    # Calculate direction for edge direction/angle
    theta = np.arctan2(Gy, Gx)

    # Returning magnitude for now
    return magnitude

    # FOR DEBUG
    # return magnitude, theta, Gx, Gy


def main():
    """
    Function for testing the sobel filter.
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

    # Then apply sobel filter on blurred image
    edge_sobel = sobel(img=blurred)

    # Show and save the image
    plt.title('Sobel Image')
    plt.imshow(edge_sobel, cmap="gray")
    plt.imsave(f"data/outputs/sobel_{FILENAME}", edge_sobel, cmap="gray")
    print("Image saved successfully to output dir.")

if __name__ == '__main__':
    main()