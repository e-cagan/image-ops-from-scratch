"""
Module for grayscaling a RGB image with NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def to_grayscale(img):
    """
    Function that grayscales the image.
    """

    # DEBUG
    print(f"Normal Image Shape: {img.shape}")
    
    # Seperating channels
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Multiplying with channel weights then converting to 8 bit unsigned integer. (Related to human eye)
    gray = (R * 0.299) + (G * 0.587) + (B * 0.114)
    gray = np.round(gray).astype(np.uint8)

    # DEBUG
    print(f"Grayscaled Image Shape: {gray.shape}")

    return gray


def main():
    """
    Testing the grayscale function
    """

    # Defining the filename
    FILENAME = "front.jpeg"

    # Read the image and convert it to np array
    img = Image.open(f"data/inputs/{FILENAME}")
    img = np.array(img)

    # Apply the implemented grayscaling function
    gray = to_grayscale(img)

    # Show and save the image
    plt.title("Grayscaled Image")
    plt.imshow(gray, cmap="gray")
    plt.imsave(f"data/outputs/grayscaled_{FILENAME}", gray, cmap="gray")
    print("Image saved successfully to output dir.")


if __name__ == '__main__':
    main()