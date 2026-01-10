"""
Module for applying histogram equalization
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from grayscale import to_grayscale


def hist_eq(img):
    """
    Function that equalizes histogram on image. (Recommended grayscaled image.)

    img -> Image (assuming image type is uint)
    """

    # Create the histogram with 256 bins
    hist, bins = np.histogram(img.flatten(), bins=256, density=True) # Probability density function
    
    # Take the cumulative distribution function (CDF) and normalize it
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    # Use linear interpolation to find new pixel values
    img_hist_equalized = np.interp(img.flatten(), bins[:-1], cdf)

    # Reshape back to original image size
    return img_hist_equalized.reshape(img.shape), cdf
    

def main():
    """
    Testing the histogram equalization function
    """

    # Define directory paths
    INPUT_DIR = "data/inputs/"
    OUTPUT_DIR = "data/outputs/"

    # Define a dict for image path prefixes
    images = {
        "low": "low_cont.jpeg",
        "high": "high_cont.jpeg"
    }

    # Iterate trough files and save their figures
    for tag, filename in images.items():

        # Load & grayscale + blur to reduce channel size and noise
        img = np.array(Image.open(INPUT_DIR + filename))
        gray = to_grayscale(img)

        # Histogram Equalization
        heq, cdf = hist_eq(gray)
        heq = np.clip(heq, 0, 255).astype(np.uint8)
        
        # Save equalized image
        plt.figure()
        plt.imshow(heq, cmap="gray")
        plt.axis("off")
        plt.savefig(f"{OUTPUT_DIR}{tag}_heq.png", bbox_inches="tight", dpi=200)
        plt.close()

        # Grayscale histogram
        plt.figure()
        plt.hist(gray.flatten(), bins=256, range=(0, 255), density=True)
        plt.title(f"{tag.upper()} Grayscale Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.savefig(f"{OUTPUT_DIR}{tag}_hist_gray.png", dpi=200)
        plt.close()

        # Equalized histogram
        plt.figure()
        plt.hist(heq.flatten(), bins=256, range=(0, 255), density=True)
        plt.title(f"{tag.upper()} Equalized Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.savefig(f"{OUTPUT_DIR}{tag}_hist_heq.png", dpi=200)
        plt.close()

        # CDF plot        
        plt.figure()
        plt.plot(cdf)
        plt.xlim(0, 255)
        plt.ylim(0, 255)
        plt.title(f"{tag.upper()} CDF")
        plt.xlabel("Intensity")
        plt.ylabel("Mapped Intensity")
        plt.savefig(f"{OUTPUT_DIR}{tag}_cdf.png", dpi=200)
        plt.close()


if __name__ == '__main__':
    main()