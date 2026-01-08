"""
Module for applying Sobel filter to images. Implemented with NumPy from scratch
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from grayscale import to_grayscale
from gaussian_blur import gaussian_blur