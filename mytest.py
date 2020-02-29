import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


kernel = np.ones((5,6))
kernel.shape
kernel.ndim

