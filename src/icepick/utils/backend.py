'''
icepick: array backend
'''

# Import external dependencies
import numpy as np
from scipy import fft
from scipy import ndimage as ndi

# xp: array namespace for volume-sized work (for gpu, xp=cupy, ndi to cupyx.scipy.ndimage)
xp = np

# asdevice: move a host array to the backend (a no-op on the CPU)
def asdevice(a: np.ndarray) -> np.ndarray:
    return a

# asnumpy: move a backend array to the host, for point lists, KD-trees, ranking and file output
def asnumpy(a: np.ndarray) -> np.ndarray:
    return a

# rfftn: real 3D FFT using all cores
def rfftn(a: np.ndarray) -> np.ndarray:
    return fft.rfftn(a, workers=-1)

# irfftn: inverse real 3D FFT to the given shape using all cores
def irfftn(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return fft.irfftn(a, shape, workers=-1)
