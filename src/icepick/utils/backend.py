'''
icepick: array backend
'''

# Import external dependencies
import numpy as np
from scipy import fft
from scipy import ndimage as ndi_cpu

# _Namespace: forwards attribute access to a module that can be swapped, so names imported once follow the active backend
class _Namespace:
    # __init__: start on the given module
    def __init__(self, module) -> None:
        self._module = module

    # __getattr__: only called for names not on the forwarder itself
    def __getattr__(self, name: str):
        return getattr(self._module, name)

    # use: point the forwarder at another module
    def use(self, module) -> None:
        self._module = module

# xp: array namespace for volume-sized work (numpy, or cupy after use_gpu)
xp = _Namespace(np)

# ndi: ndimage namespace (scipy.ndimage, or cupyx.scipy.ndimage after use_gpu)
ndi = _Namespace(ndi_cpu)

# _cupy: the cupy module while the GPU backend is active, otherwise None
_cupy = None

# use_gpu: switch xp, ndi and the FFTs to cupy (call icepick.utils.gpu.require_gpu first)
def use_gpu() -> None:
    global _cupy
    import cupy, cupyx.scipy.ndimage as cupy_ndi    # imported here so the CPU path never needs cupy
    xp.use(cupy)
    ndi.use(cupy_ndi)
    _cupy = cupy

# use_cpu: switch back to numpy and scipy
def use_cpu() -> None:
    global _cupy
    xp.use(np)
    ndi.use(ndi_cpu)
    _cupy = None

# asdevice: move a host array to the backend (a no-op on the CPU)
def asdevice(a: np.ndarray) -> np.ndarray:
    return a if _cupy is None else _cupy.asarray(a)

# asnumpy: move a backend array to the host, for point lists, KD-trees, ranking and file output
def asnumpy(a: np.ndarray) -> np.ndarray:
    return a if _cupy is None else _cupy.asnumpy(a)

# release: return cached GPU memory blocks to the driver between large tiles (a no-op on the CPU)
def release() -> None:
    if _cupy is not None:
        _cupy.get_default_memory_pool().free_all_blocks()

# rfftn: real 3D FFT using all cores, or on the GPU
def rfftn(a: np.ndarray) -> np.ndarray:
    return fft.rfftn(a, workers=-1) if _cupy is None else _cupy.fft.rfftn(a)

# irfftn: inverse real 3D FFT to the given shape using all cores, or on the GPU
def irfftn(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return fft.irfftn(a, shape, workers=-1) if _cupy is None else _cupy.fft.irfftn(a, shape)
