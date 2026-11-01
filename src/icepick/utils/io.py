'''
icepick: file i/o utilities
'''

# Import external dependencies
import mrcfile, numpy as np
from collections.abc import Callable
from pathlib import Path

# _CHUNK: input slices read at a time while binning
_CHUNK: int = 64

# mrc_info: (z, y, x) shape and voxel size in Angstrom from the MRC header
def mrc_info(path: Path) -> tuple[tuple[int, int, int], float]:
    with mrcfile.mmap(path, mode='r', permissive=True) as mrc:
        if mrc.data.ndim != 3:
            raise ValueError(f'expected a 3D volume in {path!r}, got shape {mrc.data.shape!r}')
        z, y, x = mrc.data.shape
        return (int(z), int(y), int(x)), round(float(mrc.voxel_size.x), 4)    # float32 header noise would move radii across voxel distances

# bin_volume: read an MRC in z chunks and reduce each factor-sized block, cropping to a multiple of the factor
def bin_volume(path: Path, factor: int, reduce: Callable[..., np.ndarray]) -> np.ndarray:
    with mrcfile.mmap(path, mode='r', permissive=True) as mrc:
        nz, ny, nx = (int(n) // factor * factor for n in mrc.data.shape)
        if min(nz, ny, nx) == 0:
            raise ValueError(f'volume {mrc.data.shape!r} in {path!r} is smaller than binning factor {factor}')
        out = np.empty((nz // factor, ny // factor, nx // factor), np.float32)
        step = factor * max(_CHUNK // factor, 1)    # chunks must hold whole blocks
        for z0 in range(0, nz, step):
            z1 = min(z0 + step, nz)
            slab = np.asarray(mrc.data[z0:z1, :ny, :nx], dtype=np.float32)
            blocks = slab.reshape((z1 - z0) // factor, factor, ny // factor, factor, nx // factor, factor)
            out[z0 // factor:z1 // factor] = reduce(blocks, axis=(1, 3, 5))
    return out
