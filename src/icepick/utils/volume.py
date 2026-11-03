'''
icepick: working volume
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import internal icepick objects
from icepick.config import Grid, Params
from icepick.utils.backend import asdevice, asnumpy, ndi, xp
from icepick.utils.io import bin_volume, mrc_info

# Volume: one tomogram on the working grid, shared by both streams
@dataclass(frozen=True)
class Volume:
    x: np.ndarray    # inverted z-scored density, protein positive
    membrane: np.ndarray    # raw membrane mask
    artefact: np.ndarray    # dilated bright-artefact mask
    weight: np.ndarray    # float32, 0 on dilated membrane and artefacts
    lamella: tuple[int, int]    # working-grid slice range of the specimen
    grid: Grid

# load: bin a density and its binary membrane segmentation onto the working grid and build the volume
def load(density: Path, segmentation: Path, params: Params) -> Volume:
    shape, voxel_size = mrc_info(density)
    seg_shape, _ = mrc_info(segmentation)
    if seg_shape != shape:
        raise ValueError(f'segmentation shape {seg_shape!r} differs from density shape {shape!r}')
    if voxel_size <= 0:
        raise ValueError(f'no voxel size in the header of {density!r}')
    grid = params.grid(voxel_size)
    d = asdevice(bin_volume(density, grid.factor, np.mean))
    m = asdevice(bin_volume(segmentation, grid.factor, np.max)) > 0
    near = ndi.binary_dilation(m, iterations=grid.dilate)
    keep = ~near
    x = -(d - d[keep].mean()) / d[keep].std()    # protein is dark, flip so blobs are positive
    v = x[keep]
    med = xp.median(v)
    bright = x < med - grid.artefact_sd * 1.4826 * xp.median(xp.abs(v - med))
    artefact = ndi.binary_dilation(bright, iterations=grid.artefact_dilate)
    weight = (~(near | artefact)).astype(xp.float32)
    return Volume(x, m, artefact, weight, sample_z(x), grid)

# sample_z: working-grid slice range of the specimen, between the structure minima either side of the structure peak
def sample_z(x: np.ndarray) -> tuple[int, int]:
    # per-slice spread of band-passed density peaks inside the lamella and rises again at the tomogram ends (junk)
    band = ndi.gaussian_filter(x, 1.0) - ndi.gaussian_filter(x, 4.0)    # working-voxel scales, which follow the particle
    profile = asnumpy(ndi.gaussian_filter1d(band.std(axis=(1, 2)), 3.0))
    top = int(np.argmax(profile))
    return int(np.argmin(profile[:top + 1])), top + int(np.argmin(profile[top:])) + 1
