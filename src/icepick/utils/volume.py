'''
icepick: working volume
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

# Import internal icepick objects
from icepick.config import Grid, Params
from icepick.utils import log
from icepick.utils.backend import asdevice, asnumpy, ndi, xp
from icepick.utils.io import bin_volume, mrc_info

# Volume: one tomogram on the working grid, shared by both streams
@dataclass(frozen=True)
class Volume:
    x: np.ndarray    # inverted z-scored density, protein positive
    membrane: np.ndarray    # raw membrane mask
    artefact: np.ndarray    # dilated bright-artefact mask
    weight: np.ndarray    # float32, 0 on dilated membrane and artefacts
    slices: tuple[int, int]    # working-grid slice range of the specimen
    grid: Grid

# load: bin a density and its binary membrane segmentation onto the working grid and build the volume
def load(density: Path, segmentation: Path, params: Params) -> Volume:
    shape, voxel_size = mrc_info(density)
    seg_shape, _ = mrc_info(segmentation)
    if seg_shape != shape:
        log.error_exit(f'segmentation shape {seg_shape!r} differs from density shape {shape!r}', ValueError)
    if voxel_size <= 0:
        log.error_exit(f'no voxel size in the header of {density!r}', ValueError)
    grid = params.grid(voxel_size)
    log.progress(f'file parameters: shape {shape!r}, voxel size {voxel_size} A')
    log.progress(f'binning {grid.factor}x to {grid.voxel_size:.2f} A per working voxel')
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
    sample = sample_z(x)
    logger.debug(f'using working slices: {sample[0]} - {sample[1]} (of {len(x)})')
    logger.debug(f'working slices contain: {int(m.sum())} membrane voxels, {int(artefact.sum())} artefact voxels')
    return Volume(x, m, artefact, weight, sample, grid)

# sample_z: working-grid slice range of the specimen, between the structure minima either side of the structure peak
def sample_z(x: np.ndarray, *, slab: int = 32) -> tuple[int, int]:
    profile = slice_profile(x, slab=slab)
    top = int(np.argmax(profile))
    return int(np.argmin(profile[:top + 1])), top + int(np.argmin(profile[top:])) + 1

# slice_profile: smoothed per-slice spread of band-passed density along z
def slice_profile(x: np.ndarray, *, slab: int = 32) -> np.ndarray:
    # per-slice spread of band-passed density peaks inside the sample and rises again at the tomogram ends (junk)
    halo = 16    # gaussian radius at sigma 4.0 with the default truncate of 4.0, so slab interiors match the full volume
    spread = []
    for a in range(0, len(x), slab):
        b = min(a + slab, len(x))
        lo, hi = max(a - halo, 0), min(b + halo, len(x))
        band = ndi.gaussian_filter(x[lo:hi], 1.0) - ndi.gaussian_filter(x[lo:hi], 4.0)    # working-voxel scales, which follow the particle
        spread.append(band[a - lo:b - lo].std(axis=(1, 2)))
    return asnumpy(ndi.gaussian_filter1d(xp.concatenate(spread), 3.0))
