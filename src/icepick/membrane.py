'''
icepick: membrane particle stream
'''

# Import external dependencies
import numpy as np

# Import internal icepick objects
from icepick.config import Grid
from icepick.utils.backend import asnumpy, ndi, xp
from icepick.utils.volume import Volume

# pick: candidate array (x, y, z, score, stream) in input voxels for one tomogram, score is robust z within its shell
def pick(volume: Volume, *, stream: int = 1) -> np.ndarray:
    grid = volume.grid
    lo, hi = volume.lamella
    x, m, artefact = volume.x[lo:hi], volume.membrane[lo:hi], volume.artefact[lo:hi]
    shell = _shell(m, grid)
    s = _excess(x, shell, grid)
    pk = (s == ndi.maximum_filter(s, size=grid.peak)) & (shell <= grid.band) & ~artefact
    idx = np.array([asnumpy(a) for a in xp.nonzero(pk)]).T
    zyx = (idx + [lo, 0, 0] + 0.5) * grid.factor
    return np.column_stack([zyx[:, ::-1], asnumpy(s[pk]), np.full(len(idx), stream)])

# _excess: smoothed residual over the robust local bare profile of each distance shell, robust z per shell
def _excess(x: np.ndarray, shell: np.ndarray, grid: Grid) -> np.ndarray:
    r = xp.zeros_like(x)
    for k in range(grid.band + 1):
        sel = shell == k
        mk = sel.astype(xp.float32)
        res = x - _shell_mean(x, mk, grid.along)
        mk[sel & (res >= grid.refit * res[sel].std())] = 0    # likely particles, left out of the refit
        r[sel] = (x - _shell_mean(x, mk, grid.along))[sel]
    r = ndi.gaussian_filter(r, grid.smooth)
    out = xp.full(x.shape, -xp.inf, xp.float32)
    for k in range(grid.band + 1):
        sel = shell == k
        v = r[sel]
        med = xp.median(v)
        out[sel] = (v - med) / (1.4826 * xp.median(xp.abs(v - med)) + 1e-6)
    return out

# _shell_mean: gaussian mean of x over the voxels of one shell (normalised convolution)
def _shell_mean(x: np.ndarray, mk: np.ndarray, s: float) -> np.ndarray:
    return ndi.gaussian_filter(x * mk, s) / xp.maximum(ndi.gaussian_filter(mk, s), 1e-3)

# _shell: distance shells (clamped at band + 1) from per-slab edts, each slab padded so clamped values are exact
def _shell(m: np.ndarray, grid: Grid, *, slab: int = 16) -> np.ndarray:
    halo = grid.band + 2    # any membrane voxel that can round to <= band + 1 lies within this many slices
    shell = xp.empty(m.shape, xp.int16)
    for a in range(0, len(m), slab):
        b = min(a + slab, len(m))
        lo, hi = max(a - halo, 0), min(b + halo, len(m))
        sub = m[lo:hi]
        if sub.any():    # edt of a membrane-free block is undefined
            d = ndi.distance_transform_edt(~sub)[a - lo:b - lo]
            shell[a:b] = xp.minimum(xp.round(d), grid.band + 1).astype(xp.int16)
        else:
            shell[a:b] = grid.band + 1
    return shell
