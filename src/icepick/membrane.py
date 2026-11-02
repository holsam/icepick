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
    dist = ndi.distance_transform_edt(~m)
    shell = xp.minimum(xp.round(dist), grid.band + 1).astype(xp.int16)
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
