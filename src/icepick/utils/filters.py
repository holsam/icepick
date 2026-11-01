'''
icepick: blob and matched filters
'''

# Import external dependencies
import numpy as np
from scipy.spatial import cKDTree

# Import internal icepick objects
from icepick.config import Grid
from icepick.utils.backend import asnumpy, ndi, xp

# blobs: membrane-aware multi-scale DoG candidates, rows (x, y, z, dog score) in working-grid voxels, centred on voxel centres
def blobs(x: np.ndarray, w: np.ndarray, grid: Grid) -> np.ndarray:
    found = []
    for sigma in grid.sigmas:
        r = dog(x, w, sigma, grid.min_weight)
        r[~(w > 0)] = 0
        v = r[w > 0]
        r /= 1.4826 * xp.median(xp.abs(v - xp.median(v)))    # robust z per scale
        size = int(round(3.0 * sigma)) | 1
        peaks = (r == ndi.maximum_filter(r, size=size)) & (r > grid.threshold)
        z, y, xx = (asnumpy(a) for a in xp.nonzero(peaks))
        if not len(z):
            continue
        rows = np.column_stack([asnumpy(r[peaks]), xx + 0.5, y + 0.5, z + 0.5, np.full(len(z), sigma)])
        found.append(_suppress(rows, grid.suppress))    # only within a scale, across scales small speckle peaks delete particle-scale ones
    if not found:
        return np.zeros((0, 4))
    c = np.vstack(found)
    return np.column_stack([c[:, 1:4], c[:, 0]])

# dog: difference of normalised-convolution gaussian means, ignoring voxels where w is zero
def dog(x: np.ndarray, w: np.ndarray, s: float, min_weight: float) -> np.ndarray:
    a, wa = nconv(x, w, s)
    b, _ = nconv(x, w, 1.6 * s)
    r = a - b
    r[wa < min_weight] = 0    # centre mostly membrane, response unreliable
    return r

# nconv: gaussian mean of x over voxels with weight w, plus the retained weight
def nconv(x: np.ndarray, w: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    num = ndi.gaussian_filter(x * w, s, truncate=3.0)
    den = ndi.gaussian_filter(w, s, truncate=3.0)
    return num / xp.maximum(den, 1e-3), den

# _suppress: greedy removal of candidates within slope * sigma + offset of a better one, rows (score, x, y, z, sigma)
def _suppress(c: np.ndarray, radius: tuple[float, float]) -> np.ndarray:
    c = c[np.argsort(-c[:, 0])]
    tree = cKDTree(c[:, 1:4])
    gone = np.zeros(len(c), bool)
    keep = []
    for i in range(len(c)):
        if gone[i]:
            continue
        keep.append(i)
        gone[tree.query_ball_point(c[i, 1:4], radius[0] * c[i, 4] + radius[1])] = True
    return c[keep]
