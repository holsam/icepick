'''
icepick: blob and matched filters
'''

# Import external dependencies
import numpy as np
from scipy.spatial import cKDTree

# Import internal icepick objects
from icepick.config import Grid
from icepick.utils.backend import asdevice, asnumpy, irfftn, ndi, release, rfftn, xp

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

# matched: matched-filter score map over overlapping z tiles, each whitened and normalised on its own
def matched(x: np.ndarray, w: np.ndarray, grid: Grid) -> np.ndarray:
    if min(x.shape) < 2 * grid.box + 1:
        raise ValueError(f'volume {x.shape!r} on the working grid is smaller than the template box {2 * grid.box + 1}')
    filled = xp.where(w > 0, x, nconv(x, w, grid.fill)[0])
    n, margin, depth = grid.tile, grid.box, x.shape[0]
    # half-overlapping tiles cross-faded with triangular weights, so per-tile normalisation leaves no seams
    tri = asdevice((1 - np.abs(np.linspace(-1, 1, n + 2)[1:-1])).astype(np.float32)[:, None, None])
    acc = xp.zeros(x.shape, xp.float32)
    total = xp.zeros((depth, 1, 1), xp.float32)
    kernels = [asdevice(kernel(r, grid)) for r in grid.radii]
    for c0 in range(-(n // 2), depth, n // 2):
        a, b = max(c0, 0), min(c0 + n, depth)
        if a >= b:
            continue
        # tiles at the ends reach inwards to keep full length
        lo = max(a - margin, 0)
        hi = min(max(b + margin, lo + n + 2 * margin), depth)
        lo = max(min(lo, hi - n - 2 * margin), 0)
        s = _matched_tile(filled[lo:hi], w[lo:hi], kernels, grid.box)
        acc[a:b] += tri[a - c0:b - c0] * s[a - lo:b - lo]
        total[a:b] += tri[a - c0:b - c0]
        release()
    return acc / total

# kernel: zero-mean soft ball of radius r (working voxels) with a negative shell, stretched along z
def kernel(r: float, grid: Grid) -> np.ndarray:
    b = grid.box
    zz, yy, xx = np.indices((2 * b + 1,) * 3) - b
    d = np.sqrt((zz / grid.stretch) ** 2 + yy ** 2 + xx ** 2)
    core = 1 / (1 + np.exp((d - r) / 0.7))
    ring = 1 / (1 + np.exp((d - r - grid.shell[0]) / 0.7)) - core
    k = core - grid.shell[1] * ring * core.sum() / ring.sum()
    k *= np.exp(-(np.sqrt(zz ** 2 + yy ** 2 + xx ** 2) / b) ** 4)    # soft support inside the box
    return (k - k.mean()).astype(np.float32)

# rescore: matched-filter score near each candidate (rows x, y, z in working-grid voxels)
def rescore(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    near = ndi.maximum_filter(s, size=3)    # blob centres sit a voxel or so off the filter peak
    i = np.clip(np.floor(c[:, [2, 1, 0]]).astype(int), 0, np.array(s.shape) - 1)
    return asnumpy(near[tuple(asdevice(a) for a in i.T)])

# thin: greedy removal of candidates within radius of a better one, rows (x, y, z, score)
def thin(c: np.ndarray, radius: float) -> np.ndarray:
    c = c[np.argsort(-c[:, 3])]
    tree = cKDTree(c[:, :3])
    gone = np.zeros(len(c), bool)
    keep = []
    for i in range(len(c)):
        if gone[i]:
            continue
        keep.append(i)
        gone[tree.query_ball_point(c[i, :3], radius)] = True
    return c[keep]

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

# _matched_tile: whitened matched-filter score (robust z, max over template radii) for one membrane-filled tile
def _matched_tile(filled: np.ndarray, w: np.ndarray, kernels: list[np.ndarray], box: int) -> np.ndarray:
    spec = rfftn(filled)
    # dividing by the smoothed periodogram whitens the correlated background, the key step
    spec /= ndi.gaussian_filter(xp.abs(spec) ** 2, 2.0)
    valid = w > 0
    best = xp.full(filled.shape, -xp.inf, xp.float32)
    for kern in kernels:
        k = xp.zeros(filled.shape, xp.float32)
        k[:2 * box + 1, :2 * box + 1, :2 * box + 1] = kern
        k = xp.roll(k, (-box, -box, -box), (0, 1, 2))
        resp = irfftn(spec * xp.conj(rfftn(k)), filled.shape)
        v = resp[valid]
        xp.maximum(best, (resp - xp.median(v)) / v.std(), out=best)
    return best
