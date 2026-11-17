'''
icepick: cytosolic particle stream
'''

# Import external dependencies
import numpy as np
from loguru import logger
from scipy.stats import rankdata

# Import internal icepick objects
from icepick.utils import filters
from icepick.utils.volume import Volume

# pick: candidate array (x, y, z, score, stream) in input voxels for one tomogram, score is a mean rank in (0, 1]
def pick(volume: Volume, *, stream: int = 0) -> np.ndarray:
    grid = volume.grid
    lo, hi = volume.slices
    c = filters.blobs(volume.x, volume.weight, grid)
    n = len(c)
    c = c[(c[:, 2] >= lo) & (c[:, 2] < hi)]    # outside the volume is junk
    logger.debug(f'{n} blobs above threshold, {len(c)} inside the volume')
    if not len(c):
        return np.zeros((0, 5))
    mf = filters.rescore(c, filters.matched(volume.x, volume.weight, grid))
    # the two scores err differently, so their mean rank beats either alone
    c[:, 3] = (rankdata(c[:, 3]) + rankdata(mf)) / (2 * len(c))
    n = len(c)
    c = filters.thin(c, grid.thin)
    logger.debug(f'thinned {n} rescored candidates to {len(c)} at {grid.thin:.1f} voxels')
    c[:, :3] *= grid.factor
    return np.column_stack([c, np.full(len(c), stream)])
