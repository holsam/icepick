'''
icepick: stream merging and tomogram picking
'''

# Import external dependencies
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

# Import internal icepick objects
from icepick import cytosolic, membrane
from icepick.config import Params
from icepick.utils import volume

# pick_tomogram: merged candidate array (x, y, z, score, stream) in input voxels, score is the within-stream rank in (0, 1]
def pick_tomogram(density: Path, segmentation: Path, params: Params) -> np.ndarray:
    vol = volume.load(density, segmentation, params)
    return merge([cytosolic.pick(vol, stream=0), membrane.pick(vol, stream=1)], vol.grid.duplicate * vol.grid.factor)

# merge: pool per-stream candidates, put scores on a common rank scale, drop cross-stream duplicates within radius (input voxels)
def merge(streams: list[np.ndarray], radius: float) -> np.ndarray:
    ranked = []
    for c in streams:
        if len(c):
            c = c.copy()
            c[:, 3] = rankdata(c[:, 3]) / len(c)    # stream scores are not comparable, ranks are
            ranked.append(c)
    if not ranked:
        return np.zeros((0, 5))
    c = np.vstack(ranked)
    c = c[np.argsort(-c[:, 3])]
    tree = cKDTree(c[:, :3])
    gone = np.zeros(len(c), bool)
    for i in range(len(c)):
        if gone[i]:
            continue
        near = np.array(tree.query_ball_point(c[i, :3], radius), int)
        gone[near[(c[near, 4] != c[i, 4]) & (near > i)]] = True    # only across streams, within-stream thinning is each stream's job
    return c[~gone]
