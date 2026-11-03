'''
icepick: tests for the cytosolic stream
'''

# Import external dependencies
import numpy as np
from scipy import ndimage as ndi

# Import internal icepick objects
from icepick import cytosolic
from icepick.config import Params
from icepick.utils.volume import Volume

# TestCytosolic: checks for cytosolic.py
class TestCytosolic:
    # test_pick_finds_particles_of_different_sizes_and_avoids_membrane: three planted particles are all picked, nothing is picked on the membrane slab
    def test_pick_finds_particles_of_different_sizes_and_avoids_membrane(self) -> None:
        rng = np.random.default_rng(0)
        x = ndi.gaussian_filter(rng.standard_normal((64, 96, 128)), 1.0).astype(np.float32)
        x /= x.std()
        zz, yy, xx = np.indices(x.shape)
        planted = [((32, 30, 30), 4.0), ((32, 60, 60), 5.5), ((32, 40, 100), 7.0)]    # centre (z, y, x), radius
        for (cz, cy, cx), r in planted:
            x[np.sqrt(((zz - cz) / 1.5) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) < r] += 1.5
        m = np.zeros(x.shape, bool)
        m[:, 75:78, :] = True
        x[m] += 4.0
        grid = Params().grid(15.68)    # factor 1
        near = ndi.binary_dilation(m, iterations=grid.dilate)
        volume = Volume(x, m, np.zeros_like(m), (~near).astype(np.float32), (0, 64), grid)
        c = cytosolic.pick(volume)
        assert c.shape[1] == 5 and (c[:, 4] == 0).all()
        for (cz, cy, cx), _ in planted:
            assert np.min(np.linalg.norm(c[:, :3] - [cx + 0.5, cy + 0.5, cz + 0.5], axis=1)) <= 4
        assert not near[np.floor(c[:, 2]).astype(int), np.floor(c[:, 1]).astype(int), np.floor(c[:, 0]).astype(int)].any()

    # test_pick_returns_empty_array_without_candidates: pure noise below threshold gives a well-shaped empty result
    def test_pick_returns_empty_array_without_candidates(self) -> None:
        grid = Params(threshold=1e6).grid(15.68)
        x = np.random.default_rng(0).standard_normal((40, 40, 40)).astype(np.float32)
        m = np.zeros(x.shape, bool)
        volume = Volume(x, m, m, np.ones_like(x), (0, 40), grid)
        assert cytosolic.pick(volume).shape == (0, 5)
