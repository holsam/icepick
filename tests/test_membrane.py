'''
icepick: tests for the membrane stream
'''

# Import external dependencies
import numpy as np
from scipy import ndimage as ndi

# Import internal icepick objects
from icepick import membrane
from icepick.config import Params
from icepick.utils.volume import Volume

# TestMembrane: checks for membrane.py
class TestMembrane:
    # test_pick_finds_bump_beside_membrane: a dense bump beside a flat dense membrane is the best candidate, the bare membrane is not
    def test_pick_finds_bump_beside_membrane(self) -> None:
        rng = np.random.default_rng(0)
        x = ndi.gaussian_filter(rng.standard_normal((48, 64, 64)), 1.0).astype(np.float32)
        x /= x.std()
        m = np.zeros(x.shape, bool)
        m[:, 30:32, :] = True
        x[m] += 3.0    # the membrane itself is dense, the bare profile must absorb it
        zz, yy, xx = np.indices(x.shape)
        x[(zz - 24) ** 2 + (yy - 35) ** 2 + (xx - 20) ** 2 <= 9] += 3.0
        volume = Volume(x, m, np.zeros_like(m), np.ones_like(x), (0, 48), Params().grid(15.68))    # factor 1
        c = membrane.pick(volume)
        assert c.shape[1] == 5 and (c[:, 4] == 1).all()
        best = c[np.argmax(c[:, 3])]
        assert np.linalg.norm(best[[2, 1, 0]] - (24.5, 35.5, 20.5)) <= 3

    # test_pick_scales_coordinates_to_input_voxels_and_honours_lamella: z offset of the lamella and the binning factor both enter the output coordinates
    def test_pick_scales_coordinates_to_input_voxels_and_honours_lamella(self) -> None:
        rng = np.random.default_rng(0)
        x = ndi.gaussian_filter(rng.standard_normal((48, 64, 64)), 1.0).astype(np.float32)
        m = np.zeros(x.shape, bool)
        m[:, 30:32, :] = True
        volume = Volume(x, m, np.zeros_like(m), np.ones_like(x), (10, 40), Params().grid(7.84))    # factor 2
        c = membrane.pick(volume)
        assert c[:, 2].min() >= 2 * 10 and c[:, 2].max() <= 2 * 40
