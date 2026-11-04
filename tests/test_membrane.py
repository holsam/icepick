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

    # test_shell_slabs_match_single_transform: slabbed shells equal one whole-volume edt, including membrane-free slabs and membrane on the first and last slice
    def test_shell_slabs_match_single_transform(self) -> None:
        rng = np.random.default_rng(2)
        grid = Params().grid(15.68)
        m = np.zeros((80, 24, 24), bool)
        m[20:40] = rng.random((20, 24, 24)) < 0.02    # sparse middle, long membrane-free runs either side
        m[0, 5, 5] = m[-1, 7, 9] = True    # membrane on the first and last slice
        full = np.minimum(np.round(ndi.distance_transform_edt(~m)), grid.band + 1).astype(np.int16)
        assert np.array_equal(membrane._shell(m, grid, slab=5), full)
