'''
icepick: tests for shared utilities
'''

# Import external dependencies
import numpy as np, pytest
from scipy import ndimage as ndi

# Import internal icepick objects
from icepick.config import Params
from icepick.utils import filters, io, volume
from icepick.utils.backend import asdevice, asnumpy, irfftn, rfftn

# _VOXEL: input voxel size (Angstrom) at which a 250 A particle needs no binning
_VOXEL: float = 15.68

# _GRID: the default working grid, as for a 250 A particle at 15.68 A/voxel
_GRID = Params().grid(_VOXEL)

# TestBackend: tests for utils/backend.py
class TestBackend:
    # test_fft_round_trip: forward then inverse transform returns the input, including for odd lengths
    def test_fft_round_trip(self) -> None:
        a = np.random.default_rng(0).standard_normal((8, 9, 11)).astype(np.float32)
        assert np.allclose(irfftn(rfftn(a), a.shape), a, atol=1e-5)

    # test_host_device_round_trip: moving to the backend and back keeps the values
    def test_host_device_round_trip(self) -> None:
        a = np.arange(6.0)
        assert np.array_equal(asnumpy(asdevice(a)), a)

# TestFilters: checks for utils/filters.py
class TestFilters:
    # test_blobs_find_blob_beside_excluded_slab: a planted blob next to an excluded slab is found, nothing is found inside the slab
    def test_blobs_find_blob_beside_excluded_slab(self) -> None:
        rng = np.random.default_rng(0)
        x = ndi.gaussian_filter(rng.standard_normal((48, 64, 64)), 1.0).astype(np.float32)
        x /= x.std()
        zz, yy, xx = np.indices(x.shape)
        x += 4.0 * np.exp(-((zz - 24) ** 2 + (yy - 30) ** 2 + (xx - 40) ** 2) / (2 * 3.0 ** 2))
        w = np.ones_like(x)
        w[:, :, :16] = 0    # membrane slab
        x[:, :, :16] += 8.0    # dense, so an unmasked filter would peak here
        c = filters.blobs(x, w, _GRID)
        assert len(c) and (c[:, 0] >= 16).all()
        best = c[np.argmax(c[:, 3])]
        assert np.linalg.norm(best[[2, 1, 0]] - [24.5, 30.5, 40.5]) <= 3

    # test_blobs_suppress_within_scale_only: one particle gives one candidate per scale at most
    def test_blobs_suppress_within_scale_only(self) -> None:
        x = np.zeros((40, 40, 40), np.float32)
        x[20, 20, 20] = 1.0
        x = ndi.gaussian_filter(x, 3.0) * 1000
        rng = np.random.default_rng(1)
        x += rng.standard_normal(x.shape).astype(np.float32) * 0.01
        c = filters.blobs(x, np.ones_like(x), _GRID)
        assert 1 <= len(c) <= len(_GRID.sigmas)

# TestIo: tests for utils/io.py
class TestIo:
    # test_mrc_info_reads_shape_and_voxel_size: header values come back as written
    def test_mrc_info_reads_shape_and_voxel_size(self, tmp_path, write_mrc) -> None:
        path = write_mrc(tmp_path / 'a.mrc', np.zeros((5, 6, 7), np.float32), 7.84)
        shape, voxel_size = io.mrc_info(path)
        assert shape == (5, 6, 7)
        assert voxel_size == 7.84    # exactly, not the float32 neighbour

    # test_bin_volume_matches_numpy_on_cropped_shape: mean and max binning of a shape aren't divisible by the factor
    @pytest.mark.parametrize('factor', [2, 3])
    def test_bin_volume_matches_numpy_on_cropped_shape(self, tmp_path, write_mrc, factor) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((150, 13, 17)).astype(np.float32)    # 150 slices span three 64-slice chunks
        path = write_mrc(tmp_path / 'a.mrc', data)
        nz, ny, nx = (n // factor * factor for n in data.shape)
        blocks = data[:nz, :ny, :nx].reshape(nz // factor, factor, ny // factor, factor, nx // factor, factor)
        assert np.allclose(io.bin_volume(path, factor, np.mean), blocks.mean(axis=(1, 3, 5)), atol=1e-6)
        assert np.array_equal(io.bin_volume(path, factor, np.max), blocks.max(axis=(1, 3, 5)))

    # test_bin_volume_reads_int8_masks: segmentation volumes are stored as bytes
    def test_bin_volume_reads_int8_masks(self, tmp_path, write_mrc) -> None:
        mask = np.zeros((4, 4, 4), np.int8)
        mask[1, 1, 1] = 1
        path = write_mrc(tmp_path / 'm.mrc', mask)
        out = io.bin_volume(path, 2, np.max)
        assert out.dtype == np.float32 and out.sum() == 1

    # test_bin_volume_rejects_volume_smaller_than_factor: nothing left after cropping raises an error
    def test_bin_volume_rejects_volume_smaller_than_factor(self, tmp_path, write_mrc) -> None:
        path = write_mrc(tmp_path / 'a.mrc', np.zeros((1, 4, 4), np.float32))
        with pytest.raises(ValueError):
            io.bin_volume(path, 2, np.mean)

# TestVolume: tests for utils/volume.py
class TestVolume:
    # test_load_masks_membrane_and_speck: the membrane is dilated, a bright speck is masked, density is centred outside both
    def test_load_masks_membrane_and_speck(self, tmp_path, write_mrc) -> None:
        rng = np.random.default_rng(0)
        d = rng.standard_normal((32, 48, 48)).astype(np.float32) + 5.0
        m = np.zeros(d.shape, np.int8)
        m[:, 20:22, :] = 1
        d[10:12, 8:10, 8:10] = 60.0    # far brighter than anything protein makes
        v = volume.load(write_mrc(tmp_path / 'd.mrc', d, _VOXEL), write_mrc(tmp_path / 's.mrc', m, _VOXEL), Params())
        assert v.grid.factor == 1
        assert v.weight[16, 20, 24] == 0 and v.weight[16, 19, 24] == 0    # membrane and its dilation
        assert v.weight[10, 8, 8] == 0 and v.artefact[10, 8, 8]    # speck
        assert v.weight[16, 40, 24] == 1
        assert abs(np.median(v.x[v.weight > 0])) < 0.05    # the speck stays in the normalisation, so only the centre is checked

    # test_load_rejects_mismatched_shapes: a segmentation of another size is an error
    def test_load_rejects_mismatched_shapes(self, tmp_path, write_mrc) -> None:
        d = write_mrc(tmp_path / 'd.mrc', np.zeros((8, 8, 8), np.float32), _VOXEL)
        s = write_mrc(tmp_path / 's.mrc', np.zeros((8, 8, 6), np.int8), _VOXEL)
        try:
            volume.load(d, s, Params())
        except ValueError as error:
            assert 'differs' in str(error)
        else:
            raise AssertionError('expected ValueError')

    # test_sample_z_brackets_structured_middle: structure peaks mid-stack, junk rises at both ends, the range sits between
    def test_sample_z_brackets_structured_middle(self) -> None:
        rng = np.random.default_rng(0)
        z = np.arange(120)
        amplitude = 0.3 + 2.0 * np.exp(-(((z - 60) / 15) ** 2)) + 1.5 * np.exp(-((z / 6) ** 2)) + 1.5 * np.exp(-(((z - 119) / 6) ** 2))
        x = ndi.gaussian_filter(rng.standard_normal((120, 40, 40)), 1.5) * amplitude[:, None, None]
        lo, hi = volume.sample_z(x.astype(np.float32))
        assert 10 <= lo <= 45 and 75 <= hi <= 110
