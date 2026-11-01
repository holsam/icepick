'''
icepick: tests for shared utilities
'''

# Import external dependencies
import numpy as np, pytest

# Import internal icepick objects
from icepick.utils import io
from icepick.utils.backend import asdevice, asnumpy, irfftn, rfftn

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
