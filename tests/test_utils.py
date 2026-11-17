'''
icepick: tests for shared utilities
'''

# Import external dependencies
import numpy as np, pytest, sys
from loguru import logger
from scipy import ndimage as ndi

# Import internal icepick objects
from icepick.config import Params
from icepick.utils import backend, filters, gpu, io, log, plot, volume

# _VOXEL: input voxel size (Angstrom) at which a 250 A particle needs no binning
_VOXEL: float = 15.68

# _GRID: the default working grid, as for a 250 A particle at 15.68 A/voxel
_GRID = Params().grid(_VOXEL)

# TestBackend: tests for utils/backend.py
class TestBackend:
    # test_fft_round_trip: forward then inverse transform returns the input, including for odd lengths
    def test_fft_round_trip(self) -> None:
        a = np.random.default_rng(0).standard_normal((8, 9, 11)).astype(np.float32)
        assert np.allclose(backend.irfftn(backend.rfftn(a), a.shape), a, atol=1e-5)

    # test_host_device_round_trip: moving to the backend and back keeps the values
    def test_host_device_round_trip(self) -> None:
        a = np.arange(6.0)
        assert np.array_equal(backend.asnumpy(backend.asdevice(a)), a)

    # test_use_cpu_restores_numpy: after switching back, xp and ndi forward to numpy and scipy
    def test_use_cpu_restores_numpy(self) -> None:
        backend.use_cpu()
        assert backend.xp.zeros(3).dtype == np.float64
        assert backend.ndi.gaussian_filter is ndi.gaussian_filter
        assert isinstance(backend.asdevice(np.arange(3)), np.ndarray)

# TestFilters: tests for utils/filters.py
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

    # test_matched_finds_planted_particle: top filter peak lands on a z-stretched ball hidden in correlated noise
    def test_matched_finds_planted_particle(self) -> None:
        rng = np.random.default_rng(0)
        x = ndi.gaussian_filter(rng.standard_normal((64, 96, 96)), 1.5).astype(np.float32)
        x /= x.std()
        self._plant(x, [(32, 40, 55)], 1.0)
        w = np.ones_like(x)
        w[:, :, :10] = 0    # a membrane slab, filled in and excluded as centres
        s = filters.matched(x, w, _GRID)
        s[w == 0] = -np.inf
        top = np.unravel_index(np.argmax(s), s.shape)
        assert np.linalg.norm(np.subtract(top, (32, 40, 55))) <= 2

    # test_tiled_matched_finds_particles_in_both_noise_regions: the noise spectrum and level change along z, tile seams must not hide a particle in either half
    def test_tiled_matched_finds_particles_in_both_noise_regions(self) -> None:
        rng = np.random.default_rng(0)
        x = np.empty((192, 64, 64), np.float32)
        x[:96] = ndi.gaussian_filter(rng.standard_normal((96, 64, 64)), 2.0) * 6    # smooth, strong
        x[96:] = rng.standard_normal((96, 64, 64)) * 0.5    # white, weak
        spots = [(48, 32, 32), (144, 32, 32)]
        self._plant(x, spots, 0.8)
        w = np.ones_like(x)

        def rank(s: np.ndarray, spot: tuple[int, int, int]) -> float:
            return float((s > s[spot]).mean())    # fraction of the whole volume scoring above the particle centre

        tiled = filters.matched(x, w, _GRID)
        assert all(rank(tiled, s) < 0.0001 for s in spots)

    # test_matched_rejects_volume_smaller_than_template: a volume that cannot hold the template box is an error
    def test_matched_rejects_volume_smaller_than_template(self) -> None:
        x = np.zeros((20, 64, 64), np.float32)
        with pytest.raises(SystemExit):
            filters.matched(x, np.ones_like(x), _GRID)

    # test_thin_keeps_best_of_close_pair: of two candidates closer than the radius only the better survives
    def test_thin_keeps_best_of_close_pair(self) -> None:
        c = np.array([[0, 0, 0, 1.0], [5, 0, 0, 2.0], [40, 0, 0, 0.5]])
        assert filters.thin(c, 12.0)[:, 3].tolist() == [2.0, 0.5]

    # test_rescore_takes_max_in_three_voxel_neighbourhood: a candidate one voxel off the filter peak still gets the peak
    def test_rescore_takes_max_in_three_voxel_neighbourhood(self) -> None:
        s = np.zeros((5, 5, 5), np.float32)
        s[2, 2, 2] = 7.0
        c = np.array([[3.5, 2.5, 2.5, 0.0], [0.5, 0.5, 0.5, 0.0]])    # x, y, z in voxel-centre coordinates
        assert filters.rescore(c, s).tolist() == [7.0, 0.0]

    # _plant: add z-stretched balls of the given amplitude to a volume
    @staticmethod
    def _plant(x: np.ndarray, centres: list[tuple[int, int, int]], amplitude: float, radius: float = 6.0) -> None:
        zz, yy, xx = np.indices(x.shape)
        for cz, cy, cx in centres:
            x[np.sqrt(((zz - cz) / 1.5) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) < radius] += amplitude

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
        with pytest.raises(SystemExit):
            io.bin_volume(path, 2, np.mean)
            log = (tmp_path / 'out' / 'icepick.log').read_text()
            assert 'ERROR' in log

    # test_write_candidates_round_trips_npy_and_star: both files hold the same rows, the star has one header block and one line per candidate
    def test_write_candidates_round_trips_npy_and_star(self, tmp_path) -> None:
        c = np.array([[10.5, 20.25, 30.0, 0.75, 0.0], [1.0, 2.0, 3.0, 0.5, 1.0]])
        io.write_candidates(tmp_path / 'tomo.v1', c)    # a dot in the name must survive
        assert np.array_equal(np.load(tmp_path / 'tomo.v1.npy'), c)
        lines = (tmp_path / 'tomo.v1.star').read_text().splitlines()
        assert lines[:3] == ['data_particles', '', 'loop_'] and lines[3:8] == [
            '_rlnCoordinateX #1', '_rlnCoordinateY #2', '_rlnCoordinateZ #3', '_rlnAutopickFigureOfMerit #4', '_icepickStream #5']
        rows = np.array([[float(v) for v in line.split()] for line in lines[8:]])
        assert np.allclose(rows, c, atol=0.01) and rows[:, 4].tolist() == [0.0, 1.0]

    # test_append_picks_adds_priors_and_logs_existing: one header, stream-mapped priors, an info message counting tomograms already present
    def test_append_picks_adds_priors_and_logs_existing(self, tmp_path) -> None:
        c = np.array([[10.5, 20.25, 30.0, 0.75, 0.0], [1.0, 2.0, 3.0, 0.5, 1.0]])
        path = tmp_path / 'picks.csv'
        messages = []
        logger.add(messages.append, format='{message}')
        io.append_picks(path, 'a', c, stamp_priors=True)
        assert not messages    # a new file needs no message
        io.append_picks(path, 'b', c, stamp_priors=True)
        lines = path.read_text().splitlines()
        assert lines[0] == 'tomogram_id,x,y,z,confidence,stream,prior' and len(lines) == 5
        assert lines[1].endswith(',0,none') and lines[2].endswith(',1,normal')
        assert f'appending to {path}, picks from 1 tomograms already present' in messages[0]

    # test_check_picks_rejects_duplicates_and_column_mismatch: a repeated tomogram_id or different columns is an error, a missing file holds none
    def test_check_picks_rejects_duplicates_and_column_mismatch(self, tmp_path) -> None:
        c = np.array([[1.0, 2.0, 3.0, 0.5, 1.0]])
        path = tmp_path / 'picks.csv'
        assert io.check_picks(path, 'a') == 0
        io.append_picks(path, 'a', c, stamp_priors=True)
        assert io.check_picks(path, 'b', stamp_priors=True) == 1
        with pytest.raises(SystemExit):
            io.check_picks(path, 'a', stamp_priors=True)
            log = (tmp_path / 'out' / 'icepick.log').read_text()
            assert 'ERROR' and 'already in' in log
        with pytest.raises(SystemExit):
            io.check_picks(path, 'b')
            log = (tmp_path / 'out' / 'icepick.log').read_text()
            assert 'ERROR' and 'do not match' in log
        with pytest.raises(SystemExit):
            io.append_picks(path, 'a', c, stamp_priors=True)
            log = (tmp_path / 'out' / 'icepick.log').read_text()
            assert 'ERROR' and 'already in' in log

    # test_mrc_info_does_not_map_data: the header is enough, so the data mapper is never called
    def test_mrc_info_does_not_map_data(self, tmp_path, write_mrc, monkeypatch) -> None:
        path = write_mrc(tmp_path / 'a.mrc', np.zeros((4, 5, 6), np.float32), voxel_size=7.84)
        def fail(*args, **kwargs):
            raise AssertionError('data was mapped')
        monkeypatch.setattr(io.mrcfile, 'mmap', fail)
        assert io.mrc_info(path) == ((4, 5, 6), 7.84)

    # test_mrc_info_rejects_single_image: a 2D image is not a volume
    def test_mrc_info_rejects_single_image(self, tmp_path, write_mrc) -> None:
        path = write_mrc(tmp_path / 'a.mrc', np.zeros((5, 6), np.float32))
        with pytest.raises(SystemExit):
            io.mrc_info(path)
            log = (tmp_path / 'out' / 'icepick.log').read_text()
            assert 'ERROR' in log

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
        with pytest.raises(SystemExit):
            volume.load(d, s, Params())
            log = (tmp_path / 'out' / 'icepick.log').read_text()
            assert 'ERROR' in log

    # test_sample_z_brackets_structured_middle: structure peaks mid-stack, junk rises at both ends, the range sits between
    def test_sample_z_brackets_structured_middle(self) -> None:
        rng = np.random.default_rng(0)
        z = np.arange(120)
        amplitude = 0.3 + 2.0 * np.exp(-(((z - 60) / 15) ** 2)) + 1.5 * np.exp(-((z / 6) ** 2)) + 1.5 * np.exp(-(((z - 119) / 6) ** 2))
        x = ndi.gaussian_filter(rng.standard_normal((120, 40, 40)), 1.5) * amplitude[:, None, None]
        lo, hi = volume.sample_z(x.astype(np.float32))
        assert 10 <= lo <= 45 and 75 <= hi <= 110

    # test_sample_z_slabs_match_single_slab: slabbing with a halo gives the same range as filtering the whole volume at once
    def test_sample_z_slabs_match_single_slab(self) -> None:
        rng = np.random.default_rng(1)
        z = np.arange(120)
        amplitude = 0.3 + 2.0 * np.exp(-(((z - 60) / 15) ** 2)) + 1.5 * np.exp(-((z / 6) ** 2)) + 1.5 * np.exp(-(((z - 119) / 6) ** 2))
        x = (ndi.gaussian_filter(rng.standard_normal((120, 40, 40)), 1.5) * amplitude[:, None, None]).astype(np.float32)
        assert volume.sample_z(x, slab=7) == volume.sample_z(x, slab=len(x))

    # test_slice_profile_matches_sample_z: the range from sample_z sits between the profile minima either side of its peak
    def test_slice_profile_matches_sample_z(self) -> None:
        x = np.random.default_rng(2).standard_normal((60, 24, 24)).astype(np.float32)
        p = volume.slice_profile(x)
        top = int(np.argmax(p))
        assert volume.sample_z(x) == (int(np.argmin(p[:top + 1])), top + int(np.argmin(p[top:])) + 1)

# TestPlot: tests for utils/plot.py
class TestPlot:
    # test_save_slice_range_writes_png: the plot is written as a PNG
    def test_save_slice_range_writes_png(self, tmp_path) -> None:
        x = np.random.default_rng(3).standard_normal((60, 24, 24)).astype(np.float32)
        plot.save_slice_range(tmp_path / 'p.png', x, (10, 50))
        assert (tmp_path / 'p.png').read_bytes()[:4] == b'\x89PNG'

    # test_require_matplotlib_exits_when_missing: a missing matplotlib exits with the install hint
    def test_require_matplotlib_exits_when_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, 'matplotlib', None)    # makes the import raise ImportError
        with pytest.raises(SystemExit):
            plot.require_matplotlib()

# TestGpu: tests for utils/gpu.py
class TestGpu:
    # test_cuda_visible_devices_wins: the CUDA list is used over scheduler variables
    def test_cuda_visible_devices_wins(self, monkeypatch) -> None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
        monkeypatch.setenv('SLURM_STEP_GPUS', '0')
        assert gpu.assigned_gpus() == ['2', '3']

    # test_empty_cuda_visible_devices_means_none: set but empty hides every GPU
    def test_empty_cuda_visible_devices_means_none(self, monkeypatch) -> None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
        assert gpu.assigned_gpus() == []

    # test_slurm_fallback: scheduler variables apply when CUDA_VISIBLE_DEVICES is unset
    def test_slurm_fallback(self, monkeypatch) -> None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
        monkeypatch.setenv('SLURM_JOB_GPUS', '1,4')
        assert gpu.assigned_gpus() == ['1', '4']

    # test_nvidia_smi_fallback: with nothing assigned the card list comes from nvidia-smi
    def test_nvidia_smi_fallback(self, monkeypatch) -> None:
        for name in ('CUDA_VISIBLE_DEVICES', *gpu._SLURM):
            monkeypatch.delenv(name, raising=False)
        class Done:
            stdout = 'GPU 0: A100 (UUID: GPU-aaa)\nGPU 1: A100 (UUID: GPU-bbb)\n'
        monkeypatch.setattr(gpu.subprocess, 'run', lambda *a, **k: Done())
        assert gpu.assigned_gpus() == ['0', '1']

    # test_missing_nvidia_smi_is_empty: no tool means no GPUs, not an error
    def test_missing_nvidia_smi_is_empty(self, monkeypatch) -> None:
        for name in ('CUDA_VISIBLE_DEVICES', *gpu._SLURM):
            monkeypatch.delenv(name, raising=False)
        def missing(*args, **kwargs):
            raise FileNotFoundError
        monkeypatch.setattr(gpu.subprocess, 'run', missing)
        assert gpu.assigned_gpus() == []

    # test_require_gpu_errors_when_none: nothing assigned raises a clear error
    def test_require_gpu_errors_when_none(self, monkeypatch) -> None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
        with pytest.raises(SystemExit):
            gpu.require_gpu()
            log = (tmp_path / 'out' / 'icepick.log').read_text()
            assert 'ERROR' and 'no GPU' in log

# TestLog: tests for utils/log.py
class TestLog:
    # test_setup_writes_debug_to_file_and_info_to_console: the file keeps DEBUG, the console only INFO, both use the shared format
    def test_setup_writes_debug_to_file_and_info_to_console(self, tmp_path, capsys) -> None:
        log.setup(tmp_path, 'tomo_a.mrc')
        logger.debug('fine detail')
        logger.info('headline')
        logger.remove()    # flush and close the file sink
        text = (tmp_path / 'icepick.log').read_text()
        assert '[DEBUG]    [tomo_a.mrc] fine detail' in text and '[INFO]     [tomo_a.mrc] headline' in text
        assert text.splitlines()[0][:4].isdigit()
        err = capsys.readouterr().err
        assert 'headline' in err and 'fine detail' not in err

    # test_setup_verbosity_sets_console_level: -v adds PROGRESS and -vv adds DEBUG to the console, the file always keeps both
    def test_setup_verbosity_sets_console_level(self, tmp_path, capsys) -> None:
        for verbosity, expected in ((0, (False, False)), (1, (True, False)), (2, (True, True))):
            log.setup(tmp_path, 'tomo_a.mrc', verbosity=verbosity)
            log.progress('step')
            logger.debug('detail')
            logger.remove()
            err = capsys.readouterr().err
            assert ('step' in err, 'detail' in err) == expected
        text = (tmp_path / 'icepick.log').read_text()
        assert '[PROGRESS] [tomo_a.mrc] step' in text and '[DEBUG]    [tomo_a.mrc] detail' in text
