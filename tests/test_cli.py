'''
icepick: tests for the command line
'''

# Import external dependencies
import numpy as np
from typer.testing import CliRunner

# Import internal icepick objects
from icepick.cli import app

# TestCli: tests for cli.py
class TestCli:
    # test_pick_writes_npy_and_star: the command picks a synthetic tomogram and writes both files named after the density
    def test_pick_writes_npy_and_star(self, tmp_path, write_mrc) -> None:
        rng = np.random.default_rng(0)
        d = rng.standard_normal((64, 96, 96)).astype(np.float32)
        m = np.zeros(d.shape, np.int8)
        m[:, 46:48, :] = 1
        density, seg = write_mrc(tmp_path / 'tomo_a.mrc', d), write_mrc(tmp_path / 'tomo_a_segmented.mrc', m)
        result = CliRunner().invoke(app, [str(density), str(seg), '-o', str(tmp_path / 'out'), '--stamp-priors'])
        c = np.load(tmp_path / 'out' / 'tomo_a.npy')
        star = (tmp_path / 'out' / 'tomo_a.star').read_text().splitlines()
        assert result.exit_code == 0
        assert c.shape[1] == 5 and len(c) > 0
        assert len(star) == 8 + len(c)
        assert f'{len(c)} candidates' in result.output
        picks = (tmp_path / 'out' / 'icepick.csv').read_text().splitlines()
        assert picks[0] == 'tomogram_id,x,y,z,confidence,stream,prior' and len(picks) == 1 + len(c)
        assert picks[1].startswith('tomo_a_segmented,')
        log = (tmp_path / 'out' / 'icepick.log').read_text()
        assert f'{len(c)} candidates' in log
        for stage in ('loaded tomogram', 'cytosolic:', 'membrane:', 'merged', 'using working slices:', 'wrote '):
            assert stage in log    # every stage reports, DEBUG detail included

    # test_pick_stops_on_duplicate_tomogram: a tomogram already in icepick.csv exits non-zero before picking and leaves the file alone
    def test_pick_stops_on_duplicate_tomogram(self, tmp_path, write_mrc, monkeypatch) -> None:
        f = write_mrc(tmp_path / 'tomo_a.mrc', np.zeros((8, 8, 8), np.float32))
        out = tmp_path / 'out'
        out.mkdir()
        (out / 'icepick.csv').write_text('tomogram_id,x,y,z,confidence,stream\ntomo_a,1.00,2.00,3.00,0.5,0\n')
        monkeypatch.setattr('icepick.cli.pick_tomogram', lambda *a: (_ for _ in ()).throw(AssertionError('picked')))
        result = CliRunner().invoke(app, [str(f), str(f), '-o', str(out)])
        assert result.exit_code == 1 and 'already in' in result.output
        assert len((out / 'icepick.csv').read_text().splitlines()) == 2

    # test_pick_rejects_bad_options: a non-positive diameter is a usage error, not a traceback
    def test_pick_rejects_bad_options(self, tmp_path, write_mrc) -> None:
        f = write_mrc(tmp_path / 'd.mrc', np.zeros((8, 8, 8), np.float32))
        result = CliRunner().invoke(app, [str(f), str(f), '--particle-diameter', '0'])
        assert result.exit_code == 2

    # test_gpu_without_gpu_fails: --gpu with no assigned GPU exits non-zero and names the problem
    def test_gpu_without_gpu_fails(self, tmp_path, write_mrc, monkeypatch) -> None:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
        density = write_mrc(tmp_path / 'd.mrc', np.zeros((8, 8, 8), np.float32))
        seg = write_mrc(tmp_path / 's.mrc', np.zeros((8, 8, 8), np.int8))
        result = CliRunner().invoke(app, [str(density), str(seg), '--gpu'])
        assert result.exit_code != 0
        assert 'no GPU' in result.output
