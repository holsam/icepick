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
        result = CliRunner().invoke(app, [str(density), str(seg), '-o', str(tmp_path / 'out')])
        c = np.load(tmp_path / 'out' / 'tomo_a.npy')
        star = (tmp_path / 'out' / 'tomo_a.star').read_text().splitlines()
        assert result.exit_code == 0
        assert c.shape[1] == 5 and len(c) > 0
        assert len(star) == 8 + len(c)
        assert f'tomo_a {len(c)} candidates' in result.output

    # test_pick_rejects_bad_options: a non-positive diameter is a usage error, not a traceback
    def test_pick_rejects_bad_options(self, tmp_path, write_mrc) -> None:
        f = write_mrc(tmp_path / 'd.mrc', np.zeros((8, 8, 8), np.float32))
        result = CliRunner().invoke(app, [str(f), str(f), '--particle-diameter', '0'])
        assert result.exit_code == 2
