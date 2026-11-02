'''
icepick: tests for the merge module
'''

# Import external dependencies
import numpy as np

# Import internal icepick objects
from icepick.config import Params
from icepick.merge import merge, pick_tomogram

# TestMerge: checks for merge.py
class TestMerge:
    # test_merge_drops_cross_stream_duplicates_only: a close pair from different streams keeps the better rank, a close pair within a stream keeps both
    def test_merge_drops_cross_stream_duplicates_only(self) -> None:
        cyto = np.array([[0, 0, 0, 1.0, 0], [100, 0, 0, 5.0, 0], [104, 0, 0, 2.0, 0]])
        mem = np.array([[3, 0, 0, 9.0, 1], [50, 0, 0, 1.0, 1]])
        out = merge([cyto, mem], 8.0)
        assert len(out) == 4    # x 0 and 3 collapse across streams, x 100 and 104 share a stream so both stay
        assert out[out[:, 0] < 10, 4].tolist() == [1.0]    # membrane rank 1.0 beats cytosolic rank 1/3

    # test_merge_handles_empty_streams: no candidates anywhere gives a well-shaped empty array
    def test_merge_handles_empty_streams(self) -> None:
        assert merge([np.zeros((0, 5)), np.zeros((0, 5))], 8.0).shape == (0, 5)

    # test_pick_tomogram_runs_both_streams_in_input_voxels: a synthetic tomogram with a membrane gives candidates from both streams inside the volume
    def test_pick_tomogram_runs_both_streams_in_input_voxels(self, tmp_path, write_mrc) -> None:
        rng = np.random.default_rng(0)
        d = rng.standard_normal((64, 96, 96)).astype(np.float32)
        m = np.zeros(d.shape, np.int8)
        m[:, 46:48, :] = 1
        c = pick_tomogram(write_mrc(tmp_path / 'd.mrc', d, 7.84), write_mrc(tmp_path / 's.mrc', m, 7.84), Params())    # factor 2
        assert set(np.unique(c[:, 4])) == {0.0, 1.0}
        assert (c[:, :3] >= 0).all() and (c[:, :3] <= [96, 96, 64]).all()
