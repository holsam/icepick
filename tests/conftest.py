'''
icepick: shared test fixtures
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from collections.abc import Callable
from pathlib import Path

# write_mrc: fixture returning a function that saves an array as an MRC with a voxel size in Angstrom
@pytest.fixture
def write_mrc() -> Callable[..., Path]:
    def write(path: Path, data: np.ndarray, voxel_size: float = 7.84) -> Path:
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = voxel_size
        return path
    return write
