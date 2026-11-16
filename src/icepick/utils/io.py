'''
icepick: file i/o utilities
'''

# Import external dependencies
import csv, mrcfile, numpy as np, warnings
from collections.abc import Callable
from pathlib import Path

# _CHUNK: input slices read at a time while binning
_CHUNK: int = 64

# mrc_info: (z, y, x) shape and voxel size in Angstrom from the MRC header
def mrc_info(path: Path) -> tuple[tuple[int, int, int], float]:
    with mrcfile.open(path, mode='r', permissive=True, header_only=True) as mrc:
        z, y, x = int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx)
        if z < 2:
            raise ValueError(f'expected a 3D volume in {path!r}, got shape {(z, y, x)!r}')
        return (z, y, x), round(float(mrc.voxel_size.x), 4)    # float32 header noise would move radii across voxel distances

# bin_volume: read an MRC in z chunks and reduce each factor-sized block, cropping to a multiple of the factor
def bin_volume(path: Path, factor: int, reduce: Callable[..., np.ndarray]) -> np.ndarray:
    with mrcfile.open(path, mode='r', permissive=True, header_only=True) as mrc:
        h = mrc.header
        shape = (int(h.nz), int(h.ny), int(h.nx))
        dtype, offset = mrcfile.utils.data_dtype_from_header(h), 1024 + int(h.nsymbt)
    nz, ny, nx = (n // factor * factor for n in shape)
    if min(nz, ny, nx) == 0:
        raise ValueError(f'volume {shape!r} in {path!r} is smaller than binning factor {factor}')
    out = np.empty((nz // factor, ny // factor, nx // factor), np.float32)
    step = factor * max(_CHUNK // factor, 1)    # chunks must hold whole blocks
    with open(path, 'rb') as f:
        for z0 in range(0, nz, step):
            z1 = min(z0 + step, nz)
            f.seek(offset + z0 * shape[1] * shape[2] * dtype.itemsize)
            slab = np.fromfile(f, dtype, (z1 - z0) * shape[1] * shape[2]).reshape(z1 - z0, shape[1], shape[2])[:, :ny, :nx].astype(np.float32)
            blocks = slab.reshape((z1 - z0) // factor, factor, ny // factor, factor, nx // factor, factor)
            out[z0 // factor:z1 // factor] = reduce(blocks, axis=(1, 3, 5))
    return out

# write_candidates: save an (n, 5) candidate array (x, y, z, score, stream) in input voxels as <stem>.npy and a RELION-style <stem>.star
def write_candidates(stem: Path, c: np.ndarray) -> None:
    np.save(Path(f'{stem}.npy'), c)    # not with_suffix, tomogram names may contain dots
    with open(f'{stem}.star', 'w') as f:
        f.write('data_particles\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n_rlnAutopickFigureOfMerit #4\n_icepickStream #5\n')
        np.savetxt(f, c, fmt=['%.2f', '%.2f', '%.2f', '%.6f', '%d'])

# _STAMP_PRIORS: STAMP orientation prior per icepick stream id
_STAMP_PRIORS: dict[int, str] = {0: 'none', 1: 'normal'}

# append_picks: append candidates to a shared picks CSV with a tomogram_id column, optionally adding a STAMP prior column
def append_picks(path: Path, tomogram_id: str, c: np.ndarray, *, stamp_priors: bool = False) -> None:
    header = ['tomogram_id', 'x', 'y', 'z', 'confidence', 'stream'] + (['prior'] if stamp_priors else [])
    if path.exists():
        with open(path, newline='') as f:
            rows = list(csv.reader(f))
        if not rows or rows[0] != header:
            raise ValueError(f'columns in {path!r} do not match {header!r}, remove it or match --stamp-priors')
        if any(r[0] == tomogram_id for r in rows[1:]):
            warnings.warn(f'{tomogram_id!r} is already in {path!r}, appending duplicate picks')
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(header)
        for x, y, z, score, stream in c:
            row = [tomogram_id, f'{x:.2f}', f'{y:.2f}', f'{z:.2f}', f'{score:.6f}', int(stream)]
            w.writerow(row + ([_STAMP_PRIORS[int(stream)]] if stamp_priors else []))
