'''
icepick: slice range plot
'''

# Import external dependencies
import numpy as np
from pathlib import Path

# Import internal icepick objects
from icepick.utils.backend import asnumpy
from icepick.utils.log import error_exit
from icepick.utils.volume import slice_profile

# require_matplotlib: exit with an install hint unless matplotlib imports
def require_matplotlib() -> None:
    try:
        import matplotlib
    except ImportError as error:
        error_exit("matplotlib is not installed, install with 'uv tool install \"git+https://github.com/holsam/icepick[plot]\"'", RuntimeError)

# save_slice_range: save a central x-z view of the working volume beside the per-slice spread profile, with the kept slice range marked
def save_slice_range(path: Path, x: np.ndarray, slices: tuple[int, int]) -> None:
    import matplotlib
    matplotlib.use('Agg')    # file output only, no display needed
    import matplotlib.pyplot as plt
    profile = slice_profile(x)
    view = asnumpy(x[:, x.shape[1] // 2, :])
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(8, 5), gridspec_kw={'width_ratios': [2, 1]})
    left.imshow(view, cmap='gray', origin='lower', aspect='auto', vmin=-3, vmax=3)
    left.set(xlabel='x (working voxels)', ylabel='z (working slices)', title='central x-z slice')
    right.plot(profile, np.arange(len(profile)))
    right.set(xlabel='band-passed spread', title='per-slice spread')
    for ax in (left, right):
        for z in slices:
            ax.axhline(z, color='tab:red', linestyle='--')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
