'''
icepick: command line entry point
'''

# Import external dependencies
import numpy as np, sys, time, typer
from importlib.metadata import version
from loguru import logger
from pathlib import Path
from pydantic import ValidationError
from typing import Annotated

# Import internal icepick objects
from icepick.config import Params
from icepick.merge import pick_tomogram
from icepick.utils import backend
from icepick.utils.gpu import require_gpu
from icepick.utils import log
from icepick.utils.io import append_picks, check_picks, write_candidates

# app: initialise Typer instance
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help='Template-free particle picking for cryoET workflows.',
)

# _DEFAULTS: parameter defaults, shown in the help text
_DEFAULTS: Params = Params()

# pick: pick one tomogram and write <density name>.npy and .star, and append to icepick.csv
@app.command()
def pick(
    density: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help='Path to denoised tomogram MRC file.'),
    ],
    segmentation: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help='Path to binary membrane segmentation MRC file.'),
    ],
    output: Annotated[
        Path,
        typer.Option('-o', '--output', help='Path to output directory.'),
    ] = Path('.'),
    particle_diameter: Annotated[
        float,
        typer.Option(help='Particle diameter in Ångstrom.'),
    ] = _DEFAULTS.particle_diameter,
    tilt_range: Annotated[
        float,
        typer.Option(help='Maximum tilt angle (±) in degrees.')
    ] = _DEFAULTS.tilt_range,
    gpu: Annotated[
        bool,
        typer.Option('--gpu', help='Run processing using first available GPU (requires icepick\'s gpu extra).'),
    ] = False,
    stamp_priors: Annotated[
        bool,
        typer.Option('--stamp-priors', help='Add a STAMP orientation prior column to icepick.csv (cytosolic: none, membrane: normal).'),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option('-v', '--verbose', count=True, help='Show progress on the console (-v) or debug detail (-vv).'),
    ] = 0,
) -> None:
    try:
        params = Params(particle_diameter=particle_diameter, tilt_range=tilt_range)
    except ValidationError as error:
        raise typer.BadParameter(str(error)) from error
    if gpu:
        try:
            require_gpu()
        except RuntimeError as error:
            raise typer.BadParameter(str(error), param_hint='--gpu') from error
        backend.use_gpu()
    output.mkdir(parents=True, exist_ok=True)
    log.setup(output, density.name, verbosity=verbose)
    sys.stderr.write('\n')
    logger.opt(colors=True).info(f'❄️  icepick <dim>(v{version("icepick")})</dim>')
    check_picks(output / 'icepick.csv', segmentation.stem, stamp_priors=stamp_priors)    # fail before the slow picking, not after
    logger.debug(f'icepick parameters: {", ".join(f'{k}={v}' for k,v in params.model_dump().items())}, gpu={gpu}, stamp_priors={stamp_priors}, output={output.resolve()}')
    t = time.perf_counter()
    c = pick_tomogram(density, segmentation, params)
    write_candidates(output / density.stem, c)
    append_picks(output / 'icepick.csv', segmentation.stem, c, stamp_priors=stamp_priors)
    log.progress(f'wrote {output / density.stem}.npy and .star, and rows for {segmentation.stem} to icepick.csv')
    counts = np.bincount(c[:, 4].astype(int), minlength=2)
    logger.info(f'found {len(c)} candidates (cytosolic {counts[0]}, membrane {counts[1]}) in {time.perf_counter() - t:.1f} s')
    sys.stderr.write('\n')

# main: console entry point
def main() -> None:
    app()
