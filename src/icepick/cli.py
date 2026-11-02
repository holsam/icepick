'''
icepick: command line entry point
'''

# Import external dependencies
import numpy as np, typer
from pathlib import Path
from pydantic import ValidationError
from typing import Annotated

# Import internal icepick objects
from icepick.config import Params
from icepick.merge import pick_tomogram
from icepick.utils.io import write_candidates

# app: initialise Typer instance
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help='Template-free particle picking for cryoET workflows.',
)

# _DEFAULTS: parameter defaults, shown in the help text
_DEFAULTS: Params = Params()

# pick: pick one tomogram and write <density name>.npy and .star
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
) -> None:
    try:
        params = Params(particle_diameter=particle_diameter, tilt_range=tilt_range)
    except ValidationError as error:
        raise typer.BadParameter(str(error)) from error
    c = pick_tomogram(density, segmentation, params)
    output.mkdir(parents=True, exist_ok=True)
    write_candidates(output / density.stem, c)
    counts = np.bincount(c[:, 4].astype(int), minlength=2)
    typer.echo(f'{density.stem} {len(c)} candidates (cytosolic {counts[0]}, membrane {counts[1]})')

# main: console entry point
def main() -> None:
    app()
