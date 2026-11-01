'''
icepick: parameters (in physical units) and working-grid values
'''

# Import external dependencies
import math
from dataclasses import dataclass
from pydantic import BaseModel, Field

# _REFERENCE: particle diameter (Angstrom) the particle-scaled lengths below were tuned for, a ribosome
_REFERENCE: float = 250.0

# _VOXELS: a particle spans about this many working voxels, so the grid follows the particle
_VOXELS: float = 16.0

# _COARSEST: working voxel size cap (Angstrom), the membrane dilation and noise smoothing lengths need at least about a voxel
_COARSEST: float = 24.0

# _DILATE: membrane mask dilation (Angstrom), voxels inside are excluded as centres and as data
_DILATE: float = 24.0

# _ARTEFACT: bright artefacts (ice, reconstruction specks) lie this many robust SD from the median, dilated by the given length (Angstrom)
_ARTEFACT: tuple[float, float] = (8.0, 63.0)

# _SMOOTH: gaussian sigma (Angstrom) applied to the membrane residual before peak finding, set by the noise not the particle
_SMOOTH: float = 15.68

# _TILE: z extent (Angstrom) of one whitening tile core, set by how fast the noise spectrum changes along z
_TILE: float = 501.76

# _SIGMAS: DoG blob scales (Angstrom, particle-scaled), smaller scales were noise-dominated
_SIGMAS: tuple[float, ...] = (31.36, 47.04, 70.56)

# _SUPPRESS: DoG duplicate removal radius is slope * sigma + offset (offset in Angstrom, particle-scaled)
_SUPPRESS: tuple[float, float] = (2.2, 23.52)

# _RADII: matched-filter template radii (Angstrom, particle-scaled), the score is the max over radii
_RADII: tuple[float, ...] = (78.4, 94.08, 109.76, 125.44)

# _SHELL: negative shell around the template core, (gap in Angstrom, particle-scaled; weight relative to the core)
_SHELL: tuple[float, float] = (47.04, 0.5)

# _BOX: template half-size (Angstrom, particle-scaled)
_BOX: float = 188.16

# _FILL: gaussian sigma (Angstrom, particle-scaled) of the local mean that replaces membrane voxels before filtering
_FILL: float = 62.72

# _THIN: final cytosolic duplicate removal radius (Angstrom, particle-scaled)
_THIN: float = 94.08

# _BAND: membrane candidates lie within this distance of a membrane (Angstrom, particle-scaled), including on it
_BAND: float = 156.8

# _ALONG: gaussian sigma (Angstrom, particle-scaled) of the bare-membrane estimate along each shell, larger than a protein
_ALONG: float = 125.44

# _PEAK: membrane local-maximum window (Angstrom, particle-scaled)
_PEAK: float = 78.4

# _DUPLICATE: candidates from different streams closer than this (Angstrom, particle-scaled) are one particle
_DUPLICATE: float = 62.72

# Params: options of one picking run
class Params(BaseModel):
    particle_diameter: float = Field(250.0, gt=0)    # Angstrom, sets the working grid and every particle-scaled length
    tilt_range: float = Field(60.0, gt=0, le=90)    # maximum tilt in degrees (+-), sets the template elongation along z
    threshold: float = Field(3.0, gt=0)    # DoG robust z a blob must exceed

    # grid: the working grid for a tomogram of the given voxel size (Angstrom)
    def grid(self, voxel_size: float) -> 'Grid':
        target = min(self.particle_diameter / _VOXELS, _COARSEST)    # working voxel size wanted, Angstrom
        factor = max(round(target / voxel_size), 1)
        size = voxel_size * factor
        grow = self.particle_diameter / _REFERENCE / size    # Angstrom to working voxels, particle-scaled
        fixed = 1 / size    # Angstrom to working voxels, set by the physics not the particle
        return Grid(
            factor=factor,
            voxel_size=size,
            dilate=max(int(_DILATE * fixed), 1),
            artefact_sd=_ARTEFACT[0],
            artefact_dilate=max(int(_ARTEFACT[1] * fixed), 1),
            smooth=_SMOOTH * fixed,
            tile=2 * max(round(_TILE * fixed / 2), 1),    # even, tiles overlap by half
            sigmas=tuple(s * grow for s in _SIGMAS),
            suppress=(_SUPPRESS[0], _SUPPRESS[1] * grow),
            threshold=self.threshold,
            min_weight=0.35,
            radii=tuple(r * grow for r in _RADII),
            stretch=_elongation(self.tilt_range),
            shell=(_SHELL[0] * grow, _SHELL[1]),
            box=max(round(_BOX * grow), 1),
            fill=_FILL * grow,
            thin=_THIN * grow,
            band=max(round(_BAND * grow), 1),
            along=_ALONG * grow,
            refit=0.5,
            peak=max(round(_PEAK * grow), 1) | 1,    # odd window
            duplicate=_DUPLICATE * grow,
        )

# Grid: parameters in working-grid voxels for one tomogram, lengths are voxels of voxel_size Angstrom
@dataclass(frozen=True)
class Grid:
    factor: int    # binning of the input volume, multiply working coordinates by this to get input voxels
    voxel_size: float    # working voxel size, Angstrom
    dilate: int    # membrane dilation iterations
    artefact_sd: float
    artefact_dilate: int    # artefact dilation iterations
    smooth: float
    tile: int
    sigmas: tuple[float, ...]
    suppress: tuple[float, float]    # (slope, offset), radius = slope * sigma + offset
    threshold: float
    min_weight: float    # minimum fraction of a filter's support that must lie outside the membrane
    radii: tuple[float, ...]
    stretch: float
    shell: tuple[float, float]    # (gap, weight)
    box: int
    fill: float
    thin: float
    band: int
    along: float
    refit: float    # residual (shell SD) above which voxels are left out of the refitted bare profile
    peak: int
    duplicate: float

# _elongation: z elongation of a point in a tomogram with the missing wedge, sqrt of the ratio of the sampled Fourier second moments along kx and kz, 1.55 at +-60 degrees
def _elongation(tilt_range: float) -> float:
    a = math.radians(tilt_range)
    return math.sqrt((a + math.sin(a) * math.cos(a)) / (a - math.sin(a) * math.cos(a)))
