# icepick

Template-free particle picking for cryoET workflows.

## Installation
```sh
uv tool install git+https://github.com/holsam/icepick
```

## Usage
```sh
icepick /path/to/denoised/tomogram /path/to/segmentation -o /path/to/output/directory [--particle-diameter INT] [--tilt-range [INT]]
```

### Options
Option | Default | Description
-- | -- | --
`/path/to/denoised/tomogram` | *n/a (required)* | Denoised tomogram, protein dark, voxel size in the MRC header.
`/path/to/segmentation` | *n/a (required)* | Binary membrane segmentation on the same grid.
`-o /path/to/output/directory/` | `./` (current working directory) | Output directory.
`--particle-diameter INT` | 250 | Ångstrom. Sets the working grid and every particle-scaled length. The input is binned by the whole factor that brings the voxel closest to diameter / 16 (so a particle spans about 16 working voxels), capped at 24 Å per working voxel for large particles.
`--tilt-range INT` | 60 | Maximum tilt angle in degrees (±). Sets the template elongation along z from missing wedge.
`--stamp-priors` | *n/a* | Adds a `prior` column to `icepick.csv` for `stamp pick` (cytosolic: `none`; membrane: `normal`).
`--gpu` | *n/a* | Switches processing to use first available GPU on Linux/Windows systems via `cupy`. Requires the `icepick[gpu]` optional extra to be installed.

### Outputs
Outputs are named after the denoised tomogram: `/path/to/output/directory/<name>.npy` and `/path/to/output/directory/<name>.star`.

Columns in `/path/to/output/directory/<name>.star`:
Column | Description
-- | --
`_rlnCoordinateX` | X coordinate in input-tomogram voxels
`_rlnCoordinateY` | Y coordinate in input-tomogram voxels
`_rlnCoordinateZ` | Z coordinate in input-tomogram voxels 
`_rlnAutopickFigureOfMerit` | Score/rank within stream (0-1, with higher scores preferred)
`_icepickStream` | `icepick` stream responsible for pick (0: cytosolic; 1: membrane)

Each run also appends to `/path/to/output/directory/icepick.csv` (one file for all tomograms, remove it to start afresh), which contains columns: `tomogram_id` (tomogram filename stem), `x`, `y`, `z`, `confidence`, `stream` and, with `--stamp-priors`, `prior`.

## Method
A brief description of the underlying `icepicker` method is provided below. In all steps, lengths are set in Ångstrom and scaled by particle diameter relative to 250 Å before conversion to working voxels.

### Pre-processing
Mean density and maximum segmentation values are read in z chunks and binned onto the working grid. Density is then inverted and z-scored against non-membrane voxels. Masks are applied to membranes and bright artefacts (e.g. ice): membranes are dilated 24 Å and bright artefacts (defined as densities more than 8 robust SDs from the median) are dilated 63 Å. Both masks are given zero weight, so get excluded as centres/data. Density is then band-passed (difference of Gaussians ratio 1, 4 working voxels) and the per-slice spread computed. Slices between minima either side of its peak are kept, and candidates outside this range are dropped.

### Cytosolic stream
Candidates are identified with a difference of Gaussians (ratio 1.6) at three particle-scaled scales, computed by normalised convolution. Peaks above a `z` of 3 are kept, with duplicates dropped. Each candidate is then rescored by a matched filter: a zero-mean soft ball with a negative shell, radii 78 to 125 Å, stretched along z to mimic missing-wedge elongation. The volume is filled under the mask, tiled along z with approximately half overlapping, and each tile is whitened by its smoothed periodogram before correlation to suppress the correlated background. The final score is the mean of the DoG rank and the matched-filter rank, followed by greedy thinning at 94 Å.

### Membrane stream
Voxels within 157 Å of a membrane are grouped into one-voxel distance shells. For each shell, a bare membrane profile is estimated using a wide Gaussian mean along the shell (using a width greater than that expected for a protein) and refitted after voxels well above the shell mean are removed. The residual is smoothed and converted to a z-score within each shell. Local maxima within the band are kept, inclusive of the membrane, and artefacts are excluded.

### Post-processing
Each stream then replaces its own scores with the within-stream rank, so picks can be comparable. Ranked candidates are pooled and sorted, with any candidate that has a worse score than another within 63 Å being dropped as a duplicate.
