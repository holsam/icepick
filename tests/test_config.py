'''
icepick: tests for config module
'''

# Import external dependencies
import pytest
from pydantic import ValidationError

# Import internal icepick objects
from icepick.config import Params

# TestConfig: tests for config.py
class TestConfig:
    # test_expected_binning: at 7.84 A/voxel a ribosome-sized particle bins by 2
    def test_expected_binning(self) -> None:
        g = Params().grid(7.84)
        assert g.factor == 2
        assert g.voxel_size == pytest.approx(15.68, rel=1e-6)
        assert (g.dilate, g.artefact_dilate, g.tile, g.box, g.band, g.peak) == (1, 4, 32, 12, 10, 5)
        assert g.sigmas == pytest.approx((2.0, 3.0, 4.5), rel=1e-6)
        assert g.radii == pytest.approx((5.0, 6.0, 7.0, 8.0), rel=1e-6)
        assert g.suppress == pytest.approx((2.2, 1.5), rel=1e-6)
        assert (g.shell[0], g.fill, g.thin, g.along, g.smooth, g.duplicate) == pytest.approx((3.0, 4.0, 6.0, 8.0, 1.0, 4.0), rel=1e-6)

    # test_stretch_follows_tilt_range: no missing wedge at +-90 degrees, 1.55 at +-60, more for a narrower range
    def test_stretch_follows_tilt_range(self) -> None:
        stretch = {t: Params(tilt_range=t).grid(7.84).stretch for t in (40, 60, 90)}
        assert stretch[90] == pytest.approx(1.0, abs=1e-6)
        assert stretch[60] == pytest.approx(1.5524, abs=1e-4)
        assert stretch[40] > stretch[60] > stretch[90]

    # test_binning_follows_particle_diameter: smaller particles keep a finer grid, larger ones bin harder, never below the input grid
    def test_binning_follows_particle_diameter(self) -> None:
        assert Params(particle_diameter=100).grid(7.84).factor == 1
        assert Params(particle_diameter=250).grid(2.0).factor == 8
        assert Params(particle_diameter=300).grid(7.84).factor == 2

    # test_working_voxel_is_capped_for_large_particles: beyond 384 A the grid stops coarsening so fixed-length steps stay at least a voxel
    def test_working_voxel_is_capped_for_large_particles(self) -> None:
        g = Params(particle_diameter=1000).grid(7.84)
        assert g.factor == 3 and g.voxel_size == pytest.approx(23.52, rel=1e-6)
        assert g.smooth == pytest.approx(15.68 / 23.52, rel=1e-6)

    # test_particle_lengths_scale_with_diameter: at 300 A the grid is unchanged, so particle-scaled lengths grow by 1.2 voxel for voxel
    def test_particle_lengths_scale_with_diameter(self) -> None:
        g = Params(particle_diameter=300).grid(7.84)
        assert g.sigmas == pytest.approx((2.4, 3.6, 5.4), rel=1e-6)
        assert (g.box, g.band, g.peak) == (14, 12, 7)
        assert g.tile == 32    # set by the noise, not the particle

    # test_params_reject_non_positive_values: bad options fail at the boundary
    def test_params_reject_non_positive_values(self) -> None:
        for bad in ({'particle_diameter': 0}, {'tilt_range': 0}, {'tilt_range': 95}):
            with pytest.raises(ValidationError):
                Params(**bad)
