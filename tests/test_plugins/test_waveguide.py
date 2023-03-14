import pytest
import numpy as np
import tidy3d as td
from tidy3d.plugins import waveguide
from tidy3d.log import ValidationError


def test_rectangular_dielectric_validations():
    """Rectangular dielectric waveguide validations"""
    with pytest.raises(ValidationError, match=r".* gaps .*"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=(0.5, 0.5),
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
            gap=(0.1, 0.1),
        )

    with pytest.raises(ValidationError, match=r".* sidewall thickness .*"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.5,
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
            sidewall_thickness=0.01,
        )

    with pytest.raises(ValidationError, match=r".* surface thickness .*"):
        waveguide.RectangularDielectric(
            wavelength=1.55,
            core_width=0.5,
            core_thickness=0.22,
            core_medium=td.Medium(permittivity=3.48**2),
            clad_medium=td.Medium(permittivity=1.45**2),
            surface_thickness=0.01,
        )


def test_rectangular_dielectric_strip():
    """Rectangular dielectric strip waveguide"""
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=0.5,
        core_thickness=0.22,
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=td.Medium(permittivity=1.45**2),
        mode_spec=td.ModeSpec(num_modes=2),
    )
    assert np.allclose(wg.n_eff.values, [2.4536054, 1.7850305])


def test_rectangular_dielectric_rib():
    """Rectangular dielectric rib waveguide"""
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=0.45,
        core_thickness=0.22,
        slab_thickness=0.07,
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=td.Medium(permittivity=1.0),
        box_medium=td.Medium(permittivity=1.45**2),
        sidewall_angle=0.5,
        mode_spec=td.ModeSpec(num_modes=1),
    )
    assert np.allclose(wg.n_eff.values, [2.5560536])


def test_rectangular_dielectric_coupled():
    """Rectangular dielectric coupled waveguides"""
    wg = waveguide.RectangularDielectric(
        wavelength=1.55,
        core_width=[0.38, 0.38],
        core_thickness=0.22,
        core_medium=td.Medium(permittivity=3.48**2),
        clad_medium=td.Medium(permittivity=1.45**2),
        sidewall_angle=0.2,
        gap=0.1,
        mode_spec=td.ModeSpec(num_modes=4),
    )
    assert np.allclose(wg.n_eff.values, [2.4453077, 2.2707212, 1.8694501, 1.5907708])
