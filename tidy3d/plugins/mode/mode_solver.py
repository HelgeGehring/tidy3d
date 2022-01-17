"""Turn Mode Specifications into Mode profiles 
"""

from typing import List
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ...components import Box
from ...components import Simulation
from ...components import ModeSpec
from ...components import ModeMonitor
from ...components import ModeSource, GaussianPulse
from ...components.types import Direction
from ...components.data import ScalarFieldData, FieldData
from ...log import SetupError

from .solver import compute_modes


"""
Stage:                Simulation     Mode Specs    Outputs       Viz           Export
                      ----------  +  ---------- -> ----------- -> ---------- -> ----------
Method:                        __init__()          .solve()       .plot()       .to_file()

td Objects:     Simulation     Mode    ->    FieldData   ->  image     -> ModeSource
                Plane           ^                             |           ModeMonitor
                Frequency       |_____________________________|
                                      iterative design


simulation = td.Simulation(...)        # define PML, gridsize, structures
plane = td.Box(...)                    # define plane that we solve modes on
freqs = td.FreqSampler(freqs=[1, 2])   # frequencies we care about
ms = ModeSolver(simulation, plane, freqs)

mode_spec = td.Mode(num_modes=3)       # solve for a number of modes to find the one we want
modes = ms.solve(mode_spec=mode_spec)  # list of ModeInfo objects for each mode
mode_index = 1                         # initial guess for the mode index
modes[mode_index].field_data.plot()    # inspect fields, do they look ok?

mon = ms.export_monitor(mode_spec=mode_spec)   # if we're happy with results, return td.ModeMonitor
src = ms.export_src(mode_spec=mode_spec,       # or as a td.ModeSource
    mode_index=mode_index,
    src_time=...)   

src.to_file('data/my_source.json')                 # this source /monitor can be saved to file
src = ModeSource.from_file('data/my_source.json')  # and loaded in our script
"""


@dataclass
class ModeInfo:
    """stores information about a (solved) mode.
    Attributes
    ----------
    field_data: xr.Dataset
        Contains information about the fields of the modal profile.
    mode: Mode
        Specifications of the mode.
    n_eff: float
        Real part of the effective refractive index of mode.
    k_eff: float
        Imaginary part of the effective refractive index of mode.
    """

    field_data: FieldData
    mode_spec: ModeSpec
    mode_index: int
    n_eff: float
    k_eff: float


class ModeSolver:
    """Interface for creating ``Mode`` objects."""

    def __init__(self, simulation: Simulation, plane: Box, freq: float):
        """Create a ``ModeSolver`` instance.

        Parameters
        ----------
        simulation : Simulation
            ``Simulation`` the ``Mode`` will be inserted into.
        plane : Box
            Plane where the mode will be computed in ``Simulation``.
        freq : float
            Frequency of mode (Hz).
        """

        self.simulation = simulation
        self.plane = plane
        self.freq = freq

        assert 0.0 in plane.size, "plane must have at least one axis with size=0"

    def solve(self, mode_spec: ModeSpec) -> List[ModeInfo]:
        """Solves for modal profile and effective index of ``Mode`` object.

        Parameters
        ----------
        mode : Mode
            ``Mode`` object containing specifications of mode.

        Returns
        -------
        ModeInfo
            Object containing mode profile and effective index data.
        """

        normal_axis = self.plane.size.index(0.0)

        # note discretizing, need to make consistent
        eps_xx = self.simulation.epsilon(self.plane, "Ex", self.freq)
        eps_yy = self.simulation.epsilon(self.plane, "Ey", self.freq)
        eps_zz = self.simulation.epsilon(self.plane, "Ez", self.freq)

        # make numpy array and get rid of normal axis
        eps_xx = np.squeeze(eps_xx.values, axis=normal_axis)
        eps_yy = np.squeeze(eps_yy.values, axis=normal_axis)
        eps_zz = np.squeeze(eps_zz.values, axis=normal_axis)

        # swap axes to waveguide coordinates (propagating in z)
        eps_wg_zz, (eps_wg_xx, eps_wg_yy) = self.plane.pop_axis(
            (eps_xx, eps_yy, eps_zz), axis=normal_axis
        )

        # note: from this point on, in waveguide coordinates (propagating in z)

        # construct eps_cross section to feed to mode solver
        eps_cross = np.stack((eps_wg_xx, eps_wg_yy, eps_wg_zz))

        # Nx, Ny = eps_cross.shape[1:]
        # if mode_spec.symmetries[0] != 0:
        #     eps_cross = np.stack(tuple(e[Nx // 2, :] for e in eps_cross))
        # if mode_spec.symmetries[1] != 0:
        #     eps_cross = np.stack(tuple(e[:, Ny // 2] for e in eps_cross))

        # note, internally discretizing, need to make consistent.
        mode_fields, n_eff_complex = compute_modes(
            eps_cross=eps_cross,
            freq=self.freq,
            grid_size=self.simulation.grid_size,
            pml_layers=mode_spec.num_pml,
            num_modes=mode_spec.num_modes,
            target_neff=mode_spec.target_neff,
            symmetries=mode_spec.symmetries,
            coords=None,
        )

        def rotate_field_coords(e_field, h_field):
            """move the propagation axis=z to the proper order in the array"""
            Ex, Ey, Ez = np.moveaxis(e_field, source=3, destination=1 + normal_axis)
            e_rot = np.stack(self.simulation.unpop_axis(Ez, (Ex, Ey), axis=normal_axis), axis=0)
            Hx, Hy, Hz = np.moveaxis(h_field, source=3, destination=1 + normal_axis)
            h_rot = np.stack(self.simulation.unpop_axis(Hz, (Hx, Hy), axis=normal_axis), axis=0)
            return (e_rot, h_rot)

        modes = []
        for mode_index in range(mode_spec.num_modes):

            # Get E and H fields at the current mode_index
            E, H = mode_fields[..., mode_index]

            # # Handle symmetries
            # if mode.symmetries[0] != 0:
            #     E_half = E[:, 1:, ...]
            #     H_half = H[:, 1:, ...]
            #     E = np.concatenate((+E_half[:, ::-1, ...], E_half), axis=1)
            #     H = np.concatenate((-H_half[:, ::-1, ...], H_half), axis=1)
            # if mode.symmetries[1] != 0:
            #     E_half = E[:, :, 1:, ...]
            #     H_half = H[:, :, 1:, ...]
            #     E = np.concatenate((+E_half[:, :, ::-1, ...], E_half), axis=2)
            #     H = np.concatenate((-H_half[:, :, ::-1, ...], H_half), axis=2)

            # Rotate back to original coordinates
            (Ex, Ey, Ez), (Hx, Hy, Hz) = rotate_field_coords(E, H)

            # apply -1 to H fields if a reflection was involved in the rotation
            if normal_axis == 1:
                Hx *= -1
                Hy *= -1
                Hz *= -1

            # note: from this point on, back in original coordinates
            fields = {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}

            # note: re-discretizing, need to make consistent.
            data_dict = {}
            for field_name, field in fields.items():
                plane_grid = self.simulation.discretize(self.plane)
                plane_coords = plane_grid[field_name]
                xyz_coords = [plane_coords.x, plane_coords.y, plane_coords.z]
                xyz_coords[normal_axis] = [self.plane.center[normal_axis]]
                data_dict[field_name] = ScalarFieldData(
                    x=xyz_coords[0],
                    y=xyz_coords[1],
                    z=xyz_coords[2],
                    f=np.array([self.freq]),
                    values=field[..., None],
                )

            mode_info = ModeInfo(
                field_data=FieldData(data_dict=data_dict).data,
                mode_spec=mode_spec,
                mode_index=mode_index,
                n_eff=n_eff_complex[mode_index].real,
                k_eff=n_eff_complex[mode_index].imag,
            )

            modes.append(mode_info)

        return modes

    # def make_source(self, mode_spec: ModeSpec, fwidth: float, direction: Direction) -> ModeSource:
    #     """Creates ``ModeSource`` from a Mode + additional specifications.

    #     Parameters
    #     ----------
    #     mode : Mode
    #         ``Mode`` object containing specifications of mode.
    #     fwidth : float
    #         Standard deviation of ``GaussianPulse`` of source (Hz).
    #     direction : Direction
    #         Whether source will inject in ``"+"`` or ``"-"`` direction relative to plane normal.

    #     Returns
    #     -------
    #     ModeSource
    #         Modal source containing specification in ``mode``.
    #     """
    #     center = self.plane.center
    #     size = self.plane.size
    #     source_time = GaussianPulse(freq0=self.freq, fwidth=fwidth)
    #     return ModeSource(
    #         center=center, size=size, source_time=source_time, mode=mode, direction=direction
    #     )

    # def make_monitor(self, mode: Mode, freqs: List[float], name: str) -> ModeMonitor:
    #     """Creates ``ModeMonitor`` from a Mode + additional specifications.

    #     Parameters
    #     ----------
    #     mode : Mode
    #         ``Mode`` object containing specifications of mode.
    #     freqs : List[float]
    #         Frequencies to include in Monitor (Hz).
    #     name : str
    #         Required name of monitor.
    #     Returns
    #     -------
    #     ModeMonitor
    #         Monitor that measures ``Mode`` on ``plane`` at ``freqs``.
    #     """
    #     center = self.plane.center
    #     size = self.plane.size
    #     return ModeMonitor(center=center, size=size, freqs=freqs, modes=[mode], name=name)
