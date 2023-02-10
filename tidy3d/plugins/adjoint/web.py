"""Adjoint-specific webapi."""
from typing import Tuple
from functools import partial

import numpy as np
from jax import custom_vjp

from ...components.simulation import Simulation
from ...components.monitor import ModeMonitor, FieldMonitor, PermittivityMonitor
from ...components.data.monitor_data import PermittivityData, FieldData
from ...components.data.data_array import ScalarFieldDataArray, ModeIndexDataArray
from ...components.data.sim_data import SimulationData
from ...web.webapi import run as web_run
from ...log import log

from .components.simulation import JaxSimulation
from .components.data.data_array import JaxDataArray
from .components.data.monitor_data import JaxModeData
from .components.data.sim_data import JaxSimulationData


def _task_name_fwd(task_name: str) -> str:
    """task name for forward run as a function of the original task name."""
    return task_name + "_fwd"


def _task_name_adj(task_name: str) -> str:
    """task name for adjoint run as a function of the original task name."""
    return task_name + "_adj"


def tidy3d_run_fn(simulation: Simulation, task_name: str, **kwargs) -> SimulationData:
    """Run a regular :class:`.Simulation` after conversion from jax type."""
    return web_run(simulation=simulation, task_name=task_name, **kwargs)

@partial(custom_vjp, nondiff_argnums=tuple(range(1, 5)))
def run(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str = "default",
    path: str = "simulation_data.hdf5",
    callback_url: str = None,
) -> JaxSimulationData:
    """Mocking original web.run function, using regular tidy3d components."""

    # convert to regular tidy3d (and accounting info)
    sim_tidy3d, jax_info = simulation.to_simulation()

    # run using regular tidy3d simulation running fn
    sim_data_tidy3d = tidy3d_run_fn(
        simulation=sim_tidy3d,
        task_name=task_name,
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
    )

    # convert back to jax type and return
    return JaxSimulationData.from_sim_data(sim_data_tidy3d, jax_info=jax_info)

# @partial(custom_vjp, nondiff_argnums=tuple(range(1, 5)))
# def run(
#     simulation: JaxSimulation,
#     task_name: str,
#     folder_name: str = "default",
#     path: str = "simulation_data.hdf5",
#     callback_url: str = None,
# ) -> JaxSimulationData:
#     """Mocking original web.run function. """

#     from scipy.ndimage.filters import gaussian_filter

#     def make_data(coords: dict, data_array_type: type, is_complex: bool = False) -> "data_type":
#         """make a random DataArray out of supplied coordinates and data_type."""
#         data_shape = [len(val) for val in coords.values()]
#         data = np.random.random(data_shape)
#         data = (1 + 1j) * data if is_complex else data
#         data = gaussian_filter(data, sigma=0.5)  # smooth out the data a little so it isnt random
#         try:
#             # regular data array
#             data_array = data_array_type(data, coords=coords)
#         except:
#             # jax data array
#             data_array = data_array_type(values=data, coords=coords)
#         return data_array

#     def make_field_data(monitor: FieldMonitor) -> FieldData:
#         """make a random FieldData from a FieldMonitor."""
#         field_cmps = {}
#         coords = {}
#         grid = simulation.discretize(monitor, extend=True)

#         for field_name in monitor.fields:
#             spatial_coords_dict = grid[field_name].dict()

#             for axis, dim in enumerate("xyz"):
#                 if monitor.size[axis] == 0:
#                     coords[dim] = [monitor.center[axis]]
#                 else:
#                     coords[dim] = np.array(spatial_coords_dict[dim])

#             coords["f"] = list(monitor.freqs)
#             field_cmps[field_name] = make_data(
#                 coords=coords, data_array_type=ScalarFieldDataArray, is_complex=True
#             )

#         return FieldData(
#             monitor=monitor,
#             symmetry=simulation.symmetry,
#             symmetry_center=simulation.center,
#             grid_expanded=simulation.discretize(monitor, extend=True),
#             **field_cmps
#         )

#     def make_eps_data(monitor: PermittivityMonitor) -> PermittivityData:
#         """make a random PermittivityData from a PermittivityMonitor."""
#         field_mnt = FieldMonitor(**monitor.dict(exclude={"type", "fields"}))
#         field_data = make_field_data(monitor=field_mnt)
#         return PermittivityData(
#             monitor=monitor, eps_xx=field_data.Ex, eps_yy=field_data.Ey, eps_zz=field_data.Ez
#         )

#     def make_mode_data(monitor: ModeMonitor) -> JaxModeData:
#         """make a random JaxModeData from a ModeMonitor."""
#         mode_indices = np.arange(monitor.mode_spec.num_modes)
#         coords_ind = {
#             "f": list(monitor.freqs),
#             "mode_index": np.arange(monitor.mode_spec.num_modes).tolist(),
#         }
#         n_complex = make_data(
#             coords=coords_ind, data_array_type=ModeIndexDataArray, is_complex=True
#         )
#         coords_amps = dict(direction=["+", "-"])
#         coords_amps.update(coords_ind)
#         amps = make_data(coords=coords_amps, data_array_type=JaxDataArray, is_complex=True)
#         return JaxModeData(monitor=monitor, n_complex=n_complex, amps=amps)

#     MONITOR_MAKER_MAP = {
#         FieldMonitor: make_field_data,
#         ModeMonitor: make_mode_data,
#         PermittivityMonitor: make_eps_data,
#     }

#     data = [MONITOR_MAKER_MAP[type(mnt)](mnt) for mnt in simulation.monitors]
#     output_data = [MONITOR_MAKER_MAP[type(mnt)](mnt) for mnt in simulation.output_monitors]

#     return JaxSimulationData(simulation=simulation, data=data, output_data=output_data)


def run_fwd(
    simulation: JaxSimulation,
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
) -> Tuple[JaxSimulationData, tuple]:
    """Run forward pass and stash extra objects for the backwards pass."""

    # add the gradient monitors and run the forward simulation
    grad_mnts = simulation.get_grad_monitors()
    sim_fwd = simulation.updated_copy(**grad_mnts)
    sim_data_fwd = run(
        simulation=sim_fwd,
        task_name=_task_name_fwd(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
    )

    # remove the gradient data from the returned version (not needed)
    sim_data_orig = sim_data_fwd.copy(update=dict(grad_data=(), simulation=simulation))
    return sim_data_orig, (sim_data_fwd,)


# pylint:disable=too-many-arguments
def run_bwd(
    task_name: str,
    folder_name: str,
    path: str,
    callback_url: str,
    res: tuple,
    sim_data_vjp: JaxSimulationData,
) -> Tuple[JaxSimulation]:
    """Run backward pass and return simulation storing vjp of the objective w.r.t. the sim."""

    # grab the forward simulation and its gradient monitor data
    (sim_data_fwd,) = res
    grad_data_fwd = sim_data_fwd.grad_data
    grad_eps_data_fwd = sim_data_fwd.grad_eps_data

    # make and run adjoint simulation
    fwidth_adj = sim_data_fwd.simulation._fwidth_adjoint  # pylint:disable=protected-access
    sim_adj = sim_data_vjp.make_adjoint_simulation(fwidth=fwidth_adj)
    sim_data_adj = run(
        simulation=sim_adj,
        task_name=_task_name_adj(task_name),
        folder_name=folder_name,
        path=path,
        callback_url=callback_url,
    )
    grad_data_adj = sim_data_adj.grad_data

    # get gradient and insert into the resulting simulation structure medium
    sim_vjp = sim_data_vjp.simulation.store_vjp(grad_data_fwd, grad_data_adj, grad_eps_data_fwd)

    return (sim_vjp,)


# register the custom forward and backward functions
run.defvjp(run_fwd, run_bwd)
