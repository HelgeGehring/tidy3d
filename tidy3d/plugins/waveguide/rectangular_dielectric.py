"""Rectangular dielectric waveguide utilities"""

import numpy
import pydantic

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.boundary import BoundarySpec, Periodic
from ...components.geometry import Box, PolySlab
from ...components.grid.grid_spec import GridSpec
from ...components.medium import Medium, MediumType
from ...components.mode import ModeSpec
from ...components.simulation import Simulation
from ...components.source import ModeSource, GaussianPulse
from ...components.structure import Structure
from ...components.types import ArrayLike, Axis, Coordinate, Size1D, Union
from ...constants import C_0, inf, MICROMETER, RADIAN

from ...log import log, ValidationError

from ..mode.mode_solver import ModeSolver


# TODO: consider bend_radius in mode_spec


class RectangularDielectric(Tidy3dBaseModel):
    """General rectangular dielectric waveguide

    Supports:
    - Strip and rib geometries
    - Angled sidewalls
    - Modes in waveguide bends
    - Surface and sidewall loss models
    - Coupled waveguides
    """

    wavelength: Union[float, ArrayLike[float, 1]] = pydantic.Field(
        ...,
        description="Wavelength(s) at which to calculate modes (in μm).",
        units=MICROMETER,
    )

    core_width: Union[float, ArrayLike[float, 1]] = pydantic.Field(
        ...,
        description="Core width at the top of the waveguide.  If set to an array, defines "
        "the widths of adjacent waveguides.",
        units=MICROMETER,
    )

    core_thickness: float = pydantic.Field(
        ...,
        description="Thickness of the core layer.",
        units=MICROMETER,
    )

    core_medium: MediumType = pydantic.Field(
        ...,
        description="Medium associated with the core layer.",
    )

    clad_medium: MediumType = pydantic.Field(
        ...,
        description="Medium associated with the upper cladding layer.",
    )

    box_medium: MediumType = pydantic.Field(
        None,
        description="Medium associated with the lower cladding layer.",
    )

    slab_thickness: float = pydantic.Field(
        0.0,
        description="Thickness of the slab for rib geometry.",
        units=MICROMETER,
    )

    clad_thickness: float = pydantic.Field(
        None,
        description="Domain size above the core layer.",
        units=MICROMETER,
    )

    box_thickness: float = pydantic.Field(
        None,
        description="Domain size below the core layer.",
        units=MICROMETER,
    )

    side_margin: float = pydantic.Field(
        None,
        description="Domain size to the sides of the waveguide core.",
        units=MICROMETER,
    )

    sidewall_angle: float = pydantic.Field(
        0.0,
        description="Angle of the core sidewalls measured from the vertical direction (in "
        "radians).  Positive (negative) values create waveguides with bases wider (narrower) "
        "than their tops.",
        units=RADIAN,
    )

    gap: Union[float, ArrayLike[float, 1]] = pydantic.Field(
        0.0,
        description="Distance between adjacent waveguides, measured at the top core edges.  "
        "An array can be used to define one gap per pair of adjacent waveguides.",
        units=MICROMETER,
    )

    sidewall_thickness: float = pydantic.Field(
        0.0,
        description="Sidewall layer thickness (within core).",
        units=MICROMETER,
    )

    sidewall_medium: MediumType = pydantic.Field(
        None,
        description="Medium associated with the sidewall layer to model sidewall losses.",
    )

    surface_thickness: float = pydantic.Field(
        0.0,
        description="Thickness of the surface layers defined on the top of the waveguide and  "
        "slab regions (if any).",
        units=MICROMETER,
    )

    surface_medium: MediumType = pydantic.Field(
        None,
        description="Medium associated with the surface layer to model surface losses.",
    )

    origin: Coordinate = pydantic.Field(
        (0, 0, 0),
        description="Center of the waveguide geometry.  This coordinate represents the base "
        "of the waveguides (substrate surface) in the normal axis, and center of the geometry "
        "in the remaining axes.",
        units=MICROMETER,
    )

    length: Size1D = pydantic.Field(
        1e30,
        description="Length of the waveguides in the propagation direction",
        units=MICROMETER,
    )

    propagation_axis: Axis = pydantic.Field(
        0,
        description="Axis of propagation of the waveguide",
    )

    normal_axis: Axis = pydantic.Field(
        2,
        description="Axis normal to the substrate surface",
    )

    mode_spec: ModeSpec = pydantic.Field(
        ModeSpec(),
        description=":class:`ModeSpec` defining waveguide mode properties.",
    )

    grid_resolution: int = pydantic.Field(
        15,
        description="Solver grid resolution per wavelength.",
    )

    max_grid_scaling: float = pydantic.Field(
        1.2,
        description="Maximal size increase between adjacent grid boundaries.",
    )

    @pydantic.validator("wavelength", "core_width", "gap", always=True)
    def _set_array(cls, val):
        if isinstance(val, float):
            return numpy.array((val,))
        return numpy.array(val)

    @pydantic.validator("box_medium", always=True)
    def _set_box_medium(cls, val, values):
        return values["clad_medium"] if val is None else val

    @pydantic.validator("clad_thickness", always=True)
    def _set_clad_thickness(cls, val, values):
        if val is None:
            wavelength = values["wavelength"]
            medium = values["clad_medium"]
            n = numpy.array([medium.nk_model(f)[0] for f in C_0 / wavelength])
            lda = wavelength / n
            return 1.5 * lda.max()
        return val

    @pydantic.validator("box_thickness", always=True)
    def _set_box_thickness(cls, val, values):
        if val is None:
            wavelength = values["wavelength"]
            medium = values["box_medium"]
            n = numpy.array([medium.nk_model(f)[0] for f in C_0 / wavelength])
            lda = wavelength / n
            return 1.5 * lda.max()
        return val

    @pydantic.validator("side_margin", always=True)
    def _set_side_thickness(cls, val, values):
        return max(values["clad_thickness"], values["box_thickness"]) if val is None else val

    @pydantic.validator("gap", always=True)
    def _validate_gaps(cls, val, values):
        if val.size == 1 and values["core_width"].size != 2:
            return numpy.array([val[0]] * (values["core_width"].size - 1))
        if val.size != values["core_width"].size - 1:
            raise ValidationError("Number of gaps must be 1 less than number of core widths.")
        return val

    @pydantic.root_validator
    def _ensure_consistency(cls, values):
        if values["sidewall_thickness"] > 0 and values["sidewall_medium"] is None:
            raise ValidationError(
                "Sidewall medium must be provided when sidewall thickness is greater than 0."
            )

        if values["sidewall_thickness"] == 0 and values["sidewall_medium"] is not None:
            log.warning("Sidewall medium not used because sidewall thickness is zero.")

        if values["surface_thickness"] > 0 and values["surface_medium"] is None:
            raise ValidationError(
                "Surface medium must be provided when surface thickness is greater than 0."
            )

        if values["surface_thickness"] == 0 and values["surface_medium"] is not None:
            log.warning("Surface medium not used because surface thickness is zero.")

        if values["propagation_axis"] == values["normal_axis"]:
            raise ValidationError("Propagation and normal axes must be different.")

        return values

    @cached_property
    def lateral_axis(self):
        """Lateral direction axis"""
        return 3 - self.propagation_axis - self.normal_axis

    def _transform(self, lateral_coordinate, normal_coordinate, propagation_coordinate):
        """Swap the model coordinates to desired axes and translate to origin"""
        result = list(self.origin)
        result[self.lateral_axis] += lateral_coordinate
        result[self.propagation_axis] += propagation_coordinate
        result[self.normal_axis] += normal_coordinate
        return result

    def _transform_in_plane(self, lateral_coordinate, propagation_coordinate):
        """Swap the model coordinates to desired axes in the substrate plane"""
        result = self._transform(lateral_coordinate, 0, propagation_coordinate)
        result.pop(self.normal_axis)
        return result

    @cached_property
    def height(self):
        """Domain height (size in the normal direction)"""
        return self.box_thickness + self.core_thickness + self.clad_thickness

    @cached_property
    def width(self):
        """Domain width (size in the lateral direction)"""
        w = self.core_width.sum() + self.gap.sum() + 2 * self.side_margin
        if self.sidewall_angle > 0:
            w += 2 * self.core_thickness * numpy.tan(self.sidewall_angle)
        return w

    # pylint:disable=too-many-locals,too-many-statements
    @cached_property
    def _structures_and_gridspec(self):
        """Build waveguide structure and custom grid_spec for mode solving"""

        # Domain size

        freqs = C_0 / self.wavelength
        nk_core = numpy.array([self.core_medium.nk_model(f) for f in freqs])
        nk_clad = numpy.array([self.clad_medium.nk_model(f) for f in freqs])
        nk_box = numpy.array([self.box_medium.nk_model(f) for f in freqs])
        lda_core = self.wavelength / nk_core[:, 0]
        lda_clad = self.wavelength / nk_clad[:, 0]
        lda_box = self.wavelength / nk_box[:, 0]

        # Create a local copy of these values, as they will be modified
        # according to the desired geometry
        core_w = numpy.array(self.core_width, copy=True)
        core_t = self.core_thickness
        slab_t = self.slab_thickness

        half_length = 0.5 * self.length

        normal_origin = self.origin[self.normal_axis]

        # Starting positions of each waveguide (x is the position in the lateral direction)
        core_x = [-0.5 * (self.core_width.sum() + self.gap.sum())]
        core_x.extend(core_x[0] + numpy.cumsum(self.core_width[:-1]) + numpy.cumsum(self.gap))

        # Grid resolution factor applied at the core edges
        edge_factor = 2
        i = numpy.argmin(lda_core)
        hi_index = Medium.from_nk(n=nk_core[i, 0] * edge_factor, k=0, freq=freqs[i])
        lo_index = self.box_medium if lda_box.min() < lda_clad.min() else self.clad_medium

        # Gather all waveguide edge intervals into `hi_res` list
        if self.sidewall_angle > 0:
            dx = (core_t - slab_t) * numpy.tan(self.sidewall_angle)
            hi_res = [
                pair for x, w in zip(core_x, core_w) for pair in [[x - dx, x], [x + w, x + w + dx]]
            ]
        elif self.sidewall_angle < 0:
            dx = (core_t - slab_t) * numpy.tan(self.sidewall_angle)
            hi_res = [
                pair for x, w in zip(core_x, core_w) for pair in [[x, x - dx], [x + w + dx, x + w]]
            ]
        else:
            dx = lda_core.max() / (edge_factor * self.grid_resolution)
            hi_res = [
                pair
                for x, w in zip(core_x, core_w)
                for pair in [[x - dx, x + dx], [x + w - dx, x + w + dx]]
            ]

        # The gaps between waveguides can be small enough to merge adjacent high
        # resolution intervals (specially with angled sidewalls), so we merge
        # intervals that overlap
        i = 0
        while i < len(hi_res) - 1:
            if hi_res[i][1] >= hi_res[i + 1][0]:
                hi_res[i][1] = hi_res.pop(i + 1)[1]
            else:
                i += 1

        # Create override structures to improve the mode solver grid.  We want
        # high resolution around all core edges, but not along the whole slab
        # in case of rib geometry.
        dy = 2 * lda_core.max() / self.grid_resolution
        override_structures = [
            Structure(
                geometry=Box(
                    center=self._transform(0.5 * (a + b), y, 0),
                    size=self._transform(b - a, dy, inf),
                ),
                medium=hi_index,
            )
            for (a, b) in hi_res
            for y in (slab_t, core_t)
        ] + [
            Structure(
                geometry=Box(
                    center=self._transform(0.5 * (a + b), 0, 0),
                    size=self._transform(b - a, inf, inf),
                ),
                medium=lo_index,
            )
            for (a, b) in ((-self.width, hi_res[0][0]), (hi_res[-1][1], self.width))
        ]

        # Set up the grid with overriding geometry
        grid_spec = GridSpec.auto(
            min_steps_per_wvl=self.grid_resolution,
            wavelength=self.wavelength.min(),
            override_structures=override_structures,
            max_scale=self.max_grid_scaling,
        )

        # Create the actual waveguide geometry
        structures = []

        # Surface and sidewall loss regions are created first, so that the core
        # can be applied on top.
        if self.surface_thickness > 0:
            structures.extend(
                Structure(
                    geometry=PolySlab(
                        vertices=(
                            self._transform_in_plane(x, -half_length),
                            self._transform_in_plane(x + w, -half_length),
                            self._transform_in_plane(x + w, half_length),
                            self._transform_in_plane(x, half_length),
                        ),
                        slab_bounds=(
                            normal_origin + core_t - self.surface_thickness,
                            normal_origin + core_t,
                        ),
                        sidewall_angle=self.sidewall_angle,
                        reference_plane="top",
                        axis=self.normal_axis,
                    ),
                    medium=self.surface_medium,
                )
                for x, w in zip(core_x, core_w)
            )

            # Add loss region over slab surface, if rib geometry
            if slab_t > 0:
                structures.append(
                    Structure(
                        geometry=Box(
                            center=self._transform(0, 0.5 * slab_t, 0),
                            size=self._transform(inf, slab_t, inf),
                        ),
                        medium=self.surface_medium,
                    )
                )

            # Correct core geometry to leave the lossy regions with their
            # specified thickness
            dx = self.surface_thickness * numpy.tan(self.sidewall_angle)
            core_x -= dx
            core_w += 2 * dx
            core_t -= self.surface_thickness
            slab_t = max(0, slab_t - self.surface_thickness)

        if self.sidewall_thickness > 0:
            structures.extend(
                Structure(
                    geometry=PolySlab(
                        vertices=(
                            self._transform_in_plane(x, -half_length),
                            self._transform_in_plane(x + w, -half_length),
                            self._transform_in_plane(x + w, half_length),
                            self._transform_in_plane(x, half_length),
                        ),
                        slab_bounds=(normal_origin, normal_origin + core_t),
                        sidewall_angle=self.sidewall_angle,
                        reference_plane="top",
                        axis=self.normal_axis,
                    ),
                    medium=self.sidewall_medium,
                )
                for x, w in zip(core_x, core_w)
            )

            # Core position offset and width reduction to accommodate lossy
            # regions
            dx = self.sidewall_thickness / numpy.cos(self.sidewall_angle)
            core_x += dx
            core_w -= 2 * dx

        # Waveguide cores
        structures.extend(
            Structure(
                geometry=PolySlab(
                    vertices=(
                        self._transform_in_plane(x, -half_length),
                        self._transform_in_plane(x + w, -half_length),
                        self._transform_in_plane(x + w, half_length),
                        self._transform_in_plane(x, half_length),
                    ),
                    slab_bounds=(normal_origin, normal_origin + core_t),
                    sidewall_angle=self.sidewall_angle,
                    reference_plane="top",
                    axis=self.normal_axis,
                ),
                medium=self.core_medium,
            )
            for x, w in zip(core_x, core_w)
        )

        # Slab for rib geometry
        if slab_t > 0:
            structures.append(
                Structure(
                    geometry=Box(
                        center=self._transform(0, 0.5 * slab_t, 0),
                        size=self._transform(inf, slab_t, inf),
                    ),
                    medium=self.core_medium,
                )
            )

        # Lower cladding
        if self.box_medium != self.clad_medium:
            structures.append(
                Structure(
                    geometry=Box(
                        center=self._transform(0, -self.box_thickness, 0),
                        size=self._transform(inf, 2 * self.box_thickness, inf),
                    ),
                    medium=self.box_medium,
                )
            )

        return (structures, grid_spec)

    @property
    def structures(self):
        """Waveguide structures for simulation, including the core(s), slabs (if any), and bottom
        cladding, if different from the top."""

        return self._structures_and_gridspec[0]

    @cached_property
    def mode_solver(self):
        """Create a mode solver based on this waveguide structure

        Returns
        -------
        :class:`ModeSolver`

        Example
        -------
        >>> wg = waveguide.RectangularDielectric(
        ...     wavelength=1.55,
        ...     core_width=0.5,
        ...     core_thickness=0.22,
        ...     core_medium=Medium(permittivity=3.48**2),
        ...     clad_medium=Medium(permittivity=1.45**2),
        ...     num_modes=2,
        ... )
        >>> mode_data = wg.mode_solver.solve()
        >>> mode_data.n_eff.values
        array([[2.4536054 1.7850305]], dtype=float32)

        """
        freqs = C_0 / self.wavelength
        structures, grid_spec = self._structures_and_gridspec

        plane = Box(center=self._transform(0, 0, 0), size=self._transform(inf, inf, 0))

        # Source used only to silence warnings
        mode_source = ModeSource(
            center=plane.center,
            size=plane.size,
            source_time=GaussianPulse(freq0=freqs[0], fwidth=freqs[0] / 10),
            direction="+",
            mode_spec=self.mode_spec,
        )

        simulation = Simulation(
            center=self._transform(0, 0.5 * self.height - self.box_thickness, 0),
            size=self._transform(self.width, self.height, 0),
            medium=self.clad_medium,
            structures=structures,
            boundary_spec=BoundarySpec.all_sides(Periodic()),
            grid_spec=grid_spec,
            sources=[mode_source],
            run_time=1e-12,
        )

        return ModeSolver(simulation=simulation, plane=plane, mode_spec=self.mode_spec, freqs=freqs)
