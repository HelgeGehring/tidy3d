"""Rectangular dielectric waveguide utilities"""

import numpy
import pydantic

from ...components.base import Tidy3dBaseModel, cached_property
from ...components.boundary import BoundarySpec, Periodic
from ...components.data.data_array import ModeIndexDataArray, FreqModeDataArray, FreqDataArray
from ...components.geometry import Box, PolySlab
from ...components.grid.grid_spec import GridSpec
from ...components.medium import Medium, MediumType
from ...components.mode import ModeSpec
from ...components.simulation import Simulation
from ...components.source import ModeSource, GaussianPulse
from ...components.structure import Structure
from ...components.types import ArrayLike, Axis, Coordinate, Size1D, Union
from ...constants import C_0, inf, MICROMETER, RADIAN

from ...log import log, Tidy3dError, ValidationError

from ..mode.mode_solver import ModeSolver


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
        title="Wavelength",
        description="Wavelength(s) at which to calculate modes (in μm).",
        units=MICROMETER,
    )

    core_width: Union[Size1D, ArrayLike[Size1D, 1]] = pydantic.Field(
        ...,
        title="Core width",
        description="Core width at the top of the waveguide.  If set to an array, defines "
        "the widths of adjacent waveguides.",
        units=MICROMETER,
    )

    core_thickness: Size1D = pydantic.Field(
        ...,
        title="Core Thickness",
        description="Thickness of the core layer.",
        units=MICROMETER,
    )

    core_medium: MediumType = pydantic.Field(
        ...,
        title="Core Medium",
        description="Medium associated with the core layer.",
    )

    clad_medium: MediumType = pydantic.Field(
        ...,
        title="Clad Medium",
        description="Medium associated with the upper cladding layer.",
    )

    box_medium: MediumType = pydantic.Field(
        None,
        title="Box Medium",
        description="Medium associated with the lower cladding layer.",
    )

    slab_thickness: Size1D = pydantic.Field(
        0.0,
        title="Slab Thickness",
        description="Thickness of the slab for rib geometry.",
        units=MICROMETER,
    )

    clad_thickness: Size1D = pydantic.Field(
        None,
        title="Clad Thickness",
        description="Domain size above the core layer.",
        units=MICROMETER,
    )

    box_thickness: Size1D = pydantic.Field(
        None,
        title="Box Thickness",
        description="Domain size below the core layer.",
        units=MICROMETER,
    )

    side_margin: Size1D = pydantic.Field(
        None,
        title="Side Margin",
        description="Domain size to the sides of the waveguide core.",
        units=MICROMETER,
    )

    sidewall_angle: float = pydantic.Field(
        0.0,
        title="Sidewall Angle",
        description="Angle of the core sidewalls measured from the vertical direction (in "
        "radians).  Positive (negative) values create waveguides with bases wider (narrower) "
        "than their tops.",
        units=RADIAN,
    )

    gap: Union[float, ArrayLike[float, 1]] = pydantic.Field(
        0.0,
        title="Gap",
        description="Distance between adjacent waveguides, measured at the top core edges.  "
        "An array can be used to define one gap per pair of adjacent waveguides.",
        units=MICROMETER,
    )

    sidewall_thickness: Size1D = pydantic.Field(
        0.0,
        title="Sidewall Thickness",
        description="Sidewall layer thickness (within core).",
        units=MICROMETER,
    )

    sidewall_medium: MediumType = pydantic.Field(
        None,
        title="Sidewall medium",
        description="Medium associated with the sidewall layer to model sidewall losses.",
    )

    surface_thickness: Size1D = pydantic.Field(
        0.0,
        title="Surface Thickness",
        description="Thickness of the surface layers defined on the top of the waveguide and  "
        "slab regions (if any).",
        units=MICROMETER,
    )

    surface_medium: MediumType = pydantic.Field(
        None,
        title="Surface Medium",
        description="Medium associated with the surface layer to model surface losses.",
    )

    origin: Coordinate = pydantic.Field(
        (0, 0, 0),
        title="Origin",
        description="Center of the waveguide geometry.  This coordinate represents the base "
        "of the waveguides (substrate surface) in the normal axis, and center of the geometry "
        "in the remaining axes.",
        units=MICROMETER,
    )

    length: Size1D = pydantic.Field(
        1e30,
        title="Length",
        description="Length of the waveguides in the propagation direction",
        units=MICROMETER,
    )

    propagation_axis: Axis = pydantic.Field(
        0,
        title="Propagation Axis",
        description="Axis of propagation of the waveguide",
    )

    normal_axis: Axis = pydantic.Field(
        2,
        title="Normal Axis",
        description="Axis normal to the substrate surface",
    )

    mode_spec: ModeSpec = pydantic.Field(
        ModeSpec(),
        title="Mode Specification",
        description=":class:`ModeSpec` defining waveguide mode properties.",
    )

    grid_resolution: int = pydantic.Field(
        15,
        title="Grid Resolution",
        description="Solver grid resolution per wavelength.",
    )

    max_grid_scaling: float = pydantic.Field(
        1.2,
        title="Maximal Grid Scaling",
        description="Maximal size increase between adjacent grid boundaries.",
    )

    @pydantic.validator("wavelength", always=True)
    def _set_dataarray(cls, val):
        wavelength = numpy.array(val, ndmin=1)
        freqs = C_0 / wavelength
        return FreqDataArray(wavelength, coords={"f": freqs})

    @pydantic.validator("core_width", "gap", always=True)
    def _set_array(cls, val):
        return numpy.array(val, ndmin=1)

    @pydantic.validator("box_medium", always=True)
    def _set_box_medium(cls, val, values):
        return values["clad_medium"] if val is None else val

    @pydantic.validator("clad_thickness", always=True)
    def _set_clad_thickness(cls, val, values):
        if val is None:
            freqs = values["wavelength"].coords["f"].values
            medium = values["clad_medium"]
            n = numpy.array([medium.nk_model(f)[0] for f in freqs])
            lda = values["wavelength"].values / n
            return 1.5 * lda.max()
        return val

    @pydantic.validator("box_thickness", always=True)
    def _set_box_thickness(cls, val, values):
        if val is None:
            freqs = values["wavelength"].coords["f"].values
            medium = values["box_medium"]
            n = numpy.array([medium.nk_model(f)[0] for f in freqs])
            lda = values["wavelength"].values / n
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

    def _swizzle(self, lateral_coord, normal_coord, propagation_coord):
        """Swap the model coordinates to desired axes"""
        result = [None, None, None]
        result[self.lateral_axis] = lateral_coord
        result[self.propagation_axis] = propagation_coord
        result[self.normal_axis] = normal_coord
        return result

    def _translate(self, lateral_coord, normal_coord, propagation_coord):
        """Swap the model coordinates to desired axes and translate to origin"""
        result = [
            a + b
            for a, b in zip(
                self.origin, self._swizzle(lateral_coord, normal_coord, propagation_coord)
            )
        ]
        return result

    def _transform_in_plane(self, lateral_coord, propagation_coord):
        """Swap the model coordinates to desired axes in the substrate plane"""
        result = self._translate(lateral_coord, 0, propagation_coord)
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

    # pylint:disable=too-many-locals,too-many-statements,too-many-branches
    @cached_property
    def _structures_and_gridspec(self):
        """Build waveguide structure and custom grid_spec for mode solving"""

        freqs = self.wavelength.coords["f"].values
        nk_core = numpy.array([self.core_medium.nk_model(f) for f in freqs])
        nk_clad = numpy.array([self.clad_medium.nk_model(f) for f in freqs])
        nk_box = numpy.array([self.box_medium.nk_model(f) for f in freqs])
        lda_core = self.wavelength.values / nk_core[:, 0]
        lda_clad = self.wavelength.values / nk_clad[:, 0]
        lda_box = self.wavelength.values / nk_box[:, 0]

        # Create a local copy of these values, as they will be modified
        # according to the desired geometry
        core_w = numpy.array(self.core_width, copy=True)
        core_t = self.core_thickness
        slab_t = self.slab_thickness

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
                    center=self._translate(0.5 * (a + b), y, 0),
                    size=self._swizzle(b - a, dy, inf),
                ),
                medium=hi_index,
            )
            for (a, b) in hi_res
            for y in (slab_t, core_t)
        ] + [
            Structure(
                geometry=Box(
                    center=self._translate(0.5 * (a + b), 0, 0),
                    size=self._swizzle(b - a, inf, inf),
                ),
                medium=lo_index,
            )
            for (a, b) in ((-self.width, hi_res[0][0]), (hi_res[-1][1], self.width))
        ]

        # Set up the grid with overriding geometry
        grid_spec = GridSpec.auto(
            min_steps_per_wvl=self.grid_resolution,
            wavelength=self.wavelength.values.min(),
            override_structures=override_structures,
            max_scale=self.max_grid_scaling,
        )

        if self.mode_spec.bend_radius is None or self.mode_spec.bend_radius == 0.0:
            half_length = 0.5 * self.length

            def polyslab_vertices(x, w):
                return (
                    self._transform_in_plane(x, -half_length),
                    self._transform_in_plane(x + w, -half_length),
                    self._transform_in_plane(x + w, half_length),
                    self._transform_in_plane(x, half_length),
                )

        else:
            if (self.normal_axis > self.lateral_axis) != (self.mode_spec.bend_axis == 1):
                raise Tidy3dError(
                    "Waveguide band axis must be the substrate normal "
                    f"(mode_spec.bend_axis = {1 - self.mode_spec.bend_axis})"
                )

            bend_radius = self.mode_spec.bend_radius
            x0 = -bend_radius

            # 10 nm resolution (at center)
            num_points = 1 + int(0.5 + 1.5 * numpy.pi * abs(bend_radius) / 0.01)

            angles = numpy.linspace(-0.75 * numpy.pi, 0.75 * numpy.pi, num_points)
            if bend_radius < 0:
                angles = -angles
            sin = numpy.sin(angles)
            cos = numpy.cos(angles)

            def polyslab_vertices(x, w):
                r_in = bend_radius + x
                v_in = numpy.vstack((x0 + r_in * cos, r_in * sin)).T
                r_out = r_in + w
                v_out = numpy.vstack((x0 + r_out * cos, r_out * sin)).T
                return [self._transform_in_plane(*v) for v in list(v_out) + list(v_in[::-1])]

        # Create the actual waveguide geometry
        structures = []

        # Surface and sidewall loss regions are created first, so that the core
        # can be applied on top.
        if self.surface_thickness > 0:
            structures.extend(
                Structure(
                    geometry=PolySlab(
                        vertices=polyslab_vertices(x, w),
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
                            center=self._translate(0, 0.5 * slab_t, 0),
                            size=self._swizzle(inf, slab_t, self.length),
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
                        vertices=polyslab_vertices(x, w),
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
                    vertices=polyslab_vertices(x, w),
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
                        center=self._translate(0, 0.5 * slab_t, 0),
                        size=self._swizzle(inf, slab_t, self.length),
                    ),
                    medium=self.core_medium,
                )
            )

        # Lower cladding
        if self.box_medium != self.clad_medium:
            structures.append(
                Structure(
                    geometry=Box(
                        center=self._translate(0, -self.box_thickness, 0),
                        size=self._swizzle(inf, 2 * self.box_thickness, self.length),
                    ),
                    medium=self.box_medium,
                )
            )

        return (structures, grid_spec)

    @property
    def structures(self):
        """Waveguide structures for simulation, including the core(s), slabs (if any), and bottom
        cladding, if different from the top.  For bend modes, the structure is a 270 degree bend
        regardless of :attr:`length`."""

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
        freqs = self.wavelength.coords["f"].values
        structures, grid_spec = self._structures_and_gridspec

        plane = Box(
            center=self._translate(0, 0.5 * self.height - self.box_thickness, 0),
            size=self._swizzle(self.width, self.height, 0),
        )

        # Source used only to silence warnings
        mode_source = ModeSource(
            center=plane.center,
            size=plane.size,
            source_time=GaussianPulse(freq0=freqs[0], fwidth=freqs[0] / 10),
            direction="+",
            mode_spec=self.mode_spec,
        )

        simulation = Simulation(
            center=plane.center,
            size=plane.size,
            medium=self.clad_medium,
            structures=structures,
            boundary_spec=BoundarySpec.all_sides(Periodic()),
            grid_spec=grid_spec,
            sources=[mode_source],
            run_time=1e-12,
        )

        return ModeSolver(simulation=simulation, plane=plane, mode_spec=self.mode_spec, freqs=freqs)

    @property
    def n_eff(self) -> ModeIndexDataArray:
        """Calculate the effective index."""
        return self.mode_solver.data.n_eff

    @property
    def n_complex(self) -> ModeIndexDataArray:
        """Calculate the complex effective index."""
        return self.mode_solver.data.n_complex

    @property
    def mode_area(self) -> FreqModeDataArray:
        """Calculate the effective mode area."""
        return self.mode_solver.data.mode_area

    # NOTE (lucas): This is inaccurate due to the lacking of sub-pixel smoothing in the local mode
    # solver.
    def calculate_group_index(
        self, wavelength_step: pydantic.PositiveFloat = 0.005
    ) -> ModeIndexDataArray:
        """Calculate the group index.
    
        The group index is numerically calculated as
        $$ n_g = n_\text{eff} - \lambda \frac{\mathrm d n_\text{eff}}{\mathrm d\lambda} $$
        with a central difference algorithm used to approximate the differential.
    
        Parameters
        ----------
        wavelength_step : `pydantic.PositiveFloat`
            Wavelength step used for the numerical differentiation.
    
        Returns
        -------
        :class:`FreqModeDataArray`
            Group indices for all waveguide wavelengths and modes indices.
        """
        # We have to solve all frequencies in a single waveguide because we want to take advantage
        # of mode tracking
        wavelength = [
            x for lda in self.wavelength.values
            for x in (lda - wavelength_step, lda, lda + wavelength_step)
        ]
        wg = self.copy(update={"wavelength": wavelength})
        end = len(wavelength)
        nb = wg.n_eff.isel(f=slice(0, end, 3)).values
        n0 = wg.n_eff.isel(f=slice(1, end, 3)).values
        nf = wg.n_eff.isel(f=slice(2, end, 3)).values
        ng = numpy.array(n0 - (self.wavelength.values / (2 * wavelength_step) * (nf - nb).T).T)
        return ModeIndexDataArray(ng, coords=self.n_eff.coords)
