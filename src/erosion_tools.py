import surface_tools as surf
import statistics_tools as stat
import scatter_tools as sct
import constants as cst
import levelset_tools as lv
import export_tools as ex
import import_tools as impt
import os
import math
import numpy as np
import scipy as sp
from typing import Dict, Any
import matplotlib.pyplot as plt
import argparse


def meters_to_micrometers(quantity: Any):
    """
    Convert a quantity expressed in meters to micrometers.

    Parameters
    ----------
    quantity : Any
        Numeric scalar or array-like value(s) in meters.

    Returns
    -------
    Any
        Same type/shape as input, scaled to micrometers.
    """
    return quantity * 1e6


class Eroder:
    """
    Level-set based erosion simulator with optional multi-reflection / scattering corrections.

    This class:
      - Loads simulation inputs from a project folder.
      - Builds a 3D level-set field Phi and corresponding X/Y/Z grids.
      - Builds interpolators for time-varying atmospheric/spacecraft inputs.
      - Builds interpolators for erosion yield, reaction fraction, and energy accommodation.
      - Advances Phi in time via first-reflection and multi-reflection advection terms.
      - Optionally exports plots, history files, and geometry snapshots.

    Notes
    -----
    - Spatial grid is assumed structured and uniform in each axis (dx, dy, dz).
    - Units are mixed throughout the code: positions are typically in micrometers,
      while some intermediate tables accept speeds in m/s and then convert.
    - The level-set sign convention used here is consistent with the called
      levelset_tools routines (e.g., interior nodes selected by Phi == -1).
    """

    def __init__(self, project_folder: str):
        """
        Initialize an erosion simulation instance from a project folder.

        Steps performed:
          1) Set defaults for all configuration and cached attributes.
          2) Import inputs via import_tools.import_inputs (Inputs.txt).
          3) Validate required fields.
          4) Build the initial level-set function and the mesh grids.
          5) Cache domain extents and grid spacings.
          6) Build erosion and time-series interpolators.
          7) Initialize Ash_Shadow mask to all ones.

        Parameters
        ----------
        project_folder : str
            Path or name of the project folder containing Inputs.txt and results outputs.
        """
        self.project_folder = project_folder

        # Defaults
        # Grid resolution (voxel counts) in x, y, z.
        self.Nx = self.Ny = self.Nz = None
        # Physical domain extents in x, y, z (computed from X/Y/Z grids).
        self.Lx = self.Ly = self.Lz = None
        # Numerical integration resolution used by the PolyGaussian/scatter routines.
        self.Nint = None
        # Number of scattering reflections to include in MR term.
        self.num_reflections = None
        # Main time-step and output cadences.
        self.time_step = self.poly_update_step = self.print_step = None
        # Extra padding and depth used to construct the level-set domain.
        self.domain_depth = self.domain_padding = None
        # Stored/estimated energy accommodation coefficient (can be updated on the fly).
        self.energy_accommodation_coeff = None
        # Simulation start/end times (UTC datetimes).
        self.sim_start_utc = self.sim_end_utc = None
        # Time-series inputs and erosion table data structures (dict-like).
        self.atmospheric_data = self.spacecraft_data = self.erosion_data = None
        # Surface tilt angle and characteristic length used in the PolyGaussian setup.
        self.surface_angle = self.surface_length = None
        # Initial PolyGaussian surface description used for Phi initialization.
        self.initial_polygaussian_surface = None
        # Level-set Phi, spatial grids X/Y/Z, and ash shadow mask (same shape as Phi).
        self.X = self.Y = self.Z = self.Phi = self.Ash_Shadow = None
        # Grid spacing in each axis (computed).
        self.dx = self.dy = self.dz = None
        # Time interpolation functions (built from atmospheric/spacecraft time series).
        self.density_interp = None
        self.speed_ratio_interp = None
        self.velocity_interp = None
        self.incident_angle_1_interp = None
        self.incident_angle_2_interp = None
        self.surface_temperature_interp = None
        self.surface_density_interp = None

        # Feature toggles for plotting/exporting.
        self.plot_history = self.export_history = self.plot_poly_geometry = self.export_geometry = False

        # Import inputs
        # Expects Inputs.txt to populate attributes on this instance.
        try:
            impt.import_inputs(self, project_folder, "Inputs.txt")
        except Exception as e:
            raise ValueError(f"import_inputs failed: {e}") from e

        # Required fields
        # Minimal set required to build the grid and initial level-set.
        req = ("Nx", "Ny", "Nz", "surface_angle", "surface_length", "domain_padding", "domain_depth",
               "initial_polygaussian_surface")
        missing = [k for k in req if getattr(self, k, None) is None]
        if missing:
            raise ValueError(f"Missing required inputs: {', '.join(missing)}")

        # Cast & validate grid sizes
        # Must be at least 2 points per axis to define spacings and finite differences.
        self.Nx, self.Ny, self.Nz = map(int, (self.Nx, self.Ny, self.Nz))
        if min(self.Nx, self.Ny, self.Nz) < 2:
            raise ValueError("Nx, Ny, Nz must be ≥ 2.")

        # Build level set
        # initialise_levelset_function is expected to return (Phi, X, Y, Z) arrays.
        try:
            self.Phi, self.X, self.Y, self.Z = lv.initialise_levelset_function(
                self.initial_polygaussian_surface,
                float(self.surface_angle),
                float(self.surface_length),
                self.Nx, self.Ny, self.Nz,
                float(self.domain_padding),
                float(self.domain_depth),
                verbose=False,
            )
        except Exception as e:
            raise ValueError(f"Level-set init failed: {e}") from e

        # Shape checks
        # Ensure all returned grids match the requested discretization.
        shp = (self.Nx, self.Ny, self.Nz)
        for nm, arr in (("Phi", self.Phi), ("X", self.X), ("Y", self.Y), ("Z", self.Z)):
            if not isinstance(arr, np.ndarray) or arr.shape != shp:
                raise ValueError(f"{nm} must be ndarray with shape {shp}, got {getattr(arr, 'shape', None)}")

        # Extents & spacings
        # Compute physical extents from the mesh grids, then uniform spacings.
        self.Lx = float(self.X.max() - self.X.min())
        self.Ly = float(self.Y.max() - self.Y.min())
        self.Lz = float(self.Z.max() - self.Z.min())
        self.dx = self.Lx / (self.Nx - 1.0)
        self.dy = self.Ly / (self.Ny - 1.0)
        self.dz = self.Lz / (self.Nz - 1.0)
        if not all(np.isfinite(v) and v > 0 for v in (self.dx, self.dy, self.dz)):
            raise ValueError("Invalid grid spacings computed (non-finite or ≤ 0).")

        # Build erosion/reaction/energy accommodation interpolators from erosion_data tables.
        self.erosion_interpolator, self.reaction_interpolator, self.energy_accommodation_interpolator = (
            self.build_erosion_functions()
        )
        # Build time interpolation functions from atmospheric_data and spacecraft_data.
        self.build_data_interpolators()

        # Ash shadow mask: 1 means unshadowed; 0 means blocked by ash geometry.
        self.Ash_Shadow = np.ones((self.Nx, self.Ny, self.Nz))

    def add_ash_particle(
            self,
            position: np.ndarray,
            radius: float,
            v: np.ndarray,
            *,
            rough_rms: float = 0.1,
            corr_len: float = 0.1,
            seed: int | None = None) -> None:
        """
        Carve an ash "shadow" volume as a finite cylinder (optionally roughened) into Ash_Shadow.

        This function sets Ash_Shadow voxels to 0.0 inside a cylinder:
          - Axis starts at `position` and points along `v` (velocity direction).
          - Mean radius is `radius`.
          - Axial coordinate s is measured along the axis (projection onto vhat).
          - The intended cylinder height is 2 * radius over 0 ≤ s ≤ 2r
            (but note the upper bound is currently commented out in the mask).

        Roughness model:
          - If rough_rms > 0, the cylinder radius is perturbed as:
                r_eff = radius + δr(θ, s)
            where δr is a small random field built from a few harmonics in θ and s.
          - corr_len sets the axial correlation length scale.
          - seed enables deterministic random phases.

        Parameters
        ----------
        position : np.ndarray
            3-vector giving the cylinder start point in the same units as X/Y/Z.
        radius : float
            Mean cylinder radius.
        v : np.ndarray
            3-vector indicating the axis direction (does not need to be unit length).
        rough_rms : float, optional
            RMS roughness amplitude of radius perturbations.
        corr_len : float, optional
            Axial correlation length used to scale s for the harmonic field.
        seed : int | None, optional
            RNG seed for reproducibility of roughness phases.

        Returns
        -------
        None
            Modifies self.Ash_Shadow in place.
        """
        # Ensure required grids and shadow volume are available.
        if self.Ash_Shadow is None or self.X is None or self.Y is None or self.Z is None:
            raise ValueError("Ash_Shadow and (X,Y,Z) grids must be initialized.")

        # Normalize and sanitize inputs to fixed shapes and float dtype.
        pos = np.asarray(position, dtype=float).reshape(3)
        vel = np.asarray(v, dtype=float).reshape(3)

        # Validate radius.
        r_mean = float(radius)
        if not (np.isfinite(r_mean) and r_mean > 0.0):
            return

        # Validate axis direction and normalize.
        vmag = float(np.linalg.norm(vel))
        if not (np.isfinite(vmag) and vmag > 0.0):
            return
        vhat = vel / vmag  # axis unit vector

        # Relative coordinates from cylinder start point to every grid node.
        Rx = self.X - pos[0]
        Ry = self.Y - pos[1]
        Rz = self.Z - pos[2]

        # Projection s along axis and radial distance rho from the axis.
        s = Rx * vhat[0] + Ry * vhat[1] + Rz * vhat[2]  # axial coordinate
        rel2 = Rx * Rx + Ry * Ry + Rz * Rz
        rad2 = np.maximum(rel2 - s * s, 0.0)
        rho = np.sqrt(rad2)  # radial distance

        # Construct an orthonormal basis (e1, e2) perpendicular to the axis.
        # Used to define the azimuthal angle θ around vhat for roughness harmonics.
        def _perp_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """
            Build two orthonormal vectors perpendicular to a unit direction n.

            Parameters
            ----------
            n : np.ndarray
                Unit direction vector.

            Returns
            -------
            (e1, e2) : tuple[np.ndarray, np.ndarray]
                Orthonormal basis spanning the plane normal to n.
            """
            # Choose a vector not too aligned with n, then Gram-Schmidt it.
            a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1 = a - np.dot(a, n) * n
            e1_norm = np.linalg.norm(e1)
            # Fallbacks to avoid degeneracy if chosen a aligns with n.
            if e1_norm == 0.0:
                e1 = np.array([0.0, 0.0, 1.0]) - n[2] * n
                e1_norm = np.linalg.norm(e1)
                if e1_norm == 0.0:
                    e1 = np.array([1.0, 0.0, 0.0])
                    e2 = np.cross(n, e1)
                    e2 /= np.linalg.norm(e2)
                    return e1, e2
            e1 /= e1_norm
            e2 = np.cross(n, e1)
            e2 /= np.linalg.norm(e2)
            return e1, e2

        e1, e2 = _perp_basis(vhat)

        # Angular coordinate around the cylinder axis for every grid node.
        x1 = Rx * e1[0] + Ry * e1[1] + Rz * e1[2]
        x2 = Rx * e2[0] + Ry * e2[1] + Rz * e2[2]
        theta = np.arctan2(x2, x1)

        # Roughness δr(θ, s) with specified RMS, assembled from a few cosine modes.
        if rough_rms and rough_rms > 0.0:
            # Use corr_len as axial scale; fall back to r_mean if corr_len invalid.
            Ls = float(corr_len) if (corr_len is not None and np.isfinite(corr_len) and corr_len > 0.0) else r_mean
            xi = s / max(Ls, 1e-12)
            rng = np.random.default_rng(seed)
            # Small set of (k_theta, k_s) mode pairs with random phases.
            modes = [(1, 0.0), (2, 0.0), (0.0, 1.0), (1.0, 1.0)]
            phases = rng.uniform(0.0, 2.0 * np.pi, size=len(modes))
            raw = np.zeros_like(theta)
            for (kth, ks), phi in zip(modes, phases):
                raw += np.cos(kth * theta + ks * xi + phi)
            # Empirical normalization of mode sum to have O(1) standard deviation.
            raw /= np.sqrt(2.0)  # std ≈ 1
            std = np.std(raw[np.isfinite(raw)])
            delta_r = (rough_rms / std) * raw if (std > 0 and np.any(np.isfinite(raw))) else 0.0
            r_eff = np.maximum(1e-12, r_mean + delta_r)
        else:
            # No roughness requested: radius is constant.
            r_eff = r_mean

        # ---- Finite-cylinder mask: radius condition and axial window 0 ≤ s ≤ 2r ----
        # Note: the upper axial bound is commented out, so the mask currently enforces only s ≥ 0.
        mask = (rho <= r_eff) & (s >= 0.0) #& (s <= 2.0 * r_mean)

        # Apply shadow by zeroing those voxels.
        self.Ash_Shadow[mask] = 0.0

    def add_ash(
            self,
            incident_velocity: np.ndarray,
            median_radius: float = 2,
            radius_variance: float = 0.1,
            num_particles: int = 1000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Place ash particles uniformly inside the solid (Phi == -1) and update Ash_Shadow.

        The selected positions are grid nodes chosen uniformly at random from all interior nodes.
        Radii are drawn from a uniform distribution about `median_radius` (despite the docstring
        mentioning lognormal volume sampling).

        For each placed particle, add_ash_particle is called to carve a shadow cylinder aligned
        with the incident velocity.

        Parameters
        ----------
        incident_velocity : np.ndarray
            3-vector that sets the cylinder axis direction for all particles.
        median_radius : float
            Center value for the radius draw (same units as X/Y/Z).
        radius_variance : float
            Half-width of the uniform radius interval around median_radius.
        num_particles : int
            Number of particles to place (attempted).

        Returns
        -------
        X_positions, Y_positions, Z_positions, Radii : np.ndarray
            Arrays of particle center coordinates and radii for those placed.
        """
        # --- sanity checks ---
        # Phi and coordinate grids must exist and match.
        if any(v is None for v in (self.Phi, self.X, self.Y, self.Z)):
            raise ValueError("Phi and (X, Y, Z) grids must be initialized.")
        if not (self.Phi.shape == self.X.shape == self.Y.shape == self.Z.shape):
            raise ValueError("Phi and (X, Y, Z) must have identical shapes.")
        # Ensure the per-particle routine exists.
        if not hasattr(self, "add_ash_particle"):
            raise AttributeError("Expected method self.add_particle(position, radius, velocity).")

        N = int(num_particles)
        if N <= 0:
            return (np.array([], float),) * 4

        # --- pick uniformly from solid nodes (Φ == -1) ---
        interior = np.argwhere(self.Phi == -1)
        if interior.size == 0:
            return (np.array([], float),) * 4

        rng = np.random.default_rng()
        # Sample interior node indices with replacement.
        idx = rng.integers(0, interior.shape[0], size=N)
        ii, jj, kk = interior[idx, 0], interior[idx, 1], interior[idx, 2]

        # Map voxel indices to physical coordinates.
        X_positions = self.X[ii, jj, kk].astype(float, copy=False)
        Y_positions = self.Y[ii, jj, kk].astype(float, copy=False)
        Z_positions = self.Z[ii, jj, kk].astype(float, copy=False) #* 0 + self.Lz / 2.0

        # --- radii draw ---
        # Uniform draw around median_radius; uses numpy global RNG.
        Radii = np.random.uniform(median_radius - radius_variance, median_radius + radius_variance, N)

        # --- place particles (one-by-one call) ---
        # Each particle updates Ash_Shadow according to its cylinder definition.
        for p, r in zip(np.c_[X_positions, Y_positions, Z_positions], Radii):
            self.add_ash_particle(p, float(r), incident_velocity)

        return X_positions, Y_positions, Z_positions, Radii


    def add_ash_ontop(
            self,
            incident_velocity: np.ndarray,
            median_radius: float = 2,
            radius_variance: float = 0.1,
            num_particles: int = 1000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Place ash particles with centers constrained to a plane at z = Lz/2, then update Ash_Shadow.

        This routine is identical to add_ash except that the Z positions are forced
        to a constant value (here set to Lz/2 via a multiply-by-zero trick).

        Parameters
        ----------
        incident_velocity : np.ndarray
            3-vector that sets the cylinder axis direction for all particles.
        median_radius : float
            Center value for the radius draw (same units as X/Y/Z).
        radius_variance : float
            Half-width of the uniform radius interval around median_radius.
        num_particles : int
            Number of particles to place (attempted).

        Returns
        -------
        X_positions, Y_positions, Z_positions, Radii : np.ndarray
            Arrays of particle center coordinates and radii for those placed.
        """
        # --- sanity checks ---
        if any(v is None for v in (self.Phi, self.X, self.Y, self.Z)):
            raise ValueError("Phi and (X, Y, Z) grids must be initialized.")
        if not (self.Phi.shape == self.X.shape == self.Y.shape == self.Z.shape):
            raise ValueError("Phi and (X, Y, Z) must have identical shapes.")
        if not hasattr(self, "add_ash_particle"):
            raise AttributeError("Expected method self.add_particle(position, radius, velocity).")

        N = int(num_particles)
        if N <= 0:
            return (np.array([], float),) * 4

        # --- pick uniformly from solid nodes (Φ == -1) ---
        interior = np.argwhere(self.Phi == -1)
        if interior.size == 0:
            return (np.array([], float),) * 4

        rng = np.random.default_rng()
        idx = rng.integers(0, interior.shape[0], size=N)
        ii, jj, kk = interior[idx, 0], interior[idx, 1], interior[idx, 2]

        X_positions = self.X[ii, jj, kk].astype(float, copy=False)
        Y_positions = self.Y[ii, jj, kk].astype(float, copy=False)
        # Force all particle centers to a single z-plane (Lz/2).
        Z_positions = self.Z[ii, jj, kk].astype(float, copy=False) * 0 + self.Lz / 2.0

        # --- radii draw ---
        Radii = np.random.uniform(median_radius - radius_variance, median_radius + radius_variance, N)

        # --- place particles (one-by-one call) ---
        for p, r in zip(np.c_[X_positions, Y_positions, Z_positions], Radii):
            self.add_ash_particle(p, float(r), incident_velocity)

        return X_positions, Y_positions, Z_positions, Radii

    def sample_incident_velocity(
            self,
            incident_angle_1: float,
            incident_angle_2: float,
            spacecraft_velocity: float,
            speed_ratio: float,
    ) -> np.ndarray:
        """
        Sample an incident velocity vector that arrives toward the surface (negative along surface normal).

        The mean direction is defined by spherical angles (incident_angle_1, incident_angle_2)
        with the convention of pointing toward −z. A thermal perturbation (Gaussian) is added
        to model spread about the mean direction, controlled by speed_ratio.

        The returned vector is accepted only if it points "into" the surface according to the
        surface normal direction derived from surface_angle. This is enforced with rejection
        sampling, with a last-resort fallback adjustment.

        Parameters
        ----------
        incident_angle_1 : float
            Polar angle θ (radians).
        incident_angle_2 : float
            Azimuthal angle φ (radians).
        spacecraft_velocity : float
            Mean speed scale for the incident particles (same units as used elsewhere here).
        speed_ratio : float
            Positive scale; larger values produce smaller thermal spread.

        Returns
        -------
        np.ndarray
            A (3,) velocity vector satisfying the inward-pointing condition.
        """
        # validate
        if not np.isfinite(incident_angle_1) or not np.isfinite(incident_angle_2):
            raise ValueError("incident angles must be finite.")
        if not np.isfinite(speed_ratio) or speed_ratio <= 0.0:
            raise ValueError("speed_ratio must be a positive finite number.")
        if not np.isfinite(spacecraft_velocity) or spacecraft_velocity < 0.0:
            raise ValueError("spacecraft_velocity must be a finite, non-negative number.")

        # mean (unit) direction toward the surface (−z), scaled by spacecraft_velocity.
        s1, c1 = np.sin(incident_angle_1), np.cos(incident_angle_1)
        s2, c2 = np.sin(incident_angle_2), np.cos(incident_angle_2)
        v_dir = np.array([s1 * c2, s1 * s2, -c1], dtype=float) * spacecraft_velocity

        # Surface normal direction used to define "inward" for acceptance.
        norm = np.array([np.sin(self.surface_angle), 0.0, np.cos(self.surface_angle)])

        n = np.linalg.norm(v_dir)
        if n == 0.0:
            raise ValueError("degenerate direction from angles; cannot sample velocity.")

        # thermal spread (Gaussian perturbation std per component)
        sigma = n / float(speed_ratio) / np.sqrt(2.0)

        # Rejection sampling to enforce inward-pointing condition; robust fallback after max_tries.
        max_tries = 1000
        for _ in range(max_tries):
            thermal = np.random.standard_normal(3) * sigma
            v = v_dir + thermal
            if np.dot(v, norm) < 0.0:
                return v

        # Fallback: force z negative (keeps x,y as last sampled).
        v[2] = -abs(v[2]) if np.isfinite(v[2]) else -1e-12
        if v[2] == 0.0:  # ensure strictly negative, not just non-positive
            v[2] = -1e-12
        return v

    def build_data_interpolators(self) -> None:
        """
        Build interpolation callables for time-varying atmospheric and spacecraft parameters.

        Uses linear interpolation with extrapolation outside the data range.
        Handles duplicate time stamps by averaging associated values.

        After calling, the following attributes are set:
          - self.density_interp(t)
          - self.speed_ratio_interp(t)
          - self.velocity_interp(t)
          - self.incident_angle_1_interp(t)
          - self.incident_angle_2_interp(t)
          - self.surface_temperature_interp(t)
          - self.surface_density_interp(t)
        """

        def _const_interp(value: np.ndarray):
            """
            Create a constant "interpolator" that broadcasts a stored value over input t.

            Parameters
            ----------
            value : np.ndarray
                Value to broadcast.

            Returns
            -------
            callable
                Function f(t) returning value broadcast to t's shape.
            """
            value = np.asarray(value)

            def f(t):
                t = np.asarray(t)
                return np.broadcast_to(value, t.shape + value.shape)

            return f

        def _make_interp(times, values, name: str):
            """
            Create an interp1d time interpolator with deduplication and sanitization.

            - Drops non-finite rows.
            - Sorts by time.
            - If duplicate times exist, averages values across duplicates.
            - If only one time sample exists, returns a constant function.

            Parameters
            ----------
            times : array-like
                Time stamps.
            values : array-like
                Values aligned with times.
            name : str
                Label used for error messages.

            Returns
            -------
            callable
                Interpolator function of time.
            """
            t = np.asarray(times, float).ravel()
            v = np.asarray(values, float)
            if t.ndim != 1 or v.shape[0] != t.size:
                raise ValueError(f"{name}: time and value lengths must match (got {t.size} vs {v.shape[0]}).")

            # Drop non-finite rows
            row_ok = np.isfinite(t)
            if v.ndim == 1:
                row_ok &= np.isfinite(v)
            else:
                row_ok &= np.all(np.isfinite(v), axis=tuple(range(1, v.ndim)))
            t, v = t[row_ok], v[row_ok]
            if t.size == 0:
                raise ValueError(f"{name}: no finite samples.")

            # Sort by time (stable sort keeps deterministic ordering for duplicates).
            order = np.argsort(t, kind="stable")
            t, v = t[order], v[order]

            # Deduplicate times by averaging values
            tu, inv = np.unique(t, return_inverse=True)
            if tu.size != t.size:
                if v.ndim == 1:
                    acc = np.zeros(tu.size, float)
                    cnt = np.zeros(tu.size, int)
                    np.add.at(acc, inv, v)
                    np.add.at(cnt, inv, 1)
                    v = acc / np.maximum(cnt, 1)
                else:
                    tail_shape = v.shape[1:]
                    acc = np.zeros((tu.size,) + tail_shape, float)
                    cnt = np.zeros(tu.size, int)
                    np.add.at(acc, inv, v)
                    np.add.at(cnt, inv, 1)
                    v = acc / np.maximum(cnt, 1)[:, None if tail_shape else ...]
                t = tu

            # 1 sample → constant; otherwise interp1d along axis=0
            if t.size == 1:
                return _const_interp(v[0])

            return sp.interpolate.interp1d(
                t, v, kind="linear", axis=0,
                bounds_error=False, fill_value="extrapolate", assume_sorted=True
            )

        # ---- Atmospheric ----
        # Expected keys: time, ao_density, speed_ratio, velocity.
        a = self.atmospheric_data
        if a is None:
            raise ValueError("atmospheric_data is missing.")
        self.density_interp = _make_interp(a["time"], a["ao_density"], "density")
        self.speed_ratio_interp = _make_interp(a["time"], a["speed_ratio"], "speed_ratio")
        # Convert m/s to micrometers/s for consistency with the rest of the simulator.
        self.velocity_interp = _make_interp(a["time"], meters_to_micrometers(a["velocity"]), "velocity")

        # ---- Spacecraft ----
        # Expected keys: time, incident angles, surface temperature, surface density.
        s = self.spacecraft_data
        if s is None:
            raise ValueError("spacecraft_data is missing.")
        self.incident_angle_1_interp = _make_interp(s["time"], s["incident_angle_1"], "incident_angle_1")
        self.incident_angle_2_interp = _make_interp(s["time"], s["incident_angle_2"], "incident_angle_2")
        self.surface_temperature_interp = _make_interp(s["time"], s["surface_temperature"], "surface_temperature")
        self.surface_density_interp = _make_interp(s["time"], s["surface_density"], "surface_density")

    def build_erosion_functions(self):
        """
        Build erosion, reaction, and energy-accommodation interpolator callables.

        Returns
        -------
        (Ey_fn, R_fn, EA_fn)
            Ey_fn(T, V, A) -> erosion yield on broadcastable arrays
            R_fn(T, V, A)  -> reacted fraction on broadcastable arrays
            EA_fn(T, V, A) -> energy accommodation coefficient on broadcastable arrays

        Notes
        -----
        - Inputs are clamped to the table bounds (constant extrapolation).
        - Tables are interpolated trilinearly with RegularGridInterpolator.
        - Energy accommodation is optional; if missing, EA_fn returns a constant.
        """
        ed = self.erosion_data
        if ed is None:
            raise ValueError("erosion_data is missing.")

        def _axis_from_grid(G, axis):
            """
            Extract a 1D axis vector from a provided grid.

            Accepts either:
              - 1D axis array, returned as-is
              - 3D meshgrid-like arrays, reduced along other dimensions.

            Parameters
            ----------
            G : array-like
                Grid data.
            axis : int
                Axis index 0, 1, or 2.

            Returns
            -------
            np.ndarray
                1D axis vector.
            """
            G = np.asarray(G, float)
            if G.ndim == 1:
                return G
            if G.ndim != 3:
                raise ValueError("Grids must be 1-D or 3-D.")
            return (G[:, 0, 0] if axis == 0 else (G[0, :, 0] if axis == 1 else G[0, 0, :]))

        # Extract 1D coordinate axes.
        T = _axis_from_grid(ed["T_grid"], 0)
        V = _axis_from_grid(ed["velocity_grid"], 1)
        A = _axis_from_grid(ed["angle_grid_rad"], 2)

        # Core tabulated fields.
        Ey = np.asarray(ed["erosion_yield"], float)
        Rf = np.asarray(ed["reacted_fraction"], float)

        # Validate shapes are consistent with axes.
        if Ey.shape != (T.size, V.size, A.size) or Rf.shape != Ey.shape:
            raise ValueError("Value arrays must have shape (nT, nV, nA).")

        # Optional energy-accommodation grid.
        if "energy_accommodation" in ed:
            Ea_arr = np.asarray(ed["energy_accommodation"], float)
            if Ea_arr.shape != Ey.shape:
                raise ValueError("energy_accommodation must have shape (nT, nV, nA).")
        else:
            Ea_arr = None  # will provide a constant fallback function later

        def _sort_axis(ax, arr, axid):
            """
            Sort a 1D axis and reorder a 3D array consistently along that axis.

            Parameters
            ----------
            ax : np.ndarray
                Axis vector to sort.
            arr : np.ndarray
                3D array aligned to (T,V,A).
            axid : int
                Which axis of arr to reorder: 0, 1, or 2.

            Returns
            -------
            (ax_sorted, arr_sorted)
            """
            idx = np.argsort(ax)
            ax = ax[idx]
            arr = (arr[idx, :, :] if axid == 0 else (arr[:, idx, :] if axid == 1 else arr[:, :, idx]))
            return ax, arr

        # Sort axes; keep arrays in sync
        T, Ey = _sort_axis(T, Ey, 0)
        T, Rf = _sort_axis(T, Rf, 0)
        V, Ey = _sort_axis(V, Ey, 1)
        V, Rf = _sort_axis(V, Rf, 1)
        A, Ey = _sort_axis(A, Ey, 2)
        A, Rf = _sort_axis(A, Rf, 2)
        if Ea_arr is not None:
            T, Ea_arr = _sort_axis(T, Ea_arr, 0)
            V, Ea_arr = _sort_axis(V, Ea_arr, 1)
            A, Ea_arr = _sort_axis(A, Ea_arr, 2)

        # Ensure core values are finite before interpolation.
        if not (np.all(np.isfinite(Ey)) and np.all(np.isfinite(Rf))):
            raise ValueError("Non-finite values in erosion tables.")

        # Build interpolators in (T, V, A) space.
        ey_rgi = sp.interpolate.RegularGridInterpolator((T, V, A), Ey, fill_value=None)
        rf_rgi = sp.interpolate.RegularGridInterpolator((T, V, A), Rf, fill_value=None)
        ea_rgi = (sp.interpolate.RegularGridInterpolator((T, V, A), Ea_arr, fill_value=None)
                  if Ea_arr is not None else None)

        # Cache bounds for clamping.
        Tmin, Tmax = T[0], T[-1]
        Vmin, Vmax = V[0], V[-1]
        Amin, Amax = A[0], A[-1]

        def _eval_clamped(rgi, t, v, a):
            """
            Evaluate a RegularGridInterpolator with constant extrapolation via clamping.

            Parameters
            ----------
            rgi : RegularGridInterpolator
                Interpolator over (T,V,A).
            t, v, a : array-like
                Query points (broadcastable).

            Returns
            -------
            np.ndarray
                Interpolated values with shape equal to broadcast(t,v,a).
            """
            t, v, a = np.broadcast_arrays(t, v, a)
            pts = np.column_stack([
                np.clip(np.asarray(t, float).ravel(), Tmin, Tmax),
                np.clip(np.asarray(v, float).ravel(), Vmin, Vmax),
                np.clip(np.asarray(a, float).ravel(), Amin, Amax),
            ])
            out = rgi(pts)
            return out.reshape(t.shape)

        # Public callables; preserve the code's lambda style.
        Ey_fn = lambda Tq, Vq, Aq: _eval_clamped(ey_rgi, Tq, Vq, Aq)
        R_fn = lambda Tq, Vq, Aq: _eval_clamped(rf_rgi, Tq, Vq, Aq)

        if ea_rgi is not None:
            EA_fn = lambda Tq, Vq, Aq: _eval_clamped(ea_rgi, Tq, Vq, Aq)
        else:
            # fallback: constant function using current self.energy_accommodation_coeff or default.
            const_alpha = float(self.energy_accommodation_coeff) if (
                        self.energy_accommodation_coeff is not None) else 0.7
            EA_fn = lambda Tq, Vq, Aq, _a=const_alpha: np.broadcast_to(
                np.asarray(_a, float), np.broadcast(Tq, Vq, Aq).shape
            )

        return Ey_fn, R_fn, EA_fn

    def local_to_global_angles(
            self,
            local_angle_1: np.ndarray,
            local_angle_2: np.ndarray,
            normal_angle_1: np.ndarray,
            normal_angle_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert local scattering angles into global spherical angles.

        Local angles (θ_l, φ_l) are defined in a coordinate system where the local +z axis
        coincides with the surface normal. The surface normal itself is oriented in the
        global frame by spherical angles (θ_n, φ_n). This routine maps local directions
        into the global frame and returns the resulting global angles (θ_g, φ_g).

        Parameters
        ----------
        local_angle_1 : array-like
            Local polar angle θ_l.
        local_angle_2 : array-like
            Local azimuth angle φ_l.
        normal_angle_1 : array-like
            Normal polar angle θ_n.
        normal_angle_2 : array-like
            Normal azimuth angle φ_n.

        Returns
        -------
        (theta_g, phi_g) : tuple[np.ndarray, np.ndarray]
            Global polar and azimuth angles, broadcast to the common input shape.
        """

        # Broadcast and cast
        th_l, ph_l, th_n, ph_n = np.broadcast_arrays(
            np.asarray(local_angle_1, float),
            np.asarray(local_angle_2, float),
            np.asarray(normal_angle_1, float),
            np.asarray(normal_angle_2, float),
        )

        # Trig (vectorized)
        ctn, stn = np.cos(th_n), np.sin(th_n)
        cpn, spn = np.cos(ph_n), np.sin(ph_n)
        ctl, stl = np.cos(th_l), np.sin(th_l)
        cpl, spl = np.cos(ph_l), np.sin(ph_l)

        # Helper term used by the closed-form transformation.
        C = ctn * stl * cpl + stn * ctl

        # Polar angle (θ_g); clip for numerical safety
        cos_thg = ctn * ctl - stn * stl * cpl
        th_g = np.arccos(np.clip(cos_thg, -1.0, 1.0))

        # Azimuth (φ_g), wrapped to [0, 2π).
        num = spn * C + cpn * stl * spl
        den = cpn * C - spn * stl * spl
        ph_g = np.mod(np.arctan2(num, den), 2.0*np.pi)

        return th_g, ph_g

    def compute_energy_accommodation(
            self,
            energy_function: Any | None = None,
            incident_velocity: np.ndarray | None = None,
            surface_temperature: float | None = None,
            scatterer: sct.Scatterer | None = None,
            first: bool = True,
    ) -> float:
        """
        Compute a first-reflection mean energy accommodation coefficient α_E.

        The method:
          1) Computes the incident direction angles (θ_i, φ_i).
          2) Evaluates the surface-normals PDF conditioned on that incidence direction.
          3) Applies a cosine-visibility factor max(0, n̂·î).
          4) Converts visibility into a local incidence angle θ_loc = arccos(max(0, n̂·î)).
          5) Evaluates α_E(T, |v|, θ_loc) and averages under the weighted PDF.

        Parameters
        ----------
        energy_function : callable | None
            Function α(T, V, θ) returning accommodation in [0,1].
            If None, uses self.energy_accommodation_interpolator if available.
        incident_velocity : np.ndarray | None
            Incident velocity vector; used to determine incoming direction.
        surface_temperature : float | None
            Surface temperature [K].
        scatterer : sct.Scatterer | None
            Scatterer object with a bound PolyGaussian surface and PDF methods.
        first : bool
            Flag forwarded to scatterer.get_normals_pdf (kept for interface consistency).

        Returns
        -------
        float
            Mean α_E in [0, 1] (clipped).
        """
        # --- sanity checks ---
        # Need a scatterer with poly_surface parameters for normals PDF.
        if scatterer is None or getattr(scatterer, "poly_surface", None) is None:
            raise ValueError("scatterer with a bound PolyGaussian surface is required.")
        # Default energy function to stored interpolator if not provided.
        if energy_function is None:
            energy_function = getattr(self, "energy_accommodation_interpolator", None)
            if energy_function is None:
                # fall back to a stored scalar if available
                return float(self.energy_accommodation_coeff if self.energy_accommodation_coeff is not None else 1.0)
        if incident_velocity is None or surface_temperature is None:
            raise ValueError("incident_velocity and surface_temperature must be provided.")

        # --- incident direction (unit) and its spherical angles ---
        v = np.asarray(incident_velocity, float).ravel()
        vmag = float(np.linalg.norm(v))
        if not np.isfinite(vmag) or vmag <= 0.0:
            return 0.0
        vhat = v / vmag

        # Rays arrive along î = -v̂ (toward the surface).
        i_hat = -vhat
        theta_inc = float(np.arccos(np.clip(i_hat[2], -1.0, 1.0)))
        phi_inc = float(np.mod(np.arctan2(i_hat[1], i_hat[0]), 2.0 * np.pi))

        # --- normals grid (upper hemisphere) ---
        # Integration resolution set by scatterer.N_INT (with a minimum).
        N = int(getattr(scatterer, "N_INT", 64))
        N = max(N, 4)
        theta_n = np.linspace(0.0, np.pi / 2.0, N)
        phi_n = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        dtheta = (theta_n[-1] - theta_n[0]) / max(N - 1, 1)
        dphi = (2.0 * np.pi) / N

        THn, PHn = np.meshgrid(theta_n, phi_n, indexing="ij")  # (N,N)

        # --- normals PDF conditioned on this incidence ---
        # first=True forces the scatterer to use its first-bounce shadow/visibility conventions.
        Pn = scatterer.get_normals_pdf(
            (THn.ravel(), PHn.ravel()),
            theta_inc, phi_inc,
            getattr(scatterer.poly_surface, "sigma_coeff", None),
            getattr(scatterer.poly_surface, "mu_coeff", None),
            getattr(scatterer.poly_surface, "R", None),
            N,
            first=True
        ).reshape(THn.shape)
        Pn = np.maximum(np.nan_to_num(Pn, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # --- cosine visibility scaling: max(0, n̂·î) ---
        nx = np.sin(THn) * np.cos(PHn)
        ny = np.sin(THn) * np.sin(PHn)
        nz = np.cos(THn)
        cos_vis = nx * i_hat[0] + ny * i_hat[1] + nz * i_hat[2]
        cos_vis = np.maximum(np.nan_to_num(cos_vis, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # --- local incidence angle θ_loc ---
        theta_loc = np.arccos(np.clip(cos_vis, 0.0, 1.0))

        # --- evaluate α(T, |v|, θ_loc) over normals grid ---
        Tgrid = np.full_like(theta_loc, float(surface_temperature))
        # vmag is in micrometers/s; energy_function expects V in m/s (hence /1e6).
        alpha_loc = np.asarray(energy_function(Tgrid, vmag / 1e6, theta_loc), float)
        alpha_loc = np.clip(np.nan_to_num(alpha_loc, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # --- normalized expectation with a flat θ–φ measure (dθ dφ) ---
        # The normalization uses Pn and the discrete grid spacing factors.
        weights = Pn  # density × cosine visibility (as used in your code)
        Z = float(np.sum(weights) * dtheta * dphi)  # normalization constant
        if not np.isfinite(Z) or Z <= 0.0:
            return 0.0

        alpha_mean = float(np.sum(alpha_loc * weights) * dtheta * dphi / Z)
        return float(np.clip(alpha_mean, 0.0, 1.0))

    def compute_reaction_fraction(
            self,
            incident_pdf: np.ndarray | None = None,
            reaction_function: Any | None = None,
            incident_velocity: np.ndarray | None = None,
            accommodation_coefficient: float | None = None,
            surface_temperature: float | None = None,
            velocity_lim: float | None = None,
            scatterer: sct.Scatterer | None = None,
            first: bool = False,
    ) -> float:
        """
        Compute an effective reaction fraction by integrating over local incidence conditions.

        This routine combines:
          - a normals distribution conditioned on the incident direction (via scatterer.get_normals_pdf)
          - a velocity distribution (via scatterer.get_velocity_pdf)
          - a visibility mask based on local incidence cosine
          - a reaction kernel R(T, v, θ_loc)

        Two modes exist:
          - first=True: evaluate at the instantaneous incident direction (derived from incident_velocity)
          - first=False: integrate over an incident angle PDF (incident_pdf) on the visible hemisphere

        Parameters
        ----------
        incident_pdf : np.ndarray | None
            Incident-angle PDF on an (N,N) (θ,φ) grid, or None for isotropic default.
        reaction_function : callable | None
            Reaction kernel R(T, v, θ) returning values in [0,1].
        incident_velocity : np.ndarray | None
            Incident velocity vector used for the first-bounce direction.
        accommodation_coefficient : float | None
            Effective accommodation used in the velocity PDF.
        surface_temperature : float | None
            Surface temperature [K].
        velocity_lim : float | None
            Upper bound of speed integration domain.
        scatterer : sct.Scatterer | None
            Scatterer with get_normals_pdf and get_velocity_pdf.
        first : bool
            Toggle between single-direction evaluation and multi-incident integration.

        Returns
        -------
        float
            Reaction fraction in [0,1] (clipped).
        """
        # Validate required objects and scalars.
        if reaction_function is None or scatterer is None or scatterer.poly_surface is None:
            raise ValueError("reaction_function and a scatterer with a bound surface are required.")
        if surface_temperature is None or velocity_lim is None:
            raise ValueError("surface_temperature and velocity_lim must be provided.")

        ps = scatterer.poly_surface
        # Default incident velocity if none provided (avoids crashes, but yields near-zero physics).
        v = np.asarray(incident_velocity if incident_velocity is not None else [0.0, 0.0, 1.0], float)
        speed = float(np.linalg.norm(v))
        if not np.isfinite(speed) or speed <= 0.0:
            return 0.0

        # grids
        # N controls quadrature resolution in θ, φ, and speed.
        N = int(getattr(scatterer, "N_INT", 64))
        N = max(N, 2)
        theta_n = np.linspace(0.0, np.pi / 2.0, N)
        phi_n = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
        vel = np.linspace(0.0, float(velocity_lim), N)
        denom = max(N - 1, 1)
        dtheta_n = (np.pi / 2.0) / denom
        dphi_n = 2.0 * np.pi / N
        dv = float(velocity_lim) / denom

        # precompute normals mesh & weights
        # Vg, THNg, PHNg span (speed, theta_n, phi_n).
        Vg, THNg, PHNg = np.meshgrid(vel, theta_n, phi_n, indexing="ij")  # (N,N)
        w_n = np.sin(THNg) * dtheta_n * dphi_n * dv # (N,N)
        nx, ny, nz = np.sin(THNg) * np.cos(PHNg), np.sin(THNg) * np.sin(PHNg), np.cos(THNg)

        # helper: expectation over normals (visible-only normalized PDF) at *incident speed*
        def _En(theta_i: float, phi_i: float) -> float:
            """
            Compute the conditional expectation of reaction fraction given an incident direction.

            Parameters
            ----------
            theta_i, phi_i : float
                Incident direction angles.

            Returns
            -------
            float
                Expected reaction fraction over normals and speed distributions.
            """
            base = np.asarray(
                scatterer.get_normals_pdf(
                    (THNg.ravel(), PHNg.ravel()),
                    float(theta_i), float(phi_i),
                    ps.sigma_coeff, ps.mu_coeff, ps.R, N, first=False
                ), float
            ).reshape(N, N, N)
            base = np.maximum(np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            p_v = scatterer.get_velocity_pdf(Vg, speed, accommodation_coefficient, surface_temperature, cst.m_O)
            # local incidence cosine; visible iff > 0 (left as in your code)
            sx, sy, sz = np.sin(theta_i) * np.cos(phi_i), np.sin(theta_i) * np.sin(phi_i), np.cos(theta_i)
            cos_loc_2d = (nx * sx + ny * sy + nz * sz)  # (N,N,N)
            vis_mask = (cos_loc_2d > 0.0).astype(float)

            # normalize normals PDF on visible subset with sinθ_n weight
            Zn = float(np.sum(base * p_v * vis_mask * w_n))
            if not np.isfinite(Zn) or Zn <= 0.0:
                return 0.0
            Pn_vis = (base * p_v * vis_mask) / Zn  # ∫ Pn_vis w_n = 1

            # θ_loc for reaction; evaluate at *incident speed*
            theta_loc = np.arccos(np.clip(np.maximum(cos_loc_2d, 0.0), 0.0, 1.0))
            rk = np.asarray(reaction_function(float(surface_temperature), Vg, theta_loc), float)
            rk = np.clip(np.nan_to_num(rk, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

            return float(np.sum(Pn_vis * rk * w_n))  # expectation over normals only

        if first:
            # First-bounce direction from the incident velocity vector.
            theta_i = float(np.arccos(np.clip(-v[2] / speed, -1.0, 1.0)))
            phi_i = float(np.mod(np.arctan2(v[1], v[0]), 2.0 * np.pi))
            # print("REACTION", _En(theta_i, phi_i))
            return float(np.clip(_En(theta_i, phi_i), 0.0, 1.0))

        # Multi-incident case: integrate over an incident-angle PDF on visible set (0<θ_i<π/2)
        theta = np.linspace(0.0, np.pi, N)
        phi = np.linspace(0.0, 2.0 * np.pi, N)
        dtheta = np.pi / denom
        dphi = 2.0 * np.pi / denom
        THi, PHi = np.meshgrid(theta, phi, indexing="ij")
        w_i = np.sin(THi) * dtheta * dphi

        # Default incident distribution if none provided: uniform over 4π.
        if incident_pdf is None:
            inc = np.full((N, N), 1.0 / (4.0 * np.pi), float)
        else:
            inc = np.asarray(incident_pdf, float)
            if inc.shape != (N, N):
                raise ValueError(f"incident_pdf must have shape ({N},{N}), got {inc.shape}.")

        # Restrict to visible hemisphere and renormalize.
        mask_i = (THi > 0.0) & (THi < (0.5 * np.pi))
        inc_masked = np.clip(np.nan_to_num(inc, nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf) * mask_i
        Zi = float(np.sum(inc_masked * w_i))
        if not np.isfinite(Zi) or Zi <= 0.0:
            return 0.0
        inc_norm = inc_masked / Zi  # ∫ inc_norm dΩ_i = 1 on mask

        # Discrete quadrature over the incident-angle grid.
        rf = 0.0
        for i in range(N):
            for j in range(N):
                w = float(inc_norm[i, j])
                if w == 0.0:
                    continue
                rf += w * _En(float(THi[i, j]), float(PHi[i, j])) * dphi * dtheta * np.sin(theta[i])
        # print("REACTION", rf)
        return float(np.clip(np.nan_to_num(rf, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))

    def compute_flux_scaling(
            self,
            incident_velocity: np.ndarray | None = None,
            accommodation_coefficient: np.ndarray | None = None,
            surface_temperature: float | None = None,
            velocity_lim: float | None = None,
            scatterer: sct.Scatterer | None = None,
            reflection_index: int | None = None
    ) -> float:
        """
        Compute a dimensionless flux scaling factor for the r-th reflection.

        The factor integrates the speed PDFs produced by scatterer.get_velocity_pdf.
        For r == 1, it performs a 1D integral over speed.
        For r > 1, it performs a coupled 2D integral over (current speed, previous speed).

        Parameters
        ----------
        incident_velocity : np.ndarray | None
            Incident velocity vector; only its Euclidean norm is used.
        accommodation_coefficient : np.ndarray | None
            Effective accommodation coefficient α in [0,1].
        surface_temperature : float | None
            Surface temperature [K].
        velocity_lim : float | None
            Upper bound for the speed integration domain.
        scatterer : sct.Scatterer | None
            Scatterer providing get_velocity_pdf and integration resolution N_INT.
        reflection_index : int | None
            Reflection order r (>=1).

        Returns
        -------
        float
            Dimensionless scaling factor (>=0).
        """
        # ---- validate and sanitize inputs ----
        if scatterer is None:
            raise ValueError("scatterer must be provided")
        if velocity_lim is None or not np.isfinite(velocity_lim) or velocity_lim <= 0:
            raise ValueError("velocity_lim must be a positive finite number")
        if incident_velocity is None:
            raise ValueError("incident_velocity must be provided")
        if accommodation_coefficient is None:
            raise ValueError("accommodation_coefficient must be provided")
        if surface_temperature is None or not np.isfinite(surface_temperature):
            raise ValueError("surface_temperature must be a finite number")

        # Reflection order defaults to 1 if not given.
        r = int(reflection_index or 1)
        if r < 1:
            raise ValueError("reflection_index must be >= 1")

        # Quadrature resolution.
        N = int(getattr(scatterer, "N_INT", 64))
        N = max(N, 2)
        eps = 1e-6

        # Speed integration grid (avoid zero to prevent division issues).
        vel = np.linspace(eps, float(velocity_lim), N, dtype=float)
        dvel = vel[1] - vel[0] if N > 1 else float(velocity_lim)

        # Incident speed magnitude (used as parameter to velocity PDF).
        v_inc = float(np.linalg.norm(np.asarray(incident_velocity, float)))
        if not np.isfinite(v_inc) or v_inc <= 0.0:
            # No incident flux; scaling is zero
            return 0.0

        # clamp α to [0, 1] and promote to float
        alpha = float(np.clip(np.asarray(accommodation_coefficient, dtype=float), 0.0, 1.0))
        Tsurf = float(surface_temperature)

        eps = 1e-12  # numerical safety

        # ---- first reflection: 1D integral ----
        if r == 1:
            pdf = scatterer.get_velocity_pdf(vel, v_inc, 1.0 - (1.0 - alpha) ** (r), Tsurf, cst.m_O)
            pdf = np.nan_to_num(np.asarray(pdf, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            # broadcast to 1D if needed
            if pdf.shape != vel.shape:
                try:
                    pdf = np.broadcast_to(pdf, vel.shape)
                except Exception as e:
                    raise ValueError(f"velocity PDF shape {pdf.shape} not broadcastable to {vel.shape}") from e

            # Integrand structure follows the model implemented in your code.
            integrand = pdf * (max(v_inc, eps) / vel)
            # trapezoidal integration is more accurate on a uniform grid
            return float(np.trapezoid(integrand, vel))

        # ---- higher reflections: 2D integral over (v, v_old) ----
        V, V_old = np.meshgrid(vel, vel, indexing="ij")  # V: current; V_old: previous

        pdf_curr = scatterer.get_velocity_pdf(V, v_inc, 1.0 - (1.0 - alpha) ** (r), Tsurf, cst.m_O)
        pdf_curr = np.nan_to_num(np.asarray(pdf_curr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if pdf_curr.shape != V.shape:
            try:
                pdf_curr = np.broadcast_to(pdf_curr, V.shape)
            except Exception as e:
                raise ValueError(f"current velocity PDF shape {pdf_curr.shape} not broadcastable to {V.shape}") from e

        pdf_prev = scatterer.get_velocity_pdf(V_old, v_inc, 1.0 - (1.0 - alpha) ** (r - 1), Tsurf, cst.m_O)
        pdf_prev = np.nan_to_num(np.asarray(pdf_prev, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if pdf_prev.shape != V_old.shape:
            try:
                pdf_prev = np.broadcast_to(pdf_prev, V_old.shape)
            except Exception as e:
                raise ValueError(
                    f"previous velocity PDF shape {pdf_prev.shape} not broadcastable to {V_old.shape}") from e

        # ratio uses safe denominator
        ratio = V_old / np.clip(V, eps, None)
        integrand_2d = pdf_curr * pdf_prev * ratio

        # double trapezoidal integration: first over v_old, then over v
        tmp = np.trapezoid(integrand_2d, vel, axis=1)  # integrate over v_old axis=1
        result = np.trapezoid(tmp, vel, axis=0)  # integrate over v axis=0

        # guard against tiny negative due to numerical noise
        return float(np.maximum(result, 0.0))

    def compute_mr_term(
            self,
            incident_pdf: np.ndarray,
            erosion_function,
            gas_density: float,
            incident_velocity: np.ndarray,
            accommodation_coefficient: float,
            surface_temperature: float,
            surface_density: float,
            height_lim: float,
            velocity_lim: float,
            scatterer: sct.Scatterer,
    ) -> np.ndarray:
        """
        Compute the multi-reflection source term MR_term(θ_n, φ_n, h).

        This term is evaluated on an N×N×N grid over:
          - surface-normal angles (θ_n, φ_n)
          - local height h (relative to mean surface)

        It integrates over:
          - local scattering angles (θ_l, φ_l)
          - reflected global angles (θ_g, φ_g) derived from local angles and normal angles
          - speed v via the velocity PDF

        The implementation proceeds by:
          1) Building a reflected-angle PDF on a (θ_g, φ_g) grid.
          2) Building a height PDF and an angular flux term including shadowing.
          3) Integrating erosion yield over speed first, yielding a θ_l-only kernel.
          4) Mapping local→global angles for all combinations of normals and local angles.
          5) Accumulating MR on a height grid.

        Returns
        -------
        np.ndarray
            MR array of shape (N, N, N) corresponding to (θ_n, φ_n, h).
        """

        # ---- setup discrete grids (N from scatterer) ----
        N = int(getattr(scatterer, "N_INT", 64))
        N = max(N, 2)
        v_i = float(np.linalg.norm(incident_velocity))
        # local angles (scattering), normal angles, globals for interpolation, heights, speeds
        theta_l = np.linspace(0.0, 0.5 * np.pi, N)  # [0, π/2]
        phi_l = np.linspace(0.0, 2.0 * np.pi, N)  # [0, 2π]
        theta_n = np.linspace(0.0, 0.5 * np.pi, N)  # [0, π/2]
        phi_n = np.linspace(0.0, 2.0 * np.pi, N)  # [0, 2π]
        theta_g = np.linspace(0.0, np.pi, N)  # global grid for interpolants
        phi_g = np.linspace(0.0, 2.0 * np.pi, N)
        Hgrid = np.linspace(-float(height_lim), float(height_lim), N)
        Vgrid = np.linspace(0.0, float(velocity_lim), N)
        denom = max(N - 1, 1)
        dtheta_l, dphi_l, dv, dh = (0.5 * np.pi) / denom, (2.0 * np.pi) / denom, float(velocity_lim) / denom, height_lim * 2.0 / denom
        # Incident angles derived from incident_velocity.
        theta_i = np.arccos(- incident_velocity[2] / v_i)
        phi_i = np.arctan2(incident_velocity[1], incident_velocity[0])

        # ---- reflected-angle PDF on (theta_g, phi_g) ----
        THg, PHg = np.meshgrid(theta_g, phi_g, indexing="ij")

        # Either single-bounce scattering from (theta_i, phi_i) or multi-bounce from incident_pdf.
        if incident_pdf is None:
            refl_pdf_grid = scatterer.get_scattering_pdf(
                np.vstack([THg.ravel(), PHg.ravel()]), theta_i, phi_i, first=True, accommodation_coefficient=accommodation_coefficient
            ).reshape(N, N)
        else:
            refl_pdf_grid = scatterer.get_multi_scattering_pdf(
                np.vstack([THg.ravel(), PHg.ravel()]), incident_pdf, accommodation_coefficient
            ).reshape(N, N)
        refl_pdf_grid = np.nan_to_num(refl_pdf_grid, nan=0.0, posinf=0.0, neginf=0.0)

        # Shift by π in φ on uniform grid (intended alignment convention).
        refl_pdf_grid = np.roll(refl_pdf_grid, int(round(np.pi / dphi_l)) % N,
                                axis=1)  # shift by π on a uniform φ grid)

        # Normalize reflected PDF with solid angle weights.
        refl_norm = np.sum(refl_pdf_grid * 2 * dtheta_l * dphi_l * np.sin(THg))
        refl_pdf = sp.interpolate.RegularGridInterpolator((np.pi - theta_g, phi_g), refl_pdf_grid / refl_norm,
                                                          bounds_error=True,
                                                          fill_value=None)

        # Height PDF for the PolyGaussian surface.
        height_pdf_grid = scatterer.poly_surface.get_height_pdf(Hgrid)
        height_norm = np.sum(height_pdf_grid * dh)
        height_pdf = sp.interpolate.RegularGridInterpolator((Hgrid,), height_pdf_grid / height_norm, bounds_error=True, fill_value=None)

        # ---- angular flux on (theta_g, phi_g, Hgrid) ----
        # Flux is reduced by shadow_function; stored as 1 - shadow.
        TH3, PH3, HH3 = np.meshgrid(theta_g, phi_g, Hgrid, indexing="ij")
        ang_flux_grid = 1.0 - scatterer.shadow_function(TH3.ravel(), PH3.ravel(), HH3.ravel()).reshape(N, N, N)
        ang_flux_grid = np.nan_to_num(ang_flux_grid, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize angular flux with the coupled height and reflected-angle distributions.
        ang_flux_norm = np.sum(height_pdf_grid[:, None, None] * refl_pdf_grid[::-1, :, None] * ang_flux_grid * np.sin(TH3) * dtheta_l * 2.0 * dphi_l * dh)
        ang_flux = sp.interpolate.RegularGridInterpolator((theta_g, phi_g, Hgrid), ang_flux_grid / ang_flux_norm, bounds_error=True,
                                                          fill_value=None)

        # ---- speed PDF and erosion kernel Ey(T,v,θ_l) → integrate speed out first ----
        # VelPDF(v) evaluated on Vgrid; then Ey(T, v, θ_l) is integrated over v.
        vel_pdf = scatterer.get_velocity_pdf(Vgrid, v_i, float(accommodation_coefficient),
                                             float(surface_temperature), cst.m_O)
        vel_pdf = np.nan_to_num(np.asarray(vel_pdf, float), nan=0.0, posinf=0.0, neginf=0.0)
        Tgrid_thetav = np.full((N, N), float(surface_temperature), dtype=float)
        Vmap = np.broadcast_to(Vgrid, (N, N)).T  # (Nv,Nθ) → will transpose after Ey
        Ey_thetav = np.asarray(erosion_function(Tgrid_thetav.T, Vmap.T / 1e6, theta_l[None, :]), float)  # (Nv,Nθ)
        Ey_thetav = np.nan_to_num(Ey_thetav, nan=0.0, posinf=0.0, neginf=0.0)

        # Speed-integrated erosion contribution per θ_l.
        vel_int = (Ey_thetav * v_i * vel_pdf[:, None]).sum(axis=0, dtype=float) * dv  # (Nθ_l,)

        # ---- geometry: map local→global angles for all (n1,n2,θl,φl) (no height here) ----
        # Broadcast to (N_n1,N_n2,N_θl,N_φl)
        n1 = theta_n[:, None, None, None]
        n2 = phi_n[None, :, None, None]
        l1 = theta_l[None, None, :, None]
        l2 = phi_l[None, None, None, :]
        g1, g2 = self.local_to_global_angles(l1, l2, n1, n2)  # shapes (N,N,N,N)

        # Clamp angles into interpolant domains (prevents NaNs and mimics edge extrapolation).
        g1 = np.clip(g1, theta_g[0], theta_g[-1])
        g2 = np.mod(g2, 2.0*np.pi)

        # ---- reflected PDF evaluated on (n1,n2,θl,φl) ----
        pts2 = np.c_[g1.ravel(), g2.ravel()]
        Rvals = refl_pdf(pts2).reshape(g1.shape)
        Rvals = np.nan_to_num(Rvals, nan=0.0, posinf=0.0, neginf=0.0)

        # prepare accumulation
        MR = np.zeros((N, N, N), dtype=float)  # (n1,n2,h)
        wtheta = (np.sin(theta_l) * np.cos(theta_l) * dtheta_l).astype(float)  # (Nθ)

        # Base dimensional factor converting density/geometry into a flux scaling.
        base_factor = (gas_density / surface_density) * cst.NA / cst.m_O

        # θ_l weights applied after speed integration.
        vel_w = (vel_int * base_factor) * wtheta  # (Nθ)

        eps = 1e-15

        # Height loop: evaluate angular flux at each height and assemble MR contributions.
        for k, h in enumerate(Hgrid):
            # angle flux at (g1, g2, h)
            Hk = np.full_like(g1, h, dtype=float)
            pts3 = np.c_[g1.ravel(), g2.ravel(), Hk.ravel()]
            AF = ang_flux(pts3).reshape(g1.shape)
            AF = np.nan_to_num(AF, nan=0.0, posinf=0.0, neginf=0.0)

            # sum over local φ_l (axis=-1): apply dφ weight
            K = (AF * Rvals).sum(axis=3, dtype=float) * dphi_l  # (N_n1,N_n2,N_θl)

            # Rnorm(n1,n2,h): normalization-like quantity (computed but not applied later)
            Rnorm_k = np.sum(Rvals * np.sin(l1) * dtheta_l * dphi_l, axis=(2, 3), dtype=float)  # (Nn1, Nn2)
            Rnorm_k = np.clip(Rnorm_k, eps, None)

            # Combine with speed-integrated erosion and θ-weight; sum over θ_l
            MR[:, :, k] = (K * vel_w[None, None, :]).sum(axis=2, dtype=float)

        return MR

    def first_reflection_flux(
            self,
            Phi: np.ndarray,
            incident_velocity: np.ndarray,  # (vx, vy, vz) in microns/s
            atmospheric_density: float,  # kg/m^3
            surface_density: float,  # kg/m^3
            surface_temperature: float,  # K
            old_flux_x: np.ndarray,
            old_flux_y: np.ndarray,
            old_flux_z: np.ndarray,
            erosion_function,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the first-reflection (direct incidence) erosion flux field on the grid.

        The computed flux is proportional to:
          - atmospheric-to-surface density ratio
          - erosion yield Ey(T, |v|, θ)
          - incident velocity vector components

        The local incidence angle θ is obtained from the level-set normals via:
            cos θ = - (v · n) / |v|

        Parameters
        ----------
        Phi : np.ndarray
            Level-set field.
        incident_velocity : np.ndarray
            Incident velocity vector (micrometers/s).
        atmospheric_density : float
            Atmospheric mass density (kg/m^3).
        surface_density : float
            Surface/material mass density (kg/m^3).
        surface_temperature : float
            Surface temperature (K).
        old_flux_x, old_flux_y, old_flux_z : np.ndarray
            Previous flux fields, used by lv.get_normals_full for upwinding.
        erosion_function : callable
            Ey(T, V, θ) erosion yield kernel; expects V in m/s (hence /1e6 later).

        Returns
        -------
        (flux_x, flux_y, flux_z) : tuple[np.ndarray, np.ndarray, np.ndarray]
            Flux components over the full grid, same shape as Phi.
        """
        # 1) Surface normals (uses non-periodic, upwinded gradients internally)
        nx, ny, nz = lv.get_normals_full(Phi, old_flux_x, old_flux_y, old_flux_z, self.X, self.Y, self.Z)

        # 2) Incidence angle
        vx, vy, vz = map(float, np.asarray(incident_velocity).ravel()[:3])
        vmag = (vx * vx + vy * vy + vz * vz) ** 0.5
        if not np.isfinite(vmag) or vmag <= 0.0:
            # No incidence -> no erosion
            z = np.zeros_like(Phi, dtype=float)
            return z, z, z

        vi_dot_n = nx * vx + ny * vy + nz * vz
        cos_theta = np.clip(-vi_dot_n / vmag, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # 3) Erosion yield Ey(T, |v|, θ)
        # Build broadcastable fields for T and speed magnitude.
        T = np.full_like(Phi, float(surface_temperature), dtype=float)
        V = np.full_like(Phi, vmag, dtype=float)

        Ey = erosion_function(T, V / 1e6, theta)  # must broadcast over arrays
        Ey = np.where(np.isfinite(Ey), Ey, 0.0)

        # 4) Flux components (constant scale)
        # Scale converts erosion yield to volumetric removal rate per AO flux (as encoded here).
        scale = - (float(atmospheric_density) / float(surface_density)) * (cst.NA / cst.m_O)
        Ey_scale = scale * Ey
        return Ey_scale * vx, Ey_scale * vy, Ey_scale * vz

    def multi_reflection_flux(
            self,
            num_reflections: int,
            erosion_function,
            reaction_function,
            incident_velocity: np.ndarray,
            gas_density: float,
            accommodation_coefficient: float,
            surface_temperature: float,
            surface_density: float,
            height_lim: float,
            velocity_lim: float,
            scatterer: sct.Scatterer,
    ) -> sp.interpolate.RegularGridInterpolator:
        """
        Build an interpolator for the accumulated multi-reflection erosion flux kernel.

        The returned RegularGridInterpolator is defined on:
          (normal_angle_1, normal_angle_2, height) -> MR_flux

        The algorithm iterates over reflection bounces:
          - Computes bounce-dependent accommodation alpha_i.
          - Computes reaction fraction and trapping probability to update `scale`.
          - Computes MR term for the current bounce and accumulates it.
          - Updates the incident PDF for the next bounce from the reflected PDF.

        Parameters
        ----------
        num_reflections : int
            Maximum number of reflections to include.
        erosion_function : callable
            Ey(T, V, θ) erosion yield kernel.
        reaction_function : callable
            R(T, V, θ) reaction fraction kernel.
        incident_velocity : np.ndarray
            Incident velocity vector for first bounce.
        gas_density : float
            Atmospheric mass density (kg/m^3).
        accommodation_coefficient : float
            Base accommodation coefficient for bounce scaling.
        surface_temperature : float
            Surface temperature (K).
        surface_density : float
            Material density (kg/m^3).
        height_lim : float
            Height range for PolyGaussian height integration.
        velocity_lim : float
            Speed integration upper limit.
        scatterer : sct.Scatterer
            Scatterer instance based on a fitted PolyGaussian surface.

        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Interpolator for MR_flux on (θ_n, φ_n, h).
        """

        # ---- grids & helpers ----------------------------------------------------
        N = int(getattr(scatterer, "N_INT", 64))
        N = max(N, 2)
        thetan = np.linspace(0.0, 0.5 * np.pi, N)  # normal angle 1  ∈ [0, π/2]
        phin = np.linspace(0.0, 2.0 * np.pi, N)  # normal angle 2  ∈ [0, 2π]
        H = np.linspace(-float(height_lim), float(height_lim), N)  # height ≥ 0 (as in your code)

        thetag = np.linspace(0.0, np.pi, N)  # global-angle grids for PDFs
        phig = np.linspace(0.0, 2.0 * np.pi, N)
        dtheta = (thetag[-1] - thetag[0]) / max(N - 1, 1)
        dphi = (phig[-1] - phig[0]) / max(N - 1, 1)
        W = np.sin(thetag)[:, None] * dtheta * dphi  # solid-angle weights (N,N)

        def _normalize_pdf(pdf2d: np.ndarray) -> np.ndarray:
            """
            Normalize a 2D angular PDF with respect to solid angle.

            Parameters
            ----------
            pdf2d : np.ndarray
                PDF values on the (thetag, phig) grid.

            Returns
            -------
            np.ndarray
                Normalized PDF (or zeros if normalization fails).
            """
            pdf2d = np.nan_to_num(np.asarray(pdf2d, float), nan=0.0, posinf=0.0, neginf=0.0)
            Z = float(np.sum(pdf2d * W))
            return pdf2d / Z if (Z > 0.0 and np.isfinite(Z)) else np.zeros_like(pdf2d)

        # incident angles from incident_velocity
        v = np.asarray(incident_velocity, float)
        vmag = float(np.linalg.norm(v))

        if not np.isfinite(vmag) or vmag <= 0.0:
            raise ValueError("incident_velocity must have non-zero magnitude.")
        theta_inc = float(np.arccos(np.clip(- v[2] / vmag, -1.0, 1.0)))
        phi_inc = float(np.arctan2(v[1], v[0]) % (2.0 * np.pi))

        # ---- accumulate flux over reflections -----------------------------------
        MR_flux = np.zeros((N, N, N), float)  # (θn, φn, h)
        scale = 1.0

        # First incident PDF is None (handled inside scatterer calls as delta-like).
        inc_pdf = None

        # pre-mesh for reflected PDF evaluation
        THg, PHg = np.meshgrid(thetag, phig, indexing="ij")
        PTS = np.vstack([THg.ravel(), PHg.ravel()])

        for i in range(int(max(1, num_reflections))):
            # accommodation for this bounce (cumulative)
            alpha_i = 1.0 - (1.0 - accommodation_coefficient) ** (i + 1)

            # reaction & trapping fractions (first-bounce shortcuts handled inside)
            react = float(self.compute_reaction_fraction(
                inc_pdf, reaction_function, v, alpha_i, float(surface_temperature),
                float(velocity_lim), scatterer, first=(i == 0)
            ))
            trap = float(scatterer.get_trapping_probability(
                inc_pdf, theta_inc, phi_inc, first=(i == 0)
            ))
            flux = float(self.compute_flux_scaling(
                incident_velocity, alpha_i, float(surface_temperature),
                float(velocity_lim), scatterer, i + 1
            ))
            # Clamp react/trap to physical range and update cumulative scale.
            react = 0.0 if not np.isfinite(react) or react < 0.0 else min(react, 1.0)
            trap = 0.0 if not np.isfinite(trap) or trap < 0.0 else min(trap, 1.0)
            scale *= (1.0 - react)
            # survivors for next bounce
            scale *= trap
            # print("SCALE", scale, "TRAP", trap)
            if scale <= 0.0 or i == num_reflections - 1:
                break

            # MR term for this bounce on (θn, φn, h)
            MR_i = self.compute_mr_term(
                inc_pdf, erosion_function, float(gas_density), v, alpha_i, float(surface_temperature),
                float(surface_density),
                float(height_lim), float(velocity_lim), scatterer
            )

            # Accumulate scaled contribution.
            MR_flux += scale * np.nan_to_num(MR_i, nan=0.0, posinf=0.0, neginf=0.0)

            # next incident PDF = reflected PDF(θg, φg | current inc_pdf)
            if i == 0:
                refl_vals = scatterer.get_scattering_pdf(PTS, theta_inc, phi_inc, accommodation_coefficient=alpha_i).reshape(N, N)
            else:
                refl_vals = scatterer.get_multi_scattering_pdf(PTS, inc_pdf, accommodation_coefficient=alpha_i).reshape(N, N)
            # Flip theta axis to match the convention used downstream.
            inc_pdf = _normalize_pdf(refl_vals)[::-1, :]

        # ---- interpolator (linear; bounds_error=True; NaN-free grid) --------------
        MR_flux = np.nan_to_num(MR_flux, nan=0.0, posinf=0.0, neginf=0.0)

        return sp.interpolate.RegularGridInterpolator((thetan, phin, H), MR_flux, method="linear", bounds_error=True,
                                                      fill_value=None)

    def compute_mass_loss(
            self,
            initial_surface: surf.Surface,
            current_surface: surf.Surface,
            material_density: float,
    ) -> float:
        """
        Compute mass removed between an initial and current surface.

        The mass loss is computed from the volume difference:
            V_loss = sum(Z_initial - Z_current) * dx * dy
            m_loss = rho_material * V_loss

        NaNs are treated as zero contribution.

        Parameters
        ----------
        initial_surface : surf.Surface
            Reference surface at t=0.
        current_surface : surf.Surface
            Surface at current time.
        material_density : float
            Material density used to convert volume loss to mass loss.

        Returns
        -------
        float
            Non-negative mass loss (clipped to >= 0).
        """
        Z0 = np.asarray(initial_surface.Z, dtype=float)
        Z1 = np.asarray(current_surface.Z, dtype=float)
        if Z0.shape != Z1.shape:
            raise ValueError("initial_surface.Z and current_surface.Z must have the same shape.")

        # Cell spacings: prefer precomputed, else infer from surface grids if available.
        dx = getattr(self, "dx", None)
        dy = getattr(self, "dy", None)

        def _infer_spacing(surf_obj, Z):
            """
            Infer grid spacing from Surface X/Y fields if they match Z.

            Parameters
            ----------
            surf_obj : surf.Surface
                Surface object that may provide X and Y grids.
            Z : np.ndarray
                Height field shape to match.

            Returns
            -------
            (dx, dy) : tuple[float | None, float | None]
                Spacing estimates or (None, None) if inference fails.
            """
            if hasattr(surf_obj, "X") and hasattr(surf_obj, "Y"):
                X = np.asarray(surf_obj.X, float)
                Y = np.asarray(surf_obj.Y, float)
                if X.shape == Z.shape and Y.shape == Z.shape and Z.shape[0] > 1 and Z.shape[1] > 1:
                    dx_ = (np.nanmax(X) - np.nanmin(X)) / (Z.shape[0] - 1)
                    dy_ = (np.nanmax(Y) - np.nanmin(Y)) / (Z.shape[1] - 1)
                    if np.isfinite(dx_) and dx_ > 0 and np.isfinite(dy_) and dy_ > 0:
                        return dx_, dy_
            return None, None

        # If dx/dy are not valid scalars, infer them from surface grids.
        if not (isinstance(dx, (int, float)) and dx > 0 and np.isfinite(dx) and
                isinstance(dy, (int, float)) and dy > 0 and np.isfinite(dy)):
            dx_i, dy_i = _infer_spacing(initial_surface, Z0)
            if dx_i is None:
                dx_i, dy_i = _infer_spacing(current_surface, Z1)
            if dx_i is None:
                raise ValueError("dx, dy not set and could not be inferred from surface grids.")
            dx, dy = dx_i, dy_i

        # Volume and mass loss (clip to ≥0 to avoid negative due to noise/deposition).
        dZ = np.nan_to_num(Z0 - Z1, nan=0.0, posinf=0.0, neginf=0.0)
        vol_loss = float(np.sum(dZ) * dx * dy)
        # The 1e-18 factor encodes a unit conversion consistent with your domain scaling.
        mass_loss = float(material_density) * vol_loss * 1e-18
        return max(mass_loss, 0.0)

    def simulate(self) -> None:
        """
        Run the full level set erosion simulation.

        This routine advances the signed-distance-like level set field `Phi` forward in time using:
          - A first-reflection (FR) erosion flux computed from local surface normals and the incident AO velocity.
          - A multi-reflection (MR) erosion contribution computed from a PolyGaussian surface model and a scatterer.
          - A small relaxation term to stabilize/regularize the level set evolution.

        It also:
          - Seeds an "ash shadow" mask via synthetic ash particles (used to locally inhibit erosion).
          - Tracks time histories (mass loss, AO exposure/fluence, effective erosion yield, etc.).
          - Optionally refits the PolyGaussian surface model periodically and exports plots/geometry.

        Assumptions and conventions
        ---------------------------
        - `self.Phi`, `self.X`, `self.Y`, `self.Z`, and grid spacings `dx,dy,dz` are already initialized.
        - All velocities used by the level set advection are in micrometers per second (μm/s).
        - Angles are in radians.
        - The surface normal reference direction is encoded via `self.surface_angle`.
        """

        # --- sanity on required params ---
        # The simulation window must be defined in UTC timestamps.
        if self.sim_start_utc is None or self.sim_end_utc is None:
            raise ValueError("Simulation window not set. Did you call import_inputs first?")
        # Positive timestep is mandatory for explicit advection.
        if self.time_step is None or self.time_step <= 0:
            raise ValueError("time_step_seconds must be a positive number.")
        # Nint controls the quadrature / discretization size for PolyGaussian and scattering integrals.
        if self.Nint is None or self.Nint < 1:
            raise ValueError("Nint (integration steps) must be ≥ 1.")
        # MR requires a reflection count. This should be provided in the inputs file.
        if getattr(self, "num_reflections", None) is None:
            raise ValueError("num_reflections not set. Add 'number of reflections' to inputs.")
        # If the user has not provided α_E, use a conservative default (later overwritten after first evaluation).
        if self.energy_accommodation_coeff is None:
            # Safe default if not provided elsewhere
            self.energy_accommodation_coeff = 0.7

        # --- pull class parameters ---
        # dt: physical timestep in seconds.
        dt = float(self.time_step)
        # Total simulation duration in seconds.
        time_span = float((self.sim_end_utc - self.sim_start_utc).total_seconds())
        # Period (seconds) at which the PolyGaussian geometry may be refit and MR kernel refreshed.
        update_time = float(self.poly_update_step)
        # Period (seconds) used for plotting/export cadence.
        plotting_time = float(self.print_step)
        # Number of MR bounces to consider.
        num_multi_reflections = int(self.num_reflections)
        # PolyGaussian/scattering angular and quadrature resolution.
        N_INT = int(self.Nint)

        # --- constants / limits ---
        # Upper bound of velocity integration domain for MR (converted to μm/s).
        velocity_lim = meters_to_micrometers(15000.0)  # m/s → μm/s
        # Height integration domain for the PolyGaussian height PDF and shadowing in MR.
        height_lim = 105.0  # μm
        # Used to sample a "mean-ish" incident velocity by reducing thermal spread.
        max_speed_ratio = 100.0
        # Small stabilizing term applied by advect_relax_term (acts like mild regularization).
        relax_term = 1e-16

        # --- time & update counters ---
        # Number of time steps including t=0.
        Nt = int(time_span / dt) + 1
        # Number of time steps between PolyGaussian refresh events.
        N_update = int(update_time / dt) if update_time > 0 else 0
        # Next wall-clock times at which plotting/export actions should occur.
        next_hist_plot_t = float(plotting_time) if plotting_time and plotting_time > 0 else float("inf")
        next_hist_export_t = float(plotting_time) if plotting_time and plotting_time > 0 else float("inf")
        next_geo_export_t = float(plotting_time) if plotting_time and plotting_time > 0 else float("inf")

        # --- plotting dirs & counters ---
        # Output directories. These are created if they do not exist.
        pg_dir = os.path.join(self.project_folder, "results", "polygaussian_plots")
        eh_dir = os.path.join(self.project_folder, "results", "erosion_plots")
        eh_data_dir = os.path.join(self.project_folder, "results", "erosion_data")
        geo_data_dir = os.path.join(self.project_folder, "results", "geometry_data")
        os.makedirs(pg_dir, exist_ok=True)
        os.makedirs(eh_dir, exist_ok=True)
        os.makedirs(eh_data_dir, exist_ok=True)
        os.makedirs(geo_data_dir, exist_ok=True)
        # Indices for incrementing filenames.
        pg_idx = 1
        eh_idx = 1
        geo_idx = 1

        # --- initial surfaces / fit ---
        # Extract an initial interface surface mesh/heightfield from the level set.
        x0, y0, z0 = lv.extract_surface(self.Phi, self.Lx, self.Ly, self.Lz)
        init_surface = surf.Surface(x0, y0, z0)
        # Fit an initial PolyGaussian roughness representation to the surface.
        poly_surface = surf.PolyGaussian_Surface(N_INT=N_INT, angle=self.surface_angle)
        poly_surface.fit_parameters(init_surface, verbose=False, niter=8)
        # Scatterer wraps the PolyGaussian surface and provides PDFs for scattering/velocity/shadowing/trapping.
        scatterer = sct.Scatterer(poly_surface)

        # --- fields / accumulators ---
        # FR flux components (same shape as Phi). Used by FR advection and in upwinding of normals.
        fr_flux_x = np.zeros_like(self.Phi)
        fr_flux_y = np.zeros_like(self.Phi)
        fr_flux_z = np.zeros_like(self.Phi)

        # Time histories for diagnostics and exporting.
        # t_hist: time in seconds since simulation start.
        # mass_loss_hist: cumulative mass loss (units depend on compute_mass_loss scaling).
        # AO_hist: cumulative AO particle count over the domain (scaled to SI-based counting).
        # AO_fluence_hist: cumulative AO fluence (per area, scaled to typical units).
        t_hist, mass_loss_hist, AO_hist, AO_fluence_hist = [0.0], [0.0], [0.0], [0.0]
        # EY_hist: effective erosion yield inferred from mass loss and AO exposure.
        # rho_hist: AO mass density time series (from atmospheric input).
        # sratio_hist: speed ratio time series (from atmospheric input).
        EY_hist, rho_hist, sratio_hist = [0.0], [0.0], [0.0]
        # Incident angle histories (radians).
        th1_hist, th2_hist = [0.0], [0.0]
        # PolyGaussian parameter histories (stored each step for later plotting/export).
        mu_param_hist, sigma_param_hist, r_hist = [poly_surface.mu_coeff], [poly_surface.sigma_coeff], [
            poly_surface.R]
        gamma_hist = [np.linspace(poly_surface.gamma_min, poly_surface.gamma_max, poly_surface.mu_coeff.shape[0])]

        # --- initial conditions for MR ---
        # Evaluate inputs at t=0 to seed ash placement and initial MR kernel build.
        v0 = float(self.velocity_interp(0.0))
        th1 = float(self.incident_angle_1_interp(0.0))
        th2 = float(self.incident_angle_2_interp(0.0))
        T0 = float(self.surface_temperature_interp(0.0))
        rho0 = float(self.surface_density_interp(0.0))
        dens0 = float(self.density_interp(0.0))

        # Mean incident direction vector (μm/s), constructed from angles (θ=th1, φ=th2) toward -z.
        v_vec0 = np.array([np.sin(th1) * np.cos(th2), np.sin(th1) * np.sin(th2), -np.cos(th1)]) * v0

        #################### Kapton Second Layer Simulation ################################################
        # Optional ash seeding configurations. These blocks document alternative parameter sets that
        # change ash population, size distribution, and placement depth.
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash(v_vec0, median_radius=1.0, radius_variance=0.8,
        #                                                               num_particles=3000)                   # Kapton H simulation
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash_ontop(v_vec0, median_radius=3.0, radius_variance=1.0,
        #                                                               num_particles=110)                    # Kapton H display simulation

        #################### Kapton First Layer Simulation ##################################################
        # Active ash seeding: place ash inside solid nodes; Ash_Shadow is updated via add_ash_particle calls.
        self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash(v_vec0, median_radius=1.0,
                                                                      radius_variance=0.8,
                                                                      num_particles=2500)

        # #################### Teflon ETFE Simulation ##################################################
        # Alternative ash parameters suitable for ETFE scenarios.
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash(v_vec0, median_radius=0.8, radius_variance=0.4,
        #                                                               num_particles=2000) # Rv = 0.9, corr = 0.1

        #################### Teflon FEP Simulation ###################################################
        # Alternative ash parameters suitable for FEP scenarios.
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash(v_vec0, median_radius=0.15, radius_variance=0.05,
        #                                                               num_particles=1200)  # Rv = 0.05, corr = 0.1

        #################### Teflon FEP Cone Simulation ###################################################
        # Alternative: ash placed on a mid-plane and/or explicit rough cylinder particle injection.
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash_ontop(v_vec0, median_radius=0.1, radius_variance=0.04,
        #                                                               num_particles=1500)  # Rv = 0.05, corr = 0.1
        # self.add_ash_particle(np.array([0.0, 0.0, 5.0]), 4.0, np.array([0.0, 0.0, -1.0]), rough_rms=1, corr_len=0.2)

        # #################### Teflon PTFE Simulation ###################################################
        # Alternative: PTFE on-top ash placement.
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash_ontop(v_vec0, median_radius=2.0, radius_variance=0.05,
        #                                                               num_particles=0)  # Rv = 0.05, corr = 0.1

        #################### Teflon PTFE Cone Simulation ###################################################
        # Alternative PTFE cone-like configurations with explicit particles.
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash_ontop(v_vec0, median_radius=2.0,
        #                                                                     radius_variance=0.05,
        #                                                                     num_particles=2)  # Rv = 0.05, corr = 0.1
        # self.ash_x, self.ash_y, self.ash_z, self.ash_r = self.add_ash_ontop(v_vec0, median_radius=0.05,
        #                                                                     radius_variance=0.01,
        #                                                                     num_particles=3000)  # Rv = 0.05, corr = 0.1
        # self.add_ash_particle(np.array([5.0, 0.0, 5.0]), 2.0, np.array([0.0, 0.0, -1.0]), rough_rms=1, corr_len=0.2)
        # self.add_ash_particle(np.array([-5.0, 0.0, 5.0]), 2.0, np.array([0.0, 0.0, -1.0]), rough_rms=1, corr_len=0.2)

        # Compute the first-bounce effective energy accommodation α_E from the conditioned normals distribution.
        accommodation_coefficient = self.compute_energy_accommodation(
            energy_function=self.energy_accommodation_interpolator,
            incident_velocity=v_vec0,
            surface_temperature=T0,
            scatterer=scatterer,
            first=True,
        )
        # ("Energy accommodation", accommodation_coefficient)
        # Persist α_E for downstream uses or fallbacks.
        self.energy_accommodation_coeff = accommodation_coefficient

        # Precompute the MR flux interpolator over (θ_n, φ_n, h) using the current surface fit.
        mr_interpolator = self.multi_reflection_flux(
            num_multi_reflections,
            self.erosion_interpolator,
            self.reaction_interpolator,
            v_vec0, dens0, accommodation_coefficient,
            T0, rho0, height_lim, velocity_lim, scatterer
        )

        # --- progress bar setup ---
        # Prefer tqdm if available; otherwise print percentage every ~5%.
        try:
            from tqdm import tqdm  # type: ignore
            progress_iter = tqdm(range(Nt), total=Nt, desc="Simulating erosion", unit="step")
        except Exception:
            # Simple fallback progress printer
            def _fallback_iter(n):
                last_pct = -1
                for ii in range(n):
                    pct = int(100 * ii / max(1, n - 1))
                    if pct % 5 == 0 and pct != last_pct:
                        print(f"[simulate] {pct}%")
                        last_pct = pct
                    yield ii

            progress_iter = _fallback_iter(Nt)

        # --- main loop ---
        for i in progress_iter:
            # Current simulation time since start, in seconds.
            t_cur = i * dt

            # inputs at time t_cur
            # dens: AO mass density (kg/m^3) from atmospheric model.
            dens = float(self.density_interp(t_cur))
            # v_mag: spacecraft speed magnitude in μm/s (note: velocity_interp was scaled to μm/s earlier).
            v_mag = float(self.velocity_interp(t_cur))
            # sratio: speed ratio controlling thermal spread for incident velocity sampling.
            sratio = float(self.speed_ratio_interp(t_cur))
            # Surface temperature and density for yield computations and scaling.
            T_surf = float(self.surface_temperature_interp(t_cur))
            rho_s = float(self.surface_density_interp(t_cur))
            # Incident direction angles in radians.
            th1 = float(self.incident_angle_1_interp(t_cur))
            th2 = float(self.incident_angle_2_interp(t_cur))

            # Sample a thermally-perturbed incident velocity (μm/s) consistent with current speed ratio.
            v_inc = self.sample_incident_velocity(th1, th2, v_mag, sratio)
            # "Mean" incident velocity: reduced thermal perturbation using a large speed ratio.
            v_inc_mean = self.sample_incident_velocity(th1, th2, v_mag, max_speed_ratio)

            # current surface (from Phi)
            # Extract the current Φ=0 interface and build a surface object.
            Xs, Ys, Zs = lv.extract_surface(self.Phi, self.Lx, self.Ly, self.Lz)
            cur_surface = surf.Surface(Xs, Ys, Zs)
            # Isotropic version of the surface used by MR routines (depends on implementation).
            iso_surface = cur_surface.get_isotropic_surface(self.surface_angle)

            # mass loss (m^3?) and AO fluence accumulation (SI)
            # Mass loss between the initial and current surface, using current surface density.
            mass_loss_hist.append(self.compute_mass_loss(init_surface, cur_surface, rho_s))
            # AO_hist: integrated AO exposure over the domain area (Lx*Ly), scaled to particle count.
            AO_hist.append(
                AO_hist[-1]
                + dt * np.linalg.norm(v_inc) * np.cos(
                    th1) * self.Lx * self.Ly * dens / cst.m_O * cst.NA * 1e3 * 1e-18
            )
            # AO_fluence_hist: integrated AO exposure per unit area, scaled to common fluence units.
            AO_fluence_hist.append(
                AO_fluence_hist[-1]
                + dt * np.linalg.norm(v_inc) * np.cos(th1) * dens / cst.m_O * cst.NA * 1e3 * 1e-6 * 1e-4
            )
            # instantaneous erosion yield (cm^3/AO atom)
            # Convert mass loss and AO count to an effective volumetric yield; guarded against division by zero.
            EY_hist.append(
                (mass_loss_hist[-1] / AO_hist[-1] / rho_s * 1e6) if (AO_hist[-1] > 0 and rho_s > 0) else 0.0
            )

            # first-reflection fluxes → FR advection
            # Compute FR flux vectors at each grid cell based on local normals and Ey(T,|v|,θ).
            fr_flux_x, fr_flux_y, fr_flux_z = self.first_reflection_flux(
                self.Phi, v_inc, dens, rho_s, T_surf,
                fr_flux_x, fr_flux_y, fr_flux_z, self.erosion_interpolator
            )
            # Advect Φ under FR term, modulated by Ash_Shadow (0 blocks, 1 allows).
            self.Phi = lv.advect_FR_term(
                self.Phi, self.Ash_Shadow, fr_flux_x, fr_flux_y, fr_flux_z,
                self.dx, self.dy, self.dz, dt, self.surface_angle
            )
            # multi-reflection advection (uses isotropic surface grids)
            # Apply MR term using the precomputed interpolator over (θ_n, φ_n, h).
            self.Phi = lv.advect_MR_term(
                self.Phi, self.Ash_Shadow, mr_interpolator,
                self.X, self.Y, self.Z,
                self.dx, self.dy, self.dz, dt, self.surface_angle
            )
            # Relaxation term to stabilize the level set field evolution.
            self.Phi = lv.advect_relax_term(
                self.Phi, self.Ash_Shadow, relax_term, self.dx, self.dy, self.dz, dt, self.surface_angle
            )

            # histories
            # Store scalar time series for later plotting/export.
            t_hist.append(t_cur)
            rho_hist.append(dens)
            sratio_hist.append(sratio)
            th1_hist.append(th1)
            th2_hist.append(th2)

            # --- PolyGaussian plot every update_time ---
            # If enabled, periodically refit the PolyGaussian parameters to the evolving surface and rebuild MR kernel.
            if self.plot_poly_geometry and N_update > 0 and (i % N_update == 0) and i > 0:
                # refit + refresh MR kernel
                Xs, Ys, Zs = lv.extract_surface(self.Phi, self.Lx, self.Ly, self.Lz)
                cur_surface = surf.Surface(Xs, Ys, Zs)
                # Update PolyGaussian fit to current geometry (fixed niter).
                poly_surface.fit_parameters(cur_surface, verbose=False, niter=8)
                scatterer = sct.Scatterer(poly_surface)
                # Rebuild MR interpolator using a less noisy incident direction.
                mr_interpolator = self.multi_reflection_flux(
                    num_multi_reflections,
                    self.erosion_interpolator,
                    self.reaction_interpolator,
                    v_inc_mean, dens, accommodation_coefficient,
                    T_surf, rho_s, height_lim, velocity_lim, scatterer
                )
                # Recompute α_E on the updated surface (first reflection).
                accommodation_coefficient = self.compute_energy_accommodation(
                    energy_function=self.energy_accommodation_interpolator,
                    incident_velocity=v_vec0,
                    surface_temperature=T0,
                    scatterer=scatterer,
                    first=True,
                )
                self.energy_accommodation_coeff = accommodation_coefficient
                # print("Energy accommodation", accommodation_coefficient)
                # plot & save (1.pdf, 2.pdf, …)
                pg_idx += 1

            # poly-gaussian parameter histories
            # Store evolving surface-model parameters for later analysis.
            mu_param_hist.append(poly_surface.mu_coeff)
            sigma_param_hist.append(poly_surface.sigma_coeff)
            r_hist.append(poly_surface.R)
            gamma_hist.append(gamma_hist[-1])

            # --- Erosion-history plot every plotting_time (and at final step) ---
            # If enabled, generate diagnostic plots of the evolving surface and time histories.
            if self.plot_history and ((t_cur >= next_hist_plot_t) or (i == Nt - 1)):
                ex.plot_erosion_history(
                    folder_name=eh_dir,
                    file_name=str(eh_idx),
                    X=self.X,
                    Y=self.Y,
                    Z=self.Z,
                    Phi=self.Phi,
                    sample_size=self.Ly,
                    time_data=np.asarray(t_hist),
                    erosion_yield_data=np.asarray(EY_hist),
                    AO_fluence_data=np.asarray(AO_fluence_hist),
                    density_data=np.asarray(rho_hist),
                    speed_ratio_data=np.asarray(sratio_hist),
                    incident_angle_1_data=np.asarray(th1_hist),
                    incident_angle_2_data=np.asarray(th2_hist),
                    surface_angle=self.surface_angle,
                    pos_x=self.ash_x,
                    pos_y=self.ash_y,
                    pos_z=self.ash_z,
                    radius=self.ash_r,
                    cmap_function=None
                )
                # PolyGaussian model vs surface plot for debugging/verification.
                ex.plot_polygaussian_surface_data(
                    folder_name=pg_dir,
                    file_name=str(eh_idx),
                    surface=cur_surface,
                    poly_surface=poly_surface,
                    surface_angle=self.surface_angle,
                )
                eh_idx += 1
                next_hist_plot_t += plotting_time if np.isfinite(next_hist_plot_t) else float("inf")

            # --- Erosion-history export every plotting_time (and at final step) ---
            # If enabled, export raw time series data to disk for post-processing.
            if self.export_history and ((t_cur >= next_hist_export_t) or (i == Nt - 1)):
                ex.export_erosion_history(
                    folder_name=eh_data_dir,
                    file_name="erosion_history",
                    start_date=self.sim_start_utc,
                    time_data=np.asarray(t_hist),
                    erosion_yield_data=np.asarray(EY_hist),
                    AO_fluence_data=np.asarray(AO_fluence_hist),
                    density_data=np.asarray(rho_hist),
                    speed_ratio_data=np.asarray(sratio_hist),
                    incident_angle_1_data=np.asarray(th1_hist),
                    incident_angle_2_data=np.asarray(th2_hist),
                    gamma_array_history=np.asarray(gamma_hist),
                    mu_array_history=np.asarray(mu_param_hist),
                    sigma_array_history=np.asarray(sigma_param_hist),
                    R_history=np.asarray(r_hist)
                )
                next_hist_export_t += plotting_time if np.isfinite(next_hist_export_t) else float("inf")
            # --- Erosion-history export every plotting_time (and at final step) ---
            # If enabled, export geometry snapshots (STL) and ash particle representations at the same cadence.
            if self.export_geometry and ((t_cur >= next_geo_export_t) or (i == Nt - 1)):
                ex.export_geometry_as_stl(
                    folder_name=geo_data_dir,
                    file_name=str(geo_idx),
                    X=self.X,
                    Y=self.Y,
                    Z=self.Z,
                    Phi=self.Phi,
                    refinement_factor=2
                )
                ex.export_ash_particles(
                    folder_name=geo_data_dir,
                    file_name=str(geo_idx) + "_ash",
                    pos_x=self.ash_x,
                    pos_y=self.ash_y,
                    pos_z=self.ash_z,
                    radius=self.ash_r,
                    X=self.X,
                    Y=self.Y,
                    Z=self.Z,
                    Phi=self.Phi,
                    refinement_factor=3
                )
                geo_idx += 1
                next_geo_export_t += plotting_time if np.isfinite(next_geo_export_t) else float("inf")

def main():
    """
    CLI entry point.

    Expects a single positional argument `project_name` pointing to the project folder
    containing Inputs.txt and the required data files imported by `impt.import_inputs`.
    """
    # Build a minimal command-line interface to select the project folder.
    parser = argparse.ArgumentParser(description="Run erosion simulation.")
    parser.add_argument(
        "project_name",
        type=str,
        help="Project folder name (e.g. 'test_project')."
    )
    args = parser.parse_args()

    # Instantiate and run the simulation.
    sim = Eroder(args.project_name)
    sim.simulate()

if __name__ == "__main__":
    # Standard Python executable-module guard.
    main()
