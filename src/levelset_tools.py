"""
Level set utilities for tilted periodic geometries.

This module provides helpers to:
- build an initial signed distance like level set field from a generated rough surface
- extract the zero level set surface from a 3D field
- compute surface normals using either a 2.5D surface extraction or a full 3D upwind gradient
- advance Phi under several advection and relaxation style update terms

Notes
- x is treated as periodic, with a z seam shift that depends on surface_angle
- z is treated as nonperiodic in the seam handling helper, with out of bounds values clamped to ±1
"""

import numpy as np
import scipy as sp
from scipy.interpolate import griddata

import surface_tools as surf
import scatter_tools as sct  # imported for downstream usage in the wider codebase


def _shift_z_nonperiodic_pm1(slice_yz: np.ndarray, shift_z: float, dz: float) -> np.ndarray:
    """
    Shift a (Ny, Nz) slice along z by shift_z with linear interpolation.
    z is NOT periodic.

    Values sampled below the bottom are set to -1.
    Values sampled above the top are set to +1.

    Convention
      out(z_j) = in(z_j - shift_z)

    Parameters
    slice_yz
        2D array with shape (Ny, Nz), representing a y z plane.
    shift_z
        Physical shift distance along z. Positive shift_z samples from lower indices.
    dz
        Grid spacing along z (must be positive and finite).

    Returns
    out
        Shifted slice with the same shape as slice_yz.
    """
    # Ensure we are working with a float array for interpolation
    slice_yz = np.asarray(slice_yz, dtype=float)
    Ny, Nz = slice_yz.shape

    # Trivial cases: nothing to do if the z axis is degenerate or shift is zero
    if Nz <= 1 or shift_z == 0.0:
        return slice_yz.copy()

    # Validate spacing
    dz = float(dz)
    if not np.isfinite(dz) or dz <= 0.0:
        raise ValueError("dz must be positive and finite.")

    # Convert physical shift into index shift
    s = shift_z / dz  # shift in index units

    # Destination indices j map to source indices (j - s)
    j = np.arange(Nz, dtype=float)[None, :]  # (1, Nz)
    src = j - s

    # Identify out of bounds samples before clipping
    below = src < 0.0
    above = src > (Nz - 1.0)

    # Clip source indices for interpolation within valid range
    src_clip = np.clip(src, 0.0, Nz - 1.0)
    j0 = np.floor(src_clip).astype(int)   # left index
    j1 = np.minimum(j0 + 1, Nz - 1)       # right index
    a = (src_clip - j0).astype(float)     # interpolation weight in [0, 1]

    # Gather values and interpolate along z for each y row
    rows = np.arange(Ny)[:, None]
    v0 = slice_yz[rows, j0]
    v1 = slice_yz[rows, j1]
    out = (1.0 - a) * v0 + a * v1

    # Apply the nonperiodic boundary convention for out of range samples
    out[:, below[0]] = -1.0
    out[:, above[0]] = +1.0
    return out


def rotate_geometry(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    Nx: int,
    Ny: int,
    angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate a 2D height field surface about the y axis and remesh onto a new regular grid.

    The input surface is described by (X, Y, Z) arrays. The rotation is performed in the x z plane:
      Xr =  X cos(angle) + Z sin(angle)
      Zr = -X sin(angle) + Z cos(angle)
      Yr =  Y

    The rotated points are then interpolated onto a new (Nx, Ny) mesh using:
    - linear interpolation where possible
    - nearest neighbor fallback for points outside the convex hull

    A slope safety check is performed to avoid near vertical mappings that would cause remeshing issues.

    Parameters
    X, Y, Z
        Arrays describing the surface geometry.
    Nx, Ny
        Target grid resolution in x and y for the remeshed surface.
    angle
        Rotation angle in radians.

    Returns
    X_new, Y_new, Z_new
        Remeshed rotated geometry on a regular grid with indexing="ij".
    """
    # Precompute trig terms for the rotation
    ca, sa = np.cos(angle), np.sin(angle)

    # Estimate x spacing to evaluate slope with respect to x
    dx = float(np.median(np.diff(X[0]))) if X.shape[1] > 1 else 1.0

    # Slope safety check: denom corresponds to Jacobian like term for the mapping
    dZdx = np.gradient(Z, dx, axis=1)
    denom = ca + sa * dZdx
    if np.nanmin(np.abs(denom)) < 1e-6:
        raise ValueError("Rotation creates |slope| >= 90 deg; cannot remesh safely.")

    # Apply rotation in the x z plane
    Xr = X * ca + Z * sa
    Yr = Y
    Zr = -X * sa + Z * ca

    # Determine extents to build a centered target grid
    x_min, x_max = np.min(Xr), np.max(Xr)
    y_min, y_max = np.min(Yr), np.max(Yr)
    Lx, Ly = (x_max - x_min), (y_max - y_min)

    # New grid is centered at zero, spanning the rotated extents
    x_new = np.linspace(-Lx / 2.0, Lx / 2.0, Nx)
    y_new = np.linspace(-Ly / 2.0, Ly / 2.0, Ny)
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing="ij")

    # Interpolate rotated points to the new mesh
    pts = np.c_[Xr.ravel(), Yr.ravel()]
    Z_lin = griddata(pts, Zr.ravel(), (X_new, Y_new), method="linear")
    Z_nei = griddata(pts, Zr.ravel(), (X_new, Y_new), method="nearest")

    # Use linear results when defined, fall back to nearest where needed
    Z_new = np.where(np.isfinite(Z_lin), Z_lin, Z_nei)

    return X_new, Y_new, Z_new


def initialise_levelset_function(
    initial_surface: surf.PolyGaussian_Surface | None = None,
    surface_angle: float = 0.0,
    sample_length: float = 10.0,
    Nx: int = 100,
    Ny: int = 100,
    Nz: int = 100,
    upper_padding: float = 2.0,
    lower_padding: float = 20.0,
    iterations: int = 150,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an initial level set field Phi from a rough surface model.

    Workflow
    1) Generate a poly Gaussian rough surface over a square sample.
    2) Center heights by subtracting the mean.
    3) Optionally scale Z to respect a target maximum slope implied by surface_angle.
    4) Rotate the geometry by surface_angle and remesh to (Nx, Ny).
    5) Build a 3D grid (Nx, Ny, Nz) with padding above and below the surface.
    6) Construct Phi as a normalized signed distance like field in z grid units:
         Phi = (Z_grid - Z_surface) / dz
       then clamp to [-1, 1] for stability in later advection steps.

    Parameters
    initial_surface
        PolyGaussian_Surface instance or None to construct a default one.
    surface_angle
        Tilt angle in radians used for both slope conditioning and geometry rotation.
    sample_length
        Physical side length of the generated surface patch.
    Nx, Ny, Nz
        Grid resolution for the final Phi field and coordinate arrays.
    upper_padding, lower_padding
        Extra space added above and below the surface in the z direction.
    iterations
        Iterations passed to the surface generator.
    verbose
        Forwarded to the surface generator.

    Returns
    Phi, X_phi, Y_phi, Z_phi
        Phi is clamped to [-1, 1] and all coordinate arrays are shaped (Nx, Ny, Nz).
    """
    # Default surface generator if none is provided
    if initial_surface is None:
        initial_surface = surf.PolyGaussian_Surface()

    # Generate a surface and convert to float arrays
    surface = initial_surface.generate(sample_length, Ny, iterations=iterations, verbose=verbose)
    X, Y, Z = surface.X, surface.Y, surface.Z

    # Center the surface heights around zero mean
    Z -= np.mean(Z)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    Z = np.asarray(Z, float)

    # Estimate slope magnitudes on the initial grid
    h = float(sample_length) / max(int(Ny), 1)
    dZdx = np.gradient(Z, h, axis=0)
    dZdy = np.gradient(Z, h, axis=1)
    max_slope = float(np.nanmax(np.abs([dZdx, dZdy]))) if np.isfinite(dZdx).any() and np.isfinite(dZdy).any() else 0.0

    # Compute a target slope scaling based on the angle convention used elsewhere
    target = np.tan(np.pi / 2.0 + float(surface_angle))

    # If target is finite and positive, scale Z down to avoid exceeding target
    if np.isfinite(target) and target > 0.0 and max_slope > 0.0:
        Z *= min(target / max_slope, 1.0)

    # Rotate and remesh the surface to the desired x y resolution
    X_r, Y_r, Z_r = rotate_geometry(X, Y, Z, Nx, Ny, surface_angle)

    # Compute domain extents after rotation and define the z grid with padding
    Lx = float(np.max(X_r) - np.min(X_r))
    Ly = float(np.max(Y_r) - np.min(Y_r))
    Hz = float(np.max(Z_r) - np.min(Z_r))
    Lz = Hz + float(upper_padding) + float(lower_padding)

    # Coordinate arrays for Phi grid
    x_phi = np.linspace(-Lx / 2.0, Lx / 2.0, int(Nx))
    y_phi = np.linspace(-Ly / 2.0, Ly / 2.0, int(Ny))
    z_phi = np.linspace(-Hz / 2.0 - float(lower_padding), Hz / 2.0 + float(upper_padding), int(Nz))
    X_phi, Y_phi, Z_phi = np.meshgrid(x_phi, y_phi, z_phi, indexing="ij")

    # Normalize distances by dz so Phi is expressed in grid units along z
    dz = (z_phi[1] - z_phi[0]) if Nz > 1 else 1.0
    Phi = (Z_phi - Z_r[..., None]) / dz

    # Clamp to ±1 to keep values bounded for downstream upwind logic
    return np.clip(Phi, -1.0, 1.0), X_phi, Y_phi, Z_phi


def extract_surface(
    Phi: np.ndarray,
    Lx: float,
    Ly: float,
    Lz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the zero crossing surface z0(x, y) from a 3D level set field Phi(x, y, z).

    The method searches along z for each (x, y) column to find the first index k where:
      Phi[..., k] < 0 and Phi[..., k+1] >= 0
    and then linearly interpolates the z location of Phi = 0.

    If there is no crossing in a column:
    - if all Phi >= 0, assign the minimum z
    - if all Phi < 0, assign the maximum z
    - otherwise fall back to the z index with smallest |Phi|

    Parameters
    Phi
        Level set field of shape (Nx, Ny, Nz).
    Lx, Ly, Lz
        Physical extents used to reconstruct coordinate grids.

    Returns
    X, Y, Z0
        X and Y are 2D grids shaped (Nx, Ny). Z0 is the extracted surface height field.
    """
    # Ensure floating arithmetic and unpack grid sizes
    Phi = np.asarray(Phi, float)
    Nx, Ny, Nz = Phi.shape

    # Construct z coordinate array consistent with given extent
    z = np.linspace(-Lz / 2.0, Lz / 2.0, Nz, dtype=float)

    # Construct x y coordinate grids consistent with given extents
    x = np.linspace(-Lx / 2.0, Lx / 2.0, Nx, dtype=float)
    y = np.linspace(-Ly / 2.0, Ly / 2.0, Ny, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Degenerate z axis: return a constant surface at the only z value
    if Nz < 2:
        Z0 = np.full((Nx, Ny), z[0], dtype=float)
        return X, Y, Z0

    # Identify sign changes from negative to nonnegative between adjacent z samples
    neg = Phi[..., :-1] < 0.0
    pos_next = Phi[..., 1:] >= 0.0
    cross = neg & pos_next

    # For each (x, y), find if any crossing exists and the first crossing index
    any_cross = np.any(cross, axis=2)
    k = np.argmax(cross, axis=2)
    k = np.where(any_cross, k, -1)

    # Gather neighboring Phi and z values for linear interpolation
    ii, jj = np.indices((Nx, Ny))
    k1 = np.where(k >= 0, k, 0)
    k2 = np.clip(k1 + 1, 1, Nz - 1)

    phi1 = Phi[ii, jj, k1]
    phi2 = Phi[ii, jj, k2]
    z1 = z[k1]
    z2 = z[k2]

    # Avoid divide by zero when phi2 and phi1 are extremely close
    denom = np.where(np.abs(phi2 - phi1) > 1e-14, phi2 - phi1, 1e-14)

    # Linear interpolation for the zero crossing
    z0_cross = z1 - phi1 * (z2 - z1) / denom

    # Handle columns with no crossings using sign tests
    all_pos = np.all(Phi >= 0.0, axis=2)
    all_neg = np.all(Phi < 0.0, axis=2)
    Z0 = np.where(k >= 0, z0_cross, np.nan)
    Z0 = np.where(np.isnan(Z0) & all_pos, z.min(), Z0)
    Z0 = np.where(np.isnan(Z0) & all_neg, z.max(), Z0)

    # Fallback for ambiguous columns: pick the z with minimal |Phi|
    ambig = np.isnan(Z0)
    if np.any(ambig):
        kmin = np.argmin(np.abs(Phi), axis=2)
        Z0 = np.where(ambig, z[kmin], Z0)

    return X, Y, np.clip(Z0, z.min(), z.max())


def extract_unbiased_surface(
    Phi: np.ndarray,
    Lx: float,
    Ly: float,
    Lz: float,
    surf_angle: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the zero crossing surface and then convert it to an isotropic surface representation.

    This wraps extract_surface style logic, then constructs a surf.Surface and calls
    get_isotropic_surface(surf_angle) to remove the bias introduced by the tilt convention.

    Parameters
    Phi
        Level set field of shape (Nx, Ny, Nz).
    Lx, Ly, Lz
        Physical extents used to reconstruct coordinate grids.
    surf_angle
        Surface tilt angle passed to get_isotropic_surface.

    Returns
    X_unb, Y_unb, Z_unb
        Isotropic surface coordinates returned by surf.Surface.get_isotropic_surface.
    """
    # Ensure float and unpack grid sizes
    Phi = np.asarray(Phi, float)
    Nx, Ny, Nz = Phi.shape

    # Coordinate reconstruction from extents
    z = np.linspace(-Lz / 2.0, Lz / 2.0, Nz, dtype=float)

    x = np.linspace(-Lx / 2.0, Lx / 2.0, Nx, dtype=float)
    y = np.linspace(-Ly / 2.0, Ly / 2.0, Ny, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Degenerate z axis: build a constant surface and immediately isotropize it
    if Nz < 2:
        Z0 = np.full((Nx, Ny), z[0], dtype=float)
        s_biased = surf.Surface(X, Y, Z0)
        s_unb = s_biased.get_isotropic_surface(surf_angle)
        return s_unb.X, s_unb.Y, s_unb.Z

    # Detect negative to nonnegative crossings along z
    neg = Phi[..., :-1] < 0.0
    pos_next = Phi[..., 1:] >= 0.0
    cross = neg & pos_next
    any_cross = np.any(cross, axis=2)
    k = np.argmax(cross, axis=2)
    k = np.where(any_cross, k, -1)

    # Linear interpolation setup
    ii, jj = np.indices((Nx, Ny))
    k1 = np.where(k >= 0, k, 0)
    k2 = np.clip(k1 + 1, 1, Nz - 1)

    phi1 = Phi[ii, jj, k1]
    phi2 = Phi[ii, jj, k2]
    z1 = z[k1]
    z2 = z[k2]
    denom = np.where(np.abs(phi2 - phi1) > 1e-14, phi2 - phi1, 1e-14)
    z0_cross = z1 - phi1 * (z2 - z1) / denom

    # Fill missing values using sign tests
    all_pos = np.all(Phi >= 0.0, axis=2)
    all_neg = np.all(Phi < 0.0, axis=2)
    Z0 = np.where(k >= 0, z0_cross, np.nan)
    Z0 = np.where(np.isnan(Z0) & all_pos, z.min(), Z0)
    Z0 = np.where(np.isnan(Z0) & all_neg, z.max(), Z0)

    # Ambiguous fallback to smallest |Phi|
    ambig = np.isnan(Z0)
    if np.any(ambig):
        kmin = np.argmin(np.abs(Phi), axis=2)
        Z0 = np.where(ambig, z[kmin], Z0)

    # Ensure extracted surface is within z bounds
    Z0 = np.clip(Z0, z.min(), z.max())

    # Convert from biased representation to isotropic representation
    s_biased = surf.Surface(X, Y, Z0)
    s_unb = s_biased.get_isotropic_surface(surf_angle)
    return s_unb.X, s_unb.Y, s_unb.Z


def get_normals(
    Phi: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    surface_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute surface normals on the Phi grid using an extracted 2D surface representation.

    Steps
    1) Extract the zero level set surface Zs(x, y) from Phi.
    2) Compute derivatives dZdx and dZdy on that surface, treating x as periodic with a z seam shift
       implied by surface_angle.
    3) Form the unit normal components:
         n = (-dZdx, -dZdy, 1) / sqrt(1 + dZdx^2 + dZdy^2)
    4) Broadcast the 2D normal field to 3D arrays matching Phi shape.

    Parameters
    Phi, X, Y, Z
        Arrays of identical shape (Nx, Ny, Nz) defining the level set grid.
    surface_angle
        Tilt angle used to correct the x periodic seam in the extracted surface.

    Returns
    NX, NY, NZ
        Normal components broadcast to shape (Nx, Ny, Nz).
    """
    # Validate shape consistency early
    if not (Phi.shape == X.shape == Y.shape == Z.shape):
        raise ValueError("Phi, X, Y, Z must have identical shapes.")
    Nx, Ny, Nz = Phi.shape

    # Compute extents from coordinate arrays to reconstruct consistent 2D grids
    Lx = float(np.nanmax(X) - np.nanmin(X)) if Nx > 1 else 1.0
    Ly = float(np.nanmax(Y) - np.nanmin(Y)) if Ny > 1 else 1.0
    Lz = float(np.nanmax(Z) - np.nanmin(Z)) if Nz > 1 else 1.0

    # Extract the surface height field from Phi
    Xs, Ys, Zs = extract_surface(Phi, Lx, Ly, Lz)
    Xs = np.asarray(Xs, float)
    Ys = np.asarray(Ys, float)
    Zs = np.asarray(Zs, float)

    # Helper to estimate spacing robustly from coordinate grids
    def _dx1d(a: np.ndarray, axis: int, extent: float, n: int) -> float:
        # If only one sample, treat spacing as 1 to avoid division by zero
        if n <= 1:
            return 1.0

        # Use median diffs along the appropriate axis when possible
        if axis == 0:
            d = float(np.nanmedian(np.diff(a[:, 0])))
        else:
            d = float(np.nanmedian(np.diff(a[0, :])))

        # Fallback to extent based spacing if diffs are invalid
        if not np.isfinite(d) or d == 0.0:
            d = extent / max(n - 1, 1)
        return d

    dx = _dx1d(Xs, 0, Lx, Nx)
    dy = _dx1d(Ys, 1, Ly, Ny)

    # Tilt correction convention: subtract linear trend in x before y derivatives
    m = -np.tan(float(surface_angle))
    Zc = Zs - m * Xs

    # Periodic seam in x corresponds to an offset in z for the tilted surface
    Lx_period = dx * Nx
    dz_seam = m * Lx_period

    # Central difference in x with seam corrected boundary rows
    if Nx > 1:
        Zp = np.roll(Zs, -1, axis=0)
        Zm = np.roll(Zs, 1, axis=0)
        Zm[0, :] = Zs[-1, :] - dz_seam
        Zp[-1, :] = Zs[0, :] + dz_seam
        dZdx = (Zp - Zm) / (2.0 * dx)
    else:
        dZdx = np.zeros_like(Zs)

    # Central difference in y using the corrected surface Zc
    if Ny > 1:
        dZdy = (np.roll(Zc, -1, axis=1) - np.roll(Zc, 1, axis=1)) / (2.0 * dy)
    else:
        dZdy = np.zeros_like(Zs)

    # Normalize to unit vectors, guarding against invalid magnitudes
    nmag = np.sqrt(1.0 + dZdx * dZdx + dZdy * dZdy)
    nmag = np.where(np.isfinite(nmag) & (nmag > 0.0), nmag, 1.0)

    nx2d = -dZdx / nmag
    ny2d = -dZdy / nmag
    nz2d = 1.0 / nmag

    # Replace NaNs and infs with zeros for robustness
    nx2d = np.nan_to_num(nx2d, nan=0.0, posinf=0.0, neginf=0.0)
    ny2d = np.nan_to_num(ny2d, nan=0.0, posinf=0.0, neginf=0.0)
    nz2d = np.nan_to_num(nz2d, nan=0.0, posinf=0.0, neginf=0.0)

    # Broadcast 2D normals to the full 3D grid shape
    NX = np.broadcast_to(nx2d[:, :, None], (Nx, Ny, Nz))
    NY = np.broadcast_to(ny2d[:, :, None], (Nx, Ny, Nz))
    NZ = np.broadcast_to(nz2d[:, :, None], (Nx, Ny, Nz))
    return NX, NY, NZ


def get_normals_full(
    Phi: np.ndarray,
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    flux_z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    surface_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute normals from the full 3D upwind gradient of Phi.

    This computes dPhi/dx, dPhi/dy, dPhi/dz using forward or backward differences chosen
    by the sign of the supplied flux components (fx, fy, fz). This is useful for flux aligned
    advection terms where an upwind consistent normal is desired.

    Special handling is applied for x periodicity with a tilted seam:
    the boundary differences at x = 0 and x = Nx-1 use shifted z slices to respect the
    surface_angle induced z offset.

    Parameters
    Phi
        Level set field of shape (Nx, Ny, Nz).
    flux_x, flux_y, flux_z
        Flux direction fields used to select upwind differencing.
    X, Y, Z
        Coordinate arrays of identical shape, used to infer grid spacing.
    surface_angle
        Tilt angle used to compute the x seam shift.

    Returns
    nx, ny, nz
        Unit normals computed as grad(Phi) / |grad(Phi)| with shape (Nx, Ny, Nz).
    """
    # Validate that coordinate arrays match Phi
    if not (Phi.shape == X.shape == Y.shape == Z.shape):
        raise ValueError("Phi, X, Y, Z must have identical shapes.")

    # Cast inputs to float arrays
    Phi = np.asarray(Phi, float)
    fx = np.asarray(flux_x, float)
    fy = np.asarray(flux_y, float)
    fz = np.asarray(flux_z, float)
    Nx, Ny, Nz = Phi.shape

    # Robust spacing estimator along a given axis
    def _spacing(A: np.ndarray, axis: int) -> float:
        # If axis has length 1, spacing is arbitrary but nonzero
        if A.shape[axis] <= 1:
            return 1.0

        # Median diff is robust to small irregularities
        d = float(np.nanmedian(np.diff(A, axis=axis)))

        # Fallback to extent based spacing if diffs are invalid
        if not np.isfinite(d) or d == 0.0:
            lo, hi = float(np.nanmin(A)), float(np.nanmax(A))
            n = A.shape[axis]
            d = (hi - lo) / (n - 1) if (np.isfinite([lo, hi]).all() and hi > lo and n > 1) else 1.0
        return d

    dx = _spacing(X, 0)
    dy = _spacing(Y, 1)
    dz = _spacing(Z, 2)
    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz

    # Compute the z seam shift associated with x periodicity under a tilt
    Lx_period = dx * Nx
    dz_seam = -np.tan(float(surface_angle)) * Lx_period

    # Backward and forward differences in x
    dBx = (Phi - np.roll(Phi, 1, axis=0)) * inv_dx
    dFx = (np.roll(Phi, -1, axis=0) - Phi) * inv_dx

    # Correct the x seam differences using nonperiodic z shifting for boundary slices
    if Nx > 1 and dz_seam != 0.0:
        adj = _shift_z_nonperiodic_pm1(Phi[-1, :, :], +dz_seam, dz)
        dBx[0, :, :] = (Phi[0, :, :] - adj) * inv_dx

        adj = _shift_z_nonperiodic_pm1(Phi[0, :, :], -dz_seam, dz)
        dFx[-1, :, :] = (adj - Phi[-1, :, :]) * inv_dx

    # Backward and forward differences in y (simple periodic roll)
    dBy = (Phi - np.roll(Phi, 1, axis=1)) * inv_dy
    dFy = (np.roll(Phi, -1, axis=1) - Phi) * inv_dy

    # Backward and forward differences in z (nonperiodic, one sided at boundaries)
    dBz = np.zeros_like(Phi)
    dFz = np.zeros_like(Phi)
    if Nz > 1:
        tmp = (Phi[:, :, 1:] - Phi[:, :, :-1]) * inv_dz
        dFz[:, :, :-1] = tmp
        dBz[:, :, 1:] = tmp

    # Upwind selection based on flux sign at each cell
    dphidx = np.where(fx <= 0.0, dBx, dFx)
    dphidy = np.where(fy <= 0.0, dBy, dFy)
    dphidz = np.where(fz <= 0.0, dBz, dFz)

    # Normalize the gradient to obtain unit normals
    grad_mag = np.sqrt(dphidx * dphidx + dphidy * dphidy + dphidz * dphidz)
    nx = np.divide(dphidx, grad_mag, out=np.zeros_like(Phi), where=grad_mag > 0.0)
    ny = np.divide(dphidy, grad_mag, out=np.zeros_like(Phi), where=grad_mag > 0.0)
    nz = np.divide(dphidz, grad_mag, out=np.zeros_like(Phi), where=grad_mag > 0.0)
    return nx, ny, nz


def relax_levelset(
    Phi: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    n_iters: int = 50,
    dt: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Reinitialize Phi toward a signed distance function using an iterative PDE relaxation.

    This implements a standard reinitialization style update:
      Phi_t + s(Phi0) (|grad Phi| - 1) = 0
    where s(Phi0) is a smoothed sign function of the initial field Phi0.

    The update is performed using upwind like Godunov norms for |grad Phi|.

    Parameters
    Phi
        Initial level set field, shape (Nx, Ny, Nz).
    dx, dy, dz
        Grid spacings, must be positive.
    n_iters
        Number of relaxation iterations.
    dt
        Time step for the pseudo time integration. If None, chosen as 0.3 * min(dx, dy, dz).
    eps
        Smoothing parameter for the sign function.

    Returns
    Phi
        Relaxed Phi field.
    """
    # Work on a copy to avoid in place modification of the input
    Phi = np.asarray(Phi, dtype=float).copy()
    if Phi.ndim != 3:
        raise ValueError("Phi must be 3-D (Nx,Ny,Nz).")
    if min(dx, dy, dz) <= 0:
        raise ValueError("dx, dy, dz must be positive.")

    # Choose a conservative default time step
    if dt is None:
        dt = 0.3 * min(dx, dy, dz)

    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz

    # Phi0 is used only for the sign term, per standard reinitialization
    Phi0 = Phi.copy()
    s = Phi0 / np.sqrt(Phi0 * Phi0 + eps * eps)

    Nx, Ny, Nz = Phi.shape
    for _ in range(int(max(1, n_iters))):
        # Allocate forward and backward differences in each direction
        dFx = np.zeros_like(Phi)
        dBx = np.zeros_like(Phi)
        dFy = np.zeros_like(Phi)
        dBy = np.zeros_like(Phi)
        dFz = np.zeros_like(Phi)
        dBz = np.zeros_like(Phi)

        # Forward and backward differences in x
        dFx[:-1, :, :] = (Phi[1:, :, :] - Phi[:-1, :, :]) * inv_dx
        dBx[1:, :, :] = (Phi[1:, :, :] - Phi[:-1, :, :]) * inv_dx

        # Forward and backward differences in y
        dFy[:, :-1, :] = (Phi[:, 1:, :] - Phi[:, :-1, :]) * inv_dy
        dBy[:, 1:, :] = (Phi[:, 1:, :] - Phi[:, :-1, :]) * inv_dy

        # Forward and backward differences in z
        dFz[:, :, :-1] = (Phi[:, :, 1:] - Phi[:, :, :-1]) * inv_dz
        dBz[:, :, 1:] = (Phi[:, :, 1:] - Phi[:, :, :-1]) * inv_dz

        # Godunov scheme components for s >= 0
        a_plus = np.maximum(np.maximum(dBx, 0.0) ** 2, np.maximum(-dFx, 0.0) ** 2)
        b_plus = np.maximum(np.maximum(dBy, 0.0) ** 2, np.maximum(-dFy, 0.0) ** 2)
        c_plus = np.maximum(np.maximum(dBz, 0.0) ** 2, np.maximum(-dFz, 0.0) ** 2)

        # Godunov scheme components for s < 0
        a_minus = np.maximum(np.maximum(dFx, 0.0) ** 2, np.maximum(-dBx, 0.0) ** 2)
        b_minus = np.maximum(np.maximum(dFy, 0.0) ** 2, np.maximum(-dBy, 0.0) ** 2)
        c_minus = np.maximum(np.maximum(dFz, 0.0) ** 2, np.maximum(-dBz, 0.0) ** 2)

        # Select appropriate norm based on sign
        grad_plus = np.sqrt(a_plus + b_plus + c_plus)
        grad_minus = np.sqrt(a_minus + b_minus + c_minus)
        grad = np.where(s >= 0.0, grad_plus, grad_minus)

        # Update Phi toward |grad Phi| = 1
        Phi -= dt * s * (grad - 1.0)

    return Phi


def advect_FR_term(
    Phi: np.ndarray,
    Ash_Shadow: np.ndarray,
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    flux_z: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    surface_angle: float,
) -> np.ndarray:
    """
    Apply an advection update term of the form:
      Phi_new = Phi + dt * (f · grad Phi) * Ash_Shadow

    Upwind differencing is used for each gradient component based on the flux sign.
    x periodic boundaries are corrected with a nonperiodic z shift that depends on surface_angle.

    Parameters
    Phi
        Level set field, shape (Nx, Ny, Nz).
    Ash_Shadow
        Multiplicative mask field, same shape as Phi.
    flux_x, flux_y, flux_z
        Flux components that define the advection direction and magnitude.
    dx, dy, dz
        Grid spacings.
    dt
        Time step.
    surface_angle
        Tilt angle used to compute the x seam z shift.

    Returns
    Phi_new
        Updated Phi field after applying the FR term.
    """
    # Basic numeric validation and type normalization
    dx, dy, dz, dt = map(float, (dx, dy, dz, dt))
    if not np.isfinite([dx, dy, dz, dt]).all() or min(dx, dy, dz) <= 0.0:
        raise ValueError("dx, dy, dz must be > 0 and dt finite.")

    # Cast arrays and validate shape agreement
    Phi = np.asarray(Phi, float)
    Ash_Shadow = np.asarray(Ash_Shadow, float)
    fx = np.asarray(flux_x, float)
    fy = np.asarray(flux_y, float)
    fz = np.asarray(flux_z, float)
    if Phi.shape != Ash_Shadow.shape:
        raise ValueError("Phi and Ash_Shadow must have identical shapes.")

    Nx, Ny, Nz = Phi.shape
    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz

    # x differences, periodic roll by default
    dBx = (Phi - np.roll(Phi, 1, axis=0)) * inv_dx
    dFx = (np.roll(Phi, -1, axis=0) - Phi) * inv_dx

    # Correct the x seam using the tilted z shift
    if Nx > 1:
        Lx_period = dx * Nx
        dz_shift = -np.tan(float(surface_angle)) * Lx_period

        right_shifted = _shift_z_nonperiodic_pm1(Phi[-1, :, :], -dz_shift, dz)
        dBx[0, :, :] = (Phi[0, :, :] - right_shifted) * inv_dx

        left_shifted = _shift_z_nonperiodic_pm1(Phi[0, :, :], +dz_shift, dz)
        dFx[-1, :, :] = (left_shifted - Phi[-1, :, :]) * inv_dx

    # y differences, periodic roll
    dBy = (Phi - np.roll(Phi, 1, axis=1)) * inv_dy
    dFy = (np.roll(Phi, -1, axis=1) - Phi) * inv_dy

    # z differences, nonperiodic one sided at boundaries
    dBz = np.zeros_like(Phi)
    dFz = np.zeros_like(Phi)
    if Nz > 1:
        tmp = (Phi[:, :, 1:] - Phi[:, :, :-1]) * inv_dz
        dFz[:, :, :-1] = tmp
        dBz[:, :, 1:] = tmp

    # Upwind selection per component
    dphix = np.where(fx <= 0.0, dBx, dFx)
    dphiy = np.where(fy <= 0.0, dBy, dFy)
    dphiz = np.where(fz <= 0.0, dBz, dFz)

    # print("FR term", np.average((fx * dphix + fy * dphiy + fz * dphiz) * Ash_Shadow))
    return Phi + dt * (fx * dphix + fy * dphiy + fz * dphiz) * Ash_Shadow


def advect_MR_term(
    Phi: np.ndarray,
    Ash_Shadow: np.ndarray,
    MR_function: sp.interpolate.RegularGridInterpolator,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    surface_angle: float,
) -> np.ndarray:
    """
    Apply an update term driven by a precomputed MR flux function evaluated on local angles and height.

    High level form:
      Phi_new = Phi + dt * flux_MR(theta1, theta2, height) * |grad Phi|_upwind

    Steps
    1) Extract the surface from Phi and isotropize it with surface_angle.
    2) Compute normals from Phi and convert them to spherical angles theta1 and theta2.
    3) Build a heights field compatible with Phi shape.
    4) Compute an upwind consistent approximation of |grad Phi| using one sided differences.
    5) Evaluate MR_function at (theta1, theta2, height) for every grid cell.
    6) Update Phi with flux_MR * grad_phi * dt.

    Parameters
    Phi
        Level set field, shape (Nx, Ny, Nz).
    Ash_Shadow
        Multiplicative mask field, same shape as Phi.
    MR_function
        RegularGridInterpolator mapping (theta1, theta2, height) to a flux value.
    X, Y, Z
        Coordinate arrays, used for normal computation.
    dx, dy, dz
        Grid spacings.
    dt
        Time step, must be nonnegative.
    surface_angle
        Tilt angle used for isotropization and seam corrections.

    Returns
    Phi_new
        Updated Phi field after applying the MR term.
    """
    # Numeric validation and normalization
    dx, dy, dz, dt = map(float, (dx, dy, dz, dt))
    if not np.isfinite([dx, dy, dz, dt]).all() or min(dx, dy, dz) <= 0.0 or dt < 0.0:
        raise ValueError("dx, dy, dz must be > 0 and dt >= 0.")

    # Cast and validate fields
    Phi = np.asarray(Phi, float)
    Ash_Shadow = np.asarray(Ash_Shadow, float)
    if Phi.shape != Ash_Shadow.shape:
        raise ValueError("Phi and Ash_Shadow must have identical shapes.")
    Nx, Ny, Nz = Phi.shape
    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz

    # Extract and isotropize the surface used to define height input to MR_function
    Lx, Ly, Lz = (Nx - 1.0) * dx, (Ny - 1.0) * dy, (Nz - 1.0) * dz
    Xs, Ys, Zs = extract_surface(Phi, Lx, Ly, Lz)
    iso_surface = surf.Surface(Xs, Ys, Zs).get_isotropic_surface(surface_angle)
    Xs, Ys, Zs = iso_surface.X, iso_surface.Y, iso_surface.Z

    # Center heights, typically useful when MR tables assume zero mean reference
    Zs -= np.mean(Zs)

    # Compute normals and ensure we have 2D normal components for angle conversion
    nx, ny, nz = get_normals(Phi, X, Y, Z, surface_angle)
    nx2d = nx[..., 0] if nx.ndim == 3 else nx
    ny2d = ny[..., 0] if ny.ndim == 3 else ny
    nz2d = nz[..., 0] if nz.ndim == 3 else nz

    # Convert normals to angles: theta1 is polar angle, theta2 is azimuth
    theta1_2d = np.arccos(np.clip(nz2d, -1.0, 1.0))
    theta2_2d = np.mod(np.arctan2(ny2d, nx2d), 2.0 * np.pi)

    # Broadcast angles to full 3D shape for table evaluation
    theta1 = np.broadcast_to(theta1_2d[..., None], Phi.shape)
    theta2 = np.broadcast_to(theta2_2d[..., None], Phi.shape)

    # Broadcast surface heights to full 3D field for table evaluation
    if Zs.shape == Phi.shape:
        heights = Zs.astype(float, copy=False)
    elif Zs.shape == (Nx, Ny):
        heights = np.broadcast_to(Zs[..., None], Phi.shape).astype(float, copy=False)
    else:
        raise ValueError("Z must be shape (Nx,Ny,Nz) or (Nx,Ny).")

    # x differences with seam correction
    dBx = (Phi - np.roll(Phi, 1, axis=0)) * inv_dx
    dFx = (np.roll(Phi, -1, axis=0) - Phi) * inv_dx

    if Nx > 1:
        Lx_period = dx * Nx
        dz_shift = -np.tan(float(surface_angle)) * Lx_period

        right_shifted = _shift_z_nonperiodic_pm1(Phi[-1, :, :], -dz_shift, dz)
        dBx[0, :, :] = (Phi[0, :, :] - right_shifted) * inv_dx

        left_shifted = _shift_z_nonperiodic_pm1(Phi[0, :, :], +dz_shift, dz)
        dFx[-1, :, :] = (left_shifted - Phi[-1, :, :]) * inv_dx

    # y differences
    dBy = (Phi - np.roll(Phi, 1, axis=1)) * inv_dy
    dFy = (np.roll(Phi, -1, axis=1) - Phi) * inv_dy

    # z differences
    dBz = np.zeros_like(Phi)
    dFz = np.zeros_like(Phi)
    if Nz > 1:
        tmp = (Phi[:, :, 1:] - Phi[:, :, :-1]) * inv_dz
        dFz[:, :, :-1] = tmp
        dBz[:, :, 1:] = tmp

    # Godunov style norm for |grad Phi| used in many level set advection schemes
    grad_phi = np.sqrt(
        np.minimum(dBx, 0.0) ** 2 + np.maximum(dFx, 0.0) ** 2 +
        np.minimum(dBy, 0.0) ** 2 + np.maximum(dFy, 0.0) ** 2 +
        np.minimum(dBz, 0.0) ** 2 + np.maximum(dFz, 0.0) ** 2
    )

    # Evaluate MR flux lookup table at every cell
    pts = np.column_stack((theta1.ravel(), theta2.ravel(), heights.ravel()))
    flux_MR = MR_function(pts).reshape(Phi.shape)

    # print("Heights", np.min(heights), np.max(heights))
    # print("MR term", np.average(flux_MR * grad_phi))
    return Phi + flux_MR * grad_phi * dt


def advect_relax_term(
    Phi: np.ndarray,
    Ash_Shadow: np.ndarray,
    relax_coefficient: float | np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    surface_angle: float,
) -> np.ndarray:
    """
    Apply a diffusion like relaxation update term:
      Phi_new = Phi + dt * (nu * Laplacian(Phi)) * Ash_Shadow

    The x direction uses periodic rolls with a nonperiodic z shift at the seam
    implied by surface_angle. y is periodic by roll. z uses edge padding to form
    a simple second derivative.

    A stability limit is applied to dt based on the maximum nu value and grid spacings.

    Parameters
    Phi
        Level set field, shape (Nx, Ny, Nz).
    Ash_Shadow
        Multiplicative mask field, same shape as Phi.
    relax_coefficient
        Scalar or array nu controlling diffusion strength.
    dx, dy, dz
        Grid spacings, must be positive.
    dt
        Time step, must be nonnegative. Will be reduced if required for stability.
    surface_angle
        Tilt angle used to compute the x seam z shift.

    Returns
    Phi_new
        Updated Phi field after applying the relaxation term.
    """
    # Cast arrays and validate inputs
    Phi = np.asarray(Phi, float)
    Ash_Shadow = np.asarray(Ash_Shadow, float)
    nu = np.asarray(relax_coefficient, float)
    if Phi.ndim != 3:
        raise ValueError("Phi must be (Nx,Ny,Nz).")
    if not np.isfinite([dx, dy, dz, dt]).all() or min(dx, dy, dz) <= 0.0 or dt < 0.0:
        raise ValueError("dx,dy,dz > 0 and dt >= 0 required.")

    Nx, Ny, Nz = Phi.shape
    inv_dx2, inv_dy2, inv_dz2 = 1.0 / dx ** 2, 1.0 / dy ** 2, 1.0 / dz ** 2

    # CFL like stability restriction for explicit diffusion
    nu_max = float(np.nanmax(nu)) if nu.size else float(nu)
    if np.isfinite(nu_max) and nu_max > 0.0:
        dt = min(dt, 0.49 / (nu_max * (inv_dx2 + inv_dy2 + inv_dz2)))

    # Compute the x seam z shift for a tilted periodic domain
    Lx_period = dx * Nx
    dz_shift = -np.tan(float(surface_angle)) * Lx_period

    # Periodic neighbors in x, with seam corrected boundaries via z shifts
    phi_xp = np.roll(Phi, -1, axis=0)
    phi_xm = np.roll(Phi, 1, axis=0)

    phi_xm[0, :, :] = _shift_z_nonperiodic_pm1(Phi[-1, :, :], -dz_shift, dz)
    phi_xp[-1, :, :] = _shift_z_nonperiodic_pm1(Phi[0, :, :], +dz_shift, dz)

    d2x = (phi_xp - 2.0 * Phi + phi_xm) * inv_dx2

    # Periodic neighbors in y
    phi_yp = np.roll(Phi, -1, axis=1)
    phi_ym = np.roll(Phi, 1, axis=1)
    d2y = (phi_yp - 2.0 * Phi + phi_ym) * inv_dy2

    # Nonperiodic second derivative in z using edge padding
    Pz = np.pad(Phi, ((0, 0), (0, 0), (1, 1)), mode="edge")
    phi_zp = Pz[:, :, 2:]
    phi_zm = Pz[:, :, :-2]
    d2z = (phi_zp - 2.0 * Phi + phi_zm) * inv_dz2

    # Laplacian and masked explicit diffusion update
    lap = d2x + d2y + d2z
    return Phi + dt * (nu * lap) * Ash_Shadow
