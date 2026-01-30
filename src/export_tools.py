import os
import struct
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import RegularGridInterpolator

from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import surface_tools as surf
from matplotlib.colors import LightSource
from typing import Tuple, Union, Iterable
from datetime import datetime, timedelta, timezone
from dateutil import parser
from matplotlib.colors import Normalize
from matplotlib import cm


def export_geometry_as_stl(
        folder_name: str,
        file_name: str,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        refinement_factor: int = 2,
        level: float = 0.0,
        surface_angle: float = 0.0,  # NEW: slope angle about +y (radians)
        verbose: bool = False
) -> None:
    """
    Extract Φ=level surface, add an x·tan(surface_angle) vertical bias (tilt),
    then rotate by −surface_angle about +y so the STL appears “flat”.
    Finally, write a binary STL with flipped face normals.

    Assumptions
    -----------
    - X, Y, Z are 3D meshgrids with indexing='ij':
        axis 0 -> x, axis 1 -> y, axis 2 -> z
    - Phi has shape (Nx, Ny, Nz) aligned with (X, Y, Z).
    """
    if refinement_factor < 1:
        raise ValueError("refinement_factor must be ≥ 1")

    try:
        from scipy.interpolate import RegularGridInterpolator
    except Exception as e:
        raise ImportError("SciPy is required: pip install scipy") from e

    try:
        from skimage.measure import marching_cubes
    except Exception as e:
        raise ImportError("scikit-image is required: pip install scikit-image") from e

    # ---- extract 1D axes from meshgrids (axis 0=x, 1=y, 2=z) ----
    X = np.asarray(X);
    Y = np.asarray(Y);
    Z = np.asarray(Z)
    phi = np.asarray(Phi)

    if not (X.ndim == Y.ndim == Z.ndim == 3):
        raise ValueError("X, Y, Z must be 3D meshgrids.")
    Nx, Ny, Nz = X.shape
    if phi.shape != (Nx, Ny, Nz):
        raise ValueError(f"Phi must have shape (Nx, Ny, Nz) = {X.shape}, got {phi.shape}.")

    x = X[:, 0, 0]
    y = Y[0, :, 0]
    z = Z[0, 0, :]

    # Ensure strict monotonicity; flip to ascending if needed (and flip phi consistently)
    def _is_strict_mono(a: np.ndarray) -> bool:
        d = np.diff(a)
        return np.all(d > 0) or np.all(d < 0)

    if not (_is_strict_mono(x) and _is_strict_mono(y) and _is_strict_mono(z)):
        raise ValueError("Axes must be strictly monotonic (ascending or descending).")

    if x.size > 1 and x[1] < x[0]:
        x = x[::-1];
        phi = phi[::-1, :, :]
    if y.size > 1 and y[1] < y[0]:
        y = y[::-1];
        phi = phi[:, ::-1, :]
    if z.size > 1 and z[1] < z[0]:
        z = z[::-1];
        phi = phi[:, :, ::-1]

    # ---- refine axes (uniform) & interpolate field if requested ----
    def _ref_axis(a: np.ndarray, r: int) -> np.ndarray:
        return np.linspace(a[0], a[-1], (a.size - 1) * r + 1)

    if refinement_factor == 1:
        xf, yf, zf = x, y, z
        phif = phi
    else:
        xf = _ref_axis(x, refinement_factor)
        yf = _ref_axis(y, refinement_factor)
        zf = _ref_axis(z, refinement_factor)
        interp = RegularGridInterpolator((x, y, z), phi, method="linear",
                                         bounds_error=False, fill_value=None)
        Xq, Yq, Zq = np.meshgrid(xf, yf, zf, indexing="ij")
        pts = np.column_stack([Xq.ravel(), Yq.ravel(), Zq.ravel()])
        phif = interp(pts).reshape(Xq.shape)

    # ---- Marching cubes (expects volume as (z, y, x)) ----
    dx = (xf[-1] - xf[0]) / (len(xf) - 1) if len(xf) > 1 else 1.0
    dy = (yf[-1] - yf[0]) / (len(yf) - 1) if len(yf) > 1 else 1.0
    dz = (zf[-1] - zf[0]) / (len(zf) - 1) if len(zf) > 1 else 1.0

    vol_zyx = np.transpose(phif, (2, 1, 0))  # (Nz, Ny, Nx)
    verts_zyx, faces, _, _ = marching_cubes(vol_zyx, level=level, spacing=(dz, dy, dx))

    # Convert verts from (z,y,x) to (x,y,z) and add origin offsets
    verts = np.empty_like(verts_zyx)
    verts[:, 0] = verts_zyx[:, 2] + xf[0]  # x
    verts[:, 1] = verts_zyx[:, 1] + yf[0]  # y
    verts[:, 2] = verts_zyx[:, 0] + zf[0]  # z

    # ---- Apply requested tilt & un-tilt rotation so STL appears flat ----
    # Bias: z ← z + x * tan(a)
    tan_a = np.tan(surface_angle)
    x0 = verts[:, 0]
    y0 = verts[:, 1]
    z0 = verts[:, 2] - x0 * tan_a

    # Rotate by −a about +y: R_y(-a) = [[cos a,0,-sin a],[0,1,0],[sin a,0,cos a]]
    ca = np.cos(-surface_angle)
    sa = np.sin(-surface_angle)
    x1 = ca * x0 - (-sa) * 0.0 - sa * z0  # x1 =  ca*x0 - sa*z0
    y1 = y0
    z1 = sa * x0 + ca * z0

    verts_flat = np.column_stack([x1, y1, z1]).astype(np.float32, copy=False)

    # ---- Write binary STL with flipped normals ----
    os.makedirs(folder_name, exist_ok=True)
    out_path = os.path.join(
        folder_name, file_name if file_name.lower().endswith(".stl") else f"{file_name}.stl"
    )

    with open(out_path, "wb") as f:
        header = b"Binary STL (tilted+rotated flat) by export_geometry_as_stl".ljust(80, b" ")
        f.write(header)
        f.write(struct.pack("<I", faces.shape[0]))

        for tri in faces:
            v0 = verts_flat[tri[0]].astype(np.float64)
            v1 = verts_flat[tri[1]].astype(np.float64)
            v2 = verts_flat[tri[2]].astype(np.float64)

            # Flip normals: swap the cross-product order to invert direction
            n = np.cross(v2 - v0, v1 - v0)
            nn = np.linalg.norm(n)
            if nn > 0.0:
                n = (n / nn).astype(np.float32)
            else:
                n = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # normal (nx,ny,nz) then vertices (x,y,z), attr byte count 0
            f.write(struct.pack("<3f", n[0], n[1], n[2]))
            f.write(struct.pack("<3f", float(v0[0]), float(v0[1]), float(v0[2])))
            f.write(struct.pack("<3f", float(v1[0]), float(v1[1]), float(v1[2])))
            f.write(struct.pack("<3f", float(v2[0]), float(v2[1]), float(v2[2])))
            f.write(struct.pack("<H", 0))

    if verbose:
        print(f"[export_geometry_as_stl] Saved {faces.shape[0]} triangles to {out_path} (binary STL).")


def export_ash_particles(
        folder_name: str,
        file_name: str,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        radius: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        refinement_factor: int = 2,
        delimiter: str = ";",
        float_fmt: str = "%.6e",
        level: float = 0.0,
        verbose: bool = False
) -> None:
    """
    Refine X,Y,Z,Φ on a rectilinear grid; compute z_surf(x,y) as the Φ=level crossing
    along +z (topmost surface); clamp particles above z_surf down to z_surf; save to txt.

    Parameters
    ----------
    pos_x, pos_y, pos_z : (N,) arrays
        Particle coordinates (same units as X,Y,Z).
    radius : (N,) array
        Particle radii.
    X,Y,Z : arrays
        3D meshgrids with indexing='ij' (axis 0=x, axis 1=y, axis 2=z).
    Phi : array
        Level-set field with shape (Nx, Ny, Nz).
    refinement_factor : int
        Per-axis upsampling factor ≥ 1 (1 = no refinement).
    delimiter, float_fmt : str
        Formatting options for the output text file.
    level : float
        Isosurface value (default 0.0).
    """
    if refinement_factor < 1:
        raise ValueError("refinement_factor must be ≥ 1")

    pos_x = np.asarray(pos_x, dtype=float).ravel()
    pos_y = np.asarray(pos_y, dtype=float).ravel()
    pos_z = np.asarray(pos_z, dtype=float).ravel()
    radius = np.asarray(radius, dtype=float).ravel()
    if not (pos_x.shape == pos_y.shape == pos_z.shape == radius.shape):
        raise ValueError("pos_x, pos_y, pos_z, and radius must all have the same shape")

    N = pos_x.size

    try:
        from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
    except Exception as e:
        raise ImportError("SciPy is required: pip install scipy") from e

    # ---- extract 1D axes from meshgrids (axis 0=x, 1=y, 2=z) ----
    X = np.asarray(X);
    Y = np.asarray(Y);
    Z = np.asarray(Z)
    phi = np.asarray(Phi)

    if not (X.ndim == Y.ndim == Z.ndim == 3):
        raise ValueError("X, Y, Z must be 3D meshgrids.")
    Nx, Ny, Nz = X.shape
    if phi.shape != (Nx, Ny, Nz):
        raise ValueError(f"Phi must have shape (Nx, Ny, Nz) = {X.shape}, got {phi.shape}.")

    x = X[:, 0, 0]
    y = Y[0, :, 0]
    z = Z[0, 0, :]

    # Ensure monotonic axes, flip if descending
    def _is_strict_mono(a):
        d = np.diff(a)
        return np.all(d > 0) or np.all(d < 0)

    if not (_is_strict_mono(x) and _is_strict_mono(y) and _is_strict_mono(z)):
        raise ValueError("Axes must be strictly monotonic (ascending or descending).")

    if x.size > 1 and x[1] < x[0]:
        x = x[::-1];
        phi = phi[::-1, :, :]
    if y.size > 1 and y[1] < y[0]:
        y = y[::-1];
        phi = phi[:, ::-1, :]
    if z.size > 1 and z[1] < z[0]:
        z = z[::-1];
        phi = phi[:, :, ::-1]

    # ---- refine axes & field ----
    def _ref_axis(a, r):
        return np.linspace(a[0], a[-1], (a.size - 1) * r + 1)

    if refinement_factor == 1:
        xf, yf, zf = x, y, z
        phif = phi
    else:
        xf = _ref_axis(x, refinement_factor)
        yf = _ref_axis(y, refinement_factor)
        zf = _ref_axis(z, refinement_factor)
        interp_phi = RegularGridInterpolator(
            (x, y, z), phi, method="linear", bounds_error=False, fill_value=None
        )
        Xq, Yq, Zq = np.meshgrid(xf, yf, zf, indexing="ij")
        pts = np.column_stack([Xq.ravel(), Yq.ravel(), Zq.ravel()])
        phif = interp_phi(pts).reshape(Xq.shape)

    Nxf, Nyf, Nzf = phif.shape  # (x,y,z)

    # ---- compute topmost z_surf(x,y) ----
    s0 = phif[:, :, :-1]
    s1 = phif[:, :, 1:]
    cross = (s0 - level) * (s1 - level) <= 0
    any_cross = cross.any(axis=2)

    rev = cross[:, :, ::-1]
    idx_last_rev = rev.argmax(axis=2)
    k_last = (Nzf - 2) - idx_last_rev
    k_last = np.where(any_cross, k_last, -1)

    z_surf = np.full((Nxf, Nyf), np.nan, dtype=float)
    if np.any(any_cross):
        take_idx = np.expand_dims(k_last.clip(0, Nzf - 2), axis=2)
        phi0 = np.take_along_axis(s0, take_idx, axis=2).squeeze(2)
        phi1 = np.take_along_axis(s1, take_idx, axis=2).squeeze(2)
        z0 = zf[k_last]
        z1 = zf[k_last + 1]
        denom = (phi1 - phi0)
        t = np.where(np.abs(denom) > 0, (level - phi0) / denom, 0.0)
        zc = z0 + t * (z1 - z0)
        z_surf[any_cross] = zc[any_cross]

    # ---- interpolate z_surf at particle (x,y) ----
    xy_interp = RegularGridInterpolator(
        (xf, yf), z_surf, method="linear", bounds_error=False, fill_value=np.nan
    )
    zq = xy_interp(np.column_stack([pos_x, pos_y]))

    nan_mask = ~np.isfinite(zq)
    if np.any(nan_mask):
        xx, yy = np.meshgrid(xf, yf, indexing="ij")
        good = np.isfinite(z_surf)
        if np.any(good):
            nn = NearestNDInterpolator(
                np.column_stack([xx[good], yy[good]]),
                z_surf[good]
            )
            zq[nan_mask] = nn(pos_x[nan_mask], pos_y[nan_mask])

    # ---- clamp ----
    updated_x = pos_x.copy()
    updated_y = pos_y.copy()
    updated_z = pos_z.copy()
    updated_r = radius.copy()

    above = np.isfinite(zq) & (updated_z > zq)
    updated_z[above] = zq[above]

    # ---- export ----
    os.makedirs(folder_name, exist_ok=True)
    out_path = os.path.join(
        folder_name, file_name if file_name.lower().endswith(".txt") else f"{file_name}.txt"
    )
    header = delimiter.join(["x", "y", "z", "radius"])
    data_out = np.column_stack([updated_x, updated_y, updated_z, updated_r])
    np.savetxt(out_path, data_out, delimiter=delimiter, fmt=float_fmt, header=header, comments="")

    if verbose:
        print(f"[export_ash_particles] Snapped {np.count_nonzero(above)} / {N} particles; wrote {out_path}")


def export_erosion_history(
        folder_name: str,
        file_name: str,
        start_date: Union[str, datetime],
        time_data: np.ndarray,
        erosion_yield_data: np.ndarray,
        AO_fluence_data: np.ndarray,
        density_data: np.ndarray,
        speed_ratio_data: np.ndarray,
        incident_angle_1_data: np.ndarray,
        incident_angle_2_data: np.ndarray,
        gamma_array_history: np.ndarray,
        mu_array_history: np.ndarray,
        sigma_array_history: np.ndarray,
        R_history: np.ndarray,
        verbose: bool = False
) -> None:
    """
    Export erosion simulation history as a semicolon-separated text file,
    with aligned columns and units in the header.
    """

    # ---- helpers ----
    def _parse_start(dt_like: Union[str, datetime]) -> datetime:
        if isinstance(dt_like, datetime):
            dt = dt_like
        else:
            s = str(dt_like).strip()
            fmts = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%d/%m/%Y:%H:%M:%S",
                "%m/%d/%Y:%H:%M:%S",
            ]
            dt = None
            for fmt in fmts:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                raise ValueError(f"Unrecognized start_date format: {dt_like!r}")
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    def _to_utc_strings(t0_utc: datetime, t_rel_s: np.ndarray) -> list[str]:
        return [(t0_utc + timedelta(seconds=float(ts))).strftime("%Y-%m-%d %H:%M:%S")
                for ts in t_rel_s]

    # ---- inputs & shape checks ----
    t = np.asarray(time_data, dtype=float).reshape(-1)
    Nt = t.size
    one_d_names = [
        ("erosion_yield_data", erosion_yield_data, ".6e"),
        ("AO_fluence_data", AO_fluence_data, ".6e"),
        ("density_data", density_data, ".6e"),
        ("speed_ratio_data", speed_ratio_data, ".6f"),
        ("incident_angle_1_data", incident_angle_1_data, ".6f"),
        ("incident_angle_2_data", incident_angle_2_data, ".6f"),
        ("R_history", R_history, ".6f"),
    ]
    one_d = {}
    for name, arr, _ in one_d_names:
        a = np.asarray(arr).reshape(-1)
        if a.size != Nt:
            raise ValueError(f"{name} length {a.size} != time_data length {Nt}")
        one_d[name] = a

    gamma = np.asarray(gamma_array_history, dtype=float)
    mu = np.asarray(mu_array_history, dtype=float)
    sigma = np.asarray(sigma_array_history, dtype=float)
    if gamma.ndim != 2 or mu.ndim != 2 or sigma.ndim != 2:
        raise ValueError("gamma/mu/sigma must be 2D arrays of shape (Nt, M).")
    if not (gamma.shape[0] == mu.shape[0] == sigma.shape[0] == Nt):
        raise ValueError("gamma/mu/sigma first dimension must equal Nt (len(time_data)).")
    if not (gamma.shape[1] == mu.shape[1] == sigma.shape[1]):
        raise ValueError("gamma/mu/sigma must share the same number of parameters (M).")
    M = gamma.shape[1]

    # ---- time stamps ----
    t0_utc = _parse_start(start_date)
    utc_strings = _to_utc_strings(t0_utc, t)

    # ---- header with units ----
    base_cols = [
        "time_utc",
        "t_sec [s]",
        "erosion_yield [cm^3/AO]",
        "AO_fluence [atoms/cm^2]",
        "density [kg/m^3]",
        "speed_ratio [-]",
        "incident_angle_1 [deg]",
        "incident_angle_2 [deg]",
        "R [µm]",
    ]
    gamma_cols = [f"gamma_{j + 1} [-]" for j in range(M)]
    mu_cols = [f"mu_{j + 1} [µm]" for j in range(M)]
    sigma_cols = [f"sigma_{j + 1} [µm]" for j in range(M)]
    headers = base_cols + gamma_cols + mu_cols + sigma_cols

    # ---- format sample values to determine widths ----
    col_samples = []
    col_samples.append(utc_strings[0])  # time_utc
    col_samples.append(f"{t[-1]:.6f}")
    for (name, _, fmt) in one_d_names:
        col_samples.append(f"{one_d[name][-1]:{fmt}}")
    col_samples.extend([f"{gamma[-1, j]:.6e}" for j in range(M)])
    col_samples.extend([f"{mu[-1, j]:.6e}" for j in range(M)])
    col_samples.extend([f"{sigma[-1, j]:.6e}" for j in range(M)])

    widths = [max(len(h), len(s)) + 2 for h, s in zip(headers, col_samples)]

    # ---- write file ----
    os.makedirs(folder_name, exist_ok=True)
    out_path = os.path.join(
        folder_name, file_name if file_name.lower().endswith(".txt") else f"{file_name}.txt"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        # header
        header_line = "".join(h.ljust(w) for h, w in zip(headers, widths))
        f.write(header_line + "\n")
        # rows
        for i in range(Nt):
            row = []
            row.append(utc_strings[i].ljust(widths[0]))
            row.append(f"{t[i]:.6f}".rjust(widths[1]))
            k = 2
            for (name, _, fmt) in one_d_names:
                row.append(f"{one_d[name][i]:{fmt}}".rjust(widths[k]))
                k += 1
            for j in range(M):
                row.append(f"{gamma[i, j]:.6e}".rjust(widths[k]));
                k += 1
            for j in range(M):
                row.append(f"{mu[i, j]:.6e}".rjust(widths[k]));
                k += 1
            for j in range(M):
                row.append(f"{sigma[i, j]:.6e}".rjust(widths[k]));
                k += 1
            f.write("".join(row) + "\n")

    if verbose:
        print(f"[export_erosion_history] Wrote Nt={Nt}, M={M} to {out_path}")


def plot_polygaussian_surface_data(
        folder_name: str,
        file_name: str,
        surface: surf.Surface,
        poly_surface: surf.PolyGaussian_Surface,
        surface_angle: float
) -> None:
    """
    Save a publication-quality PDF comparing the eroded surface vs. its fitted
    PolyGaussian model. Top row: 3D surfaces; bottom row: height & slope PDFs.
    The PolyGaussian sample is winsorized (percentile clip) to remove outliers.
    """

    # ----- IO -----
    os.makedirs(folder_name, exist_ok=True)
    if not file_name.lower().endswith(".jpg"):
        file_name += ".jpg"
    out_path = os.path.join(folder_name, file_name)

    # ----- Eroded (isotropic) geometry -----
    s_iso = surface.get_isotropic_surface(surface_angle)
    X = np.asarray(s_iso.X, float)
    Y = np.asarray(s_iso.Y, float)
    Z = np.asarray(s_iso.Z, float)

    # Heights & slopes (eroded)
    dx = float(np.nanmedian(np.diff(X[:, 0]))) if X.shape[0] > 1 else 1.0
    dZdx = np.gradient(Z, dx, axis=0)
    heights_e = Z.ravel()
    slopes_e = dZdx.ravel()
    heights_e = heights_e[np.isfinite(heights_e)]
    slopes_e = slopes_e[np.isfinite(slopes_e)]

    # ----- Fitted PolyGaussian sample (winsorize outliers) -----
    Ls = float(np.nanmax(Y) - np.nanmin(Y)) if np.isfinite(Y).all() else 1.0
    pg = poly_surface.generate(sample_length=Ls, samples=220, iterations=60, verbose=False).get_isotropic_surface(surface_angle)
    Xp = np.asarray(pg.X, float)
    Yp = np.asarray(pg.Y, float)
    Zp = np.asarray(pg.Z, float)

    # Winsorize Zp to remove extreme outliers
    p_lo, p_hi = np.nanpercentile(Zp, [1.0, 99.0])
    Zp = np.clip(Zp, p_lo, p_hi)

    # Heights & slopes (model)
    dxp = float(np.nanmedian(np.diff(Xp[:, 0]))) if Xp.shape[0] > 1 else 1.0
    dZdx_p = np.gradient(Zp, dxp, axis=0)
    heights_m = Zp.ravel()
    slopes_m = dZdx_p.ravel()
    heights_m = heights_m[np.isfinite(heights_m)]
    slopes_m = slopes_m[np.isfinite(slopes_m)]

    # ----- Shared bins for fair comparison -----
    def _bins(a, b, nbins=70):
        if a.size and b.size:
            lo = float(min(np.min(a), np.min(b)))
            hi = float(max(np.max(a), np.max(b)))
            if lo == hi: hi = lo + 1.0
        elif a.size:
            lo, hi = float(np.min(a)), float(np.max(a))
            if lo == hi: hi = lo + 1.0
        elif b.size:
            lo, hi = float(np.min(b)), float(np.max(b))
            if lo == hi: hi = lo + 1.0
        else:
            lo, hi = -1.0, 1.0
        bins = np.linspace(lo, hi, nbins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        return bins, centers

    h_bins, h_cent = _bins(heights_e, heights_m, nbins=70)
    s_bins, s_cent = _bins(slopes_e, slopes_m, nbins=70)

    # PDFs (histograms)
    h_emp, _ = np.histogram(heights_e, bins=h_bins, density=True)
    h_mod, _ = np.histogram(heights_m, bins=h_bins, density=True)
    s_emp, _ = np.histogram(slopes_e, s_bins, density=True)
    s_mod, _ = np.histogram(slopes_m, s_bins, density=True)

    # ----- Helpers -----
    def _set_equal_box(ax, X_, Y_, Z_):
        rx = float(np.nanmax(X_) - np.nanmin(X_))
        ry = float(np.nanmax(Y_) - np.nanmin(Y_))
        rz = float(np.nanmax(Z_) - np.nanmin(Z_))
        rx = rx if rx > 0 else 1.0
        ry = ry if ry > 0 else 1.0
        rz = rz if rz > 0 else 1.0
        ax.set_box_aspect((rx, ry, rz))

    # ----- Figure -----
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.direction": "in", "ytick.direction": "in",
    })
    fig = plt.figure(figsize=(10.5, 7.0))
    gs = gridspec.GridSpec(2, 2, wspace=0.28, hspace=0.34)

    # (a) Eroded surface (top-left)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.96, cmap="viridis")
    ax1.set_title("Eroded surface (isotropic)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    _set_equal_box(ax1, X, Y, Z)
    ax1.view_init(elev=45, azim=-60)

    # (b) PolyGaussian sample (top-right)
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    ax2.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.96, cmap="plasma")
    ax2.set_title("Fitted PolyGaussian (winsorized sample)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    _set_equal_box(ax2, Xp, Yp, Zp)
    ax2.view_init(elev=45, azim=-60)

    # (c) Height PDF (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.step(h_cent, h_emp, where="mid", lw=1.6, label="Eroded (empirical)")
    ax3.step(h_cent, h_mod, where="mid", lw=1.6, ls="--", label="PolyGaussian (sample)")
    ax3.set_title("Height PDF")
    ax3.set_xlabel("height")
    ax3.set_ylabel("density")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best")

    # (d) Slope PDF (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.step(s_cent, s_emp, where="mid", lw=1.6, label="Eroded (empirical)")
    ax4.step(s_cent, s_mod, where="mid", lw=1.6, ls="--", label="PolyGaussian (sample)")
    ax4.set_title("Slope (∂z/∂x) PDF")
    ax4.set_xlabel("slope")
    ax4.set_ylabel("density")
    ax4.grid(alpha=0.25)
    ax4.legend(loc="best")

    fig.suptitle(f"Eroded vs PolyGaussian (angle={np.degrees(surface_angle):.1f}°)", y=0.99)
    fig.savefig(out_path, format="jpg", bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_erosion_history(
        folder_name: str,
        file_name: str,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        Phi: np.ndarray,
        sample_size: float,
        time_data: np.ndarray,
        erosion_yield_data: np.ndarray,  # cm^3 / AO atom
        AO_fluence_data: np.ndarray,  # SI (atoms/cm^2)
        density_data: np.ndarray,  # SI (kg/m^3)
        speed_ratio_data: np.ndarray,  # dimensionless
        incident_angle_1_data: np.ndarray,  # radians
        incident_angle_2_data: np.ndarray,  # radians
        surface_angle: float,  # radians; rotate about +y
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        pos_z: np.ndarray,
        radius: np.ndarray,
        cmap_function: sp.interpolate.RegularGridInterpolator | None
) -> None:
    # ---------- IO ----------
    os.makedirs(folder_name, exist_ok=True)
    if not file_name.lower().endswith(".pdf"):  # keep your current behavior (save JPG)
        file_name += ".jpg"
    out_path = os.path.join(folder_name, file_name)

    # ---------- time & series (unchanged) ----------
    def _months(t):
        t = np.asarray(t, float).ravel()
        if t.size == 0 or not np.isfinite(t).all():
            raise ValueError("time_data must be finite and non-empty.")
        return (t - t[0]) / (30.0 * 24.0 * 3600.0)

    def _prep(ref, arr):
        a = np.asarray(arr, float).ravel()
        if a.size != ref.size:
            raise ValueError("All time-series must match time_data length.")
        return np.nan_to_num(a, nan=np.nan, posinf=np.nan, neginf=np.nan)

    tm = _months(time_data)
    EY = _prep(tm, erosion_yield_data)
    AO = _prep(tm, AO_fluence_data)
    rho = _prep(tm, density_data)
    Sr = _prep(tm, speed_ratio_data)
    th1d = _prep(tm, incident_angle_1_data)
    th2d = _prep(tm, incident_angle_2_data)
    th1_deg = np.degrees(np.clip(th1d, 0.0, np.pi))
    th2_deg = np.mod(np.degrees(th2d), 360.0)

    # ---------- extract surface from Φ on a refined grid (unchanged) ----------
    Nx, Ny, Nz = Phi.shape
    x = np.asarray(X[:, 0, 0], float) if X.ndim == 3 else np.linspace(np.nanmin(X), np.nanmax(X), Nx)
    y = np.asarray(Y[0, :, 0], float) if Y.ndim == 3 else np.linspace(np.nanmin(Y), np.nanmax(Y), Ny)
    z = np.asarray(Z[0, 0, :], float) if Z.ndim == 3 else np.linspace(np.nanmin(Z), np.nanmax(Z), Nz)

    refine = 3
    xr = np.linspace(x[0], x[-1], refine * Nx)
    yr = np.linspace(y[0], y[-1], refine * Ny)
    zr = np.linspace(z[0], z[-1], refine * Nz)

    rgi = sp.interpolate.RegularGridInterpolator(
        (x, y, z), np.asarray(Phi, float), bounds_error=False, fill_value=None
    )
    Xg, Yg, Zg = np.meshgrid(xr, yr, zr, indexing="ij")
    Phi_hr = rgi(np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])).reshape(Xg.shape)

    pos = Phi_hr >= 0.0
    any_pos = np.any(pos, axis=2)
    k2 = np.argmax(pos, axis=2)
    k2 = np.where(any_pos, k2, Phi_hr.shape[2] - 1)
    k1 = np.clip(k2 - 1, 0, Phi_hr.shape[2] - 1)

    ii, jj = np.indices(k2.shape)
    phi1 = Phi_hr[ii, jj, k1]
    phi2 = Phi_hr[ii, jj, k2]
    z1 = zr[k1]
    z2 = zr[k2]
    denom = np.where(np.abs(phi2 - phi1) > 1e-14, phi2 - phi1, 1e-14)
    Zs = z1 - phi1 * (z2 - z1) / denom  # surface height from PolyGaussian iso-surface
    Xs, Ys = np.meshgrid(xr, yr, indexing="ij")

    # ---------- rotate surface about +y (unchanged) ----------
    ca, sa = np.cos(-surface_angle), np.sin(-surface_angle)
    Xr = Xs * ca + Zs * sa
    Yr = Ys
    Zr = -Xs * sa + Zs * ca

    # ---------- incident vector (unchanged) ----------
    th1, th2 = float(th1d[-1]), float(th2d[-1])
    v_lab = np.array([np.sin(th1) * np.cos(th2), np.sin(th1) * np.sin(th2), -np.cos(th1)], float)
    v_rot = np.array([v_lab[0] * ca + v_lab[2] * sa, v_lab[1], -v_lab[0] * sa + v_lab[2] * ca], float)
    nrm = np.linalg.norm(v_rot)
    v_rot = v_rot / (nrm if nrm > 0 else 1.0)

    xmid, ymid = 0.5 * (Xr.min() + Xr.max()), 0.5 * (Yr.min() + Yr.max())
    ztop = Zr.max()
    llen = 0.35 * max(1e-9, min(np.ptp(Xr), np.ptp(Yr), np.ptp(Zr)))

    # ---------- figure & axes (unchanged) ----------

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.0, "legend.fontsize": 8.0,
        "xtick.direction": "in", "ytick.direction": "in"
    })
    fig = plt.figure(figsize=(10.6, 5.8), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.35, 1.65], height_ratios=[1, 1, 1])

    axS = fig.add_subplot(gs[:, 0], projection="3d")

    # ---------- NEW: color the surface from cmap_function(lat, lon, Zs) ----------
    if cmap_function is not None:
        # normals from PolyGaussian surface Zs(Xs, Ys) — BEFORE rotation
        # dz/dx, dz/dy with proper spacing (xr along axis=0, yr along axis=1)
        dzdx, dzdy = np.gradient(Zs, xr, yr, edge_order=2)
        nx = -dzdx
        ny = -dzdy
        nz = np.ones_like(Zs)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        # unit normal
        nx /= np.where(norm > 0, norm, 1.0)
        ny /= np.where(norm > 0, norm, 1.0)
        nz /= np.where(norm > 0, norm, 1.0)

        # normal angles (latitude & longitude)
        lat = np.arccos(np.clip(nz, -1.0, 1.0))  # [-pi/2, pi/2]
        lon = np.mod(np.arctan2(ny, nx), 2.0 * np.pi) * 0.0  # [0, 2π)

        # evaluate user-provided color mapping on (lat, lon, height=Zs)
        Zs -= np.mean(Zs)
        pts = np.column_stack([lat.ravel(), lon.ravel(), Zs.ravel()])
        C = cmap_function(pts)
        C = np.asarray(C)

        # Accept scalar or RGB(A). Map scalar → Viridis.
        if C.ndim == 1 or (C.ndim == 2 and C.shape[1] == 1):
            Cs = C.ravel().reshape(Zs.shape)
            finite = np.isfinite(Cs)
            if np.any(finite):
                vmin, vmax = np.nanpercentile(Cs[finite], [2, 98])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = float(np.nanmin(Cs)), float(np.nanmax(Cs))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = 0.0, 1.0
            normc = Normalize(vmin=vmin, vmax=vmax)
            facecolors = cm.get_cmap("viridis")(normc(np.nan_to_num(Cs, nan=vmin)))
        elif C.ndim == 2 and C.shape[1] in (3, 4):
            # Already RGB(A) in [0,1]; reshape to (M,N,3/4). Add alpha if missing.
            k = C.shape[1]
            facecolors = C.reshape(Zs.shape + (k,))
            if k == 3:
                alpha = np.ones(Zs.shape + (1,), dtype=facecolors.dtype)
                facecolors = np.concatenate([facecolors, alpha], axis=-1)
        else:
            raise ValueError("cmap_function must return scalar or RGB(A) per grid point.")

        # Ensure finite & clipped
        facecolors = np.clip(np.nan_to_num(facecolors, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        axS.plot_surface(Xr, Yr, Zr,
                         facecolors=facecolors,
                         linewidth=0, antialiased=False,
                         rstride=1, cstride=1, shade=False)
    else:
        # fallback: your original LightSource grayscale
        ls = LightSource(azdeg=0, altdeg=25)
        rgb = ls.shade(Zr, cmap=plt.cm.gray, vert_exag=4.0, blend_mode='soft')
        axS.plot_surface(Xr, Yr, Zr, facecolors=rgb, linewidth=0, antialiased=True,
                         rstride=1, cstride=1, shade=True)

    # incident arrow (unchanged)
    axS.quiver(xmid, ymid, ztop + 0.12 * llen, v_rot[0], v_rot[1], v_rot[2],
               length=llen, color="crimson", linewidth=1.6, arrow_length_ratio=0.12)
    axS.set_title("Eroded surface (rotated) + incident vector")
    axS.set_xlabel("x")
    axS.set_ylabel("y")
    axS.set_zlabel("z")
    axS.set_xlim(- sample_size / 2.0, sample_size / 2.0)
    axS.set_ylim(- sample_size / 2.0, sample_size / 2.0)
    axS.set_zlim(- sample_size / 2.0, sample_size / 2.0)

    def _set_equal_box(ax, X_, Y_, Z_):
        xm, ym, zm = 0.5 * (X_.min() + X_.max()), 0.5 * (Y_.min() + Y_.max()), 0.5 * (Z_.min() + Z_.max())
        r = 0.5 * max(np.ptp(X_), np.ptp(Y_), np.ptp(Z_), 1e-9)
        ax.set_xlim(xm - r, xm + r)
        ax.set_ylim(ym - r, ym + r)
        ax.set_zlim(zm - r, zm + r)
        ax.set_aspect('equal')

    _set_equal_box(axS, Xr, Yr, Zr)
    axS.view_init(elev=45, azim=45)

    # -------------------- ash spheres overlay (unchanged) --------------------

    zr_interp = RegularGridInterpolator((xr, yr), Zs, bounds_error=False, fill_value=np.nan)
    px = np.asarray(pos_x, float).ravel()
    py = np.asarray(pos_y, float).ravel()
    pz = np.asarray(pos_z, float).ravel()
    pr = np.asarray(radius, float).ravel()
    if not (px.shape == py.shape == pz.shape == pr.shape):
        raise ValueError("pos_x, pos_y, pos_z, radius must have the same shape")
    zsurf = zr_interp(np.column_stack([px, py]))
    mask = np.isfinite(zsurf) & (pz > zsurf)

    eps = 1e-6 * max(np.ptp(Zr), 1.0)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    def _icosphere(center, r, subdiv=1):
        # ... (unchanged from your version) ...
        import numpy as np
        t = (1.0 + np.sqrt(5.0)) / 2.0
        V = np.array([
            (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
        ], dtype=float)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        F = np.array([
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
        ], dtype=int)

        def subdivide(V, F):
            cache = {}
            verts = V.tolist()
            faces = []

            def midpoint(i, j):
                key = (i, j) if i < j else (j, i)
                if key in cache: return cache[key]
                m = (V[i] + V[j]) / 2.0
                m /= np.linalg.norm(m)
                idx = len(verts)
                verts.append(m.tolist())
                cache[key] = idx
                return idx

            for a, b, c in F:
                ab = midpoint(a, b)
                bc = midpoint(b, c)
                ca = midpoint(c, a)
                faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
            return np.asarray(verts, float), np.asarray(faces, int)

        for _ in range(max(0, int(subdiv))):
            V, F = subdivide(V, F)
        P = V * float(r) + np.asarray(center, float)
        return [P[idx] for idx in F]

    if np.any(mask):
        sort_zpos = float(np.nanmax(Zr)) + 10.0 * max(np.ptp(Zr), 1.0)
        for x0, y0, z0, r0, zs in zip(px[mask], py[mask], pz[mask], pr[mask], zsurf[mask]):
            cx, cy, cz = float(x0), float(y0), float(zs) + eps
            cx_r = cx * ca + cz * sa
            cy_r = cy
            cz_r = -cx * sa + cz * ca
            tris = _icosphere((cx_r, cy_r, cz_r), float(r0), subdiv=1)
            coll = Poly3DCollection(tris, facecolor="#A9A9A9", edgecolor="w", linewidth=0.001, alpha=0.2)
            coll.set_zorder(1e6)
            try:
                coll.set_sort_zpos(sort_zpos)
            except Exception:
                try:
                    coll.set_zsort('max')
                except Exception:
                    pass
            axS.add_collection3d(coll)
    Zs -= np.mean(Zs)
    # ---------- right-side time plots (unchanged) ----------
    ax1 = fig.add_subplot(gs[0, 1])
    ax1b = ax1.twinx()
    ax1.grid(alpha=0.25)
    l1, = ax1.plot(tm[1:], EY[1:], lw=1.8, color="#1f77b4", label="Erosion yield (cm³/AO)")
    l2, = ax1b.plot(tm, AO, lw=1.5, ls="--", color="#7f7f7f", label="AO fluence")
    ax1.set_ylabel("Erosion yield (cm³/AO)")
    ax1b.set_ylabel("AO fluence (atoms/cm²)")
    ax1.set_yscale('log')
    ax1.set_title("Erosion yield & AO fluence")
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", frameon=False)

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2b = ax2.twinx()
    ax2.grid(alpha=0.25)
    l3, = ax2.plot(tm[1:], rho[1:], lw=1.8, color="#2ca02c", label=r"Density (kg/m$^3$)")
    l4, = ax2b.plot(tm[1:], Sr[1:], lw=0.9, ls="--", color="#d62728", label="Speed ratio (–)")
    ax2.set_ylabel(r"Density (kg/m$^3$)");
    ax2b.set_ylabel("Speed ratio (–)")
    ax2.set_title("Ambient density & speed ratio")
    ax2.legend([l3, l4], [l3.get_label(), l4.get_label()], loc="upper left", frameon=False)

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax3.grid(alpha=0.25)
    ax3.plot(tm[1:], th1_deg[1:], lw=1.6, color="#9467bd", label="θ (deg)")
    ax3.plot(tm[1:], th2_deg[1:], lw=1.6, ls="--", color="#8c564b", label="φ (deg)")
    ax3.set_xlabel("Time (months)");
    ax3.set_ylabel("Angle (deg)")
    ax3.set_title("Incident angles")
    ax3.legend(loc="upper left", frameon=False)

    fig.suptitle("Erosion history", y=0.995)
    fig.savefig(out_path, format="jpg", dpi=800)
    plt.close(fig)
