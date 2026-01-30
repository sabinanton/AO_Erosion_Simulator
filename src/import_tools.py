import os
import numpy as np
import scipy as sp
from typing import Dict, Any
from typing import Tuple, Union, Iterable
from datetime import datetime, timezone
import surface_tools as surf  # must provide PolyGaussian_Surface


def import_initial_polygaussian_surface(
        project_folder: str,
        surface_file: str,
        n_int: int = 40,
) -> surf.PolyGaussian_Surface:
    """
    Import a PolyGaussian surface from a strictly formatted text file and
    return a `PolyGaussian_Surface` with **angle forced to 0.0**.

    Expected file format (numbers separated by spaces/newlines):
        1) num_param
        2) mu_coeff            # num_param values
        3) sigma_coeff         # num_param values
        4) R                   # scalar > 0
        5) num_local_param
        6) local_params        # num_local_param values (ignored by this loader)
        7) surface_angle       # read but IGNORED — returned surface uses angle=0.0

    Parameters
    ----------
    project_folder : str
        Directory containing the surface file.
    surface_file : str
        Filename of the surface description (e.g., 'surface.srf').
    n_int : int, optional
        Integration resolution to pass to `PolyGaussian_Surface` (default 40).

    Returns
    -------
    surf.PolyGaussian_Surface
        Constructed surface with angle=0.0.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    ValueError
        If the file content does not match the required format.
    """
    path = os.path.join(os.path.abspath(project_folder), surface_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Surface file not found: {path}")

    # Read all numeric tokens (whitespace-separated)
    with open(path, "r", encoding="utf-8") as f:
        tokens = []
        for line in f:
            # no comments specified; treat everything as data
            parts = line.strip().split()
            for p in parts:
                try:
                    tokens.append(float(p))
                except ValueError:
                    raise ValueError(f"Non-numeric token '{p}' encountered in {path}")

    it = iter(tokens)
    try:
        n = int(round(next(it)))
        if n < 1:
            raise ValueError("num_param must be >= 1.")

        mu = np.array([next(it) for _ in range(n)], dtype=float)
        sigma = np.array([next(it) for _ in range(n)], dtype=float)

        R = float(next(it))
        if not np.isfinite(R) or R <= 0.0:
            raise ValueError("R must be a positive finite scalar.")

        nloc = int(round(next(it)))
        if nloc < 0:
            raise ValueError("num_local_param must be >= 0.")
        # consume (but ignore) local parameters
        _local = [next(it) for _ in range(nloc)]

        # read (and ignore) surface angle from file; we force 0.0
        _angle_from_file = float(next(it))

    except StopIteration:
        raise ValueError("File ended unexpectedly; check the required 7-part format.")
    except Exception as exc:
        # re-raise as ValueError for a clean API
        raise ValueError(f"Failed to parse '{path}': {exc}") from exc

    # Build surface with angle forced to 0.0
    return surf.PolyGaussian_Surface(
        sigma_coeff=sigma,
        mu_coeff=mu,
        R=R,
        angle=0.0,  # force horizontal
        N_INT=int(max(2, n_int)),
    )


def import_atmospheric_properties(
        project_folder: str,
        atmosphere_file: str,
        sim_start_iso: Union[str, datetime],
        sim_end_iso: Union[str, datetime],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load atmospheric time-series with UTC timestamps and clip to a simulation window.

    Expected file format (semicolon-separated; one header line):
        date/time [UTC] ; AO density [kg/m^3] ; speed ratio ; velocity [m/s]

    Notes
    ; Lines starting with '#' are ignored
    ; Rows with any non-finite value are discarded
    ; Result is sorted by time ascending
    ; Returns 1D NumPy arrays

    Parameters
    ----------
    project_folder : str
        Directory containing the atmosphere file.
    atmosphere_file : str
        Filename, for example "iss_atmosphere.txt".
    sim_start_iso : str or datetime
        Simulation start time in UTC, for example "2025-09-05 12:00:00".
    sim_end_iso : str or datetime
        Simulation end time in UTC.

    Returns
    -------
    t_sec : np.ndarray
        Time array in seconds; starts at 0; ends at sim_end − sim_start.
    ao_density : np.ndarray
        NRLMSISE total mass density [kg/m^3] over the requested window.
    speed_ratio : np.ndarray
        Speed ratio over the requested window.
    velocity : np.ndarray
        Velocity magnitude [m/s] over the requested window.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If parsing fails, time span is invalid, or fewer than two data points remain.
    """

    def _parse_utc(s: Union[str, datetime]) -> datetime:
        if isinstance(s, datetime):
            # Assume naive datetimes are UTC
            return s if s.tzinfo is not None else s.replace(tzinfo=timezone.utc)
        s = s.strip()
        # Accept common timestamp styles
        fmts: Iterable[str] = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Unrecognized UTC time format: {s!r}")

    def _to_epoch_s(dt: datetime) -> float:
        dt = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    path = os.path.join(os.path.abspath(project_folder), atmosphere_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Atmosphere file not found: {path}")

    # Read file; skip comments and blank lines; ignore the header line
    times_utc: list[datetime] = []
    ao_list: list[float] = []
    sr_list: list[float] = []
    vel_list: list[float] = []

    with open(path, "r", encoding="utf-8") as f:
        first_data_seen = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 4:
                # Likely a header or malformed row
                continue
            # Detect header once by testing float conversion on numeric fields
            try:
                ao = float(parts[1])
                sr = float(parts[2])
                vel = float(parts[3])
            except ValueError:
                # Header row; skip
                continue

            try:
                t_dt = _parse_utc(parts[0])
            except ValueError:
                # If first field is not a recognizable datetime; skip row
                continue

            # Filter non-finite rows
            if not (np.isfinite(ao) and np.isfinite(sr) and np.isfinite(vel)):
                continue

            times_utc.append(t_dt)
            ao_list.append(ao)
            sr_list.append(sr)
            vel_list.append(vel)
            first_data_seen = True

    if not times_utc:
        raise ValueError("No valid data rows found in atmosphere file.")

    # Convert to arrays and sort by time
    t_abs = np.array([_to_epoch_s(dt) for dt in times_utc], dtype=float)
    ao = np.array(ao_list, dtype=float)
    sr = np.array(sr_list, dtype=float)
    vel = np.array(vel_list, dtype=float)

    order = np.argsort(t_abs, kind="mergesort")
    t_abs = t_abs[order]
    ao = ao[order]
    sr = sr[order]
    vel = vel[order]

    # Remove duplicate timestamps by keeping the first occurrence
    uniq_mask = np.r_[True, np.diff(t_abs) > 0.0]
    t_abs = t_abs[uniq_mask]
    ao = ao[uniq_mask]
    sr = sr[uniq_mask]
    vel = vel[uniq_mask]

    if t_abs.size < 2:
        raise ValueError("Need at least two time samples for interpolation.")

    # Parse simulation window
    t0_dt = _parse_utc(sim_start_iso)
    t1_dt = _parse_utc(sim_end_iso)
    t0 = _to_epoch_s(t0_dt)
    t1 = _to_epoch_s(t1_dt)
    if not (t1 > t0):
        raise ValueError("sim_end_iso must be later than sim_start_iso.")

    # Ensure the requested window lies within the data coverage
    if t0 < t_abs[0] or t1 > t_abs[-1]:
        raise ValueError(
            f"Requested simulation window [{t0_dt.isoformat()} ; {t1_dt.isoformat()}] "
            f"is outside data coverage [{datetime.fromtimestamp(t_abs[0]).isoformat()}Z ; "
            f"{datetime.fromtimestamp(t_abs[-1]).isoformat()}Z]."
        )

    # Select all interior samples within (t0, t1) inclusive
    inside = (t_abs >= t0) & (t_abs <= t1)
    t_mid = t_abs[inside]
    ao_mid = ao[inside]
    sr_mid = sr[inside]
    vel_mid = vel[inside]

    # Guarantee exact endpoints via linear interpolation
    # Build the new time grid: [t0] + interior unique times (excluding endpoints) + [t1]
    # Exclude exact duplicates of endpoints to avoid repeating
    eps = 0.0
    mid_mask = (t_mid > t0 + eps) & (t_mid < t1 - eps)
    t_grid = np.concatenate(([t0], t_mid[mid_mask], [t1]))

    # Interpolate each series at t0 and t1; numpy.interp requires ascending x
    def _interp_at(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        return np.interp(x_new, x, y)

    ao_edge = _interp_at(t_abs, ao, np.array([t0, t1]))
    sr_edge = _interp_at(t_abs, sr, np.array([t0, t1]))
    vel_edge = _interp_at(t_abs, vel, np.array([t0, t1]))

    # Assemble y values aligned with t_grid
    ao_new = np.concatenate(([ao_edge[0]], ao_mid[mid_mask], [ao_edge[1]]))
    sr_new = np.concatenate(([sr_edge[0]], sr_mid[mid_mask], [sr_edge[1]]))
    vel_new = np.concatenate(([vel_edge[0]], vel_mid[mid_mask], [vel_edge[1]]))

    # Relative time in seconds
    t_rel = t_grid - t0
    # Numerical hygiene; snap first and last
    t_rel[0] = 0.0
    t_rel[-1] = t1 - t0

    return t_rel, ao_new, sr_new, vel_new


def import_spacecraft_properties(
        project_folder: str,
        spacecraft_file: str,
        sim_start_iso: Union[str, datetime],
        sim_end_iso: Union[str, datetime],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load spacecraft time-series with UTC timestamps; clip to [sim_start, sim_end]; interpolate endpoints.

    Expected file format (semicolon-separated; one header line allowed):
        date/time [UTC] ; incident_angle_1 [deg] ; incident_angle_2 [deg] ; surface_temperature [K]

    Returns
    -------
    t_sec : np.ndarray
        Seconds since sim_start; first element 0; last element sim_end − sim_start.
    incident_angle_1 : np.ndarray
        Polar incidence θ_i [deg]; aligned with t_sec.
    incident_angle_2 : np.ndarray
        Azimuth incidence φ_i [deg]; aligned with t_sec.
    surface_temperature : np.ndarray
        Surface temperature [K]; aligned with t_sec.
    """

    def _parse_utc(s: Union[str, datetime]) -> datetime:
        if isinstance(s, datetime):
            return s if s.tzinfo is not None else s.replace(tzinfo=timezone.utc)
        s = s.strip()
        fmts: Iterable[str] = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Unrecognized UTC time format: {s!r}")

    def _to_epoch_s(dt: datetime) -> float:
        dt = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    path = os.path.join(os.path.abspath(project_folder), spacecraft_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Spacecraft file not found: {path}")

    # Parse file manually to handle datetime in col 0 and skip header/comments
    t_list, a1_list, a2_list, T_list = [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 4:
                continue
            # Try numeric fields first to filter out headers
            try:
                a1 = float(parts[1])
                a2 = float(parts[2])
                T = float(parts[3])
            except ValueError:
                continue
            try:
                t_dt = _parse_utc(parts[0])
            except ValueError:
                continue
            if not (np.isfinite(a1) and np.isfinite(a2) and np.isfinite(T)):
                continue
            t_list.append(t_dt)
            a1_list.append(a1)
            a2_list.append(a2)
            T_list.append(T)

    if not t_list:
        raise ValueError("No valid data rows found in spacecraft file.")

    t_abs = np.array([_to_epoch_s(dt) for dt in t_list], dtype=float)
    a1 = np.array(a1_list, dtype=float)
    a2 = np.array(a2_list, dtype=float)
    Temp = np.array(T_list, dtype=float)

    # Sort; drop duplicate timestamps
    order = np.argsort(t_abs, kind="mergesort")
    t_abs = t_abs[order]
    a1 = a1[order]
    a2 = a2[order]
    Temp = Temp[order]
    uniq = np.r_[True, np.diff(t_abs) > 0.0]
    t_abs = t_abs[uniq]
    a1 = a1[uniq]
    a2 = a2[uniq]
    Temp = Temp[uniq]

    if t_abs.size < 2:
        raise ValueError("Need at least two time samples for interpolation.")

    # Simulation window
    t0_dt = _parse_utc(sim_start_iso)
    t1_dt = _parse_utc(sim_end_iso)
    t0 = _to_epoch_s(t0_dt)
    t1 = _to_epoch_s(t1_dt)
    if not (t1 > t0):
        raise ValueError("sim_end_iso must be later than sim_start_iso.")

    # Coverage check
    if t0 < t_abs[0] or t1 > t_abs[-1]:
        from datetime import datetime as _dt  # local alias
        raise ValueError(
            f"Requested window [{t0_dt.isoformat()} ; {t1_dt.isoformat()}] is outside data coverage "
            f"[{_dt.fromtimestamp(t_abs[0]).isoformat()}Z ; {_dt.fromtimestamp(t_abs[-1]).isoformat()}Z]."
        )

    # Interior selection
    inside = (t_abs >= t0) & (t_abs <= t1)
    t_mid = t_abs[inside]
    a1_mid = a1[inside]
    a2_mid = a2[inside]
    T_mid = Temp[inside]

    # Build exact endpoints; interpolate values at t0 and t1
    mid_mask = (t_mid > t0) & (t_mid < t1)
    t_grid = np.concatenate(([t0], t_mid[mid_mask], [t1]))

    def _interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
        return np.interp(xq, x, y)

    a1_edge = _interp(t_abs, a1, np.array([t0, t1]))
    a2_edge = _interp(t_abs, a2, np.array([t0, t1]))
    T_edge = _interp(t_abs, Temp, np.array([t0, t1]))

    a1_new = np.concatenate(([a1_edge[0]], a1_mid[mid_mask], [a1_edge[1]]))
    a2_new = np.concatenate(([a2_edge[0]], a2_mid[mid_mask], [a2_edge[1]]))
    T_new = np.concatenate(([T_edge[0]], T_mid[mid_mask], [T_edge[1]]))

    # Relative seconds
    t_rel = t_grid - t0
    t_rel[0] = 0.0
    t_rel[-1] = t1 - t0

    return t_rel, a1_new, a2_new, T_new


def import_material_data(material_name: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load tabulated material data and reshape it onto a regular (T, V, angle) mesh.

    File location
    -------------
    material_data/{material_name}.txt

    Expected whitespace-delimited columns (no header; lines starting with '#' are ignored):
        1) temperature [K]
        2) incident_velocity [Å/fs]   <-- converted to m/s (1 Å/fs = 1e5 m/s)
        3) incidence_angle [deg]
        4) erosion_yield [-]
        5) reacted_fraction [-]
        6) energy_accommodation [-]   <-- OPTIONAL; if missing, filled with 1.0

    Behavior
    --------
    - Validates that data can form a complete Cartesian grid over unique temperatures,
      velocities (after conversion), and angles.
    - Duplicate (T, V, angle) entries are averaged.
    - Raises if any grid cell is missing.

    Returns
    -------
    T_grid : (nT, nV, nA)
    V_grid : (nT, nV, nA) in m/s
    A_grid : (nT, nV, nA) in degrees
    erosion_yield : (nT, nV, nA)
    reacted_fraction : (nT, nV, nA)
    energy_accommodation : (nT, nV, nA)
    """
    path = os.path.join("material_data", f"{material_name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Material data file not found: {path}")

    data = np.genfromtxt(path, comments="#", dtype=float)
    if data.ndim == 1:
        if data.size < 5:
            raise ValueError(f"Expected ≥5 columns, found {data.size}.")
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"Expected ≥5 columns, found {data.shape[1]}.")

    # Keep first 6 columns if present; otherwise first 5
    max_cols = 6 if data.shape[1] >= 6 else 5
    data = data[:, :max_cols]

    # Drop rows with any non-finite entries
    finite = np.all(np.isfinite(data), axis=1)
    data = data[finite]
    if data.size == 0:
        raise ValueError("No valid (finite) rows in material data file.")

    if max_cols == 6:
        T_raw, V_raw_A_per_fs, A_raw, Y_raw, R_raw, Ea_raw = data.T
        has_ea = True
    else:
        T_raw, V_raw_A_per_fs, A_raw, Y_raw, R_raw = data.T
        Ea_raw = None
        has_ea = False

    # Convert velocity Å/fs -> m/s  (1 Å = 1e-10 m, 1 fs = 1e-15 s => 1e5 m/s)
    V_raw = V_raw_A_per_fs * 1e5

    # Unique sorted axes
    T_vals = np.unique(T_raw)
    V_vals = np.unique(V_raw)
    A_vals = np.unique(A_raw)
    nT, nV, nA = T_vals.size, V_vals.size, A_vals.size

    # Map rows to grid indices
    Ti = np.searchsorted(T_vals, T_raw)
    Vi = np.searchsorted(V_vals, V_raw)
    Ai = np.searchsorted(A_vals, A_raw)

    # Accumulate sums and counts on the 3D grid (vectorized via flat indices)
    flat = (Ti * nV + Vi) * nA + Ai
    size = nT * nV * nA
    sum_Y  = np.bincount(flat, weights=Y_raw, minlength=size)
    sum_R  = np.bincount(flat, weights=R_raw, minlength=size)
    sum_Ea = np.bincount(flat, weights=Ea_raw if has_ea else np.ones_like(Y_raw), minlength=size)
    cnt    = np.bincount(flat, minlength=size)

    Y_grid  = np.full(size, np.nan, dtype=float)
    R_grid  = np.full(size, np.nan, dtype=float)
    Ea_grid = np.full(size, np.nan, dtype=float)

    nz = cnt > 0
    Y_grid[nz]  = sum_Y[nz] / cnt[nz]
    R_grid[nz]  = sum_R[nz] / cnt[nz]
    if has_ea:
        Ea_grid[nz] = sum_Ea[nz] / cnt[nz]
    else:
        Ea_grid[nz] = 1.0  # default if column absent

    Y_grid  = Y_grid.reshape(nT, nV, nA)
    R_grid  = R_grid.reshape(nT, nV, nA)
    Ea_grid = Ea_grid.reshape(nT, nV, nA)

    # Verify a complete rectangular grid
    if np.any(cnt.reshape(nT, nV, nA) == 0):
        missing = np.argwhere(cnt.reshape(nT, nV, nA) == 0)
        i0, j0, k0 = missing[0]
        raise ValueError(
            "Material data do not form a complete (T, V, angle) grid: "
            f"missing {missing.shape[0]} of {size} cells. "
            f"Example missing at T={T_vals[i0]!r}, V={V_vals[j0]!r} m/s, angle={A_vals[k0]!r} deg."
        )

    # Build meshgrid (indexing='ij') with the same shape
    T_grid, V_grid, A_grid = np.meshgrid(T_vals, V_vals, A_vals, indexing="ij")

    return T_grid, V_grid, A_grid, Y_grid, R_grid, Ea_grid


def import_inputs(eroder: "Eroder", project_folder: str, inputs_file: str) -> None:
    """Parse inputs; load referenced files; populate `eroder` in place."""

    # ---------- helpers ----------
    def _clean(line: str) -> str | None:
        s = line.split("#", 1)[0].strip()
        return s or None

    def _as_bool(v: str | None, default=False) -> bool:
        if v is None:
            return default
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}

    def _as_float(v: str | None, name: str) -> float:
        if v is None:
            raise ValueError(f"Missing value for '{name}'.")
        try:
            return float(v.replace(",", " "))
        except Exception as e:
            raise ValueError(f"Invalid float for '{name}': {v}") from e

    def _as_int(v: str | None, name: str) -> int:
        return int(round(_as_float(v, name)))

    def _get(block: Dict[str, str], key: str) -> str | None:
        return block.get(key.lower())

    def _ensure_dict(obj, name: str) -> Dict[str, Any]:
        d = getattr(obj, name, None)
        if not isinstance(d, dict):
            d = {}
            setattr(obj, name, d)
        return d

    def _parse_utc_like(s: str) -> datetime:
        s = s.strip()
        fmts = [
            "%d/%m/%Y:%H:%M:%S",
            "%m/%d/%Y:%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise ValueError(f"Unrecognized UTC date format: {s!r}")

    # ---------- read & sectionize ----------
    path = os.path.join(os.path.abspath(project_folder), inputs_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Inputs file not found: {path}")

    env, sc, sim, exp = {}, {}, {}, {}
    cur = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = _clean(raw)
            if not s:
                continue
            low = s.lower()
            if "environment inputs" in low:
                cur = env; continue
            if "spacecraft inputs" in low:
                cur = sc; continue
            if "simulation inputs" in low:
                cur = sim; continue
            if "exporting inputs" in low:
                cur = exp; continue
            if cur is None or "=" not in s:
                continue
            k, v = map(str.strip, s.split("=", 1))
            cur[k.lower()] = v

    # ---------- simulation window ----------
    sim_start_str = _get(sim, "start date")
    if sim_start_str is None:
        raise ValueError("Missing 'start date' in Simulation block.")
    sim_end_str = _get(sim, "end_date")
    if sim_end_str is None:
        raise ValueError("Missing 'end_date' in Simulation block.")
    sim_start_dt = _parse_utc_like(sim_start_str)
    sim_end_dt = _parse_utc_like(sim_end_str)
    if not (sim_end_dt > sim_start_dt):
        raise ValueError("Simulation 'end_date' must be later than 'start date'.")
    eroder.sim_start_utc = sim_start_dt
    eroder.sim_end_utc = sim_end_dt

    # ---------- ensure dict containers ----------
    atmospheric = _ensure_dict(eroder, "atmospheric_data")
    spacecraft = _ensure_dict(eroder, "spacecraft_data")
    erosion = _ensure_dict(eroder, "erosion_data")

    # ---------- Environment ----------
    if _as_bool(_get(env, "constant conditions"), False):
        atmospheric["time"] = None
        atmospheric["ao_density"] = _as_float(_get(env, "neutral density"), "neutral density")
        atmospheric["speed_ratio"] = _as_float(_get(env, "speed ratio"), "speed ratio")
        atmospheric["velocity"] = None
    else:
        atm_file = _get(env, "atmospheric conditions file")
        if not atm_file:
            raise ValueError("Missing 'atmospheric conditions file' for non-constant Environment.")
        t_env, rho_env, sratio_env, vel_env = import_atmospheric_properties(
            project_folder, atm_file, sim_start_dt, sim_end_dt
        )
        atmospheric.update(time=t_env, ao_density=rho_env, speed_ratio=sratio_env, velocity=vel_env)

    # ---------- Spacecraft ----------
    if _as_bool(_get(sc, "constant conditions"), False):
        t = np.array([0.0, (sim_end_dt - sim_start_dt).total_seconds()], dtype=float)
        spacecraft["time"] = t
        th1 = _as_float(_get(sc, "incidence angle 1"), "incidence angle 1")
        th2 = _as_float(_get(sc, "incidence angle 2"), "incidence angle 2")
        Ts  = _as_float(_get(sc, "surface temperature"), "surface temperature")
        v_in = _as_float(_get(sc, "incident velocity"), "incident velocity")
        rho_surf = _as_float(_get(sc, "surface density"), "surface density")

        spacecraft["incident_angle_1_deg"] = th1 * np.ones_like(t)
        spacecraft["incident_angle_2_deg"] = th2 * np.ones_like(t)
        spacecraft["incident_angle_1"] = np.deg2rad(spacecraft["incident_angle_1_deg"])
        spacecraft["incident_angle_2"] = np.deg2rad(spacecraft["incident_angle_2_deg"])
        spacecraft["surface_temperature"] = Ts * np.ones_like(t)
        spacecraft["incident_velocity"] = v_in * np.ones_like(t)
        spacecraft["surface_density"] = rho_surf * np.ones_like(t)
    else:
        sc_file = _get(sc, "spacecraft data file")
        if not sc_file:
            raise ValueError("Missing 'spacecraft data file' for non-constant Spacecraft.")
        t_sc, th1_deg, th2_deg, Ts = import_spacecraft_properties(
            project_folder, sc_file, sim_start_dt, sim_end_dt
        )
        spacecraft["time"] = t_sc
        spacecraft["incident_angle_1_deg"] = th1_deg
        spacecraft["incident_angle_2_deg"] = th2_deg
        spacecraft["incident_angle_1"] = np.deg2rad(th1_deg)
        spacecraft["incident_angle_2"] = np.deg2rad(th2_deg)
        spacecraft["surface_temperature"] = Ts
        spacecraft["surface_density"] = _as_float(_get(sc, "surface density"), "surface density") * np.ones_like(t_sc)
        spacecraft["incident_velocity"] = (
            atmospheric.get("velocity") if isinstance(atmospheric.get("velocity"), np.ndarray) else None
        )

    # --- Energy accommodation coefficient from Spacecraft block; optional; default 1.0 ---
    eac_keys = [
        "energy acc coefficient",          # your current input file key
        "energy acc. coefficient",         # variant with dot
        "energy accommodation coefficient",
        "energy accommodation coeff",
    ]
    eac_str = None
    for k in eac_keys:
        if _get(sc, k) is not None:
            eac_str = _get(sc, k)
            break
    eroder.energy_accommodation_coeff = (
        _as_float(eac_str, "energy acc coefficient") if eac_str is not None else 1.0
    )

    # ---------- Material tables ----------
    mat_name = _get(sc, "surface material")
    if not mat_name:
        raise ValueError("Missing 'surface material' in Spacecraft block.")
    T_grid, V_grid, A_grid_deg, Y_grid, R_grid, EA_grid = import_material_data(mat_name.strip())
    erosion.update(
        T_grid=T_grid,
        velocity_grid=V_grid,
        angle_grid_rad=A_grid_deg * np.pi / 180.0,
        erosion_yield=Y_grid,
        reacted_fraction=R_grid,
        energy_accommodation=EA_grid,  # NEW
    )

    # ---------- PolyGaussian surface ----------
    srf_file = _get(sc, "polygaussian surface file")
    if not srf_file:
        raise ValueError("Missing 'polygaussian surface file' in Spacecraft block.")
    eroder.initial_polygaussian_surface = import_initial_polygaussian_surface(project_folder, srf_file, n_int=40)

    # ---------- Simulation grid; sizes; steps ----------
    eroder.Nx = _as_int(_get(sim, "grid points x"), "grid points x")
    eroder.Ny = _as_int(_get(sim, "grid points y"), "grid points y")
    eroder.Nz = _as_int(_get(sim, "grid points z"), "grid points z")

    eroder.surface_length  = _as_float(_get(sim, "sample size"), "sample size")       # µm
    eroder.domain_depth    = _as_float(_get(sim, "domain depth"), "domain depth")     # µm
    eroder.domain_padding  = _as_float(_get(sim, "domain padding"), "domain padding") # µm
    eroder.Lx, eroder.Ly, eroder.Lz = eroder.surface_length, eroder.domain_depth, eroder.domain_padding

    eroder.Nint = _as_int(_get(sim, "integration steps"), "integration steps")
    eroder.surface_angle = np.deg2rad(_as_float(_get(sim, "surface angle"), "surface angle"))

    # time steps in seconds
    eroder.time_step        = _as_float(_get(sim, "time step"), "time step") * 3600.0
    eroder.poly_update_step = _as_float(_get(sim, "polygaussian update step"), "polygaussian update step") * 3600.0
    eroder.print_step       = _as_float(_get(sim, "print time step"), "print time step") * 3600.0

    # number of reflections
    eroder.num_reflections = _as_int(_get(sim, "number of reflections"), "number of reflections")

    # ---------- Exporting flags ----------
    eroder.plot_history       = _as_bool(_get(exp, "plot erosion history?"), False)
    eroder.plot_poly_geometry = _as_bool(_get(exp, "plot polygaussian fitting?"), False)
    eroder.export_history     = _as_bool(_get(exp, "export erosion history?"), False)
    eroder.export_geometry    = _as_bool(_get(exp, "export surface geometry?"), False)



