import numpy as np


def normalize(
        pdf: np.ndarray,
        *deltas: float,
        nonnegative: bool = True,
        eps: float = 1e-12,
        rule: str = "trap",
) -> np.ndarray:
    """
    Normalize an N-D PDF sampled on a uniform rectilinear grid so it integrates to 1.

    You provide the grid spacings per axis (dx, dy, dz, ...), not the coordinate arrays.
    The integral is computed with either a simple Riemann sum or an N-D trapezoidal rule.

    Parameters
    ----------
    pdf : np.ndarray
        N-D array of (possibly unnormalized) PDF values on a uniform grid.
    *deltas : float
        Grid spacings per axis in the same order as `pdf` dimensions:
        e.g. (dx,) for 1D, (dx, dy) for 2D, (dx, dy, dz) for 3D, etc.
        You may also pass a single iterable, e.g. normalize_nd(pdf, (dx, dy)).
    nonnegative : bool, default True
        If True, clip PDF to be >= 0 before integrating/normalizing.
    eps : float, default 1e-12
        Small threshold to guard against near-zero or invalid integrals.
    rule : {"trap","sum"}, default "trap"
        - "trap": N-D trapezoidal rule (0.5 weight on each boundary along every axis).
        - "sum" : simple Riemann sum (uniform weights).

    Returns
    -------
    np.ndarray
        Normalized PDF with the same shape as `pdf`. If the integral is invalid or
        below `eps`, returns an array of zeros.

    Examples
    --------
    >>> x = np.linspace(0, 1, 101); dx = x[1]-x[0]
    >>> pdf = np.exp(-((x-0.5)**2)/0.01)
    >>> pdf_n = normalize_nd(pdf, dx)

    >>> nx, ny = 64, 64; dx = dy = 0.1
    >>> pdf2 = np.random.rand(nx, ny)
    >>> pdf2_n = normalize_nd(pdf2, dx, dy, rule="sum")
    """
    # Parse spacings (allow a single iterable)
    if len(deltas) == 1 and hasattr(deltas[0], "__iter__"):
        spacings = tuple(float(v) for v in deltas[0])
    else:
        spacings = tuple(float(v) for v in deltas)
    if len(spacings) != pdf.ndim:
        raise ValueError(f"Expected {pdf.ndim} spacings, got {len(spacings)}.")

    vol_elem = float(np.prod(spacings))

    # Sanitize PDF
    p = np.asarray(pdf, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    if nonnegative:
        p = np.clip(p, 0.0, np.inf)

    # Weights
    if rule.lower() == "trap":
        w = 1.0
        for ax, n in enumerate(p.shape):
            wa = np.ones(n, dtype=float)
            if n >= 2:
                wa[0] = wa[-1] = 0.5
            # reshape for broadcasting along axis ax
            shape = [1] * p.ndim
            shape[ax] = n
            w = w * wa.reshape(shape)
        integral = float(np.sum(p * w) * vol_elem)
    elif rule.lower() == "sum":
        integral = float(np.sum(p) * vol_elem)
    else:
        raise ValueError("rule must be 'trap' or 'sum'.")

    if not np.isfinite(integral) or abs(integral) < eps:
        return np.zeros_like(p, dtype=float)

    return p / integral
