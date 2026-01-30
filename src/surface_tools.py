import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statistics_tools as st


class Surface:
    """
    Lightweight container for a rectilinear height field Z(X, Y) with
    helpers to extract heights and compute slopes (∂Z/∂X, ∂Z/∂Y)
    using first-order forward differences.

    All outputs are sanitized to contain only finite numbers:
      - NaN and ±Inf in inputs are replaced with finite values.
      - Divisions by zero are guarded by a tiny epsilon.
    """

    _EPS = 1e-12

    @staticmethod
    def _safe_div(n: np.ndarray, d: np.ndarray, eps: float) -> np.ndarray:
        """Elementwise safe division n/d with zero/near-zero guards."""
        d_safe = np.where(np.isfinite(d), d, 0.0)
        d_safe = np.where(np.abs(d_safe) > eps, d_safe, np.where(d_safe >= 0.0, eps, -eps))
        out = n / d_safe
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _sanitize(arr: np.ndarray) -> np.ndarray:
        """
        Replace NaN with finite mean (or 0 if no finite values),
        and clip ±Inf to finite min/max of the array (or 0 if none).
        """
        arr = np.asarray(arr, dtype=float)
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return np.zeros_like(arr, dtype=float)
        finite_vals = arr[finite_mask]
        mean = float(np.mean(finite_vals))
        amin = float(np.min(finite_vals))
        amax = float(np.max(finite_vals))
        # First, map NaN to mean; then clip Infs to [amin, amax]
        out = np.where(np.isnan(arr), mean, arr)
        out = np.nan_to_num(out, nan=mean, posinf=amax, neginf=amin)
        # tiny final clip to ensure no spill due to numerical noise
        out = np.clip(out, amin, amax)
        return out

    def __init__(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        """
        Parameters
        ----------
        X, Y, Z : np.ndarray
            2D arrays of identical shape (nx, ny) describing a rectilinear grid.
            X[i,j] and Y[i,j] give coordinates; Z[i,j] is the surface height.

        Raises
        ------
        ValueError
            If inputs are not 2D or shapes do not match.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=float)

        if X.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            raise ValueError("X, Y, Z must be 2D arrays.")
        if X.shape != Y.shape or X.shape != Z.shape:
            raise ValueError("X, Y, Z must have identical shapes.")

        # Sanitize and store
        self.X = self._sanitize(X)
        self.Y = self._sanitize(Y)
        self.Z = self._sanitize(Z)

    def get_heights(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Flattened 1D array of surface heights (finite values only).
        """
        return self._sanitize(self.Z - np.average(self.Z)).ravel()

    def get_slopes_x(self) -> np.ndarray:
        """
        Forward-difference estimate of ∂Z/∂X along the first axis.

        Returns
        -------
        np.ndarray
            Array of shape (nx-1 * ny) containing finite slope values.

        Notes
        -----
        Uses (Z[i+1,j] - Z[i,j]) / (X[i+1,j] - X[i,j]) with robust division.
        """
        Z0 = self._sanitize(self.Z[1:, :])
        Z1 = self._sanitize(self.Z[:-1, :])
        dZ = Z0 - Z1

        X0 = self._sanitize(self.X[1:, :])
        X1 = self._sanitize(self.X[:-1, :])
        dX = X0 - X1

        slopes_x = self._safe_div(dZ, dX, self._EPS)
        return slopes_x.flatten()

    def get_slopes_y(self) -> np.ndarray:
        """
        Forward-difference estimate of ∂Z/∂Y along the second axis.

        Returns
        -------
        np.ndarray
            Array of shape (nx * ny-1) containing finite slope values.

        Notes
        -----
        Uses (Z[i,j+1] - Z[i,j]) / (Y[i,j+1] - Y[i,j]) with robust division.
        """
        Z0 = self._sanitize(self.Z[:, 1:])
        Z1 = self._sanitize(self.Z[:, :-1])
        dZ = Z0 - Z1

        Y0 = self._sanitize(self.Y[:, 1:])
        Y1 = self._sanitize(self.Y[:, :-1])
        dY = Y0 - Y1

        slopes_y = self._safe_div(dZ, dY, self._EPS)
        return slopes_y.flatten()

    def get_slopes(self, azimuth_angle: float) -> np.ndarray:
        """
        Directional slope samples along a given azimuth in the XY-plane.

        Parameters
        ----------
        azimuth_angle : float
            Azimuth (radians) measured from +X toward +Y. 0 → +X, π/2 → +Y.

        Returns
        -------
        np.ndarray
            Flattened array of shape ((nx-1) * (ny-1),) containing the
            directional slopes (finite values only) evaluated on the interior
            cell grid where ∂Z/∂X and ∂Z/∂Y overlap.

        Notes
        -----
        The directional derivative along unit vector u = (cos φ, sin φ) is
           s_φ = (∂Z/∂X) * cos φ + (∂Z/∂Y) * sin φ

        We reuse the forward-difference fields from `get_slopes_x()` and
        `get_slopes_y()`. Because they live on staggered grids—(nx-1, ny)
        and (nx, ny-1)—we crop to their common interior (nx-1, ny-1)
        before combining.
        """
        # sanitize angle and compute unit direction
        phi = float(np.nan_to_num(azimuth_angle, nan=0.0, posinf=0.0, neginf=0.0))
        c, s = np.cos(phi), np.sin(phi)

        nx, ny = self.Z.shape

        # Recover 2D slope fields from the flattened versions
        sx = self.get_slopes_x().reshape(nx - 1, ny)  # ∂Z/∂X on (nx-1, ny)
        sy = self.get_slopes_y().reshape(nx, ny - 1)  # ∂Z/∂Y on (nx, ny-1)

        # Crop to common interior (nx-1, ny-1)
        sx_int = sx[:, :-1]  # (nx-1, ny-1)
        sy_int = sy[:-1, :]  # (nx-1, ny-1)

        # Directional slope: s = sx * cosφ + sy * sinφ
        s_dir = self._sanitize(c * sx_int + s * sy_int)

        return s_dir.flatten()

    def get_isotropic_surface(self, angle: float | None = None) -> "Surface":
        """
        Return a new Surface with the large-scale tilt along +X removed
        (i.e., an "isotropized" surface).

        Parameters
        ----------
        angle : float | None, optional
            Tilt angle (radians) to remove from Z via Z_bias = -tan(angle) * X.
            If None, the angle is estimated from the data using `extract_angle()`.

        Returns
        -------
        Surface
            A new Surface instance with:
              X_iso = X,
              Y_iso = Y,
              Z_iso = Z - tan(angle) * X,
            with all arrays sanitized to contain only finite values.

        Notes
        -----
        - The input `angle` is clamped away from ±π/2 to avoid tan() overflow.
        - Any NaN/Inf encountered in intermediate steps are replaced with finite values.
        - If `angle` is non-finite, 0.0 is used as a safe fallback.
        """
        # Choose angle
        if angle is None:
            angle = 0 #self.extract_angle()

        # Ensure scalar float and finite
        try:
            angle = float(angle)
        except Exception:
            angle = 0.0

        if not np.isfinite(angle):
            angle = 0.0

        # Clamp away from tan singularities
        delta = 1e-6
        limit = np.pi / 2 - delta
        angle = float(np.clip(angle, -limit, limit))

        # Compute bias safely
        tan_angle = np.tan(angle)
        # X, Y, Z stored in this class are already sanitized, but keep it robust
        Xs = self._sanitize(self.X)
        Ys = self._sanitize(self.Y)
        Zs = self._sanitize(self.Z - np.mean(self.Z))

        Z_bias = np.nan_to_num(-tan_angle * Xs, nan=0.0, posinf=0.0, neginf=0.0)
        Z_iso = Zs - Z_bias
        Z_iso = self._sanitize(Z_iso)

        # Return a new sanitized Surface
        return Surface(Xs, Ys, Z_iso)

    def estimate_autocorrelation_length(self,
                                        fit_range: tuple[float, float] = (0.2, 0.9),
                                        nbins: int = 512,
                                        use_hann: bool = True) -> float:
        """
        Estimate the Gaussian autocorrelation length R from a height map via:
            1) FFT -> power spectrum  |F|^2
            2) IFFT of |F|^2 -> autocovariance (Wiener–Khinchin)
            3) radial average -> C(r) normalized to C(0)=1
            4) LSQ fit of  -ln C(r) = (1/R^2) * r^2  over a correlation range

        Parameters
        ----------
        fit_range : (float, float), default (0.2, 0.9)
            Use only radii where C(r) is between these bounds for the fit.
            (Keeps the small/medium-r region and avoids noisy tails.)
        nbins : int, default 512
            Number of radial bins for averaging.
        use_hann : bool, default True
            Apply 2D Hann window before FFT to reduce wrap-around bias.

        Returns
        -------
        float
            Estimated R (>= 0). Returns 0.0 if the surface is flat or unusable.
        """

        # --- data & spacing (safe) ---
        Z = self._sanitize(self.Z).astype(float, copy=False)
        X = self._sanitize(self.X).astype(float, copy=False)
        Y = self._sanitize(self.Y).astype(float, copy=False)
        nx, ny = Z.shape
        if nx < 2 or ny < 2:
            return 0.0

        with np.errstate(invalid="ignore"):
            dx = float(np.nanmedian(np.diff(X[:, 0]))) if nx > 1 else 1.0
            dy = float(np.nanmedian(np.diff(Y[0, :]))) if ny > 1 else 1.0
        if not np.isfinite(dx) or dx <= 0: dx = 1.0
        if not np.isfinite(dy) or dy <= 0: dy = 1.0

        # zero-mean & optional taper
        Z = Z - float(np.nanmean(Z))
        if use_hann:
            wx = np.hanning(nx)
            wy = np.hanning(ny)
            Z = (Z * wx[:, None]) * wy[None, :]

        # --- ACF via FFT (circular) ---
        F = np.fft.fft2(np.nan_to_num(Z, nan=0.0))
        S = np.abs(F) ** 2
        acov = np.fft.ifft2(S).real  # autocovariance (unnormalized)
        acov = np.fft.fftshift(acov)  # center at (0,0)
        c0 = float(acov[nx // 2, ny // 2])
        if not np.isfinite(c0) or c0 <= 0:
            return 0.0
        C = np.clip(acov / c0, 0.0, 1.0)  # normalized ACF

        # --- radial average ---
        iy = np.arange(ny) - ny // 2
        ix = np.arange(nx) - nx // 2
        IX, IY = np.meshgrid(ix, iy, indexing="ij")
        Rgrid = np.sqrt((IX * dx) ** 2 + (IY * dy) ** 2)

        r_max = 0.5 * min(dx * nx, dy * ny)  # safe radius (avoid far wrap)
        nb = max(int(nbins), 32)
        edges = np.linspace(0.0, r_max, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ridx = np.clip(np.digitize(Rgrid.ravel(), edges) - 1, 0, nb - 1)

        sums = np.bincount(ridx, weights=C.ravel(), minlength=nb)
        cnts = np.bincount(ridx, minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            C_rad = np.nan_to_num(sums / np.maximum(cnts, 1), nan=0.0)
        C_rad[0] = 1.0  # enforce C(0)=1

        # --- LSQ fit:  -ln C(r) = (1/R^2) * r^2  over chosen correlation band ---
        cmin, cmax = fit_range
        cmin = float(np.clip(cmin, 1e-6, 0.999))
        cmax = float(np.clip(cmax, cmin + 1e-6, 0.999))

        # keep small-to-mid radii where C in (cmin, cmax)
        mask = (centers > 0) & np.isfinite(C_rad) & (C_rad > cmin) & (C_rad < cmax)
        if np.count_nonzero(mask) < 5:
            # fallback: use everything above cmin if the band is too narrow
            mask = (centers > 0) & np.isfinite(C_rad) & (C_rad > cmin)

        if np.count_nonzero(mask) < 5:
            # final fallback: 1/e crossing (C=exp(-1)) => R ≈ r_e
            target = np.exp(-1.0)
            idx = np.where(C_rad[1:] < target)[0]
            if idx.size:
                k = idx[0] + 1
                r0, r1 = centers[k - 1], centers[k]
                c0, c1 = C_rad[k - 1], C_rad[k]
                t = (target - c0) / (c1 - c0 + 1e-12)
                Re = r0 + np.clip(t, 0.0, 1.0) * (r1 - r0)
                return float(np.nan_to_num(Re, nan=0.0, posinf=0.0, neginf=0.0))
            return 0.0

        r = centers[mask]
        y = -np.log(np.clip(C_rad[mask], 1e-12, 1.0))
        x = r * r

        denom = float(np.dot(x, x))
        if denom <= 0 or not np.isfinite(denom):
            return 0.0
        slope = float(np.dot(x, y) / denom)  # slope = 1/R^2
        if not np.isfinite(slope) or slope <= 0:
            return 0.0

        R = 1.0 / np.sqrt(slope)
        return float(np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0))


class PolyGaussian_Surface:
    """
        PolyGaussian surface model with piecewise-linear μ(γ) and σ(γ).

        The model assumes a latent standard normal variable γ ~ N(0, 1) and
        represents heights as

            H = μ(γ) + σ(γ) * ε,   with ε ~ N(0, 1),

        where μ(·) and σ(·) are defined by coefficients sampled uniformly over
        γ ∈ [gamma_min, gamma_max] and interpolated linearly with flat
        extrapolation beyond the knot range. The class also provides PDFs/CDFs
        for heights and slopes, parameter fitting against histogram data, and
        synthetic surface generation.

        Attributes
        ----------
        sigma_coeff : np.ndarray | None
            Coefficients defining σ(γ) at uniformly spaced γ-knots.
        mu_coeff : np.ndarray | None
            Coefficients defining μ(γ) at uniformly spaced γ-knots.
        R : float
            Autocorrelation length for the Gaussian ACF.
        angle : float
            Global tilt angle (radians) applied along +X when generating surfaces.
        parameter_type : str
            Reserved for future extensions; currently 'linear'.
        gamma_min, gamma_max : float
            Knot interval for μ/σ definitions.
        gamma_min_int, gamma_max_int : float
            Integration interval for numeric quadrature over γ.
        N_INT : int
            Number of quadrature points per γ-dimension.

        Notes
        -----
        - All public methods are robust to NaN/Inf in inputs and avoid divisions by
          zero by applying small floors where needed.
        - Piecewise-linear interpolation uses flat extrapolation outside the knot
          range, and the derivative is stepwise (constant per segment, zero outside).
        """

    _EPS = 1e-12

    def __init__(
            self,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float = 1.0,
            angle: float = 0.0,
            parameter_type: str = "linear",
            N_INT=40
    ):
        """
        Parameters
        ----------
        sigma_coeff, mu_coeff : np.ndarray | None
            Coefficients for σ(γ) and μ(γ); if None, safe defaults are used.
        R : float, default 1.0
            Autocorrelation length of the Gaussian ACF.
        angle : float, default 0.0
            Global tilt (radians) along +X used in `generate`.
        parameter_type : str, default 'linear'
            Interpolation type; currently only 'linear'.
        """
        self.sigma_coeff = sigma_coeff
        self.mu_coeff = mu_coeff
        self.R = float(R)
        self.angle = float(angle)
        self.parameter_type = str(parameter_type)

        # γ domain for knots and for integration
        self.gamma_min = -4.0
        self.gamma_max = 4.0
        self.gamma_min_int = -4.0
        self.gamma_max_int = 4.0

        # Quadrature resolution
        self.N_INT = N_INT

    def get_function(self, gamma_values: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        """
        Evaluate a safe piecewise-linear interpolant of a coefficient over gamma.

        The coefficient values `coeff` are assumed to be samples on uniformly spaced
        gamma knots spanning [self.gamma_min, self.gamma_max]. The function is
        linearly interpolated between knots and **flat-extrapolated** outside the
        knot range. Both inputs and coefficients are sanitized so the output
        contains only finite numbers.

        Parameters
        ----------
        gamma_values : np.ndarray
            Points (any shape) at which to evaluate the interpolant.
        coeff : np.ndarray
            1D (or broadcastable to 1D) array of coefficient samples. Length
            determines the number of knots; knots are placed at
            linspace(self.gamma_min, self.gamma_max, len(coeff)).

        Returns
        -------
        np.ndarray
            Interpolated values with the same shape as `gamma_values`. Guaranteed
            finite (NaN/±Inf are mapped to finite values).

        Notes
        -----
        - If `coeff` is empty, zeros are returned.
        - If `coeff` has no finite values, it is treated as all zeros.
        - If exactly one finite value exists, the function is constant at that value.
        - Non-finite `gamma_values` are mapped to the interval by replacing NaN with
          the midpoint and ±Inf with the nearest bound.
        """
        g = np.asarray(gamma_values, dtype=float)
        c = np.asarray(coeff, dtype=float).ravel()

        if c.size == 0:
            return np.zeros_like(g)

        # Repair coeffs: interpolate across finite entries, flat-fill outside
        finite = np.isfinite(c)
        if finite.sum() == 0:
            c[:] = 0.0
        elif finite.sum() == 1:
            c[:] = float(c[finite][0])
        else:
            idx = np.arange(c.size)
            vi, vv = idx[finite], c[finite].astype(float)
            c = np.interp(idx, vi, vv)
            c[:vi[0]] = vv[0]
            c[vi[-1] + 1:] = vv[-1]
        c = np.nan_to_num(c, nan=float(c[0]), neginf=float(c[0]), posinf=float(c[-1]))

        gmin = float(getattr(self, "gamma_min", -3.0))
        gmax = float(getattr(self, "gamma_max", 3.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            return np.full_like(g, float(c[-1]))

        knots = np.linspace(gmin, gmax, c.size)

        # Sanitize evaluation points
        g_mid = 0.5 * (gmin + gmax)
        g = np.nan_to_num(g, nan=g_mid, neginf=gmin, posinf=gmax)

        y = np.interp(g, knots, c, left=float(c[0]), right=float(c[-1]))
        return np.nan_to_num(y, nan=float(c[0]), neginf=float(c[0]), posinf=float(c[-1]))

    def get_function_derivative(self, gamma_values: np.ndarray, coeff: np.ndarray) -> np.ndarray:
        """
        Exact derivative of `get_function`'s piecewise-linear interpolant with flat
        extrapolation outside the knot range.

        - Knots are uniformly spaced: linspace(self.gamma_min, self.gamma_max, len(coeff)).
        - Between knots: derivative is constant (stepwise).
        - Outside [knots[0], knots[-1]]: derivative = 0 (flat extrapolation).
        - NaN/Inf in inputs or coeff are repaired; output is finite.

        Parameters
        ----------
        gamma_values : np.ndarray
            Points where the derivative is evaluated (any shape).
        coeff : np.ndarray
            Coefficient values at the knots (1D or broadcastable to 1D).

        Returns
        -------
        np.ndarray
            Derivative values with the same shape as `gamma_values`.
        """
        g = np.asarray(gamma_values, dtype=float)
        out_shape = g.shape
        g = g.ravel()

        c = np.asarray(coeff, dtype=float).ravel()
        if c.size <= 1:
            return np.zeros(out_shape, dtype=float)

        # Repair coeffs the same way as in get_function: interpolate across finite entries, flat-fill outside
        finite = np.isfinite(c)
        if finite.sum() == 0:
            c[:] = 0.0
        elif finite.sum() == 1:
            c[:] = float(c[finite][0])
        else:
            idx_all = np.arange(c.size)
            idx_f = idx_all[finite]
            val_f = c[finite].astype(float)
            c = np.interp(idx_all, idx_f, val_f)
            c[:idx_f[0]] = val_f[0]
            c[idx_f[-1] + 1:] = val_f[-1]
        c = np.nan_to_num(c, nan=float(c[0]), neginf=float(c[0]), posinf=float(c[-1]))

        # Knot range
        gmin = float(getattr(self, "gamma_min", -5.0))
        gmax = float(getattr(self, "gamma_max", 5.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            return np.zeros(out_shape, dtype=float)

        # Uniform knots and segment slopes
        n = c.size
        knots = np.linspace(gmin, gmax, n)
        denom = (gmax - gmin) / (n - 1)
        if not np.isfinite(denom) or denom == 0.0:
            return np.zeros(out_shape, dtype=float)
        slopes = np.diff(c) / denom  # length n-1

        # Sanitize evaluation points and look up segment index
        g_mid = 0.5 * (gmin + gmax)
        g_eval = np.nan_to_num(g, nan=g_mid, neginf=gmin, posinf=gmax)
        idx = np.searchsorted(knots, g_eval, side="right") - 1  # segment index

        # Derivative: piecewise constant on [knots[i], knots[i+1]); 0 outside
        dy = np.zeros_like(g_eval, dtype=float)
        mask = (idx >= 0) & (idx < n - 1)
        if np.any(mask):
            dy[mask] = slopes[idx[mask]]

        dy = np.nan_to_num(dy, nan=0.0, neginf=0.0, posinf=0.0)
        return dy.reshape(out_shape)

    def fit_parameters(
            self,
            surface: Surface,
            num_param: int = 10,
            method: str = 'hsr',
            bins: int = 40,
            lim_min: float = 0.5,
            niter: int = 40,
            verbose: bool = False,
    ) -> None:
        """
        Fit PolyGaussian coefficients to histogram data from a surface.

        Modes
        -----
        - 'hsr' : fit μ(gamma), σ(gamma), and R using height & slope histograms.
        - 's'   : fit μ(gamma) and σ(gamma) using slope histogram; R is fixed.
        - 'h'   : fit μ(gamma) and σ(gamma) using height histogram; R is fixed.

        Parameters
        ----------
        surface : Surface
            Input surface providing heights and x-slopes (can be angled).
        num_param : int
            The number of surface parameters to fit
        method : {'hsr','s','h'}, default 'hsr'
            Fitting mode (see above).
        bins : int, default 40
            Number of histogram bins.
        lim : float, default 8.0
            Symmetric histogram range [-lim, lim] for both heights and slopes.
        niter : int, default 20
            Number of basinhopping iterations.
        verbose : bool, default False
            If True, prints progress and a final summary.

        Notes
        -----
        - Uses density=True histograms.
        - σ coefficients are constrained positive via bounds.
        - Updates self.mu_coeff, self.sigma_coeff, and (if 'hsr') self.R in place.
        """
        # --- Build histograms (remove mean tilt) ---
        try:
            surf_angle = float(self.angle)
        except Exception:
            surf_angle = 0.0
        surf_angle = np.nan_to_num(surf_angle, nan=0.0)
        surface = surface.get_isotropic_surface(surf_angle)
        R0 = surface.estimate_autocorrelation_length()

        surf_heights = np.asarray(surface.get_heights(), dtype=float)
        surf_slopes_x = np.asarray(surface.get_slopes_x(), dtype=float)

        lim = max(lim_min, np.max(surf_heights) - np.min(surf_heights))
        lim_s = lim / max(0.1, R0)

        # Sanitize inputs
        surf_heights = np.nan_to_num(surf_heights, nan=0.0, posinf=0.0, neginf=0.0)
        surf_slopes_x = np.nan_to_num(surf_slopes_x, nan=0.0, posinf=0.0, neginf=0.0)

        p_h, bins_h = np.histogram(surf_heights, density=True, bins=bins, range=(-lim, lim))
        h_centers = 0.5 * (bins_h[:-1] + bins_h[1:])
        p_s, bins_s = np.histogram(surf_slopes_x, density=True, bins=bins, range=(-lim_s, lim_s))
        s_centers = 0.5 * (bins_s[:-1] + bins_s[1:])

        # --- Initial guesses ---
        mu0 = np.asarray(getattr(self, "mu_coeff", np.random.random(num_param)), dtype=float).ravel()
        s0 = np.asarray(getattr(self, "sigma_coeff", np.ones_like(mu0)), dtype=float).ravel()
        if mu0.size != s0.size or mu0.size < 2:
            n = num_param
            mu0 = np.zeros(n, dtype=float)
            s0 = np.ones(n, dtype=float)
        n_c = mu0.size

        if not np.isfinite(R0) or R0 <= 0:
            R0 = 1.0

        # --- Bounds (μ free, σ > 0, R > 0) ---
        mu_bounds = [(- lim, lim)] * n_c
        sig_bounds = [(1e-6, lim)] * n_c
        R_bounds = [(1e-3, 1e2)]

        # --- Pack params by mode ---
        if method == 'hsr':
            x0 = np.concatenate([mu0, s0, [R0]])
            bounds = mu_bounds + sig_bounds + R_bounds
        elif method in ('hs'):
            x0 = np.concatenate([mu0, s0, [R0]])
            bounds = mu_bounds + sig_bounds + R_bounds
        else:
            raise ValueError("method must be one of: 'hsr', 's', 'h'")

        EPS, BIG = 1e-10, 1e9

        def _cost(x: np.ndarray) -> float:
            # Unpack
            if method == 'hsr':
                mu_i = x[:n_c]
                sig_i = x[n_c:2 * n_c]
                R_ = float(x[-1])
            else:
                mu_i = x[:n_c]
                sig_i = x[n_c:2 * n_c]
                R_ = R0  # fixed

            # Sanity
            if (not np.isfinite(R_)) or (R_ <= 0) or (np.min(sig_i) <= EPS) \
                    or (not np.all(np.isfinite(mu_i))) or (not np.all(np.isfinite(sig_i))):
                return BIG

            loss = 0.0
            if method in ('hsr', 'hs'):
                ph = self.get_height_pdf(h_centers, sigma_coeff=sig_i, mu_coeff=mu_i)
                if (ph is None) or (not np.all(np.isfinite(ph))):
                    return BIG
                loss += np.linalg.norm(ph - p_h) / max(len(h_centers), 1) ** 0.5 * 1.0

            if method in ('hsr', 'hs'):
                ps = self.get_slope_pdf(s_centers, sigma_coeff=sig_i, mu_coeff=mu_i, R=R_)
                if (ps is None) or (not np.all(np.isfinite(ps))):
                    return BIG
                loss += np.linalg.norm(ps - p_s) / max(len(s_centers), 1) ** 0.5

            if method in ('hsr'):
                if (R_ is None) or (not np.all(np.isfinite(R_))):
                    return BIG
                #loss += np.abs(R_ - R0) * 2.0

            return float(np.nan_to_num(loss, nan=BIG, posinf=BIG, neginf=BIG))

        # Verbose header
        if verbose:
            print(f"[fit] mode={method}, bins={bins}, lim={lim}, niter={niter}, n_coeff={n_c}, R0={R0:.4g}")

        step = {"k": 0}

        def _bh_callback(x, f, accept):
            if verbose:
                step["k"] += 1
                print(f"[fit] iter {step['k']:>3}/{niter}: loss={f:.6g}  accepted={bool(accept)}")
            return False  # continue

        # --- Optimize ---
        result = sp.optimize.basinhopping(
            _cost,
            x0,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
            niter=niter,
            callback=_bh_callback if verbose else None,
        )

        optres = getattr(result, "lowest_optimization_result", None)
        success = bool(getattr(optres, "success", True))
        xbest = np.asarray(result.x, dtype=float)

        # --- Update parameters ---
        self.mu_coeff = xbest[:n_c]
        self.sigma_coeff = xbest[n_c:2 * n_c]
        self.R = float(xbest[-1])

        self.angle = surf_angle

        if verbose:
            print(f"[fit] done. success={success}  best_loss={result.fun:.6g}  iters={getattr(result, 'nit', 'NA')}")
            if method == 'hsr':
                print(f"[fit] R={self.R:.6g}")

    def get_height_pdf(
            self,
            points: np.ndarray,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            N_INT: int | None = None
    ) -> np.ndarray:
        """
        Height PDF p(h) of the PolyGaussian surface component via marginalization over a standard-normal latent γ:
            p(h) = ∫ N(γ;0,1) · N(h | μ(γ), σ(γ)) dγ
        evaluated by uniform quadrature on [gamma_min, gamma_max].

        Parameters
        ----------
        points : np.ndarray
            Heights where the PDF is evaluated (any shape).
        sigma_coeff : np.ndarray, optional
            Coefficients for σ(γ). Defaults to self.sigma_coeff.
        mu_coeff : np.ndarray, optional
            Coefficients for μ(γ). Defaults to self.mu_coeff.
        N_INT : int, optional
            The integration order

        Returns
        -------
        np.ndarray
            PDF values with the same shape as `points`. Finite and non-negative.
        """
        # Defaults
        sigma_coeff = self.sigma_coeff if sigma_coeff is None else sigma_coeff
        mu_coeff = self.mu_coeff if mu_coeff is None else mu_coeff
        N = self.N_INT if N_INT is None else N_INT

        # Sanitize evaluation points
        x = np.asarray(points, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        orig_shape = x.shape
        x = x.ravel()

        # Quadrature grid
        gmin = float(getattr(self, "gamma_min_int", -8.0))
        gmax = float(getattr(self, "gamma_max_int", 8.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            gmin, gmax = -5.0, 5.0
        gamma = np.linspace(gmin, gmax, N)
        dgamma = (gmax - gmin) / (N - 1)

        # μ(γ), σ(γ) (safe piecewise-linear with flat extrapolation)
        mu_g = np.asarray(self.get_function(gamma, mu_coeff), dtype=float)
        sigma_g = np.asarray(self.get_function(gamma, sigma_coeff), dtype=float)
        sigma_g = np.maximum(np.abs(np.nan_to_num(sigma_g, nan=1.0, posinf=1.0, neginf=1.0)), 1e-12)

        # Prior over γ: N(0,1)
        p_gamma = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * gamma ** 2)

        # Broadcast and integrate
        muG = mu_g[:, None]
        sigG = sigma_g[:, None]
        xB = x[None, :]
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigG)
        expo = np.exp(-0.5 * ((xB - muG) / sigG) ** 2)
        p = np.sum(p_gamma[:, None] * norm * expo, axis=0) * dgamma

        return np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).reshape(orig_shape)

    def get_height_cdf(
            self,
            points: np.ndarray,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            N_INT: int | None = None
    ) -> np.ndarray:
        """
        Height CDF F(h) of the PolyGaussian surface component via marginalization over a standard-normal latent γ:
            F(h) = ∫ N(γ;0,1) · Φ((h - μ(γ)) / σ(γ)) dγ,
        evaluated by uniform quadrature on [gamma_min_int, gamma_max_int].

        Parameters
        ----------
        points : np.ndarray
            Heights where the CDF is evaluated (any shape).
        sigma_coeff : np.ndarray, optional
            Coefficients for σ(γ). Defaults to self.sigma_coeff.
        mu_coeff : np.ndarray, optional
            Coefficients for μ(γ). Defaults to self.mu_coeff.
        N_INT : int, optional
            The integration order.

        Returns
        -------
        np.ndarray
            CDF values with the same shape as `points`. Finite and in [0, 1].
        """
        # Defaults
        sigma_coeff = self.sigma_coeff if sigma_coeff is None else sigma_coeff
        mu_coeff = self.mu_coeff if mu_coeff is None else mu_coeff
        N = self.N_INT if N_INT is None else N_INT

        # Sanitize evaluation points
        x = np.asarray(points, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        orig_shape = x.shape
        x = x.ravel()

        # Quadrature grid
        gmin = float(getattr(self, "gamma_min_int", -8.0))
        gmax = float(getattr(self, "gamma_max_int", 8.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            gmin, gmax = -5.0, 5.0
        gamma = np.linspace(gmin, gmax, N)
        dgamma = (gmax - gmin) / (N - 1)

        # μ(γ), σ(γ) (safe piecewise-linear with flat extrapolation)
        mu_g = np.asarray(self.get_function(gamma, mu_coeff), dtype=float)
        sigma_g = np.asarray(self.get_function(gamma, sigma_coeff), dtype=float)
        sigma_g = np.maximum(np.abs(np.nan_to_num(sigma_g, nan=1.0, posinf=1.0, neginf=1.0)), 1e-12)

        # Prior over γ: N(0,1)
        p_gamma = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * gamma ** 2)

        # Broadcast and integrate CDF Φ
        muG = mu_g[:, None]
        sigG = sigma_g[:, None]
        xB = x[None, :]
        z = (xB - muG) / sigG
        # Standard normal CDF Φ(z) = 0.5*(1 + erf(z/√2))
        Phi = 0.5 * (1.0 + sp.special.erf(z / np.sqrt(2.0)))

        F = np.sum(p_gamma[:, None] * Phi, axis=0) * dgamma
        F = np.clip(np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        return F.reshape(orig_shape)

    def get_slope_pdf(
            self,
            points: np.ndarray,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None
    ) -> np.ndarray:
        """
        Compute the slope PDF p(s) of the PolyGaussian surface component via numerical integration over (gamma, gamma_dot).

        Model:
            gamma ~ N(0, 1)
            gamma_dot ~ N(0, 2 / R^2)   (i.e., pdf = R / sqrt(4π) * exp(-(R*gamma_dot)^2 / 4))
            μ'(gamma) = derivative of the piecewise-linear μ(gamma)
            σ(gamma)  = piecewise-linear σ(gamma)
            σ'(gamma) = derivative of σ(gamma)

            Conditional slope distribution:
                s | gamma, gamma_dot ~ N( μ'(gamma) * gamma_dot,
                                          sqrt( (σ'(gamma) * gamma_dot)^2 + 2 * (σ(gamma)/R)^2 ) )

        The PDF is:
            p(s) = ∬ p_gamma(gamma) p_gamma_dot(gamma_dot)
                   N(s; μ'(gamma) * gamma_dot, sqrt((σ'(gamma) * gamma_dot)^2 + 2(σ(gamma)/R)^2))
                   dgamma dgamma_dot

        Parameters
        ----------
        points : np.ndarray
            Evaluation points (any shape).
        sigma_coeff : np.ndarray, optional
            Coefficients for σ(gamma). Defaults to self.sigma_coeff.
        mu_coeff : np.ndarray, optional
            Coefficients for μ(gamma). Defaults to self.mu_coeff.
        R : float, optional
            Autocorrelation length parameter. Defaults to self.R.
        N_INT : int, optional
            The integration order.

        Returns
        -------
        np.ndarray
            p_slope evaluated at `points`, same shape as `points`.
        """
        # Defaults and safety
        EPS = 1e-12
        sigma_coeff = self.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = self.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        N = self.N_INT if N_INT is None else N_INT
        R = float(self.R if R is None else R)
        if not np.isfinite(R) or R <= 0:
            R = 1.0

        # Integration grids
        N = max(N, 2)  # need at least 2 points
        gmin = float(getattr(self, "gamma_min_int", -8.0))
        gmax = float(getattr(self, "gamma_max_int", 8.0))
        gdmin = gmin * np.sqrt(2) / float(getattr(self, "R", 1.0))
        gdmax = gmax * np.sqrt(2) / float(getattr(self, "R", 1.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            gmin, gmax = -8.0, 8.0
        if not np.isfinite(gdmin) or not np.isfinite(gdmax) or gdmin == gdmax:
            gdmin, gdmax = -16.0, 16.0

        gamma = np.linspace(gmin, gmax, N)
        gdot = np.linspace(gdmin, gdmax, N)
        dgamma = (gmax - gmin) / (N - 1)
        dgd = (gdmax - gdmin) / (N - 1)

        pts = np.asarray(points, dtype=float)
        pts_flat = np.nan_to_num(pts.ravel(), nan=0.0, posinf=0.0, neginf=0.0)

        gamma_map, gdot_map, s_map = np.meshgrid(gamma, gdot, pts_flat, indexing="ij")

        # Distributions of gamma and gamma_dot
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            p_gamma = inv_sqrt_2pi * np.exp(-0.5 * gamma_map ** 2)
            p_gdot = (R / np.sqrt(4.0 * np.pi)) * np.exp(-0.25 * (R * gdot_map) ** 2)

        # μ' and σ, σ'
        mu_gprime = self.get_function_derivative(gamma_map, mu_coeff)
        sigma_g = self.get_function(gamma_map, sigma_coeff)
        sigma_gprime = self.get_function_derivative(gamma_map, sigma_coeff)

        # Ensure positive, non-zero scales
        sigma_g = np.maximum(np.abs(np.nan_to_num(sigma_g, nan=0.0, posinf=0.0, neginf=0.0)), EPS)

        # Conditional parameters of s | gamma, gamma_dot
        mu_dot = mu_gprime * gdot_map
        sigma_dot = np.sqrt((sigma_gprime * gdot_map) ** 2 + 2.0 * (sigma_g / R) ** 2)
        sigma_dot = np.maximum(np.abs(np.nan_to_num(sigma_dot, nan=0.0, posinf=0.0, neginf=0.0)), EPS)

        # Gaussian density in s
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            norm = inv_sqrt_2pi / sigma_dot
            z2 = ((s_map - mu_dot) / sigma_dot) ** 2
            gauss_s = norm * np.exp(-0.5 * z2)

            integrand = p_gamma * p_gdot * gauss_s

        p_slope_flat = np.sum(integrand, axis=(0, 1)) * dgamma * dgd
        p_slope_flat = np.nan_to_num(p_slope_flat, nan=0.0, posinf=0.0, neginf=0.0)

        return p_slope_flat.reshape(pts.shape)

    def get_slope_xy_pdf(
            self,
            points: np.ndarray,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
    ) -> np.ndarray:
        """
        Joint PDF of slope components (s_x, s_y) evaluated at given points.

        Model
        -----
        γ ~ N(0, 1),   γ̇_x, γ̇_y ~ N(0, 2 / R²) (independent of each other and of γ)

        With piecewise-linear μ(γ), σ(γ) and their stepwise derivatives μ'(γ), σ'(γ):
            s_x | γ,γ̇_x ~ N( μ'(γ) γ̇_x + tan(angle),
                             √{ (σ'(γ) γ̇_x)² + 2[σ(γ)/R]² } )
            s_y | γ,γ̇_y ~ N( μ'(γ) γ̇_y,
                             √{ (σ'(γ) γ̇_y)² + 2[σ(γ)/R]² } )

        The joint PDF factorizes conditionally on γ, so we compute:
            p(s_x, s_y) = ∫ p(γ) [ ∫ p(γ̇) N_x dγ̇ ] · [ ∫ p(γ̇) N_y dγ̇ ] dγ
        using simple Riemann quadrature (uniform weights).

        Parameters
        ----------
        points : np.ndarray
            Evaluation points with last dimension 2 (…, 2), where points[..., 0]=s_x and
            points[..., 1]=s_y. Values are sanitized to be finite.
        sigma_coeff : np.ndarray, optional
            Coefficients defining σ(γ). Defaults to `self.sigma_coeff`.
        mu_coeff : np.ndarray, optional
            Coefficients defining μ(γ). Defaults to `self.mu_coeff`.
        R : float, optional
            Autocorrelation length. Defaults to `self.R`.

        Returns
        -------
        np.ndarray
            p(s_x, s_y) with shape `points.shape[:-1]`. Values are finite and ≥ 0.

        Notes
        -----
        - Safer and more memory-efficient than forming a 5D mesh; integrates over γ̇
          separately for x and y, then integrates over γ.
        - All variances are clamped to a small positive floor to avoid divide-by-zero.
        """
        EPS = 1e-12

        # ---- Defaults & safety
        sigma_coeff = self.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = self.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(self.R if R is None else R)
        if not np.isfinite(R) or R <= 0:
            R = 1.0

        P = np.asarray(points, dtype=float)
        if P.ndim < 1 or P.shape[-1] != 2:
            raise ValueError("`points` must have last dimension 2: (..., 2) for (s_x, s_y).")
        orig_shape = P.shape[:-1]
        sx = np.nan_to_num(P[..., 0].ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        sy = np.nan_to_num(P[..., 1].ravel(), nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Quadrature grids
        N = max(int(getattr(self, "N_INT", 15)), 2)

        gmin = float(getattr(self, "gamma_min_int", -8.0))
        gmax = float(getattr(self, "gamma_max_int", 8.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            gmin, gmax = -8.0, 8.0
        gamma = np.linspace(gmin, gmax, N)
        dgamma = (gmax - gmin) / (N - 1)

        # For γ̇ ~ N(0, 2/R²), choose symmetric bounds that cover several std devs
        # std(γ̇) = √2 / R  → span proportional to (gmax - gmin) * √2 / R
        gd_span = (gmax - gmin) * np.sqrt(2.0) / R
        gdmin, gdmax = -gd_span, gd_span
        if not np.isfinite(gdmin) or not np.isfinite(gdmax) or gdmin == gdmax:
            gdmin, gdmax = -16.0, 16.0
        gdot = np.linspace(gdmin, gdmax, N)
        dgd = (gdmax - gdmin) / (N - 1)

        # ---- Prior over γ and γ̇
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        p_gamma = inv_sqrt_2pi * np.exp(-0.5 * gamma ** 2)  # (N,)
        p_gdot = (R / np.sqrt(4.0 * np.pi)) * np.exp(-0.25 * (R * gdot) ** 2)  # (N,)

        # ---- μ, σ and their derivatives at γ-knots
        sig_g = self.get_function(gamma, sigma_coeff)  # (N,)
        mu_gp = self.get_function_derivative(gamma, mu_coeff)  # (N,)
        sig_gp = self.get_function_derivative(gamma, sigma_coeff)  # (N,)

        sig_g = np.maximum(np.abs(np.nan_to_num(sig_g, nan=0.0, posinf=0.0, neginf=0.0)), EPS)

        # ---- Angle shift
        angle = float(getattr(self, "angle", 0.0))
        tan_angle = np.tan(angle) if np.isfinite(angle) else 0.0

        # ---- Prepare broadcast shapes
        # gdot: (N, 1) to combine with many evaluation points
        gdot_col = gdot[:, None]  # (N, 1)
        p_gdot_col = p_gdot[:, None]  # (N, 1)
        # evaluation points vectors
        sx_row = sx[None, :]  # (1, M)
        sy_row = sy[None, :]  # (1, M)

        # ---- Integrate over γ̇ for x and y, for each γ (loop over γ only; vectorize others)
        # We accumulate lx(gamma_i, sx) and ly(gamma_i, sy) then integrate over γ.
        lx = np.empty((N, sx_row.shape[1]), dtype=float)  # (N, M)
        ly = np.empty((N, sy_row.shape[1]), dtype=float)  # (N, M)

        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            for i in range(N):
                mu_dot_x = mu_gp[i] * gdot_col + tan_angle  # (N,1)
                mu_dot_y = mu_gp[i] * gdot_col  # (N,1)
                sig_dot_x = np.sqrt((sig_gp[i] * gdot_col) ** 2 + 2.0 * (sig_g[i] / R) ** 2)  # (N,1)
                sig_dot_y = np.sqrt((sig_gp[i] * gdot_col) ** 2 + 2.0 * (sig_g[i] / R) ** 2)  # (N,1)

                sig_dot_x = np.maximum(np.abs(np.nan_to_num(sig_dot_x, nan=0.0, posinf=0.0, neginf=0.0)), EPS)
                sig_dot_y = np.maximum(np.abs(np.nan_to_num(sig_dot_y, nan=0.0, posinf=0.0, neginf=0.0)), EPS)

                # Gaussian densities in s_x and s_y, integrated over γ̇ with weights p_gdot
                # Shapes: (N,1) vs (1,M) → broadcast to (N,M), then sum over γ̇ (axis=0)
                norm_x = inv_sqrt_2pi / sig_dot_x
                norm_y = inv_sqrt_2pi / sig_dot_y

                zx2 = ((sx_row - mu_dot_x) / sig_dot_x) ** 2
                zy2 = ((sy_row - mu_dot_y) / sig_dot_y) ** 2

                gx = (norm_x * np.exp(-0.5 * zx2)) * p_gdot_col  # (N,M)
                gy = (norm_y * np.exp(-0.5 * zy2)) * p_gdot_col  # (N,M)

                lx[i, :] = np.sum(gx, axis=0) * dgd  # (M,)
                ly[i, :] = np.sum(gy, axis=0) * dgd  # (M,)

        # ---- Integrate over γ
        # For each γ_i: factorized contribution lx[i,:] * ly[i,:]
        # Weight by p_gamma[i] and dgamma and sum over i
        out = np.sum((p_gamma[:, None] * lx * ly), axis=0) * dgamma  # (M,)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out.reshape(orig_shape)

    def sample_height_pdf(self, samples: int | tuple[int, ...]) -> np.ndarray:
        """
        Draw random height samples from the PolyGaussian height model.

        The model assumes a latent normal variable γ ~ N(0, 1). Heights are
        generated as:
            H = μ(γ) + σ(γ) * epsilon,   with epsilon ~ N(0, 1),
        where μ(·) and σ(·) are the piecewise-linear functions defined by
        `self.mu_coeff` and `self.sigma_coeff` (via `get_function`).

        This implementation is robust to NaN/Inf in μ/σ evaluations and enforces
        a small positive floor on σ.

        Parameters
        ----------
        samples : int or tuple[int, ...]
            Number of samples (if int) or the desired output shape.

        Returns
        -------
        np.ndarray
            Array of sampled heights with shape `samples`.
        """
        rng = np.random.default_rng()
        shape = (samples,) if isinstance(samples, int) else tuple(samples)

        # Latent normal draws
        gamma = rng.standard_normal(size=shape)

        # Evaluate μ(γ) and σ(γ) safely
        mu = np.asarray(self.get_function(gamma, getattr(self, "mu_coeff", np.array([0.0]))), dtype=float)
        sigma = np.asarray(self.get_function(gamma, getattr(self, "sigma_coeff", np.array([1.0]))), dtype=float)

        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        sigma = np.maximum(sigma, 1e-12)  # ensure strictly positive std dev

        # Height samples
        epsilon = rng.standard_normal(size=shape)
        heights = mu + sigma * epsilon
        return np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

    def generate(self, sample_length: float, samples: int, iterations: int = 10, verbose: bool = False) -> Surface:
        """
        Synthesize a PolyGaussian surface via IAFFT (Iterative Amplitude-Adjusted FFT).

        The method generates two latent Gaussian fields, γ(x,y) and ε(x,y), then
        iteratively enforces a target power spectrum (Gaussian ACF with length R)
        while preserving standard-normal marginals by rank-ordering. The height
        field is constructed as:
            Z(x,y) = μ(γ) + σ(γ) * ε  +  tan(angle) * X

        All computations are made robust to NaN/Inf and degenerate parameters.

        Parameters
        ----------
        sample_length : float
            Physical side length of the square patch (same units as R).
        samples : int
            Number of grid points per side (output is samples×samples).
        iterations : int, default 10
            IAFFT iterations; higher → better PSD match (slower).
        verbose : bool, default False
            If True, prints simple progress messages.

        Returns
        -------
        Surface
            A `Surface(X, Y, Z)` object with the synthesized height field.
        """
        # --- validate & basic setup ---
        samples = int(max(2, samples))
        iterations = int(max(0, iterations))
        R = float(getattr(self, "R", 1.0))
        angle = float(getattr(self, "angle", 0.0))
        R = R if np.isfinite(R) and R > 0 else 1.0

        # Grid
        x = np.linspace(-0.5 * sample_length, 0.5 * sample_length, samples)
        y = np.linspace(-0.5 * sample_length, 0.5 * sample_length, samples)
        X, Y = np.meshgrid(x, y, indexing="ij")

        rng = np.random.default_rng()
        shp = (samples, samples)

        # Latent fields with standard-normal marginals
        gamma = rng.standard_normal(shp)
        eps = rng.standard_normal(shp)
        ref_gamma_sorted = np.sort(gamma.ravel())
        ref_eps_sorted = np.sort(eps.ravel())

        # Target amplitude spectrum (Gaussian ACF kernel)
        kernel = np.exp(-((X * X + Y * Y) / (R * R * 0.5)))
        psd_amp = np.abs(np.fft.fft2(kernel))
        psd_amp = np.nan_to_num(psd_amp, nan=0.0, posinf=0.0, neginf=0.0)

        def _enforce_psd_and_marginals(field: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
            """One IAFFT step: set amplitude to psd_amp, keep phase; then rank-match."""
            F = np.fft.fft2(field)
            F = psd_amp * np.exp(1j * np.angle(F))
            x_spatial = np.real(np.fft.ifft2(F))
            flat = x_spatial.ravel()
            order = np.argsort(flat)
            out = np.empty_like(flat)
            out[order] = ref_sorted  # assign sorted reference to sorted positions
            return out.reshape(shp)

        # --- IAFFT loop ---
        for it in range(iterations):
            gamma = _enforce_psd_and_marginals(gamma, ref_gamma_sorted)
            eps = _enforce_psd_and_marginals(eps, ref_eps_sorted)
            if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
                print(f"IAFFT iteration {it + 1}/{iterations}")

        # Evaluate μ(γ) and σ(γ) safely
        mu = np.asarray(self.get_function(gamma, getattr(self, "mu_coeff", np.array([0.0]))), dtype=float)
        sigma = np.asarray(self.get_function(gamma, getattr(self, "sigma_coeff", np.array([1.0]))), dtype=float)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = np.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0)
        sigma = np.maximum(sigma, 1e-12)

        # Construct height field with planar bias along +x
        Z = mu + sigma * eps
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        tan_angle = np.tan(angle) if np.isfinite(angle) else 0.0
        Z = Z - tan_angle * X

        return Surface(X, Y, Z)
