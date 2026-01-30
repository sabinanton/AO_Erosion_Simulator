import surface_tools as surf
import statistics_tools as stat
import constants as cst
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)


def meters_to_micrometers(quantity: any):
    return quantity * 1e6


class Scatterer:
    """
    Scattering/visibility helper built on a PolyGaussian surface model.

    This class provides a set of radiometric kernels and integrals for
    PolyGaussian-shaped random rough surfaces (height and slope statistics are
    driven by a latent standard normal γ with piecewise-linear μ(γ) and σ(γ)).
    It exposes:
      - Shadowing/visibility terms in angle–only or angle–height form,
      - Latent-variable mixture PDFs,
      - Normal-angle PDFs conditioned on incidence (with visibility masking),
      - Scattering PDFs over reflected directions,
      - Trapping probability over reflected/incident hemispheres,
      - Angular flux for rays arriving at (θ, φ) from height h.

    Conventions
    -----------
    - Angles are in **radians**.
    - Polar angle θ ∈ [0, π], azimuth φ ∈ [0, 2π).
    - Incident direction: (θ_i, φ_i). Reflected direction: (θ_r, φ_r).
    - Surface normal: (θ_n, φ_n) with θ_n ∈ [0, π/2] (upper hemisphere).
    - All integrals use uniform grids with trapezoidal or Riemann summation.
    - Inputs are sanitized; divisions are guarded with small epsilons.

    Parameters
    ----------
    surface : surf.PolyGaussian_Surface, optional
        The PolyGaussian surface providing μ/σ coefficient arrays, their
        piecewise-linear evaluators and derivatives, surface tilt `angle`,
        autocorrelation length `R`, and γ-domain bounds for quadrature.

    Attributes
    ----------
    poly_surface : surf.PolyGaussian_Surface | None
        Bound PolyGaussian surface model. Required for all computations that
        depend on μ(γ), σ(γ), their derivatives, `R`, or `angle`.
    N_INT : int
        Default quadrature resolution used across methods (minimum 2).
        Methods accept an `N_INT` override per call.

    Notes
    -----
    - All returned PDFs are non-negative; some are normalized over their natural
      domain if a rectilinear grid is detected (see method docstrings).
    - Set `self.N_INT` (or pass `N_INT=`) to trade off accuracy vs. speed.
    - The class is numerically defensive (NaN/Inf mapping, epsilon guards,
      clamping of angles near tan/cot singularities).
    """

    def __init__(self, surface: surf.PolyGaussian_Surface = None):
        """
        Initialize a Scatterer bound to a PolyGaussian surface.

        Parameters
        ----------
        surface : surf.PolyGaussian_Surface, optional
            The surface model providing μ/σ functions, their derivatives,
            and global parameters (R, angle, γ-integration bounds).
            You can assign/replace it later via `self.poly_surface`.

        Notes
        -----
        The default quadrature resolution `N_INT` is set to 15; increase for
        higher-accuracy integrals at the cost of runtime.
        """
        self.poly_surface = surface
        self.N_INT = 15

    def shadow_function(
            self,
            angle_1: np.ndarray,
            angle_2: np.ndarray,
            height: np.ndarray,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
    ) -> np.ndarray:
        ps = self.poly_surface
        EPS = 1e-12

        # --- validate & sanitize 1D inputs ---
        a1 = np.asarray(angle_1, float).ravel()
        a2 = np.asarray(angle_2, float).ravel()
        h = np.asarray(height, float).ravel()
        if not (a1.ndim == a2.ndim == h.ndim == 1 and a1.size == a2.size == h.size and a1.size > 0):
            raise ValueError("angle_1, angle_2, and height must be 1D arrays of the same non-zero length.")
        a1 = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)
        a2 = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # --- NEW: block any rays with θ1 > π/2 ---
        m_block = (a1 > (0.5 * np.pi))
        if np.all(m_block):
            # everything is blocked -> S = 0
            return np.zeros_like(a1)

        # defaults
        sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(ps.R if R is None else R)
        R = 1.0 if (not np.isfinite(R) or R <= 0) else R
        N = int(self.N_INT if N_INT is None else N_INT)
        N = max(N, 2)

        # height CDF F(h)
        try:
            F = ps.get_height_cdf(h, sigma_coeff, mu_coeff)
        except TypeError:
            F = ps.get_height_cdf(h)
        F = np.clip(np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        # quadrature & setup (unchanged) ...
        gmin = float(getattr(ps, "gamma_min_int", -8.0))
        gmax = float(getattr(ps, "gamma_max_int", 8.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            gmin, gmax = -8.0, 8.0
        gamma = np.linspace(gmin, gmax, N)
        gdot = np.linspace(gmin * np.sqrt(2) / R, gmax * np.sqrt(2) / R, N)

        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        p_g = inv_sqrt_2pi * np.exp(-0.5 * gamma ** 2)
        p_gd = (R / np.sqrt(4.0 * np.pi)) * np.exp(-0.25 * (R * gdot) ** 2)

        mu_gprime = ps.get_function_derivative(gamma, coeff=None if mu_coeff is None else mu_coeff)
        sigma_g = ps.get_function(gamma, sigma_coeff)
        sigma_g = np.maximum(np.abs(np.nan_to_num(sigma_g, 0.0)), EPS)
        sigma_gprim = ps.get_function_derivative(gamma, sigma_coeff)

        a1_map, g_map, gd_map = np.meshgrid(a1, gamma, gdot, indexing="ij")
        a2_map = np.broadcast_to(a2[:, None, None], a1_map.shape)
        mu_gp = mu_gprime[None, :, None]
        sig_g = sigma_g[None, :, None]
        sig_gp = sigma_gprim[None, :, None]
        pG = p_g[None, :, None]
        pGD = p_gd[None, None, :]

        tan_angle = np.tan(float(ps.angle)) if np.isfinite(ps.angle) else 0.0
        mu_dot = mu_gp * gd_map
        sigma_dot = np.sqrt((sig_gp * gd_map) ** 2 + 2.0 * (sig_g / R) ** 2)
        sigma_dot = np.maximum(np.abs(np.nan_to_num(sigma_dot, 0.0)), EPS)

        t1 = np.tan(a1_map)
        t1 = np.where(np.isfinite(t1), t1, 0.0)
        t1 = np.where(np.abs(t1) > EPS, t1, np.sign(t1) * EPS + (np.abs(t1) <= EPS) * EPS)
        eta = (1.0 / t1) + tan_angle * np.cos(a2_map)
        eta = np.where(np.abs(eta) > EPS, eta, np.sign(eta) * EPS + (np.abs(eta) <= EPS) * EPS)

        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            z = (eta - mu_dot)
            term1 = (sigma_dot * inv_sqrt_2pi) * np.exp(-0.5 * (z / sigma_dot) ** 2)
            term2 = 0.5 * z * sp.special.erfc(z / (np.sqrt(2.0) * sigma_dot))
            integr = (term1 - term2) / eta * pG * pGD
            integr = np.nan_to_num(integr, 0.0)

        Delta = np.trapezoid(np.trapezoid(integr, gdot, axis=2), gamma, axis=1)
        Delta = np.nan_to_num(Delta, 0.0)

        Shadow = np.exp(np.clip(Delta, -1e6, 1e6) * np.log(np.clip(F, EPS, 1.0)))
        Shadow = np.clip(np.nan_to_num(Shadow, 0.0), 0.0, 1.0)

        # --- NEW: force shadow = 0 for blocked rays (θ1 > π/2) ---
        Shadow[m_block] = 0.0
        return Shadow

    def shadow_angle_function(
            self,
            angle_1: np.ndarray,
            angle_2: np.ndarray,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
    ) -> np.ndarray:
        ps = self.poly_surface
        EPS = 1e-12

        # ---- validate & sanitize 1D angle arrays ----
        a1 = np.asarray(angle_1, float).ravel()
        a2 = np.asarray(angle_2, float).ravel()
        if a1.ndim != 1 or a2.ndim != 1 or a1.size != a2.size or a1.size == 0:
            raise ValueError("angle_1 and angle_2 must be 1D arrays of the same non-zero length.")
        a1 = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)
        a2 = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- NEW: block rays with θ > π/2 ----
        m_block = (a1 > (0.5 * np.pi))
        if np.all(m_block):
            return np.zeros_like(a1)

        # ---- defaults / parameters ----
        sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(ps.R if R is None else R)
        if not np.isfinite(R) or R <= 0:
            R = 1.0
        N = int(self.N_INT if N_INT is None else N_INT)
        N = max(N, 2)

        # ---- quadrature grids (γ, γ̇) ----
        gmin = float(getattr(ps, "gamma_min_int", -8.0))
        gmax = float(getattr(ps, "gamma_max_int", 8.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            gmin, gmax = -8.0, 8.0
        gamma = np.linspace(gmin, gmax, N)  # (G,)
        gdot = np.linspace(gmin * np.sqrt(2) / R, gmax * np.sqrt(2) / R, N)  # (D,)

        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        p_g = inv_sqrt_2pi * np.exp(-0.5 * gamma ** 2)  # (G,)
        p_gd = (R / np.sqrt(4.0 * np.pi)) * np.exp(-0.25 * (R * gdot) ** 2)  # (D,)

        mu_gprime = np.asarray(ps.get_function_derivative(gamma, mu_coeff), float)  # (G,)
        sigma_g = np.asarray(ps.get_function(gamma, sigma_coeff), float)  # (G,)
        sigma_g = np.maximum(np.abs(np.nan_to_num(sigma_g, nan=1.0, posinf=1.0, neginf=1.0)), EPS)
        sigma_gprim = np.asarray(ps.get_function_derivative(gamma, sigma_coeff), float)  # (G,)

        # ---- broadcast to (A, G, D) ----
        a1_map, g_map, gd_map = np.meshgrid(a1, gamma, gdot, indexing="ij")
        a2_map = np.broadcast_to(a2[:, None, None], a1_map.shape)
        mu_gp = mu_gprime[None, :, None]
        sig_g = sigma_g[None, :, None]
        sig_gp = sigma_gprim[None, :, None]
        pG = p_g[None, :, None]
        pGD = p_gd[None, None, :]

        # Surface tilt
        angle_s = float(np.nan_to_num(getattr(ps, "angle", 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        angle_s = float(np.clip(angle_s, -(np.pi / 2 - 1e-6), (np.pi / 2 - 1e-6)))
        tan_s = np.tan(angle_s)

        # Conditional params
        mu_dot = mu_gp * gd_map# + tan_s * np.cos(a2_map)
        sigma_dot = np.sqrt((sig_gp * gd_map) ** 2 + 2.0 * (sig_g / R) ** 2)
        sigma_dot = np.maximum(np.abs(np.nan_to_num(sigma_dot, nan=0.0, posinf=0.0, neginf=0.0)), EPS)

        # η = cot(theta) + tan(angle_s)*cos(phi)
        s1 = np.sin(a1_map)
        c1 = np.cos(a1_map)
        s1_safe = np.where(np.abs(s1) > EPS, s1, np.sign(s1) * EPS + (np.abs(s1) <= EPS) * EPS)
        cot_a1 = c1 / s1_safe
        eta = cot_a1 + tan_s * np.cos(a2_map)
        eta = np.where(np.abs(eta) > EPS, eta, np.sign(eta) * EPS + (np.abs(eta) <= EPS) * EPS)

        # ---- Δ integrand and integration ----
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            z = (eta - mu_dot)
            term1 = (sigma_dot * inv_sqrt_2pi) * np.exp(-0.5 * (z / sigma_dot) ** 2)
            term2 = 0.5 * z * sp.special.erfc(z / (np.sqrt(2.0) * sigma_dot))
            integr = (term1 - term2) / eta * pG * pGD
            integr = np.nan_to_num(integr, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            Delta = np.trapezoid(np.trapezoid(integr, gdot, axis=2), gamma, axis=1)
        except Exception:
            dgd = (gdot[-1] - gdot[0]) / max(N - 1, 1)
            dgam = (gamma[-1] - gamma[0]) / max(N - 1, 1)
            Delta = np.sum(integr, axis=(1, 2)) * dgd * dgam

        Delta = np.nan_to_num(Delta, nan=0.0, posinf=0.0, neginf=0.0)
        Shadow = 1.0 / (1.0 + np.maximum(Delta, 0.0))
        Shadow = np.clip(np.nan_to_num(Shadow, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

        # ---- NEW: enforce S=0 for θ > π/2 ----
        Shadow[m_block] = 0.0
        return Shadow

    def get_mixture_pdf(
            self,
            points: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
            incident_angle_1: float,
            incident_angle_2: float,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
            first: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the (normalized) 3D mixture PDF over latent variables (γ, γ̇x, γ̇y)
        at a *vector* of points, using the same convention as elsewhere:

          - `points` is either a tuple of three (N,) arrays (gamma, gdot_x, gdot_y),
            or an array of shape (3, N) or (N, 3).
          - Returns a (N,) array: one value per provided point.
          - The density is normalized via a Riemann sum using spacings inferred
            from the unique coordinates in each dimension.

        Model ingredients
        -----------------
        γ ~ N(0, 1)
        γ̇ ~ N(0, 2/R²)  →  p(γ̇) = R / sqrt(4π) * exp(-(R γ̇)² / 4)
        μ(γ), σ(γ) piecewise-linear; μ'(γ), σ'(γ) piecewise-constant.
        Occlusion(γ, γ̇x, γ̇y) matches your provided formula.
        Optional Shadow term depends only on γ (via height CDF).

        Parameters
        ----------
        points : array-like
            (gamma, gdot_x, gdot_y) as (3, N), (N, 3), or a tuple of three (N,) arrays.
        incident_angle_1 : float
            Incident polar angle θ (radians).
        incident_angle_2 : float
            Incident azimuth φ (radians).
        sigma_coeff, mu_coeff : np.ndarray, optional
            Coeffs for σ(γ) and μ(γ). Defaults to self.poly_surface values.
        R : float, optional
            Autocorrelation length; defaults to self.poly_surface.R (>0 enforced).
        N_INT : int, optional
            Passed through to shadow_function for its internal quadrature (if used).
        first : bool, default False
            If True, multiplies by Shadow(γ); else Shadow=1.

        Returns
        -------
        np.ndarray
            Normalized mixture density evaluated at the provided points, shape (N,).
        """
        ps = self.poly_surface
        EPS = 1e-12

        # --- Parse points into 1D arrays: gamma, gdx, gdy (all length N) ---
        if isinstance(points, (tuple, list)) and len(points) == 3:
            gamma, gdx, gdy = (np.asarray(a, float).ravel() for a in points)
        else:
            P = np.asarray(points, float)
            if P.ndim == 2 and P.shape[0] == 3:
                gamma, gdx, gdy = P[0].ravel(), P[1].ravel(), P[2].ravel()
            elif P.ndim == 2 and P.shape[1] == 3:
                gamma, gdx, gdy = P[:, 0].ravel(), P[:, 1].ravel(), P[:, 2].ravel()
            else:
                raise ValueError("`points` must be (gamma, gdot_x, gdot_y) or array of shape (3,N) or (N,3).")

        Np = gamma.size
        if Np == 0 or gdx.size != Np or gdy.size != Np:
            return np.zeros(Np, dtype=float)

        # Sanitize inputs
        gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        gdx = np.nan_to_num(gdx, nan=0.0, posinf=0.0, neginf=0.0)
        gdy = np.nan_to_num(gdy, nan=0.0, posinf=0.0, neginf=0.0)

        # Parameters / coeffs
        sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(ps.R if R is None else R)
        if not np.isfinite(R) or R <= 0:
            R = 1.0

        N = int(self.N_INT if N_INT is None else N_INT)
        N = max(N, 2)

        # Clamp incident polar angle away from tan singularity
        def _clamp(a: float) -> float:
            try:
                a = float(a)
            except Exception:
                a = 0.0
            lim = np.pi / 2 - 1e-6
            return float(np.clip(a, -lim, lim))

        th = _clamp(incident_angle_1)
        ph = float(np.nan_to_num(incident_angle_2, nan=0.0, posinf=0.0, neginf=0.0))
        tan_th = np.tan(th)
        # Safe cotangent
        tan_safe = tan_th if abs(tan_th) > EPS else (EPS if tan_th >= 0 else -EPS)
        cot_th = 1.0 / tan_safe
        tan_angle = np.tan(float(np.nan_to_num(ps.angle, nan=0.0, posinf=0.0, neginf=0.0)))

        # Priors
        inv_s2pi = 1.0 / np.sqrt(2.0 * np.pi)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            p_gamma = inv_s2pi * np.exp(-0.5 * gamma ** 2)
            p_gdx = (R / np.sqrt(4.0 * np.pi)) * np.exp(-0.25 * (R * gdx) ** 2)
            p_gdy = (R / np.sqrt(4.0 * np.pi)) * np.exp(-0.25 * (R * gdy) ** 2)

        # μ(γ), σ(γ) and derivatives at the provided γ values
        mu_g = np.asarray(ps.get_function(gamma, mu_coeff), float)
        sig_g = np.asarray(ps.get_function(gamma, sigma_coeff), float)
        mu_gprime = np.asarray(ps.get_function_derivative(gamma, mu_coeff), float)
        sig_gprime = np.asarray(ps.get_function_derivative(gamma, sigma_coeff), float)

        sig_g = np.maximum(np.abs(np.nan_to_num(sig_g, nan=1.0, posinf=1.0, neginf=1.0)), EPS)
        mu_gprime = np.nan_to_num(mu_gprime, nan=0.0, posinf=0.0, neginf=0.0)
        sig_gprime = np.nan_to_num(sig_gprime, nan=0.0, posinf=0.0, neginf=0.0)

        # Conditional slope stats at each point
        mu_dx = mu_gprime * gdx
        mu_dy = mu_gprime * gdy
        sig_dx = np.sqrt((sig_gprime * gdx) ** 2 + 2.0 * (sig_g / R) ** 2)
        sig_dy = np.sqrt((sig_gprime * gdy) ** 2 + 2.0 * (sig_g / R) ** 2)
        sig_dx = np.maximum(np.abs(np.nan_to_num(sig_dx, nan=0.0, posinf=0.0, neginf=0.0)), EPS)
        sig_dy = np.maximum(np.abs(np.nan_to_num(sig_dy, nan=0.0, posinf=0.0, neginf=0.0)), EPS)

        # Occlusion term (your formula), with safety guards
        sth, cph, sph = np.sin(th), np.cos(ph), np.sin(ph)
        eta = np.nan_to_num(cot_th - tan_angle * cph, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            Phi = 0.5 * (1.0 + sp.special.erf((eta + mu_dx) / (np.sqrt(2.0) * sig_dx)))
            term1 = Phi * (np.cos(th) + mu_dx * sth * cph)
            term2 = (sig_dx / np.sqrt(2.0 * np.pi)) * sth * cph * np.exp(-0.5 * ((eta + mu_dx) / sig_dx) ** 2)
            term3 = sth * sph * mu_dy
            Occlusion = np.nan_to_num(term1 + term2 + term3, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional shadow (function of γ only)
        if first:
            ang1 = np.full_like(gamma, th, dtype=float)
            ang2 = np.full_like(gamma, ph, dtype=float)
            try:
                Shadow = np.asarray(self.shadow_function(ang1, ang2, mu_g, sigma_coeff, mu_coeff, R, N_INT), float)
            except Exception:
                Shadow = np.ones_like(gamma, dtype=float)
            Shadow = np.clip(np.nan_to_num(Shadow, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        else:
            Shadow = 1.0

        # Unnormalized integrand (pointwise)
        integrand = p_gamma * p_gdx * p_gdy * Shadow
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Normalize over the implied 3D product grid (Riemann sum) ---
        dgamma = (ps.gamma_max_int - ps.gamma_min_int) / (N - 1.0)
        dgdx = (ps.gamma_max_int - ps.gamma_min_int) / (N - 1.0) * np.sqrt(2.0) / R
        dgdy = (ps.gamma_max_int - ps.gamma_min_int) / (N - 1.0) * np.sqrt(2.0) / R

        total = float(np.sum(integrand) * dgamma * dgdx * dgdy)
        if not np.isfinite(total) or total <= EPS:
            return np.zeros_like(integrand, dtype=float)

        return integrand / total

    def get_normals_pdf(
            self,
            points: np.ndarray | tuple[np.ndarray, np.ndarray],
            incident_angle_1: float,
            incident_angle_2: float,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
            first: bool = False,
    ) -> np.ndarray:
        """
        Joint normal-angle density p(θ_n, φ_n) given an incident direction,
        masked to include **only visible normals** satisfying n · v_i ≤ 0.

        Model (as in your original):
          - Integrates a latent mixture over (γ, γ̇x, γ̇y) to produce the slope
            likelihoods N(s_x; μ̇_x, σ̇_x) and N(s_y; μ̇_y, σ̇_y), with the
            Jacobian sec^4(θ_n) for (s_x, s_y) ↔ (θ_n, φ_n).

        Visibility rule:
          - Incident direction v_i has polar/azimuth (θ_i, φ_i).
          - Surface normal n(θ_n, φ_n) is visible here iff n · v_i ≤ 0.
          - This function zeros out p where visibility is false; normalization
            (if desired) should be done by the caller (e.g., in get_scattering_pdf).

        Parameters
        ----------
        points : (N,2) array or tuple of two (N,) arrays
            Normal angles (θ_n, φ_n) in radians.
        incident_angle_1, incident_angle_2 : float
            Incident polar/azimuth angles (radians).
        sigma_coeff, mu_coeff : np.ndarray, optional
            Coefficients for σ(γ) and μ(γ). Defaults from `self.poly_surface`.
        R : float, optional
            Autocorrelation length (>0). Default from `self.poly_surface.R`.
        N_INT : int, optional
            Integration resolution for γ and γ̇ grids. Default `self.N_INT`.
        first : bool, optional
            If True, includes the shadow term via `get_mixture_pdf`.

        Returns
        -------
        np.ndarray
            Nonnegative density values of shape (N,), with non-visible normals zeroed.
            (Not normalized; caller may renormalize over its domain.)
        """
        ps = self.poly_surface
        EPS = 1e-12

        # ---- unpack points -> (theta_n, phi_n) ----
        if isinstance(points, tuple):
            theta_n = np.asarray(points[0], float).ravel()
            phi_n = np.asarray(points[1], float).ravel()
            if theta_n.size != phi_n.size:
                raise ValueError("theta_n and phi_n must have the same length.")
        else:
            arr = np.asarray(points, float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("points must be (N,2) or a tuple of two (N,) arrays.")
            theta_n, phi_n = arr[:, 0].ravel(), arr[:, 1].ravel()

        Np = theta_n.size
        if Np == 0:
            return np.zeros(0, dtype=float)

        # Clamp θ away from ±π/2 to keep tan finite; sanitize
        delta = 1e-6
        lim = np.pi / 2 - delta
        theta_n = np.nan_to_num(theta_n, nan=0.0)
        theta_n = np.clip(theta_n, -lim, lim)
        phi_n = np.nan_to_num(phi_n, nan=0.0)

        # Incident angles (sanitized & clamped)
        ia1 = float(np.nan_to_num(incident_angle_1, nan=0.0))
        ia2 = float(np.nan_to_num(incident_angle_2, nan=0.0))
        ia1 = float(np.clip(ia1, -lim, lim))

        # ---- parameters & latent grids ----
        sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(ps.R if R is None else R)
        R = 1.0 if (not np.isfinite(R) or R <= 0) else R

        N = int(self.N_INT if N_INT is None else N_INT)
        N = max(N, 2)

        gmin = float(getattr(ps, "gamma_min_int", -8.0))
        gmax = float(getattr(ps, "gamma_max_int", 8.0))
        if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin == gmax:
            gmin, gmax = -8.0, 8.0

        gamma_array = np.linspace(gmin, gmax, N)
        gdx_array = np.linspace(gmin, gmax, N) * (np.sqrt(2.0) / R)
        gdy_array = np.linspace(gmin, gmax, N) * (np.sqrt(2.0) / R)
        dgamma = (gmax - gmin) / (N - 1.0)
        dgd = (gmax - gmin) / (N - 1.0) * (np.sqrt(2.0) / R)

        # 3D latent mesh
        G, Gx, Gy = np.meshgrid(gamma_array, gdx_array, gdy_array, indexing="ij")

        # ---- mixture PDF over latent grid ----
        latent_stacked = np.array([G.ravel(), Gx.ravel(), Gy.ravel()])  # (3, Nlat)
        mix = self.get_mixture_pdf(
            latent_stacked, ia1, ia2,
            sigma_coeff=sigma_coeff, mu_coeff=mu_coeff, R=R, N_INT=N, first=first
        )
        mix = np.nan_to_num(mix, nan=0.0, posinf=0.0, neginf=0.0).reshape(G.shape)  # (Ng,Ngdx,Ngdy)

        # ---- PolyGaussian fields on latent grid ----
        sigma_g = ps.get_function(G, sigma_coeff)
        mu_gprime = ps.get_function_derivative(G, mu_coeff)
        sigma_gprime = ps.get_function_derivative(G, sigma_coeff)

        sigma_g = np.maximum(np.abs(np.nan_to_num(sigma_g, nan=0.0, posinf=0.0, neginf=0.0)), EPS)
        mu_gprime = np.nan_to_num(mu_gprime, nan=0.0, posinf=0.0, neginf=0.0)
        sigma_gprime = np.nan_to_num(sigma_gprime, nan=0.0, posinf=0.0, neginf=0.0)

        # Means/SDs of slope conditionals
        mu_dx = mu_gprime * Gx - np.tan(ps.angle)
        mu_dy = mu_gprime * Gy
        sigma_dx = np.sqrt((sigma_gprime * Gx) ** 2 + 2.0 * (sigma_g / R) ** 2)
        sigma_dy = np.sqrt((sigma_gprime * Gy) ** 2 + 2.0 * (sigma_g / R) ** 2)

        sigma_dx = np.maximum(np.abs(np.nan_to_num(sigma_dx, nan=0.0, posinf=0.0, neginf=0.0)), EPS)
        sigma_dy = np.maximum(np.abs(np.nan_to_num(sigma_dy, nan=0.0, posinf=0.0, neginf=0.0)), EPS)

        # ---- normals -> slopes with Jacobian (broadcast last dim = Np) ----
        t = np.tan(theta_n)  # (Np,)
        sx = (-t * np.cos(phi_n))[None, None, None, :]  # (1,1,1,Np)
        sy = (-t * np.sin(phi_n))[None, None, None, :]
        jac = ((1.0 + t * t) ** 1.5)[None, None, None, :]  # sec^3(theta_n)

        mu_dx = mu_dx[..., None]
        mu_dy = mu_dy[..., None]
        sd_x = sigma_dx[..., None]
        sd_y = sigma_dy[..., None]
        mix = mix[..., None]  # (Ng,Ngdx,Ngdy,1)

        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            gx = inv_sqrt_2pi / sd_x * np.exp(-0.5 * ((sx - mu_dx) / sd_x) ** 2)
            gy = inv_sqrt_2pi / sd_y * np.exp(-0.5 * ((sy - mu_dy) / sd_y) ** 2)
            integrand = mix * gx * gy * jac

        pdf = np.sum(integrand, axis=(0, 1, 2)) * dgamma * dgd * dgd  # (Np,)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdf[pdf < 0] = 0.0

        # ---- Visibility mask: keep only normals with n · v_i ≤ 0 ----
        # n = (sinθn cosφn, sinθn sinφn, cosθn)
        # v_i = (sinθi cosφi, sinθi sinφi, cosθi)
        s_thn, c_thn = np.sin(theta_n), np.cos(theta_n)
        s_thi, c_thi = np.sin(ia1), np.cos(ia1)
        n_dot_vi = s_thn * s_thi * np.cos(phi_n - ia2) - c_thn * c_thi  # (Np,)
        vis_mask = (n_dot_vi < 0.0)
        pdf = np.where(vis_mask, pdf, 0.0)

        return pdf

    def get_scattering_pdf_diffuse(
            self,
            points: np.ndarray | tuple[np.ndarray, np.ndarray],
            incident_angle_1: float,
            incident_angle_2: float,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
            first: bool = False,
    ) -> np.ndarray:
        """
        Scattering PDF over reflected directions via integration over surface normals.

        Conventions
        -----------
        - Reflected directions are pairs (theta_r, phi_r):
            * pass a tuple of two (N,) arrays (theta_r, phi_r), OR
            * a 2×N array, OR an N×2 array.
          The function returns an array of shape (N,) with one value per pair.

        - Internally integrates over surface normals (theta_n ∈ [0, π/2], phi_n ∈ [0, 2π)):
            scatter(θr, φr) = ∬ cos(θ_local) · p_n(θn, φn | incidence)
                               · scaling(θn, φn; θi, φi) · sin(θn) dθn dφn
          where cos(θ_local) = sinθr sinθn cos(φr − φn) + cosθr cosθn, clamped ≥ 0.

        Normalization behavior
        ----------------------
        - If the provided (theta_r, phi_r) form a rectilinear grid (uniform in both
          directions), the output is **normalized** so that:
              ∬ PDF(θr, φr) · sin(θr) dθr dφr = 1
          over the supplied grid domain.
        - If the inputs are not a grid (e.g., a 1D slice), the function returns the
          unnormalized density for those directions (you can normalize that subset
          externally if desired).

        Safety
        ------
        - All inputs are sanitized (NaN/Inf → finite).
        - Angles near tan/cot singularities are guarded with tiny epsilons.
        - Negative results are clamped to zero before normalization.

        Parameters
        ----------
        points : array-like
            Reflected directions as described above.
        incident_angle_1 : float
            Incident polar angle (radians).
        incident_angle_2 : float
            Incident azimuth (radians).
        sigma_coeff, mu_coeff : np.ndarray, optional
            PolyGaussian coefficient vectors; defaults are taken from `self.poly_surface`.
        R : float, optional
            Autocorrelation length; defaults to `self.poly_surface.R` (clamped > 0).
        N_INT : int, optional
            Quadrature resolution used internally for normal/latent integrals.
        first : bool, default False
            If True, includes the shadowing term in `get_normals_pdf`.

        Returns
        -------
        np.ndarray
            Scattering PDF evaluated at the provided reflected directions, shape (N,).
        """
        ps = self.poly_surface
        EPS = 1e-12

        # --- parse points → (theta_r, phi_r) length-N vectors ---
        if isinstance(points, (tuple, list)) and len(points) == 2:
            theta_r, phi_r = (np.asarray(a, float).ravel() for a in points)
        else:
            P = np.asarray(points, float)
            if P.ndim == 2 and P.shape[0] == 2:
                theta_r, phi_r = P[0].ravel(), P[1].ravel()
            elif P.ndim == 2 and P.shape[1] == 2:
                theta_r, phi_r = P[:, 0].ravel(), P[:, 1].ravel()
            else:
                raise ValueError(
                    "`points` must be (theta_r, phi_r) as (2,N), (N,2), or a tuple of two (N,) arrays."
                )

        Np = theta_r.size
        if Np == 0 or phi_r.size != Np:
            return np.zeros(Np, dtype=float)

        theta_r = np.nan_to_num(theta_r, nan=0.0, posinf=0.0, neginf=0.0)
        phi_r = np.nan_to_num(phi_r, nan=0.0, posinf=0.0, neginf=0.0)

        # parameters / coeffs
        sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(ps.R if R is None else R)
        if not np.isfinite(R) or R <= 0:
            R = 1.0

        # clamp incident polar away from tan singularity
        def _clamp(a: float) -> float:
            try:
                a = float(a)
            except Exception:
                a = 0.0
            lim = np.pi / 2 - 1e-6
            return float(np.clip(a, -lim, lim))

        theta_i = _clamp(incident_angle_1)
        phi_i = float(np.nan_to_num(incident_angle_2, nan=0.0, posinf=0.0, neginf=0.0))

        # normal-angle quadrature
        N = int(self.N_INT if N_INT is None else N_INT)
        N = max(N, 2)

        theta_n = np.linspace(0.0, np.pi / 2.0, N)  # [0, π/2]
        phi_n = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)  # [0, 2π)
        dtheta = (theta_n[-1] - theta_n[0]) / max(N - 1, 1)
        dphi = (2.0 * np.pi) / N

        # normals PDF on the (theta_n, phi_n) grid
        thn_grid, phn_grid = np.meshgrid(theta_n, phi_n, indexing="ij")  # (N, N)
        normals_pdf = self.get_normals_pdf(
            (thn_grid.ravel(), phn_grid.ravel()),
            theta_i, phi_i, sigma_coeff, mu_coeff, R, N, first
        ).reshape(thn_grid.shape)

        normals_pdf = np.maximum(np.nan_to_num(normals_pdf, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # normalize normals PDF over hemisphere with sin(theta_n) Jacobian
        weight_n = np.sin(thn_grid)
        total_n = float(np.sum(normals_pdf * weight_n) * dtheta * dphi)
        if np.isfinite(total_n) and total_n > EPS:
            normals_pdf /= total_n
        else:
            return np.zeros(Np, dtype=float)

        # precompute normal trig arrays (expand to broadcast with reflected directions)
        sin_thn = np.sin(thn_grid)[:, :, None]  # (N, N, 1)
        cos_thn = np.cos(thn_grid)[:, :, None]  # (N, N, 1)
        phi_n_b = phn_grid[:, :, None]  # (N, N, 1)

        # reflected trig (as (1,1,Np))
        sin_thr = np.sin(theta_r)[None, None, :]
        cos_thr = np.cos(theta_r)[None, None, :]
        phi_r_b = phi_r[None, None, :]

        # cos of local angle between normal and reflected direction, clamped ≥ 0
        cos_local = sin_thr * sin_thn * np.cos(phi_r_b - phi_n_b) + cos_thr * cos_thn
        cos_local = np.maximum(np.nan_to_num(cos_local, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # scaling term: cos θi − tan θn · sin θi · cos(φn − φi)
        sin_thi, cos_thi = np.sin(theta_i), np.cos(theta_i)
        cos_phi_diff = np.cos(phn_grid - phi_i)[:, :, None]
        cos_thn_safe = np.where(cos_thn > EPS, cos_thn, EPS)
        tan_thn = sin_thn / cos_thn_safe
        scaling = cos_thi - tan_thn * sin_thi * cos_phi_diff
        scaling = np.nan_to_num(scaling, nan=0.0, posinf=0.0, neginf=0.0)

        # integrate over normals (θ_n, φ_n)
        normals_pdf_b = normals_pdf[:, :, None]
        weight_b = weight_n[:, :, None]
        integrand = cos_local * normals_pdf_b * scaling * weight_b
        scatter = np.sum(integrand, axis=(0, 1)) * dtheta * dphi  # (Np,)
        scatter = np.maximum(np.nan_to_num(scatter, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # --- Attempt global normalization over (theta_r, phi_r) if inputs form a grid ---
        def _normalize_over_reflected_grid(theta, phi, vals):
            # try to detect a rectilinear grid
            th_u = np.unique(np.round(theta, 12))
            ph_u = np.unique(np.round(phi, 12))
            n_th, n_ph = th_u.size, ph_u.size
            if n_th * n_ph != vals.size or n_th < 2 or n_ph < 2:
                return vals  # not a grid → leave as-is

            # estimate steps (robust to roundoff)
            dth = float(np.median(np.diff(th_u))) if n_th > 1 else np.pi
            dph_raw = np.median(np.diff(np.sort(ph_u))) if n_ph > 1 else 2.0 * np.pi
            # if azimuth spans full circle, prefer 2π/n_ph
            span_ph = float(ph_u.max() - ph_u.min())
            dph = 2.0 * np.pi / n_ph if span_ph > np.pi else float(dph_raw)

            # assemble grid by sorting pairs (θ, φ) to (i,j) bins
            th_map = {v: i for i, v in enumerate(th_u)}
            ph_map = {v: j for j, v in enumerate(ph_u)}
            # map with rounding to avoid FP mismatches
            th_idx = np.vectorize(lambda x: th_map.get(np.round(x, 12), -1))(theta)
            ph_idx = np.vectorize(lambda x: ph_map.get(np.round(x, 12), -1))(phi)
            ok = (th_idx >= 0) & (ph_idx >= 0)

            grid = np.zeros((n_th, n_ph), dtype=float)
            counts = np.zeros_like(grid)
            grid[th_idx[ok], ph_idx[ok]] += vals[ok]
            counts[th_idx[ok], ph_idx[ok]] += 1.0
            with np.errstate(invalid="ignore"):
                grid = np.where(counts > 0, grid / counts, 0.0)

            # normalize so ∬ grid(θ,φ) sinθ dθ dφ = 1 over provided grid
            sin_th = np.sin(th_u)[:, None]
            total = float(np.sum(grid * sin_th) * dth * dph)
            if np.isfinite(total) and total > EPS:
                grid /= total
            # return flattened in the original (θ, φ) order
            out = np.zeros_like(vals)
            out[ok] = grid[th_idx[ok], ph_idx[ok]]
            return np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        scatter = _normalize_over_reflected_grid(theta_r, phi_r, scatter)
        return scatter

    def get_scattering_pdf_specular(
            self,
            points: np.ndarray | tuple[np.ndarray, np.ndarray],
            incident_angle_1: float,
            incident_angle_2: float,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
            first: bool = False,
    ) -> np.ndarray:
        """
        Scattering PDF for *perfectly specular* local reflection (Dirac at the specular normal).

        For each reflected direction (θr, φr) there is a unique surface normal
            n̂ = (î + r̂) / |î + r̂|
        and the scattering PDF evaluates to:
            P_r(θr, φr) = P_n(θn, φn) / (4 * |î·n̂|)

        - No integration over (θn, φn): we directly evaluate the normals PDF at the
          specular normal and apply the Jacobian.
        - If (θr, φr) form a rectilinear grid, the output is normalized so that:
              ∬ P_r(θr,φr) sinθr dθr dφr = 1
          over the provided grid domain. Otherwise, raw analytical values are returned.

        Parameters are the same as before; N_INT is ignored here but accepted for API parity.
        """
        import numpy as np
        ps = self.poly_surface
        EPS = 1e-12

        # --- parse reflected directions → (theta_r, phi_r) ---
        if isinstance(points, (tuple, list)) and len(points) == 2:
            theta_r, phi_r = (np.asarray(a, float).ravel() for a in points)
        else:
            P = np.asarray(points, float)
            if P.ndim == 2 and P.shape[0] == 2:
                theta_r, phi_r = P[0].ravel(), P[1].ravel()
            elif P.ndim == 2 and P.shape[1] == 2:
                theta_r, phi_r = P[:, 0].ravel(), P[:, 1].ravel()
            else:
                raise ValueError("`points` must be (theta_r, phi_r) as (2,N), (N,2), or a tuple of two (N,) arrays.")

        Np = theta_r.size
        if Np == 0 or phi_r.size != Np:
            return np.zeros(Np, dtype=float)

        theta_r = np.nan_to_num(theta_r, nan=0.0, posinf=0.0, neginf=0.0)
        phi_r = np.nan_to_num(phi_r, nan=0.0, posinf=0.0, neginf=0.0)

        # --- parameters / coeffs (pass-through to get_normals_pdf) ---
        sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        Rv = float(ps.R if R is None else R)
        if not np.isfinite(Rv) or Rv <= 0.0:
            Rv = 1.0

        # --- incident direction (clamp polar slightly below π/2 for safety) ---
        def _clamp(a: float) -> float:
            try:
                a = float(a)
            except Exception:
                a = 0.0
            lim = np.pi / 2 - 1e-6
            return float(np.clip(a, 0.0, lim))

        theta_i = _clamp(incident_angle_1)
        phi_i = float(np.nan_to_num(incident_angle_2, nan=0.0, posinf=0.0, neginf=0.0))

        si, ci = np.sin(theta_i), np.cos(theta_i)
        i_hat = np.array([si * np.cos(phi_i), si * np.sin(phi_i), ci], dtype=float)

        # --- reflected unit vectors ---
        sr, cr = np.sin(theta_r), np.cos(theta_r)
        r_hat = np.stack([sr * np.cos(phi_r), sr * np.sin(phi_r), cr], axis=1)  # (N,3)

        # --- specular normal: n̂ = (î + r̂)/||…||; invalid if î ≈ −r̂ (n_norm→0) ---
        n_sum = i_hat[None, :] + r_hat  # (N,3)
        n_norm = np.linalg.norm(n_sum, axis=1)  # (N,)
        valid = n_norm > EPS
        n_hat = np.zeros_like(n_sum)
        n_hat[valid] = n_sum[valid] / n_norm[valid, None]

        # --- spherical angles of n̂ (for evaluating P_n) ---
        nz = np.clip(n_hat[:, 2], -1.0, 1.0)
        theta_n = np.arccos(nz)
        phi_n = np.mod(np.arctan2(n_hat[:, 1], n_hat[:, 0]), 2.0 * np.pi)

        # --- evaluate surface-normal PDF at (θn, φn) ---
        Pn = self.get_normals_pdf(
            (theta_n, phi_n),
            theta_i, phi_i,
            sigma_coeff, mu_coeff,
            Rv, getattr(self, "N_INT", 64),  # N_INT unused here; keep signature compatibility
            first
        )
        Pn = np.maximum(np.nan_to_num(Pn, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # --- Jacobian: dΩ_r / dΩ_n = 4 |î·n̂|  ⇒  P_r = P_n / (4 |î·n̂|)
        dot_in = np.einsum("i,ni->n", i_hat, n_hat)
        dot_in = np.clip(np.abs(dot_in), 1e-12, 1.0)
        Pr = Pn / (4.0 * dot_in)
        Pr[~valid] = 0.0
        Pr = np.maximum(np.nan_to_num(Pr, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # --- If (θr, φr) is a rectilinear grid, normalize over that grid domain ---
        th_u = np.unique(np.round(theta_r, 12))
        ph_u = np.unique(np.round(phi_r, 12))
        n_th, n_ph = th_u.size, ph_u.size

        if n_th >= 2 and n_ph >= 2 and n_th * n_ph == Np:
            # robust step sizes
            dth = float(np.median(np.diff(th_u)))
            span_ph = float(ph_u.max() - ph_u.min())
            dph = 2.0 * np.pi / n_ph if span_ph > np.pi else float(np.median(np.diff(ph_u)))

            # build grid in the original (θr,φr) order
            th_map = {v: i for i, v in enumerate(np.round(th_u, 12))}
            ph_map = {v: j for j, v in enumerate(np.round(ph_u, 12))}
            th_idx = np.vectorize(lambda x: th_map.get(np.round(x, 12), -1))(theta_r)
            ph_idx = np.vectorize(lambda x: ph_map.get(np.round(x, 12), -1))(phi_r)
            ok = (th_idx >= 0) & (ph_idx >= 0)

            grid = np.zeros((n_th, n_ph), dtype=float)
            cnts = np.zeros_like(grid)
            grid[th_idx[ok], ph_idx[ok]] += Pr[ok]
            cnts[th_idx[ok], ph_idx[ok]] += 1.0
            with np.errstate(invalid="ignore"):
                grid = np.where(cnts > 0, grid / cnts, 0.0)

            # normalize so ∬ P_r sinθ dθ dφ = 1 on the provided grid
            sin_th = np.sin(th_u)[:, None]
            total = float(np.sum(grid * sin_th) * dth * dph)
            if np.isfinite(total) and total > EPS:
                grid /= total

            out = np.zeros_like(Pr)
            out[ok] = grid[th_idx[ok], ph_idx[ok]]
            return np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        # otherwise return analytical values for the queried directions
        return Pr

    def get_scattering_pdf(
            self,
            points: np.ndarray | tuple[np.ndarray, np.ndarray],
            incident_angle_1: float,
            incident_angle_2: float,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
            first: bool = False,
            accommodation_coefficient: float = 1.0,
    ) -> np.ndarray:
        # accommodation_coefficient = 1.0
        scatter_specular = self.get_scattering_pdf_specular(points, incident_angle_1, incident_angle_2, sigma_coeff, mu_coeff, R, N_INT, first)
        scatter_diffuse = self.get_scattering_pdf_diffuse(points, incident_angle_1, incident_angle_2, sigma_coeff, mu_coeff, R, N_INT, first)

        return (1.0 - accommodation_coefficient) * scatter_specular + accommodation_coefficient * scatter_diffuse

    def get_multi_scattering_pdf(
            self,
            points: np.ndarray | tuple[np.ndarray, np.ndarray],
            incident_angle_pdf: np.ndarray | None,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
            accommodation_coefficient = 1.0,
    ) -> np.ndarray:
        """
        Mixture scattering PDF at reflected angles by integrating single-bounce
        PDFs over an incident-angle distribution on the sphere.

        Parameters
        ----------
        points : ((N,), (N,)) or (2,N) or (N,2)
            Reflected directions (theta_r, phi_r) in radians.
        incident_angle_pdf : (Nθ,Nφ) or None
            Incident PDF p(θ,φ) (per steradian). If None → uniform.
            It is normalized internally (via solid-angle weights).
        sigma_coeff, mu_coeff, R, N_INT
            Optional surface parameters / quadrature resolution (fallbacks to
            self.poly_surface.* and self.N_INT).

        Returns
        -------
        (N,) array
            Scattering PDF evaluated at `points`.
        """

        ps = self.poly_surface
        sigma = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(ps.R if R is None else R)
        N = int(self.N_INT if N_INT is None else N_INT)
        if not (np.isfinite(R) and R > 0):  # safe fallback
            R = 1.0
        N = max(N, 2)

        # ---- parse reflected points → 1D arrays (θ_r, φ_r) ----
        if isinstance(points, (tuple, list)) and len(points) == 2:
            th_r, ph_r = (np.asarray(a, float).ravel() for a in points)
        else:
            P = np.asarray(points, float)
            if P.ndim == 2 and P.shape[0] == 2:
                th_r, ph_r = P[0].ravel(), P[1].ravel()
            elif P.ndim == 2 and P.shape[1] == 2:
                th_r, ph_r = P[:, 0].ravel(), P[:, 1].ravel()
            else:
                raise ValueError("`points` must be (θ,φ) as (2,N), (N,2), or a tuple of two (N,) arrays.")
        Np = th_r.size
        if Np == 0 or ph_r.size != Np:
            return np.zeros(0, dtype=float)

        # ---- incident grid + weights (solid angle) ----
        theta_i = np.linspace(0.0, np.pi, N)
        phi_i = np.linspace(0.0, 2.0 * np.pi, N)
        dtheta = (theta_i[-1] - theta_i[0]) / (N - 1)
        dphi = (phi_i[-1] - phi_i[0]) / (N - 1)
        TH, PH = np.meshgrid(theta_i, phi_i, indexing="ij")  # (N,N)
        w = np.sin(TH) * dtheta * dphi
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        # incident PDF (per steradian) → discrete weights normalized to 1
        if incident_angle_pdf is None:
            inc_pdf = np.full((N, N), 1.0 / (4.0 * np.pi), float)
        else:
            arr = np.asarray(incident_angle_pdf, float)
            if arr.shape == (N, N):
                inc_pdf = arr
            elif arr.size == N * N:
                inc_pdf = arr.reshape(N, N)
            else:
                raise ValueError("incident_angle_pdf must have shape (N,N) or size N*N.")
        inc_pdf = np.nan_to_num(inc_pdf, nan=0.0, posinf=0.0, neginf=0.0)

        weights = inc_pdf * w
        Z = float(np.sum(weights))
        if not np.isfinite(Z) or Z <= 0.0:
            return np.zeros(Np, dtype=float)
        weights /= Z  # now ∑_grid weights ≈ 1

        # ---- integrate single-bounce scattering over incident distribution ----
        out = np.zeros(Np, dtype=float)
        for i in range(N):
            for j in range(N):
                if weights[i, j] == 0.0:
                    continue
                vals = self.get_scattering_pdf((th_r, ph_r), float(TH[i, j]), float(PH[i, j]),
                                               sigma, mu, R, N, False, accommodation_coefficient)
                v = np.nan_to_num(np.asarray(vals, float).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
                if v.size != Np:
                    raise ValueError("get_scattering_pdf must return shape (N_points,) for given `points`.")
                out += weights[i, j] * v

        return out

    def get_trapping_probability(
            self,
            incident_angle_pdf: np.ndarray | None = None,
            incident_angle_1: float | None = None,
            incident_angle_2: float | None = None,
            sigma_coeff: np.ndarray | None = None,
            mu_coeff: np.ndarray | None = None,
            R: float | None = None,
            N_INT: int | None = None,
            first: bool = False,
            accommodation_coefficient: float = 1.0,
    ) -> float:
        """
        Compute the probability that a ray becomes *trapped* (i.e., not escaping after
        a reflection) by integrating scattering over reflected directions and, optionally,
        over a distribution of incident directions.

        Two modes
        ----------
        - `first=False` (default): Integrate over an **incident-angle PDF** on a regular
          (θ_i ∈ [0,π], φ_i ∈ [0,2π)) grid of size N×N. In this mode you should pass
          `incident_angle_pdf` with total probability 1 (w.r.t. solid angle).
          If omitted, a uniform distribution over the sphere (1 / 4π) is assumed.
        - `first=True`: Use a **single** incident direction given by
          (`incident_angle_1`, `incident_angle_2`), and integrate only over reflected
          angles.

        Integrand (schematic)
        ---------------------
            P_trap = ∬_{inc} p_i(θ_i,φ_i) ∬_{ref} [1 - Shadow(θ_r,φ_r)]
                     · ScatterPDF((θ_r,φ_r) | θ_i,φ_i) · dΩ_r · dΩ_i

        where dΩ = sin(θ) dθ dφ, and `Shadow(θ_r,φ_r)` is the angle-only shadow term.

        Parameters
        ----------
        incident_angle_pdf : np.ndarray, optional
            (N×N) array for p_i(θ_i,φ_i) on a uniform grid, or a flat array with N*N
            elements that will be reshaped to (N,N). If None (and `first=False`),
            a uniform 1/(4π) PDF is used and normalized.
        incident_angle_1 : float, optional
            Incident polar angle θ_i in radians (only used if `first=True`).
        incident_angle_2 : float, optional
            Incident azimuth φ_i in radians (only used if `first=True`).
        sigma_coeff, mu_coeff : np.ndarray, optional
            PolyGaussian σ(γ) and μ(γ) coefficient vectors. Defaults to `self.poly_surface`.
        R : float, optional
            Autocorrelation length. Defaults to `self.poly_surface.R`. Must be > 0.
        N_INT : int, optional
            Angular grid resolution N used for the (θ,φ) meshes. Defaults to `self.N_INT`.
        first : bool, default False
            If True, compute trapping probability for a single incident direction;
            otherwise integrate over `incident_angle_pdf`.

        Returns
        -------
        float
            Trapping probability in [0, 1]. Returns 0.0 on invalid inputs.

        Notes
        -----
        - Uses uniform Riemann quadrature on regular angular grids.
        - All intermediate arrays are sanitized to avoid NaN/Inf propagation.
        - The shadow term depends **only** on reflected angles and is reused across
          all incident directions for efficiency.
        """
        ps = self.poly_surface
        EPS = 1e-12

        # --- Parameters & safety ---
        sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
        mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
        R = float(ps.R if R is None else R)
        if not np.isfinite(R) or R <= 0:
            R = 1.0

        N = int(self.N_INT if N_INT is None else N_INT)
        N = max(N, 2)

        # Uniform angular grids
        theta = np.linspace(0.0, np.pi, N)
        phi = np.linspace(0.0, 2.0 * np.pi, N)
        dtheta = (theta[-1] - theta[0]) / (N - 1)
        dphi = (phi[-1] - phi[0]) / (N - 1)

        # Reflected grid (used in both modes)
        th_r, ph_r = np.meshgrid(theta, phi, indexing="ij")
        points_ref = np.array([th_r.ravel(), ph_r.ravel()])

        # Shadow(θ_r, φ_r) depends only on reflected angles; cache once
        shadow = self.shadow_angle_function(points_ref[0], points_ref[1], sigma_coeff, mu_coeff, R, N)
        shadow = np.clip(np.nan_to_num(np.asarray(shadow, float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        shadow = shadow.reshape(N, N)

        # Reflected solid-angle weights
        w_ref = np.sin(th_r) * dtheta * dphi
        w_ref = np.nan_to_num(w_ref, nan=0.0, posinf=0.0, neginf=0.0)
        # Multiplicative kernel for inner integral over reflected angles
        K_ref = (1.0 - shadow) * w_ref

        if not first:
            # ---- Integrate over *incident-angle PDF* ----
            # Incident grid & weights
            th_i, ph_i = np.meshgrid(theta, phi, indexing="ij")

            # Incident PDF: shape (N,N). If None, uniform over sphere (1/4π)
            if incident_angle_pdf is None:
                inc_pdf = np.full((N, N), 1.0 / (4.0 * np.pi), dtype=float)
            else:
                arr = np.asarray(incident_angle_pdf, float)
                if arr.size == N * N:
                    inc_pdf = arr.reshape(N, N)
                else:
                    raise ValueError("incident_angle_pdf must have size N*N or shape (N,N).")

            inc_pdf = np.nan_to_num(inc_pdf, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize incident PDF (w.r.t. solid angle) for safety
            w_inc = np.sin(th_i) * dtheta * dphi
            w_inc = np.nan_to_num(w_inc, nan=0.0, posinf=0.0, neginf=0.0)
            Z = float(np.sum(inc_pdf * w_inc))
            if not np.isfinite(Z) or Z <= EPS:
                return 0.0
            inc_pdf /= Z

            # Accumulate trapping probability:
            # For each (θ_i, φ_i), integrate scattering over reflected angles with kernel K_ref
            P_trap = 0.0
            for i in range(N):
                # (N,N) scattering for the whole reflected grid given this incident row
                for j in range(N):
                    th_i_ = float(th_i[i, j])
                    ph_i_ = float(ph_i[i, j])
                    scat = self.get_scattering_pdf(points_ref, th_i_, ph_i_, sigma_coeff, mu_coeff, R, N, False, accommodation_coefficient)
                    scat = np.nan_to_num(np.asarray(scat, float), nan=0.0, posinf=0.0, neginf=0.0).reshape(N, N)

                    # Inner integral over reflected angles
                    inner = float(np.sum(scat * K_ref))
                    if np.isfinite(inner) and inner >= 0.0:
                        P_trap += inc_pdf[i, j] * inner * np.sin(th_i_) * dtheta * dphi

            return float(np.clip(P_trap, 0.0, 1.0))

        # ---- Single-incident mode (first=True) ----
        # Sanitize incident angles
        th_i = float(np.nan_to_num(incident_angle_1 if incident_angle_1 is not None else 0.0))
        ph_i = float(np.nan_to_num(incident_angle_2 if incident_angle_2 is not None else 0.0))

        # Scattering for this single incident direction
        scat = self.get_scattering_pdf(points_ref, th_i, ph_i, sigma_coeff, mu_coeff, R, N, True, accommodation_coefficient)
        scat = np.nan_to_num(np.asarray(scat, float), nan=0.0, posinf=0.0, neginf=0.0).reshape(N, N)

        # Integrate over reflected directions only
        P_trap = float(np.sum(scat * K_ref))
        return float(np.clip(P_trap, 0.0, 1.0))

    # def get_angular_flux(
    #         self,
    #         angle_1: np.ndarray,
    #         angle_2: np.ndarray,
    #         height: np.ndarray,
    #         sigma_coeff: np.ndarray | None = None,
    #         mu_coeff: np.ndarray | None = None,
    #         R: float | None = None,
    #         N_INT: int | None = None,
    # ) -> np.ndarray:
    #     """
    #     Compute the angular flux for rays arriving at (angle_1, angle_2) from a given height.
    #
    #     Model outline (ASCII notation)
    #     ------------------------------
    #     Let g ~ N(0,1) and gdot ~ N(0, 2/R^2). From the PolyGaussian surface we have
    #     functions mu(g), mu'(g), sigma(g), sigma'(g). For each query tuple
    #     (theta1, theta2, h), define
    #
    #         eta(theta1, theta2) = cot(theta1) + tan(surface_angle) * cos(theta2)
    #
    #     and the auxiliary standardized variable
    #
    #         z = (eta - mu'(g)*gdot) / sqrt( (sigma'(g)*gdot)^2 + 2*(sigma(g)/R)^2 )
    #
    #     The quantity Delta(theta1, theta2) is obtained by integrating an analytic
    #     kernel over (g, gdot). Given the height CDF F(h) of the PolyGaussian surface,
    #     the flux is
    #
    #         if eta > 0:  flux = [ F^Delta * (1 - F^(1-Delta)) ] / [ (1-Delta) * eta ]
    #         else:        flux = - F / [ (1-Delta) * eta ]
    #
    #     with safe guards for divisions and exponents.
    #
    #     Parameters
    #     ----------
    #     angle_1, angle_2, height : np.ndarray
    #         1D arrays (same length) of polar angle, azimuthal angle (radians),
    #         and height. Non-finite inputs are sanitized to zero.
    #     sigma_coeff, mu_coeff : np.ndarray, optional
    #         PolyGaussian coefficient vectors. Defaults to self.poly_surface values.
    #     R : float, optional
    #         Autocorrelation length (>0). Defaults to self.poly_surface.R (clamped to >0).
    #     N_INT : int, optional
    #         Quadrature resolution for g and gdot (>=2). Defaults to self.N_INT.
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         Flux values, shape (N,), finite and non-negative.
    #     """
    #     ps = self.poly_surface
    #     EPS = 1e-12
    #
    #     # --- inputs: 1D, same size ---
    #     a1 = np.asarray(angle_1, float).ravel()
    #     a2 = np.asarray(angle_2, float).ravel()
    #     h = np.asarray(height, float).ravel()
    #     if not (a1.ndim == a2.ndim == h.ndim == 1 and a1.size == a2.size == h.size and a1.size > 0):
    #         raise ValueError("angle_1, angle_2, and height must be 1D arrays of equal non-zero length.")
    #     a1 = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)
    #     a2 = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)
    #     h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #     # --- parameters / coefficients ---
    #     sigma_coeff = ps.sigma_coeff if sigma_coeff is None else np.asarray(sigma_coeff, float)
    #     mu_coeff = ps.mu_coeff if mu_coeff is None else np.asarray(mu_coeff, float)
    #     R = float(ps.R if R is None else R)
    #     R = 1.0 if (not np.isfinite(R) or R <= 0.0) else R
    #
    #     N = int(self.N_INT if N_INT is None else N_INT)
    #     N = max(N, 2)
    #
    #     # --- height CDF F(h) in [0,1] ---
    #     try:
    #         F = ps.get_height_cdf(h, sigma_coeff, mu_coeff)
    #     except TypeError:
    #         F = ps.get_height_cdf(h)
    #     F = np.clip(np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    #
    #     # --- quadrature grids for g and gdot ---
    #     gmin = float(getattr(ps, "gamma_min_int", -8.0))
    #     gmax = float(getattr(ps, "gamma_max_int", 8.0))
    #     if not np.isfinite(gmin) or not np.isfinite(gmax) or gmin >= gmax:
    #         gmin, gmax = -8.0, 8.0
    #
    #     g = np.linspace(gmin, gmax, N)  # (G,)
    #     gdot = np.linspace(gmin * np.sqrt(2) / R, gmax * np.sqrt(2) / R, N)  # (D,)
    #
    #     inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    #     p_g = inv_sqrt_2pi * np.exp(-0.5 * g ** 2)  # (G,)
    #     p_gd = (R / np.sqrt(4.0 * np.pi)) * np.exp(-0.25 * (R * gdot) ** 2)  # (D,)
    #
    #     # PolyGaussian primitives
    #     mu_gprime = ps.get_function_derivative(g, mu_coeff)  # (G,)
    #     sigma_g = ps.get_function(g, sigma_coeff)  # (G,)
    #     sigma_g = np.maximum(np.abs(np.nan_to_num(sigma_g, nan=0.0)), EPS)
    #     sigma_gpr = ps.get_function_derivative(g, sigma_coeff)  # (G,)
    #
    #     # --- broadcast to (A, G, D) ---
    #     A = a1.size
    #     a1_map, g_map, gd_map = np.meshgrid(a1, g, gdot, indexing="ij")
    #     a2_map = np.broadcast_to(a2[:, None, None], (A, N, N))
    #
    #     mu_gp = mu_gprime[None, :, None]
    #     sig_g = sigma_g[None, :, None]
    #     sig_gp = sigma_gpr[None, :, None]
    #     pG = p_g[None, :, None]
    #     pGD = p_gd[None, None, :]
    #
    #     # conditional std and mean terms
    #     surface_angle = float(getattr(ps, "angle", 0.0))
    #     tan_angle = np.tan(surface_angle) if np.isfinite(surface_angle) else 0.0
    #
    #     mu_dot = mu_gp * gd_map
    #     sigma_sq = (sig_gp * gd_map) ** 2 + 2.0 * (sig_g / R) ** 2
    #     sigma_dot = np.sqrt(np.maximum(np.nan_to_num(sigma_sq, nan=0.0), EPS))
    #
    #     # eta = cot(theta1) + tan(surface_angle) * cos(theta2)
    #     t1 = np.tan(a1_map)
    #     t1 = np.where(np.isfinite(t1), t1, 0.0)
    #     t1 = np.where(np.abs(t1) > EPS, t1, np.sign(t1) * EPS + (t1 == 0.0) * EPS)
    #     eta = (1.0 / t1) - tan_angle * np.cos(a2_map)
    #     eta = np.where(np.abs(eta) > EPS, eta, np.sign(eta) * EPS + (eta == 0.0) * EPS)
    #
    #     # integrand for Delta
    #     with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
    #         z = (eta - mu_dot) / sigma_dot
    #         term1 = sigma_dot * inv_sqrt_2pi * np.exp(-0.5 * z * z)
    #         term2 = 0.5 * (eta - mu_dot) * sp.special.erfc(z / np.sqrt(2.0))
    #         integr = (term1 - term2) / eta * pG * pGD
    #         integr = np.nan_to_num(integr, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #     # trapezoidal integration over gdot then g  -> Delta shape (A,)
    #     Delta = np.trapezoid(np.trapezoid(integr, gdot, axis=2), g, axis=1)
    #     Delta = np.nan_to_num(Delta, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #     # --- robust power terms without assuming 0<=Delta<1 ---
    #     EPS = 1e-12
    #     MAX_LOG = 700.0  # ~ ln(1e304), safe for double
    #
    #     F_clip = np.clip(F, EPS, 1.0 - EPS)
    #     logF = np.log(F_clip)  # < 0
    #
    #     eta_s = eta[:, 0, 0]
    #     eta_s = np.where(np.abs(eta_s) < EPS, np.sign(eta_s) * EPS + (eta_s == 0) * EPS, eta_s)
    #
    #     denom = 1.0 - Delta
    #     denom = np.where(np.abs(denom) < EPS, np.sign(denom) * EPS + (denom == 0) * EPS, denom)
    #
    #     # exp(Delta * logF)  and  exp((1-Delta) * logF) with clipped exponents
    #     e1 = np.clip(Delta * logF, -MAX_LOG, MAX_LOG)
    #     e2 = np.clip((1.0 - Delta) * logF, -MAX_LOG, MAX_LOG)
    #
    #     F_pow_D = np.exp(e1)  # == F_clip ** Delta
    #     one_minus_F_pow_1mD = -np.expm1(e2)  # == 1 - exp(e2) == 1 - F_clip**(1-Delta)
    #
    #     # piecewise in eta
    #     flux = np.empty_like(F_clip)
    #     pos = eta_s > 0.0
    #     flux[pos] = (F_pow_D[pos] * one_minus_F_pow_1mD[pos]) / (denom[pos] * eta_s[pos])
    #     flux[~pos] = -(F_clip[~pos]) / (denom[~pos] * eta_s[~pos])
    #
    #     # final cleanup
    #     flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
    #
    #     return np.clip(np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0), 0.0, np.inf)

    def get_velocity_pdf(
            self,
            points: np.ndarray,
            incident_velocity: float,
            accommodation_coefficient: float,
            surface_temperature: float,
            gas_molar_mass: float,
    ) -> np.ndarray:
        """
        Speed PDF for a gas after surface interaction under a simple
        Maxwellian–accommodation model.

        Model
        -----
        A post-collision scalar speed v_r > 0 is modeled with a flux-weighted
        Maxwellian in speed:

            p(v_r) = (μ_AO^2 / (2 R^2 T^2)) v_r^3 exp(- μ_AO v_r^2 / (2 R T)),

        which is normalized on [0, ∞) for a given effective temperature T.

        The effective temperature T mixes incident and surface temperatures via a
        (clamped) accommodation coefficient alpha ∈ [0, 1]:

            T_i = (M u^2) / (3 R)          (incident “temperature”)
            T   = (1 - alpha) T_i + alpha T_s,

        where:
          - R is the ideal gas constant,
          - T_s is the surface temperature,
          - M = μ_AO is the molar mass,
          - u is the incident speed.

        Parameters
        ----------
        points : np.ndarray
            Speeds v_r at which to evaluate the PDF (same units as incident_velocity
            before the internal scaling). Any shape; values < 0 are treated as
            zero probability. Returned array matches this shape.
        incident_velocity : float
            Magnitude of the incident speed u.
        accommodation_coefficient : float
            Accommodation coefficient alpha (clamped to [0, 1]).
        surface_temperature : float
            Surface temperature T_s in kelvin (clamped to ≥ 0).
        gas_molar_mass : float
            Molar mass μ_AO = M in kg/mol (must be > 0).

        Returns
        -------
        np.ndarray
            PDF values p(v_r) with the same shape as `points`. Non-negative and finite.
        """
        # --- constants & safety guards ---
        EPS = 1e-15
        R_gas = float(getattr(cst, "R", 8.314462618))

        # Internal unit scaling (keep as in your original code)
        points = points * 1e-6
        incident_velocity = incident_velocity * 1e-6

        # Clamp parameters to safe numeric ranges
        alpha = float(np.nan_to_num(accommodation_coefficient, nan=0.0,
                                    posinf=1.0, neginf=0.0))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        T_s = float(np.nan_to_num(surface_temperature, nan=0.0,
                                  posinf=0.0, neginf=0.0))
        T_s = max(T_s, 0.0)

        M = float(np.nan_to_num(gas_molar_mass, nan=0.0,
                                posinf=0.0, neginf=0.0))
        if not np.isfinite(M) or M <= 0.0:
            return np.zeros_like(np.asarray(points, dtype=float))

        u = float(np.nan_to_num(incident_velocity, nan=0.0,
                                posinf=0.0, neginf=0.0))
        u2 = u * u

        # --- effective temperature ---
        # T_i = M u^2 / (3 R), then mix with surface temperature
        T_i = (M * u2) / (3.0 * R_gas)
        T_i = max(T_i, 0.0)
        T = (1.0 - alpha) * T_i + alpha * T_s
        T = max(T, 0.0)

        if not np.isfinite(T) or T <= EPS:
            return np.zeros_like(np.asarray(points, dtype=float))

        # --- Maxwellian speed PDF, flux weighted ~ v_r^3 exp(- μ v_r^2 / (2 R T)) ---
        v = np.asarray(points, dtype=float)
        v_nonneg = np.clip(v, 0.0, None)

        # normalization constant C = μ^2 / (2 R^2 T^2)
        C = (M * M) / (2.0 * R_gas * R_gas * T * T)

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            pdf = C * (v_nonneg ** 3) * np.exp(- (M * v_nonneg ** 2) / (2.0 * R_gas * T))

        # Zero out negatives explicitly; apply Jacobian for the 1e-6 scaling
        pdf = np.where(v >= 0.0, pdf, 0.0) * 1e-6
        return np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)

# # --------------------
# # setup
# # --------------------
# test_surf = surf.PolyGaussian_Surface(
#     np.array([0.6, 0.6, 0.6, 0.6]),
#     np.array([-0.5, -0.5, 0.5, 0.5]),
#     R=0.2,
#     angle=np.pi / 2.5,
#     N_INT=45
# )
# scatterer = Scatterer(test_surf)
#
# theta_i1 = float(np.pi / 12)  # polar incidence
# theta_i2 = float(0.0)          # azimuth incidence
#
# # --- helpers: rotate around y-axis by +angle (to remove x-tilt) ---
# def _ry(x, z, ang):
#     """Rotate coordinates (x,z) about +y by angle `ang` (y unchanged)."""
#     c, s = np.cos(-ang), np.sin(-ang)
#     xr = x * c + z * s
#     zr = -x * s + z * c
#     return xr, zr
#
# def _wrap_pi(a):
#     """Wrap angle to [-pi, pi]."""
#     return (a + np.pi) % (2.0 * np.pi) - np.pi
#
# a_tilt = float(getattr(test_surf, "angle", 0.0))
#
# # --------------------
# # trapping probability tests
# # --------------------
# P_single = scatterer.get_trapping_probability(
#     incident_angle_1=theta_i1,
#     incident_angle_2=theta_i2,
#     N_INT=15,
#     first=True
# )
# print(f"[trapping] single-incident (θi={theta_i1:.3f}, φi={theta_i2:.3f}): {P_single:.6f}")
#
# P_uniform = scatterer.get_trapping_probability(
#     incident_angle_pdf=None,  # defaults to uniform inside the function
#     N_INT=15,
#     first=False
# )
# print(f"[trapping] uniform-incident PDF: {P_uniform:.6f}")
#
# # --------------------
# # 1) x–z plane polar slice
# # --------------------
# alpha = np.linspace(-np.pi, np.pi, 721)     # signed in-plane angle
# theta_r = np.abs(alpha)                      # polar (0..π)
# phi_r = np.where(alpha >= 0.0, 0.0, np.pi)   # azimuth: +x for α≥0, −x for α<0
#
# pdf_slice = scatterer.get_scattering_pdf((theta_r, phi_r), theta_i1, theta_i2, N_INT=25, first=True)
# pdf_slice = np.maximum(np.nan_to_num(pdf_slice, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
# r_slice_max = float(np.max(pdf_slice)) if np.max(pdf_slice) > 0 else 1.0
#
# # rotate the 2D angular coordinate so the surface looks horizontal
# alpha_rot = _wrap_pi(alpha - a_tilt)
#
# # incidence arrow (incoming, from -theta_i1 along the x–z slice)
# sgn = 1.0 if np.cos(theta_i2) >= 0 else -1.0
# alpha_inc = -sgn * float(theta_i1)
# alpha_inc_rot = _wrap_pi(alpha_inc - a_tilt)
# r_tail = 1.10 * r_slice_max
# r_head = 0.05 * r_slice_max
#
# # --------------------
# # 2) full 3D scattering lobe
# # --------------------
# Nr, Np = 120, 240
# theta_r1 = np.linspace(0.0, np.pi, Nr)        # polar
# theta_r2 = np.linspace(-np.pi, np.pi, Np)     # azimuth
# Th, Ph = np.meshgrid(theta_r1, theta_r2, indexing='ij')
#
# pdf_full = scatterer.get_scattering_pdf((Th.ravel(), Ph.ravel()), theta_i1, theta_i2, N_INT=25, first=True)
# pdf_full = np.maximum(np.nan_to_num(pdf_full, nan=0.0, posinf=0.0, neginf=0.0), 0.0).reshape(Th.shape)
#
# r = pdf_full / (np.max(pdf_full) if np.max(pdf_full) > 0 else 1.0)
# X = r * np.sin(Th) * np.cos(Ph)
# Y = r * np.sin(Th) * np.sin(Ph)
# Z = r * np.cos(Th)
#
# # rotate the lobe by +angle around y to flatten the surface
# Xr, Zr = _ry(X, Z, a_tilt)
#
# # incidence vector in Cartesian (unit length), then rotate it too
# ix = np.sin(-theta_i1) * np.cos(theta_i2)
# iy = np.sin(-theta_i1) * np.sin(theta_i2)
# iz = np.cos(-theta_i1)
# ixr, izr = _ry(ix, iz, a_tilt)  # y-component unchanged
#
# # --------------------
# # 3) generate a surface (shown at bottom) and rotate it
# # --------------------
# gen = test_surf.generate(sample_length=10.0, samples=150, iterations=50, verbose=False)
# Xs, Ys, Zs = gen.X, gen.Y, gen.Z
# Xsr, Zsr = _ry(Xs, Zs, a_tilt)  # rotate to remove tilt
#
# # --------------------
# # plot
# # --------------------
# from matplotlib.gridspec import GridSpec
#
# fig = plt.figure(figsize=(12, 10))
# gs = GridSpec(2, 2, height_ratios=[1.0, 1.2], figure=fig)
#
# # (a) polar slice – rotated angles so the plane is horizontal
# ax0 = fig.add_subplot(gs[0, 0], projection='polar')
# ax0.set_theta_zero_location('N')  # 0 at +z
# ax0.set_theta_direction(-1)       # clockwise positive
# ax0.plot(alpha_rot, pdf_slice, linewidth=2)
# # incidence arrow (inward)
# ax0.annotate('', xy=(alpha_inc_rot, r_head), xytext=(alpha_inc_rot, r_tail),
#              arrowprops=dict(arrowstyle='->', lw=2))
# ax0.set_title('Scattering PDF slice (x–z plane, rotated by surface angle)\n(arrow = incidence projection)', pad=10)
# ax0.set_rlabel_position(180)
# ax0.grid(True)
#
# # (b) 3D scattering lobe – rotated coordinates
# ax1 = fig.add_subplot(gs[0, 1], projection='3d')
# surf_pdf = ax1.plot_surface(Xr, Y, Zr, rstride=2, cstride=2, cmap='viridis', linewidth=0, antialiased=True)
# # incidence arrow (3D), start at tip, point toward origin
# arrow_len = 1.2
# x0, y0, z0 = arrow_len * ixr, arrow_len * iy, arrow_len * izr
# dx, dy, dz = -arrow_len * ixr, -arrow_len * iy, -arrow_len * izr
# ax1.quiver(x0, y0, z0, dx, dy, dz, arrow_length_ratio=0.1, linewidth=2)
# ax1.set_title('Full 3D Scattering PDF (rotated to horizontal surface)\n(arrow = incidence)')
# ax1.set_box_aspect((1, 1, 1))
# cbar = fig.colorbar(surf_pdf, ax=ax1, shrink=0.7, pad=0.05)
# cbar.set_label('PDF / max')
#
# # (c) generated surface – rotated to appear horizontal
# ax2 = fig.add_subplot(gs[1, :], projection='3d')
# surf_s = ax2.plot_surface(Xsr, Ys, Zsr, rstride=1, cstride=1, cmap='terrain', linewidth=0, antialiased=True)
# ax2.set_aspect('equal')
# ax2.set_title('Generated PolyGaussian Surface (rotated to horizontal)')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# fig.colorbar(surf_s, ax=ax2, shrink=0.7, pad=0.05, label='Height')
#
# plt.tight_layout()
# plt.show()
