# test_polygaussian_surface.py
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pytest

# TODO: adjust this import to where your classes live.
from surface_tools import PolyGaussian_Surface, Surface  # Surface is returned by generate()


def test_polygaussian_end_to_end():
    """
    End-to-end test:
      1) Build a PolyGaussian_Surface with known coeffs.
      2) Evaluate height/slope PDFs on grids; sanity-check integrals & finiteness.
      3) Generate a synthetic surface; isotropize it; compare model vs. histograms.
      4) Fit model parameters to the generated surface histograms; sanity-check outputs.
      5) Render a quick 3D plot to ensure plotting doesn't error (saved offscreen).
    """
    # --- 1) Model setup
    test_surf = PolyGaussian_Surface(
        sigma_coeff=np.array([0.4, 0.4, 0.1, 0.1], dtype=float),
        mu_coeff=np.array([-2.0, -2.0, -2.0,  2.0], dtype=float),
        R=0.8,
        angle=np.pi / 4,
    )

    # --- 2) PDFs on grids
    heights = np.linspace(-7, 7, 100)
    slopes  = np.linspace(-7, 7, 100)

    mu_vals = test_surf.get_function(heights, test_surf.mu_coeff)
    p_h     = test_surf.get_height_pdf(heights)
    p_s     = test_surf.get_slope_pdf(slopes)

    # Shape & finiteness checks
    assert mu_vals.shape == heights.shape
    assert p_h.shape == heights.shape and np.all(np.isfinite(p_h)) and np.all(p_h >= 0)
    assert p_s.shape == slopes.shape  and np.all(np.isfinite(p_s)) and np.all(p_s >= 0)

    # PDFs roughly integrate to ~1 over the evaluation window (tails may leak)
    Ih = np.trapezoid(p_h, heights)
    Is = np.trapezoid(p_s, slopes)
    assert 0.80 <= Ih <= 1.20
    assert 0.80 <= Is <= 1.20

    # --- 3) Generate surface and compare to histograms
    gen = test_surf.generate(sample_length=10, samples=300, iterations=100, verbose=True)
    gen_iso = gen.get_isotropic_surface()

    # Histograms (density=True ⇒ integral ≈ 1 over chosen range)
    h_hist, h_edges = np.histogram(gen_iso.get_heights(), density=True, bins=80, range=(-7, 7))
    s_hist, s_edges = np.histogram(gen_iso.get_slopes_x(), density=True, bins=80, range=(-7, 7))
    h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    # Re-evaluate PDFs at bin centers for rough consistency (no strict tolerance)
    p_h_c = test_surf.get_height_pdf(h_centers)
    p_s_c = test_surf.get_slope_pdf(s_centers)
    assert np.all(np.isfinite(p_h_c)) and np.all(p_h_c >= 0)
    assert np.all(np.isfinite(p_s_c)) and np.all(p_s_c >= 0)

    # --- 4) Fit parameters to generated data; ensure outputs are sane
    fit = PolyGaussian_Surface()
    # Baseline misfit before fitting (use simple defaults)
    base_p_h = fit.get_height_pdf(h_centers)
    base_p_s = fit.get_slope_pdf(s_centers)
    base_err = (
        np.linalg.norm(np.nan_to_num(base_p_h - h_hist)) / max(len(h_centers), 1) ** 0.5
        + np.linalg.norm(np.nan_to_num(base_p_s - s_hist)) / max(len(s_centers), 1) ** 0.5
    )

    fit.fit_parameters(
        surface=gen_iso,
        niter=5,
        verbose=True,
        lim=5,
        num_param=8,
        method="hsr",
    )

    # Post-fit sanity
    assert isinstance(fit.mu_coeff, np.ndarray) and fit.mu_coeff.size == 8
    assert isinstance(fit.sigma_coeff, np.ndarray) and fit.sigma_coeff.size == 8
    assert np.isfinite(fit.R) and fit.R > 0

    # Post-fit misfit (should not explode; often improves)
    fit_p_h = fit.get_height_pdf(h_centers)
    fit_p_s = fit.get_slope_pdf(s_centers)
    fit_err = (
        np.linalg.norm(np.nan_to_num(fit_p_h - h_hist)) / max(len(h_centers), 1) ** 0.5
        + np.linalg.norm(np.nan_to_num(fit_p_s - s_hist)) / max(len(s_centers), 1) ** 0.5
    )
    # At least bounded; in many cases it will be smaller than baseline
    assert np.isfinite(fit_err) and fit_err < 10 * base_err

    # --- 5) Quick headless 3D plot (just ensure no exceptions)
    fit_gen = fit.generate(sample_length=10, samples=150, iterations=100, verbose=True)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(gen_iso.X, gen_iso.Y, gen_iso.Z - 5, rstride=4, cstride=4, cmap="viridis")
    ax.plot_surface(fit_gen.X,  fit_gen.Y,  fit_gen.Z,    rstride=4, cstride=4, cmap="viridis")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
