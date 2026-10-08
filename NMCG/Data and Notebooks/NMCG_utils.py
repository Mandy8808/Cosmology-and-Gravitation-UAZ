"""
NMCG_utils.py

Helper module for the:
     Graphs_Lambda for obtaining the f values of CDM
     f(R) scalaron perturbation notebook (Graphs_Scalaron).

Groups together:
  - The algebraic relations between the model parameter f, H0 and the
    scalaron mass m.
  - The two ODE systems for h(tau) and h(z) (dimensionless scalaron
    perturbation), plus convenience solvers built on scipy's solve_ivp.
  - The exact (Bessel-function) solution h_exact(z), evaluated with
    arbitrary precision via mpmath.
  - A helper to build the upper/lower envelope of an oscillating h(z)
    curve using scipy.signal.find_peaks + interpolation.
  - A small legend-drawing helper to avoid repeating the same
    axes.plot/axes.text block in every figure.
  - The symbolic solve used to back out which f value reproduces a
    given physical (mass, frequency) pair.
"""

import numpy as np
import sympy as sp
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from matplotlib.tri import Triangulation


# ---------------------------------------------------------------------------
# Algebraic relations: scalaron mass and the m/H0 ratio as functions of f
# ---------------------------------------------------------------------------

def m(H0, beta, f):
    """
    Scalaron mass m(H0, beta, f) obtained from the background f(R) + f Lm
    field equations (beta is the coupling constant, f the model parameter).
    """
    return (41472 * H0**10 - 1440 * f * H0**6 * beta + 4 * f**2 * H0**2 * beta**2) / \
           (216 * f * H0**4 * beta - f**2 * beta**2)


def m2_over_H02(f):
    """
    (m / H0)^2 written purely as a function of f, for the case beta = H0**4
    (i.e. the curve plotted in the "mH0_vs_f" figure).
    """
    return 4 * (f**2 - 3600 * f + 10368) / (f * (216 - f))


# ---------------------------------------------------------------------------
# ODE systems and solvers for the scalaron perturbation h
# ---------------------------------------------------------------------------

def system_tau(tau, y, m2H02, qtilde):
    """
    First-order system for h(tau), where tau is the dimensionless
    (e-fold-like) time variable:
        h'' + 3 h' + (qtilde^2 * exp(-2 tau) + m2H02) h = 0
    """
    h, hp = y
    hpp = -3 * hp - (qtilde**2 * np.exp(-2 * tau) + m2H02) * h
    return [hp, hpp]


def solve_h_tau(H0val, fval, qval, tau_span=(0, 4), y0=(1, 0)):
    """
    Build m2H02 for a given (H0, f) pair and integrate h(tau) with
    solve_ivp. Returns the solve_ivp solution object (dense_output=True).
    """
    m2H02 = m(H0val, H0val**4, fval) / H0val**2
    sol = solve_ivp(
        system_tau, tau_span, y0,
        args=(m2H02, qval),
        dense_output=True,
    )
    return sol


def system_z(z, y, m2H02, qtilde):
    """
    First-order system for h(z), the redshift-domain version of the
    scalaron perturbation equation:
        (1+z)^2 h'' - 2(1+z) h' + [qtilde^2 (1+z)^2 + m2H02] h = 0
    """
    h, hp = y
    hpp = (
        2 * (1 + z) * hp
        - (qtilde**2 * (1 + z)**2 + m2H02) * h
    ) / (1 + z)**2
    return [hp, hpp]


def solve_h_z(H0val, fval, qval, z_span=(0, 1), y0=(1, 0), max_step=1e-3):
    """
    Build m2H02 for a given (H0, f) pair and integrate h(z) with
    solve_ivp. Returns the solve_ivp solution object (dense_output=True).
    """
    m2H02 = m(H0val, H0val**4, fval) / H0val**2
    sol = solve_ivp(
        system_z, z_span, y0,
        args=(m2H02, qval),
        dense_output=True,
        max_step=max_step,
    )
    return sol


# ---------------------------------------------------------------------------
# Exact solution (Bessel functions), arbitrary precision via mpmath
# ---------------------------------------------------------------------------

def h_exact(z, q, m, dps=500):
    """
    Exact solution h(z) in terms of Bessel functions J and Y, evaluated
    with mpmath at `dps` decimal digits of precision.

    z, q, m are expected as mpmath mpf/mpc-compatible numbers (or plain
    numbers, which mpmath will coerce).
    """
    mp.mp.dps = dps
    s = mp.sqrt(9 - 4 * m**2)

    return (
        mp.pi / 4
        * (1 + z)**(mp.mpf(3) / 2)
        * (
            (
                -(3 + s) * mp.besselj(s / 2, q)
                + 2 * q * mp.besselj((2 + s) / 2, q)
            )
            * mp.bessely(s / 2, q + q * z)
            +
            mp.besselj(s / 2, q + q * z)
            * (
                (3 + s) * mp.bessely(s / 2, q)
                - 2 * q * mp.bessely((2 + s) / 2, q)
            )
        )
    )


def h_exact_array(z_values, q, m, dps=500):
    """
    Vectorized convenience wrapper: evaluate h_exact over a list of z
    values (given as plain floats or mpmath mpf) and return a list of
    real-part floats, ready for plotting with matplotlib.
    """
    mp.mp.dps = dps
    q = mp.mpf(q) if not isinstance(q, mp.mpf) else q
    m = mp.mpf(m) if not isinstance(m, mp.mpf) else m
    h_values = [h_exact(mp.mpf(z), q, m, dps=dps) for z in z_values]
    return [float(mp.re(h)) for h in h_values]


# ---------------------------------------------------------------------------
# Envelope extraction for an oscillating h(z) curve
# ---------------------------------------------------------------------------

def find_envelope(z_plot, h, kind="cubic"):
    """
    Find the upper and lower envelope of an oscillating array h(z_plot)
    by locating its peaks/troughs and interpolating through them.
    Returns (env_upper, env_lower), each the same length as z_plot.
    """
    peaks, _ = find_peaks(h)
    troughs, _ = find_peaks(-h)

    env_upper = interp1d(
        z_plot[peaks], h[peaks],
        kind=kind, bounds_error=False, fill_value="extrapolate",
    )(z_plot)

    env_lower = interp1d(
        z_plot[troughs], h[troughs],
        kind=kind, bounds_error=False, fill_value="extrapolate",
    )(z_plot)

    return env_upper, env_lower


# ---------------------------------------------------------------------------
# Plot helper: small colored line + label, used for the H0=67/74 legends
# ---------------------------------------------------------------------------

def draw_line_legend(ax, entries, legend_x=0.725, y_start=0.87, y_step=0.08,
                      line_dx=0.07, label_dx=0.09, fontsize=11):
    """
    Draw a compact hand-made legend of colored line segments + labels in
    axes-fraction coordinates, to avoid repeating the same block of
    ax.plot/ax.text calls in every figure.

    entries: list of dicts, each with keys
        'color', 'linestyle', 'label', and optionally 'linewidth'.
    """
    y = y_start
    for entry in entries:
        color = entry["color"]
        linestyle = entry.get("linestyle", "-")
        linewidth = entry.get("linewidth", 2.2)
        label = entry["label"]

        ax.plot(
            [legend_x, legend_x + line_dx], [y, y],
            color=color, linestyle=linestyle, linewidth=linewidth,
            transform=ax.transAxes, clip_on=False,
        )
        ax.text(
            legend_x + label_dx, y, label,
            transform=ax.transAxes,
            fontsize=fontsize, ha="left", va="center", color="black",
        )
        y -= y_step


# ---------------------------------------------------------------------------
# Physical <-> dimensionless conversion and symbolic solve for f
# ---------------------------------------------------------------------------

# Constants used in the physical -> dimensionless conversion

MPC_KM = 3.0856775814913673e19   # km per Mpc
HBAR = 6.582119569e-16           # reduced Planck constant, eV*s


def solve_f_for_mass(H0Val, m_eV):
    """
    Given H0 (km/s/Mpc), a gravitational-wave frequency fGW (Hz) and a
    scalaron mass m_eV (eV), convert to dimensionless (adimensional)
    mass/frequency and symbolically solve m2_over_H02(f) == m_ad^2 for f.

    Returns fsol.
    """
    mval = m_eV / HBAR

    H0SI = H0Val / MPC_KM

    f = sp.symbols("f")
    kval = sp.nsimplify((mval / H0SI)**2)

    eq = -4 * (10368 + (-360 + f) * f) / ((-216 + f) * f) - kval
    numerator, _ = sp.fraction(sp.together(eq))
    fsol = sp.solve(sp.Eq(numerator, 0), f)

    return fsol


def real_positive_roots(fsol, tol=1e-9):
    """
    Given a list of sympy solutions (as returned by solve_f_for_mass,
    generally 2 roots from the quadratic-in-f equation), keep only the
    ones that are real and positive, as plain floats.
    """
    roots = []
    for sol in fsol:
        val = sp.N(sol)
        re, im = sp.re(val), sp.im(val)
        if abs(im) < tol and re > 0:
            roots.append(float(re))
 
    if not roots:
        return None
    return min(roots)

def f_bounds_from_mass_table(df, H0Val, mass_min_col, mass_max_col):
    """
    Given a DataFrame with one row per experiment/bound and columns for
    the minimum/maximum mass (in eV), solve for the corresponding f
    roots (via solve_f_for_mass + real_positive_roots) for every row.
    """
    
    df = df.copy()
    df["f_min_roots"] = df[mass_min_col].apply(
        lambda m: real_positive_roots(solve_f_for_mass(H0Val, m))
    )
    df["f_max_roots"] = df[mass_max_col].apply(
        lambda m: real_positive_roots(solve_f_for_mass(H0Val, m))
    )
    return df



def masked_triangulation(x, y, max_edge=None, percentile=95):
    """Delaunay triangulation of (x, y), masking triangles whose 
    longest edge exceeds a threshold (auto-set via percentile if not given) 
    to hide spurious long triangles over gaps/concave regions."""
    
    tri = Triangulation(x, y)

    # Longitudes de los 3 lados de cada triángulo
    triangles = tri.triangles
    xt, yt = x[triangles], y[triangles]
    edge1 = np.hypot(xt[:, 0] - xt[:, 1], yt[:, 0] - yt[:, 1])
    edge2 = np.hypot(xt[:, 1] - xt[:, 2], yt[:, 1] - yt[:, 2])
    edge3 = np.hypot(xt[:, 2] - xt[:, 0], yt[:, 2] - yt[:, 0])
    max_edge_per_triangle = np.max([edge1, edge2, edge3], axis=0)

    if max_edge is None:
        # Umbral automático: percentil de las longitudes de lado
        max_edge = np.percentile(max_edge_per_triangle, percentile)

    mask = max_edge_per_triangle > max_edge
    tri.set_mask(mask)
    return tri