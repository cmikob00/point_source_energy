#!/usr/bin/env python3
"""
Taylor–Sedov–von Neumann blast wave (self-similar point explosion)
Supports planar (1-D), cylindrical (2-D) and spherical (3-D) geometry.
Supports free-air, ideal-surface, and intermediate ("surface-interacting") bursts.
Includes Rankine–Hugoniot post-shock states and diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------------
# Physical / numerical parameters (edit these)
# ---------------------------------------------------------------------------
geometry           = 3                  # 1 = planar, 2 = cylindrical, 3 = spherical
burst_type         = "free_air"         # "free_air" or "surface"

gamma              = 1.4                # ratio of specific heats
E                  = 1.0e12             # true explosion energy [J] (or energy per unit length / area)
rho0               = 1.225              # ambient density   [kg m^-3]
p0                 = 1.01325e5          # ambient pressure  [Pa]
R_gas              = 287.0              # specific gas constant for air [J kg^-1 K^-1]
T0                 = p0 / (rho0 * R_gas)

t_min              = 1.0e-6             # earliest time of interest [s]
t_max              = 1.0e-1             # latest time of interest  [s]
n_times            = 400                # number of time samples

# Approximate self-similar constant ξ₀ for γ = 1.4
XI0 = {
    1: 1.105,   # planar
    2: 1.121,   # cylindrical
    3: 1.033    # spherical
}

# Ideal surface reflection factor (classic method-of-images value)
SURFACE_FACTOR = 2.0

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
d     = geometry
xi0   = XI0[d]
alpha = 2.0 / (d + 2)          # R ∝ t^α
beta  = 1.0 / (d + 2)          # R ∝ E^β

# Effective energy seen by the blast wave
bt = burst_type.lower().strip()
if bt == "free_air":
    E_eff = E
    burst_label = "Free-Air"
elif bt == "surface":
    E_eff = E * SURFACE_FACTOR
    burst_label = f"Surface (×{SURFACE_FACTOR:.1f})"
else:
    raise ValueError("burst_type must be 'free_air' or 'surface'")

# Strong-shock Rankine–Hugoniot factors
rho_ratio = (gamma + 1.0) / (gamma - 1.0)          # ρ₂ / ρ₀
u_factor  = 2.0 / (gamma + 1.0)                    # u₂ / D
p_factor  = 2.0 / (gamma + 1.0)                    # p₂ / (ρ₀ D²)

# ---------------------------------------------------------------------------
# Core self-similar solution
# ---------------------------------------------------------------------------
def blast_radius(t):
    """Sedov–Taylor blast-wave radius R(t)."""
    return xi0 * (E_eff / rho0)**beta * t**alpha

def shock_speed(t, R=None):
    """Shock propagation velocity D = dR/dt."""
    if R is None:
        R = blast_radius(t)
    return alpha * R / t

def post_shock_state(t):
    """
    Return post-shock quantities at the shock front:
        R, D, u2, rho2, p2, T2
    """
    R  = blast_radius(t)
    D  = shock_speed(t, R)
    u2 = u_factor * D
    # Force rho2 to have the same shape as t (constant in strong-shock limit)
    rho2 = np.full_like(np.asarray(t, dtype=float), rho_ratio * rho0, dtype=float)
    p2   = p_factor * rho0 * D**2
    T2   = p2 / (rho2 * R_gas)
    return R, D, u2, rho2, p2, T2

# ---------------------------------------------------------------------------
# Time grid and evaluation
# ---------------------------------------------------------------------------
t = np.logspace(np.log10(t_min), np.log10(t_max), n_times)

R, D, u2, rho2, p2, T2 = post_shock_state(t)

# ---------------------------------------------------------------------------
# Diagnostic output at a few representative times
# ---------------------------------------------------------------------------
print(f"Geometry          : {d}-D  (1=planar, 2=cylindrical, 3=spherical)")
print(f"Burst type        : {burst_label}")
print(f"True energy E     : {E:.4g} J")
print(f"Effective energy  : {E_eff:.4g} J")
print(f"γ                 : {gamma}")
print(f"ξ₀                : {xi0}")
print(f"Ambient density   : {rho0:.4g} kg m⁻³")
print()
print(f"{'t [s]':>10}  {'R [m]':>12}  {'D [m/s]':>12}  "
      f"{'u₂ [m/s]':>12}  {'ρ₂ [kg/m³]':>12}  {'p₂ [Pa]':>12}  {'T₂ [K]':>10}")
print("-" * 90)

for tt in np.logspace(np.log10(t_min), np.log10(t_max), 6):
    Rr, Dd, uu, rr, pp, TT = post_shock_state(tt)
    print(f"{tt:10.3e}  {Rr:12.4e}  {Dd:12.4e}  "
          f"{uu:12.4e}  {rr:12.4e}  {pp:12.4e}  {TT:10.1f}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
ax = axes.ravel()

def style(ax, ylabel, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.xaxis.set_major_formatter(ScalarFormatter())

ax[0].plot(t, R, "k-", lw=2)
style(ax[0], "R [m]", "Blast-wave radius")

ax[1].plot(t, D, "C0-", lw=2)
style(ax[1], "D [m s⁻¹]", "Shock propagation velocity")

ax[2].plot(t, u2, "C1-", lw=2)
style(ax[2], "u₂ [m s⁻¹]", "Post-shock particle velocity")

ax[3].plot(t, rho2, "C2-", lw=2)
style(ax[3], "ρ₂ [kg m⁻³]", "Post-shock density")
ax[3].set_yscale("linear")

ax[4].plot(t, p2, "C3-", lw=2)
style(ax[4], "p₂ [Pa]", "Post-shock pressure")

ax[5].plot(t, T2, "C4-", lw=2)
style(ax[5], "T₂ [K]", "Post-shock temperature")

geo_name = {1: "Planar", 2: "Cylindrical", 3: "Spherical"}[d]
fig.suptitle(f"Taylor–Sedov–von Neumann blast wave\n"
             f"{geo_name}, {burst_label}, γ = {gamma}",
             fontsize=13, fontweight="bold")

plt.show()