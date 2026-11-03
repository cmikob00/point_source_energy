#!/usr/bin/env python3
"""
Axisymmetric 3-D Taylor-Sedov blast wave with a reflecting ground plane.

Physics:
  - Compressible, inviscid Euler equations in cylindrical (r,z) coordinates.
  - Axisymmetry: no azimuthal velocity or azimuthal variation.
  - Ideal-gas equation of state with constant gamma.
  - An ideal, rigid, impermeable, perfectly reflecting ground at z = 0.
  - Finite-radius thermal energy deposition, normalized to total energy E
    in the physical half-space z >= 0.

The finite burst height H is a continuous input. There is no empirical
surface-interaction factor. Reflection and the evolving shock geometry arise
from the Euler equations and the ground boundary condition.

Numerics:
  - Second-order MUSCL finite-volume spatial reconstruction in primitive variables.
  - Monotonized-central (MC) slope limiter with positivity-preserving slope scaling.
  - Local Rusanov (local Lax-Friedrichs) numerical flux.
  - Two-stage SSP-RK2 time integration (method of lines).
  - Cylindrical cell volumes and face areas are included explicitly.
  - Explicit CFL time stepping, with a stage-1 CFL rejection check.

Phase-1 verification and diagnostics:
  - Free-air Sedov-Taylor radius comparison in the unreflected, point-source-like
    time window (diagnostic comparison, not an automatic pass/fail certification).
  - Local normal Rankine-Hugoniot comparisons using interpolated upstream and
    downstream states sampled on either side of a gradient-refined shock front.
  - Selected-ray density and pressure profiles through the detected shock,
    including cases where the front is detected but the jump is not yet resolvable.
  - Mass/energy budgets including signed energy and mass flow through open
    boundaries; positivity is enforced by conservative flux blending rather
    than post-update cell-local mass, momentum, or energy repairs.
  - Configurable virtual pressure gauges, peak overpressure, threshold arrival
    time, and positive-phase impulse.
  - Run summaries and optional comparison of separately executed grid/domain
    cases to support convergence studies.
  - Local shock-profile diagnostics use bilinear ray sampling, maximum-gradient
    front refinement, normal-offset upstream/downstream states, and estimated normal speed.
  - Optional Numba JIT acceleration of repeated numerical kernels; NumPy fallback.

Important limitations:
  - This is an inviscid ideal-gas model, not a soil/ground-response model.
  - The initial energy deposition has finite radius; grid studies must hold its
    physical width fixed to isolate discretization effects.
  - MC-limited MUSCL reconstruction reduces (but does not eliminate) numerical
    diffusion; Rusanov fluxes remain comparatively dissipative.
  - A conservative high-/low-order Rusanov flux blend is applied only when a
    MUSCL forward-Euler stage would violate density/internal-energy positivity.
  - The low-order fallback is piecewise-constant Rusanov; if it is not admissible
    at the selected step, the step is rejected and retried at a smaller dt.
  - The reconstruction is dimension-by-dimension in r and z; it is not a fully
    multidimensional unsplit gradient reconstruction.
  - The shock-front/volume diagnostic uses a density threshold and is therefore
    an operational numerical estimate, not an exact discontinuity location.
  - Sedov and Rankine-Hugoniot comparisons are approximate diagnostics subject
    to finite deposition size, grid resolution, front detection, and boundary
    effects. They are not substitutes for formal verification and validation.

Requires: numpy, matplotlib; optional numba for acceleration and imageio/imageio-ffmpeg for MP4 output
"""

from datetime import datetime
from pathlib import Path
import csv

import numpy as np

# Optional JIT acceleration. The code retains a NumPy fallback if Numba is absent.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on systems without a working Numba install
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm


# =============================================================================
# USER INPUTS
# =============================================================================

# Explosion and ambient atmosphere
E_kt = 20.0                 # explosion energy [kt]
HOB = 30.0                # height of burst above reflecting plane [m]
rho0 = 1.15               # ambient density [kg/m^3]
p0 = 1.01325e5              # ambient pressure [Pa]
gamma = 1.4                # ratio of specific heats

# Time interval
t_end = 0.1               # simulation end time [s]
CFL = 0.40                 # Courant number; keep <= 0.5 for this scheme
MAX_STEPS = 500000

# Phase-2 higher-order numerical method controls
SPATIAL_RECONSTRUCTION = "MUSCL"  # Current implementation: MUSCL
SLOPE_LIMITER = "MC"               # "MC" (recommended) or "minmod"
TIME_INTEGRATOR = "SSP_RK2"        # Two-stage, second-order SSP Runge-Kutta
MAX_CFL_RETRIES = 20               # Retry when stage CFL / low-order positivity requires a smaller dt
CFL_RETRY_SHRINK = 0.80             # Multiplicative safety factor applied after a rejected step

# Conservative positivity-preserving flux-correction controls.  The low-order
# baseline is piecewise-constant Rusanov using the same EOS, boundaries, and
# cylindrical source as the MUSCL operator.  A single scalar alpha blends the
# low-/high-order face fluxes for a stage; alpha=1 means no correction.
RHO_POSITIVITY_FLOOR = 1.0e-12      # kg/m^3; admissibility threshold, not a cell repair
INTERNAL_ENERGY_POSITIVITY_FLOOR = 1.0e-10 / (gamma - 1.0)  # J/m^3; matches prior p-floor scale
FLUX_BLEND_BISECTION_ITERS = 48
FLUX_BLEND_SAFETY_FACTOR = 1.0 - 1.0e-8
MIN_TIMESTEP_FRACTION = 1.0e-16
INITIAL_CFL_SAFETY_FACTOR = 0.90  # Predictor margin for SSP-RK2 stage-2 wave-speed growth

# Runtime optimization. Numba accelerates the repeated MUSCL/Rusanov spatial
# operator and CFL estimate. The positivity fallback is evaluated only when
# the high-order candidate is inadmissible and uses vectorized NumPy arrays.
# Set False to use the NumPy MUSCL operator instead of the compiled kernel.
USE_NUMBA = True

# Computational domain / resolution
NR = 180                   # number of radial cells
NZ = 360                   # number of vertical cells
DOMAIN_FACTOR = 1.30       # domain extends this many free-air reference radii
DEPOSITION_CELLS = 2.5     # Original Gaussian width in cell widths
# Optional fixed physical deposition width [m].  Leave as None to preserve
# the original DEPOSITION_CELLS behavior.  Set a numeric value when comparing
# different grids so the physical initial condition is identical.
DEPOSITION_WIDTH_M = None

# Shock-front diagnostic
SHOCK_DENSITY_EXCESS = 0.02  # shock mask: rho > rho0*(1 + this value)
N_ANGLE_BINS = 240
# Shock-structure diagnostic controls. These affect diagnostics only, not the solver.
SHOCK_PROFILE_SAMPLE_OFFSET_CELLS = 3.5  # sample states this many max-cell widths from front
SHOCK_PROFILE_MIN_POSTSOURCE_FACTOR = 1.5  # downstream sample must be beyond this * deposition width
SHOCK_PROFILE_HALF_WIDTH_CELLS = 8.0  # plotted profile extent on either side of front
SHOCK_PROFILE_RAY_ANGLES_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
SHOCK_PROFILE_MAX_GRADIENT_WINDOW_CELLS = 2.5  # locate steepest density drop near threshold crossing
SHOCK_PROFILE_DS_FACTOR = 0.5  # ray interpolation spacing as fraction of min(dr,dz)

# Phase-1 verification / diagnostic controls
MAKE_PHASE1_DIAGNOSTICS = True
SEDOV_CLEARANCE_FRACTION = 0.80
SEDOV_MIN_RADIUS_TO_DEPOSITION = 5.0
SEDOV_MIN_DETECTED_ANGLE_FRACTION = 0.75
GAUGE_ARRIVAL_THRESHOLD_PA = 100.0  # first overpressure crossing used as arrival
# Gauge positions are absolute (r,z) coordinates in meters.  A requested
# ground gauge at z=0 samples the first cell center immediately above the wall.
GAUGE_LOCATIONS = (
    ("surface_axis", 0.0, 0.0),
    ("surface_50m", 50.0, 0.0),
    ("surface_100m", 100.0, 0.0),
    ("surface_150m", 150.0, 0.0),
    ("surface_200m", 200.0, 0.0),
    ("burst_axis", 0.0, None),  # None means the current HOB value
)
# Add paths to earlier run directories containing run_summary.csv to compare
# grid/domain studies.  The current run is automatically included.
# Grid study: hold DEPOSITION_WIDTH_M and all physical inputs fixed; increase
# NR,NZ.  Domain study: vary DOMAIN_FACTOR while scaling NR,NZ to keep dr,dz
# approximately fixed.  Then list the prior run directories here.
CONVERGENCE_REFERENCE_RUNS = ()

# Output
PRINT_EVERY = 100         # print progress every N time steps
MAKE_PLOTS = True
# Logarithmically spaced snapshot targets resolve the early blast expansion.
# Targets later than t_end are ignored; the final state is always included.
SNAPSHOT_TIMES = (
    1.0e-5, 3.0e-5, 5.e-5,
    1.0e-4, 3.0e-4, 5.e-4,
    1.0e-3, 3.0e-3, 5.e-3,
    1.0e-2, 3.0e-2, 5.e-2,
    1.0e-1, 3.0e-1, 5.e-1,
    1.0e+0, 3.0e+0, 5.e+0,
)

SNAPSHOT_OUTPUT_DIR = "blast_wave_snapshots"
SNAPSHOT_DPI = 300

# MP4 video output
MAKE_MP4 = True
MP4_FPS = 1
MP4_FILENAME = "blast_wave.mp4"


# =============================================================================
# SEDOV-TAYLOR REFERENCE (SPHERICAL, FREE AIR)
# =============================================================================

# Approximate strong-shock similarity constant for gamma = 1.4.
# This is retained as a reference scale, not used to advance the Euler solution.
XI0 = 1.033

alpha = 2.0 / 5.0
beta = 1.0 / 5.0

E = 4.184e12 * E_kt  # convert energy from kt to Joules


def sedov_radius(t, energy=E):
    """Spherical free-air Sedov-Taylor reference radius [m]."""
    t = np.asarray(t, dtype=float)
    return XI0 * (energy / rho0)**beta * t**alpha


def interaction_time(height):
    """
    Characteristic free-air time at which R_ST = HOB.
    For HOB = 0, the interaction time is zero by definition.
    """
    if height <= 0.0:
        return 0.0
    return (height / (XI0 * (E / rho0)**beta))**(1.0 / alpha)


# =============================================================================
# AXISYMMETRIC FINITE-VOLUME EULER SOLVER
# =============================================================================

# Conserved variables:
# U = [rho, rho*u_r, rho*u_z, total_energy_density]
#
# Cylindrical conservation law:
# dU/dt + (1/r) d(r F_r)/dr + dF_z/dz = S
# S = [0, p/r, 0, 0]
#
# The p/r geometric source is included in the radial momentum equation.


def primitive_from_conserved(U):
    """Convert conserved state to primitive variables."""
    rho = np.maximum(U[0], 1.0e-12)
    ur = U[1] / rho
    uz = U[2] / rho
    kinetic = 0.5 * rho * (ur**2 + uz**2)
    internal = np.maximum(U[3] - kinetic, 1.0e-12)
    pressure = (gamma - 1.0) * internal
    return rho, ur, uz, pressure


def conserved_from_primitive(rho, ur, uz, pressure):
    """Convert primitive variables to conserved variables."""
    total_energy = pressure / (gamma - 1.0) + 0.5 * rho * (ur**2 + uz**2)
    return np.array([rho, rho * ur, rho * uz, total_energy])


def physical_flux_r(U):
    rho, ur, uz, pressure = primitive_from_conserved(U)
    return np.array([
        rho * ur,
        rho * ur**2 + pressure,
        rho * ur * uz,
        (U[3] + pressure) * ur
    ])


def physical_flux_z(U):
    rho, ur, uz, pressure = primitive_from_conserved(U)
    return np.array([
        rho * uz,
        rho * ur * uz,
        rho * uz**2 + pressure,
        (U[3] + pressure) * uz
    ])


def sound_speed(rho, pressure):
    return np.sqrt(np.maximum(gamma * pressure / rho, 1.0e-20))


def rusanov_flux(UL, UR, direction):
    """Local Rusanov flux between left/right states."""
    if direction == "r":
        FL = physical_flux_r(UL)
        FR = physical_flux_r(UR)
        rhoL, urL, uzL, pL = primitive_from_conserved(UL)
        rhoR, urR, uzR, pR = primitive_from_conserved(UR)
        amax = np.maximum(np.abs(urL) + sound_speed(rhoL, pL),
                          np.abs(urR) + sound_speed(rhoR, pR))
    else:
        FL = physical_flux_z(UL)
        FR = physical_flux_z(UR)
        rhoL, urL, uzL, pL = primitive_from_conserved(UL)
        rhoR, urR, uzR, pR = primitive_from_conserved(UR)
        amax = np.maximum(np.abs(uzL) + sound_speed(rhoL, pL),
                          np.abs(uzR) + sound_speed(rhoR, pR))
    return 0.5 * (FL + FR) - 0.5 * amax[None, ...] * (UR - UL)


def make_initial_state(r_centers, z_centers, dr, dz):
    """
    Ambient gas plus Gaussian thermal energy deposition.

    A mirrored Gaussian is used in z so that the initial energy distribution
    is symmetric about z=0, consistent with the reflecting-wall condition.
    The energy deposited in z >= 0 is normalized to exactly E (up to
    floating-point summation).
    """
    rr, zz = np.meshgrid(r_centers, z_centers, indexing="ij")

    if DEPOSITION_WIDTH_M is None:
        a = DEPOSITION_CELLS * max(dr, dz)
    else:
        a = float(DEPOSITION_WIDTH_M)
        if a <= 0.0:
            raise ValueError("DEPOSITION_WIDTH_M must be positive when specified.")
    radial_term = rr**2
    gaussian_upper = np.exp(-(radial_term + (zz - HOB)**2) / (2.0 * a**2))
    gaussian_mirror = np.exp(-(radial_term + (zz + HOB)**2) / (2.0 * a**2))
    shape = gaussian_upper + gaussian_mirror

    # Exact axisymmetric cell volumes: pi*(r_outer^2-r_inner^2)*dz
    r_faces = np.linspace(0.0, r_centers[-1] + 0.5 * dr, len(r_centers) + 1)
    cell_volumes = np.pi * (r_faces[1:]**2 - r_faces[:-1]**2)[:, None] * dz

    normalization = np.sum(shape * cell_volumes)
    energy_density = E * shape / normalization

    rho = np.full_like(shape, rho0)
    ur = np.zeros_like(shape)
    uz = np.zeros_like(shape)
    pressure = p0 + (gamma - 1.0) * energy_density

    return conserved_from_primitive(rho, ur, uz, pressure), cell_volumes, a


def radial_volume_centroids(r_faces):
    """Volume centroids of cylindrical annular cells.

    The finite-volume weight is proportional to r dr.  Using the corresponding
    volume centroid for radial reconstruction makes a linear radial profile's
    reconstructed cell average consistent with the annular volume measure.
    """
    rin = r_faces[:-1]
    rout = r_faces[1:]
    denominator = np.maximum(rout**2 - rin**2, 1.0e-300)
    return (2.0 / 3.0) * (rout**3 - rin**3) / denominator


def minmod3(a, b, c):
    """Three-argument minmod function, applied elementwise."""
    same_positive = (a > 0.0) & (b > 0.0) & (c > 0.0)
    same_negative = (a < 0.0) & (b < 0.0) & (c < 0.0)
    result = np.zeros_like(a)
    result = np.where(same_positive, np.minimum(np.minimum(a, b), c), result)
    result = np.where(same_negative, np.maximum(np.maximum(a, b), c), result)
    return result


def mc_limited_slope(q_minus, q_center, q_plus, dx_minus, dx_plus):
    """Return a monotonized-central slope on a possibly nonuniform mesh.

    The MC limiter is minmod(2*dq/dx|-, centered secant, 2*dq/dx|+).
    On a uniform mesh this is the standard monotonized-central limiter.
    """
    d_minus = (q_center - q_minus) / dx_minus
    d_plus = (q_plus - q_center) / dx_plus
    d_center = (q_plus - q_minus) / (dx_minus + dx_plus)
    if SLOPE_LIMITER.upper() == "MC":
        return minmod3(2.0 * d_minus, d_center, 2.0 * d_plus)
    if SLOPE_LIMITER.lower() == "minmod":
        return minmod3(d_minus, d_plus, d_center)
    raise ValueError("SLOPE_LIMITER must be 'MC' or 'minmod'.")


def primitive_reconstruction_slopes(U, r_faces, z_centers, dz):
    """Calculate positivity-limited MUSCL slopes of primitive variables.

    Reconstructed variables are [rho, u_r, u_z, p].  Reflection ghost values
    enforce odd symmetry of the wall-normal velocity at r=0 and z=0; scalar
    quantities and tangential velocity are even.  Zero-gradient outflow ghosts
    are used at the two outer boundaries.

    Slopes are derivatives per meter.  A single cell-wise scaling factor is
    applied to all primitive slopes if needed to keep every reconstructed face
    value of density and pressure positive.  This preserves the cell-centred
    state while reducing only the reconstruction gradient.
    """
    rho, ur, uz, pressure = primitive_from_conserved(U)
    W = np.stack((rho, ur, uz, pressure), axis=0)
    nr, nz = rho.shape

    # Radial reconstruction: use cylindrical volume centroids, not arithmetic
    # midpoints, because cell averages are weighted by annular volume.
    r_centroid = radial_volume_centroids(r_faces)
    W_r = np.empty((4, nr + 2, nz), dtype=float)
    W_r[:, 1:-1, :] = W
    W_r[:, 0, :] = W[:, 0, :]
    W_r[1, 0, :] *= -1.0  # reflected radial velocity at the axis
    W_r[:, -1, :] = W[:, -1, :]  # zero-gradient outer radial ghost
    r_coord = np.concatenate(([-r_centroid[0]], r_centroid,
                              [r_centroid[-1] + (r_centroid[-1] - r_centroid[-2])]))
    dxm_r = (r_coord[1:-1] - r_coord[:-2])[None, :, None]
    dxp_r = (r_coord[2:] - r_coord[1:-1])[None, :, None]
    slope_r = mc_limited_slope(
        W_r[:, :-2, :], W_r[:, 1:-1, :], W_r[:, 2:, :], dxm_r, dxp_r
    )

    # Vertical reconstruction: reflected ground ghost and zero-gradient top.
    W_z = np.empty((4, nr, nz + 2), dtype=float)
    W_z[:, :, 1:-1] = W
    W_z[:, :, 0] = W[:, :, 0]
    W_z[2, :, 0] *= -1.0  # reflected vertical velocity at the ground
    W_z[:, :, -1] = W[:, :, -1]
    z_coord = np.concatenate(([-z_centers[0]], z_centers,
                              [z_centers[-1] + (z_centers[-1] - z_centers[-2])]))
    dxm_z = (z_coord[1:-1] - z_coord[:-2])[None, None, :]
    dxp_z = (z_coord[2:] - z_coord[1:-1])[None, None, :]
    slope_z = mc_limited_slope(
        W_z[:, :, :-2], W_z[:, :, 1:-1], W_z[:, :, 2:], dxm_z, dxp_z
    )

    # Positivity-preserving slope scaling.  Examine both radial faces and both
    # vertical faces for rho and p; scale all four primitive gradients together.
    theta = np.ones((nr, nz), dtype=float)
    dr_minus = r_faces[:-1] - r_centroid
    dr_plus = r_faces[1:] - r_centroid
    face_offsets = (
        slope_r[0] * dr_minus[:, None],
        slope_r[0] * dr_plus[:, None],
        slope_z[0] * (-0.5 * dz),
        slope_z[0] * (0.5 * dz),
    )
    pressure_offsets = (
        slope_r[3] * dr_minus[:, None],
        slope_r[3] * dr_plus[:, None],
        slope_z[3] * (-0.5 * dz),
        slope_z[3] * (0.5 * dz),
    )
    rho_floor = 1.0e-12
    pressure_floor = (gamma - 1.0) * 1.0e-10
    for center, offsets, floor in (
        (W[0], face_offsets, rho_floor),
        (W[3], pressure_offsets, pressure_floor),
    ):
        for offset in offsets:
            predicted = center + offset
            bad = predicted < floor
            allowable = np.ones_like(theta)
            denominator = center - predicted
            valid = bad & (denominator > 0.0)
            allowable[valid] = (center[valid] - floor) / denominator[valid]
            theta = np.minimum(theta, allowable)
    theta = np.clip(theta, 0.0, 1.0)
    slope_r *= theta[None, :, :]
    slope_z *= theta[None, :, :]

    return W, slope_r, slope_z, r_centroid


def primitive_face_to_conserved(W_face):
    """Convert [rho, ur, uz, p] face arrays to conserved face arrays."""
    return conserved_from_primitive(
        W_face[0], W_face[1], W_face[2], W_face[3]
    )


def apply_muscl_boundaries(U, r_faces, z_centers, dz):
    """Build second-order MUSCL left/right states at all cell faces.

    At reflecting boundaries, the reconstructed interior face state is mirrored
    in the normal velocity to make the ghost-side Riemann state.  At outflow
    boundaries, left and right states are equal to the extrapolated face state.
    """
    W, slope_r, slope_z, r_centroid = primitive_reconstruction_slopes(
        U, r_faces, z_centers, dz
    )
    nr, nz = U.shape[1:]
    dr_minus = r_faces[:-1] - r_centroid
    dr_plus = r_faces[1:] - r_centroid

    UL_r = np.empty((4, nr + 1, nz), dtype=float)
    UR_r = np.empty_like(UL_r)
    UL_z = np.empty((4, nr, nz + 1), dtype=float)
    UR_z = np.empty_like(UL_z)

    # Interior radial faces: left cell extrapolated to its outer face; right
    # cell extrapolated to its inner face.
    W_left = W[:, :-1, :] + slope_r[:, :-1, :] * dr_plus[:-1][None, :, None]
    W_right = W[:, 1:, :] + slope_r[:, 1:, :] * dr_minus[1:][None, :, None]
    UL_r[:, 1:-1, :] = primitive_face_to_conserved(W_left)
    UR_r[:, 1:-1, :] = primitive_face_to_conserved(W_right)

    # Axis reflection.
    W_axis = W[:, 0, :] + slope_r[:, 0, :] * dr_minus[0]
    W_axis_ghost = W_axis.copy()
    W_axis_ghost[1] *= -1.0
    UL_r[:, 0, :] = primitive_face_to_conserved(W_axis_ghost)
    UR_r[:, 0, :] = primitive_face_to_conserved(W_axis)

    # Outer radial zero-gradient outflow.
    W_outer = W[:, -1, :] + slope_r[:, -1, :] * dr_plus[-1]
    outer_state = primitive_face_to_conserved(W_outer)
    UL_r[:, -1, :] = outer_state
    UR_r[:, -1, :] = outer_state

    # Interior vertical faces.
    W_lower = W[:, :, :-1] + slope_z[:, :, :-1] * (0.5 * dz)
    W_upper = W[:, :, 1:] + slope_z[:, :, 1:] * (-0.5 * dz)
    UL_z[:, :, 1:-1] = primitive_face_to_conserved(W_lower)
    UR_z[:, :, 1:-1] = primitive_face_to_conserved(W_upper)

    # Reflecting ground.
    W_ground = W[:, :, 0] + slope_z[:, :, 0] * (-0.5 * dz)
    W_ground_ghost = W_ground.copy()
    W_ground_ghost[2] *= -1.0
    UL_z[:, :, 0] = primitive_face_to_conserved(W_ground_ghost)
    UR_z[:, :, 0] = primitive_face_to_conserved(W_ground)

    # Top zero-gradient outflow.
    W_top = W[:, :, -1] + slope_z[:, :, -1] * (0.5 * dz)
    top_state = primitive_face_to_conserved(W_top)
    UL_z[:, :, -1] = top_state
    UR_z[:, :, -1] = top_state

    return UL_r, UR_r, UL_z, UR_z


@njit
def _stable_timestep_numba(U, gamma_value, cfl_value, dr_value, dz_value):
    max_rate = 0.0
    nr = U.shape[1]
    nz = U.shape[2]
    for i in range(nr):
        for j in range(nz):
            rho = max(U[0, i, j], 1.0e-12)
            ur = U[1, i, j] / rho
            uz = U[2, i, j] / rho
            kinetic = 0.5 * rho * (ur * ur + uz * uz)
            internal = max(U[3, i, j] - kinetic, 1.0e-12)
            pressure = (gamma_value - 1.0) * internal
            c = np.sqrt(max(gamma_value * pressure / rho, 1.0e-20))
            rate = (abs(ur) + c) / dr_value + (abs(uz) + c) / dz_value
            if rate > max_rate:
                max_rate = rate
    return cfl_value / max(max_rate, 1.0e-30)


def stable_timestep(U, dr, dz):
    """Multidimensional CFL estimate; Numba path is algebraically identical."""
    if USE_NUMBA and NUMBA_AVAILABLE:
        return float(_stable_timestep_numba(U, gamma, CFL, dr, dz))
    rho, ur, uz, pressure = primitive_from_conserved(U)
    c = sound_speed(rho, pressure)
    max_rate = np.max((np.abs(ur) + c) / dr + (np.abs(uz) + c) / dz)
    return CFL / max(max_rate, 1.0e-30)



def cfl_state_diagnostics(U, dr, dz, r_centers=None, z_centers=None):
    """Locate and report the cell controlling the multidimensional CFL step."""
    rho_raw = U[0]
    rho = np.maximum(rho_raw, RHO_POSITIVITY_FLOOR)
    ur = U[1] / rho
    uz = U[2] / rho
    kinetic = 0.5 * rho * (ur**2 + uz**2)
    internal = U[3] - kinetic
    pressure = (gamma - 1.0) * np.maximum(internal, INTERNAL_ENERGY_POSITIVITY_FLOOR)
    c = sound_speed(rho, pressure)
    rate = (np.abs(ur) + c) / dr + (np.abs(uz) + c) / dz
    rate = np.where(np.isfinite(rate), rate, np.inf)
    i, j = np.unravel_index(int(np.argmax(rate)), rate.shape)
    diag = state_admissibility_diagnostics(U, r_centers, z_centers)
    diag.update({
        "cfl_rate_1_s": float(rate[i, j]),
        "local_cfl_dt_s": float(CFL / max(rate[i, j], 1.0e-30)),
        "cfl_i": int(i), "cfl_j": int(j),
        "cfl_r_m": float(r_centers[i]) if r_centers is not None else np.nan,
        "cfl_z_m": float(z_centers[j]) if z_centers is not None else np.nan,
        "cfl_rho_kg_m3": float(rho_raw[i, j]),
        "cfl_pressure_Pa": float(pressure[i, j]),
        "cfl_ur_m_s": float(ur[i, j]),
        "cfl_uz_m_s": float(uz[i, j]),
        "cfl_sound_speed_m_s": float(c[i, j]),
        "cfl_fluid_mach": float(np.hypot(ur[i, j], uz[i, j]) / max(c[i, j], 1.0e-30)),
        "cfl_max_rate_cell_count": int(np.count_nonzero(rate >= np.max(rate) * (1.0 - 1.0e-12))),
    })
    return diag


class StageCFLViolation(RuntimeError):
    """A step rejection with a suggested smaller dt and state diagnostics."""
    def __init__(self, suggested_dt, reason, diagnostics=None):
        super().__init__(reason)
        self.suggested_dt = float(suggested_dt)
        self.reason = str(reason)
        self.diagnostics = diagnostics or {}


def state_admissibility(U):
    """Return an EOS-based admissibility mask for conservative Euler states.

    An ideal-gas Euler state is admissible when rho is positive and the
    internal-energy density E - |m|^2/(2 rho) is positive.  Finite numerical
    thresholds are used here to avoid dividing by zero and to keep pressure
    well-defined; they are not applied as nonconservative state corrections.
    """
    rho = U[0]
    finite = np.all(np.isfinite(U), axis=0)
    rho_ok = np.isfinite(rho) & (rho >= RHO_POSITIVITY_FLOOR)
    rho_safe = np.maximum(np.where(np.isfinite(rho), rho, RHO_POSITIVITY_FLOOR),
                          RHO_POSITIVITY_FLOOR)
    kinetic = 0.5 * (U[1]**2 + U[2]**2) / rho_safe
    internal = U[3] - kinetic
    internal_ok = np.isfinite(internal) & (internal >= INTERNAL_ENERGY_POSITIVITY_FLOOR)
    admissible = finite & rho_ok & internal_ok
    return admissible, internal


def is_admissible_state(U):
    """Fast whole-grid admissibility test; no clipping or repair is performed."""
    admissible, _ = state_admissibility(U)
    return bool(np.all(admissible))


def state_admissibility_diagnostics(U, r_centers=None, z_centers=None):
    """Summarize the worst-density and worst-internal-energy cells."""
    rho = U[0]
    finite = np.all(np.isfinite(U), axis=0)
    rho_safe = np.maximum(np.where(np.isfinite(rho), rho, RHO_POSITIVITY_FLOOR),
                          RHO_POSITIVITY_FLOOR)
    ur = U[1] / rho_safe
    uz = U[2] / rho_safe
    internal = U[3] - 0.5 * rho_safe * (ur**2 + uz**2)
    pressure = (gamma - 1.0) * internal

    bad_rho = (~np.isfinite(rho)) | (rho < RHO_POSITIVITY_FLOOR)
    bad_internal = (~np.isfinite(internal)) | (internal < INTERNAL_ENERGY_POSITIVITY_FLOOR)
    bad_any = (~finite) | bad_rho | bad_internal

    # Select the most severe violated state for the reported location.
    if np.any(bad_rho):
        metric = np.where(bad_rho, rho, np.inf)
        i_bad, j_bad = np.unravel_index(int(np.argmin(metric)), rho.shape)
        selected_reason = "minimum density"
    elif np.any(bad_internal):
        metric = np.where(bad_internal, internal, np.inf)
        i_bad, j_bad = np.unravel_index(int(np.argmin(metric)), rho.shape)
        selected_reason = "minimum internal energy"
    else:
        # For valid states, report the cell controlling the CFL rate.
        c = np.sqrt(np.maximum(gamma * pressure / rho_safe, 1.0e-20))
        rate = (np.abs(ur) + c) / max(float(np.min(np.diff(r_centers))) if r_centers is not None else 1.0, 1.0e-30)
        if z_centers is not None:
            dz_local = float(np.min(np.diff(z_centers)))
        else:
            dz_local = 1.0
        rate += (np.abs(uz) + c) / max(dz_local, 1.0e-30)
        i_bad, j_bad = np.unravel_index(int(np.argmax(rate)), rho.shape)
        selected_reason = "maximum wave-speed/CFL rate"

    rho_b = float(rho[i_bad, j_bad])
    rho_d = max(rho_b if np.isfinite(rho_b) else RHO_POSITIVITY_FLOOR,
                RHO_POSITIVITY_FLOOR)
    ur_b = float(ur[i_bad, j_bad])
    uz_b = float(uz[i_bad, j_bad])
    p_b = float(pressure[i_bad, j_bad])
    c_b = float(np.sqrt(max(gamma * p_b / rho_d, 1.0e-20))) if np.isfinite(p_b) else np.nan
    speed = float(np.hypot(ur_b, uz_b))
    x_r = float(r_centers[i_bad]) if r_centers is not None else np.nan
    x_z = float(z_centers[j_bad]) if z_centers is not None else np.nan

    return {
        "selected_reason": selected_reason,
        "i": int(i_bad), "j": int(j_bad),
        "r_m": x_r, "z_m": x_z,
        "rho_kg_m3": rho_b,
        "pressure_Pa": p_b,
        "internal_energy_density_J_m3": float(internal[i_bad, j_bad]),
        "ur_m_s": ur_b, "uz_m_s": uz_b,
        "sound_speed_m_s": c_b,
        "flow_speed_m_s": speed,
        "fluid_mach": speed / c_b if np.isfinite(c_b) and c_b > 0.0 else np.nan,
        "bad_density_cells": int(np.count_nonzero(bad_rho)),
        "bad_internal_energy_cells": int(np.count_nonzero(bad_internal)),
        "nonfinite_cells": int(np.count_nonzero(~finite)),
        "minimum_density_kg_m3": float(np.nanmin(rho)),
        "minimum_internal_energy_density_J_m3": float(np.nanmin(internal)),
        "any_inadmissible": bool(np.any(bad_any)),
    }


def format_state_diagnostics(diag):
    """Compact user-facing description of the cell controlling a retry."""
    if not diag:
        return "no cell diagnostics available"
    return (
        f"cell=({diag.get('i')},{diag.get('j')}), "
        f"(r,z)=({diag.get('r_m', np.nan):.5g},{diag.get('z_m', np.nan):.5g}) m, "
        f"rho={diag.get('rho_kg_m3', np.nan):.4e} kg/m^3, "
        f"p={diag.get('pressure_Pa', np.nan):.4e} Pa, "
        f"u=({diag.get('ur_m_s', np.nan):.4e},{diag.get('uz_m_s', np.nan):.4e}) m/s, "
        f"c={diag.get('sound_speed_m_s', np.nan):.4e} m/s, "
        f"M_flow={diag.get('fluid_mach', np.nan):.4e}, "
        f"bad rho/e cells=({diag.get('bad_density_cells', 0)},"
        f"{diag.get('bad_internal_energy_cells', 0)})"
    )

def _spatial_operator_numpy(U, r_faces, z_centers, dz, cell_volumes):
    """Reference NumPy implementation of the original MUSCL-Rusanov operator."""
    UL_r, UR_r, UL_z, UR_z = apply_muscl_boundaries(U, r_faces, z_centers, dz)
    Fr = rusanov_flux(UL_r, UR_r, "r")
    Fz = rusanov_flux(UL_z, UR_z, "z")
    radial_areas = (2.0 * np.pi * r_faces * dz)[None, :, None]
    axial_areas = np.pi * (r_faces[1:]**2 - r_faces[:-1]**2)
    radial_net = (Fr[:, 1:, :] * radial_areas[:, 1:, :]
                  - Fr[:, :-1, :] * radial_areas[:, :-1, :])
    vertical_net = (Fz[:, :, 1:] * axial_areas[None, :, None]
                    - Fz[:, :, :-1] * axial_areas[None, :, None])
    rhs = -(radial_net + vertical_net) / cell_volumes[None, :, :]
    rho, ur, uz, pressure = primitive_from_conserved(U)
    r_mid_sum = r_faces[1:] + r_faces[:-1]
    rhs[1] += 2.0 * pressure / np.maximum(r_mid_sum[:, None], 1.0e-30)
    outer_radial_area = 2.0 * np.pi * r_faces[-1] * dz
    mass_out_rate = (
        np.sum(Fr[0, -1, :]) * outer_radial_area
        + np.sum(Fz[0, :, -1] * axial_areas)
    )
    energy_out_rate = (
        np.sum(Fr[3, -1, :]) * outer_radial_area
        + np.sum(Fz[3, :, -1] * axial_areas)
    )
    return rhs, float(mass_out_rate), float(energy_out_rate)



def _spatial_operator_low_order_numpy(U, r_faces, dz, cell_volumes):
    """Piecewise-constant Rusanov finite-volume RHS (positivity baseline).

    This is the low-order member of the same conservative flux family: same
    Euler fluxes, local Rusanov wave speed, reflecting/outflow boundaries,
    cylindrical face areas/volumes, and geometric radial source.  Only MUSCL
    reconstruction is suppressed.
    """
    nr, nz = U.shape[1:]
    UL_r = np.empty((4, nr + 1, nz), dtype=float)
    UR_r = np.empty_like(UL_r)
    UL_z = np.empty((4, nr, nz + 1), dtype=float)
    UR_z = np.empty_like(UL_z)

    UL_r[:, 1:-1, :] = U[:, :-1, :]
    UR_r[:, 1:-1, :] = U[:, 1:, :]
    axis_ghost = U[:, 0, :].copy()
    axis_ghost[1] *= -1.0
    UL_r[:, 0, :] = axis_ghost
    UR_r[:, 0, :] = U[:, 0, :]
    UL_r[:, -1, :] = U[:, -1, :]
    UR_r[:, -1, :] = U[:, -1, :]

    UL_z[:, :, 1:-1] = U[:, :, :-1]
    UR_z[:, :, 1:-1] = U[:, :, 1:]
    ground_ghost = U[:, :, 0].copy()
    ground_ghost[2] *= -1.0
    UL_z[:, :, 0] = ground_ghost
    UR_z[:, :, 0] = U[:, :, 0]
    UL_z[:, :, -1] = U[:, :, -1]
    UR_z[:, :, -1] = U[:, :, -1]

    Fr = rusanov_flux(UL_r, UR_r, "r")
    Fz = rusanov_flux(UL_z, UR_z, "z")
    radial_areas = (2.0 * np.pi * r_faces * dz)[None, :, None]
    axial_areas = np.pi * (r_faces[1:]**2 - r_faces[:-1]**2)
    radial_net = (Fr[:, 1:, :] * radial_areas[:, 1:, :]
                  - Fr[:, :-1, :] * radial_areas[:, :-1, :])
    vertical_net = (Fz[:, :, 1:] * axial_areas[None, :, None]
                    - Fz[:, :, :-1] * axial_areas[None, :, None])
    rhs = -(radial_net + vertical_net) / cell_volumes[None, :, :]

    _, _, _, pressure = primitive_from_conserved(U)
    r_sum = r_faces[1:] + r_faces[:-1]
    rhs[1] += 2.0 * pressure / np.maximum(r_sum[:, None], 1.0e-30)

    outer_area = 2.0 * np.pi * r_faces[-1] * dz
    mass_rate = (np.sum(Fr[0, -1, :]) * outer_area
                 + np.sum(Fz[0, :, -1] * axial_areas))
    energy_rate = (np.sum(Fr[3, -1, :]) * outer_area
                   + np.sum(Fz[3, :, -1] * axial_areas))
    return rhs, float(mass_rate), float(energy_rate)


def low_order_spatial_operator(U, r_faces, dz, cell_volumes):
    """Return the piecewise-constant Rusanov positivity-baseline operator."""
    return _spatial_operator_low_order_numpy(U, r_faces, dz, cell_volumes)


def maximum_admissible_blend(U_low, U_high):
    """Largest global alpha for U=U_low+alpha*(U_high-U_low) to stay admissible.

    Because the ideal-gas Euler admissible set (rho>0, internal energy>0) is
    convex, bisection on this line segment gives a conservative flux-blend
    coefficient.  A single alpha is used for every face in a stage, so shared
    face fluxes remain equal-and-opposite and the finite-volume update remains
    conservative.  This is intentionally a robust global fallback, not a
    per-cell repair of the conserved state.
    """
    if not is_admissible_state(U_low):
        raise ValueError("Low-order Rusanov candidate is not admissible.")
    if is_admissible_state(U_high):
        return U_high.copy(), 1.0

    lo = 0.0
    hi = 1.0
    delta = U_high - U_low
    for _ in range(FLUX_BLEND_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        trial = U_low + mid * delta
        if is_admissible_state(trial):
            lo = mid
        else:
            hi = mid
    alpha = lo
    if alpha > 0.0:
        alpha *= FLUX_BLEND_SAFETY_FACTOR
    candidate = U_low + alpha * delta
    if not is_admissible_state(candidate):
        alpha = 0.0
        candidate = U_low.copy()
    return candidate, float(alpha)


def maximum_admissible_low_order_dt(U, rhs_low, dt_requested):
    """Find a safe FE dt for the low-order baseline by bisection."""
    lo = 0.0
    hi = float(dt_requested)
    for _ in range(FLUX_BLEND_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        trial = U + mid * rhs_low
        if is_admissible_state(trial):
            lo = mid
        else:
            hi = mid
    # The retry controller applies CFL_RETRY_SHRINK once after receiving this
    # bound; do not apply the safety factor a second time here.
    return float(lo)


@njit(inline="always")
def _minmod3_scalar(a, b, c):
    if a > 0.0 and b > 0.0 and c > 0.0:
        return min(a, b, c)
    if a < 0.0 and b < 0.0 and c < 0.0:
        return max(a, b, c)
    return 0.0


@njit(inline="always")
def _limited_slope_scalar(qm, qc, qp, dxm, dxp, limiter_code):
    dm = (qc - qm) / dxm
    dp = (qp - qc) / dxp
    dc = (qp - qm) / (dxm + dxp)
    if limiter_code == 0:  # MC: minmod(2*dm, dc, 2*dp)
        return _minmod3_scalar(2.0 * dm, dc, 2.0 * dp)
    # minmod: minmod(dm, dp, dc)
    return _minmod3_scalar(dm, dp, dc)


@njit(inline="always")
def _rusanov_r_primitive(rl, url, uzl, pl, rr, urr, uzr, pr, gamma_value):
    el = pl / (gamma_value - 1.0) + 0.5 * rl * (url * url + uzl * uzl)
    er = pr / (gamma_value - 1.0) + 0.5 * rr * (urr * urr + uzr * uzr)
    fl0 = rl * url
    fr0 = rr * urr
    fl1 = rl * url * url + pl
    fr1 = rr * urr * urr + pr
    fl2 = rl * url * uzl
    fr2 = rr * urr * uzr
    fl3 = (el + pl) * url
    fr3 = (er + pr) * urr
    cl = np.sqrt(max(gamma_value * pl / rl, 1.0e-20))
    cr = np.sqrt(max(gamma_value * pr / rr, 1.0e-20))
    amax = max(abs(url) + cl, abs(urr) + cr)
    return (0.5 * (fl0 + fr0) - 0.5 * amax * (rr - rl),
            0.5 * (fl1 + fr1) - 0.5 * amax * (rr * urr - rl * url),
            0.5 * (fl2 + fr2) - 0.5 * amax * (rr * uzr - rl * uzl),
            0.5 * (fl3 + fr3) - 0.5 * amax * (er - el))


@njit(inline="always")
def _rusanov_z_primitive(rl, url, uzl, pl, rr, urr, uzr, pr, gamma_value):
    el = pl / (gamma_value - 1.0) + 0.5 * rl * (url * url + uzl * uzl)
    er = pr / (gamma_value - 1.0) + 0.5 * rr * (urr * urr + uzr * uzr)
    fl0 = rl * uzl
    fr0 = rr * uzr
    fl1 = rl * url * uzl
    fr1 = rr * urr * uzr
    fl2 = rl * uzl * uzl + pl
    fr2 = rr * uzr * uzr + pr
    fl3 = (el + pl) * uzl
    fr3 = (er + pr) * uzr
    cl = np.sqrt(max(gamma_value * pl / rl, 1.0e-20))
    cr = np.sqrt(max(gamma_value * pr / rr, 1.0e-20))
    amax = max(abs(uzl) + cl, abs(uzr) + cr)
    return (0.5 * (fl0 + fr0) - 0.5 * amax * (rr - rl),
            0.5 * (fl1 + fr1) - 0.5 * amax * (rr * urr - rl * url),
            0.5 * (fl2 + fr2) - 0.5 * amax * (rr * uzr - rl * uzl),
            0.5 * (fl3 + fr3) - 0.5 * amax * (er - el))


@njit
def _spatial_operator_numba(U, r_faces, z_centers, dz, cell_volumes,
                            gamma_value, limiter_code):
    """Fused loop implementation of the same MUSCL + Rusanov finite-volume RHS.

    It performs no fast-math transformations and retains the NumPy reference
    method's reconstruction, limiter, positivity scaling, flux, and geometry.
    """
    nr = U.shape[1]
    nz = U.shape[2]
    W = np.empty((4, nr, nz), dtype=np.float64)
    r_centroid = np.empty(nr, dtype=np.float64)
    slope_r = np.zeros((4, nr, nz), dtype=np.float64)
    slope_z = np.zeros((4, nr, nz), dtype=np.float64)

    for i in range(nr):
        rin = r_faces[i]
        rout = r_faces[i + 1]
        r_centroid[i] = (2.0 / 3.0) * (rout**3 - rin**3) / max(rout**2 - rin**2, 1.0e-300)
        for j in range(nz):
            rho = max(U[0, i, j], 1.0e-12)
            ur = U[1, i, j] / rho
            uz = U[2, i, j] / rho
            kinetic = 0.5 * rho * (ur * ur + uz * uz)
            internal = max(U[3, i, j] - kinetic, 1.0e-12)
            pressure = (gamma_value - 1.0) * internal
            W[0, i, j] = rho
            W[1, i, j] = ur
            W[2, i, j] = uz
            W[3, i, j] = pressure

    # Radial and vertical limited primitive slopes, including reflected ghosts.
    for i in range(nr):
        for j in range(nz):
            if i == 0:
                dxm = 2.0 * r_centroid[0]
            else:
                dxm = r_centroid[i] - r_centroid[i - 1]
            if i == nr - 1:
                dxp = r_centroid[nr - 1] - r_centroid[nr - 2]
            else:
                dxp = r_centroid[i + 1] - r_centroid[i]
            for c in range(4):
                if i == 0:
                    qm = -W[c, 0, j] if c == 1 else W[c, 0, j]
                else:
                    qm = W[c, i - 1, j]
                qc = W[c, i, j]
                qp = W[c, i + 1, j] if i < nr - 1 else W[c, i, j]
                slope_r[c, i, j] = _limited_slope_scalar(qm, qc, qp, dxm, dxp, limiter_code)

            if j == 0:
                dzm = 2.0 * z_centers[0]
            else:
                dzm = z_centers[j] - z_centers[j - 1]
            if j == nz - 1:
                dzp = z_centers[nz - 1] - z_centers[nz - 2]
            else:
                dzp = z_centers[j + 1] - z_centers[j]
            for c in range(4):
                if j == 0:
                    qm = -W[c, i, 0] if c == 2 else W[c, i, 0]
                else:
                    qm = W[c, i, j - 1]
                qc = W[c, i, j]
                qp = W[c, i, j + 1] if j < nz - 1 else W[c, i, j]
                slope_z[c, i, j] = _limited_slope_scalar(qm, qc, qp, dzm, dzp, limiter_code)

    # Positivity-preserving common slope scaling for rho and pressure.
    rho_floor = 1.0e-12
    pressure_floor = (gamma_value - 1.0) * 1.0e-10
    for i in range(nr):
        dr_minus = r_faces[i] - r_centroid[i]
        dr_plus = r_faces[i + 1] - r_centroid[i]
        for j in range(nz):
            theta = 1.0
            for offset in (slope_r[0, i, j] * dr_minus,
                           slope_r[0, i, j] * dr_plus,
                           slope_z[0, i, j] * (-0.5 * dz),
                           slope_z[0, i, j] * (0.5 * dz)):
                predicted = W[0, i, j] + offset
                if predicted < rho_floor:
                    denom = W[0, i, j] - predicted
                    if denom > 0.0:
                        theta = min(theta, (W[0, i, j] - rho_floor) / denom)
            for offset in (slope_r[3, i, j] * dr_minus,
                           slope_r[3, i, j] * dr_plus,
                           slope_z[3, i, j] * (-0.5 * dz),
                           slope_z[3, i, j] * (0.5 * dz)):
                predicted = W[3, i, j] + offset
                if predicted < pressure_floor:
                    denom = W[3, i, j] - predicted
                    if denom > 0.0:
                        theta = min(theta, (W[3, i, j] - pressure_floor) / denom)
            theta = min(max(theta, 0.0), 1.0)
            if theta < 1.0:
                for c in range(4):
                    slope_r[c, i, j] *= theta
                    slope_z[c, i, j] *= theta

    Fr = np.empty((4, nr + 1, nz), dtype=np.float64)
    Fz = np.empty((4, nr, nz + 1), dtype=np.float64)

    # Radial face fluxes.
    for iface in range(nr + 1):
        for j in range(nz):
            if iface == 0:
                i = 0
                off = r_faces[0] - r_centroid[0]
                rr = W[0, i, j] + slope_r[0, i, j] * off
                ur = W[1, i, j] + slope_r[1, i, j] * off
                uz = W[2, i, j] + slope_r[2, i, j] * off
                pp = W[3, i, j] + slope_r[3, i, j] * off
                flux = _rusanov_r_primitive(rr, -ur, uz, pp, rr, ur, uz, pp, gamma_value)
            elif iface == nr:
                i = nr - 1
                off = r_faces[nr] - r_centroid[i]
                rr = W[0, i, j] + slope_r[0, i, j] * off
                ur = W[1, i, j] + slope_r[1, i, j] * off
                uz = W[2, i, j] + slope_r[2, i, j] * off
                pp = W[3, i, j] + slope_r[3, i, j] * off
                flux = _rusanov_r_primitive(rr, ur, uz, pp, rr, ur, uz, pp, gamma_value)
            else:
                il = iface - 1
                ir = iface
                offl = r_faces[iface] - r_centroid[il]
                offr = r_faces[iface] - r_centroid[ir]
                rl = W[0, il, j] + slope_r[0, il, j] * offl
                url = W[1, il, j] + slope_r[1, il, j] * offl
                uzl = W[2, il, j] + slope_r[2, il, j] * offl
                pl = W[3, il, j] + slope_r[3, il, j] * offl
                rr = W[0, ir, j] + slope_r[0, ir, j] * offr
                urr = W[1, ir, j] + slope_r[1, ir, j] * offr
                uzr = W[2, ir, j] + slope_r[2, ir, j] * offr
                pr = W[3, ir, j] + slope_r[3, ir, j] * offr
                flux = _rusanov_r_primitive(rl, url, uzl, pl, rr, urr, uzr, pr, gamma_value)
            for c in range(4):
                Fr[c, iface, j] = flux[c]

    # Vertical face fluxes.
    for i in range(nr):
        for jface in range(nz + 1):
            if jface == 0:
                j = 0
                off = 0.0 - z_centers[0]
                rr = W[0, i, j] + slope_z[0, i, j] * off
                ur = W[1, i, j] + slope_z[1, i, j] * off
                uz = W[2, i, j] + slope_z[2, i, j] * off
                pp = W[3, i, j] + slope_z[3, i, j] * off
                flux = _rusanov_z_primitive(rr, ur, -uz, pp, rr, ur, uz, pp, gamma_value)
            elif jface == nz:
                j = nz - 1
                off = 0.5 * dz
                rr = W[0, i, j] + slope_z[0, i, j] * off
                ur = W[1, i, j] + slope_z[1, i, j] * off
                uz = W[2, i, j] + slope_z[2, i, j] * off
                pp = W[3, i, j] + slope_z[3, i, j] * off
                flux = _rusanov_z_primitive(rr, ur, uz, pp, rr, ur, uz, pp, gamma_value)
            else:
                jl = jface - 1
                jr = jface
                offl = 0.5 * dz
                offr = -0.5 * dz
                rl = W[0, i, jl] + slope_z[0, i, jl] * offl
                url = W[1, i, jl] + slope_z[1, i, jl] * offl
                uzl = W[2, i, jl] + slope_z[2, i, jl] * offl
                pl = W[3, i, jl] + slope_z[3, i, jl] * offl
                rr = W[0, i, jr] + slope_z[0, i, jr] * offr
                urr = W[1, i, jr] + slope_z[1, i, jr] * offr
                uzr = W[2, i, jr] + slope_z[2, i, jr] * offr
                pr = W[3, i, jr] + slope_z[3, i, jr] * offr
                flux = _rusanov_z_primitive(rl, url, uzl, pl, rr, urr, uzr, pr, gamma_value)
            for c in range(4):
                Fz[c, i, jface] = flux[c]

    rhs = np.empty_like(U)
    mass_out_rate = 0.0
    energy_out_rate = 0.0
    outer_radial_area = 2.0 * np.pi * r_faces[nr] * dz
    for j in range(nz):
        mass_out_rate += Fr[0, nr, j] * outer_radial_area
        energy_out_rate += Fr[3, nr, j] * outer_radial_area
    for i in range(nr):
        axial_area = np.pi * (r_faces[i + 1]**2 - r_faces[i]**2)
        mass_out_rate += Fz[0, i, nz] * axial_area
        energy_out_rate += Fz[3, i, nz] * axial_area
        r_mid_sum = max(r_faces[i + 1] + r_faces[i], 1.0e-30)
        for j in range(nz):
            volume = cell_volumes[i, j]
            for c in range(4):
                radial_net = (Fr[c, i + 1, j] * (2.0 * np.pi * r_faces[i + 1] * dz)
                              - Fr[c, i, j] * (2.0 * np.pi * r_faces[i] * dz))
                vertical_net = (Fz[c, i, j + 1] * axial_area - Fz[c, i, j] * axial_area)
                rhs[c, i, j] = -(radial_net + vertical_net) / volume
            rhs[1, i, j] += 2.0 * W[3, i, j] / r_mid_sum

    return rhs, mass_out_rate, energy_out_rate


def spatial_operator(U, r_faces, dz, cell_volumes, z_centers=None):
    """MUSCL-Rusanov RHS, with optional compiled kernel and identical NumPy fallback."""
    if z_centers is None:
        z_centers = np.linspace(0.5 * dz, (U.shape[2] - 0.5) * dz, U.shape[2])
    if USE_NUMBA and NUMBA_AVAILABLE:
        limiter_code = 0 if SLOPE_LIMITER.upper() == "MC" else 1
        rhs, mass_rate, energy_rate = _spatial_operator_numba(
            U, r_faces, z_centers, dz, cell_volumes, gamma, limiter_code
        )
        return rhs, float(mass_rate), float(energy_rate)
    return _spatial_operator_numpy(U, r_faces, z_centers, dz, cell_volumes)


def _forward_euler_stage_with_positivity(U, dt, r_faces, dz, cell_volumes,
                                          z_centers=None, r_centers=None):
    """One FE stage with a conservative high-/low-order flux positivity blend."""
    rhs_high, mass_high, energy_high = spatial_operator(
        U, r_faces, dz, cell_volumes, z_centers=z_centers
    )
    U_high = U + dt * rhs_high
    high_mask, _ = state_admissibility(U_high)
    if np.all(high_mask):
        return U_high, mass_high, energy_high, 1.0, False, 0
    high_order_bad_cells = int(np.count_nonzero(~high_mask))

    # High-order MUSCL update is inadmissible. Build the low-order, piecewise-
    # constant Rusanov update using exactly the same physical fluxes/source.
    rhs_low, mass_low, energy_low = low_order_spatial_operator(
        U, r_faces, dz, cell_volumes
    )
    U_low = U + dt * rhs_low
    if not is_admissible_state(U_low):
        safe_dt = maximum_admissible_low_order_dt(U, rhs_low, dt)
        diag = state_admissibility_diagnostics(U_low, r_centers, z_centers)
        raise StageCFLViolation(
            safe_dt,
            "Low-order piecewise-constant Rusanov update violates density/internal-energy positivity.",
            diag,
        )

    U_blended, alpha = maximum_admissible_blend(U_low, U_high)
    # Blend the physical boundary flux rates with the same alpha.  This keeps
    # the mass/energy budget consistent with the fluxes used in the state update.
    if alpha <= 0.0:
        # Avoid 0*NaN if a failed high-order candidate generated non-finite
        # boundary fluxes; alpha=0 means the accepted update is entirely low-order.
        mass_rate = mass_low
        energy_rate = energy_low
    else:
        mass_rate = mass_low + alpha * (mass_high - mass_low)
        energy_rate = energy_low + alpha * (energy_high - energy_low)
    return U_blended, mass_rate, energy_rate, alpha, True, high_order_bad_cells


def advance_one_step(U, dt, r_faces, dz, cell_volumes, dr=None,
                     return_budget=False, z_centers=None, r_centers=None):
    """Advance one step with MUSCL-Rusanov + SSP-RK2 and conservative positivity.

    Each forward-Euler stage first attempts the full MC-MUSCL update.  If that
    candidate leaves the ideal-gas admissible set, the stage is recomputed from
    the same cell-average state with piecewise-constant Rusanov fluxes.  The
    shared face fluxes are then blended by one global coefficient alpha chosen
    so the conservative update remains admissible.  If even the low-order
    update is inadmissible, the time step is rejected and a safe smaller dt is
    suggested.  There are no post-update cell-local density, momentum, or
    energy repairs.

    SSP-RK2 stages:
      U1      = FE_positive(U^n, dt)
      U^{n+1} = 0.5*U^n + 0.5*FE_positive(U1, dt)

    The two boundary-flux evaluations retain the SSP-RK2 half-step weights.
    """
    if dr is None:
        dr = float(r_faces[1] - r_faces[0])

    U1, mass_rate0, energy_rate0, alpha0, used_low0, bad_cells0 = (
        _forward_euler_stage_with_positivity(
            U, dt, r_faces, dz, cell_volumes,
            z_centers=z_centers, r_centers=r_centers
        )
    )

    # SSP-RK2 requires the second forward-Euler stage to satisfy the CFL
    # restriction evaluated on the intermediate state.
    stage1_dt_limit = stable_timestep(U1, dr, dz)
    if dt > stage1_dt_limit * (1.0 + 1.0e-12):
        diag = cfl_state_diagnostics(U1, dr, dz, r_centers, z_centers)
        raise StageCFLViolation(
            stage1_dt_limit,
            "SSP-RK2 intermediate state requires a smaller CFL time step.",
            diag,
        )

    U2_fe, mass_rate1, energy_rate1, alpha1, used_low1, bad_cells1 = (
        _forward_euler_stage_with_positivity(
            U1, dt, r_faces, dz, cell_volumes,
            z_centers=z_centers, r_centers=r_centers
        )
    )
    U2 = 0.5 * U + 0.5 * U2_fe

    if not is_admissible_state(U2):
        # Convexity says this should not occur if U and U2_fe are admissible;
        # treat it as a fatal numerical inconsistency rather than repairing it.
        diag = state_admissibility_diagnostics(U2, r_centers, z_centers)
        raise RuntimeError(
            "SSP-RK2 convex combination unexpectedly left the admissible set: "
            + format_state_diagnostics(diag)
        )

    if return_budget:
        boundary_mass_out = 0.5 * dt * (mass_rate0 + mass_rate1)
        boundary_energy_out = 0.5 * dt * (energy_rate0 + energy_rate1)
        limited_stages = int(used_low0) + int(used_low1)
        alphas = [alpha0, alpha1]
        return U2, {
            "boundary_mass_out_kg": float(boundary_mass_out),
            "boundary_energy_out_J": float(boundary_energy_out),
            # Compatibility fields retained.  The conservative flux correction
            # makes no post-update floor additions; these terms are identically 0.
            "mass_floor_added_kg": 0.0,
            "energy_floor_added_J": 0.0,
            "stage1_cfl_limit_s": float(stage1_dt_limit),
            "flux_limited_stages": int(limited_stages),
            "low_order_fallbacks": int(limited_stages),
            "flux_blend_alpha_stage1": float(alpha0),
            "flux_blend_alpha_stage2": float(alpha1),
            "minimum_flux_blend_alpha": float(min(alphas)),
            "high_order_inadmissible_cells": int(bad_cells0 + bad_cells1),
            "radial_momentum_correction_kg_m_s": 0.0,
            "vertical_momentum_correction_kg_m_s": 0.0,
        }
    return U2


# =============================================================================
# SHOCK AND VOLUME DIAGNOSTICS
# =============================================================================

def shock_geometry(r_centers, z_centers, rho):
    """
    Estimate the outer shock radius along rays from the burst center.

    The shock is identified by rho > rho0*(1 + SHOCK_DENSITY_EXCESS).
    The result is threshold- and resolution-dependent; it is a diagnostic,
    not an exact mathematical shock location.
    """
    rr, zz = np.meshgrid(r_centers, z_centers, indexing="ij")
    distance = np.sqrt(rr**2 + (zz - HOB)**2)
    theta = np.arctan2(rr, zz - HOB)  # polar angle from upward vertical
    mask = rho > rho0 * (1.0 + SHOCK_DENSITY_EXCESS)

    edges = np.linspace(0.0, np.pi, N_ANGLE_BINS + 1)
    theta_centers = 0.5 * (edges[:-1] + edges[1:])
    r_shock = np.zeros(N_ANGLE_BINS)

    bin_index = np.digitize(theta[mask], edges) - 1
    dist_values = distance[mask]
    valid = (bin_index >= 0) & (bin_index < N_ANGLE_BINS)
    np.maximum.at(r_shock, bin_index[valid], dist_values[valid])

    # A downward ray terminates at the reflecting ground plane.
    downward = np.cos(theta_centers) < 0.0
    ground_distance = np.full_like(theta_centers, np.inf)
    ground_distance[downward] = HOB / np.maximum(-np.cos(theta_centers[downward]), 1.0e-14)

    # If the shock reaches the ground, the physical domain is truncated there.
    r_limited = np.minimum(r_shock, ground_distance)

    # Solid-angle integration: V = (2*pi/3) integral R(theta)^3 sin(theta) dtheta.
    dtheta = edges[1] - edges[0]
    volume = (2.0 * np.pi / 3.0) * np.sum(
        r_limited**3 * np.sin(theta_centers) * dtheta
    )

    return theta_centers, r_shock, volume


def total_energy(U, cell_volumes):
    """Explosion energy above the uniform ambient internal-energy baseline."""
    ambient_energy_density = p0 / (gamma - 1.0)
    return float(np.sum((U[3] - ambient_energy_density) * cell_volumes))


def total_mass(U, cell_volumes):
    """Total gas mass in the computational domain [kg]."""
    return float(np.sum(U[0] * cell_volumes))


def rankine_hugoniot_ratios(shock_mach):
    """Ideal-gas normal-shock density and pressure ratios for upstream Mach M."""
    mach = np.asarray(shock_mach, dtype=float)
    m2 = np.maximum(mach**2, 1.0)
    density_ratio = ((gamma + 1.0) * m2) / ((gamma - 1.0) * m2 + 2.0)
    pressure_ratio = (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)
    return density_ratio, pressure_ratio


def initialize_gauges(r_centers, z_centers, r_max, z_max):
    """Map requested physical gauge coordinates to nearest cell centers."""
    gauges = []
    names = set()
    for item in GAUGE_LOCATIONS:
        if len(item) != 3:
            raise ValueError("Each GAUGE_LOCATIONS entry must be (name, r, z).")
        name, r_requested, z_requested = item
        name = str(name)
        if not name or name in names:
            raise ValueError("Gauge names must be nonempty and unique.")
        names.add(name)
        r_requested = float(r_requested)
        z_requested = float(HOB if z_requested is None else z_requested)
        if not (0.0 <= r_requested <= r_max and 0.0 <= z_requested <= z_max):
            print(
                f"WARNING: skipping gauge {name!r} at (r={r_requested:g}, "
                f"z={z_requested:g}) m because it is outside the domain."
            )
            continue
        i = int(np.argmin(np.abs(r_centers - r_requested)))
        j = int(np.argmin(np.abs(z_centers - z_requested)))
        gauges.append({
            "name": name,
            "r_requested_m": r_requested,
            "z_requested_m": z_requested,
            "i": i,
            "j": j,
            "r_sampled_m": float(r_centers[i]),
            "z_sampled_m": float(z_centers[j]),
        })
    return gauges


def sample_gauge_pressures(U, gauges, time_s, gauge_history):
    """Record cell-centered pressure at each virtual gauge."""
    gauge_history["time_s"].append(float(time_s))
    for gauge in gauges:
        i, j = gauge["i"], gauge["j"]
        rho_cell = max(float(U[0, i, j]), 1.0e-12)
        ur_cell = float(U[1, i, j]) / rho_cell
        uz_cell = float(U[2, i, j]) / rho_cell
        kinetic = 0.5 * rho_cell * (ur_cell**2 + uz_cell**2)
        internal = max(float(U[3, i, j]) - kinetic, 1.0e-12)
        pressure_cell = (gamma - 1.0) * internal
        gauge_history["pressure_Pa"][gauge["name"]].append(pressure_cell)


def gauge_summary(gauges, gauge_history):
    """Calculate peak overpressure, arrival time, and positive-phase impulse."""
    times = np.asarray(gauge_history["time_s"], dtype=float)
    summaries = []
    for gauge in gauges:
        pressure = np.asarray(
            gauge_history["pressure_Pa"][gauge["name"]], dtype=float
        )
        overpressure = pressure - p0
        positive = np.maximum(overpressure, 0.0)
        peak_overpressure = float(np.max(overpressure)) if pressure.size else np.nan
        min_overpressure = float(np.min(overpressure)) if pressure.size else np.nan
        impulse = float(np.trapz(positive, times)) if times.size > 1 else 0.0

        arrival_time = np.nan
        crossed = np.flatnonzero(overpressure >= GAUGE_ARRIVAL_THRESHOLD_PA)
        if crossed.size:
            k = int(crossed[0])
            if k == 0:
                arrival_time = float(times[0])
            else:
                y0 = overpressure[k - 1]
                y1 = overpressure[k]
                if y1 != y0:
                    fraction = (GAUGE_ARRIVAL_THRESHOLD_PA - y0) / (y1 - y0)
                    arrival_time = float(times[k - 1] + fraction * (times[k] - times[k - 1]))
                else:
                    arrival_time = float(times[k])

        row = dict(gauge)
        row.update({
            "peak_overpressure_Pa": max(peak_overpressure, 0.0),
            "minimum_overpressure_Pa": min_overpressure,
            "positive_phase_impulse_Pa_s": impulse,
            "arrival_time_s": arrival_time,
        })
        summaries.append(row)
    return summaries


def write_gauge_outputs(output_dir, gauges, gauge_history, summaries):
    """Write full pressure histories and compact gauge summary CSVs/plots."""
    output_dir = Path(output_dir)
    times = np.asarray(gauge_history["time_s"], dtype=float)

    history_path = output_dir / "pressure_gauges.csv"
    fieldnames = ["time_s"]
    for gauge in gauges:
        name = gauge["name"]
        fieldnames.extend([f"{name}_pressure_Pa", f"{name}_overpressure_Pa"])
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k, time_s in enumerate(times):
            row = {"time_s": float(time_s)}
            for gauge in gauges:
                name = gauge["name"]
                pressure = gauge_history["pressure_Pa"][name][k]
                row[f"{name}_pressure_Pa"] = pressure
                row[f"{name}_overpressure_Pa"] = pressure - p0
            writer.writerow(row)

    summary_path = output_dir / "gauge_summary.csv"
    summary_fields = [
        "name", "r_requested_m", "z_requested_m", "r_sampled_m", "z_sampled_m",
        "i", "j", "peak_overpressure_Pa", "minimum_overpressure_Pa",
        "positive_phase_impulse_Pa_s", "arrival_time_s",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in summary_fields})

    if gauges and times.size > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        positive_time = times > 0.0
        ax = axes[0]
        for gauge in gauges:
            name = gauge["name"]
            pressure = np.asarray(gauge_history["pressure_Pa"][name])
            ax.plot(times[positive_time], (pressure - p0)[positive_time], label=name)
        ax.axhline(GAUGE_ARRIVAL_THRESHOLD_PA, color="k", ls="--", lw=1,
                   label="Arrival threshold")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=max(GAUGE_ARRIVAL_THRESHOLD_PA, 1.0))
        ax.set_xlabel("Simulation time [s]")
        ax.set_ylabel("Overpressure, p - p0 [Pa]")
        ax.set_title("Virtual-gauge pressure histories")
        ax.grid(True, which="both", ls=":")
        ax.legend(fontsize=8)

        surface = [row for row in summaries if np.isclose(row["z_requested_m"], 0.0)]
        ax = axes[1]
        if surface:
            x = [row["r_requested_m"] for row in surface]
            y = [row["peak_overpressure_Pa"] for row in surface]
            ax.plot(x, y, "o-")
        ax.set_xlabel("Requested ground-gauge radius, r [m]")
        ax.set_ylabel("Peak overpressure [Pa]")
        ax.set_title("Ground peak overpressure")
        ax.grid(True, ls=":")

        ax = axes[2]
        if surface:
            x = [row["r_requested_m"] for row in surface]
            y = [row["positive_phase_impulse_Pa_s"] for row in surface]
            ax.plot(x, y, "o-")
        ax.set_xlabel("Requested ground-gauge radius, r [m]")
        ax.set_ylabel("Positive-phase impulse [Pa s]")
        ax.set_title("Ground positive-phase impulse")
        ax.grid(True, ls=":")

        fig.suptitle(
            "Virtual pressure gauges (ground gauges sample the first cell above z=0)",
            fontsize=13, fontweight="bold"
        )
        fig.savefig(output_dir / "pressure_gauges.png", dpi=SNAPSHOT_DPI)
        plt.close(fig)


def _bilinear_sample(field, r_query, z_query, r_centers, z_centers):
    """Bilinearly interpolate a cell-centred field; clamp at physical boundaries."""
    rq = np.clip(np.asarray(r_query, dtype=float), r_centers[0], r_centers[-1])
    zq = np.clip(np.asarray(z_query, dtype=float), z_centers[0], z_centers[-1])
    ir = np.searchsorted(r_centers, rq, side="right") - 1
    iz = np.searchsorted(z_centers, zq, side="right") - 1
    ir = np.clip(ir, 0, len(r_centers) - 2)
    iz = np.clip(iz, 0, len(z_centers) - 2)
    r0 = r_centers[ir]
    r1 = r_centers[ir + 1]
    z0 = z_centers[iz]
    z1 = z_centers[iz + 1]
    fr = np.clip((rq - r0) / np.maximum(r1 - r0, 1.0e-30), 0.0, 1.0)
    fz = np.clip((zq - z0) / np.maximum(z1 - z0, 1.0e-30), 0.0, 1.0)
    f00 = field[ir, iz]
    f10 = field[ir + 1, iz]
    f01 = field[ir, iz + 1]
    f11 = field[ir + 1, iz + 1]
    return ((1.0 - fr) * (1.0 - fz) * f00
            + fr * (1.0 - fz) * f10
            + (1.0 - fr) * fz * f01
            + fr * fz * f11)


def _ray_sampling_limit(theta, r_centers, z_centers):
    """Maximum ray distance within the sampled cell-centre domain."""
    st = float(np.sin(theta))
    ct = float(np.cos(theta))
    limits = []
    if st > 1.0e-12:
        limits.append(r_centers[-1] / st)
    if ct > 1.0e-12:
        limits.append((z_centers[-1] - HOB) / ct)
    elif ct < -1.0e-12:
        limits.append((HOB - z_centers[0]) / (-ct))
    positive = [v for v in limits if v > 0.0]
    return max(0.0, min(positive)) if positive else 0.0


def _detect_ray_front(state, theta, r_centers, z_centers, dr, dz):
    """Find an outer density-threshold crossing and refine it with max |grad rho|."""
    s_max = _ray_sampling_limit(theta, r_centers, z_centers)
    ds = SHOCK_PROFILE_DS_FACTOR * min(dr, dz)
    if s_max <= 2.0 * ds or ds <= 0.0:
        return np.nan, np.nan, None
    s = np.arange(0.0, s_max + 0.25 * ds, ds)
    st = np.sin(theta)
    ct = np.cos(theta)
    rq = s * st
    zq = HOB + s * ct
    rho_line = _bilinear_sample(state["rho"], rq, zq, r_centers, z_centers)
    threshold = rho0 * (1.0 + SHOCK_DENSITY_EXCESS)
    crossing = np.flatnonzero((rho_line[:-1] >= threshold) & (rho_line[1:] < threshold))
    if crossing.size == 0:
        return np.nan, np.nan, (s, rho_line)

    # Select the outermost compression front along this ray.
    icross = int(crossing[-1])
    frac = (rho_line[icross] - threshold) / max(rho_line[icross] - rho_line[icross + 1], 1.0e-30)
    threshold_radius = float(s[icross] + np.clip(frac, 0.0, 1.0) * ds)

    # Refine front location to the steepest outward density drop close to the
    # threshold crossing. This identifies the shock's gradient peak rather
    # than selecting the maximum state in a broad region behind it.
    half_window = SHOCK_PROFILE_MAX_GRADIENT_WINDOW_CELLS * np.hypot(dr, dz)
    gradients = -(rho_line[1:] - rho_line[:-1]) / ds
    midpoints = 0.5 * (s[1:] + s[:-1])
    candidate = np.flatnonzero(np.abs(midpoints - threshold_radius) <= half_window)
    if candidate.size:
        ig = int(candidate[np.argmax(gradients[candidate])])
        if gradients[ig] > 0.0:
            front_radius = float(midpoints[ig])
        else:
            front_radius = threshold_radius
    else:
        front_radius = threshold_radius
    return front_radius, float(np.max(gradients[candidate])) if candidate.size else np.nan, (s, rho_line)


def _front_radii_for_snapshots(snapshot_states, r_centers, z_centers, dr, dz):
    """Detect shock radius on each fixed angular ray for all saved states."""
    theta = snapshot_states[0]["theta"]
    radii = np.full((len(snapshot_states), len(theta)), np.nan, dtype=float)
    gradients = np.full_like(radii, np.nan)
    for it, state in enumerate(snapshot_states):
        for k, angle in enumerate(theta):
            radius, gradient, _ = _detect_ray_front(
                state, float(angle), r_centers, z_centers, dr, dz
            )
            radii[it, k] = radius
            gradients[it, k] = gradient
    return theta, radii, gradients


def _time_derivative_front_radius(snapshot_states, radii, it, ik):
    """Finite-difference radial front speed using the nearest valid time states."""
    valid = np.flatnonzero(np.isfinite(radii[:, ik]))
    if valid.size < 2:
        return np.nan
    position = int(np.searchsorted(valid, it))
    left = valid[position - 1] if position > 0 else None
    right = valid[position] if position < valid.size and valid[position] != it else None
    if right is None and position + 1 < valid.size:
        right = valid[position + 1]
    if left is None and position > 0:
        left = valid[position - 1]
    if left is None:
        # First valid front: forward difference.
        a = it
        b = valid[position + 1] if position + 1 < valid.size else None
    elif right is None:
        a = left
        b = it
    else:
        a = left
        b = right
    if a is None or b is None or a == b:
        return np.nan
    ta = float(snapshot_states[a]["time"])
    tb = float(snapshot_states[b]["time"])
    if tb <= ta:
        return np.nan
    return float((radii[b, ik] - radii[a, ik]) / (tb - ta))


def _angular_front_slope(theta, radii, index):
    """Local least-squares dR/dtheta using valid neighboring rays."""
    valid = np.isfinite(radii)
    indices = np.flatnonzero(valid & (np.abs(np.arange(len(theta)) - index) <= 3))
    if indices.size < 3:
        return np.nan
    coeff = np.polyfit(theta[indices], radii[indices], 1)
    return float(coeff[0])


def _linear_crossing_position(s, values, target, start_index, end_index):
    """Return the first interpolated crossing of target in a directed interval."""
    lo = max(0, int(start_index))
    hi = min(len(values) - 1, int(end_index))
    if hi <= lo:
        return np.nan
    for i in range(lo, hi):
        a = values[i] - target
        b = values[i + 1] - target
        if a == 0.0:
            return float(s[i])
        if a * b <= 0.0:
            frac = a / (a - b) if a != b else 0.0
            return float(s[i] + frac * (s[i + 1] - s[i]))
    return np.nan


def analyze_shock_structure(snapshot_states, r_centers, z_centers,
                            dr, dz, deposition_width):
    """Build local, normal-direction shock-jump and through-front profiles.

    Each angular ray is sampled by bilinear interpolation. The shock is located
    from the outer density-threshold crossing and refined to the local maximum
    outward density gradient. Upstream/downstream states are sampled at fixed
    distances on opposite sides of that front. Shock speed is converted from
    radial-graph motion to a local normal speed using the angular front shape.
    All operations are post-processing; the Euler solution is untouched.
    """
    if not snapshot_states:
        return [], [], []
    theta, radii, front_gradients = _front_radii_for_snapshots(
        snapshot_states, r_centers, z_centers, dr, dz
    )
    h = max(dr, dz)
    offset = SHOCK_PROFILE_SAMPLE_OFFSET_CELLS * h
    minimum_downstream_radius = SHOCK_PROFILE_MIN_POSTSOURCE_FACTOR * deposition_width
    angle_rows = []
    summary_rows = []
    profile_rows = []

    for it, state in enumerate(snapshot_states):
        per_snapshot = []
        for k, angle in enumerate(theta):
            R = radii[it, k]
            status = "front_not_detected"
            row = {
                "time_s": float(state["time"]),
                "theta_deg": float(np.degrees(angle)),
                "front_radius_m": float(R) if np.isfinite(R) else np.nan,
                "front_max_density_gradient_kg_m4": float(front_gradients[it, k]) if np.isfinite(front_gradients[it, k]) else np.nan,
                "normal_sample_offset_m": float(offset),
                "normal_sample_offset_cells": float(SHOCK_PROFILE_SAMPLE_OFFSET_CELLS),
                "radial_front_speed_m_s": np.nan,
                "normal_front_speed_m_s": np.nan,
                "normal_r_component": np.nan,
                "normal_z_component": np.nan,
                "shock_mach_relative_upstream": np.nan,
                "upstream_density_kg_m3": np.nan,
                "downstream_density_kg_m3": np.nan,
                "upstream_pressure_Pa": np.nan,
                "downstream_pressure_Pa": np.nan,
                "upstream_sample_r_m": np.nan,
                "upstream_sample_z_m": np.nan,
                "downstream_sample_r_m": np.nan,
                "downstream_sample_z_m": np.nan,
                "upstream_normal_velocity_m_s": np.nan,
                "downstream_normal_velocity_m_s": np.nan,
                "observed_density_ratio": np.nan,
                "rh_density_ratio": np.nan,
                "observed_pressure_ratio": np.nan,
                "rh_pressure_ratio": np.nan,
                "observed_postshock_normal_velocity_m_s": np.nan,
                "rh_postshock_normal_velocity_m_s": np.nan,
                "shock_thickness_m": np.nan,
                "shock_thickness_cells": np.nan,
                "status": status,
            }
            if not np.isfinite(R):
                angle_rows.append(row)
                continue

            st = float(np.sin(angle))
            ct = float(np.cos(angle))
            # Save the local density/pressure profile even when a formal RH
            # jump cannot yet be evaluated (e.g. the early front is too close
            # to the finite initial deposition region).
            selected_deg = np.asarray(SHOCK_PROFILE_RAY_ANGLES_DEG, dtype=float)
            angle_deg = float(np.degrees(angle))
            if selected_deg.size and np.min(np.abs(selected_deg - angle_deg)) <= (180.0 / N_ANGLE_BINS):
                ds_profile = SHOCK_PROFILE_DS_FACTOR * min(dr, dz)
                half_profile = SHOCK_PROFILE_HALF_WIDTH_CELLS * h
                smax_profile = _ray_sampling_limit(float(angle), r_centers, z_centers)
                s_profile = np.arange(max(0.0, R - half_profile),
                                      min(smax_profile, R + half_profile) + 0.25 * ds_profile,
                                      ds_profile)
                rho_profile = _bilinear_sample(
                    state["rho"], s_profile * st, HOB + s_profile * ct,
                    r_centers, z_centers
                )
                pressure_profile = _bilinear_sample(
                    state["pressure"], s_profile * st, HOB + s_profile * ct,
                    r_centers, z_centers
                )
                for ss, rrho, pp in zip(s_profile, rho_profile, pressure_profile):
                    profile_rows.append({
                        "time_s": float(state["time"]),
                        "theta_deg": angle_deg,
                        "distance_from_front_cells": float((ss - R) / h),
                        "distance_from_front_m": float(ss - R),
                        "density_kg_m3": float(rrho),
                        "pressure_Pa": float(pp),
                        "density_over_ambient": float(rrho / rho0),
                        "pressure_over_ambient": float(pp / p0),
                    })

            radial_speed = _time_derivative_front_radius(snapshot_states, radii, it, k)
            dR_dtheta = _angular_front_slope(theta, radii[it], k)
            if not np.isfinite(radial_speed) or not np.isfinite(dR_dtheta):
                row["status"] = "insufficient_space_time_front_data"
                angle_rows.append(row)
                continue

            norm = np.sqrt(R * R + dR_dtheta * dR_dtheta)
            if norm <= 0.0:
                row["status"] = "invalid_front_normal"
                angle_rows.append(row)
                continue
            nr = (R * st - dR_dtheta * ct) / norm
            nz = (R * ct + dR_dtheta * st) / norm
            normal_speed = radial_speed * R / norm
            row["radial_front_speed_m_s"] = radial_speed
            row["normal_front_speed_m_s"] = normal_speed
            row["normal_r_component"] = nr
            row["normal_z_component"] = nz

            s_down = R - offset
            s_up = R + offset
            if s_down <= minimum_downstream_radius:
                row["status"] = "postshock_sample_in_source_region"
                angle_rows.append(row)
                continue
            s_max = _ray_sampling_limit(float(angle), r_centers, z_centers)
            if s_up >= s_max:
                row["status"] = "insufficient_upstream_domain_clearance"
                angle_rows.append(row)
                continue

            # Sample on either side of the local front along the estimated
            # outward normal, not merely along the radial ray.
            r_front = R * st
            z_front = HOB + R * ct
            r_down, z_down = r_front - offset * nr, z_front - offset * nz
            r_up, z_up = r_front + offset * nr, z_front + offset * nz
            if (r_down < 0.0 or r_up < 0.0
                    or z_down < 0.0 or z_up < 0.0
                    or r_down > r_centers[-1] or r_up > r_centers[-1]
                    or z_down > z_centers[-1] or z_up > z_centers[-1]):
                row["status"] = "normal_sample_intersects_domain_boundary"
                angle_rows.append(row)
                continue
            rho2 = float(_bilinear_sample(state["rho"], r_down, z_down, r_centers, z_centers))
            p2 = float(_bilinear_sample(state["pressure"], r_down, z_down, r_centers, z_centers))
            ur2 = float(_bilinear_sample(state["ur"], r_down, z_down, r_centers, z_centers))
            uz2 = float(_bilinear_sample(state["uz"], r_down, z_down, r_centers, z_centers))
            rho1 = float(_bilinear_sample(state["rho"], r_up, z_up, r_centers, z_centers))
            p1 = float(_bilinear_sample(state["pressure"], r_up, z_up, r_centers, z_centers))
            ur1 = float(_bilinear_sample(state["ur"], r_up, z_up, r_centers, z_centers))
            uz1 = float(_bilinear_sample(state["uz"], r_up, z_up, r_centers, z_centers))
            u1n = ur1 * nr + uz1 * nz
            u2n = ur2 * nr + uz2 * nz
            c1 = float(sound_speed(rho1, p1))
            mach = (normal_speed - u1n) / max(c1, 1.0e-20)
            row.update({
                "upstream_density_kg_m3": rho1,
                "downstream_density_kg_m3": rho2,
                "upstream_sample_r_m": r_up,
                "upstream_sample_z_m": z_up,
                "downstream_sample_r_m": r_down,
                "downstream_sample_z_m": z_down,
                "upstream_pressure_Pa": p1,
                "downstream_pressure_Pa": p2,
                "upstream_normal_velocity_m_s": u1n,
                "downstream_normal_velocity_m_s": u2n,
                "shock_mach_relative_upstream": mach,
            })
            if mach <= 1.0:
                row["status"] = "subsonic_or_noncompressive_front"
                angle_rows.append(row)
                continue

            compression_rh, pressure_rh = rankine_hugoniot_ratios(mach)
            u2n_rh = u1n + (normal_speed - u1n) * (1.0 - 1.0 / compression_rh)
            rho_ratio = rho2 / max(rho1, 1.0e-30)
            pressure_ratio = p2 / max(p1, 1.0e-30)

            # Estimate the 10-90% density-transition thickness from a ray profile.
            ds = SHOCK_PROFILE_DS_FACTOR * min(dr, dz)
            half = max(SHOCK_PROFILE_HALF_WIDTH_CELLS * h, 4.0 * offset)
            s0 = max(0.0, R - half)
            s1 = min(s_max, R + half)
            sample_s = np.arange(s0, s1 + 0.25 * ds, ds)
            sample_rho = _bilinear_sample(
                state["rho"], sample_s * st, HOB + sample_s * ct,
                r_centers, z_centers
            )
            rho10 = rho1 + 0.1 * (rho2 - rho1)
            rho90 = rho1 + 0.9 * (rho2 - rho1)
            # For outward-increasing s, density generally falls through rho90 then rho10.
            i_near = int(np.argmin(np.abs(sample_s - R)))
            x90 = _linear_crossing_position(sample_s, sample_rho, rho90,
                                            max(0, i_near - int(half / ds)), i_near + 1)
            x10 = _linear_crossing_position(sample_s, sample_rho, rho10,
                                            i_near, min(len(sample_s) - 1, i_near + int(half / ds)))
            thickness = x10 - x90 if np.isfinite(x10) and np.isfinite(x90) and x10 >= x90 else np.nan

            row.update({
                "observed_density_ratio": rho_ratio,
                "rh_density_ratio": float(compression_rh),
                "observed_pressure_ratio": pressure_ratio,
                "rh_pressure_ratio": float(pressure_rh),
                "observed_postshock_normal_velocity_m_s": u2n,
                "rh_postshock_normal_velocity_m_s": float(u2n_rh),
                "shock_thickness_m": float(thickness * R / norm) if np.isfinite(thickness) else np.nan,
                "shock_thickness_cells": float((thickness * R / norm) / h) if np.isfinite(thickness) else np.nan,
                "status": "valid",
            })
            angle_rows.append(row)
            per_snapshot.append(row)

        valid_rows = [r for r in per_snapshot if r["status"] == "valid"]
        def med(field):
            vals = np.asarray([r[field] for r in valid_rows], dtype=float)
            return float(np.nanmedian(vals)) if vals.size and np.any(np.isfinite(vals)) else np.nan
        summary_rows.append({
            "time_s": float(state["time"]),
            "valid_shock_angle_bins": int(len(valid_rows)),
            "median_local_shock_mach": med("shock_mach_relative_upstream"),
            "median_observed_density_ratio": med("observed_density_ratio"),
            "median_rh_density_ratio": med("rh_density_ratio"),
            "median_observed_pressure_ratio": med("observed_pressure_ratio"),
            "median_rh_pressure_ratio": med("rh_pressure_ratio"),
            "median_observed_postshock_normal_velocity_m_s": med("observed_postshock_normal_velocity_m_s"),
            "median_rh_postshock_normal_velocity_m_s": med("rh_postshock_normal_velocity_m_s"),
            "median_shock_thickness_cells": med("shock_thickness_cells"),
        })

    return angle_rows, summary_rows, profile_rows


def write_shock_structure_outputs(output_dir, angle_rows, summary_rows, profile_rows):
    """Write angle-resolved shock jumps and selected ray profiles."""
    output_dir = Path(output_dir)
    specs = (
        ("shock_jump_by_angle.csv", angle_rows),
        ("shock_jump_summary.csv", summary_rows),
        ("shock_structure_profiles.csv", profile_rows),
    )
    for filename, rows in specs:
        path = output_dir / filename
        if rows:
            fields = list(rows[0].keys())
        elif filename == "shock_structure_profiles.csv":
            fields = ["time_s", "theta_deg", "distance_from_front_cells",
                      "distance_from_front_m", "density_kg_m3", "pressure_Pa",
                      "density_over_ambient", "pressure_over_ambient"]
        else:
            continue
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    if not summary_rows:
        return
    times = np.asarray([r["time_s"] for r in summary_rows], dtype=float)
    valid = np.asarray([r["valid_shock_angle_bins"] > 0 for r in summary_rows])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    latest_profile_time = max((r["time_s"] for r in profile_rows), default=np.nan)
    for angle in SHOCK_PROFILE_RAY_ANGLES_DEG:
        selected = [r for r in profile_rows
                    if abs(r["theta_deg"] - angle) <= 180.0 / N_ANGLE_BINS
                    and r["time_s"] == latest_profile_time]
        if not selected:
            continue
        profile = selected
        profile.sort(key=lambda r: r["distance_from_front_cells"])
        ax.plot([r["distance_from_front_cells"] for r in profile],
                [r["density_over_ambient"] for r in profile],
                label=f"{angle:g}°")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("Distance from detected front [maximum cell widths]")
    ax.set_ylabel(r"Density / ambient density, $\rho/\rho_0$")
    ax.set_title("Density profiles through latest detected shock")
    ax.grid(True, ls=":")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8)

    ax = axes[0, 1]
    for angle in SHOCK_PROFILE_RAY_ANGLES_DEG:
        selected = [r for r in profile_rows
                    if abs(r["theta_deg"] - angle) <= 180.0 / N_ANGLE_BINS
                    and r["time_s"] == latest_profile_time]
        if not selected:
            continue
        profile = selected
        profile.sort(key=lambda r: r["distance_from_front_cells"])
        ax.plot([r["distance_from_front_cells"] for r in profile],
                [r["pressure_over_ambient"] for r in profile],
                label=f"{angle:g}°")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("Distance from detected front [maximum cell widths]")
    ax.set_ylabel(r"Pressure / ambient pressure, $p/p_0$")
    ax.set_title("Pressure profiles through latest detected shock")
    ax.grid(True, ls=":")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    if np.any(valid):
        good_t = times[valid]
        for field, label, style in (
            ("median_observed_density_ratio", "Sampled", "o-"),
            ("median_rh_density_ratio", "Rankine-Hugoniot", "s--"),
        ):
            values = np.asarray([r[field] for r in summary_rows], dtype=float)[valid]
            ax.plot(good_t, values, style, label=label)
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No resolved, valid shock-jump samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(r"Density ratio, $\rho_2/\rho_1$")
    ax.set_title("Local normal density jump")
    ax.grid(True, which="both", ls=":")

    ax = axes[1, 1]
    if np.any(valid):
        good_t = times[valid]
        for field, label, style in (
            ("median_observed_pressure_ratio", "Sampled", "o-"),
            ("median_rh_pressure_ratio", "Rankine-Hugoniot", "s--"),
        ):
            values = np.asarray([r[field] for r in summary_rows], dtype=float)[valid]
            ax.plot(good_t, values, style, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No resolved, valid shock-jump samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(r"Pressure ratio, $p_2/p_1$")
    ax.set_title("Local normal pressure jump")
    ax.grid(True, which="both", ls=":")

    fig.suptitle(
        "Shock structure: bilinear ray profiles and local normal Rankine-Hugoniot checks\n"
        "Diagnostics only; no influence on the Euler solution",
        fontsize=12, fontweight="bold"
    )
    fig.savefig(output_dir / "shock_structure_diagnostics.png", dpi=SNAPSHOT_DPI)
    plt.close(fig)


def calculate_verification_rows(snapshot_states, shock_speed_history,
                                 shock_mach_history, r_centers, z_centers,
                                 dr, dz, deposition_width, r_max, z_max,
                                 shock_summary_rows=None):
    """Build Sedov radius/speed comparisons and merge local shock-jump summaries."""
    rows = []
    gamma_matches_xi0 = np.isclose(gamma, 1.4, rtol=0.0, atol=1.0e-8)
    nearest_boundary = min(HOB, r_max, z_max - HOB) if HOB > 0.0 else 0.0
    shock_summary_rows = shock_summary_rows or []

    for i, state in enumerate(snapshot_states):
        time_s = float(state["time"])
        detected = state["r_shock"][state["r_shock"] > 0.0]
        radius_median = float(np.median(detected)) if detected.size else np.nan
        radius_p90 = float(np.percentile(detected, 90)) if detected.size else np.nan
        r_st = float(sedov_radius(max(time_s, 1.0e-30))) if time_s > 0.0 else np.nan
        detected_angle_fraction = float(
            np.count_nonzero(state["r_shock"] > 0.0) / max(len(state["r_shock"]), 1)
        )
        sedov_valid = bool(
            gamma_matches_xi0 and HOB > 0.0 and np.isfinite(r_st)
            and r_st >= SEDOV_MIN_RADIUS_TO_DEPOSITION * deposition_width
            and r_st <= SEDOV_CLEARANCE_FRACTION * nearest_boundary
            and detected_angle_fraction >= SEDOV_MIN_DETECTED_ANGLE_FRACTION
        )
        sedov_ratio = radius_median / r_st if sedov_valid and np.isfinite(radius_median) else np.nan
        speed = shock_speed_history[i] if i < len(shock_speed_history) else np.full(N_ANGLE_BINS, np.nan)
        usable_speeds = speed[np.isfinite(speed) & (speed > 0.0)]
        speed_median = float(np.median(usable_speeds)) if usable_speeds.size else np.nan
        sedov_speed = alpha * r_st / time_s if time_s > 0.0 and np.isfinite(r_st) else np.nan
        sedov_speed_ratio = (
            speed_median / sedov_speed
            if sedov_valid and np.isfinite(speed_median) and sedov_speed > 0.0
            else np.nan
        )
        row = {
            "time_s": time_s,
            "sedov_reference_radius_m": r_st,
            "detected_radius_median_m": radius_median,
            "detected_radius_p90_m": radius_p90,
            "detected_angle_fraction": detected_angle_fraction,
            "sedov_comparison_valid": int(sedov_valid),
            "detected_to_sedov_radius_ratio": sedov_ratio,
            "sedov_reference_shock_speed_m_s": sedov_speed,
            "detected_radial_ray_speed_median_m_s": speed_median,
            "detected_to_sedov_speed_ratio": sedov_speed_ratio,
            "gamma14_xi0_applicable": int(gamma_matches_xi0),
        }
        if i < len(shock_summary_rows):
            row.update(shock_summary_rows[i])
        rows.append(row)
    return rows


def write_conservation_history(output_dir, history):
    """Write mass/energy budget histories to CSV."""
    path = Path(output_dir) / "conservation_history.csv"
    fields = [
        "time_s", "excess_energy_remaining_J", "cumulative_boundary_energy_out_J",
        "cumulative_floor_energy_added_J", "budget_expected_energy_J",
        "energy_budget_residual_J", "energy_budget_residual_percent",
        "mass_remaining_kg", "cumulative_boundary_mass_out_kg",
        "cumulative_floor_mass_added_kg", "budget_expected_mass_kg",
        "mass_budget_residual_kg", "mass_budget_residual_percent",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, time_s in enumerate(history["time_s"]):
            row = {key: history[key][i] for key in fields}
            writer.writerow(row)
    return path


def write_verification_outputs(output_dir, rows):
    """Write Sedov and local normal Rankine-Hugoniot summary diagnostics."""
    output_dir = Path(output_dir)
    path = output_dir / "verification_diagnostics.csv"
    fields = [
        "time_s", "sedov_reference_radius_m", "detected_radius_median_m",
        "detected_radius_p90_m", "detected_angle_fraction", "sedov_comparison_valid",
        "detected_to_sedov_radius_ratio", "sedov_reference_shock_speed_m_s",
        "detected_radial_ray_speed_median_m_s", "detected_to_sedov_speed_ratio",
        "gamma14_xi0_applicable", "valid_shock_angle_bins", "median_local_shock_mach",
        "median_observed_density_ratio", "median_rh_density_ratio",
        "median_observed_pressure_ratio", "median_rh_pressure_ratio",
        "median_observed_postshock_normal_velocity_m_s",
        "median_rh_postshock_normal_velocity_m_s", "median_shock_thickness_cells",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    sedov_valid = np.asarray([bool(row["sedov_comparison_valid"]) for row in rows])
    sedov_ratio = np.asarray([row["detected_to_sedov_radius_ratio"] for row in rows], dtype=float)
    valid_counts = np.asarray([row.get("valid_shock_angle_bins", 0) for row in rows], dtype=int)
    has_jump = valid_counts > 0
    rho_obs = np.asarray([row.get("median_observed_density_ratio", np.nan) for row in rows], dtype=float)
    rho_rh = np.asarray([row.get("median_rh_density_ratio", np.nan) for row in rows], dtype=float)
    p_obs = np.asarray([row.get("median_observed_pressure_ratio", np.nan) for row in rows], dtype=float)
    p_rh = np.asarray([row.get("median_rh_pressure_ratio", np.nan) for row in rows], dtype=float)
    vel_obs = np.asarray([row.get("median_observed_postshock_normal_velocity_m_s", np.nan) for row in rows], dtype=float)
    vel_rh = np.asarray([row.get("median_rh_postshock_normal_velocity_m_s", np.nan) for row in rows], dtype=float)
    sedov_speed_ratio = np.asarray([row["detected_to_sedov_speed_ratio"] for row in rows], dtype=float)
    detected_fraction = np.asarray([row.get("detected_angle_fraction", 0.0) for row in rows], dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    ax = axes[0, 0]
    good = sedov_valid & np.isfinite(sedov_ratio) & (times > 0.0)
    if np.any(good):
        ax.plot(times[good], sedov_ratio[good], "o-")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="Sedov-Taylor reference")
        ax.set_xscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid unreflected free-air comparison window",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Median detected radius / Sedov radius")
    ax.set_title("Free-air Sedov-Taylor radius comparison")
    ax.grid(True, which="both", ls=":")

    ax = axes[0, 1]
    good_speed = sedov_valid & np.isfinite(sedov_speed_ratio) & (times > 0.0)
    if np.any(good_speed):
        ax.plot(times[good_speed], sedov_speed_ratio[good_speed], "o-")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="Sedov-Taylor reference")
        ax.set_xscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid Sedov shock-speed comparison",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Median radial-ray speed / Sedov speed")
    ax.set_title("Free-air Sedov-Taylor shock-speed comparison")
    ax.grid(True, which="both", ls=":")

    ax = axes[1, 0]
    if np.any(has_jump):
        ax.plot(times[has_jump], rho_obs[has_jump], "o-", label="Sampled local jump")
        ax.plot(times[has_jump], rho_rh[has_jump], "s--", label="Rankine-Hugoniot")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No resolved shock-jump samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(r"Density ratio, $\rho_2/\rho_1$")
    ax.set_title("Local normal density jump")
    ax.grid(True, which="both", ls=":")

    ax = axes[1, 1]
    if np.any(has_jump):
        ax.plot(times[has_jump], p_obs[has_jump], "o-", label="Sampled local jump")
        ax.plot(times[has_jump], p_rh[has_jump], "s--", label="Rankine-Hugoniot")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No resolved shock-jump samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(r"Pressure ratio, $p_2/p_1$")
    ax.set_title("Local normal pressure jump")
    ax.grid(True, which="both", ls=":")

    ax = axes[2, 0]
    if np.any(has_jump):
        ax.plot(times[has_jump], vel_obs[has_jump], "o-", label="Sampled local jump")
        ax.plot(times[has_jump], vel_rh[has_jump], "s--", label="Rankine-Hugoniot")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No resolved shock-jump samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Post-shock normal velocity [m/s]")
    ax.set_title("Local normal post-shock velocity")
    ax.grid(True, which="both", ls=":")

    ax = axes[2, 1]
    ax.plot(times, detected_fraction, "o-", label="Density-threshold coverage")
    ax.axhline(SEDOV_MIN_DETECTED_ANGLE_FRACTION, color="k", ls="--", lw=1,
               label="Minimum Sedov coverage")
    ax2 = ax.twinx()
    ax2.plot(times, valid_counts, "C1s-", label="Valid local jump rays")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Fraction of detected angular bins")
    ax2.set_ylabel("Number of locally valid shock rays")
    ax.set_title("Shock-front coverage and jump-diagnostic sample count")
    ax.grid(True, ls=":")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    fig.suptitle(
        "Verification diagnostics\n"
        "Sedov comparison is global; Rankine-Hugoniot comparison uses local shock-normal states",
        fontsize=12, fontweight="bold"
    )
    fig.savefig(output_dir / "verification_diagnostics.png", dpi=SNAPSHOT_DPI)
    plt.close(fig)
    return path


def write_run_summary(output_dir, summary):
    """Write one-row run metadata/results file for later convergence comparison."""
    path = Path(output_dir) / "run_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return path


def compare_convergence_runs(current_summary, output_dir):
    """Compare this run with user-selected prior runs from grid/domain studies.

    Different grid sizes and DOMAIN_FACTOR values are allowed.  Physical
    inputs, shock threshold, and physical Gaussian width must match.
    """
    if not CONVERGENCE_REFERENCE_RUNS:
        return

    records = [{**current_summary, "run_label": "current"}]
    for directory in CONVERGENCE_REFERENCE_RUNS:
        summary_path = Path(directory) / "run_summary.csv"
        if not summary_path.exists():
            print(f"WARNING: convergence reference has no run_summary.csv: {directory}")
            continue
        with summary_path.open("r", newline="") as f:
            row = next(csv.DictReader(f), None)
        if row is None:
            continue
        records.append({**row, "run_label": str(directory)})

    if len(records) < 2:
        print("No usable prior runs found for convergence comparison.")
        return

    invariant_keys = [
        "E_kt", "E_J", "HOB_m", "t_end_s", "final_time_s", "rho0_kg_m3", "p0_Pa", "gamma",
        "deposition_width_m", "shock_density_excess",
        "spatial_reconstruction", "slope_limiter", "time_integrator", "numerical_flux",
    ]
    reference = records[0]
    mismatches = []
    for row in records[1:]:
        for key in invariant_keys:
            try:
                a = float(reference[key])
                b = float(row[key])
                matched = np.isclose(a, b, rtol=1.0e-8, atol=1.0e-12)
            except (KeyError, TypeError, ValueError):
                matched = reference.get(key) == row.get(key)
            if not matched:
                mismatches.append((row["run_label"], key, reference.get(key), row.get(key)))
    if mismatches:
        print("WARNING: convergence comparison skipped because physical inputs differ:")
        for label, key, ref_value, other_value in mismatches:
            print(f"  {label}: {key} current={ref_value}, comparison={other_value}")
        print("Use identical physical inputs and a fixed DEPOSITION_WIDTH_M for grid studies.")
        return

    gauge_positions_match = all(
        row.get("gauge_positions", "") == reference.get("gauge_positions", "")
        for row in records[1:]
    )
    fields = [
        "run_label", "spatial_reconstruction", "slope_limiter", "time_integrator",
        "numerical_flux", "NR", "NZ", "DOMAIN_FACTOR", "dr_m", "dz_m", "max_cell_width_m",
        "final_median_shock_radius_m", "final_p90_shock_radius_m",
        "energy_budget_residual_percent", "gauge_positions",
        "max_gauge_peak_overpressure_Pa", "max_gauge_positive_impulse_Pa_s",
    ]
    path = Path(output_dir) / "convergence_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    cell_width = np.asarray([float(row["max_cell_width_m"]) for row in records])
    domain_factor = np.asarray([float(row["DOMAIN_FACTOR"]) for row in records])
    radius = np.asarray([float(row["final_median_shock_radius_m"]) for row in records])
    residual = np.asarray([abs(float(row["energy_budget_residual_percent"])) for row in records])
    labels = [f"{row['NR']}x{row['NZ']}\nDF={row['DOMAIN_FACTOR']}" for row in records]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].plot(cell_width, radius, "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Maximum cell width, max(dr,dz) [m]")
    axes[0].set_ylabel("Final median detected shock radius [m]")
    axes[0].set_title("Grid-resolution sensitivity")
    axes[0].grid(True, which="both", ls=":")

    axes[1].plot(domain_factor, radius, "o-")
    axes[1].set_xlabel("DOMAIN_FACTOR")
    axes[1].set_ylabel("Final median detected shock radius [m]")
    axes[1].set_title("Computational-domain sensitivity")
    axes[1].grid(True, ls=":")

    axes[2].plot(cell_width, residual, "o-")
    axes[2].set_xscale("log")
    axes[2].set_yscale("symlog", linthresh=1.0e-10)
    axes[2].set_xlabel("Maximum cell width, max(dr,dz) [m]")
    axes[2].set_ylabel("Absolute energy-budget residual [%]")
    axes[2].set_title("Energy-budget closure")
    axes[2].grid(True, which="both", ls=":")
    for ax in axes[:2]:
        for x, y, label in zip(cell_width if ax is axes[0] else domain_factor, radius, labels):
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    gauge_note = ("Gauge coordinates match across runs." if gauge_positions_match
                  else "Gauge coordinates differ; do not compare gauge maxima across these runs.")
    fig.suptitle(
        "Grid/domain comparison — only matched physical-input runs are included\n" + gauge_note,
        fontsize=12, fontweight="bold"
    )
    fig.savefig(Path(output_dir) / "convergence_comparison.png", dpi=SNAPSHOT_DPI)
    plt.close(fig)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    if E <= 0.0 or rho0 <= 0.0 or p0 <= 0.0:
        raise ValueError("E, rho0, and p0 must be positive.")
    if HOB < 0.0:
        raise ValueError("HOB must be >= 0.")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1.")
    if NR < 20 or NZ < 20:
        raise ValueError("Use at least 20 cells in each direction.")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive.")
    if SHOCK_PROFILE_SAMPLE_OFFSET_CELLS <= 0.0:
        raise ValueError("SHOCK_PROFILE_SAMPLE_OFFSET_CELLS must be positive.")
    if SHOCK_PROFILE_DS_FACTOR <= 0.0:
        raise ValueError("SHOCK_PROFILE_DS_FACTOR must be positive.")
    if SPATIAL_RECONSTRUCTION.upper() != "MUSCL":
        raise ValueError("SPATIAL_RECONSTRUCTION must be 'MUSCL'.")
    if TIME_INTEGRATOR.upper() != "SSP_RK2":
        raise ValueError("TIME_INTEGRATOR must be 'SSP_RK2'.")
    if SLOPE_LIMITER.upper() not in ("MC", "MINMOD"):
        raise ValueError("SLOPE_LIMITER must be 'MC' or 'minmod'.")
    if MAX_CFL_RETRIES < 1:
        raise ValueError("MAX_CFL_RETRIES must be at least 1.")
    if not (0.0 < CFL_RETRY_SHRINK < 1.0):
        raise ValueError("CFL_RETRY_SHRINK must be between 0 and 1.")
    if not (0.0 < INITIAL_CFL_SAFETY_FACTOR <= 1.0):
        raise ValueError("INITIAL_CFL_SAFETY_FACTOR must be in (0, 1].")
    if RHO_POSITIVITY_FLOOR <= 0.0 or INTERNAL_ENERGY_POSITIVITY_FLOOR <= 0.0:
        raise ValueError("Positivity admissibility thresholds must be positive.")
    if FLUX_BLEND_BISECTION_ITERS < 20:
        raise ValueError("FLUX_BLEND_BISECTION_ITERS should be at least 20.")

    t_interaction = interaction_time(HOB)
    R_end_ref = float(sedov_radius(t_end))
    r_max = DOMAIN_FACTOR * R_end_ref
    z_max = HOB + DOMAIN_FACTOR * R_end_ref

    r_faces = np.linspace(0.0, r_max, NR + 1)
    z_faces = np.linspace(0.0, z_max, NZ + 1)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    dr = r_faces[1] - r_faces[0]
    dz = z_faces[1] - z_faces[0]

    cell_volumes = (np.pi * (r_faces[1:]**2 - r_faces[:-1]**2))[:, None] * dz
    cell_volumes = np.broadcast_to(cell_volumes, (NR, NZ)).copy()

    U, _, deposition_width = make_initial_state(r_centers, z_centers, dr, dz)

    # Phase-1 output directory is independent of whether spatial PNGs are enabled.
    run_output_dir = None
    if MAKE_PLOTS or MAKE_PHASE1_DIAGNOSTICS:
        run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_output_dir = Path(SNAPSHOT_OUTPUT_DIR) / run_stamp
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run outputs will be saved in: {run_output_dir.resolve()}")

    # Time-history diagnostics retained from the benchmark version.
    times = []
    eta_history = []
    tau_history = []
    volume_history = []
    volume_ratio_history = []
    radius_history = []
    radius_median_history = []
    radius_p90_history = []
    energy_history = []

    # Snapshot states are also used by the Sedov/RH diagnostics.  They do not
    # participate in, or feed back into, the numerical flow update.
    snapshot_states = []
    target_times = sorted(set(float(x) for x in SNAPSHOT_TIMES if 0.0 < x <= t_end))
    next_target_index = 0
    capture_states = MAKE_PLOTS or MAKE_PHASE1_DIAGNOSTICS

    if capture_states:
        if any(x <= 0.0 for x in SNAPSHOT_TIMES):
            raise ValueError("All SNAPSHOT_TIMES must be positive.")
        rho_initial, ur_initial, uz_initial, pressure_initial = primitive_from_conserved(U)
        theta_initial, r_shock_initial, _ = shock_geometry(
            r_centers, z_centers, rho_initial
        )
        snapshot_states.append({
            "time": 0.0,
            "rho": rho_initial.copy(),
            "ur": ur_initial.copy(),
            "uz": uz_initial.copy(),
            "pressure": pressure_initial.copy(),
            "theta": theta_initial.copy(),
            "r_shock": r_shock_initial.copy(),
        })

    gauges = initialize_gauges(r_centers, z_centers, r_max, z_max) if MAKE_PHASE1_DIAGNOSTICS else []
    gauge_history = {"time_s": [], "pressure_Pa": {g["name"]: [] for g in gauges}}
    if MAKE_PHASE1_DIAGNOSTICS:
        sample_gauge_pressures(U, gauges, 0.0, gauge_history)

    initial_energy = total_energy(U, cell_volumes)
    initial_mass = total_mass(U, cell_volumes)
    cumulative_boundary_energy_out = 0.0
    cumulative_floor_energy_added = 0.0
    cumulative_boundary_mass_out = 0.0
    cumulative_floor_mass_added = 0.0

    budget_history = {
        "time_s": [],
        "excess_energy_remaining_J": [],
        "cumulative_boundary_energy_out_J": [],
        "cumulative_floor_energy_added_J": [],
        "budget_expected_energy_J": [],
        "energy_budget_residual_J": [],
        "energy_budget_residual_percent": [],
        "mass_remaining_kg": [],
        "cumulative_boundary_mass_out_kg": [],
        "cumulative_floor_mass_added_kg": [],
        "budget_expected_mass_kg": [],
        "mass_budget_residual_kg": [],
        "mass_budget_residual_percent": [],
    }

    def record_budget(time_s, state):
        energy_now = total_energy(state, cell_volumes)
        expected_energy = (initial_energy - cumulative_boundary_energy_out
                           + cumulative_floor_energy_added)
        energy_residual = energy_now - expected_energy
        mass_now = total_mass(state, cell_volumes)
        expected_mass = (initial_mass - cumulative_boundary_mass_out
                         + cumulative_floor_mass_added)
        mass_residual = mass_now - expected_mass
        budget_history["time_s"].append(float(time_s))
        budget_history["excess_energy_remaining_J"].append(energy_now)
        budget_history["cumulative_boundary_energy_out_J"].append(cumulative_boundary_energy_out)
        budget_history["cumulative_floor_energy_added_J"].append(cumulative_floor_energy_added)
        budget_history["budget_expected_energy_J"].append(expected_energy)
        budget_history["energy_budget_residual_J"].append(energy_residual)
        budget_history["energy_budget_residual_percent"].append(
            100.0 * energy_residual / max(abs(initial_energy), 1.0e-30)
        )
        budget_history["mass_remaining_kg"].append(mass_now)
        budget_history["cumulative_boundary_mass_out_kg"].append(cumulative_boundary_mass_out)
        budget_history["cumulative_floor_mass_added_kg"].append(cumulative_floor_mass_added)
        budget_history["budget_expected_mass_kg"].append(expected_mass)
        budget_history["mass_budget_residual_kg"].append(mass_residual)
        budget_history["mass_budget_residual_percent"].append(
            100.0 * mass_residual / max(abs(initial_mass), 1.0e-30)
        )

    record_budget(0.0, U)

    print("=" * 76)
    print("AXISYMMETRIC 3-D BLAST WAVE / REFLECTING-GROUND EULER SOLVER")
    print("=" * 76)
    print(f"Input explosion energy             : {E_kt:.8g} kt")
    print(f"Converted explosion energy         : {E:.8e} J  (4.184e12 J/kt)")
    print(f"Height of burst                    : {HOB:.5g} m")
    print(f"Ambient density                    : {rho0:.5g} kg/m^3")
    print(f"Ambient pressure                   : {p0:.5g} Pa")
    print(f"gamma                              : {gamma:.5g}")
    print(f"Spatial reconstruction              : {SPATIAL_RECONSTRUCTION} (primitive variables)")
    print(f"Slope limiter                       : {SLOPE_LIMITER}")
    print(f"Time integrator                     : {TIME_INTEGRATOR}")
    print(f"Initial CFL predictor safety factor  : {INITIAL_CFL_SAFETY_FACTOR:.3f}")
    print(f"Numerical flux                      : local Rusanov")
    print("Positivity treatment                 : conservative MUSCL/first-order flux blend")
    print(f"Density/internal-energy admissibility: {RHO_POSITIVITY_FLOOR:.1e} kg/m^3 / "
          f"{INTERNAL_ENERGY_POSITIVITY_FLOOR:.1e} J/m^3")
    print(f"Runtime acceleration                : {'Numba JIT' if USE_NUMBA and NUMBA_AVAILABLE else 'NumPy fallback'}")
    if USE_NUMBA and not NUMBA_AVAILABLE:
        print("  Numba is not installed/working; install it with `pip install numba` for acceleration.")
    print(f"Grid                               : {NR} x {NZ}")
    print(f"Domain                             : r=[0,{r_max:.4g}] m, z=[0,{z_max:.4g}] m")
    print(f"Cell dimensions                    : dr={dr:.4g} m, dz={dz:.4g} m")
    print(f"Gaussian deposition width          : {deposition_width:.4g} m")
    print(f"Free-air ST radius at t_end         : {R_end_ref:.5g} m")
    if HOB > 0.0:
        print(f"Characteristic interaction time    : {t_interaction:.5e} s")
    else:
        print("Characteristic interaction time    : 0 (burst at surface)")
    if capture_states:
        print("Diagnostic snapshot targets [s]     : " + ", ".join(f"{x:.3g}" for x in target_times))
    print(f"Initial excess energy               : {initial_energy:.8e} J")
    print(f"Initial energy normalization error  : {(initial_energy / E - 1.0) * 100.0:.6e} %")
    print(f"Initial atmospheric domain mass     : {initial_mass:.8e} kg")
    if MAKE_PHASE1_DIAGNOSTICS:
        print(f"Virtual pressure gauges             : {len(gauges)}")
        for gauge in gauges:
            print(
                f"  {gauge['name']}: requested (r,z)=({gauge['r_requested_m']:g},"
                f"{gauge['z_requested_m']:g}) m; sampled cell center "
                f"({gauge['r_sampled_m']:.4g},{gauge['z_sampled_m']:.4g}) m"
            )
    if not np.isclose(gamma, 1.4, rtol=0.0, atol=1.0e-8):
        print("WARNING: XI0=1.033 is the configured gamma=1.4 Sedov reference; "
              "Sedov radius comparison will be marked inapplicable.")
    print()

    t = 0.0
    step = 0
    total_cfl_retries = 0
    total_flux_limited_stages = 0
    total_low_order_fallbacks = 0
    total_high_order_inadmissible_cells = 0
    minimum_flux_blend_alpha_seen = 1.0
    while t < t_end and step < MAX_STEPS:
        dt = min(INITIAL_CFL_SAFETY_FACTOR * stable_timestep(U, dr, dz), t_end - t)
        retry_count = 0
        while True:
            try:
                U_new, step_budget = advance_one_step(
                    U, dt, r_faces, dz, cell_volumes, dr=dr, return_budget=True,
                    z_centers=z_centers, r_centers=r_centers
                )
                U = U_new
                break
            except StageCFLViolation as exc:
                retry_count += 1
                total_cfl_retries += 1
                diag_text = format_state_diagnostics(exc.diagnostics)
                cfl_details = ""
                if exc.diagnostics and "local_cfl_dt_s" in exc.diagnostics:
                    d = exc.diagnostics
                    cfl_details = (
                        f" | CFL cell=({d.get('cfl_i')},{d.get('cfl_j')}), "
                        f"(r,z)=({d.get('cfl_r_m', np.nan):.4g},"
                        f"{d.get('cfl_z_m', np.nan):.4g}) m, "
                        f"rho={d.get('cfl_rho_kg_m3', np.nan):.3e}, "
                        f"p={d.get('cfl_pressure_Pa', np.nan):.3e} Pa, "
                        f"u=({d.get('cfl_ur_m_s', np.nan):.3e},"
                        f"{d.get('cfl_uz_m_s', np.nan):.3e}) m/s, "
                        f"M={d.get('cfl_fluid_mach', np.nan):.3e}, "
                        f"local dt={d.get('local_cfl_dt_s', np.nan):.3e} s"
                    )
                print(
                    f"  STEP RETRY {retry_count}/{MAX_CFL_RETRIES}: {exc.reason} "
                    f"current dt={dt:.4e} s, suggested dt={exc.suggested_dt:.4e} s; "
                    f"{diag_text}{cfl_details}"
                )
                if retry_count > MAX_CFL_RETRIES:
                    raise RuntimeError(
                        "SSP-RK2 step rejected repeatedly after "
                        f"{MAX_CFL_RETRIES} retries. Last reason: {exc.reason} "
                        f"Suggested dt={exc.suggested_dt:.6e} s; {diag_text}"
                    )
                shrink = min(CFL_RETRY_SHRINK, 0.8)
                dt = min(shrink * dt, shrink * exc.suggested_dt, t_end - t)
                if dt <= MIN_TIMESTEP_FRACTION * max(t_end, 1.0e-30):
                    raise RuntimeError(
                        "Adaptive SSP-RK2 timestep fell below the minimum threshold. "
                        f"Last reason: {exc.reason}; {diag_text}"
                    )

        cumulative_boundary_mass_out += step_budget["boundary_mass_out_kg"]
        cumulative_boundary_energy_out += step_budget["boundary_energy_out_J"]
        cumulative_floor_mass_added += step_budget["mass_floor_added_kg"]
        cumulative_floor_energy_added += step_budget["energy_floor_added_J"]
        total_flux_limited_stages += step_budget["flux_limited_stages"]
        total_low_order_fallbacks += step_budget["low_order_fallbacks"]
        total_high_order_inadmissible_cells += step_budget["high_order_inadmissible_cells"]
        minimum_flux_blend_alpha_seen = min(
            minimum_flux_blend_alpha_seen, step_budget["minimum_flux_blend_alpha"]
        )
        t += dt
        step += 1

        if MAKE_PHASE1_DIAGNOSTICS:
            sample_gauge_pressures(U, gauges, t, gauge_history)

        # Capture the current state when a logarithmic output target is crossed.
        if capture_states and next_target_index < len(target_times):
            if t >= target_times[next_target_index]:
                rho_snap, ur_snap, uz_snap, pressure_snap = primitive_from_conserved(U)
                theta_snap, r_shock_snap, _ = shock_geometry(
                    r_centers, z_centers, rho_snap
                )
                snapshot_states.append({
                    "time": float(t),
                    "rho": rho_snap.copy(),
                    "ur": ur_snap.copy(),
                    "uz": uz_snap.copy(),
                    "pressure": pressure_snap.copy(),
                    "theta": theta_snap.copy(),
                    "r_shock": r_shock_snap.copy(),
                })
                print(f"  captured diagnostic state at t={t:.5e} s "
                      f"(target {target_times[next_target_index]:.5e} s)")
                next_target_index += 1
                while (next_target_index < len(target_times)
                       and t >= target_times[next_target_index]):
                    print("  NOTE: snapshot target "
                          f"{target_times[next_target_index]:.5e} s was crossed "
                          "within the same time step and has no separate state.")
                    next_target_index += 1

        if step == 1 or step % PRINT_EVERY == 0 or t >= t_end:
            rho, ur, uz, pressure = primitive_from_conserved(U)
            theta, r_shock, volume = shock_geometry(r_centers, z_centers, rho)
            R_ref = float(sedov_radius(max(t, 1.0e-30)))
            eta = HOB / R_ref
            tau = (t / t_interaction) if t_interaction > 0.0 else np.inf
            volume_ref = (4.0 / 3.0) * np.pi * R_ref**3
            volume_ratio = volume / volume_ref
            detected_radii = r_shock[r_shock > 0.0]
            max_radius = float(np.max(detected_radii)) if detected_radii.size else 0.0
            median_radius = float(np.median(detected_radii)) if detected_radii.size else np.nan
            p90_radius = float(np.percentile(detected_radii, 90)) if detected_radii.size else np.nan
            e_now = total_energy(U, cell_volumes)

            times.append(t)
            eta_history.append(eta)
            tau_history.append(tau)
            volume_history.append(volume)
            volume_ratio_history.append(volume_ratio)
            radius_history.append(max_radius)
            radius_median_history.append(median_radius)
            radius_p90_history.append(p90_radius)
            energy_history.append(e_now)
            record_budget(t, U)

            energy_residual_pct = budget_history["energy_budget_residual_percent"][-1]
            print(
                f"step={step:7d}  t={t:10.4e} s  dt={dt:9.2e} s  "
                f"eta={eta:9.3g}  tau={tau:9.3g}  V/V_ST={volume_ratio:7.3f}  "
                f"E-budget residual={energy_residual_pct:+.3e}%  "
                f"CFL retries={retry_count}  "
                f"flux-limited stages={total_flux_limited_stages}  "
                f"bad HO cells={total_high_order_inadmissible_cells}  "
                f"min alpha={minimum_flux_blend_alpha_seen:.5g}"
            )

    if step >= MAX_STEPS and t < t_end:
        print("\nWARNING: MAX_STEPS reached before t_end.")

    # Final fields and diagnostics
    rho, ur, uz, pressure = primitive_from_conserved(U)
    theta, r_shock, final_volume = shock_geometry(r_centers, z_centers, rho)
    R_ref_final = float(sedov_radius(max(t, 1.0e-30)))
    V_ref_final = (4.0 / 3.0) * np.pi * R_ref_final**3
    final_volume_ratio = final_volume / V_ref_final
    final_energy = total_energy(U, cell_volumes)
    final_mass = total_mass(U, cell_volumes)
    final_detected_radii = r_shock[r_shock > 0.0]
    final_radius_median = float(np.median(final_detected_radii)) if final_detected_radii.size else np.nan
    final_radius_p90 = float(np.percentile(final_detected_radii, 90)) if final_detected_radii.size else np.nan

    if not budget_history["time_s"] or not np.isclose(budget_history["time_s"][-1], t):
        record_budget(t, U)

    expected_energy_final = initial_energy - cumulative_boundary_energy_out + cumulative_floor_energy_added
    energy_residual_final = final_energy - expected_energy_final
    expected_mass_final = initial_mass - cumulative_boundary_mass_out + cumulative_floor_mass_added
    mass_residual_final = final_mass - expected_mass_final

    print("\n" + "=" * 76)
    print("FINAL DIAGNOSTICS")
    print("=" * 76)
    print(f"Final time                         : {t:.6e} s")
    print(f"Time steps                         : {step}")
    print(f"Total SSP-RK2 CFL retries           : {total_cfl_retries}")
    print(f"Flux-limited RK stages              : {total_flux_limited_stages}")
    print(f"Low-order Rusanov fallbacks         : {total_low_order_fallbacks}")
    print(f"High-order inadmissible cell updates: {total_high_order_inadmissible_cells}")
    print(f"Minimum accepted flux-blend alpha   : {minimum_flux_blend_alpha_seen:.8g}")
    print("Cell-local radial/vertical momentum repairs: 0 (none are applied)")
    print(f"Free-air ST reference radius       : {R_ref_final:.6e} m")
    print(f"Scaled height of burst, eta        : {HOB / R_ref_final:.6g}")
    if t_interaction > 0.0:
        print(f"Scaled time, tau                   : {t / t_interaction:.6g}")
    else:
        print("Scaled time, tau                   : undefined (HOB = 0)")
    print(f"Threshold-based shock volume       : {final_volume:.6e} m^3")
    print(f"Volume / free-air sphere volume    : {final_volume_ratio:.6g}")
    print(f"Median detected shock radius       : {final_radius_median:.6e} m")
    print(f"P90 detected shock radius          : {final_radius_p90:.6e} m")
    print(f"Excess gas energy remaining        : {final_energy:.8e} J")
    print(f"Cumulative outward energy flux     : {cumulative_boundary_energy_out:.8e} J")
    print(f"Cumulative energy added by floors  : {cumulative_floor_energy_added:.8e} J")
    print(f"Energy budget residual             : {energy_residual_final:.8e} J "
          f"({100.0 * energy_residual_final / max(abs(initial_energy), 1.0e-30):+.6e} %)")
    print(f"Cumulative outward mass flux       : {cumulative_boundary_mass_out:.8e} kg")
    print(f"Cumulative mass added by floors    : {cumulative_floor_mass_added:.8e} kg")
    print(f"Mass budget residual               : {mass_residual_final:.8e} kg "
          f"({100.0 * mass_residual_final / max(abs(initial_mass), 1.0e-30):+.6e} %)")
    print()
    print("Numerical method: MC-limited MUSCL in primitive variables + SSP-RK2 + Rusanov.")
    print("NOTE: Shock radius/volume use a density threshold. Local RH checks require a resolved front, ")
    print("      adequate upstream/downstream sampling distance, and valid neighboring front data.")
    print("Positivity is enforced by conservative blending of shared high-/low-order")
    print("Rusanov face fluxes; no cell-local momentum or energy clipping is applied.")
    print("The compatibility floor-budget terms are therefore zero; conservation")
    print("accounting includes the blended boundary fluxes used by the actual update.")

    if capture_states:
        # Always include the final numerical state, even if it is not a target.
        if (not snapshot_states
                or not np.isclose(snapshot_states[-1]["time"], t,
                                  rtol=1.0e-10, atol=1.0e-14)):
            snapshot_states.append({
                "time": float(t),
                "rho": rho.copy(),
                "ur": ur.copy(),
                "uz": uz.copy(),
                "pressure": pressure.copy(),
                "theta": theta.copy(),
                "r_shock": r_shock.copy(),
            })

        shock_speed_history, shock_mach_history = calculate_shock_front_history(
            snapshot_states
        )
    else:
        shock_speed_history, shock_mach_history = [], []

    gauge_summaries = []
    verification_rows = []
    if MAKE_PHASE1_DIAGNOSTICS:
        gauge_summaries = gauge_summary(gauges, gauge_history)
        write_gauge_outputs(run_output_dir, gauges, gauge_history, gauge_summaries)
        write_conservation_history(run_output_dir, budget_history)
        plot_conservation_diagnostics(run_output_dir, budget_history)
        shock_angle_rows, shock_summary_rows, shock_profile_rows = analyze_shock_structure(
            snapshot_states, r_centers, z_centers, dr, dz, deposition_width
        )
        write_shock_structure_outputs(
            run_output_dir, shock_angle_rows, shock_summary_rows, shock_profile_rows
        )
        verification_rows = calculate_verification_rows(
            snapshot_states, shock_speed_history, shock_mach_history,
            r_centers, z_centers, dr, dz, deposition_width, r_max, z_max,
            shock_summary_rows=shock_summary_rows
        )
        write_verification_outputs(run_output_dir, verification_rows)

        summary = {
            "run_timestamp": run_output_dir.name,
            "E_kt": float(E_kt),
            "E_J": float(E),
            "HOB_m": float(HOB),
            "t_end_s": float(t_end),
            "final_time_s": float(t),
            "time_steps": int(step),
            "rho0_kg_m3": float(rho0),
            "p0_Pa": float(p0),
            "gamma": float(gamma),
            "spatial_reconstruction": SPATIAL_RECONSTRUCTION,
            "slope_limiter": SLOPE_LIMITER,
            "time_integrator": TIME_INTEGRATOR,
            "numerical_flux": "local_Rusanov",
            "runtime_acceleration": "Numba JIT" if USE_NUMBA and NUMBA_AVAILABLE else "NumPy",
            "total_cfl_retries": int(total_cfl_retries),
            "total_flux_limited_stages": int(total_flux_limited_stages),
            "total_low_order_fallbacks": int(total_low_order_fallbacks),
            "total_high_order_inadmissible_cells": int(total_high_order_inadmissible_cells),
            "minimum_flux_blend_alpha": float(minimum_flux_blend_alpha_seen),
            "initial_cfl_safety_factor": float(INITIAL_CFL_SAFETY_FACTOR),
            "cell_local_momentum_corrections": 0,
            "positivity_enforcement": "conservative_global_Rusanov_flux_blend",
            "rho_positivity_floor_kg_m3": float(RHO_POSITIVITY_FLOOR),
            "internal_energy_positivity_floor_J_m3": float(INTERNAL_ENERGY_POSITIVITY_FLOOR),
            "NR": int(NR),
            "NZ": int(NZ),
            "DOMAIN_FACTOR": float(DOMAIN_FACTOR),
            "dr_m": float(dr),
            "dz_m": float(dz),
            "max_cell_width_m": float(max(dr, dz)),
            "deposition_width_m": float(deposition_width),
            "shock_density_excess": float(SHOCK_DENSITY_EXCESS),
            "shock_profile_sample_offset_cells": float(SHOCK_PROFILE_SAMPLE_OFFSET_CELLS),
            "shock_profile_min_postsource_factor": float(SHOCK_PROFILE_MIN_POSTSOURCE_FACTOR),
            "shock_profile_ds_factor": float(SHOCK_PROFILE_DS_FACTOR),
            "gauge_positions": ";".join(
                f"{g['name']}:{g['r_requested_m']:.9g}:{g['z_requested_m']:.9g}"
                for g in gauges
            ),
            "initial_excess_energy_J": float(initial_energy),
            "final_excess_energy_J": float(final_energy),
            "outward_energy_flux_J": float(cumulative_boundary_energy_out),
            "floor_energy_added_J": float(cumulative_floor_energy_added),
            "energy_budget_residual_percent": float(
                100.0 * energy_residual_final / max(abs(initial_energy), 1.0e-30)
            ),
            "initial_mass_kg": float(initial_mass),
            "final_mass_kg": float(final_mass),
            "outward_mass_flux_kg": float(cumulative_boundary_mass_out),
            "floor_mass_added_kg": float(cumulative_floor_mass_added),
            "mass_budget_residual_percent": float(
                100.0 * mass_residual_final / max(abs(initial_mass), 1.0e-30)
            ),
            "final_median_shock_radius_m": float(final_radius_median),
            "final_p90_shock_radius_m": float(final_radius_p90),
            "final_sedov_reference_radius_m": float(R_ref_final),
            "final_sedov_comparison_valid": int(verification_rows[-1]["sedov_comparison_valid"]) if verification_rows else 0,
            "final_detected_to_sedov_radius_ratio": float(verification_rows[-1]["detected_to_sedov_radius_ratio"]) if verification_rows else np.nan,
            "final_detected_to_sedov_speed_ratio": float(verification_rows[-1]["detected_to_sedov_speed_ratio"]) if verification_rows else np.nan,
            "final_volume_ratio": float(final_volume_ratio),
            "final_valid_shock_angle_bins": int(shock_summary_rows[-1]["valid_shock_angle_bins"]) if shock_summary_rows else 0,
            "final_median_local_shock_mach": float(shock_summary_rows[-1]["median_local_shock_mach"]) if shock_summary_rows else np.nan,
            "final_median_shock_thickness_cells": float(shock_summary_rows[-1]["median_shock_thickness_cells"]) if shock_summary_rows else np.nan,
            "max_gauge_peak_overpressure_Pa": float(
                max([g["peak_overpressure_Pa"] for g in gauge_summaries], default=np.nan)
            ),
            "max_gauge_positive_impulse_Pa_s": float(
                max([g["positive_phase_impulse_Pa_s"] for g in gauge_summaries], default=np.nan)
            ),
        }
        write_run_summary(run_output_dir, summary)
        compare_convergence_runs(summary, run_output_dir)

    if MAKE_PLOTS:
        print("\nRendering spatial snapshots with individually autoscaled color limits...")
        for i, state in enumerate(snapshot_states):
            plot_scales = determine_snapshot_plot_scales(
                state, shock_mach_history[i]
            )
            save_spatial_snapshot(
                r_faces, z_faces, r_centers, z_centers,
                state["rho"], state["ur"], state["uz"], state["pressure"],
                state["time"], run_output_dir / f"snapshot_{i:04d}_t_{state['time']:.6e}s.png",
                plot_scales,
                state["theta"], state["r_shock"], shock_mach_history[i]
            )

        if MAKE_MP4:
            create_mp4_from_snapshots(
                run_output_dir,
                fps=MP4_FPS,
                output_name=MP4_FILENAME
            )

        make_diagnostic_plots(
            theta, r_shock,
            np.asarray(times), np.asarray(eta_history), np.asarray(tau_history),
            np.asarray(volume_ratio_history), np.asarray(energy_history),
            t_interaction, initial_energy,
            snapshot_states, shock_speed_history, shock_mach_history,
            run_output_dir / "diagnostics.png"
        )

    print(f"\nSaved run outputs to:\n  {run_output_dir.resolve() if run_output_dir else 'current directory'}")
    if MAKE_PHASE1_DIAGNOSTICS:
        print("Diagnostic files: conservation_history.csv, verification_diagnostics.csv, "
              "shock_jump_by_angle.csv, shock_jump_summary.csv, "
              "shock_structure_profiles.csv, pressure_gauges.csv, gauge_summary.csv, "
              "run_summary.csv, verification_diagnostics.png, "
              "shock_structure_diagnostics.png, pressure_gauges.png")
        if CONVERGENCE_REFERENCE_RUNS:
            print("Convergence comparison requested; see convergence_comparison.csv/png if matched runs were found.")


# =============================================================================
# PLOTTING AND POST-PROCESSING DIAGNOSTICS
# =============================================================================

def calculate_shock_front_history(snapshot_states):
    """Estimate radial-ray shock speed and Mach number between snapshots.

    The shock radius is the existing density-threshold estimate along rays
    from the burst center.  Its time derivative is a ray-propagation speed,
    not the exact normal speed of a strongly distorted shock surface.
    """
    ambient_sound_speed = float(sound_speed(rho0, p0))
    speed_history = []
    mach_history = []

    for i, state in enumerate(snapshot_states):
        speed = np.full_like(state["r_shock"], np.nan, dtype=float)
        mach = np.full_like(state["r_shock"], np.nan, dtype=float)

        if i > 0:
            previous = snapshot_states[i - 1]
            dt = state["time"] - previous["time"]
            if dt > 0.0:
                current_radius = state["r_shock"]
                previous_radius = previous["r_shock"]
                valid = (current_radius > 0.0) & (previous_radius > 0.0)
                speed_estimate = (current_radius - previous_radius) / dt
                # A detected inward movement is generally threshold/grid
                # jitter, not the outward-propagating leading shock.
                valid &= speed_estimate > 0.0
                speed[valid] = speed_estimate[valid]
                mach[valid] = speed[valid] / ambient_sound_speed

        speed_history.append(speed)
        mach_history.append(mach)

    return speed_history, mach_history


def determine_snapshot_plot_scales(state, shock_mach):
    """Determine color limits from one snapshot for per-image autoscaling."""
    rho = state["rho"]
    ur = state["ur"]
    uz = state["uz"]
    pressure = state["pressure"]

    density_ratio = rho / rho0
    pressure_ratio = pressure / p0
    overpressure_ratio = (pressure - p0) / p0
    speed = np.sqrt(ur**2 + uz**2)
    flow_mach = speed / np.maximum(sound_speed(rho, pressure), 1.0e-20)

    density_min = max(float(np.nanmin(density_ratio)), 1.0e-8)
    density_max = max(float(np.nanmax(density_ratio)), density_min * 1.001)
    pressure_min = max(float(np.nanmin(pressure_ratio)), 1.0e-8)
    pressure_max = max(float(np.nanmax(pressure_ratio)), pressure_min * 1.001)
    overpressure_limit = max(
        float(np.nanmax(np.abs(overpressure_ratio))), 1.0e-3
    )
    velocity_limit = max(
        float(np.nanmax(np.abs(ur))),
        float(np.nanmax(np.abs(uz))),
        1.0e-8,
    )
    speed_max = max(float(np.nanmax(speed)), 1.0e-12)
    speed_min = max(speed_max * 1.0e-5, 1.0e-12)
    flow_mach_max = max(float(np.nanmax(flow_mach)), 1.0e-12)
    flow_mach_min = max(flow_mach_max * 1.0e-5, 1.0e-12)

    shock_values = shock_mach[np.isfinite(shock_mach) & (shock_mach > 0.0)]
    if shock_values.size:
        shock_mach_min = max(float(np.min(shock_values)), 1.0e-3)
        shock_mach_max = max(
            float(np.max(shock_values)), shock_mach_min * 1.001
        )
    else:
        shock_mach_min, shock_mach_max = 1.0e-3, 1.0

    return {
        "density_vmin": density_min,
        "density_vmax": density_max,
        "pressure_vmin": pressure_min,
        "pressure_vmax": pressure_max,
        "overpressure_limit": overpressure_limit,
        "overpressure_linthresh": max(overpressure_limit * 1.0e-3, 1.0e-12),
        "velocity_limit": velocity_limit,
        "speed_vmin": speed_min,
        "speed_vmax": max(speed_max, speed_min * 1.001),
        "flow_mach_vmin": flow_mach_min,
        "flow_mach_vmax": max(flow_mach_max, flow_mach_min * 1.001),
        "shock_mach_vmin": shock_mach_min,
        "shock_mach_vmax": shock_mach_max,
    }


def save_spatial_snapshot(r_faces, z_faces, r_centers, z_centers,
                          rho, ur, uz, pressure, t, output_path, plot_scales,
                          theta, r_shock, shock_mach):
    """Save a spatial r-z snapshot using limits autoscaled for this image.

    The horizontal coordinate is radial distance from the symmetry axis (r),
    and the vertical coordinate is height above the reflecting ground (z).
    This is a meridional cross-section of the axisymmetric 3-D solution.
    """
    density_ratio = np.maximum(rho / rho0, 1.0e-12)
    pressure_ratio = np.maximum(pressure / p0, 1.0e-12)
    overpressure_ratio = (pressure - p0) / p0
    speed = np.sqrt(ur**2 + uz**2)
    local_sound_speed = sound_speed(rho, pressure)
    flow_mach = speed / np.maximum(local_sound_speed, 1.0e-20)

    fig, axes = plt.subplots(2, 4, figsize=(21, 11), constrained_layout=True)
    axes = axes.ravel()

    def decorate_spatial_axis(ax, title):
        ax.set_title(title)
        ax.set_xlabel("Radial distance from symmetry axis, r [m]")
        ax.set_ylabel("Height above reflecting ground, z [m]")
        ax.set_xlim(r_faces[0], r_faces[-1])
        ax.set_ylim(z_faces[0], z_faces[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.plot(0.0, HOB, marker="*", color="white", markeredgecolor="black",
                markersize=10, zorder=8)

    # 1. Density ratio: logarithmic scale autoscaled for this snapshot.
    ax = axes[0]
    im = ax.pcolormesh(
        r_faces, z_faces, density_ratio.T, shading="flat", cmap="viridis",
        norm=LogNorm(vmin=plot_scales["density_vmin"],
                     vmax=plot_scales["density_vmax"])
    )
    threshold = 1.0 + SHOCK_DENSITY_EXCESS
    if np.nanmin(density_ratio) <= threshold <= np.nanmax(density_ratio):
        ax.contour(r_centers, z_centers, density_ratio.T,
                   levels=[threshold], colors="white", linewidths=1.0)
    decorate_spatial_axis(ax, r"Density ratio, $\rho/\rho_0$")
    fig.colorbar(im, ax=ax, label=r"$\rho/\rho_0$ [dimensionless]")

    # 2. Pressure ratio, retained for comparison with the original output.
    ax = axes[1]
    im = ax.pcolormesh(
        r_faces, z_faces, pressure_ratio.T, shading="flat", cmap="inferno",
        norm=LogNorm(vmin=plot_scales["pressure_vmin"],
                     vmax=plot_scales["pressure_vmax"])
    )
    decorate_spatial_axis(ax, r"Pressure ratio, $p/p_0$")
    fig.colorbar(im, ax=ax, label=r"$p/p_0$ [dimensionless]")

    # 3. Overpressure ratio: zero is ambient pressure; negative values are
    # underpressure.  SymLogNorm displays both signs on this snapshot's scale.
    ax = axes[2]
    lim = plot_scales["overpressure_limit"]
    im = ax.pcolormesh(
        r_faces, z_faces, overpressure_ratio.T, shading="flat", cmap="coolwarm",
        norm=SymLogNorm(linthresh=plot_scales["overpressure_linthresh"],
                        vmin=-lim, vmax=lim, base=10)
    )
    decorate_spatial_axis(ax, r"Overpressure ratio, $\Delta p/p_0$")
    fig.colorbar(im, ax=ax, label=r"$(p-p_0)/p_0$ [dimensionless]")

    # 4. Radial velocity, using a symmetric range for this snapshot.
    ax = axes[3]
    vlim = plot_scales["velocity_limit"]
    im = ax.pcolormesh(
        r_faces, z_faces, ur.T, shading="flat", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    )
    decorate_spatial_axis(ax, r"Radial gas velocity, $u_r$")
    fig.colorbar(im, ax=ax, label=r"$u_r$ [m/s]")

    # 5. Vertical velocity, using the same per-snapshot range as radial velocity.
    ax = axes[4]
    im = ax.pcolormesh(
        r_faces, z_faces, uz.T, shading="flat", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    )
    decorate_spatial_axis(ax, r"Vertical gas velocity, $u_z$")
    fig.colorbar(im, ax=ax, label=r"$u_z$ [m/s]")

    # 6. Velocity magnitude with direction arrows.
    ax = axes[5]
    speed_floor = plot_scales["speed_vmin"]
    speed_for_plot = np.maximum(speed, speed_floor)
    im = ax.pcolormesh(
        r_faces, z_faces, speed_for_plot.T, shading="flat", cmap="magma",
        norm=LogNorm(vmin=plot_scales["speed_vmin"],
                     vmax=plot_scales["speed_vmax"])
    )
    stride_r = max(1, len(r_centers) // 24)
    stride_z = max(1, len(z_centers) // 24)
    Rq, Zq = np.meshgrid(r_centers, z_centers, indexing="xy")
    if float(np.nanmax(speed)) > 0.0:
        ax.quiver(
            Rq[::stride_z, ::stride_r], Zq[::stride_z, ::stride_r],
            ur.T[::stride_z, ::stride_r], uz.T[::stride_z, ::stride_r],
            color="white", alpha=0.75, pivot="mid"
        )
    decorate_spatial_axis(ax, r"Gas velocity magnitude, $|\mathbf{u}|$")
    fig.colorbar(im, ax=ax, label=r"$|\mathbf{u}|$ [m/s]")

    # 7. Fluid Mach number, distinct from the shock-front Mach number below.
    # Its logarithmic color limits are autoscaled for this snapshot.
    ax = axes[6]
    flow_mach_for_plot = np.maximum(flow_mach, plot_scales["flow_mach_vmin"])
    im = ax.pcolormesh(
        r_faces, z_faces, flow_mach_for_plot.T, shading="flat", cmap="cividis",
        norm=LogNorm(vmin=plot_scales["flow_mach_vmin"],
                     vmax=plot_scales["flow_mach_vmax"])
    )
    decorate_spatial_axis(ax, r"Fluid Mach number, $|\mathbf{u}|/c$")
    fig.colorbar(im, ax=ax, label="Fluid Mach number [dimensionless]")

    # 8. Color the detected shock front by its ray-based propagation Mach.
    # This is a front diagnostic, not a cell-centered fluid variable.
    ax = axes[7]
    decorate_spatial_axis(ax, "Detected shock front: radial-ray Mach number")
    valid_front = (np.isfinite(shock_mach) & (shock_mach > 0.0) & (r_shock > 0.0))
    if np.any(valid_front):
        r_front = r_shock[valid_front] * np.sin(theta[valid_front])
        z_front = HOB + r_shock[valid_front] * np.cos(theta[valid_front])
        points = ax.scatter(
            r_front, z_front, c=shock_mach[valid_front], s=15,
            cmap="plasma",
            norm=LogNorm(vmin=plot_scales["shock_mach_vmin"],
                         vmax=plot_scales["shock_mach_vmax"]),
            edgecolors="none", zorder=5
        )
        fig.colorbar(points, ax=ax, label="Radial-ray shock Mach [dimensionless]")
    else:
        ax.text(0.5, 0.5,
                "Not yet available\n(requires two snapshots with a detected front)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)

    fig.suptitle(
        "Axisymmetric blast-wave flow field\n"
        f"Simulation time = {t:.5e} s | HOB = {HOB:g} m | "
        "individually autoscaled color limits",
        fontsize=15, fontweight="bold"
    )
    fig.savefig(output_path, dpi=SNAPSHOT_DPI)
    plt.close(fig)


def _finite_summary(values, percentile):
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return np.nan
    return float(np.percentile(finite, percentile))


def make_diagnostic_plots(theta, r_shock, times, eta, tau,
                          volume_ratio, energy_history, t_interaction,
                          initial_energy, snapshot_states, shock_speed_history,
                          shock_mach_history, output_path):
    """Save time-history and shock-front diagnostics separately from maps."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
    axes = axes.ravel()

    # 1. Final density-threshold shock radius.
    ax = axes[0]
    ax.plot(theta * 180.0 / np.pi, r_shock, "k-", lw=1.8)
    ax.set_xlabel(r"Polar angle from upward vertical, $\theta$ [degrees]")
    ax.set_ylabel("Estimated shock distance from burst center [m]")
    ax.set_title("Final threshold-based shock-front radius")
    ax.grid(True, which="both", ls=":")

    # 2. Enclosed-volume history.
    ax = axes[1]
    if len(times) > 0:
        ax.plot(times, volume_ratio, lw=2, label=r"$V/V_{\mathrm{ST,sphere}}$")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="Free-air sphere reference")
        ax.axhline(0.5, color="gray", ls=":", lw=1.5, label="Hemisphere reference")
        ax.set_xscale("log")
        ax.set_xlabel("Simulation time [s]")
        ax.set_ylabel("Estimated volume ratio [dimensionless]")
        ax.set_title("Enclosed-volume history")
        ax.grid(True, which="both", ls=":")
        ax.legend()

    # 3. Excess energy history.
    ax = axes[2]
    if len(times) > 0:
        ax.plot(times, energy_history / max(initial_energy, 1.0e-30), lw=2)
        ax.set_xscale("log")
        ax.set_xlabel("Simulation time [s]")
        ax.set_ylabel("Excess gas energy / initial excess energy")
        ax.set_title("Energy diagnostic")
        ax.grid(True, which="both", ls=":")

    # 4. Dimensionless height-of-burst evolution.
    ax = axes[3]
    if len(times) > 0:
        ax.plot(times, eta, label=r"$\eta=H/R_{\mathrm{ST}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Simulation time [s]")
        ax.set_ylabel(r"Scaled height of burst, $\eta$")
        if t_interaction > 0.0:
            ax2 = ax.twinx()
            ax2.plot(times, tau, "C1--", label=r"$\tau=t/t_H$")
            ax2.set_yscale("log")
            ax2.set_ylabel(r"Scaled time, $\tau$")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax.legend()
        ax.set_title("Dimensionless height-of-burst evolution")
        ax.grid(True, which="both", ls=":")

    # Get the final snapshot's angle-dependent ray speed and Mach number.
    final_speed = shock_speed_history[-1] if shock_speed_history else np.array([])
    final_mach = shock_mach_history[-1] if shock_mach_history else np.array([])
    final_theta = snapshot_states[-1]["theta"] if snapshot_states else np.array([])

    # 5. Final radial-ray propagation speed versus angle.
    ax = axes[4]
    if final_speed.size and np.any(np.isfinite(final_speed)):
        ax.plot(final_theta * 180.0 / np.pi, final_speed, "C2-", lw=1.8)
    else:
        ax.text(0.5, 0.5, "No valid front-speed estimate yet",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel(r"Polar angle from upward vertical, $\theta$ [degrees]")
    ax.set_ylabel("Radial-ray shock speed [m/s]")
    ax.set_title("Final shock-front radial propagation speed")
    ax.grid(True, which="both", ls=":")

    # 6. Final radial-ray shock Mach number versus angle.
    ax = axes[5]
    if final_mach.size and np.any(np.isfinite(final_mach)):
        ax.plot(final_theta * 180.0 / np.pi, final_mach, "C3-", lw=1.8)
        ax.axhline(1.0, color="k", ls="--", lw=1, label="Ambient sound speed")
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid front-Mach estimate yet",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel(r"Polar angle from upward vertical, $\theta$ [degrees]")
    ax.set_ylabel("Radial-ray shock Mach [dimensionless]")
    ax.set_title("Final shock-front radial-ray Mach number")
    ax.grid(True, which="both", ls=":")

    # Summarize per-snapshot angular distributions using median and 90th
    # percentile to reduce sensitivity to isolated noisy angular bins.
    snapshot_times = np.asarray([st["time"] for st in snapshot_states], dtype=float)
    speed_median = np.asarray([_finite_summary(x, 50) for x in shock_speed_history])
    speed_p90 = np.asarray([_finite_summary(x, 90) for x in shock_speed_history])
    mach_median = np.asarray([_finite_summary(x, 50) for x in shock_mach_history])
    mach_p90 = np.asarray([_finite_summary(x, 90) for x in shock_mach_history])

    # 7. Shock propagation speed history.
    ax = axes[6]
    valid = np.isfinite(speed_median)
    if np.any(valid):
        ax.plot(snapshot_times[valid], speed_median[valid], "C2-", lw=2,
                label="Median over detected angular bins")
        valid90 = np.isfinite(speed_p90)
        ax.plot(snapshot_times[valid90], speed_p90[valid90], "C2--", lw=1.5,
                label="90th percentile")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid front-speed estimates",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Radial-ray shock speed [m/s]")
    ax.set_title("Shock-front propagation speed history")
    ax.grid(True, which="both", ls=":")

    # 8. Shock Mach history, normalized by the undisturbed ambient sound speed.
    ax = axes[7]
    valid = np.isfinite(mach_median)
    if np.any(valid):
        ax.plot(snapshot_times[valid], mach_median[valid], "C3-", lw=2,
                label="Median over detected angular bins")
        valid90 = np.isfinite(mach_p90)
        ax.plot(snapshot_times[valid90], mach_p90[valid90], "C3--", lw=1.5,
                label="90th percentile")
        ax.axhline(1.0, color="k", ls=":", lw=1.2,
                   label="Ambient sound speed (Mach 1)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid front-Mach estimates",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Radial-ray shock Mach [dimensionless]")
    ax.set_title("Shock-front radial-ray Mach history")
    ax.grid(True, which="both", ls=":")

    fig.suptitle(
        "Blast-wave diagnostics\n"
        "Shock speed/Mach are density-threshold radial-ray estimates",
        fontsize=14, fontweight="bold"
    )
    fig.savefig(output_path, dpi=SNAPSHOT_DPI, bbox_inches="tight")
    plt.close(fig)



def plot_conservation_diagnostics(output_dir, history):
    """Plot budget closure and remaining/cumulative mass-energy terms."""
    t = np.asarray(history["time_s"], dtype=float)
    if t.size == 0:
        return

    e0 = max(abs(history["excess_energy_remaining_J"][0]), 1.0e-30)
    m0 = max(abs(history["mass_remaining_kg"][0]), 1.0e-30)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t, np.asarray(history["excess_energy_remaining_J"]) / e0,
            label="Measured remaining excess energy")
    ax.plot(t, np.asarray(history["budget_expected_energy_J"]) / e0,
            "--", label="Budget-predicted remaining energy")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Energy / initial excess energy")
    ax.set_title("Excess-energy accounting")
    ax.grid(True, ls=":")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, history["energy_budget_residual_percent"])
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Energy-budget residual [% of initial excess energy]")
    ax.set_title("Energy conservation closure")
    ax.grid(True, ls=":")

    ax = axes[1, 0]
    ax.plot(t, np.asarray(history["mass_remaining_kg"]) / m0,
            label="Measured remaining mass")
    ax.plot(t, np.asarray(history["budget_expected_mass_kg"]) / m0,
            "--", label="Budget-predicted remaining mass")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Mass / initial domain mass")
    ax.set_title("Mass accounting")
    ax.grid(True, ls=":")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, history["mass_budget_residual_percent"])
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Mass-budget residual [% of initial domain mass]")
    ax.set_title("Mass conservation closure")
    ax.grid(True, ls=":")

    fig.suptitle(
        "Conservation accounting: signed open-boundary fluxes; no cell-local positivity repairs",
        fontsize=12, fontweight="bold"
    )
    fig.savefig(Path(output_dir) / "conservation_diagnostics.png", dpi=SNAPSHOT_DPI)
    plt.close(fig)

def create_mp4_from_snapshots(snapshot_dir, fps=2,
                              output_name="blast_wave.mp4"):
    """
    Assemble saved spatial snapshot PNGs into an MP4 video.

    Images are automatically padded to a common frame size.
    Their original aspect ratios and visual content are preserved.
    """

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "MP4 generation requires imageio and imageio-ffmpeg. "
            "Install with: pip install imageio imageio-ffmpeg"
        ) from exc

    snapshot_dir = Path(snapshot_dir)

    image_files = sorted(snapshot_dir.glob("snapshot_*.png"))

    if len(image_files) < 2:
        raise ValueError(
            f"At least two snapshot PNG files are required in {snapshot_dir}."
        )

    if fps <= 0:
        raise ValueError("Video frame rate must be positive.")

    output_path = snapshot_dir / output_name

    # ---------------------------------------------------------
    # First pass: determine the largest image dimensions.
    # ---------------------------------------------------------
    image_dimensions = []

    for image_path in image_files:
        frame = imageio.imread(image_path)

        height, width = frame.shape[:2]
        image_dimensions.append((height, width))

    max_height = max(h for h, w in image_dimensions)
    max_width = max(w for h, w in image_dimensions)

    # H.264/yuv420p requires even frame dimensions.
    video_height = max_height + (max_height % 2)
    video_width = max_width + (max_width % 2)

    print("\nGenerating MP4 video...")
    print(f"  Number of frames : {len(image_files)}")
    print(f"  Frame rate       : {fps} FPS")
    print(f"  Video dimensions : {video_width} x {video_height}")

    # ---------------------------------------------------------
    # Second pass: encode each image after padding.
    # ---------------------------------------------------------
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None
    ) as writer:

        for i, image_path in enumerate(image_files):

            frame = imageio.imread(image_path)

            # Convert grayscale images to RGB.
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)

            # Composite RGBA images against a white background.
            elif frame.shape[2] == 4:
                frame = frame.astype(np.float32)
                alpha = frame[:, :, 3:4] / 255.0
                frame = (
                    frame[:, :, :3] * alpha
                    + 255.0 * (1.0 - alpha)
                )

            frame = np.clip(frame, 0, 255).astype(np.uint8)

            height, width = frame.shape[:2]

            # Create a white canvas with the common video dimensions.
            canvas = np.full(
                (video_height, video_width, 3),
                255,
                dtype=np.uint8
            )

            # Center the original image without resizing it.
            top = (video_height - height) // 2
            left = (video_width - width) // 2

            canvas[
                top:top + height,
                left:left + width
            ] = frame

            writer.append_data(canvas)

            print(
                f"\r  Encoding frame {i + 1}/{len(image_files)}",
                end=""
            )

    print(f"\n\nMP4 successfully saved to:\n  {output_path.resolve()}")

    return output_path


if __name__ == "__main__":
    main()
