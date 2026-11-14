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
  - First-order finite-volume method.
  - Local Rusanov (local Lax-Friedrichs) numerical flux.
  - Cylindrical cell volumes and face areas are included explicitly.
  - Explicit CFL time stepping.

Phase-1 verification and diagnostics:
  - Free-air Sedov-Taylor radius comparison in the unreflected, point-source-like
    time window (diagnostic comparison, not an automatic pass/fail certification).
  - Approximate Rankine-Hugoniot jump checks using the detected front speed and
    a narrow post-front sampling band.
  - Mass/energy budgets including signed energy and mass flow through open
    boundaries and changes caused by positivity floors.
  - Configurable virtual pressure gauges, peak overpressure, threshold arrival
    time, and positive-phase impulse.
  - Run summaries and optional comparison of separately executed grid/domain
    cases to support convergence studies.

Important limitations:
  - This is an inviscid ideal-gas model, not a soil/ground-response model.
  - The initial energy deposition has finite radius; grid studies must hold its
    physical width fixed to isolate discretization effects.
  - First-order Rusanov fluxes are robust but numerically diffusive.
  - The shock-front/volume diagnostic uses a density threshold and is therefore
    an operational numerical estimate, not an exact discontinuity location.
  - Sedov and Rankine-Hugoniot comparisons are approximate diagnostics subject
    to finite deposition size, grid resolution, front detection, and boundary
    effects. They are not substitutes for formal verification and validation.

Requires: numpy, matplotlib; imageio and imageio-ffmpeg only for MP4 output
"""

from datetime import datetime
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm


# =============================================================================
# USER INPUTS
# =============================================================================

# Explosion and ambient atmosphere
E_kt = 10.0                 # explosion energy [kt]
HOB = 220.0                # height of burst above reflecting plane [m]
rho0 = 1.225               # ambient density [kg/m^3]
p0 = 1.01325e5              # ambient pressure [Pa]
gamma = 1.4                # ratio of specific heats

# Time interval
t_end = 0.1               # simulation end time [s]
CFL = 0.40                 # Courant number; keep <= 0.5 for this scheme
MAX_STEPS = 500000

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
SHOCK_JUMP_BAND_CELLS = 4.0  # post-front sampling-band width in cell diagonals

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


def apply_boundaries(U):
    """
    Construct radial and vertical face states.

    r=0: axis of symmetry (normal velocity ur is reflected).
    r=r_max: zero-gradient outflow.
    z=0: rigid reflecting wall (normal velocity uz is reflected).
    z=z_max: zero-gradient outflow.
    """
    # Radial faces: arrays have shape (4, NR+1, NZ)
    UL_r = np.empty((4, U.shape[1] + 1, U.shape[2]))
    UR_r = np.empty_like(UL_r)

    # Interior faces: at face i+1/2, left cell is i and right cell is i+1.
    UL_r[:, 1:-1, :] = U[:, :-1, :]
    UR_r[:, 1:-1, :] = U[:, 1:, :]

    # Axis: reflected radial velocity in ghost state
    axis_ghost = U[:, 0, :].copy()
    axis_ghost[1] *= -1.0
    UL_r[:, 0, :] = axis_ghost
    UR_r[:, 0, :] = U[:, 0, :]

    # Outer radial outflow
    UL_r[:, -1, :] = U[:, -1, :]
    UR_r[:, -1, :] = U[:, -1, :]

    # Vertical faces: arrays have shape (4, NR, NZ+1)
    UL_z = np.empty((4, U.shape[1], U.shape[2] + 1))
    UR_z = np.empty_like(UL_z)

    # Interior faces
    UL_z[:, :, 1:-1] = U[:, :, :-1]
    UR_z[:, :, 1:-1] = U[:, :, 1:]

    # Ground: reflected vertical velocity in ghost state
    ground_ghost = U[:, :, 0].copy()
    ground_ghost[2] *= -1.0
    UL_z[:, :, 0] = ground_ghost
    UR_z[:, :, 0] = U[:, :, 0]

    # Top outflow
    UL_z[:, :, -1] = U[:, :, -1]
    UR_z[:, :, -1] = U[:, :, -1]

    return UL_r, UR_r, UL_z, UR_z


def stable_timestep(U, dr, dz):
    rho, ur, uz, pressure = primitive_from_conserved(U)
    c = sound_speed(rho, pressure)
    max_rate = np.max((np.abs(ur) + c) / dr + (np.abs(uz) + c) / dz)
    return CFL / max(max_rate, 1.0e-30)


def advance_one_step(U, dt, r_faces, dz, cell_volumes, return_budget=False):
    """One explicit finite-volume update with cylindrical geometry.

    When return_budget=True, also return signed boundary fluxes and positivity-
    floor corrections.  These calculations do not alter the numerical update.
    """
    UL_r, UR_r, UL_z, UR_z = apply_boundaries(U)

    Fr = rusanov_flux(UL_r, UR_r, "r")
    Fz = rusanov_flux(UL_z, UR_z, "z")

    # Face areas (axisymmetric 3-D volumes)
    radial_areas = (2.0 * np.pi * r_faces * dz)[None, :, None]
    axial_areas = np.pi * (r_faces[1:]**2 - r_faces[:-1]**2)

    radial_net = (Fr[:, 1:, :] * radial_areas[:, 1:, :]
                  - Fr[:, :-1, :] * radial_areas[:, :-1, :])
    vertical_net = (Fz[:, :, 1:] * axial_areas[None, :, None]
                    - Fz[:, :, :-1] * axial_areas[None, :, None])

    U_new = U - dt * (radial_net + vertical_net) / cell_volumes[None, :, :]

    # Signed outward flux through the two open boundaries.  The axis and
    # reflecting ground have zero normal mass/energy flux by construction.
    outer_radial_area = 2.0 * np.pi * r_faces[-1] * dz
    boundary_mass_out = dt * (
        np.sum(Fr[0, -1, :]) * outer_radial_area
        + np.sum(Fz[0, :, -1] * axial_areas)
    )
    boundary_energy_out = dt * (
        np.sum(Fr[3, -1, :]) * outer_radial_area
        + np.sum(Fz[3, :, -1] * axial_areas)
    )

    # Cylindrical geometric source in radial momentum:
    # volume-average(p/r) = 2*p/(r_outer + r_inner).
    rho, ur, uz, pressure = primitive_from_conserved(U)
    r_mid_sum = r_faces[1:] + r_faces[:-1]
    radial_source = 2.0 * pressure / np.maximum(r_mid_sum[:, None], 1.0e-30)
    U_new[1] += dt * radial_source

    # Guard against tiny negative internal energies from roundoff.  Record the
    # exact energy/mass added by the existing positivity floors for accounting.
    energy_before_floor = U_new[3].copy()
    mass_before_floor = U_new[0].copy()
    rho_n, ur_n, uz_n, p_n = primitive_from_conserved(U_new)
    min_internal = 1.0e-10
    kinetic_n = 0.5 * rho_n * (ur_n**2 + uz_n**2)
    U_new[3] = np.maximum(U_new[3], kinetic_n + min_internal / (gamma - 1.0))
    U_new[0] = np.maximum(U_new[0], 1.0e-12)

    if return_budget:
        energy_floor_added = np.sum((U_new[3] - energy_before_floor) * cell_volumes)
        mass_floor_added = np.sum((U_new[0] - mass_before_floor) * cell_volumes)
        budget = {
            "boundary_mass_out_kg": float(boundary_mass_out),
            "boundary_energy_out_J": float(boundary_energy_out),
            "mass_floor_added_kg": float(mass_floor_added),
            "energy_floor_added_J": float(energy_floor_added),
        }
        return U_new, budget

    return U_new


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


def sample_front_jump(r_centers, z_centers, dr, dz, state, shock_speed, shock_mach):
    """Approximate post-front state sampling for a Rankine-Hugoniot comparison.

    For each angular bin with a valid radial-ray speed, take the maximum density
    and pressure in a narrow band behind the detected threshold front.  Because
    the front speed, front location, and cell sampling are approximate, this is
    a consistency diagnostic rather than a formal shock-jump verification.
    """
    rho = state["rho"]
    pressure = state["pressure"]
    theta_centers = state["theta"]
    r_shock = state["r_shock"]
    if not len(theta_centers):
        return {}

    rr, zz = np.meshgrid(r_centers, z_centers, indexing="ij")
    distance = np.sqrt(rr**2 + (zz - HOB)**2)
    theta_cells = np.arctan2(rr, zz - HOB)
    edges = np.linspace(0.0, np.pi, N_ANGLE_BINS + 1)
    bins = np.digitize(theta_cells.ravel(), edges) - 1
    mask = (rho > rho0 * (1.0 + SHOCK_DENSITY_EXCESS)).ravel()
    masked_bins = bins[mask]
    masked_distance = distance.ravel()[mask]
    masked_density = rho.ravel()[mask]
    masked_pressure = pressure.ravel()[mask]
    masked_flat_indices = np.flatnonzero(mask)
    ur_flat = state["ur"].ravel()
    uz_flat = state["uz"].ravel()
    band_width = SHOCK_JUMP_BAND_CELLS * np.hypot(dr, dz)

    observed_density = []
    observed_pressure = []
    predicted_density = []
    predicted_pressure = []
    observed_radial_velocity = []
    predicted_radial_velocity = []
    used_mach = []

    valid_bins = np.flatnonzero(
        (r_shock > 0.0) & np.isfinite(shock_mach) & (shock_mach > 1.0)
    )
    for k in valid_bins:
        local = (masked_bins == k) & (masked_distance >= max(0.0, r_shock[k] - band_width))
        if not np.any(local):
            continue
        mach = float(shock_mach[k])
        rho_rh, p_rh = rankine_hugoniot_ratios(mach)
        local_indices = np.flatnonzero(local)
        best_local_index = local_indices[int(np.argmax(masked_density[local]))]
        flat_index = masked_flat_indices[best_local_index]
        theta_k = theta_centers[k]
        measured_u_radial = (
            ur_flat[flat_index] * np.sin(theta_k)
            + uz_flat[flat_index] * np.cos(theta_k)
        )
        predicted_u_radial = float(shock_speed[k]) * (1.0 - 1.0 / float(rho_rh))

        observed_density.append(float(np.max(masked_density[local]) / rho0))
        observed_pressure.append(float(np.max(masked_pressure[local]) / p0))
        predicted_density.append(float(rho_rh))
        predicted_pressure.append(float(p_rh))
        observed_radial_velocity.append(float(measured_u_radial))
        predicted_radial_velocity.append(float(predicted_u_radial))
        used_mach.append(mach)

    if not used_mach:
        return {}
    return {
        "valid_angle_bins": int(len(used_mach)),
        "median_front_mach": float(np.median(used_mach)),
        "observed_density_ratio_median": float(np.median(observed_density)),
        "rh_density_ratio_median": float(np.median(predicted_density)),
        "observed_pressure_ratio_median": float(np.median(observed_pressure)),
        "rh_pressure_ratio_median": float(np.median(predicted_pressure)),
        "observed_postshock_radial_velocity_median_m_s": float(np.median(observed_radial_velocity)),
        "rh_postshock_radial_velocity_median_m_s": float(np.median(predicted_radial_velocity)),
    }


def calculate_verification_rows(snapshot_states, shock_speed_history,
                                 shock_mach_history, r_centers, z_centers,
                                 dr, dz, deposition_width, r_max, z_max):
    """Build Sedov and Rankine-Hugoniot comparison rows for captured states."""
    rows = []
    gamma_matches_xi0 = np.isclose(gamma, 1.4, rtol=0.0, atol=1.0e-8)
    nearest_boundary = min(HOB, r_max, z_max - HOB) if HOB > 0.0 else 0.0

    for i, state in enumerate(snapshot_states):
        time_s = float(state["time"])
        detected = state["r_shock"][state["r_shock"] > 0.0]
        radius_median = float(np.median(detected)) if detected.size else np.nan
        radius_p90 = float(np.percentile(detected, 90)) if detected.size else np.nan
        r_st = float(sedov_radius(max(time_s, 1.0e-30))) if time_s > 0.0 else np.nan

        detected_angle_fraction = float(np.count_nonzero(state["r_shock"] > 0.0) / max(len(state["r_shock"]), 1))
        sedov_valid = bool(
            gamma_matches_xi0 and HOB > 0.0 and np.isfinite(r_st)
            and r_st >= SEDOV_MIN_RADIUS_TO_DEPOSITION * deposition_width
            and r_st <= SEDOV_CLEARANCE_FRACTION * nearest_boundary
            and detected_angle_fraction >= SEDOV_MIN_DETECTED_ANGLE_FRACTION
        )
        sedov_ratio = radius_median / r_st if sedov_valid and np.isfinite(radius_median) else np.nan

        speed = shock_speed_history[i] if i < len(shock_speed_history) else np.full(N_ANGLE_BINS, np.nan)
        mach = shock_mach_history[i] if i < len(shock_mach_history) else np.full(N_ANGLE_BINS, np.nan)
        usable_speeds = speed[np.isfinite(speed) & (speed > 0.0)]
        speed_median = float(np.median(usable_speeds)) if usable_speeds.size else np.nan
        sedov_speed = alpha * r_st / time_s if time_s > 0.0 and np.isfinite(r_st) else np.nan
        sedov_speed_ratio = (
            speed_median / sedov_speed
            if sedov_valid and np.isfinite(speed_median) and sedov_speed > 0.0
            else np.nan
        )
        jump = sample_front_jump(
            r_centers, z_centers, dr, dz, state, speed, mach
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
            "front_sampling_band_width_m": SHOCK_JUMP_BAND_CELLS * np.hypot(dr, dz),
        }
        row.update(jump)
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
    """Write Sedov/RH verification diagnostics and a compact figure."""
    output_dir = Path(output_dir)
    path = output_dir / "verification_diagnostics.csv"
    fields = [
        "time_s", "sedov_reference_radius_m", "detected_radius_median_m",
        "detected_radius_p90_m", "detected_angle_fraction", "sedov_comparison_valid",
        "detected_to_sedov_radius_ratio", "sedov_reference_shock_speed_m_s",
        "detected_radial_ray_speed_median_m_s", "detected_to_sedov_speed_ratio",
        "gamma14_xi0_applicable", "front_sampling_band_width_m", "valid_angle_bins",
        "median_front_mach", "observed_density_ratio_median", "rh_density_ratio_median",
        "observed_pressure_ratio_median", "rh_pressure_ratio_median",
        "observed_postshock_radial_velocity_median_m_s",
        "rh_postshock_radial_velocity_median_m_s",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    sedov_valid = np.asarray([bool(row["sedov_comparison_valid"]) for row in rows])
    sedov_ratio = np.asarray([row["detected_to_sedov_radius_ratio"] for row in rows], dtype=float)
    rh_time = np.asarray([row["time_s"] for row in rows if row.get("valid_angle_bins", 0)], dtype=float)
    rho_obs = np.asarray([row["observed_density_ratio_median"] for row in rows if row.get("valid_angle_bins", 0)], dtype=float)
    rho_rh = np.asarray([row["rh_density_ratio_median"] for row in rows if row.get("valid_angle_bins", 0)], dtype=float)
    p_obs = np.asarray([row["observed_pressure_ratio_median"] for row in rows if row.get("valid_angle_bins", 0)], dtype=float)
    p_rh = np.asarray([row["rh_pressure_ratio_median"] for row in rows if row.get("valid_angle_bins", 0)], dtype=float)
    mach = np.asarray([row["median_front_mach"] for row in rows if row.get("valid_angle_bins", 0)], dtype=float)

    sedov_speed_ratio = np.asarray([row["detected_to_sedov_speed_ratio"] for row in rows], dtype=float)
    velocity_obs = np.asarray([row.get("observed_postshock_radial_velocity_median_m_s", np.nan)
                               for row in rows if row.get("valid_angle_bins", 0)], dtype=float)
    velocity_rh = np.asarray([row.get("rh_postshock_radial_velocity_median_m_s", np.nan)
                              for row in rows if row.get("valid_angle_bins", 0)], dtype=float)
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
    if rh_time.size:
        ax.plot(rh_time, rho_obs, "o-", label="Sampled post-front density ratio")
        ax.plot(rh_time, rho_rh, "s--", label="Rankine-Hugoniot prediction")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No usable shock-speed samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(r"Density ratio, $\rho_2/\rho_1$")
    ax.set_title("Approximate shock compression check")
    ax.grid(True, which="both", ls=":")

    ax = axes[1, 1]
    if rh_time.size:
        ax.plot(rh_time, p_obs, "o-", label="Sampled post-front pressure ratio")
        ax.plot(rh_time, p_rh, "s--", label="Rankine-Hugoniot prediction")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No usable shock-speed samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(r"Pressure ratio, $p_2/p_1$")
    ax.set_title("Approximate shock pressure-jump check")
    ax.grid(True, which="both", ls=":")

    ax = axes[2, 0]
    if rh_time.size:
        ax.plot(rh_time, velocity_obs, "o-", label="Sampled post-front radial velocity")
        ax.plot(rh_time, velocity_rh, "s--", label="Rankine-Hugoniot prediction")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No usable shock-speed samples",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Post-front radial velocity [m/s]")
    ax.set_title("Approximate post-shock velocity check")
    ax.grid(True, which="both", ls=":")

    ax = axes[2, 1]
    ax.plot(times, detected_fraction, "o-")
    ax.axhline(SEDOV_MIN_DETECTED_ANGLE_FRACTION, color="k", ls="--", lw=1,
               label="Minimum fraction for Sedov comparison")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Fraction of detected angular bins")
    ax.set_title("Shock-front angular coverage")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Phase-1 verification diagnostics\n"
        "Sedov comparisons require an unreflected, point-source-like full front; "
        "Rankine-Hugoniot sampling is approximate",
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
        "run_label", "NR", "NZ", "DOMAIN_FACTOR", "dr_m", "dz_m", "max_cell_width_m",
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
    if SHOCK_JUMP_BAND_CELLS <= 0.0:
        raise ValueError("SHOCK_JUMP_BAND_CELLS must be positive.")

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
    while t < t_end and step < MAX_STEPS:
        dt = stable_timestep(U, dr, dz)
        dt = min(dt, t_end - t)

        U, step_budget = advance_one_step(
            U, dt, r_faces, dz, cell_volumes, return_budget=True
        )
        cumulative_boundary_mass_out += step_budget["boundary_mass_out_kg"]
        cumulative_boundary_energy_out += step_budget["boundary_energy_out_J"]
        cumulative_floor_mass_added += step_budget["mass_floor_added_kg"]
        cumulative_floor_energy_added += step_budget["energy_floor_added_J"]
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
                f"E-budget residual={energy_residual_pct:+.3e}%"
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
    print("NOTE: Shock radius/volume use a density threshold; RH front sampling is approximate.")
    print("The conservation budget accounts for signed outflow through the top/outer radial")
    print("boundaries and energy/mass introduced by positivity floors.")

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
        verification_rows = calculate_verification_rows(
            snapshot_states, shock_speed_history, shock_mach_history,
            r_centers, z_centers, dr, dz, deposition_width, r_max, z_max
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
            "NR": int(NR),
            "NZ": int(NZ),
            "DOMAIN_FACTOR": float(DOMAIN_FACTOR),
            "dr_m": float(dr),
            "dz_m": float(dz),
            "max_cell_width_m": float(max(dr, dz)),
            "deposition_width_m": float(deposition_width),
            "shock_density_excess": float(SHOCK_DENSITY_EXCESS),
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
        print("Phase-1 files: conservation_history.csv, verification_diagnostics.csv, "
              "pressure_gauges.csv, gauge_summary.csv, run_summary.csv, "
              "verification_diagnostics.png, pressure_gauges.png")
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
        "Conservation accounting: signed open-boundary fluxes and positivity-floor corrections",
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
