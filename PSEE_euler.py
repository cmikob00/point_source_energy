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

Important limitations:
  - This is an inviscid ideal-gas model, not a soil/ground-response model.
  - The initial energy deposition has finite radius; results depend somewhat
    on deposition radius and grid resolution until convergence is established.
  - First-order Rusanov fluxes are robust but numerically diffusive.
  - The shock-front/volume diagnostic uses a density threshold and is therefore
    an operational numerical estimate, not an exact discontinuity location.
  - Validate grid convergence and the free-air/surface limiting cases before
    using results quantitatively.

Requires: numpy, matplotlib
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm


# =============================================================================
# USER INPUTS
# =============================================================================

# Explosion and ambient atmosphere
E_kt = 330.0                 # explosion energy [kt]
HOB = 500.0                # height of burst above reflecting plane [m]
rho0 = 1.15               # ambient density [kg/m^3]
p0 = 1.01325e5              # ambient pressure [Pa]
gamma = 1.4                # ratio of specific heats

# Time interval
t_end = 0.3               # simulation end time [s]
CFL = 0.40                 # Courant number; keep <= 0.5 for this scheme
MAX_STEPS = 500000

# Computational domain / resolution
NR = 180                   # number of radial cells
NZ = 360                   # number of vertical cells
DOMAIN_FACTOR = 1.30       # domain extends this many free-air reference radii
DEPOSITION_CELLS = 2.5     # Gaussian energy-deposition width in cell widths

# Shock-front diagnostic
SHOCK_DENSITY_EXCESS = 0.02  # shock mask: rho > rho0*(1 + this value)
N_ANGLE_BINS = 240

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

    a = DEPOSITION_CELLS * max(dr, dz)
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


def advance_one_step(U, dt, r_faces, dz, cell_volumes):
    """One explicit finite-volume update with cylindrical geometry."""
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

    # Cylindrical geometric source in radial momentum:
    # volume-average(p/r) = 2*p/(r_outer + r_inner).
    rho, ur, uz, pressure = primitive_from_conserved(U)
    r_mid_sum = r_faces[1:] + r_faces[:-1]
    radial_source = 2.0 * pressure / np.maximum(r_mid_sum[:, None], 1.0e-30)
    U_new[1] += dt * radial_source

    # Guard against tiny negative internal energies from roundoff.
    rho_n, ur_n, uz_n, p_n = primitive_from_conserved(U_new)
    min_internal = 1.0e-10
    kinetic_n = 0.5 * rho_n * (ur_n**2 + uz_n**2)
    U_new[3] = np.maximum(U_new[3], kinetic_n + min_internal / (gamma - 1.0))
    U_new[0] = np.maximum(U_new[0], 1.0e-12)

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
    return np.sum((U[3] - ambient_energy_density) * cell_volumes)


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

    # Time-history diagnostics (the solver itself is unchanged).
    times = []
    eta_history = []
    tau_history = []
    volume_history = []
    volume_ratio_history = []
    radius_history = []
    energy_history = []

    # Snapshot states are retained until the run finishes for post-processing
    # of shock-front speed and for rendering the saved spatial images.  This
    # affects only output; these arrays are never fed back into the Euler solver.
    snapshot_states = []
    run_output_dir = None
    snapshot_index = 0
    target_times = sorted(set(float(x) for x in SNAPSHOT_TIMES if 0.0 < x <= t_end))
    next_target_index = 0

    if MAKE_PLOTS:
        if any(x <= 0.0 for x in SNAPSHOT_TIMES):
            raise ValueError("All SNAPSHOT_TIMES must be positive.")
        run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_output_dir = Path(SNAPSHOT_OUTPUT_DIR) / run_stamp
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Spatial snapshots will be saved in: {run_output_dir.resolve()}")

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

    print("=" * 76)
    print("AXISYMMETRIC 3-D BLAST WAVE / REFLECTING-GROUND EULER SOLVER")
    print("=" * 76)
    print(f"Explosion energy                 : {E:.5e} J")
    print(f"Height of burst                  : {HOB:.5g} m")
    print(f"Ambient density                  : {rho0:.5g} kg/m^3")
    print(f"gamma                            : {gamma:.5g}")
    print(f"Grid                             : {NR} x {NZ}")
    print(f"Domain                           : r=[0,{r_max:.4g}] m, z=[0,{z_max:.4g}] m")
    print(f"Cell dimensions                  : dr={dr:.4g} m, dz={dz:.4g} m")
    print(f"Gaussian deposition width        : {deposition_width:.4g} m")
    print(f"Free-air ST radius at t_end      : {R_end_ref:.5g} m")
    if HOB > 0.0:
        print(f"Characteristic interaction time  : {t_interaction:.5e} s")
    else:
        print("Characteristic interaction time  : 0 (burst at surface)")
    if MAKE_PLOTS:
        print("Snapshot targets [s]             : " + ", ".join(f"{x:.3g}" for x in target_times))
    print()

    initial_energy = total_energy(U, cell_volumes)
    print(f"Initial deposited energy         : {initial_energy:.6e} J")
    print()

    t = 0.0
    step = 0
    while t < t_end and step < MAX_STEPS:
        dt = stable_timestep(U, dr, dz)
        dt = min(dt, t_end - t)

        U = advance_one_step(U, dt, r_faces, dz, cell_volumes)
        t += dt
        step += 1

        # Capture the current state when a logarithmic output target is crossed.
        # Snapshot generation does not modify the flow state or time step.
        if MAKE_PLOTS and next_target_index < len(target_times):
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
                print(f"  captured spatial state at t={t:.5e} s "
                      f"(target {target_times[next_target_index]:.5e} s)")
                next_target_index += 1
                # A single numerical step can cross more than one target.  Do
                # not fabricate intermediate states; skip those targets and
                # report this limitation explicitly.
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
            max_radius = np.max(r_shock)
            e_now = total_energy(U, cell_volumes)

            times.append(t)
            eta_history.append(eta)
            tau_history.append(tau)
            volume_history.append(volume)
            volume_ratio_history.append(volume_ratio)
            radius_history.append(max_radius)
            energy_history.append(e_now)

            print(
                f"step={step:7d}  t={t:10.4e} s  "
                f"dt={dt:9.2e} s  eta={eta:9.3g}  "
                f"tau={tau:9.3g}  V/V_ST={volume_ratio:7.3f}"
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

    print("\n" + "=" * 76)
    print("FINAL DIAGNOSTICS")
    print("=" * 76)
    print(f"Final time                       : {t:.6e} s")
    print(f"Time steps                       : {step}")
    print(f"Free-air ST reference radius     : {R_ref_final:.6e} m")
    print(f"Scaled height of burst, eta      : {HOB / R_ref_final:.6g}")
    if t_interaction > 0.0:
        print(f"Scaled time, tau                 : {t / t_interaction:.6g}")
    else:
        print("Scaled time, tau                 : undefined (HOB = 0)")
    print(f"Threshold-based shock volume     : {final_volume:.6e} m^3")
    print(f"Volume / free-air sphere volume  : {final_volume_ratio:.6g}")
    print(f"Excess gas energy above ambient  : {final_energy:.6e} J")
    print(f"Energy change from initial       : {(final_energy/initial_energy - 1.0)*100:.4f} %")
    print()
    print("NOTE: The volume is estimated from a density-threshold shock mask.")
    print("It is not an exact shock-discontinuity volume; check threshold and grid convergence.")

    if MAKE_PLOTS:
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

        # Derive ray-based shock speeds from the sequence of threshold-front
        # positions.  This is post-processing only; the Euler state is unchanged.
        shock_speed_history, shock_mach_history = calculate_shock_front_history(
            snapshot_states
        )
        print("\nRendering snapshots with individually autoscaled color limits...")
        for i, state in enumerate(snapshot_states):
            # Determine color limits from this snapshot alone.  This changes
            # only the visualization; the numerical solution is untouched.
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
        
        # Create an MP4 from the saved spatial snapshots.
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
        print(f"\nSaved {len(snapshot_states)} spatial snapshots and diagnostics to:")
        print(f"  {run_output_dir.resolve()}")


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
