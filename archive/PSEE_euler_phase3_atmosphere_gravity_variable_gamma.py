#!/usr/bin/env python3
"""
Axisymmetric 3-D blast-wave Euler solver with reflecting ground, a 1976 U.S.
Standard Atmosphere background, gravity, and a thermally-perfect frozen-air EOS.

PHYSICS
  - Axisymmetric inviscid Euler equations in cylindrical (r,z) coordinates.
  - Reflecting, impermeable ground at z=0.
  - Hydrostatic 1976 U.S. Standard Atmosphere (or optional uniform test atmosphere).
  - Altitude-dependent gravitational acceleration.
  - Thermally-perfect, frozen-composition air: ideal-gas pressure law with
    temperature-dependent cv(T), cp(T), and gamma(T)=cp/cv. N2/O2 vibrational
    excitation is represented with harmonic-oscillator partition functions.
  - Finite-radius Gaussian thermal energy deposition, normalized to E in z>=0.

NUMERICS
  - Original first-order finite-volume architecture retained.
  - Local Rusanov flux; cylindrical cell volumes and face areas.
  - Explicit CFL time stepping.
  - Hydrostatic-perturbation reconstruction at vertical faces makes the
    prescribed standard atmosphere a discrete stationary solution (to roundoff),
    while retaining first-order reconstruction of blast perturbations.
  - Gravity acts in vertical momentum and gas-energy equations. The latter
    includes gravitational work, -rho*u_z*g.

THERMODYNAMIC MODEL LIMITATIONS
  - Frozen dry-air composition; no dissociation, ionization, radiation transport,
    chemical nonequilibrium, or plasma effects.
  - Molecular vibrational excitation makes gamma temperature-dependent. For air,
    dissociation becomes important at temperatures of order 1500 K; calculations
    above this are a mathematical frozen-composition continuation, not a validated
    high-temperature air/plasma EOS. The code warns when this range is exceeded.
  - The ideal-gas law is retained. This is not a real-gas or equilibrium-chemistry EOS.

ATMOSPHERE
  - 1976 U.S. Standard Atmosphere layer equations are evaluated using geopotential
    altitude; geometric altitude is used in the Euler grid and gravity law.
  - The standard atmosphere is used as the unperturbed hydrostatic background.
  - The Sedov-Taylor reference remains a constant-density, gamma~1.4 scaling
    diagnostic only; it is not an exact solution for a stratified, variable-gamma
    atmosphere with gravity.

Requires: numpy, matplotlib. MP4 output additionally requires imageio and
imageio-ffmpeg.
"""

from datetime import datetime
from pathlib import Path
import csv
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, SymLogNorm


# =============================================================================
# USER INPUTS
# =============================================================================

# Explosion input -- energy is entered in kilotons and converted to Joules.
E_kt = 20.0
HOB = 30.0                         # height above ground [m]

# Atmosphere and gravity
ATMOSPHERE_MODEL = "ussa1976"       # "ussa1976" or "uniform" (diagnostic only)
GROUND_ALTITUDE_M = 0.0             # ground elevation above mean sea level [m]
GRAVITY_ENABLED = True

# Used only when ATMOSPHERE_MODEL = "uniform". The uniform atmosphere with
# gravity enabled is intentionally NOT hydrostatic; it is a comparison mode.
RHO0_UNIFORM = 1.15                 # [kg/m^3]
P0_UNIFORM = 1.01325e5              # [Pa]

# Gas model
VARIABLE_GAMMA = True               # temperature-dependent frozen-air cp/cv
GAMMA_CONSTANT = 1.4                # used only if VARIABLE_GAMMA=False
THERMO_WARN_TEMPERATURE_K = 1500.0  # dissociation/chemistry warning threshold
EOS_LOOKUP_T_MIN_K = 1.0
EOS_LOOKUP_T_MAX_K = 1.0e11         # numerical inversion range, NOT physical validity
EOS_LOOKUP_POINTS = 100000

# Time interval
T_END = 0.3                         # [s]
CFL = 0.40                         # Courant number
MAX_STEPS = 500000

# Computational domain / resolution
NR = 180
NZ = 360
DOMAIN_FACTOR = 1.30
DEPOSITION_CELLS = 2.5              # Gaussian width in grid-cell widths

# Shock-front diagnostic
SHOCK_DENSITY_EXCESS = 0.02
N_ANGLE_BINS = 240

# Output
PRINT_EVERY = 100
MAKE_PLOTS = True
SNAPSHOT_TIMES = (
    1.0e-5, 3.0e-5, 5.0e-5,
    1.0e-4, 3.0e-4, 5.0e-4,
    1.0e-3, 3.0e-3, 5.0e-3,
    1.0e-2, 3.0e-2, 5.0e-2,
    1.0e-1, 3.0e-1, 5.0e-1,
    1.0, 3.0, 5.0,
)
SNAPSHOT_OUTPUT_DIR = "blast_wave_snapshots_phase3"
SNAPSHOT_DPI = 250

MAKE_MP4 = True
MP4_FPS = 1
MP4_FILENAME = "blast_wave.mp4"


# =============================================================================
# PHYSICAL CONSTANTS AND THERMALLY-PERFECT FROZEN-AIR EOS
# =============================================================================

E = 4.184e12 * E_kt              # [J]; preserve kt input convention
G0 = 9.80665                     # standard gravity [m/s^2]
EARTH_RADIUS_M = 6.356766e6      # geopotential Earth radius used by USSA-1976 [m]
R_UNIVERSAL = 8314.462618        # [J/(kmol K)]

# Four-species dry-air mass fractions from the NASA Langley TPG default air:
# N2, O2, Ar, CO2. Characteristic vibrational temperatures are for N2 and O2.
AIR_MASS_FRACTIONS = np.array([0.75530, 0.23140, 0.01290, 0.00040])
AIR_MOLECULAR_WEIGHTS = np.array([28.0160, 32.0000, 39.9480, 44.0100])  # kg/kmol
R_SPECIES = R_UNIVERSAL / AIR_MOLECULAR_WEIGHTS
Y_N2, Y_O2, Y_AR, Y_CO2 = AIR_MASS_FRACTIONS
R_N2, R_O2, R_AR, R_CO2 = R_SPECIES
R_AIR = float(np.dot(AIR_MASS_FRACTIONS, R_SPECIES))
THETA_V_N2 = 3390.0              # [K]
THETA_V_O2 = 2270.0              # [K]

# Frozen mixture cv(T): translational + rotational modes for diatomic species,
# plus harmonic-oscillator vibrational modes. Ar is monatomic; trace CO2 is
# represented with translational + rotational modes (its vibrational modes are
# neglected because its mass fraction is only 0.04%).
def _vibrational_energy_factor(theta, temperature):
    T = np.maximum(np.asarray(temperature, dtype=float), 1.0e-12)
    x = theta / T
    out = np.zeros_like(T)
    mask = x < 350.0
    out[mask] = theta / np.expm1(x[mask])
    return out


def _vibrational_cv_factor(theta, temperature):
    T = np.maximum(np.asarray(temperature, dtype=float), 1.0e-12)
    x = theta / T
    out = np.zeros_like(T)
    mask = x < 350.0
    em = np.exp(-x[mask])
    denom = -np.expm1(-x[mask])
    out[mask] = x[mask]**2 * em / np.maximum(denom**2, 1.0e-300)
    return out


def air_internal_energy(temperature):
    """Specific internal energy [J/kg], measured from the frozen-gas T=0 limit."""
    T = np.maximum(np.asarray(temperature, dtype=float), 0.0)
    if not VARIABLE_GAMMA:
        return (R_AIR / (GAMMA_CONSTANT - 1.0)) * T
    e_n2 = R_N2 * (2.5 * T + _vibrational_energy_factor(THETA_V_N2, T))
    e_o2 = R_O2 * (2.5 * T + _vibrational_energy_factor(THETA_V_O2, T))
    e_ar = R_AR * (1.5 * T)
    e_co2 = R_CO2 * (2.5 * T)
    return Y_N2*e_n2 + Y_O2*e_o2 + Y_AR*e_ar + Y_CO2*e_co2


def air_cv(temperature):
    """Frozen-mixture cv(T) [J/(kg K)]."""
    T = np.maximum(np.asarray(temperature, dtype=float), 1.0e-12)
    if not VARIABLE_GAMMA:
        return np.full_like(T, R_AIR / (GAMMA_CONSTANT - 1.0))
    cv_n2 = R_N2 * (2.5 + _vibrational_cv_factor(THETA_V_N2, T))
    cv_o2 = R_O2 * (2.5 + _vibrational_cv_factor(THETA_V_O2, T))
    cv_ar = R_AR * 1.5
    cv_co2 = R_CO2 * 2.5
    return Y_N2*cv_n2 + Y_O2*cv_o2 + Y_AR*cv_ar + Y_CO2*cv_co2


def air_cp(temperature):
    if VARIABLE_GAMMA:
        return air_cv(temperature) + R_AIR
    return R_AIR * GAMMA_CONSTANT / (GAMMA_CONSTANT - 1.0)


def air_gamma(temperature):
    if not VARIABLE_GAMMA:
        return np.full_like(np.asarray(temperature, dtype=float), GAMMA_CONSTANT)
    cv = air_cv(temperature)
    return (cv + R_AIR) / cv


# Precomputed monotone e(T) table makes conservative-to-primitive recovery
# efficient and robust over the wide temperatures encountered in blast problems.
_EOS_T_GRID = np.geomspace(EOS_LOOKUP_T_MIN_K, EOS_LOOKUP_T_MAX_K, EOS_LOOKUP_POINTS)
if VARIABLE_GAMMA:
    _EOS_E_GRID = air_internal_energy(_EOS_T_GRID)
else:
    _EOS_E_GRID = (R_AIR / (GAMMA_CONSTANT - 1.0)) * _EOS_T_GRID
if np.any(np.diff(_EOS_E_GRID) <= 0.0):
    raise RuntimeError("The configured EOS internal-energy table is not monotonic.")
_EOS_E_MIN = float(_EOS_E_GRID[0])
_EOS_E_MAX = float(_EOS_E_GRID[-1])


def temperature_from_internal_energy(e_specific):
    """Invert e(T) by monotone table interpolation; returns T in kelvin."""
    e = np.asarray(e_specific, dtype=float)
    if np.any(~np.isfinite(e)):
        raise FloatingPointError("Non-finite specific internal energy encountered.")
    if np.any(e > _EOS_E_MAX):
        emax = float(np.max(e))
        raise ValueError(
            f"Specific internal energy {emax:.4e} J/kg exceeds EOS lookup range "
            f"({_EOS_E_MAX:.4e} J/kg; T_max={EOS_LOOKUP_T_MAX_K:.3e} K)."
        )
    return np.interp(np.maximum(e, _EOS_E_MIN), _EOS_E_GRID, _EOS_T_GRID)


def gamma_from_rho_pressure(rho, pressure):
    T = np.maximum(pressure / np.maximum(rho * R_AIR, 1.0e-300), EOS_LOOKUP_T_MIN_K)
    return air_gamma(T)


# =============================================================================
# 1976 U.S. STANDARD ATMOSPHERE AND GRAVITY
# =============================================================================

# Geopotential layer bases [m], base temperatures [K], and lapse rates [K/m].
_USSA_H = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0,
                    51000.0, 71000.0, 84852.0])
_USSA_TB = np.array([288.15, 216.65, 216.65, 228.65, 270.65,
                     270.65, 214.65, 186.946])
_USSA_L = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0,
                    -0.0028, -0.0020, 0.0])
_USSA_PB = np.empty_like(_USSA_H)
_USSA_PB[0] = 101325.0
for _k in range(len(_USSA_H)-1):
    _dh = _USSA_H[_k+1] - _USSA_H[_k]
    _L = _USSA_L[_k]
    if abs(_L) > 0.0:
        _t_next = _USSA_TB[_k] + _L*_dh
        _USSA_PB[_k+1] = _USSA_PB[_k] * (_t_next/_USSA_TB[_k])**(-G0/(R_AIR*_L))
    else:
        _USSA_PB[_k+1] = _USSA_PB[_k] * np.exp(-G0*_dh/(R_AIR*_USSA_TB[_k]))


def geometric_to_geopotential_altitude(z_geometric):
    z = np.asarray(z_geometric, dtype=float)
    denom = EARTH_RADIUS_M + z
    if np.any(denom <= 0.0):
        raise ValueError("Geometric altitude is outside the supported range.")
    return EARTH_RADIUS_M * z / denom


def standard_atmosphere(z_geometric):
    """Return (T, p, rho) for USSA-1976 at geometric altitude [m]."""
    z = np.asarray(z_geometric, dtype=float)
    H = geometric_to_geopotential_altitude(z)
    if np.any(H > _USSA_H[-1] + 1.0e-8):
        raise ValueError(
            "USSA-1976 implementation is supported only through 84.852 km "
            "geopotential altitude (approximately 86 km geometric). Reduce "
            "the domain or select an appropriate upper-atmosphere model."
        )
    # Tropospheric layer is extrapolated downward to -5 km geopotential altitude.
    idx = np.searchsorted(_USSA_H, H, side="right") - 1
    idx = np.clip(idx, 0, len(_USSA_H)-1)
    L = _USSA_L[idx]
    Tb = _USSA_TB[idx]
    Pb = _USSA_PB[idx]
    Hb = _USSA_H[idx]
    dH = H - Hb
    T = Tb + L*dH
    p = np.empty_like(T, dtype=float)
    lapse = np.abs(L) > 0.0
    p[lapse] = Pb[lapse] * (T[lapse]/Tb[lapse])**(-G0/(R_AIR*L[lapse]))
    p[~lapse] = Pb[~lapse] * np.exp(-G0*dH[~lapse]/(R_AIR*Tb[~lapse]))
    rho = p/(R_AIR*T)
    if np.ndim(z_geometric) == 0:
        return float(T), float(p), float(rho)
    return T, p, rho


def atmosphere_state(z_geometric):
    """Configured ambient model. z is geometric altitude above mean sea level."""
    if ATMOSPHERE_MODEL.lower() == "ussa1976":
        return standard_atmosphere(z_geometric)
    if ATMOSPHERE_MODEL.lower() == "uniform":
        rho = np.zeros_like(np.asarray(z_geometric, dtype=float)) + RHO0_UNIFORM
        p = np.zeros_like(np.asarray(z_geometric, dtype=float)) + P0_UNIFORM
        T = p/(rho*R_AIR)
        if np.ndim(z_geometric) == 0:
            return float(T), float(p), float(rho)
        return T, p, rho
    raise ValueError("ATMOSPHERE_MODEL must be 'ussa1976' or 'uniform'.")


def gravity_acceleration(z_geometric):
    """Downward gravitational acceleration magnitude at geometric altitude [m/s^2]."""
    z = np.asarray(z_geometric, dtype=float)
    return G0 * (EARTH_RADIUS_M/(EARTH_RADIUS_M + z))**2


def gravitational_potential(z_geometric):
    """Specific gravitational potential relative to sea level [J/kg]."""
    z = np.asarray(z_geometric, dtype=float)
    return G0 * EARTH_RADIUS_M * z/(EARTH_RADIUS_M + z)


# Ground reference values used for reporting and the approximate Sedov scaling.
T_GROUND, p0, rho0 = atmosphere_state(GROUND_ALTITUDE_M)
T_BURST, p_burst, rho_burst = atmosphere_state(GROUND_ALTITUDE_M + HOB)


# =============================================================================
# SEDOV-TAYLOR REFERENCE (APPROXIMATE; NOT THE SOLVER)
# =============================================================================

XI0 = 1.033                         # spherical gamma=1.4 strong-shock scaling
alpha = 2.0/5.0
beta = 1.0/5.0
RHO_SEDOV_REFERENCE = rho_burst


def sedov_radius(t, energy=E):
    """Constant-density Sedov reference radius using burst-altitude ambient rho."""
    t = np.asarray(t, dtype=float)
    return XI0 * (energy/RHO_SEDOV_REFERENCE)**beta * t**alpha


def interaction_time(height):
    if height <= 0.0:
        return 0.0
    return (height/(XI0*(E/RHO_SEDOV_REFERENCE)**beta))**(1.0/alpha)


# =============================================================================
# AXISYMMETRIC FINITE-VOLUME EULER SOLVER
# =============================================================================

RHO_FLOOR = 1.0e-12
TEMPERATURE_FLOOR_K = 1.0


def primitive_from_conserved(U):
    """Return rho, ur, uz, pressure for the thermally-perfect ideal-gas EOS."""
    rho = np.maximum(U[0], RHO_FLOOR)
    ur = U[1]/rho
    uz = U[2]/rho
    kinetic = 0.5*rho*(ur**2 + uz**2)
    internal_density = np.maximum(U[3]-kinetic, rho*float(air_internal_energy(TEMPERATURE_FLOOR_K)))
    e_specific = internal_density/rho
    T = temperature_from_internal_energy(e_specific)
    pressure = rho*R_AIR*T
    return rho, ur, uz, pressure


def primitive_thermodynamics(U):
    """Return primitive state plus temperature and local gamma."""
    rho, ur, uz, pressure = primitive_from_conserved(U)
    T = pressure/(rho*R_AIR)
    g_local = air_gamma(T)
    return rho, ur, uz, pressure, T, g_local


def conserved_from_primitive(rho, ur, uz, pressure):
    """Convert rho,u,p to conserved state using e(T), not p/(gamma-1)."""
    rho = np.maximum(np.asarray(rho, dtype=float), RHO_FLOOR)
    T = np.maximum(np.asarray(pressure, dtype=float)/(rho*R_AIR), TEMPERATURE_FLOOR_K)
    internal = rho*air_internal_energy(T)
    total_energy = internal + 0.5*rho*(np.asarray(ur)**2 + np.asarray(uz)**2)
    return np.array([rho, rho*np.asarray(ur), rho*np.asarray(uz), total_energy])


def physical_flux_r(U):
    rho, ur, uz, pressure = primitive_from_conserved(U)
    return np.array([rho*ur, rho*ur**2+pressure, rho*ur*uz, (U[3]+pressure)*ur])


def physical_flux_z(U):
    rho, ur, uz, pressure = primitive_from_conserved(U)
    return np.array([rho*uz, rho*ur*uz, rho*uz**2+pressure, (U[3]+pressure)*uz])


def sound_speed(rho, pressure):
    """Frozen-composition ideal-gas sound speed, c^2=gamma(T)*p/rho."""
    T = np.maximum(np.asarray(pressure)/(np.maximum(np.asarray(rho), RHO_FLOOR)*R_AIR),
                   TEMPERATURE_FLOOR_K)
    g_local = air_gamma(T)
    return np.sqrt(np.maximum(g_local*np.asarray(pressure)/np.maximum(rho, RHO_FLOOR), 1.0e-20))


def rusanov_flux(UL, UR, direction):
    if direction == "r":
        FL, FR = physical_flux_r(UL), physical_flux_r(UR)
        rhoL, urL, uzL, pL = primitive_from_conserved(UL)
        rhoR, urR, uzR, pR = primitive_from_conserved(UR)
        amax = np.maximum(np.abs(urL)+sound_speed(rhoL,pL),
                          np.abs(urR)+sound_speed(rhoR,pR))
    else:
        FL, FR = physical_flux_z(UL), physical_flux_z(UR)
        rhoL, urL, uzL, pL = primitive_from_conserved(UL)
        rhoR, urR, uzR, pR = primitive_from_conserved(UR)
        amax = np.maximum(np.abs(uzL)+sound_speed(rhoL,pL),
                          np.abs(uzR)+sound_speed(rhoR,pR))
    return 0.5*(FL+FR)-0.5*amax[None,...]*(UR-UL)


def make_background_profiles(z_centers, z_faces, dz):
    """Create hydrostatic ambient profiles and a discretely balanced gravity source."""
    zc_abs = GROUND_ALTITUDE_M + z_centers
    zf_abs = GROUND_ALTITUDE_M + z_faces
    T_cell, p_cell, rho_cell = atmosphere_state(zc_abs)
    T_face, p_face, rho_face = atmosphere_state(zf_abs)
    g_point = gravity_acceleration(zc_abs)
    if ATMOSPHERE_MODEL.lower() == "ussa1976" and GRAVITY_ENABLED:
        # This cell-average g is the hydrostatic pressure drop divided by
        # rho(center)*dz. It differs from point g only by a small O(dz^2) term
        # and makes the discrete reference atmosphere exactly well-balanced.
        g_cell = (p_face[:-1]-p_face[1:])/(rho_cell*dz)
    else:
        g_cell = g_point
    if not GRAVITY_ENABLED:
        g_cell = np.zeros_like(g_cell)
    return {
        "T_cell": T_cell, "p_cell": p_cell, "rho_cell": rho_cell,
        "T_face": T_face, "p_face": p_face, "rho_face": rho_face,
        "g_cell": g_cell, "g_point": g_point,
        "z_cell_abs": zc_abs, "z_face_abs": zf_abs,
    }


def _vertical_face_conserved(rho_face, ur_face, uz_face, p_face):
    return conserved_from_primitive(rho_face, ur_face, uz_face, p_face)


def apply_boundaries(U, background):
    """Face states; vertical faces use hydrostatic-background perturbations.

    For a stationary standard atmosphere, the two states at every vertical face
    are identical ambient face states. This suppresses Rusanov's otherwise
    nonphysical mass diffusion across the background density gradient.
    """
    nr, nz = U.shape[1:]
    UL_r = np.empty((4,nr+1,nz)); UR_r = np.empty_like(UL_r)
    UL_r[:,1:-1,:] = U[:,:-1,:]
    UR_r[:,1:-1,:] = U[:,1:,:]
    axis = U[:,0,:].copy(); axis[1] *= -1.0
    UL_r[:,0,:] = axis; UR_r[:,0,:] = U[:,0,:]
    UL_r[:,-1,:] = U[:,-1,:]; UR_r[:,-1,:] = U[:,-1,:]

    rho, ur, uz, pressure = primitive_from_conserved(U)
    rho_a = background["rho_cell"][None,:]
    p_a = background["p_cell"][None,:]
    rho_af = background["rho_face"]
    p_af = background["p_face"]
    UL_z = np.empty((4,nr,nz+1)); UR_z = np.empty_like(UL_z)

    # Ground face: hydrostatic perturbation from first cell, with reflected uz.
    rf = np.maximum(rho_af[0] + rho[:,0]-rho_a[:,0], RHO_FLOOR)
    pf = np.maximum(p_af[0] + pressure[:,0]-p_a[:,0], 1.0e-8)
    left = _vertical_face_conserved(rf, ur[:,0], -uz[:,0], pf)
    right = _vertical_face_conserved(rf, ur[:,0], uz[:,0], pf)
    UL_z[:,:,0] = left; UR_z[:,:,0] = right

    # Interior vertical faces: preserve the ambient hydrostatic profile exactly,
    # while each side carries its own first-order blast perturbation.
    for k in range(1,nz):
        rfL = np.maximum(rho_af[k] + rho[:,k-1]-rho_a[:,k-1], RHO_FLOOR)
        rfR = np.maximum(rho_af[k] + rho[:,k]-rho_a[:,k], RHO_FLOOR)
        pfL = np.maximum(p_af[k] + pressure[:,k-1]-p_a[:,k-1], 1.0e-8)
        pfR = np.maximum(p_af[k] + pressure[:,k]-p_a[:,k], 1.0e-8)
        UL_z[:,:,k] = _vertical_face_conserved(rfL, ur[:,k-1], uz[:,k-1], pfL)
        UR_z[:,:,k] = _vertical_face_conserved(rfR, ur[:,k], uz[:,k], pfR)

    # Top outflow: zero-gradient perturbation extrapolated to the top ambient face.
    rf = np.maximum(rho_af[-1] + rho[:,-1]-rho_a[:,-1], RHO_FLOOR)
    pf = np.maximum(p_af[-1] + pressure[:,-1]-p_a[:,-1], 1.0e-8)
    top = _vertical_face_conserved(rf, ur[:,-1], uz[:,-1], pf)
    UL_z[:,:,-1] = top; UR_z[:,:,-1] = top
    return UL_r, UR_r, UL_z, UR_z


def stable_timestep(U, dr, dz):
    rho, ur, uz, pressure = primitive_from_conserved(U)
    c = sound_speed(rho, pressure)
    max_rate = np.max((np.abs(ur)+c)/dr + (np.abs(uz)+c)/dz)
    return CFL/max(max_rate, 1.0e-30)


def advance_one_step(U, dt, r_faces, dz, cell_volumes, background):
    UL_r, UR_r, UL_z, UR_z = apply_boundaries(U, background)
    Fr = rusanov_flux(UL_r, UR_r, "r")
    Fz = rusanov_flux(UL_z, UR_z, "z")
    radial_areas = (2.0*np.pi*r_faces*dz)[None,:,None]
    axial_areas = np.pi*(r_faces[1:]**2-r_faces[:-1]**2)
    radial_net = (Fr[:,1:,:]*radial_areas[:,1:,:]-Fr[:,:-1,:]*radial_areas[:,:-1,:])
    vertical_net = (Fz[:,:,1:]*axial_areas[None,:,None]-Fz[:,:,:-1]*axial_areas[None,:,None])
    U_new = U-dt*(radial_net+vertical_net)/cell_volumes[None,:,:]

    # Cylindrical radial momentum source: volume-average(p/r).
    rho, ur, uz, pressure = primitive_from_conserved(U)
    r_mid_sum = r_faces[1:]+r_faces[:-1]
    U_new[1] += dt*2.0*pressure/np.maximum(r_mid_sum[:,None],1.0e-30)

    # Gravity source: vertical momentum and corresponding work on gas energy.
    if GRAVITY_ENABLED:
        g = background["g_cell"][None,:]
        U_new[2] -= dt*rho*g
        U_new[3] -= dt*rho*uz*g

    # Numerical positivity guard, expressed through the selected thermodynamic EOS.
    rho_n = np.maximum(U_new[0], RHO_FLOOR)
    kinetic_n = 0.5*(U_new[1]**2+U_new[2]**2)/rho_n
    e_floor = float(air_internal_energy(TEMPERATURE_FLOOR_K))
    U_new[3] = np.maximum(U_new[3], kinetic_n+rho_n*e_floor)
    U_new[0] = rho_n
    return U_new


def gravity_work_rate(U, cell_volumes, background):
    if not GRAVITY_ENABLED:
        return 0.0
    rho = np.maximum(U[0], RHO_FLOOR)
    uz = U[2]/rho
    work_density = -rho*uz*background["g_cell"][None,:]
    return float(np.sum(work_density*cell_volumes))


# =============================================================================
# SHOCK AND VOLUME DIAGNOSTICS
# =============================================================================

def shock_geometry(r_centers, z_centers, rho, ambient_rho):
    rr, zz = np.meshgrid(r_centers, z_centers, indexing="ij")
    distance = np.sqrt(rr**2+(zz-HOB)**2)
    theta = np.arctan2(rr, zz-HOB)
    mask = rho > ambient_rho*(1.0+SHOCK_DENSITY_EXCESS)
    edges = np.linspace(0.0,np.pi,N_ANGLE_BINS+1)
    theta_centers = 0.5*(edges[:-1]+edges[1:])
    r_shock = np.zeros(N_ANGLE_BINS)
    bin_index = np.digitize(theta[mask],edges)-1
    dist_values = distance[mask]
    valid = (bin_index>=0)&(bin_index<N_ANGLE_BINS)
    np.maximum.at(r_shock,bin_index[valid],dist_values[valid])
    downward = np.cos(theta_centers)<0.0
    ground_distance = np.full_like(theta_centers,np.inf)
    ground_distance[downward] = HOB/np.maximum(-np.cos(theta_centers[downward]),1.0e-14)
    r_limited = np.minimum(r_shock,ground_distance)
    dtheta = edges[1]-edges[0]
    volume = (2.0*np.pi/3.0)*np.sum(r_limited**3*np.sin(theta_centers)*dtheta)
    return theta_centers,r_shock,volume


def total_energy(U, cell_volumes, background):
    """Excess gas energy above the altitude-dependent hydrostatic atmosphere."""
    ambient_e_density = background["rho_cell"]*air_internal_energy(background["T_cell"])
    return float(np.sum((U[3]-ambient_e_density[None,:])*cell_volumes))


def total_mechanical_excess(U, cell_volumes, background):
    """Gas excess plus excess gravitational potential energy vs ambient profile."""
    gas = total_energy(U,cell_volumes,background)
    phi = gravitational_potential(background["z_cell_abs"])
    rho_amb = background["rho_cell"][None,:]
    potential_excess = np.sum((U[0]-rho_amb)*phi[None,:]*cell_volumes)
    return float(gas+potential_excess)


def write_atmosphere_diagnostics(z_centers, z_faces, background, output_dir):
    csv_path = output_dir/"ambient_atmosphere_profile.csv"
    with csv_path.open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["z_above_ground_m","geometric_altitude_m","geopotential_altitude_m",
                         "T_ambient_K","p_ambient_Pa","rho_ambient_kg_m3",
                         "g_point_m_s2","g_hydrostatic_cell_m_s2"])
        for i,z in enumerate(z_centers):
            zabs=background["z_cell_abs"][i]
            H=float(geometric_to_geopotential_altitude(zabs))
            writer.writerow([z,zabs,H,background["T_cell"][i],background["p_cell"][i],
                             background["rho_cell"][i],background["g_point"][i],
                             background["g_cell"][i]])
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    axes=axes.ravel()
    axes[0].plot(background["T_cell"],z_centers/1000.0)
    axes[0].set_xlabel("Ambient temperature [K]"); axes[0].set_ylabel("Height above ground [km]")
    axes[0].set_title("Standard-atmosphere temperature")
    axes[1].semilogx(background["p_cell"],z_centers/1000.0)
    axes[1].set_xlabel("Ambient pressure [Pa]"); axes[1].set_ylabel("Height above ground [km]")
    axes[1].set_title("Standard-atmosphere pressure")
    axes[2].semilogx(background["rho_cell"],z_centers/1000.0)
    axes[2].set_xlabel(r"Ambient density [kg/m$^3$]"); axes[2].set_ylabel("Height above ground [km]")
    axes[2].set_title("Standard-atmosphere density")
    axes[3].plot(background["g_point"],z_centers/1000.0,label="Point gravity")
    axes[3].plot(background["g_cell"],z_centers/1000.0,"--",label="Hydrostatic cell-average")
    axes[3].set_xlabel(r"Gravity magnitude [m/s$^2$]"); axes[3].set_ylabel("Height above ground [km]")
    axes[3].set_title("Gravity profile"); axes[3].legend()
    fig.suptitle(f"{ATMOSPHERE_MODEL.upper()} ambient profile")
    fig.savefig(output_dir/"ambient_atmosphere_profile.png",dpi=SNAPSHOT_DPI)
    plt.close(fig)
    return csv_path


def main():
    if E_kt <= 0.0 or E <= 0.0:
        raise ValueError("Explosion energy E_kt must be positive.")
    if HOB < 0.0 or GROUND_ALTITUDE_M < 0.0:
        raise ValueError("HOB and GROUND_ALTITUDE_M must be nonnegative.")
    if ATMOSPHERE_MODEL.lower()=="ussa1976" and not GRAVITY_ENABLED:
        warnings.warn("USSA atmosphere selected with gravity disabled; the background is not dynamically hydrostatic.")
    if NR<20 or NZ<20:
        raise ValueError("Use at least 20 cells in each direction.")
    if AIR_MASS_FRACTIONS.size!=4 or not np.isclose(np.sum(AIR_MASS_FRACTIONS),1.0):
        raise RuntimeError("Air mixture mass fractions must sum to one.")

    t_interaction=interaction_time(HOB)
    R_end_ref=float(sedov_radius(T_END))
    r_max=DOMAIN_FACTOR*R_end_ref
    z_max=HOB+DOMAIN_FACTOR*R_end_ref
    if ATMOSPHERE_MODEL.lower()=="ussa1976":
        zmax_abs=GROUND_ALTITUDE_M+z_max
        Hmax=float(geometric_to_geopotential_altitude(zmax_abs))
        if Hmax>_USSA_H[-1]:
            raise ValueError(
                f"Top of domain maps to {Hmax/1000:.2f} km geopotential altitude, "
                "above the 84.852 km USSA-1976 limit. Reduce T_END/DOMAIN_FACTOR "
                "or use a suitable upper-atmosphere model."
            )

    r_faces=np.linspace(0.0,r_max,NR+1)
    z_faces=np.linspace(0.0,z_max,NZ+1)
    r_centers=0.5*(r_faces[:-1]+r_faces[1:])
    z_centers=0.5*(z_faces[:-1]+z_faces[1:])
    dr=r_faces[1]-r_faces[0]; dz=z_faces[1]-z_faces[0]
    cell_volumes=(np.pi*(r_faces[1:]**2-r_faces[:-1]**2))[:,None]*dz
    cell_volumes=np.broadcast_to(cell_volumes,(NR,NZ)).copy()
    background=make_background_profiles(z_centers,z_faces,dz)

    # Initial ambient state plus exactly normalized excess internal energy.
    rr,zz=np.meshgrid(r_centers,z_centers,indexing="ij")
    deposition_width=DEPOSITION_CELLS*max(dr,dz)
    qshape=np.exp(-(rr**2+(zz-HOB)**2)/(2.0*deposition_width**2))
    qmirror=np.exp(-(rr**2+(zz+HOB)**2)/(2.0*deposition_width**2))
    shape=qshape+qmirror
    normalization=np.sum(shape*cell_volumes)
    energy_density=E*shape/normalization
    rho=np.broadcast_to(background["rho_cell"][None,:],(NR,NZ)).copy()
    T_ambient=np.broadcast_to(background["T_cell"][None,:],(NR,NZ))
    e_ambient=air_internal_energy(T_ambient)
    e_initial=e_ambient+energy_density/rho
    T_initial=temperature_from_internal_energy(e_initial)
    pressure=rho*R_AIR*T_initial
    zeros=np.zeros_like(rho)
    U=conserved_from_primitive(rho,zeros,zeros,pressure)
    initial_energy=total_energy(U,cell_volumes,background)

    run_output_dir=None
    if MAKE_PLOTS:
        if any(x<=0 for x in SNAPSHOT_TIMES):
            raise ValueError("All SNAPSHOT_TIMES must be positive.")
        run_stamp=datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_output_dir=Path(SNAPSHOT_OUTPUT_DIR)/run_stamp
        run_output_dir.mkdir(parents=True,exist_ok=True)
        write_atmosphere_diagnostics(z_centers,z_faces,background,run_output_dir)

    print("="*78)
    print("AXISYMMETRIC BLAST-WAVE EULER SOLVER — PHASE 3")
    print("="*78)
    print(f"Explosion energy                 : {E_kt:.6g} kt = {E:.6e} J")
    print(f"Height of burst                  : {HOB:.6g} m above ground")
    print(f"Atmosphere model                 : {ATMOSPHERE_MODEL}")
    print(f"Ground altitude                  : {GROUND_ALTITUDE_M:.6g} m MSL")
    print(f"Ground ambient T/p/rho           : {T_GROUND:.4f} K / {p0:.6e} Pa / {rho0:.6e} kg/m^3")
    print(f"Burst ambient T/p/rho            : {T_BURST:.4f} K / {p_burst:.6e} Pa / {rho_burst:.6e} kg/m^3")
    print(f"Variable gamma                   : {VARIABLE_GAMMA}")
    print(f"Ambient gamma at burst           : {float(air_gamma(T_BURST)):.6f}")
    print(f"Gravity enabled                  : {GRAVITY_ENABLED}")
    print(f"Gravity at ground/burst          : {float(gravity_acceleration(GROUND_ALTITUDE_M)):.6f} / {float(gravity_acceleration(GROUND_ALTITUDE_M+HOB)):.6f} m/s^2")
    print(f"Grid                             : {NR} x {NZ}")
    print(f"Domain                           : r=[0,{r_max:.5g}] m, z=[0,{z_max:.5g}] m")
    print(f"Cell dimensions                  : dr={dr:.5g} m, dz={dz:.5g} m")
    print(f"Gaussian deposition width        : {deposition_width:.5g} m")
    print(f"Initial deposited energy         : {initial_energy:.6e} J")
    print(f"Initial peak gas temperature     : {float(np.max(T_initial)):.6e} K")
    if float(np.max(T_initial))>THERMO_WARN_TEMPERATURE_K:
        print("WARNING: Initial temperatures exceed the approximate frozen-air/thermally-perfect validity range.")
        print("         Dissociation, ionization, radiation, and nonequilibrium chemistry are not modeled.")
    print(f"Approx. Sedov radius at t_end    : {R_end_ref:.5g} m (scaling diagnostic only)")
    print(f"Characteristic interaction time : {t_interaction:.6e} s")
    print()

    # Ambient fields are carried with every output snapshot so plot ratios are
    # normalized to the local altitude-dependent state rather than sea level.
    times=[]; eta_history=[]; tau_history=[]; volume_ratio_history=[]
    energy_history=[]; mechanical_energy_history=[]; gravity_work_history=[]
    snapshot_states=[]
    target_times=sorted(set(float(x) for x in SNAPSHOT_TIMES if 0.0<x<=T_END))
    next_target_index=0

    def append_snapshot(time_value, state_U):
        rrho,u_r,u_z,p=primitive_from_conserved(state_U)
        T=p/(rrho*R_AIR)
        gam=air_gamma(T)
        theta,rs,_=shock_geometry(r_centers,z_centers,rrho,
                                  np.broadcast_to(background["rho_cell"][None,:],rrho.shape))
        snapshot_states.append({
            "time":float(time_value),"rho":rrho.copy(),"ur":u_r.copy(),"uz":u_z.copy(),
            "pressure":p.copy(),"temperature":T.copy(),"gamma":np.asarray(gam).copy(),
            "rho_ambient":np.broadcast_to(background["rho_cell"][None,:],rrho.shape).copy(),
            "p_ambient":np.broadcast_to(background["p_cell"][None,:],rrho.shape).copy(),
            "theta":theta.copy(),"r_shock":rs.copy(),
        })

    if MAKE_PLOTS:
        append_snapshot(0.0,U)

    t=0.0; step=0; cumulative_gravity_work=0.0
    while t<T_END and step<MAX_STEPS:
        dt=min(stable_timestep(U,dr,dz),T_END-t)
        work_rate=gravity_work_rate(U,cell_volumes,background)
        U=advance_one_step(U,dt,r_faces,dz,cell_volumes,background)
        cumulative_gravity_work += dt*work_rate
        t += dt; step += 1

        if MAKE_PLOTS and next_target_index<len(target_times) and t>=target_times[next_target_index]:
            append_snapshot(t,U)
            print(f"  captured spatial state at t={t:.5e} s (target {target_times[next_target_index]:.5e} s)")
            next_target_index+=1
            while next_target_index<len(target_times) and t>=target_times[next_target_index]:
                print(f"  NOTE: target {target_times[next_target_index]:.5e} s crossed in same step; no separate state.")
                next_target_index+=1

        if step==1 or step%PRINT_EVERY==0 or t>=T_END:
            rrho,u_r,u_z,p=primitive_from_conserved(U)
            T=p/(rrho*R_AIR); gam=air_gamma(T)
            ambient_rho_grid=np.broadcast_to(background["rho_cell"][None,:],rrho.shape)
            theta,rs,vol=shock_geometry(r_centers,z_centers,rrho,ambient_rho_grid)
            Rref=float(sedov_radius(max(t,1.0e-30)))
            eta=HOB/Rref
            tau=t/t_interaction if t_interaction>0 else np.inf
            volume_ref=(4.0/3.0)*np.pi*Rref**3
            vol_ratio=vol/volume_ref
            e_gas=total_energy(U,cell_volumes,background)
            e_mech=total_mechanical_excess(U,cell_volumes,background)
            times.append(t); eta_history.append(eta); tau_history.append(tau)
            volume_ratio_history.append(vol_ratio); energy_history.append(e_gas)
            mechanical_energy_history.append(e_mech); gravity_work_history.append(cumulative_gravity_work)
            print(f"step={step:7d}  t={t:10.4e} s  dt={dt:9.2e} s  "
                  f"eta={eta:8.3g}  tau={tau:8.3g}  V/V_ST={vol_ratio:7.3f}  "
                  f"Egas/E0={e_gas/initial_energy:8.5f}  "
                  f"Tmax={np.max(T):.3e} K  gamma_min={np.min(gam):.4f}  "
                  f"Wg={cumulative_gravity_work:.3e} J")
            if np.max(T)>THERMO_WARN_TEMPERATURE_K and step==1:
                print("WARNING: Temperatures exceed ~1500 K; fixed-composition air omits dissociation and ionization.")

    if step>=MAX_STEPS and t<T_END:
        print("WARNING: MAX_STEPS reached before T_END.")

    rho,ur,uz,pressure=primitive_from_conserved(U)
    temperature=pressure/(rho*R_AIR); gamma_field=air_gamma(temperature)
    ambient_rho_grid=np.broadcast_to(background["rho_cell"][None,:],rho.shape)
    theta,r_shock,final_volume=shock_geometry(r_centers,z_centers,rho,ambient_rho_grid)
    R_ref_final=float(sedov_radius(max(t,1.0e-30)))
    V_ref_final=(4.0/3.0)*np.pi*R_ref_final**3
    final_volume_ratio=final_volume/V_ref_final
    final_energy=total_energy(U,cell_volumes,background)
    final_mechanical=total_mechanical_excess(U,cell_volumes,background)

    print("\n"+"="*78)
    print("FINAL PHASE-3 DIAGNOSTICS")
    print("="*78)
    print(f"Final time                       : {t:.6e} s")
    print(f"Time steps                       : {step}")
    print(f"Approx. Sedov reference radius   : {R_ref_final:.6e} m")
    print(f"Scaled height of burst, eta      : {HOB/R_ref_final:.6g}")
    if t_interaction>0:
        print(f"Scaled time, tau                 : {t/t_interaction:.6g}")
    print(f"Threshold-based shock volume     : {final_volume:.6e} m^3")
    print(f"Volume / free-air sphere volume  : {final_volume_ratio:.6g}")
    print(f"Excess gas energy                : {final_energy:.6e} J")
    print(f"Excess mechanical energy         : {final_mechanical:.6e} J")
    print(f"Cumulative gravitational work    : {cumulative_gravity_work:.6e} J")
    print(f"Maximum gas temperature         : {np.max(temperature):.6e} K")
    print(f"Minimum/maximum gamma            : {np.min(gamma_field):.6f} / {np.max(gamma_field):.6f}")
    if np.max(temperature)>THERMO_WARN_TEMPERATURE_K:
        print("WARNING: High-temperature frozen-air results are not chemically equilibrated or plasma-valid.")

    if MAKE_PLOTS:
        if not snapshot_states or not np.isclose(snapshot_states[-1]["time"],t,rtol=1e-10,atol=1e-14):
            append_snapshot(t,U)
        shock_speed_history,shock_mach_history=calculate_shock_front_history(snapshot_states)
        print("\nRendering spatial snapshots...")
        for i,state in enumerate(snapshot_states):
            scales=determine_snapshot_plot_scales(state,shock_mach_history[i])
            save_spatial_snapshot(
                r_faces,z_faces,r_centers,z_centers,
                state["rho"],state["ur"],state["uz"],state["pressure"],
                state["temperature"],state["gamma"],state["rho_ambient"],state["p_ambient"],
                state["time"],run_output_dir/f"snapshot_{i:04d}_t_{state['time']:.6e}s.png",
                scales,state["theta"],state["r_shock"],shock_mach_history[i]
            )
        if MAKE_MP4:
            create_mp4_from_snapshots(run_output_dir,fps=MP4_FPS,output_name=MP4_FILENAME)
        make_diagnostic_plots(
            theta,r_shock,np.asarray(times),np.asarray(eta_history),np.asarray(tau_history),
            np.asarray(volume_ratio_history),np.asarray(energy_history),t_interaction,
            initial_energy,snapshot_states,shock_speed_history,shock_mach_history,
            run_output_dir/"diagnostics.png"
        )
        # Gravity/energy time history as CSV for transparent accounting.
        with (run_output_dir/"gravity_energy_history.csv").open("w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["time_s","excess_gas_energy_J","excess_mechanical_energy_J","cumulative_gravity_work_J"])
            for row in zip(times,energy_history,mechanical_energy_history,gravity_work_history):
                writer.writerow(row)
        print(f"\nSaved {len(snapshot_states)} spatial snapshots and diagnostics to:")
        print(f"  {run_output_dir.resolve()}")


# =============================================================================
# PLOTTING AND POST-PROCESSING DIAGNOSTICS
# =============================================================================

def calculate_shock_front_history(snapshot_states):
    """Threshold-ray shock speed and Mach; Mach uses local ambient sound speed at HOB."""
    _,p_burst_local,rho_burst_local=atmosphere_state(GROUND_ALTITUDE_M+HOB)
    c_ambient=float(sound_speed(rho_burst_local,p_burst_local))
    speed_history=[]; mach_history=[]
    for i,state in enumerate(snapshot_states):
        speed=np.full_like(state["r_shock"],np.nan,dtype=float)
        mach=np.full_like(state["r_shock"],np.nan,dtype=float)
        if i>0:
            prev=snapshot_states[i-1]
            dt=state["time"]-prev["time"]
            if dt>0:
                rnow=state["r_shock"]; rprev=prev["r_shock"]
                valid=(rnow>0)&(rprev>0)
                estimate=(rnow-rprev)/dt
                valid &= estimate>0
                speed[valid]=estimate[valid]
                mach[valid]=speed[valid]/c_ambient
        speed_history.append(speed); mach_history.append(mach)
    return speed_history,mach_history


def determine_snapshot_plot_scales(state,shock_mach):
    rho_ratio=np.maximum(state["rho"]/state["rho_ambient"],1.0e-12)
    p_ratio=np.maximum(state["pressure"]/state["p_ambient"],1.0e-12)
    over=(state["pressure"]-state["p_ambient"])/state["p_ambient"]
    speed=np.sqrt(state["ur"]**2+state["uz"]**2)
    flow_mach=speed/np.maximum(sound_speed(state["rho"],state["pressure"]),1e-20)
    dmin=max(float(np.nanmin(rho_ratio)),1e-8); dmax=max(float(np.nanmax(rho_ratio)),dmin*1.001)
    pmin=max(float(np.nanmin(p_ratio)),1e-8); pmax=max(float(np.nanmax(p_ratio)),pmin*1.001)
    olim=max(float(np.nanmax(np.abs(over))),1e-3)
    vlim=max(float(np.nanmax(np.abs(state["ur"]))),float(np.nanmax(np.abs(state["uz"]))),1e-8)
    smax=max(float(np.nanmax(speed)),1e-12); smin=max(smax*1e-5,1e-12)
    mmax=max(float(np.nanmax(flow_mach)),1e-12); mmin=max(mmax*1e-5,1e-12)
    tmin=max(float(np.nanmin(state["temperature"])),1e-6); tmax=max(float(np.nanmax(state["temperature"])),tmin*1.001)
    gmin=float(np.nanmin(state["gamma"])); gmax=max(float(np.nanmax(state["gamma"])),gmin*1.001)
    sv=shock_mach[np.isfinite(shock_mach)&(shock_mach>0)]
    smn=max(float(np.min(sv)),1e-3) if sv.size else 1e-3
    smx=max(float(np.max(sv)),smn*1.001) if sv.size else 1.0
    return {"density_vmin":dmin,"density_vmax":dmax,"pressure_vmin":pmin,"pressure_vmax":pmax,
            "overpressure_limit":olim,"overpressure_linthresh":max(olim*1e-3,1e-12),
            "velocity_limit":vlim,"speed_vmin":smin,"speed_vmax":max(smax,smin*1.001),
            "flow_mach_vmin":mmin,"flow_mach_vmax":max(mmax,mmin*1.001),
            "temperature_vmin":tmin,"temperature_vmax":tmax,
            "gamma_vmin":gmin,"gamma_vmax":gmax,
            "shock_mach_vmin":smn,"shock_mach_vmax":smx}


def save_spatial_snapshot(r_faces,z_faces,r_centers,z_centers,rho,ur,uz,pressure,
                          temperature,gamma_field,rho_ambient,p_ambient,t,output_path,
                          scales,theta,r_shock,shock_mach):
    density_ratio=np.maximum(rho/rho_ambient,1e-12)
    pressure_ratio=np.maximum(pressure/p_ambient,1e-12)
    overpressure=(pressure-p_ambient)/p_ambient
    speed=np.sqrt(ur**2+uz**2)
    flow_mach=speed/np.maximum(sound_speed(rho,pressure),1e-20)
    fig,axes=plt.subplots(2,5,figsize=(25,10.5),constrained_layout=True)
    axes=axes.ravel()
    def decorate(ax,title):
        ax.set_title(title); ax.set_xlabel("Radial distance r [m]"); ax.set_ylabel("Height z [m]")
        ax.set_xlim(r_faces[0],r_faces[-1]); ax.set_ylim(z_faces[0],z_faces[-1])
        ax.set_aspect("equal",adjustable="box")
        ax.plot(0.0,HOB,marker="*",color="white",markeredgecolor="black",markersize=9,zorder=8)
    im=axes[0].pcolormesh(r_faces,z_faces,density_ratio.T,shading="flat",cmap="viridis",
       norm=LogNorm(vmin=scales["density_vmin"],vmax=scales["density_vmax"]))
    decorate(axes[0],r"Density ratio, $\rho/\rho_{amb}(z)$"); fig.colorbar(im,ax=axes[0])
    im=axes[1].pcolormesh(r_faces,z_faces,pressure_ratio.T,shading="flat",cmap="inferno",
       norm=LogNorm(vmin=scales["pressure_vmin"],vmax=scales["pressure_vmax"]))
    decorate(axes[1],r"Pressure ratio, $p/p_{amb}(z)$"); fig.colorbar(im,ax=axes[1])
    lim=scales["overpressure_limit"]
    im=axes[2].pcolormesh(r_faces,z_faces,overpressure.T,shading="flat",cmap="coolwarm",
       norm=SymLogNorm(linthresh=scales["overpressure_linthresh"],vmin=-lim,vmax=lim,base=10))
    decorate(axes[2],r"Overpressure ratio, $(p-p_{amb})/p_{amb}$"); fig.colorbar(im,ax=axes[2])
    vlim=scales["velocity_limit"]
    im=axes[3].pcolormesh(r_faces,z_faces,ur.T,shading="flat",cmap="coolwarm",
       norm=TwoSlopeNorm(vmin=-vlim,vcenter=0,vmax=vlim))
    decorate(axes[3],r"Radial velocity, $u_r$ [m/s]"); fig.colorbar(im,ax=axes[3])
    im=axes[4].pcolormesh(r_faces,z_faces,uz.T,shading="flat",cmap="coolwarm",
       norm=TwoSlopeNorm(vmin=-vlim,vcenter=0,vmax=vlim))
    decorate(axes[4],r"Vertical velocity, $u_z$ [m/s]"); fig.colorbar(im,ax=axes[4])
    im=axes[5].pcolormesh(r_faces,z_faces,np.maximum(speed,scales["speed_vmin"]).T,shading="flat",cmap="magma",
       norm=LogNorm(vmin=scales["speed_vmin"],vmax=scales["speed_vmax"]))
    decorate(axes[5],r"Velocity magnitude, $|u|$ [m/s]"); fig.colorbar(im,ax=axes[5])
    stride_r=max(1,len(r_centers)//22); stride_z=max(1,len(z_centers)//22)
    Rq,Zq=np.meshgrid(r_centers,z_centers,indexing="xy")
    if float(np.nanmax(speed)) > 0.0:
        axes[5].quiver(Rq[::stride_z,::stride_r],Zq[::stride_z,::stride_r],
                       ur.T[::stride_z,::stride_r],uz.T[::stride_z,::stride_r],color="white",alpha=.7)
    im=axes[6].pcolormesh(r_faces,z_faces,np.maximum(flow_mach,scales["flow_mach_vmin"]).T,shading="flat",cmap="cividis",
       norm=LogNorm(vmin=scales["flow_mach_vmin"],vmax=scales["flow_mach_vmax"]))
    decorate(axes[6],r"Fluid Mach number, $|u|/c$"); fig.colorbar(im,ax=axes[6])
    im=axes[7].pcolormesh(r_faces,z_faces,np.maximum(temperature,1e-9).T,shading="flat",cmap="plasma",
       norm=LogNorm(vmin=max(scales["temperature_vmin"],1e-6),vmax=max(scales["temperature_vmax"],scales["temperature_vmin"]*1.001)))
    decorate(axes[7],"Gas temperature [K]"); fig.colorbar(im,ax=axes[7])
    im=axes[8].pcolormesh(r_faces,z_faces,gamma_field.T,shading="flat",cmap="viridis",
       vmin=scales["gamma_vmin"],vmax=scales["gamma_vmax"])
    decorate(axes[8],r"Local $\gamma$ = $c_p/c_v$"); fig.colorbar(im,ax=axes[8])
    decorate(axes[9],"Detected shock front: radial-ray Mach")
    valid=np.isfinite(shock_mach)&(shock_mach>0)&(r_shock>0)
    if np.any(valid):
        rf=r_shock[valid]*np.sin(theta[valid]); zf=HOB+r_shock[valid]*np.cos(theta[valid])
        sc=axes[9].scatter(rf,zf,c=shock_mach[valid],s=14,cmap="plasma",
           norm=LogNorm(vmin=scales["shock_mach_vmin"],vmax=scales["shock_mach_vmax"]),edgecolors="none")
        fig.colorbar(sc,ax=axes[9],label="Radial-ray shock Mach")
    else:
        axes[9].text(.5,.5,"Not yet available",transform=axes[9].transAxes,ha="center")
    fig.suptitle(f"Axisymmetric blast field | t={t:.5e} s | HOB={HOB:g} m | {ATMOSPHERE_MODEL}")
    fig.savefig(output_path,dpi=SNAPSHOT_DPI)
    plt.close(fig)


def _finite_summary(values,percentile):
    finite=values[np.isfinite(values)&(values>0)]
    return float(np.percentile(finite,percentile)) if finite.size else np.nan


def make_diagnostic_plots(theta,r_shock,times,eta,tau,volume_ratio,energy_history,
                          t_interaction,initial_energy,snapshot_states,shock_speed_history,
                          shock_mach_history,output_path):
    fig,axes=plt.subplots(4,2,figsize=(15,17),constrained_layout=True); axes=axes.ravel()
    axes[0].plot(theta*180/np.pi,r_shock,"k-",lw=1.8)
    axes[0].set_xlabel("Polar angle from upward vertical [deg]")
    axes[0].set_ylabel("Estimated shock distance [m]"); axes[0].set_title("Final threshold shock radius"); axes[0].grid(True,ls=":")
    if len(times):
        axes[1].plot(times,volume_ratio,lw=2,label=r"$V/V_{ST}$")
        axes[1].axhline(1,color="k",ls="--",label="Free-air sphere reference")
        axes[1].axhline(.5,color="gray",ls=":",label="Hemisphere reference")
        axes[1].set_xscale("log"); axes[1].set_xlabel("Time [s]"); axes[1].set_ylabel("Volume ratio")
        axes[1].set_title("Shock volume (approximate Sedov scaling)"); axes[1].grid(True,which="both",ls=":"); axes[1].legend()
        axes[2].plot(times,energy_history/max(initial_energy,1e-30),lw=2)
        axes[2].axhline(1,color="k",ls="--",lw=1)
        axes[2].set_xscale("log"); axes[2].set_xlabel("Time [s]"); axes[2].set_ylabel("Excess gas energy / initial")
        axes[2].set_title("Excess gas energy (gravity work changes gas energy)"); axes[2].grid(True,which="both",ls=":")
        axes[3].plot(times,eta,label=r"$H/R_{ST}$"); axes[3].set_xscale("log"); axes[3].set_yscale("log")
        axes[3].set_xlabel("Time [s]"); axes[3].set_ylabel(r"$\eta$")
        if t_interaction>0:
            ax2=axes[3].twinx(); ax2.plot(times,tau,"C1--",label=r"$t/t_H$"); ax2.set_yscale("log"); ax2.set_ylabel(r"$t/t_H$")
        axes[3].set_title("Dimensionless blast evolution"); axes[3].grid(True,which="both",ls=":")
    final_speed=shock_speed_history[-1] if shock_speed_history else np.array([])
    final_mach=shock_mach_history[-1] if shock_mach_history else np.array([])
    final_theta=snapshot_states[-1]["theta"] if snapshot_states else np.array([])
    if final_speed.size and np.any(np.isfinite(final_speed)):
        axes[4].plot(final_theta*180/np.pi,final_speed,"C2-")
    else: axes[4].text(.5,.5,"No valid front-speed estimate",transform=axes[4].transAxes,ha="center")
    axes[4].set_xlabel("Polar angle [deg]"); axes[4].set_ylabel("Radial-ray shock speed [m/s]"); axes[4].set_title("Final shock speed"); axes[4].grid(True,ls=":")
    if final_mach.size and np.any(np.isfinite(final_mach)):
        axes[5].plot(final_theta*180/np.pi,final_mach,"C3-"); axes[5].set_yscale("log"); axes[5].axhline(1,color="k",ls="--")
    else: axes[5].text(.5,.5,"No valid front-Mach estimate",transform=axes[5].transAxes,ha="center")
    axes[5].set_xlabel("Polar angle [deg]"); axes[5].set_ylabel("Shock Mach"); axes[5].set_title("Final shock Mach (HOB ambient sound speed)"); axes[5].grid(True,which="both",ls=":")
    stimes=np.asarray([s["time"] for s in snapshot_states])
    smed=np.asarray([_finite_summary(x,50) for x in shock_speed_history]); sp90=np.asarray([_finite_summary(x,90) for x in shock_speed_history])
    mmed=np.asarray([_finite_summary(x,50) for x in shock_mach_history]); mp90=np.asarray([_finite_summary(x,90) for x in shock_mach_history])
    v=np.isfinite(smed)
    if np.any(v):
        axes[6].plot(stimes[v],smed[v],label="Median"); axes[6].plot(stimes[np.isfinite(sp90)],sp90[np.isfinite(sp90)],"--",label="90th percentile")
        axes[6].set_xscale("log"); axes[6].set_yscale("log"); axes[6].legend()
    else: axes[6].text(.5,.5,"No valid front-speed estimates",transform=axes[6].transAxes,ha="center")
    axes[6].set_xlabel("Time [s]"); axes[6].set_ylabel("Speed [m/s]"); axes[6].set_title("Shock speed history"); axes[6].grid(True,which="both",ls=":")
    v=np.isfinite(mmed)
    if np.any(v):
        axes[7].plot(stimes[v],mmed[v],label="Median"); axes[7].plot(stimes[np.isfinite(mp90)],mp90[np.isfinite(mp90)],"--",label="90th percentile")
        axes[7].axhline(1,color="k",ls=":",label="Mach 1"); axes[7].set_xscale("log"); axes[7].set_yscale("log"); axes[7].legend()
    else: axes[7].text(.5,.5,"No valid front-Mach estimates",transform=axes[7].transAxes,ha="center")
    axes[7].set_xlabel("Time [s]"); axes[7].set_ylabel("Shock Mach"); axes[7].set_title("Shock Mach history"); axes[7].grid(True,which="both",ls=":")
    fig.suptitle("Blast-wave diagnostics — stratified atmosphere, gravity, variable gamma")
    fig.savefig(output_path,dpi=SNAPSHOT_DPI,bbox_inches="tight"); plt.close(fig)


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
