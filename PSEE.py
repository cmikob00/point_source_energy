# Point-Source Energy Expansion Code
# C.J. Miko
# Unclassified

# Uses the von-Neumann derived equation for a point-source expansion with a given energy
# and a constant gamma-law fluid medium
# E = G_gam*rho*v**2*R**3

# General solving method
# step 0 - read in E, Alt, initial radius, timestep, number of timesteps, EOS
# step 1 - calculate initial velocity
# step 2 - calculate change in radius during timestep, assuming constand velocity over the timestep
# step 3 - calculate new radius after timestep
# step 4 - update timestep counter
# step 5 - calculate updated velocity
# step 6 - calculate change in radius
# step 7 - calculate updated radius
# step 8 - update timestep counter
# repeat steps 5 - 8 until simulation is complete

# importing math
import math


# declaring global constants
pi = math.pi

k_B = 1.380648e-23  # Boltzmann Constant in J/K

s = 5.670374419e-8  # Stefan-Boltzmann Constant in W*m**-2*K**-4

rho_0 = 0.001225  # g/cm**3

H = 8.  # scale height of atmosphere in km

ergkt = 2.39006e-20  # conversion from erg to kT


# defining functions

# function to convert inputs into string and write to ee output file
def initial_conds_write(E_kt, Alt, R_0, dt, tend, G_gam):

    E_kt_str = str(E_kt)
    Alt_str = str(Alt)
    R_0_str = str(R_0)
    dt_str = str(dt)
    tend_str = str(tend)
    G_gam_str = str(G_gam)

    ee_owt.write('Point-Source Energy Expansion Simulation for ' +
                 ' Energy (kT): ' + E_kt_str +
                 ' Altitude (km): ' + Alt_str +
                 ' Initial Radius (cm): ' + R_0_str +
                 ' Timestep size (sec): ' + dt_str +
                 ' Ending Time (sec): ' + tend_str +
                 ' Adiabatic Index (EOS): ' + G_gam_str + '\n' + '\n')
    
    eos_owt.write('Equation-of-State-related Outputs for ' +
                  ' Energy (kT): ' + E_kt_str +
                  ' Altitude (km): ' + Alt_str +
                  ' Initial Radius (cm): ' + R_0_str +
                  ' Timestep size (sec): ' + dt_str +
                  ' Ending Time (sec): ' + tend_str +
                  ' Adiabatic Index (EOS): ' + G_gam_str + '\n' + '\n')
    
    rtvar_owt.write('Radius vs Time Values for ' +
                    ' Energy (kT): ' + E_kt_str +
                    ' Altitude (km): ' + Alt_str +
                    ' Initial Radius (cm): ' + R_0_str +
                    ' Timestep size (sec): ' + dt_str +
                    ' Ending Time (sec): ' + tend_str +
                    ' Adiabatic Index (EOS): ' + G_gam_str + '\n' + '\n')
    
    rtvar_owt.write('Time (sec)    ' + 'Radius (cm)    ' + 'Velocity (cm/sec)' + '\n')

    return E_kt_str, Alt_str, R_0_str, dt_str, tend_str, G_gam_str


# defining function to find rho of medium
def find_rho(Alt):

    rho = rho_0*math.e**(-Alt/H)

    print('density of medium is: ', rho, 'g.cm**3')

    rho_str = str(rho)

    ee_owt.write('rho (g/cm**3)    ' + rho_str + '\n')

    eos_owt.write('rho (g/cm**3)    ' + rho_str + '\n')

    return rho, rho_str

def find_rhorat(rho, rho_0):

    rhorat = rho/rho_0

    print('atmospheric density ratio is: ', rhorat)

    rhorat_str = str(rhorat)

    ee_owt.write('atm density ratio' + rhorat_str + '\n')

    eos_owt.write('atm density ratio' + rhorat_str + '\n')

    return rhorat, rhorat_str

# defining function to convert energy in kT to energy in erg
def kt_to_erg(E):

    E = E_kt/ergkt

    print('energy in ergs is: ', E, 'erg')

    E_str = str(E)

    ee_owt.write('E (erg)          ' + E_str + '\n')

    eos_owt.write('E (erg)          ' + E_str + '\n')

    return E, E_str

# function to calculate number of timesteps
def calc_timesteps(dt, tend):

    ntps = (tend / dt) + 1

    ntps_int = int(ntps)

    print('number of timesteps: ', ntps)

    ntps_str = str(ntps)

    ee_owt.write('number of timesteps:  ' + ntps_str + '\n' + '\n')

    return ntps, ntps_int, ntps_str

# defining time initialization function
def time_init(tend):

    t = tend - (tend - 1.e-7)

    print('time = ', t, 'sec')
    t_str = str(t)
    ee_owt.write('time (sec)       ' + t_str + '\n')

    return t, t_str

# defining initial velocity calculation function
def v_init(R_0, t):

    # v_0 = (E/(G_gam*rho*R_0**3))**(1/2)
    v_0 = 0.4 * (R_0 / t)

    print('initial velocity is: ', v_0, 'cm/sec')

    v_0_str = str(v_0)

    ee_owt.write('v_0 (cm/sec)     ' + v_0_str + '\n')

    return v_0, v_0_str

# defining change in R due to velocity function
def drdt(v):

    dR = v*dt

    print('change in radius during timestep is: ', dR, 'cm')

    dR_str = str(dR)

    ee_owt.write('dR (cm)          ' + dR_str + '\n')

    return dR, dR_str

# defining new R calculation function
def R_update(R_i, dR):

    R = R_i + dR

    print('new radius after timestep is: ', R, 'cm')

    R_str = str(R)

    ee_owt.write('R (cm)           ' + R_str + '\n')
	
    return R, R_str

# defining function to update time
def time_updated(t):

    print('time = ', t , 'sec')
    t_str = str(t)
    ee_owt.write('time (sec)       ' + t_str + '\n')

    return t, t_str

# defining function to calculate new velocity
def v_update(R, t):

    # v = (E/(G_gam*rho*R**3))**(1/2)
    v = 0.4 * (R / t)

    print('new velocity value is: ', v, 'cm/sec')

    v_str = str(v)

    ee_owt.write('v (cm/sec)       ' + v_str + '\n')

    return v, v_str

# defining function to timestep forward
def timestep_fwd(t, dt):
    
    t = t + dt

    return t

# function to write to rtvar output file
def rtvar_write(t_str, R_str, v_str):
    
    rtvar_owt.write(t_str + '    ' + R_str + '    ' + v_str + '\n')

    return t_str, R_str, v_str


#######################################
#
# Main Program
#
#######################################

# Opening input deck
ee_in = open('eein.txt', 'r')

E_kt = float(ee_in.readline())
Alt = float(ee_in.readline())
R_0 = float(ee_in.readline())
dt = float(ee_in.readline())
tend = float(ee_in.readline())
G_gam = float(ee_in.readline())

# Closing input deck
ee_in.close()

# opening output files
ee_owt = open('eeowt.txt', 'a')
eos_owt = open('eosowt.txt', 'a')
rtvar_owt = open('rtvar.txt', 'a')

# writing initially to ee, EOS, and rtvar output files
E_kt_str, Alt_str, R_0_str, dt_str, tend_str, G_gam_str = initial_conds_write(E_kt, Alt, R_0, dt, tend, G_gam)

# simulation initialization

# calculate density of medium
rho, rho_str = find_rho(Alt)

# calculate atmospheric density ratio
rhorat, rhorat_str = find_rhorat(rho, rho_0)

# convert kT to erg
E, E_str = kt_to_erg(E_kt)

# calculate number of timesteps
ntps, ntps_int, ntps_str = calc_timesteps(dt, tend)

# initializing time
t, t_str = time_init(tend)

# calculate initial velocity
v_0, v_0_str = v_init(R_0, t)

# writing initial values to rtvar output file
t_str, R_str, v_str = rtvar_write(t_str, R_0_str, v_0_str)

# calculate change in R up to next timestep
dR, dR_str = drdt(v_0)

# calculate new R at next timestep
R, R_str = R_update(R_0, dR) 

# timestep forward - one timestep done, ntps - 1 to go
t = t + dt

# for loop to complete the simulation for the remainder of the timesteps
for i in range(1, ntps_int):

    t, t_str = time_updated(t)

    # calculate velocity
    v, v_str = v_update(R, t)

    # writing initial values to rtvar output file
    t_str, R_str, v_str = rtvar_write(t_str, R_str, v_str)

    # calculate dR
    dR, dR_str = drdt(v)

    # calculate new R
    R, R_str = R_update(R, dR)  

    # timestep forward dt
    t = timestep_fwd(t, dt)

# close output files
ee_owt.close()
rtvar_owt.close()
eos_owt.close()

print('STOP all done')
print('point source energy expansion simulation complete')