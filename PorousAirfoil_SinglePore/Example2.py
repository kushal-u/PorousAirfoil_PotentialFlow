"""
This script provides an example application of the Source Panel Vortex Method (SPVP)
for simulating the aerodynamic behavior of a solid airfoil. It computes the pressure 
distribution and aerodynamic coefficients (lift and drag) over a NACA 4-digit airfoil 
profile, using panel discretization. Although the full porous panel model is imported, 
this example focuses on the solid (non-porous) case as a baseline for comparison. 
It also includes various plotting functions for visualizing results such as Cp 
distributions and geometry features.
"""

import numpy as np
import math as math
import os                       # Added: To handle directory creation
import matplotlib.pyplot as plt # Added: To handle figure saving

from PLOT import *
from COMPUTATION.Hydraulic_Resistance import Hydraulic_Resistance
from COMPUTATION.Porous_SPVP import POROUS_SPVP, INIT_POROUS_GEOMETRY

if __name__ == "__main__":
    
    #%% USER-DEFINED PARAMETERS
    
    # Freestream conditions
    Vinf = 1                # Freestream velocity [arbitrary units]
    rhoinf = 1              # Freestream density [arbitrary units]
    Re = 100000             # Reynolds number
    AoA = 1                 # Angle of attack [degrees]
    
    # Airfoil definition
    numPan = 50            # Number of panels on the airfoil surface
    power = 1               # Spacing control parameter (point clustering)
    NameAirfoil = "0018"    # NACA 4-digit airfoil designation
    
    # Plotting options (1 = plot, 0 = skip)
    flagPlot = [1,      # Airfoil shape and panel normals
                1,      # Boundary and control points
                1,      # Pressure coefficient vectors
                1,      # Cp comparison: porous vs solid
                0,      # Streamlines
                0]      # Cp contour

    # Porous region geometry
    type = 'rectangle'                          # Pore shape
    pore_geometry = [0.157, 0.0125]             # Dimensions of the pore section
    L = 0.89                                    # Length of the porous medium
    a = 0.0125                                  # Height of the pores
    n = 1 / 0.166                               # Porosity coefficient (1/spacing)
    y0 = -0.02
    angle_pore = 0

    # Convergence criteria for the porous solver
    max_iter = 100                              # Max number of iterations
    tol = 1e-8                                  # Convergence tolerance
    err = 100                                   # Initial error value (arbitrary)

    #%% INITIALIZATION

    # Convert AoA to radians
    AoAR = AoA * np.pi / 180
    
    # Compute dynamic viscosity from Re number
    mu = Vinf * rhoinf / Re

    # Compute pore characteristics (hydraulic resistance, diameter, area)
    Rs, Dh, A = Hydraulic_Resistance(mu, L, type, pore_geometry)

    # Generate airfoil geometry and porous geometry configuration
    AoAR = AoA * np.pi / 180  # Recompute AoAR (redundant but harmless)
    XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point,numPan,pore_entry,pore_exit,omega_in,omega_out,low_point,high_point,\
        pore_intern_co_XB_low,pore_intern_co_YB_low,pore_intern_co_XB_high,\
        pore_intern_co_YB_high,S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low,S_pore_high,phi_pore_high, \
        pore_intern_co_XC_high, pore_intern_co_YC_high,entry_point, out_point = \
        INIT_POROUS_GEOMETRY(AoA,NameAirfoil,numPan,y0,angle_pore, a, power=power, is_straight=1)

    #%% DEFINE MAIN DICTIONARIES

    # Fluid properties
    Fluid_characteristics = {
        "Vinf": Vinf,
        "rhoinf": rhoinf,
        "Re": Re,
        "mu": mu,
        "AoA": AoA,
        "AoAR": AoAR
    }

    # Pore characteristics and geometry
    Pore_characteristics = {
        'type': type,
        'pore_geometry': pore_geometry,
        'L': L,
        'a': a,
        'n': n,
        'entry_point': entry_point,
        'out_point': out_point,
        'omega_in': omega_in,
        'omega_out': omega_out,
        'Rs': Rs,
        'pore_entry': pore_entry,
        'pore_exit': pore_exit,
        'Dh': Dh,
        'A': A,
        'pore_intern_co_XB_low': pore_intern_co_XB_low,
        'pore_intern_co_YB_low': pore_intern_co_YB_low,
        'pore_intern_co_XB_high': pore_intern_co_XB_high,
        'pore_intern_co_YB_high': pore_intern_co_YB_high,
        'S_pore_low': S_pore_low,
        'phi_pore_low': phi_pore_low,
        'pore_intern_co_XC_low': pore_intern_co_XC_low,
        'pore_intern_co_YC_low': pore_intern_co_YC_low,
        'S_pore_high': S_pore_high,
        'phi_pore_high': phi_pore_high,
        'pore_intern_co_XC_high': pore_intern_co_XC_high,
        'pore_intern_co_YC_high': pore_intern_co_YC_high,
        'low_point': low_point,
        'high_point': high_point
    }

    # Airfoil geometric data
    Airfoil_geometry = {
        'XB': XB,
        'YB': YB,
        'XC': XC,
        'YC': YC,
        'S': S,
        'phi': phi,
        'delta': delta,
        'beta': beta,
        'numPan': numPan,
        'power': power,
        'NameAirfoil': NameAirfoil
    }

    # %% CALCULATION

    # Run the porous SPVP solver
    Cp, Cp_Solid, Cp_inter_low, Cp_inter_high, CL, CL_Solid, CD, CD_Solid, lam, gamma = \
        POROUS_SPVP(tol, max_iter, Pore_characteristics, Fluid_characteristics, Airfoil_geometry)

    # %% PLOTTING RESULTS
    
    # Turn on interactive mode so code continues even if PLOT functions use plt.show()
    plt.ion() 

    # Plot airfoil with porous sections
    PLOT_AIRFOIL(XB, YB, low_point, high_point, alone=0)

    # Plot comparison between porous and solid Cp distributions
    PLOT_CP_COMPARISON(XB, XC, Cp, Cp_Solid, pore_entry, pore_exit,
                       label1='Porous', label2='Solid', alone=False)

    # Plot Cp distribution on the pressure side
    """PLOT_CP_PRESSURE_SIDE(XC, YC, Cp, Cp_inter_low, low_point,
                           pore_intern_co_XC_low, alone=False)

    # Plot Cp distribution on the suction side
    PLOT_CP_SUCCION_SIDE(XC, YC, Cp, Cp_inter_high, high_point,
                          pore_intern_co_XC_high, alone=False)
    """
    
    # Final plot: full diagnostics depending on flags
    PLOT_ALL(flagPlot, XB, YB, numPan, XC, YC, S, delta, Cp, phi, Vinf, AoA, lam, gamma)

    # Print aerodynamic coefficients
    print('CL_Porous = ', CL)
    print('CL_solid = ', CL_Solid)
    print('CD_Porous = ', CD)
    print('CD_solid = ', CD_Solid)
    print('CL_improvement = ', (CL-CL_Solid)/CL)

    # %% SAVE GRAPHS
    print("======= SAVING FIGURES =======")
    
    # 1. Create directory
    save_dir = "Results_Porous"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 2. Iterate through all open figures and save them
    # This works regardless of which PLOT functions above were called
    fignums = plt.get_fignums()
    
    for i in fignums:
        filename = f"Porous_NACA{NameAirfoil}_AoA{AoA}_Fig{i}.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.figure(i) # Select the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

    # 3. Finalize
    plt.ioff() # Turn off interactive mode
    plt.show() # Keep windows open for user to view