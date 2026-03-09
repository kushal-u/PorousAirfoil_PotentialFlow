"""
===========================================
    Porous Panel Method (SPVP) Driver
===========================================

This script simulates the aerodynamic behavior of a NACA 4-digit airfoil using 
a custom implementation of the Source–Panel–Vortex–Porous (SPVP) method.

"""

import numpy as np
import math as math
import os                       # Added for directory handling
import matplotlib.pyplot as plt # Added for saving figures

# Importing custom modules from the project
from COMPUTATION.SPVP_Airfoil import SPVP                          # Main SPVP computation function
from PLOT import PLOT_ALL                              # Function to generate all plots
from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY        # Function to compute geometry from boundary points
from GEOMETRY.NACA import GENERATE_NACA4, GENERATE_EQUAL_NACA4               # Function to generate NACA 4-digit airfoils

# Main entry point of the script
if __name__ == '__main__':
    #%% ================================
    #          User-defined inputs
    #==================================
    Vinf = 1               # Freestream velocity [unitless, normalized]
    rhoinf = 1             # Freestream density
    Re = 160000              # Reynolds number
    mu = Vinf * rhoinf / Re  # Dynamic viscosity
    AoA  = 0               # Angle of attack [degrees]
    AoAR = np.pi * AoA / 180  # Angle of attack [radians]

    # Plot flags: each 1 enables a specific visualization
    flagPlot = [1,      # 0: Airfoil with panel normal vectors
                1,      # 1: Geometry (boundary pts, control pts, panel indicators)
                1,      # 2: Cp vectors along the airfoil surface
                1,      # 3: Cp comparison plot (XFOIL vs SPVP)
                1,      # 4: Streamlines over the airfoil
                1]      # 5: Cp contour plot over domain

    # Airfoil parameters
    numPan = 100             # Number of panels (discretization segments)
    NameAirfoil = "0018"     # NACA 4-digit airfoil code
    power = 1                # Point spacing exponent for clustering near leading/trailing edges

    #%% ================================
    #          Geometry initialization
    #==================================
    # Generate boundary coordinates for the NACA airfoil
    XB, YB = GENERATE_NACA4(NameAirfoil, NumPan=numPan, power=power)

    # Compute panel geometry based on boundary coordinates and angle of attack
    XC, YC, S, phi, delta, beta = GENERATE_GEOMETRY(numPan, XB, YB, AoAR)

    #%% ================================
    #       Dictionary-based storage
    #==================================
    # Store fluid parameters
    Fluid_characteristics = {
        'Vinf' : Vinf,
        'rhoinf' : rhoinf,
        'AoA' : AoA,
        'AoAR' : AoAR,
        'mu' : mu
    }

    # Store airfoil geometry data
    Airfoil_geometry = {
        'XB' : XB,
        'YB' : YB,
        'XC' : XC,
        'YC' : YC,
        'S' : S,
        'phi' : phi,
        'delta' : delta,
        'beta' : beta,
        'numPan' : numPan,
        'power' : power,
        'NameAirfoil' : NameAirfoil
    }

    #%% ================================
    #        SPVP Main Calculation
    #==================================
    # Compute aerodynamic results using the SPVP method
    Cp, lam, gamma, CL, CM, CD = SPVP(
        Fluid_characteristics,
        Airfoil_geometry,
        is_porous=0  # 0 = non-porous airfoil
    )

    #%% ================================
    #           Visualization
    #==================================
    
    # 1. Turn on interactive mode to prevent code from stopping if PLOT_ALL has plt.show()
    plt.ion()

    # Generate all requested plots
    PLOT_ALL(
        flagPlot,XB,YB,numPan,XC,YC,S,delta,Cp,phi,Vinf,AoA,lam,gamma
    )

    # 2. Save Logic
    # Create a directory for results if it doesn't exist
    save_dir = "Results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Get all open figure numbers
    fignums = plt.get_fignums()
    
    print("======= SAVING FIGURES =======")
    for i in fignums:
        # Define a filename based on Airfoil name, AoA, and Figure Number
        filename = f"NACA{NameAirfoil}_AoA{AoA}_Fig{i}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Switch to the figure and save it
        plt.figure(i)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

    # 3. Turn interactive mode off and show plots to user
    plt.ioff()
    
    #%% ================================
    #            Output results
    #==================================
    # Print aerodynamic coefficients to the console
    print('NumPan = ',numPan)
    print("======= RESULTS =======")
    print("Lift Coefficient (CL)")
    print("  SPVP : %2.8f" % CL)

    print("Moment Coefficient (CM)")
    print("  SPVP : %2.8f" % CM)

    print("Drag Coefficient (CD)")
    print("  SPVP : %2.8f" % CD)

    # Keep windows open until manually closed
    plt.show()