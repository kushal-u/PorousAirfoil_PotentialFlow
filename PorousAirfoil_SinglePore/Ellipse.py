"""
===========================================
    Porous Panel Method (SPVP) Driver
===========================================

This script simulates the aerodynamic behavior of a NACA 4-digit airfoil using 
a custom implementation of the Source–Panel–Vortex–Porous (SPVP) method.

"""

import numpy as np
import math as math

# Importing custom modules from the project
from COMPUTATION.SPVP_Airfoil import SPVP                          # Main SPVP computation function
from PLOT import PLOT_ALL                              # Function to generate all plots
from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY        # Function to compute geometry from boundary points
from GEOMETRY.ELLIPSE import GENERATE_ELLIPSE, GENERATE_EQUAL_ELLIPSE               # Function to generate NACA 4-digit airfoils

# Main entry point of the script
if __name__ == '__main__':
    #%% ================================
    #          User-defined inputs
    #==================================
    Vinf = 1               # Freestream velocity [unitless, normalized]
    rhoinf = 1             # Freestream density
    Re = 160000              # Reynolds number
    mu = Vinf * rhoinf / Re  # Dynamic viscosity
    AoA  = 3               # Angle of attack [degrees]
    AoAR = np.pi * AoA / 180  # Angle of attack [radians]

    # Plot flags: each 1 enables a specific visualization
    flagPlot = [1,      # 0: Airfoil with panel normal vectors
                0,      # 1: Geometry (boundary pts, control pts, panel indicators)
                0,      # 2: Cp vectors along the airfoil surface
                0,      # 3: Cp comparison plot (XFOIL vs SPVP)
                0,      # 4: Streamlines over the airfoil
                0]      # 5: Cp contour plot over domain

    # Airfoil parameters
    numPan = 200             # Number of panels (discretization segments)
    NameAirfoil = "0018"     # NACA 4-digit airfoil code
    power = 1                # Point spacing exponent for clustering near leading/trailing edges

    #%% ================================
    #          Geometry initialization
    #==================================
    # Generate boundary coordinates for the NACA airfoil
    XB, YB = GENERATE_ELLIPSE(NumPan=numPan, power=power)

    #YB[0], YB[-1] = 0, 0

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
    # Generate all requested plots
    PLOT_ALL(
        flagPlot, XB, YB, numPan,
        XC, YC, S, delta, Cp, phi,
        Vinf, AoA, lam, gamma
    )

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
