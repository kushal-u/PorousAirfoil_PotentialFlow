# SOURCE/VORTEX PANEL METHOD

import numpy as np
import math as math

from COMPUTATION.COMPUTE import *
from PLOT import PLOT_ALL
from GEOMETRY.NACA import GENERATE_NACA4
from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY
from GEOMETRY.PANEL_DIRECTIONS import PANEL_DIRECTIONS

def SPVP(Fluid_characteristics,Airfoil_geometry,Pore_characteristics={},is_porous = 1,Delta_Cp=0,low_point = [],high_point = []):
    #%%Unpack maro variables
    AoAR = Fluid_characteristics['AoAR']                                                          # Angle of attack [rad]
    XB = Airfoil_geometry['XB']
    YB = Airfoil_geometry['YB']
    XC = Airfoil_geometry['XC']
    YC = Airfoil_geometry['YC']
    phi = Airfoil_geometry['phi']
    delta = Airfoil_geometry['delta']
    beta = Airfoil_geometry['beta']
    S = Airfoil_geometry['S']
    numPan = Airfoil_geometry['numPan']

    if is_porous:
        rhoinf = Fluid_characteristics['rhoinf']
        Vinf = Fluid_characteristics['Vinf']
        Rs = Pore_characteristics['Rs']
        a = Pore_characteristics['a']
        n = Pore_characteristics['n']
        pore_entry = Pore_characteristics['pore_entry']
        pore_out = Pore_characteristics['pore_exit']
        omega_in = Pore_characteristics['omega_in']
        omega_out = Pore_characteristics['omega_out']
        Dh = Pore_characteristics['Dh']
        mu = Fluid_characteristics['mu']
    else:
        rhoinf = 1
        Vinf = 1
        Rs = 1
        a = 1
        n = 0
        pore_entry = []
        pore_out = []
        omega_in = 0
        omega_out = 0
        Dh = 0
        mu = 1

    #%% CHECK PANEL DIRECTIONS - FLIP IF NECESSARY
    PANEL_DIRECTIONS(numPan,XB,YB)

    #%% COMPUTE SOURCE AND VORTEX PANEL STRENGTHS - REF [10]
    I, J = COMPUTE_IJ_SPM(XC,YC,XB,YB,phi,S)                                        # Call COMPUTE_IJ_SPM function (Refs [2] and [3])
    K, L = COMPUTE_KL_VPM(XC,YC,XB,YB,phi,S)                                        # Call COMPUTE_KL_VPM function (Refs [6] and [7])

    A = COMPUTE_A_SPVP(numPan,I,K,J,L)
    b = COMPUTE_b_SPVP(Airfoil_geometry,Fluid_characteristics,Delta_Cp,Pore_characteristics,is_porous)
    lam, gamma = COMPUTE_SOLUTION_SPVP(A,b)

    #%% COMPUTE PANEL VELOCITIES AND PRESSURE COEFFICIENTS
    Vt, Cp = COMPUTE_Vt_Cp(Airfoil_geometry,Fluid_characteristics,Delta_Cp,gamma,lam,b,J,L,Pore_characteristics,is_porous = is_porous)

    #%% COMPUTE LIFT AND MOMENT COEFFICIENTS
    CL,CM,CD = COMPUTE_LIFT_MOMENT(Cp,Fluid_characteristics,Airfoil_geometry,Pore_characteristics)
    return Cp,lam,gamma,CL,CM,CD

def SOLID_AIRFOIL(NameAirfoil,numPan,power,AoA,Vinf,rhoinf,Re):
    #%% initialisation
    AoAR = AoA*np.pi/180
    XB, YB = GENERATE_NACA4(NameAirfoil,NumPan=numPan,power=power)
    XC,YC,S,phi,delta,beta = GENERATE_GEOMETRY(numPan,XB,YB,AoAR)
    mu = Vinf*rhoinf/Re

    #%% Macro Variables
    Fluid_characteristics = {
        'Vinf' : Vinf,
        'rhoinf' : rhoinf,
        'AoA' : AoA,
        'AoAR' : AoAR,
        'mu' : mu,
        'Re' : Re
    }

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

    #%% SPVP CALCULATION
    Cp,lam,gamma,CL,CM,CD = SPVP(Fluid_characteristics,Airfoil_geometry,is_porous=0)

    return Fluid_characteristics,Airfoil_geometry,Cp,lam,gamma,CL,CM,CD



if __name__ == '__main__':
    #%% User-defined knowns
    Vinf = 1                                                                        # Freestream velocity [] (just leave this at 1)
    rhoinf = 1
    Re = 1000
    mu = Vinf*rhoinf/Re
    AoA  = 4                                                                        # Angle of attack [deg]
    AoAR = np.pi*AoA/180
    
    # Plotting flags
    flagPlot = [1,      # Airfoil with panel normal vectors
                1,      # Geometry boundary pts, control pts, first panel, second panel
                1,      # Cp vectors at airfoil surface panels
                1,      # Pressure coefficient comparison (XFOIL vs. VPM)
                0,      # Airfoil streamlines
                0]      # Pressure coefficient contour

    # AirFoil panels
    numPan = 100
    NameAirfoil = "0018"
    power = 1

    Fluid_characteristics,Airfoil_geometry,Cp,lam,gamma,CL,CM,CD = SOLID_AIRFOIL(NameAirfoil,numPan,power,AoA,Vinf,rhoinf,Re)

    # Extract Result
    XB = Airfoil_geometry['XB']
    YB = Airfoil_geometry['YB']
    XC = Airfoil_geometry['XC']
    YC = Airfoil_geometry['YC']
    S = Airfoil_geometry['S']
    delta = Airfoil_geometry['delta']
    phi = Airfoil_geometry['phi']

    # %% PLOT
    PLOT_ALL(flagPlot,XB,YB,numPan,XC,YC,S,delta,Cp,phi,Vinf,AoA,lam,gamma)


    #%% RESULTS
    # Print the results to the Console
    print("======= RESULTS =======")
    print("Lift Coefficient (CL)")
    print("  SPVP : %2.8f" % CL)                                                    # From this SPVP code
    print("Moment Coefficient (CM)")
    print("  SPVP : %2.8f" % CM)                                                    # From this SPVP code