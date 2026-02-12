import numpy as np
import math as math
import matplotlib.pyplot as plt


from COMPUTATION.SPVP_Airfoil import SPVP
from COMPUTATION.Hydraulic_Resistance import Hydraulic_Resistance
from PLOT import *
from GEOMETRY.Hydraulic_GEOMETRY import *
from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY
from GEOMETRY.NACA import GENERATE_NACA4, GENERATE_EQUAL_NACA4
from GEOMETRY.ELLIPSE import GENERATE_ELLIPSE, GENERATE_EQUAL_ELLIPSE
from COMPUTATION.COMPUTE import COMPUTE_LIFT_MOMENT

def INIT_POROUS_GEOMETRY(AoA,NameAirfoil,numPan,y0,angle_pore,a,power=1,is_straight=1):
    AoAR = AoA*(np.pi/180)                                                          # Angle of attack [rad]
    XB, YB = GENERATE_NACA4(NameAirfoil,NumPan=numPan,power=power)
    XC,YC,S,phi,delta,beta = GENERATE_GEOMETRY(numPan,XB,YB,AoAR)
    omega_in = angle_pore
    omega_out = angle_pore
    XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point = Refine_GEOMETRY(XB,YB,NameAirfoil,y0,angle_pore,AoAR)
    numPan = len(XB)-1

    pore_entry = Hydraulic_GEOMETRY(XC,YC,omega_in,a,entry_point)
    pore_exit = Hydraulic_GEOMETRY(XC,YC,omega_out,a,out_point)
    low_point,high_point = pressure_succion_side(numPan,pore_entry,pore_exit)
    
    pore_intern_co_XB_low = np.linspace(XB[pore_entry[0]],XB[pore_exit[-1]],20)
    pore_intern_co_YB_low = np.linspace(YB[pore_entry[0]],YB[pore_exit[-1]],20)

    #print(XB[low_point[-1]+1],"     ",XB[low_point[0]])
    #print(YB[low_point[-1]+1],"     ",YB[low_point[0]])


    pore_intern_co_XB_high = np.linspace(XB[pore_exit[0]],XB[pore_entry[-1]],20)
    pore_intern_co_YB_high = np.linspace(YB[pore_exit[0]],YB[pore_entry[-1]],20)

    """print(pore_intern_co_XB_high)
    print(pore_intern_co_YB_high)
    print(pore_intern_co_XB_low)
    print(pore_intern_co_YB_low)"""
    S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low = Pore_Geometry(pore_intern_co_XB_low,pore_intern_co_YB_low)
    S_pore_high,phi_pore_high, pore_intern_co_XC_high, pore_intern_co_YC_high = Pore_Geometry(pore_intern_co_XB_high,pore_intern_co_YB_high)
    
    return XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point,numPan,pore_entry,pore_exit,omega_in,omega_out,low_point,high_point,pore_intern_co_XB_low,pore_intern_co_YB_low,pore_intern_co_XB_high,pore_intern_co_YB_high,S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low,S_pore_high,phi_pore_high, pore_intern_co_XC_high, pore_intern_co_YC_high,entry_point, out_point

def INIT_POROUS_GEOMETRY_ELLIPSE(AoA,numPan,y0,angle_pore,a,power=1,is_straight=1):
    AoAR = AoA*(np.pi/180)                                                          # Angle of attack [rad]
    XB, YB = GENERATE_ELLIPSE(NumPan=numPan,power=power)
    XC,YC,S,phi,delta,beta = GENERATE_GEOMETRY(numPan,XB,YB,AoAR)
    omega_in = angle_pore
    omega_out = angle_pore
    XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point = Refine_GEOMETRY_ELLIPSE(XB,YB,y0,angle_pore,AoAR)
    numPan = len(XB)-1

    pore_entry = Hydraulic_GEOMETRY(XC,YC,omega_in,a,entry_point)
    pore_exit = Hydraulic_GEOMETRY(XC,YC,omega_out,a,out_point)
    low_point,high_point = pressure_succion_side(numPan,pore_entry,pore_exit)
    
    pore_intern_co_XB_low = np.linspace(XB[pore_entry[0]],XB[pore_exit[-1]],20)
    pore_intern_co_YB_low = np.linspace(YB[pore_entry[0]],YB[pore_exit[-1]],20)

    #print(XB[low_point[-1]+1],"     ",XB[low_point[0]])
    #print(YB[low_point[-1]+1],"     ",YB[low_point[0]])


    pore_intern_co_XB_high = np.linspace(XB[pore_exit[0]],XB[pore_entry[-1]],20)
    pore_intern_co_YB_high = np.linspace(YB[pore_exit[0]],YB[pore_entry[-1]],20)

    S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low = Pore_Geometry(pore_intern_co_XB_low,pore_intern_co_YB_low)
    S_pore_high,phi_pore_high, pore_intern_co_XC_high, pore_intern_co_YC_high = Pore_Geometry(pore_intern_co_XB_high,pore_intern_co_YB_high)
    return XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point,numPan,pore_entry,pore_exit,omega_in,omega_out,low_point,high_point,pore_intern_co_XB_low,pore_intern_co_YB_low,pore_intern_co_XB_high,pore_intern_co_YB_high,S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low,S_pore_high,phi_pore_high, pore_intern_co_XC_high, pore_intern_co_YC_high,entry_point, out_point

def POROUS_SPVP(tol,max_iter,Pore_characteristics,Fluid_characteristics,Airfoil_geometry):
    
    entry_point = Pore_characteristics['entry_point']
    out_point = Pore_characteristics['out_point']
    pore_intern_co_XC_low = Pore_characteristics['pore_intern_co_XC_low']
    pore_intern_co_XC_high = Pore_characteristics['pore_intern_co_XC_high']
    pore_intern_co_YC_low = Pore_characteristics['pore_intern_co_YC_low']
    pore_intern_co_YC_high = Pore_characteristics['pore_intern_co_YC_high']
    A = Pore_characteristics['A']
    low_point = Pore_characteristics['low_point']
    high_point = Pore_characteristics['high_point']

    XB = Airfoil_geometry['XB']
    YB = Airfoil_geometry['YB']

    #%% First round without porous

    Cp,lam,gamma,CL,CM,CD = SPVP(Fluid_characteristics,Airfoil_geometry,is_porous = 0)
    Cp_Solid = Cp
    CL_Solid = CL
    CM_Solid = CM
    CD_Solid = CD
    err = 100000
    iter =0
    #%% Loop with porous
    while err > tol and iter < max_iter: 
        Delta_Cp = Cp[entry_point]-Cp[out_point]
        Cp,lam,gamma,CL,CM,CD = SPVP(Fluid_characteristics,Airfoil_geometry,Pore_characteristics,is_porous = 1,Delta_Cp=Delta_Cp, low_point= low_point, high_point = high_point)
        err = abs((Delta_Cp-(Cp[entry_point]-Cp[out_point]))/Delta_Cp)
        print("Delta_cp = ", Delta_Cp, '     err = ', err)
        #print("Cp_entry = ",Cp[entry_point], "      Cp_out = ",Cp[out_point])
        iter += 1
        if iter == max_iter:
            print("Maximum number of iterations reached")
        if err < tol:
            print("Convergence reached in ", iter, ' iterations')
    
    # %% CALCULATION OF FINAL CP
    Cp_inter_low = np.zeros(len(pore_intern_co_YC_low))
    for i in range(len(pore_intern_co_YC_low)):
        Cp_inter_low[i] = Cp[entry_point] - Delta_Cp*(pore_intern_co_XC_low[i]-XB[entry_point])/(XB[out_point]-XB[entry_point])
        #Cp_inter_low[i] = Cp[entry_point] - Delta_Cp*(pore_intern_co_YC_low[i]-YB[entry_point])/(YB[out_point]-YB[entry_point])
    
    Cp_inter_high = np.zeros(len(pore_intern_co_YC_high))
    for i in range(len(pore_intern_co_YC_high)):
        Cp_inter_low[i] = Cp[entry_point] - Delta_Cp*(pore_intern_co_XC_high[i]-XB[entry_point])/(XB[out_point]-XB[entry_point])
        #Cp_inter_low[i] = Cp[entry_point] - Delta_Cp*(pore_intern_co_YC_high[i]-YB[entry_point])/(YB[out_point]-YB[entry_point])

    CL,CM,CD = COMPUTE_LIFT_MOMENT(Cp,Fluid_characteristics,Airfoil_geometry,Pore_characteristics,Delta_Cp,Cp_inter_low,Cp_inter_high,A,is_porous=1)
    return Cp, Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma

def POROUS_AIRFOIL(numPan,AoA,Vinf,rhoinf,Re,L,pore_geometry,type,NameAirfoil,y0,angle_pore,a,n,power,tol,max_iter):
    #%% Initialisation 
    # Fluid characteristics
    AoAR = AoA*np.pi/180
    mu = Vinf*rhoinf/Re
    #Pores characteristics
    Rs,Dh,A = Hydraulic_Resistance(mu,L,type,pore_geometry)


    AoAR = AoA*np.pi/180
    XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point,numPan,pore_entry, pore_exit,omega_in,omega_out,low_point,high_point,pore_intern_co_XB_low,pore_intern_co_YB_low,pore_intern_co_XB_high,pore_intern_co_YB_high,S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low,S_pore_high,phi_pore_high, pore_intern_co_XC_high, pore_intern_co_YC_high,entry_point, out_point = INIT_POROUS_GEOMETRY(AoA,NameAirfoil,numPan,y0,angle_pore,a,power=power,is_straight=1)
    
    #%% Macro variable
    Fluid_characteristics = {
        "Vinf" : Vinf,
        "rhoinf" : rhoinf,
        "Re" : Re,
        "mu" : mu,
        'AoA' : AoA,
        'AoAR' : AoAR
    }

    Pore_characteristics = {
        'type' : type,
        'pore_geometry' : pore_geometry,
        'L' : L,
        'a' : a,
        'n' : n,
        'entry_point' : entry_point,
        'out_point' : out_point,
        'omega_in' : omega_in,
        'omega_out' : omega_out,
        'Rs' : Rs,
        'pore_entry' : pore_entry,
        'pore_exit' : pore_exit,
        'Dh' : Dh,
        'A' : A,
        'pore_intern_co_XB_low' : pore_intern_co_XB_low,
        'pore_intern_co_YB_low' : pore_intern_co_YB_low,
        'pore_intern_co_XB_high' : pore_intern_co_XB_high,
        'pore_intern_co_YB_high' : pore_intern_co_YB_high,
        'S_pore_low' : S_pore_low,
        'phi_pore_low' : phi_pore_low,
        'pore_intern_co_XC_low' : pore_intern_co_XC_low,
        'pore_intern_co_YC_low' : pore_intern_co_YC_low,
        'S_pore_high' : S_pore_high,
        'phi_pore_high' : phi_pore_high,
        'pore_intern_co_XC_high' : pore_intern_co_XC_high,
        'pore_intern_co_YC_high' : pore_intern_co_YC_high,
        'low_point' : low_point,
        'high_point' : high_point
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

    #%% CALCULATION
    Cp, Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_SPVP(tol,max_iter,Pore_characteristics,Fluid_characteristics,Airfoil_geometry)

    return Fluid_characteristics,Pore_characteristics,Airfoil_geometry,Cp,Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma

def POROUS_ELLIPSE(numPan,AoA,Vinf,rhoinf,Re,L,pore_geometry,type,y0,angle_pore,a,n,power,tol,max_iter):
    #%% Initialisation 
    # Fluid characteristics
    AoAR = AoA*np.pi/180
    mu = Vinf*rhoinf/Re
    #Pores characteristics
    Rs,Dh,A = Hydraulic_Resistance(mu,L,type,pore_geometry)


    AoAR = AoA*np.pi/180
    XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point,numPan,pore_entry, pore_exit,omega_in,omega_out,low_point,high_point,pore_intern_co_XB_low,pore_intern_co_YB_low,pore_intern_co_XB_high,pore_intern_co_YB_high,S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low,S_pore_high,phi_pore_high, pore_intern_co_XC_high, pore_intern_co_YC_high,entry_point, out_point = INIT_POROUS_GEOMETRY_ELLIPSE(AoA,numPan,y0,angle_pore,a,power=power,is_straight=1)
    
    #%% Macro variable
    Fluid_characteristics = {
        "Vinf" : Vinf,
        "rhoinf" : rhoinf,
        "Re" : Re,
        "mu" : mu,
        'AoA' : AoA,
        'AoAR' : AoAR
    }

    Pore_characteristics = {
        'type' : type,
        'pore_geometry' : pore_geometry,
        'L' : L,
        'a' : a,
        'n' : n,
        'entry_point' : entry_point,
        'out_point' : out_point,
        'omega_in' : omega_in,
        'omega_out' : omega_out,
        'Rs' : Rs,
        'pore_entry' : pore_entry,
        'pore_exit' : pore_exit,
        'Dh' : Dh,
        'A' : A,
        'pore_intern_co_XB_low' : pore_intern_co_XB_low,
        'pore_intern_co_YB_low' : pore_intern_co_YB_low,
        'pore_intern_co_XB_high' : pore_intern_co_XB_high,
        'pore_intern_co_YB_high' : pore_intern_co_YB_high,
        'S_pore_low' : S_pore_low,
        'phi_pore_low' : phi_pore_low,
        'pore_intern_co_XC_low' : pore_intern_co_XC_low,
        'pore_intern_co_YC_low' : pore_intern_co_YC_low,
        'S_pore_high' : S_pore_high,
        'phi_pore_high' : phi_pore_high,
        'pore_intern_co_XC_high' : pore_intern_co_XC_high,
        'pore_intern_co_YC_high' : pore_intern_co_YC_high,
        'low_point' : low_point,
        'high_point' : high_point
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
        'power' : power
    }

    #%% CALCULATION
    Cp, Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_SPVP(tol,max_iter,Pore_characteristics,Fluid_characteristics,Airfoil_geometry)

    return Fluid_characteristics,Pore_characteristics,Airfoil_geometry,Cp,Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma



if __name__ == "__main__":
    
    #%% User-defined knowns
    Vinf = 1                                                                        # Freestream velocity [] (just leave this at 1)
    rhoinf = 1                                                                      # Density [] (just leave this at 1)
    Re = 160000                                                                      # Reynolds number
    AoA  = -5                                                                     # Angle of attack [deg]
    
    numPan = 250
    power = 1
    NameAirfoil = "0018"
    
    # Plotting flags
    flagPlot = [1,      # Airfoil with panel normal vectors
                1,      # Geometry boundary pts, control pts, first panel, second panel
                1,      # Cp vectors at airfoil surface panels
                1,      # Pressure coefficient comparison (XFOIL vs. VPM)
                0,      # Airfoil streamlines
                0]      # Pressure coefficient contour
    
    #Pore geometry
    type = 'rectangle'
    pore_geometry = [0.157,0.007]
    L = 0.89
    a = 0.007                       #Height of the pores
    n = 1/0.166
    y0 = 0.2
    angle_pore = 90


    #Convergence Variables
    max_iter = 100
    tol = 1e-8
    err = 100

    Fluid_characteristics,Pore_characteristics,Airfoil_geometry,Cp,Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_AIRFOIL(numPan,AoA,Vinf,rhoinf,Re,L,pore_geometry,type,NameAirfoil,y0,angle_pore,a,n,power,tol,max_iter)

    # Extract Result
    XB = Airfoil_geometry['XB']
    YB = Airfoil_geometry['YB']
    XC = Airfoil_geometry['XC']
    YC = Airfoil_geometry['YC']
    S = Airfoil_geometry['S']
    delta = Airfoil_geometry['delta']
    phi = Airfoil_geometry['phi']
    low_point = Pore_characteristics['low_point']
    high_point = Pore_characteristics['high_point']
    pore_entry = Pore_characteristics['pore_entry']
    pore_exit = Pore_characteristics['pore_exit']


    # %% PLOT Result
    PLOT_AIRFOIL(XB,YB,low_point,high_point,alone=0)
    PLOT_CP_COMPARISON(XB,XC,Cp,Cp_Solid,pore_entry,pore_exit,label1='Porous',label2='Solid',alone = False)
    #PLOT_CP_PRESSURE_SIDE(XC,YC, Cp, Cp_inter_low, low_point, pore_intern_co_XC_low, alone = False)
    #PLOT_CP_SUCCION_SIDE(XC,YC, Cp, Cp_inter_high, high_point, pore_intern_co_XC_high, alone = False)
    
    print('CL_Porous = ', CL)
    print('CL_solid = ', CL_Solid)
    print('CD_Porous = ', CD)
    print('CD_solid = ', CD_Solid)
    PLOT_ALL(flagPlot,XB,YB,numPan,XC,YC,S,delta,Cp,phi,Vinf,AoA,lam,gamma)



