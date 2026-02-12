import numpy as np
import math as math
import matplotlib.pyplot as plt

from COMPUTATION.Hydraulic_Resistance import Hydraulic_Resistance
from COMPUTATION.Porous_SPVP import INIT_POROUS_GEOMETRY, POROUS_SPVP

if __name__ == "__main__":
    CL_porous_list = []
    CD_porous_list = []
    CL_solid_list = []
    CD_solid_list = []
    AoA_list  = np.linspace(0,10,11)                                                                        # Angle of attack [deg]

    for AoA in AoA_list:
        #%% User-defined knowns
        Vinf = 1                                                                        # Freestream velocity [] (just leave this at 1)
        rhoinf = 1                                                                      # Density [] (just leave this at 1)
        Re = 160000                                                                      # Reynolds number
        
        numPan = 50
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
        pore_geometry = [0.157,0.01]
        L = 0.89
        a = 0.01                       #Height of the pores
        n = 1/0.166
        y0 = -0.01
        angle_pore = 0

        #Convergence Variables
        max_iter = 100
        tol = 1e-8
        err = 100

        #%% Initialisation 
        # Fluid characteristics
        mu = Vinf*rhoinf/Re
        #Pores characteristics
        Rs,Dh,A = Hydraulic_Resistance(mu,L,type,pore_geometry)
        
        print('AoA = ',AoA)
        AoAR = AoA*np.pi/180
        #Pores characteristics
        Rs,Dh,A = Hydraulic_Resistance(mu,L,type,pore_geometry)
        XB,YB,XC,YC,S,phi,delta,beta,entry_point,out_point,numPan,pore_entry,pore_exit,omega_in,omega_out,low_point,high_point,pore_intern_co_XB_low,pore_intern_co_YB_low,pore_intern_co_XB_high,pore_intern_co_YB_high,S_pore_low,phi_pore_low, pore_intern_co_XC_low, pore_intern_co_YC_low,S_pore_high,phi_pore_high, pore_intern_co_XC_high, pore_intern_co_YC_high,entry_point, out_point = INIT_POROUS_GEOMETRY(AoA,NameAirfoil,numPan,y0,angle_pore,a,power=power,is_straight=1)

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

        # %% CALCULATION
        Cp, Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_SPVP(tol,max_iter,Pore_characteristics,Fluid_characteristics,Airfoil_geometry)
        CL_solid_list.append(CL_Solid)
        CD_solid_list.append(CD_Solid)
        CL_porous_list.append(CL)
        CD_porous_list.append(CD)
    
    CL_CD_solid_list = []
    CL_CD_porous_list = []
    for i in range(len(CL_solid_list)):
        CL_CD_solid_list.append(CL_solid_list[i]/CD_solid_list[i])
        CL_CD_porous_list.append(CL_porous_list[i]/CD_porous_list[i])
    
    # %% PLOT Result
    fig = plt.figure() 
    plt.plot(AoA_list,CL_porous_list,label='Porous')
    plt.plot(AoA_list,CL_solid_list, label = 'Solid')
    plt.xlabel('Angle of Attack (AoA)')
    plt.ylabel('Lift Coefficient (CL)')
    plt.title('Lift Coefficient vs Angle of Attack')
    plt.legend()

    fig = plt.figure() 
    plt.plot(AoA_list,CD_porous_list,label='Porous')
    plt.plot(AoA_list,CD_solid_list, label = 'Solid')
    plt.xlabel('Angle of Attack (AoA)')
    plt.ylabel('Drag Coefficient (CD)')
    plt.title('Drag Coefficient vs Angle of Attack')
    plt.legend()
    print(AoA_list)
    print(CD_solid_list)
    fig = plt.figure() 
    plt.plot(AoA_list,CL_CD_porous_list,label='Porous')
    plt.plot(AoA_list,CL_CD_solid_list, label = 'Solid')
    plt.xlabel('Angle of Attack (AoA)')
    plt.ylabel('CL/CD')
    plt.title('CL/CD vs Angle of Attack')
    plt.legend()

    plt.show()
    #PLOT_AIRFOIL(XB,YB,low_point,high_point,alone=1)
    #PLOT_CP_COMPARISON(XB,XC,Cp,Cp_Solid,pore_entry,pore_exit,label1='Porous',label2='Solid',alone = False)
    #PLOT_CP_PRESSURE_SIDE(XC,YC, Cp, Cp_inter_low, low_point, pore_intern_co_XC_low, alone = False)
    #PLOT_CP_SUCCION_SIDE(XC,YC, Cp, Cp_inter_high, high_point, pore_intern_co_XC_high, alone = False)
    
    #PLOT_ALL(flagPlot,XB,YB,numPan,XC,YC,S,delta,Cp,phi,Vinf,AoA,lam,gamma)



