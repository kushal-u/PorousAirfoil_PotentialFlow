import numpy as np
import math as math
import matplotlib.pyplot as plt
from matplotlib import path
import os

# Chemin complet du fichier
path = "SAVE_FIGURE/XFOIL_COMPARAISON/Cp_AoA4.pdf"
# Créer le dossier si nécessaire
os.makedirs(os.path.dirname(path), exist_ok=True)

from COMPUTATION.SPVP_Airfoil import SPVP
from X_FOIL.XFOIL import XFOIL_DATA
from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY
from GEOMETRY.NACA import GENERATE_NACA4
from COMPUTATION.COMPUTE import COMPUTE_LIFT_MOMENT

Vinf = 1               # Freestream velocity [unitless, normalized]
rhoinf = 1             # Freestream density
Re = 100000              # Reynolds number
mu = Vinf * rhoinf / Re  # Dynamic viscosity
AoA  =  4            # Angle of attack [degrees]
AoAR = np.pi * AoA / 180  # Angle of attack [radians]


# Airfoil parameters
numPan = 100             # Number of panels (discretization segments)
NameAirfoil = "0018"     # NACA 4-digit airfoil code
power = 3                # Point spacing exponent for clustering near leading/trailing edges

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
x_xfoil,cp_xfoil = XFOIL_DATA(NameAirfoil,4)

t = 0.24
c=1
y_xfoil = 5 * t * c * (0.2969 * np.sqrt(x_xfoil / c) - 0.1260 * (x_xfoil / c) - 0.3516 * (x_xfoil / c)**2 
                      + 0.2843 * (x_xfoil / c)**3 - 0.1036 * (x_xfoil / c)**4)

XC_xfoil,YC_xfoil,S_xfoil,phi_xfoil,delta_xfoil,beta_xfoil = GENERATE_GEOMETRY(len(x_xfoil)-1,x_xfoil,y_xfoil,AoAR)



#%% PLOT

print('CL = ', CL)
fig = plt.figure()
midIndS_xfoil = int(np.floor(len(cp_xfoil)/2))                                          # Airfoil middle index for XFOIL data
plt.plot(x_xfoil[midIndS_xfoil:len(x_xfoil)], cp_xfoil[midIndS_xfoil:len(x_xfoil)],
         color='r',label='XFOIL Lower')
plt.plot(x_xfoil[0:midIndS_xfoil], cp_xfoil[0:midIndS_xfoil],
         color='b',label='XFOIL Upper')

midIndS = int(np.floor(len(Cp)/2))                                          # Airfoil middle index for SPVP data
plt.plot(XC[midIndS+1:len(XC)],Cp[midIndS+1:len(XC)],                       # Plot Cp for upper surface of airfoil from panel method
            'ks',markerfacecolor='b',label='Panel Method Upper')
plt.plot(XC[0:midIndS],Cp[0:midIndS],                                       # Plot Cp for lower surface of airfoil from panel method
            'ks',markerfacecolor='r',label='Panel Method Lower')

plt.gca().invert_yaxis()  # Cp plots ont souvent l'axe Y inversé
plt.xlabel("x/c",fontsize = 14)
plt.ylabel("Cp",fontsize = 14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("SAVE_FIGURE/XFOIL_COMPARAISON/Cp_AoA4.pdf", format="pdf")
#plt.show()

AoA_sweep = np.linspace(0,10,11)
CL_list = np.array([2.2416e-15, 0.1260, 0.2521,0.3781,0.5040,0.6298,0.7553,0.8807, 1.0059, 1.13079, 1.2554])
CL_list_xfoil = np.array([0,0.1265, 0.2529, 0.3792, 0.5055, 0.6316, 0.7575, 0.8831, 1.0085, 1.1336, 1.2583])


fig = plt.figure()
plt.plot(AoA_sweep,CL_list,color='black',label='This study')
plt.plot(AoA_sweep,CL_list_xfoil,'ks',markerfacecolor='r',label='XFOIL')
plt.xlabel("α (°)",fontsize = 14)
plt.ylabel("CL",fontsize = 14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("SAVE_FIGURE/XFOIL_COMPARAISON/CL_AoA_Sweep.pdf", format="pdf")


fig = plt.figure()
plt.plot(AoA_sweep,CL_list-CL_list_xfoil,color='black',label='CL-CL_XFOIL')
plt.xlabel("α (°)",fontsize = 14)
plt.ylabel("CL",fontsize = 14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("SAVE_FIGURE/XFOIL_COMPARAISON/CL_Diff_AoA_Sweep.pdf", format="pdf")


plt.show()