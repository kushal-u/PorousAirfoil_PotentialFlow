# FUNCTION - COMPUTE AIRFOIL GEOMETRY PARAMETERS FOR SOURCE PANEL METHOD

import numpy as np
import math as math

def GENERATE_GEOMETRY(numPan,XB,YB,AoAR):
    # Initialize variables
    XC  = np.zeros(numPan)                                                          # Initialize control point X-coordinate array
    YC  = np.zeros(numPan)                                                          # Initialize control point Y-coordinate array
    S   = np.zeros(numPan)                                                          # Initialize panel length array
    phi = np.zeros(numPan)                                                          # Initialize panel orientation angle array [deg]

    # Find geometric quantities of the airfoil
    for i in range(numPan):                                                         # Loop over all panels
        XC[i]   = 0.5*(XB[i]+XB[i+1])                                               # X-value of control point
        YC[i]   = 0.5*(YB[i]+YB[i+1])                                               # Y-value of control point
        dx      = XB[i+1]-XB[i]                                                     # Change in X between boundary points
        dy      = YB[i+1]-YB[i]                                                     # Change in Y between boundary points
        S[i]    = (dx**2 + dy**2)**0.5                                              # Length of the panel
        phi[i]  = math.atan2(dy,dx)                                                 # Angle of panel (positive X-axis to inside face)
        if (phi[i] < 0):                                                            # Make all panel angles positive [rad]
            phi[i] = phi[i] + 2*np.pi

    # Compute angle of panel normal w.r.t. horizontal and include AoA
    delta                = phi + (np.pi/2)                                          # Angle from positive X-axis to outward normal vector [rad]
    beta                 = delta - AoAR                                             # Angle between freestream vector and outward normal vector [rad]
    beta[beta > 2*np.pi] = beta[beta > 2*np.pi] - 2*np.pi                           # Make all panel angles between 0 and 2pi [rad]

    return XC,YC,S,phi,delta,beta