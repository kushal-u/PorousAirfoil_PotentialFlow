# FUNCTION - CHECK AND ENSURE PANEL COORDINATES ARE ORIENTED CLOCKWISE

import numpy as np
import math as math

def PANEL_DIRECTIONS(numPan,XB,YB):
    # Check for direction of points
    edge = np.zeros(numPan)                                                         # Initialize edge value array
    for i in range(numPan):                                                         # Loop over all panels
        edge[i] = (XB[i+1]-XB[i])*(YB[i+1]+YB[i])                                   # Compute edge values

    sumEdge = np.sum(edge)                                                          # Sum all edge values

    # If panels are CCW, flip them (don't if CW)
    if (sumEdge < 0):                                                               # If panels are CCW
        print('IM HERE')
        XB[:] = np.flipud(XB)                                                          # Flip the X-data array
        YB[:] = np.flipud(YB)                                                          # Flip the Y-data array