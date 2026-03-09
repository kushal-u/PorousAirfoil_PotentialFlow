# FUNCTION - GENERATE AIRFOIL COORDINATES FOR NACA 4-DIGIT SERIES AIRFOILS

import numpy as np
import math as math
import matplotlib.pyplot as plt

from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY

def GENERATE_NACA4(NameAirfoil, c=1.0, NumPan=100, power=1.0):
    n = int(NumPan / 2 + 1)
    m = int(NameAirfoil[0])*0.01
    p = int(NameAirfoil[1])*0.1
    t = int(NameAirfoil[2:4])*0.01
    beta = np.linspace(0, np.pi, n)
    if NumPan%2==1:
        beta = np.linspace(-np.pi,np.pi,NumPan+1)
        beta = beta[n-1:]
    x_dist = (1 - np.cos(beta)) / 2
    x = (x_dist**power) * c
    #yt = np.sqrt((0.25-(x-0.5)*(x-0.5))/9)
    yt = 5 * t * c * (0.2969 * np.sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * (x / c)**2 
                      + 0.2843 * (x / c)**3 - 0.1036 * (x / c)**4)
    
    if m == 0 and p == 0:
        xu, yu = x, yt
        xl, yl = x, -yt
    else:
        xu,yu,xl,yl = NON_SYMETRIES(x,c,p,m,yt)
    
    return np.append(xl[::-1], xu[1:]), np.append(yl[::-1], yu[1:])

def NON_SYMETRIES(x,c,p,m,yt):
    yc = np.where(x < p * c, m * x / (p**2) * (2 * p - x / c),
                      m * (c - x) / ((1 - p)**2) * (1 + x / c - 2 * p))
    dyc_dx = np.where(x < p * c, 2 * m / (p**2) * (p - x / c),
                        2 * m / ((1 - p)**2) * (p - x / c))
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu,yu,xl,yl

def GENERATE_EQUAL_NACA4(NameAirfoil, c=1.0, NumPan=100, power=1.0):
    n = int(NumPan / 2 + 1)
    m = int(NameAirfoil[0])*0.01
    p = int(NameAirfoil[1])*0.1
    t = int(NameAirfoil[2:4])*0.01

    beta = np.linspace(0, np.pi, 10000)
    x_dist = (1 - np.cos(beta)) / 2
    x = (x_dist**power) * c
    yt = 5 * t * c * (0.2969 * np.sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * (x / c)**2 
                      + 0.2843 * (x / c)**3 - 0.1036 * (x / c)**4)
    
    if m == 0 and p == 0:
        xu, yu = x, yt
        xl, yl = x, -yt
    else:
        xu,yu,xl,yl = NON_SYMETRIES(x,c,p,m,yt)
    
    XB_10000 = np.append(xl[::-1], xu[1:])
    YB_10000 = np.append(yl[::-1], yu[1:])

    S_10000 = GENERATE_GEOMETRY(20000-2,XB_10000,YB_10000,0)[2]
    LENGTH = sum(S_10000)
    S_CUMULATIVE = np.zeros(20000-2)
    S_CUMULATIVE[0] = S_10000[0]
    for i in range(1,20000-2):
        S_CUMULATIVE[i] = S_CUMULATIVE[i-1]+S_10000[i]
    XB = np.ones(NumPan+1)
    YB = np.zeros(NumPan+1)

    for i in range(1,NumPan):
        length = i/(NumPan)*LENGTH
        index = np.where(S_CUMULATIVE >= length)[0]
        XB[i] = XB_10000[index[0]+1]
        YB[i] = YB_10000[index[0]+1]
    
    return XB,YB


    

