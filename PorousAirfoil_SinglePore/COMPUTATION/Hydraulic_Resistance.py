import numpy as np
import math as math

def Hydraulic_Resistance(mu,L,type='circular',geometry=[1]):
    # if type = 'circular' ==> geometry = [radius]
    # if type = 'rectangular' ==> geometry = [length,width]
    # if type = 'any' ==> geometry = [hydraulic diameter]
    if type == 'circular' and len(geometry) == 1:
        Dh = 2*geometry[0]
        R = 128*mu*L/np.pi/Dh**4
        A = geometry[0]**2*math.pi
    elif type == 'rectangle' and len(geometry) == 2:
        w=geometry[0]
        h=geometry[1]
        Dh = 4*w*h/(2*(w+h))
        R = 12*mu*L/((w*h**3)*(1-0.63*(h/w)))
        A = h*w
    elif type == 'any' and len(geometry) == 1:
        Dh = geometry[0]
        R = 128*mu*L/np.pi/Dh**4
        A = math.pi*Dh**2/4
    else:
        print('Error: geometry is not defined correctly')
        print('type = ',type, '       geometry = ',geometry)
    return R,Dh,A
    
