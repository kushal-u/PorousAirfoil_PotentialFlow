import numpy as np
import math as math
import matplotlib.pyplot as plt


from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY


def Pore_Geometry(XB_pore,YB_pore):
    numPan = len(XB_pore)-1
    S = np.zeros(numPan)
    phi = np.zeros(numPan)
    XC_pore = np.zeros(numPan)
    YC_pore = np.zeros(numPan)
    j = 0

    for i in range(numPan):                                                         # Loop over all panels
        XC_pore[i] = 0.5*(XB_pore[i]+XB_pore[i+1])
        YC_pore[i] = 0.5*(YB_pore[i]+YB_pore[i+1])
        dx      = XB_pore[i+1]-XB_pore[i]                                                     # Change in X between boundary points
        dy      = YB_pore[j+1]-YB_pore[j]                                                     # Change in Y between boundary points
        S[j]    = (dx**2 + dy**2)**0.5                                              # Length of the panel
        phi[j]  = math.atan2(dy,dx)                                                 # Angle of panel (positive X-axis to inside face)
        if (phi[j] < 0):                                                            # Make all panel angles positive [rad]
            phi[j] = phi[j] + 2*np.pi
        j+=1
    return S, phi, XC_pore, YC_pore
def Hydraulic_GEOMETRY(XC,YC,omega,a,centre_point):
    # Define the list of control points that are in the pores
    control_points_in_pores = []
    a_slope = math.tan(omega)
    if omega == 0:
        b_low = YC[centre_point] - a/2
        b_high = YC[centre_point] + a/2
    else:
        b_low = YC[centre_point]- math.tan(omega)*(XC[centre_point]+a/2/math.sin(omega))
        b_high = YC[centre_point] - math.tan(omega)*(XC[centre_point]-a/2/math.sin(omega))
    if b_low > b_high:
        b_low, b_high = b_high, b_low
    for i in range(len(XC)):
        if YC[i] >= a_slope*XC[i]+b_low and YC[i] <= a_slope*XC[i]+b_high:
            control_points_in_pores.append(i)
    return filter_circulaire(control_points_in_pores,centre_point,len(XC))

def filter_circulaire(liste, A,numPan):
    """
    Retourne tous les éléments connectés à A via des pas de +1 ou -1 circulaires dans la liste.
    """
    liste_set = set(liste)
    visited = set()
    à_visiter = [A]

    min_val = 0
    max_val = numPan-1

    while à_visiter:
        courant = à_visiter.pop()
        if courant in liste_set and courant not in visited:
            visited.add(courant)
            
            voisin_plus = courant + 1 if courant < max_val else min_val
            voisin_moins = courant - 1 if courant > min_val else max_val

            if voisin_plus in liste_set:
                à_visiter.append(voisin_plus)
            if voisin_moins in liste_set:
                à_visiter.append(voisin_moins)
    
    return sorted(list(visited), key=liste.index)

def pressure_succion_side(numPan,pore_entry,pore_exit):
    if any(x>numPan/2 for x in pore_exit):
        pore_exit_succion_side = [x for x in pore_exit if x > numPan/2]
        if any(x<numPan/2 for x in pore_exit):
            pore_exit_pressure_side = [x for x in pore_exit if x < numPan/2]
            high_point1 = list(range(numPan))
            low_point1 = list(range(numPan))
            high_point = high_point1[max(pore_entry)+1:min(pore_exit_succion_side)]
            low_point = low_point1[max(pore_exit_pressure_side)+1:min(pore_entry)]
            print('1')
        else:
            high_point1 = list(range(numPan))
            low_point1 = list(range(numPan))
            high_point = high_point1[max(pore_entry)+1:min(pore_exit)]
            low_point = low_point1[0:min(pore_entry)]
            if (max(pore_exit)<numPan-1):
                low_point.extend(low_point1[max(pore_exit)+1:numPan])
            print('2')
    else:
        pore_exit_pressure_side = [x for x in pore_exit if x < numPan/2]
        high_point1 = list(range(numPan))
        low_point1 = list(range(numPan))
        high_point = high_point1[max(pore_entry)+1:numPan]
        low_point = low_point1[max(pore_exit_pressure_side)+1:min(pore_entry)]
        if (min(pore_entry)>0):
                high_point.extend(high_point1[0:min(pore_exit)])
    
    return low_point, high_point

def segments_traverses(X, Y, y0, angle_deg):
    traverses = []
    if angle_deg ==90 or angle_deg == -90:
        for i in range(len(X) - 1):
            x1, x2 = X[i], X[i+1]
            # On vérifie si la droite verticale x = A est entre x1 et x2
            if (x1 - y0) * (x2 - y0) <= 0 and x1 != x2:
                traverses.append(i)
        vertical = 1
    else:
        # Convertir l'angle en radians et calculer le vecteur direction de la droite
        angle_rad = np.radians(angle_deg)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # Deux points sur la droite pour créer un segment de test
        P1 = np.array([0, y0])
        P2 = P1 + 100*np.array([dx, dy])  # point plus loin sur la droite
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        for i in range(len(X) - 1):
            A = np.array([X[i], Y[i]])
            B = np.array([X[i+1], Y[i+1]])
            if intersect(P1, P2, A, B):
                traverses.append(i)
        vertical =0
    print(traverses)
    return traverses,vertical

def Refine_GEOMETRY(XB,YB,NameAirfoil, y0,angle_pore,AoAR,n_refinement=10):
    m = int(NameAirfoil[0])*0.01
    p = int(NameAirfoil[1])*0.1
    t = int(NameAirfoil[2:4])*0.01
    c = 1
    seg_edge_pore, vertical = segments_traverses(XB,YB,y0,angle_deg=angle_pore)
    if not vertical:
        if XB[seg_edge_pore[0]]>XB[seg_edge_pore[1]]:
            entry_point = seg_edge_pore[1]
            out_point = seg_edge_pore[0]
        else:
            entry_point = seg_edge_pore[0]
            out_point = seg_edge_pore[1]
        if entry_point < 0 or entry_point >= len(XB)-1:
            raise ValueError("i doit être un indice interne (1 <= i <= len(X)-2)")
    
    else:
        if YB[seg_edge_pore[0]]>YB[seg_edge_pore[1]]:
            entry_point = seg_edge_pore[1]
            out_point = seg_edge_pore[0]
        else:
            entry_point = seg_edge_pore[0]
            out_point = seg_edge_pore[1]
        if entry_point < 0 or entry_point >= len(XB)-1:
            raise ValueError("i doit être un indice interne (1 <= i <= len(X)-2)")

    x_prev_in = XB[entry_point-1]
    x_curr_in = XB[entry_point+2]

    x_prev_out = XB[out_point-1]
    x_curr_out = XB[out_point+2]

    # Points de transition
    trans_in = np.linspace(x_prev_in, x_curr_in, n_refinement + 2)[1:-1]
    trans_out = np.linspace(x_prev_out, x_curr_out, n_refinement + 2)[1:-1]
    if entry_point < out_point:
        XB = np.concatenate([XB[:entry_point],trans_in, XB[entry_point+2:out_point],trans_out,XB[out_point+2:]])
        entry_point += (n_refinement-1)/2
        out_point +=n_refinement+(n_refinement-1)/2
    elif out_point < entry_point:
        XB = np.concatenate([XB[:out_point],trans_out,XB[out_point+2:entry_point],trans_in,XB[entry_point+2:]])
        out_point += (n_refinement-1)/2
        entry_point +=n_refinement+(n_refinement-1)/2
    
    for i in range(len(XB)):
        if XB[i] < XB[i+1]:
            lim = i
            break
    
    #YB_DOWN = -np.sqrt((0.25-(XB[:lim+1]-0.5)*(XB[:lim+1]-0.5))/9)
    YB_DOWN = -5 * t * c * (0.2969 * np.sqrt(XB[:lim+1] / c) - 0.1260 * (XB[:lim+1] / c) - 0.3516 * (XB[:lim+1] / c)**2 
                      + 0.2843 * (XB[:lim+1] / c)**3 - 0.1036 * (XB[:lim+1] / c)**4)
    #YB_UP = np.sqrt((0.25-(XB[lim+1:]-0.5)*(XB[lim+1:]-0.5))/9)
    YB_UP = 5 * t * c * (0.2969 * np.sqrt(XB[lim+1:] / c) - 0.1260 * (XB[lim+1:] / c) - 0.3516 * (XB[lim+1:] / c)**2 
                      + 0.2843 * (XB[lim+1:] / c)**3 - 0.1036 * (XB[lim+1:] / c)**4)
    YB = np.concatenate([YB_DOWN,YB_UP])
    numPan = len(XB)-1
    XC,YC,S,phi,delta,beta = GENERATE_GEOMETRY(numPan,XB,YB,AoAR)
    return XB,YB,XC,YC,S,phi,delta,beta,int(entry_point),int(out_point)


def Refine_GEOMETRY_ELLIPSE(XB,YB, y0,angle_pore,AoAR,n_refinement=10):
    c = 1
    seg_edge_pore, vertical = segments_traverses(XB,YB,y0,angle_deg=angle_pore)
    if not vertical:
        if XB[seg_edge_pore[0]]>XB[seg_edge_pore[1]]:
            entry_point = seg_edge_pore[1]
            out_point = seg_edge_pore[0]
        else:
            entry_point = seg_edge_pore[0]
            out_point = seg_edge_pore[1]
        if entry_point < 0 or entry_point >= len(XB)-1:
            raise ValueError("i doit être un indice interne (1 <= i <= len(X)-2)")
    
    else:
        if YB[seg_edge_pore[0]]>YB[seg_edge_pore[1]]:
            entry_point = seg_edge_pore[1]
            out_point = seg_edge_pore[0]
        else:
            entry_point = seg_edge_pore[0]
            out_point = seg_edge_pore[1]
        if entry_point < 0 or entry_point >= len(XB)-1:
            raise ValueError("i doit être un indice interne (1 <= i <= len(X)-2)")

    x_prev_in = XB[entry_point-1]
    x_curr_in = XB[entry_point+2]

    x_prev_out = XB[out_point-1]
    x_curr_out = XB[out_point+2]

    # Points de transition
    trans_in = np.linspace(x_prev_in, x_curr_in, n_refinement + 2)[1:-1]
    trans_out = np.linspace(x_prev_out, x_curr_out, n_refinement + 2)[1:-1]
    if entry_point < out_point:
        XB = np.concatenate([XB[:entry_point],trans_in, XB[entry_point+2:out_point],trans_out,XB[out_point+2:]])
        entry_point += (n_refinement-1)/2
        out_point +=n_refinement+(n_refinement-1)/2
    elif out_point < entry_point:
        XB = np.concatenate([XB[:out_point],trans_out,XB[out_point+2:entry_point],trans_in,XB[entry_point+2:]])
        out_point += (n_refinement-1)/2
        entry_point +=n_refinement+(n_refinement-1)/2
    
    for i in range(len(XB)):
        if XB[i] < XB[i+1]:
            lim = i
            break
    
    YB_DOWN = -np.sqrt((0.25-(XB[:lim+1]-0.5)*(XB[:lim+1]-0.5))/9)
    YB_UP = np.sqrt((0.25-(XB[lim+1:]-0.5)*(XB[lim+1:]-0.5))/9)
    YB = np.concatenate([YB_DOWN,YB_UP])
    numPan = len(XB)-1
    XC,YC,S,phi,delta,beta = GENERATE_GEOMETRY(numPan,XB,YB,AoAR)
    return XB,YB,XC,YC,S,phi,delta,beta,int(entry_point),int(out_point)

     
def plot_line(a, b, x_min=0, x_max=1):
        """
        Trace la droite y = a*x + b entre x_min et x_max

        Paramètres :
        - a : pente de la droite
        - b : ordonnée à l'origine
        - x_min, x_max : bornes de l'axe x
        """
        x = np.linspace(x_min, x_max, 100)  # 100 points entre x_min et x_max
        y = a * x + b  # équation de la droite
        print('Im here')

        plt.plot(x, y, label=f'y = {a}x + {b}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Représentation d\'une droite')
        plt.legend()
        plt.grid(True)
