# FUNCTION - GENERATE VARIOUS PLOTS FOR SPVP METHOD INCLUDING GEOMETRY, PRESSURE COEFFICIENTS, STREAMLINES, AND CONTOURS

import numpy as np
import math as math
import matplotlib.pyplot as plt
from matplotlib import path
from scipy.interpolate import interp1d
import os

# Chemin complet du fichier
path_file = "SAVE_FIGURE/RESULT/Cp.pdf"
# Créer le dossier si nécessaire
os.makedirs(os.path.dirname(path_file), exist_ok=True)


from COMPUTATION.COMPUTE import STREAMLINE_SPM
from COMPUTATION.COMPUTE import STREAMLINE_VPM


def PLOT_NORMAL_VECTOR(XB,YB,numPan,XC,YC,S,delta,alone=1):
    fig = plt.figure()                                                         # Create the figure
    plt.cla()                                                                   # Clear the axes
    plt.fill(XB,YB,'k')                                                         # Plot the airfoil
    X = np.zeros(2)                                                             # Initialize 'X'
    Y = np.zeros(2)                                                             # Initialize 'Y'
    for i in range(numPan):                                                     # Loop over all panels
        X[0] = XC[i]                                                            # Set X start of panel orientation vector
        X[1] = XC[i] + S[i]*np.cos(delta[i])                                    # Set X end of panel orientation vector
        Y[0] = YC[i]                                                            # Set Y start of panel orientation vector
        Y[1] = YC[i] + S[i]*np.sin(delta[i])                                    # Set Y end of panel orientation vector
        if (i == 0):                                                            # If it's the first panel index
            plt.plot(X,Y,'b-',label='First Panel')                              # Plot normal vector for first panel
        elif (i == 1):                                                          # If it's the second panel index
            plt.plot(X,Y,'g-',label='Second Panel')                             # Plot normal vector for second panel
        else:                                                                   # If it's neither the first nor second panel index
            plt.plot(X,Y,'r-')                                                  # Plot normal vector for all other panels
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.title('Panel Geometry')                                                 # Set title
    plt.axis('equal')                                                           # Set axes equal
    plt.legend()                                                                # Display legend
    if alone:                                                          # If alone is 1
        plt.show()

def PLOT_GEOMETRY(XB,YB,XC,YC,alone=1):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    plt.plot(XB,YB,'k-')                                                        # Plot airfoil panels
    plt.plot([XB[0], XB[1]],[YB[0], YB[1]],'b-',label='First Panel')            # Plot first panel
    plt.plot([XB[1], XB[2]],[YB[1], YB[2]],'g-',label='Second Panel')           # Plot second panel
    plt.plot(XB,YB,'ko',markerfacecolor='k',label='Boundary Pts')               # Plot boundary points (black circles)
    plt.plot(XC,YC,'ko',markerfacecolor='r',label='Control Pts')                # Plot control points (red circles)
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.axis('equal')                                                           # Set axes equal
    plt.legend()                                                                # Display legend
    if alone:                                                          # If alone is 1
        plt.show()

def PLOT_CP_AIRFOIL(Cp,XC,YC,delta,XB,YB,alone=1):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    Cps = np.absolute(Cp*0.15)                                                  # Scale and make positive all Cp values
    X = np.zeros(2)                                                             # Initialize X values
    Y = np.zeros(2)                                                             # Initialize Y values
    for i in range(len(Cps)):                                                   # Loop over all panels
        X[0] = XC[i]                                                            # Control point X-coordinate
        X[1] = XC[i] + Cps[i]*np.cos(delta[i])                                  # Ending X-value based on Cp magnitude
        Y[0] = YC[i]                                                            # Control point Y-coordinate
        Y[1] = YC[i] + Cps[i]*np.sin(delta[i])                                  # Ending Y-value based on Cp magnitude
        
        if (Cp[i] < 0):                                                         # If pressure coefficient is negative
            plt.plot(X,Y,'r-')                                                  # Plot as a red line
        elif (Cp[i] >= 0):                                                      # If pressure coefficient is zero or positive
            plt.plot(X,Y,'b-')                                                  # Plot as a blue line
    plt.fill(XB,YB,'k')                                                         # Plot the airfoil as black polygon
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.gca().set_aspect('equal')                                               # Set aspect ratio equal
    if alone:                                                          # If alone is 1
        plt.show()

def PLOT_CP(XB,XC,Cp,alone=1):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    for i in range(len(XB)):
        if XB[i+1]>XB[i]:
            midIndS = i
            break
    plt.plot(XC[midIndS+1:len(XC)],Cp[midIndS+1:len(XC)],                       # Plot Cp for upper surface of airfoil from panel method
                'ks',markerfacecolor='b',label='VPM Upper')
    plt.plot(XC[0:midIndS+1],Cp[0:midIndS+1],                                   # Plot Cp for lower surface of airfoil from panel method
                'ks',markerfacecolor='r',label='VPM Lower')
    plt.xlim(0,1)                                                               # Set X-limits
    plt.xlabel('X Coordinate')                                                  # Set X-label
    plt.ylabel('Cp')                                                            # Set Y-label
    plt.title('Pressure Coefficient')                                           # Set title
                                                                        # Display plot
    plt.legend()                                                                # Display legend
    plt.gca().invert_yaxis()                                                    # Invert Cp (Y) axis
    if alone:                                                          # If alone is 1
        plt.show()

def PLOT_CP_COMPARISON(XB,XC,Cp1,Cp2,pore_entry,pore_exit,label1,label2,alone=1):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    XC_porous = []
    Cp_porous = []
    for i in range(len(XC)):
        if i not in pore_entry and i not in pore_exit:
            XC_porous.append(XC[i])
            Cp_porous.append(Cp1[i])
    for i in range(len(XB)):
        if XB[i+1]>XB[i]:
            midIndS = i
            break
    for i in range(len(XC_porous)-1):
        if XC_porous[i+1]>XC_porous[i]:
            midIndS_porous = i
            break
    """plt.plot(XC_porous[midIndS_porous+1:len(XC_porous)],Cp_porous[midIndS_porous+1:len(XC_porous)],                       # Plot Cp for upper surface of porous airfoil
                color='r',label=label1+' Upper porous')
    plt.plot(XC_porous[0:midIndS_porous+1],Cp_porous[0:midIndS_porous+1],                                   # Plot Cp for lower surface of porous airfoil
                color = 'g',label=label1+' Lower porous')
    plt.plot(XC[midIndS+1:len(XC)],Cp2[midIndS+1:len(XC)],                       # Plot Cp for upper surface of solid airfoil
                color='b',label=label2+' Upper')
    plt.plot(XC[0:midIndS+1],Cp2[0:midIndS+1],                                   # Plot Cp for lower surface of solid airfoil
                color='r',label=label2+' Lower')"""
    plt.plot(XC_porous,Cp_porous,                       # Plot Cp for upper surface of porous airfoil
                color='r',label=label1+' Porous')
    plt.plot(XC,Cp2,                       # Plot Cp for upper surface of solid airfoil
                color='black',label=label2+' Solid')
    plt.xlim(-0.01,1.01)                                                              # Set X-limits
    plt.ylim(-2,1.1)
    plt.grid(True)
    plt.xlabel('X Coordinate')                                                  # Set X-label
    plt.ylabel('Cp')                                                            # Set Y-label
    plt.title('Pressure Coefficient')                                           # Set title
                                                                        # Display plot
    plt.legend()                                                                # Display legend
    plt.gca().invert_yaxis()                                                    # Invert Cp (Y) axis
    if alone:                                                          # If alone is 1
        plt.show()

def PLOT_STREAMLINE(XX,YY,Vx,Vy,XYsl,XB,YB,xVals,yVals,alone=1):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    np.seterr(under="ignore")                                                   # Ignore underflow error message
    plt.streamplot(XX,YY,Vx,Vy, linewidth=0.5, density=40, color='r',           # Plot streamlines
                    arrowstyle='-', start_points=XYsl)
    plt.clim(vmin=0, vmax=2)
    plt.fill(XB,YB,'k')                                                         # Plot airfoil as black polygon
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.gca().set_aspect('equal')                                               # Set axes equal
    plt.xlim(xVals)                                                             # Set X-limits
    plt.ylim(yVals)                                                             # Set Y-limits
    if alone:                                                          # If alone is 1
        plt.show()

def PLOT_PRESSURE(XX,YY,CpXY,XB,YB,xVals,yVals,alone=1):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting
    plt.contourf(XX,YY,CpXY,500,cmap='jet')                                     # Plot contour
    plt.fill(XB,YB,'k')                                                         # Plot airfoil as black polygon
    plt.xlabel('X Units')                                                       # Set X-label
    plt.ylabel('Y Units')                                                       # Set Y-label
    plt.gca().set_aspect('equal')                                               # Set axes equal
    plt.xlim(xVals)                                                             # Set X-limits
    plt.ylim(yVals)                                                             # Set Y-limits
    if alone:                                                          # If alone is 1
        plt.show()

def GRID_CALCULATION(XB,YB,phi,S,Vinf,AoAR,lam,gamma):
    # Grid parameters
    nGridX = 100                                                                # X-grid for streamlines and contours
    nGridY = 100                                                                # Y-grid for streamlines and contours
    xVals  = [min(XB)-0.5, max(XB)+0.5]                                         # X-grid extents [min, max]
    yVals  = [min(YB)-0.3, max(YB)+0.3]                                         # Y-grid extents [min, max]
    
    # Streamline parameters
    slPct  = 25                                                                 # Percentage of streamlines of the grid
    Ysl    = np.linspace(yVals[0],yVals[1],int((slPct/100)*nGridY))             # Create array of Y streamline starting points
    Xsl    = xVals[0]*np.ones(len(Ysl))                                         # Create array of X streamline starting points
    XYsl   = np.vstack((Xsl.T,Ysl.T)).T                                         # Concatenate X and Y streamline starting points
    
    # Generate the grid points
    Xgrid  = np.linspace(xVals[0],xVals[1],nGridX)                              # X-values in evenly spaced grid
    Ygrid  = np.linspace(yVals[0],yVals[1],nGridY)                              # Y-values in evenly spaced grid
    XX, YY = np.meshgrid(Xgrid,Ygrid)                                           # Create meshgrid from X and Y grid arrays
    
    # Initialize velocities
    Vx     = np.zeros([nGridX,nGridY])                                          # Initialize X velocity matrix
    Vy     = np.zeros([nGridX,nGridY])                                          # Initialize Y velocity matrix
    
    # Path to figure out if grid point is inside polygon or not
    AF     = np.vstack((XB.T,YB.T)).T                                           # Concatenate XB and YB geometry points
    afPath = path.Path(AF)                                                      # Create a path for the geometry
    
    # Solve for grid point X and Y velocities
    for m in range(nGridX):                                                     # Loop over X-grid points
        print("m: %i" % m)
        for n in range(nGridY):                                                 # Loop over Y-grid points
            XP     = XX[m,n]                                                    # Current iteration's X grid point
            YP     = YY[m,n]                                                    # Current iteration's Y grid point
            Mx, My = STREAMLINE_SPM(XP,YP,XB,YB,phi,S)                          # Compute streamline Mx and My values
            Nx, Ny = STREAMLINE_VPM(XP,YP,XB,YB,phi,S)                          # Compute streamline Nx and Ny values
            
            # Check if grid points are in object
            # - If they are, assign a velocity of zero
            if afPath.contains_points([(XP,YP)]):                               # If (XP,YP) is in the body
                Vx[m,n] = 0                                                     # Set X-velocity equal to zero
                Vy[m,n] = 0                                                     # Set Y-velocity equal to zero
            else:
                Vx[m,n] = (Vinf*np.cos(AoAR) + sum(lam*Mx/(2*np.pi))            # Compute X-velocity
                                            + sum(-gamma*Nx/(2*np.pi)))
                Vy[m,n] = (Vinf*np.sin(AoAR) + sum(lam*My/(2*np.pi))            # Compute Y-velocity
                                            + sum(-gamma*Ny/(2*np.pi)))
    
    # Compute grid point velocity magnitude and pressure coefficient
    Vxy  = np.sqrt(Vx**2 + Vy**2)                                               # Compute magnitude of velocity vector []
    CpXY = 1 - (Vxy/Vinf)**2                                                    # Pressure coefficient []
    return XX,YY,XYsl,Vx,Vy,xVals,yVals,CpXY

def PLOT_AIRFOIL(XB, YB, low_point, high_point, alone=True):
    
    """
    Trace l'airfoil en remplissant la partie supérieure et inférieure.

    XB, YB : listes des coordonnées des points de contour.
    low_point : indices des points de la partie inférieure.
    high_point : indices des points de la partie supérieure.
    alone : si True, crée une nouvelle figure ; sinon, ajoute au plot existant.
    """
    if alone:
        plt.figure(figsize=(6, 3))

    # Coordonnées partie inférieure
    XB_low = [XB[i] for i in low_point]
    YB_low = [YB[i] for i in low_point]

    # Coordonnées partie supérieure
    XB_high = [XB[i] for i in high_point]
    YB_high = [YB[i] for i in high_point]

    # Remplissage avec plt.fill
    plt.fill(XB_high, YB_high, color='skyblue', label='Suction side')
    plt.fill(XB_low, YB_low, color='lightcoral', label='Pressure side')

    plt.axis('equal')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    if alone:
        plt.show()

def PLOT_CP_PRESSURE_SIDE(XC,YC, Cp_extern, Cp_intern, low_point, X_intern, alone = True):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting

    XC_low = np.array([XC[i] for i in low_point])
    YC_low = np.array([YC[i] for i in low_point])
    Cp_low = np.array([Cp_extern[i] for i in low_point])
    Cp_extern = np.array(np.array)

    x_common = np.linspace(max(min(XC_low), min(X_intern)), min(max(XC_low), max(X_intern)), 200)
    # Interpolation
    f1 = interp1d(XC_low, Cp_low, kind='cubic')
    f2 = interp1d(X_intern, Cp_intern, kind='cubic')

    y1_interp = f1(x_common)
    y2_interp = f2(x_common)

    plt.plot(XC_low,Cp_low,color = 'red',label = "Pressure Side")
    plt.plot(X_intern,Cp_intern, color = 'black', label = "Lower Pore Wall")

    plt.fill_between(x_common,y1_interp,y2_interp,where=(y1_interp>y2_interp), interpolate=False, color = 'green', alpha=0.5)
    plt.fill_between(x_common,y1_interp,y2_interp,where=(y1_interp<=y2_interp), interpolate=False, color = 'red', alpha=0.5)
    plt.xlim(-0.01,1.01)
    plt.ylim(-2,1.1)                                                               # Set X-limits
    plt.xlabel('X Coordinate')                                                  # Set X-label
    plt.ylabel('Cp')                                                            # Set Y-label
    plt.title('Pressure Coefficient')                                           # Set title
                                                                        # Display plot
    plt.legend()                                                                # Display legend
    plt.grid(True)
    plt.gca().invert_yaxis()                                                    # Invert Cp (Y) axis
    if alone:                                                          # If alone is 1
        plt.show()

def PLOT_CP_SUCCION_SIDE(XC,YC, Cp_extern, Cp_intern, high_point, X_intern, alone = True):
    fig = plt.figure()                                                         # Create figure
    plt.cla()                                                                   # Get ready for plotting

    XC_low = np.array([XC[i] for i in high_point])
    YC_low = np.array([YC[i] for i in high_point])

    Cp_low = np.array([Cp_extern[i] for i in high_point])
    Cp_extern = np.array(np.array)

    x_common = np.linspace(max(min(XC_low), min(X_intern)), min(max(XC_low), max(X_intern)), 200)
    # Interpolation
    f1 = interp1d(XC_low, Cp_low, kind='linear')
    f2 = interp1d(X_intern, Cp_intern, kind='linear')

    y1_interp = f1(x_common)
    y2_interp = f2(x_common)

    plt.plot(XC_low,Cp_low,color = 'black',label = "Suction Side")
    plt.plot(X_intern,Cp_intern, color = 'red', label = "Higher Pore Wall")

    plt.fill_between(x_common,y1_interp,y2_interp,where=(y1_interp>y2_interp), interpolate=True, color = 'red', alpha=0.5)
    plt.fill_between(x_common,y1_interp,y2_interp,where=(y1_interp<=y2_interp), interpolate=True, color = 'green', alpha=0.5)
    plt.xlim(-0.01,1.01)
    plt.ylim(-2,1.1)                                                               # Set X-limits
    plt.xlabel('X Coordinate')                                                  # Set X-label
    plt.ylabel('Cp')                                                            # Set Y-label
    plt.title('Pressure Coefficient')                                           # Set title
                                                                        # Display plot
    plt.legend()                                                                # Display legend
    plt.grid(True)
    plt.gca().invert_yaxis()                                                    # Invert Cp (Y) axis
    if alone:                                                          # If alone is 1
        plt.show()

def plot_convergence(h_list, CL_matrix, CD_matrix, CL_extrapolated, CD_extrapolated, aoa_values):
    num_aoa = len(aoa_values)

    for i in range(num_aoa):
        aoa = aoa_values[i]
        print(i)
        print(num_aoa)

        # Original data
        h_vals = np.array(h_list)
        cl_vals = CL_matrix[i, :]
        cd_vals = CD_matrix[i, :]

        # --- CL ---
        plt.figure()
        plt.plot(h_vals, cl_vals, 'o-', label='CL(h)', color='blue')
        # Extrapolated point
        plt.plot(0, CL_extrapolated[i], 'o', color='red', label='Extrapolated')
        # Dashed line to the last point
        plt.plot([h_vals[-1], 0], [cl_vals[-1], CL_extrapolated[i]], 'r--')
        plt.xlabel('h (panel size ~ 1/NumPan)')
        plt.ylabel('CL')
        plt.title(f'CL Convergence for AoA = {aoa}°')
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- CD ---
        plt.figure()
        plt.plot(h_vals, cd_vals, 'o-', label='CD(h)', color='blue')
        # Extrapolated point
        plt.plot(0, CD_extrapolated[i], 'o', color='red', label='Extrapolated')
        # Dashed line to the last point
        plt.plot([h_vals[-1], 0], [cd_vals[-1], CD_extrapolated[i]], 'r--')
        plt.xlabel('h (panel size ~ 1/NumPan)')
        plt.ylabel('CD')
        plt.title(f'CD Convergence for AoA = {aoa}°')
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_extrapolated_vs_aoa(aoa_values, CL_extrapolated1, CD_extrapolated1,  CL_extrapolated2, CD_extrapolated2,  CL_extrapolated3, CD_extrapolated3):
    plt.rcParams["text.usetex"] = True
    
    # --- CL vs AoA ---
    plt.figure()
    plt.plot(aoa_values, CL_extrapolated1, 'o-', color='green', label='Solid')
    plt.plot(aoa_values, CL_extrapolated2, 'o-', color='black', label='Horizontal')
    plt.plot(aoa_values, CL_extrapolated3, 'o-', color='red', label='Vertical')
    plt.xlabel('Angle of Attack (°)',fontsize = 14)
    plt.ylabel(r" $C_{L_{Extrapolated}}$",fontsize = 18)
    plt.tick_params(axis='both', labelsize=14)
    #plt.title('Extrapolated CL as a Function of AoA',fontsize = 16)
    plt.grid(True)
    #plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("SAVE_FIGURE/RESULT_ELLIPSE/CL_vs_AoA.pdf", format="pdf")

    # --- CD vs AoA ---
    plt.figure()
    plt.plot(aoa_values, CD_extrapolated1, 'o-', color='green', label='Solid')
    plt.plot(aoa_values, CD_extrapolated2, 'o-', color='black', label='Diagonal')
    plt.plot(aoa_values, CD_extrapolated3, 'o-', color='red', label='Vertical')
    plt.xlabel('Angle of Attack (°)',fontsize = 14)
    plt.ylabel(r" $C_{D_{Extrapolated}}$",fontsize = 18)
    plt.tick_params(axis='both', labelsize=14)
    #plt.title('Extrapolated CD as a Function of AoA',fontsize = 16)
    plt.grid(False)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("SAVE_FIGURE/RESULT_ELLIPSE/CD_vs_AoA.pdf", format="pdf")

    # --- CL-CL_SOLID vs AoA ---
    plt.figure()
    plt.plot(aoa_values, (CL_extrapolated2 - CL_extrapolated1), 'o-', color='black', label='Horizontal')
    plt.plot(aoa_values, (CL_extrapolated3 - CL_extrapolated1), 'o-', color='red', label='Vertical')
    plt.xlabel('Angle of Attack (°)',fontsize = 14)
    plt.ylabel(r"$\Delta C_L$",fontsize = 18)
    plt.tick_params(axis='both', labelsize=14)
    #plt.title('Difference in Extrapolated CL Compared to CL_SOLID',fontsize = 16)
    plt.grid(True)
    #plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("SAVE_FIGURE/RESULT_ELLIPSE/CL_Diff_vs_AoA.pdf", format="pdf")

    # --- Relative Improvement vs AoA ---
    plt.figure()
    aoa_values = np.delete(aoa_values,5)
    CL_extrapolated1 = np.delete(CL_extrapolated1,5)
    CL_extrapolated2 = np.delete(CL_extrapolated2,5)
    CL_extrapolated3 = np.delete(CL_extrapolated3,5)
    print(aoa_values)
    plt.plot(aoa_values, (CL_extrapolated2 - CL_extrapolated1)/abs(CL_extrapolated1), 'o-', color='black', label='Horizontal')
    plt.plot(aoa_values, (CL_extrapolated3 - CL_extrapolated1)/abs(CL_extrapolated2), 'o-', color='red', label='Vertical')
    plt.xlabel("Angle of Attack (°)",fontsize = 14)
    plt.ylabel(r"$\Delta_r C_L$",fontsize = 18)
    plt.tick_params(axis='both', labelsize=14)
    #plt.title('Relative improvement of CL Compared to CL_SOLID',fontsize = 16)
    plt.grid(True)
    #plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig("SAVE_FIGURE/RESULT_ELLIPSE/CL_Improvement_vs_AoA.pdf", format="pdf")
    plt.show()

    
def PLOT_ALL(flagPlot,XB,YB,numPan,XC,YC,S,delta,Cp,phi,Vinf,AoA,lam,gamma):
    AoAR = AoA*(np.pi/180)  
    if (flagPlot[4] == 1 or flagPlot[5] == 1):                                      # If we are plotting streamlines or pressure coefficient contours
        XX,YY,XYsl,Vx,Vy,xVals,yVals,CpXY = GRID_CALCULATION(XB,YB,phi,S,Vinf,AoAR,lam,gamma)
    if (flagPlot[0] == 1):
        PLOT_NORMAL_VECTOR(XB,YB,numPan,XC,YC,S,delta,alone=0)

    # FIGURE: Geometry with the following indicated:
    # - Boundary points, control points, first panel, second panel
    if (flagPlot[1] == 1):
        PLOT_GEOMETRY(XB,YB,XC,YC,alone=0)

    # FIGURE: Cp vectors at airfoil control points
    if (flagPlot[2] == 1):
        PLOT_CP_AIRFOIL(Cp,XC,YC,delta,XB,YB,alone=0)

    # FIGURE: Pressure coefficient
    if (flagPlot[3] == 1):
        PLOT_CP(XB,XC,Cp,alone=0)

    # FIGURE: Airfoil streamlines
    if (flagPlot[4] == 1):
        PLOT_STREAMLINE(XX,YY,Vx,Vy,XYsl,XB,YB,xVals,yVals,alone=0)

    # FIGURE: Pressure coefficient contour
    if (flagPlot[5] == 1):
        PLOT_PRESSURE(XX,YY,CpXY,XB,YB,xVals,yVals,alone=0)

    if (flagPlot != [0,0,0,0,0,0]):
        plt.show()                                                                  # Display plots