import numpy as np
import math as math
import matplotlib.pyplot as plt

def COMPUTE_A_SPVP(numPan,I,K,J,L):
    # Populate A matrix
    A = np.zeros([numPan,numPan])                                                   # Initialize the A matrix
    for i in range(numPan):                                                         # Loop over all i panels
        for j in range(numPan):                                                     # Loop over all j panels
            if (i == j):                                                            # If the panels are the same
                A[i,j] = np.pi                                                      # Set A equal to pi
            else:                                                                   # If panels are not the same
                A[i,j] = I[i,j]                                                     # Set A equal to I

    # Right column of A matrix
    newAV = np.zeros((numPan,1))                                                    # Used to enlarge the A matrix to account for gamma column
    A     = np.hstack((A,newAV))                                                    # Horizontally stack the A matrix with newAV to get enlarged matrix
    for i in range(numPan):                                                         # Loop over all i panels (rows)
        A[i,numPan] = -sum(K[i,:])                                                  # Add gamma term to right-most column of A matrix

    # Bottom row of A matrix
    newAH = np.zeros((1,numPan+1))                                                  # Used to enlarge the A matrix to account for Kutta condition equation
    A     = np.vstack((A,newAH))                                                    # Vertically stack the A matrix with newAH to get enlarged matrix
    for j in range(numPan):                                                         # Loop over all j panels (columns)
        A[numPan,j] = J[0,j] + J[numPan-1,j]                                        # Source contribution of Kutta condition equation
    A[numPan,numPan] = -(sum(L[0,:] + L[numPan-1,:])) + 2*np.pi                     # Vortex contribution of Kutta condition equation 
    return A

def COMPUTE_b_SPVP(Airfoil_geometry,Fluid_characteristics,Delta_Cp,Pore_characteristics={},is_porous = False):
    numPan = Airfoil_geometry['numPan']
    delta = Airfoil_geometry['delta']
    beta = Airfoil_geometry['beta']

    Vinf = Fluid_characteristics['Vinf']
    rhoinf = Fluid_characteristics['rhoinf']
    mu = Fluid_characteristics['mu']
    if is_porous:

        Rs = Pore_characteristics['Rs']
        a = Pore_characteristics['a']
        n = Pore_characteristics['n']
        pore_entry = Pore_characteristics['pore_entry']
        pore_out = Pore_characteristics['pore_exit']
        omega_in = Pore_characteristics['omega_in']
        omega_out = Pore_characteristics['omega_out']
        Dh = Pore_characteristics['Dh']
        L = Pore_characteristics['L']
    


    b = np.zeros(numPan)                                                            # Initialize the b array
    if is_porous:
        V_mean = 0.5*rhoinf*Vinf**2*Delta_Cp/Rs*n/a
        Re_laminar = Dh*rhoinf*abs(V_mean)/mu
        if Re_laminar >= 2000:                                                      #Turbulence
            Delta_P = Delta_Cp*0.5*rhoinf*Vinf**2
            V_mean=2.868*((Delta_P**4)*(Dh**5)/((L**4)*mu*(rhoinf**3)))**(1/7)
            Re_turbulent = Dh*rhoinf*abs(V_mean)/mu
            print("TURBULENT.    RE =",Re_turbulent)
        else: 
            print("LAMINAR.    RE =", Re_laminar)
    for i in range(numPan):                                                         # Loop over all i panels (rows)
        if is_porous:
            #normal_vs_pore_in = delta[i]-np.pi*(omega_in)/180
            #normal_vs_pore_out = delta[i]-np.pi*(omega_out)/180
            if i in pore_entry:
                b[i] = -Vinf*2*np.pi*np.cos(beta[i]) + V_mean#*np.cos(normal_vs_pore_in) # Compute RHS array
            elif i in pore_out:
                b[i] = -Vinf*2*np.pi*np.cos(beta[i]) + V_mean#*np.cos(normal_vs_pore_out)
            else:
                b[i] = -Vinf*2*np.pi*np.cos(beta[i])        
        else:
            b[i] = -Vinf*2*np.pi*np.cos(beta[i])

    

    # Last element of b array (Kutta condition)
    b = np.append(b,-Vinf*2*np.pi*(np.sin(beta[0]) + np.sin(beta[numPan-1])))       # Add Kutta condition equation RHS to b array
    return b

def COMPUTE_IJ_SPM(XC,YC,XB,YB,phi,S):
    
    # Number of panels
    numPan = len(XC)                                                                # Number of panels/control points
    
    # Initialize arrays
    I = np.zeros([numPan,numPan])                                                   # Initialize I integral matrix
    J = np.zeros([numPan,numPan])                                                   # Initialize J integral matrix
    
    # Compute integral
    for i in range(numPan):                                                         # Loop over i panels
        for j in range(numPan):                                                     # Loop over j panels
            if (j != i):                                                            # If the i and j panels are not the same
                # Compute intermediate values
                A  = -(XC[i]-XB[j])*np.cos(phi[j])-(YC[i]-YB[j])*np.sin(phi[j])     # A term
                B  = (XC[i]-XB[j])**2 + (YC[i]-YB[j])**2                            # B term
                Cn = np.sin(phi[i]-phi[j])                                          # C term (normal)
                Dn = -(XC[i]-XB[j])*np.sin(phi[i])+(YC[i]-YB[j])*np.cos(phi[i])     # D term (normal)
                Ct = -np.cos(phi[i]-phi[j])                                         # C term (tangential)
                Dt = (XC[i]-XB[j])*np.cos(phi[i])+(YC[i]-YB[j])*np.sin(phi[i])      # D term (tangential)
                E  = np.sqrt(B-A**2)                                                # E term
                if (E == 0 or np.iscomplex(E) or np.isnan(E) or np.isinf(E)):       # If E term is 0 or complex or a NAN or an INF
                    I[i,j] = 0                                                      # Set I value equal to zero
                    J[i,j] = 0                                                      # Set J value equal to zero
                else:
                    # Compute I (needed for normal velocity), Ref [1]
                    term1  = 0.5*Cn*np.log((S[j]**2 + 2*A*S[j] + B)/B)              # First term in I equation
                    term2  = ((Dn-A*Cn)/E)*(math.atan2((S[j]+A),E)-math.atan2(A,E)) # Second term in I equation
                    I[i,j] = term1 + term2                                          # Compute I integral
                    
                    # Compute J (needed for tangential velocity), Ref [2]
                    term1  = 0.5*Ct*np.log((S[j]**2 + 2*A*S[j] + B)/B)              # First term in I equation
                    term2  = ((Dt-A*Ct)/E)*(math.atan2((S[j]+A),E)-math.atan2(A,E)) # Second term in I equation
                    J[i,j] = term1 + term2                                          # Compute J integral
                
            # Zero out any problem values
            if (np.iscomplex(I[i,j]) or np.isnan(I[i,j]) or np.isinf(I[i,j])):      # If I term is complex or a NAN or an INF
                I[i,j] = 0                                                          # Set I value equal to zero
            if (np.iscomplex(J[i,j]) or np.isnan(J[i,j]) or np.isinf(J[i,j])):      # If J term is complex or a NAN or an INF
                J[i,j] = 0                                                          # Set J value equal to zero
    
    return I, J                                                                     # Return both I and J matrices

def COMPUTE_KL_VPM(XC,YC,XB,YB,phi,S):
    
    # Number of panels
    numPan = len(XC)                                                                # Number of panels
    
    # Initialize arrays
    K = np.zeros([numPan,numPan])                                                   # Initialize K integral matrix
    L = np.zeros([numPan,numPan])                                                   # Initialize L integral matrix
    
    # Compute integral
    for i in range(numPan):                                                         # Loop over i panels
        for j in range(numPan):                                                     # Loop over j panels
            if (j != i):                                                            # If panel j is not the same as panel i
                # Compute intermediate values
                A  = -(XC[i]-XB[j])*np.cos(phi[j])-(YC[i]-YB[j])*np.sin(phi[j])     # A term
                B  = (XC[i]-XB[j])**2 + (YC[i]-YB[j])**2                            # B term
                Cn = -np.cos(phi[i]-phi[j])                                         # C term (normal)
                Dn = (XC[i]-XB[j])*np.cos(phi[i])+(YC[i]-YB[j])*np.sin(phi[i])      # D term (normal)
                Ct = np.sin(phi[j]-phi[i])                                          # C term (tangential)
                Dt = (XC[i]-XB[j])*np.sin(phi[i])-(YC[i]-YB[j])*np.cos(phi[i])      # D term (tangential)
                E  = np.sqrt(B-A**2)                                                # E term
                if (E == 0 or np.iscomplex(E) or np.isnan(E) or np.isinf(E)):       # If E term is 0 or complex or a NAN or an INF
                    K[i,j] = 0                                                      # Set K value equal to zero
                    L[i,j] = 0                                                      # Set L value equal to zero
                else:
                    # Compute K
                    term1  = 0.5*Cn*np.log((S[j]**2 + 2*A*S[j] + B)/B)              # First term in K equation
                    term2  = ((Dn-A*Cn)/E)*(math.atan2((S[j]+A),E)-math.atan2(A,E)) # Second term in K equation
                    K[i,j] = term1 + term2                                          # Compute K integral
                    
                    # Compute L
                    term1  = 0.5*Ct*np.log((S[j]**2 + 2*A*S[j] + B)/B)              # First term in L equation
                    term2  = ((Dt-A*Ct)/E)*(math.atan2((S[j]+A),E)-math.atan2(A,E)) # Second term in L equation
                    L[i,j] = term1 + term2                                          # Compute L integral
            
            # Zero out any problem values
            if (np.iscomplex(K[i,j]) or np.isnan(K[i,j]) or np.isinf(K[i,j])):      # If K term is complex or a NAN or an INF
                K[i,j] = 0                                                          # Set K value equal to zero
            if (np.iscomplex(L[i,j]) or np.isnan(L[i,j]) or np.isinf(L[i,j])):      # If L term is complex or a NAN or an INF
                L[i,j] = 0                                                          # Set L value equal to zero
    
    return K, L                                                                     # Return both K and L matrices

def COMPUTE_LIFT_MOMENT(Cp,Fluid_characteristics,Airfoil_geometry,Pore_characteristics, Delta_Cp = 0,Cp_inter_low=0,Cp_inter_high=0,A=0,is_porous = 0):
    S = Airfoil_geometry['S']
    beta = Airfoil_geometry['beta']
    phi = Airfoil_geometry['phi']
    XC = Airfoil_geometry['XC']
    YC = Airfoil_geometry['YC']
    if not is_porous:
        CL = sum(-Cp*S*np.sin(beta))
        CD = sum(-Cp*S*np.cos(beta))

        CM = sum(Cp*(XC-0.25)*S*np.cos(phi) + Cp*YC*S*np.sin(phi))                           # Moment coefficient []
    # Compute normal and axial force coefficients
    else:
        low_point = Pore_characteristics['low_point']
        high_point = Pore_characteristics['high_point']
        S_pore_low = Pore_characteristics['S_pore_low']
        S_pore_high = Pore_characteristics['S_pore_high']
        phi_pore_low = Pore_characteristics['phi_pore_low']
        phi_pore_high = Pore_characteristics['phi_pore_high']
        AoAR = Fluid_characteristics['AoAR']

        CL = sum(-Cp[i]*S[i]*np.sin(beta[i]) for i in low_point)                                    # Normal force coefficient []
        CL += sum(-Cp[i]*S[i]*np.sin(beta[i]) for i in high_point)
        """CL += sum(-Cp_inter_low*S_pore_low*np.sin(phi_pore_low+(np.pi/2)-AoAR))
        CL += sum(-Cp_inter_high*S_pore_high*np.sin(phi_pore_high+(np.pi/2)-AoAR))"""
        CL += -2*Delta_Cp*A*np.sin(phi_pore_high[0])

        CD = sum(-Cp[i]*S[i]*np.cos(beta[i]) for i in low_point)
        #print('1 : ',CD)
        CD += sum(-Cp[i]*S[i]*np.cos(beta[i]) for i in high_point)
        #print('2 : ',CD)
        """CD += sum(-Cp_inter_low*S_pore_low*np.cos(phi_pore_low+(np.pi/2)-AoAR))
        print('3 : ',CD)
        print("Cp_inter_low  : ",Cp_inter_low)
        print("cos : ",np.cos(phi_pore_low+(np.pi/2)-AoAR))
        CD += sum(-Cp_inter_high*S_pore_high*np.cos(phi_pore_high+(np.pi/2)-AoAR))
        print('4 : ',CD)
        print("Cp_inter_high  : ",Cp_inter_high)
        print("cos : ",np.cos(phi_pore_high+(np.pi/2)-AoAR))"""
        CD += -2*Delta_Cp*A*np.cos(phi_pore_high[0])
        #print('5 : ',CD)

        
        Y = np.zeros(len(XC))
        for i in high_point:
            Y[i] = -Cp[i]*S[i]*np.cos(beta[i]) 
        for i in low_point:
            Y[i] = -Cp[i]*S[i]*np.cos(beta[i])


        CM = sum(Cp[i]*(XC[i]-0.25)*S[i]*np.cos(phi[i]) + Cp[i]*YC[i]*S[i]*np.sin(phi[i]) for i in low_point)           
        CM += sum(Cp[i]*(XC[i]-0.25)*S[i]*np.cos(phi[i]) + Cp[i]*YC[i]*S[i]*np.sin(phi[i]) for i in high_point)                 

    return CL,CM,CD

def COMPUTE_SOLUTION_SPVP(A,b):
    # Compute result array
    resArr = np.linalg.solve(A,b)                                                   # Solve system of equation for all source strengths and single vortex strength

    # Separate lam and gamma values from result 
    lam   = resArr[0:len(resArr)-1]                                                 # All panel source strengths
    gamma = resArr[len(resArr)-1]                                                   # Constant vortex strength
    return lam, gamma

def COMPUTE_Vt_Cp(Airfoil_geometry,Fluid_characteristics,Delta_Cp,gamma,lam,b,J,L,Pore_characteristics={},is_porous = False):
    numPan = Airfoil_geometry['numPan']
    delta = Airfoil_geometry['delta']
    beta = Airfoil_geometry['beta']

    Vinf = Fluid_characteristics['Vinf']
    rhoinf = Fluid_characteristics['rhoinf']
    mu = Fluid_characteristics['mu']
    if is_porous:

        Rs = Pore_characteristics['Rs']
        a = Pore_characteristics['a']
        n = Pore_characteristics['n']
        pore_entry = Pore_characteristics['pore_entry']
        pore_out = Pore_characteristics['pore_exit']
        omega_in = Pore_characteristics['omega_in']
        omega_out = Pore_characteristics['omega_out']
        Dh = Pore_characteristics['Dh']
        Length = Pore_characteristics['L']
    # Compute velocities
    Vt = np.zeros(numPan)                                                           # Initialize tangential velocity
    Vn = np.zeros(numPan)
    V = np.zeros(numPan) 
    Cp = np.zeros(numPan)                                                           # Initialize pressure coefficient
    for i in range(numPan):                                                         # Loop over all panels
        if is_porous:
            normal_vs_pore_in = delta[i]-np.pi*(omega_in)/180
            normal_vs_pore_out = delta[i]-np.pi*(omega_out)/180
        term1 = Vinf*np.sin(beta[i])                                                # Uniform flow term
        term2 = (1/(2*np.pi))*sum(lam*J[i,:])                                       # Source panel terms when j is not equal to i
        term3 = gamma/2                                                             # Vortex panel term when j is equal to i
        term4 = -(gamma/(2*np.pi))*sum(L[i,:])                                      # Vortex panel terms when j is not equal to i
        if is_porous:
            V_mean = 0.5*rhoinf*Vinf**2*Delta_Cp/Rs*n/a
            Re_laminar = Dh*rhoinf*abs(V_mean)/mu
            if Re_laminar >= 2000 and is_porous:                                                      #Turbulence 
                Delta_P = Delta_Cp*0.5*rhoinf*Vinf**2
                V_mean=2.868*((Delta_P**4)*(Dh**5)/((Length**4)*mu*(rhoinf**3)))**(1/7)
            if i in pore_entry:
                term5 = 0#V_mean*np.sin(normal_vs_pore_in) 
            elif i in pore_out:
                term5 = 0#V_mean*np.sin(normal_vs_pore_out)
            else:
                term5 = 0
        else:
            term5 = 0
        Vt[i] = term1 + term2 + term3 + term4 + term5                                      # Compute tangential velocity on panel i
        Vn[i] = b[i] + Vinf*2*np.pi*np.cos(beta[i])
        V[i] =math.sqrt(Vt[i]**2+Vn[i]**2)
        Cp[i] = 1-(Vt[i]/Vinf)**2                                                   # Compute pressure coefficient on panel i
    """print('V_mean. = ',V_mean)
    print('V = ',V[pore_out[int(len(pore_out)/2)]])
    print('Vn = ',Vn[pore_out[int(len(pore_out)/2)]])
    print('Vt = ',Vt[pore_out[int(len(pore_out)/2)]])
    print('V_mean*sin() = ',V_mean*np.sin(delta[pore_out[int(len(pore_out)/2)]]-np.pi*(omega_in)/180) )
    print(omega_out)
    print(delta[pore_out[int(len(pore_out)/2)]]*180/np.pi)"""

    return Vt, Cp

def STREAMLINE_SPM(XP,YP,XB,YB,phi,S):
    
    # Number of panels
    numPan = len(XB)-1                                                          # Number of panels
    
    # Initialize arrays
    Mx = np.zeros(numPan)                                                       # Initialize Ix integral array
    My = np.zeros(numPan)                                                       # Initialize Iy integral array
    
    # Compute integral
    for j in range(numPan):                                                     # Loop over all panels
        # Compute intermediate values
        A = -(XP-XB[j])*np.cos(phi[j]) - (YP-YB[j])*np.sin(phi[j])              # A term
        B  = (XP-XB[j])**2 + (YP-YB[j])**2;                                     # B term
        Cx = -np.cos(phi[j]);                                                   # C term (X-direction)
        Dx = XP - XB[j];                                                        # D term (X-direction)
        Cy = -np.sin(phi[j]);                                                   # C term (Y-direction)
        Dy = YP - YB[j];                                                        # D term (Y-direction)
        E  = math.sqrt(B-A**2);                                                 # E term
        if (E == 0 or np.iscomplex(E) or np.isnan(E) or np.isinf(E)):           # If E term is 0 or complex or a NAN or an INF
            Mx[j] = 0                                                           # Set Mx value equal to zero
            My[j] = 0                                                           # Set My value equal to zero
        else:
            # Compute Mx, Ref [1]
            term1 = 0.5*Cx*np.log((S[j]**2 + 2*A*S[j]+B)/B);                    # First term in Mx equation
            term2 = ((Dx-A*Cx)/E)*(math.atan2((S[j]+A),E) - math.atan2(A,E));   # Second term in Mx equation
            Mx[j] = term1 + term2;                                              # Compute Mx integral
            
            # Compute My, Ref [1]
            term1 = 0.5*Cy*np.log((S[j]**2 + 2*A*S[j]+B)/B);                    # First term in My equation
            term2 = ((Dy-A*Cy)/E)*(math.atan2((S[j]+A),E) - math.atan2(A,E));   # Second term in My equation
            My[j] = term1 + term2;                                              # Compute My integral

        # Zero out any problem values
        if (np.iscomplex(Mx[j]) or np.isnan(Mx[j]) or np.isinf(Mx[j])):         # If Mx term is complex or a NAN or an INF
            Mx[j] = 0                                                           # Set Mx value equal to zero
        if (np.iscomplex(My[j]) or np.isnan(My[j]) or np.isinf(My[j])):         # If My term is complex or a NAN or an INF
            My[j] = 0                                                           # Set My value equal to zero
    
    return Mx, My                                                               # Return both Mx and My matrices

def STREAMLINE_VPM(XP,YP,XB,YB,phi,S):
    
    # Number of panels
    numPan = len(XB)-1                                                          # Number of panels (control points)
    
    # Initialize arrays
    Nx = np.zeros(numPan)                                                       # Initialize Nx integral array
    Ny = np.zeros(numPan)                                                       # Initialize Ny integral array
    
    # Compute Nx and Ny
    for j in range(numPan):                                                     # Loop over all panels
        # Compute intermediate values
        A = -(XP-XB[j])*np.cos(phi[j]) - (YP-YB[j])*np.sin(phi[j])              # A term
        B  = (XP-XB[j])**2 + (YP-YB[j])**2                                      # B term
        Cx = np.sin(phi[j])                                                     # Cx term (X-direction)
        Dx = -(YP-YB[j])                                                        # Dx term (X-direction)
        Cy = -np.cos(phi[j])                                                    # Cy term (Y-direction)
        Dy = XP-XB[j]                                                           # Dy term (Y-direction)
        E  = math.sqrt(B-A**2)                                                  # E term
        if (E == 0 or np.iscomplex(E) or np.isnan(E) or np.isinf(E)):           # If E term is 0 or complex or a NAN or an INF
            Nx[j] = 0                                                           # Set Nx value equal to zero
            Ny[j] = 0                                                           # Set Ny value equal to zero
        else:
            # Compute Nx, Ref [1]
            term1 = 0.5*Cx*np.log((S[j]**2 + 2*A*S[j]+B)/B);                    # First term in Nx equation
            term2 = ((Dx-A*Cx)/E)*(math.atan2((S[j]+A),E) - math.atan2(A,E));   # Second term in Nx equation
            Nx[j] = term1 + term2;                                              # Compute Nx integral
            
            # Compute Ny, Ref [1]
            term1 = 0.5*Cy*np.log((S[j]**2 + 2*A*S[j]+B)/B);                    # First term in Ny equation
            term2 = ((Dy-A*Cy)/E)*(math.atan2((S[j]+A),E) - math.atan2(A,E));   # Second term in Ny equation
            Ny[j] = term1 + term2;                                              # Compute Ny integral
            
        # Zero out any problem values
        if (np.iscomplex(Nx[j]) or np.isnan(Nx[j]) or np.isinf(Nx[j])):         # If Nx term is complex or a NAN or an INF
            Nx[j] = 0                                                           # Set Nx value equal to zero
        if (np.iscomplex(Ny[j]) or np.isnan(Ny[j]) or np.isinf(Ny[j])):         # If Ny term is complex or a NAN or an INF
            Ny[j] = 0                                                           # Set Ny value equal to zero
    
    return Nx, Ny                                                               # Return both Nx and Ny matrices
