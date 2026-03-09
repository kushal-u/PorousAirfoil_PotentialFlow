#%% IMPORT
import numpy as np
import math as math
import matplotlib.pyplot as plt
import pandas as pd

from COMPUTATION.Porous_SPVP import POROUS_AIRFOIL, POROUS_ELLIPSE
from COMPUTATION.SPVP_Airfoil import SOLID_AIRFOIL
from COMPUTATION.SPVP_Ellipse import SOLID_ELLIPSE
from GEOMETRY.NACA import GENERATE_NACA4
from GEOMETRY.GEOMETRY import GENERATE_GEOMETRY
from PLOT import PLOT_AIRFOIL




#%% MAIN FUNCTION
if __name__ == "__main__":
    #%% Parameters for all cases
    AoA_sweep = np.linspace(-5,5,11)
    NumPan_sweep = np.array([250,500,1000])
    Vinf = 1                                                                        # Freestream velocity [] (just leave this at 1)
    rhoinf = 1                                                                      # Density [] (just leave this at 1)
    Re = 100000                                                                      # Reynolds number
    power = 1
    NameAirfoil = "0018"


    #Convergence Variables
    max_iter = 100
    tol = 1e-8


    #%% Case 1 : Solid NACA 0018
    XB,YB = GENERATE_NACA4(NameAirfoil,NumPan=10000)
    S = GENERATE_GEOMETRY(10000,XB,YB,AoAR=0)[2]
    print("Length =", sum(S))
    CL_list_1 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    CD_list_1 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    j = 0
    for numPan in NumPan_sweep:
        i = 0
        for AoA in AoA_sweep:
            Fluid_characteristics,Airfoil_geometry,Cp,lam,gamma,CL,CM,CD = SOLID_AIRFOIL(NameAirfoil,numPan,power,AoA,Vinf,rhoinf,Re)
            print('Case 1 :NumPan : ', numPan, '     AoA : ', AoA, '   Finished! \n')
            CL_list_1[i,j] = CL
            CD_list_1[i,j] = CD
            i+=1
        j+=1
    
    """CL_1 = np.zeros([len(AoA_sweep),len(NumPan_sweep)+1])
    CD_1 = np.zeros([len(AoA_sweep),len(NumPan_sweep)+1])
    for i in range(len(AoA_sweep)):
        F_CL1,h_CL1 = RICHARDSON_EXTRAPOLATION(NumPan_list=NumPan_sweep,Result_list=CL_list_1[i])[:2]
        F_CD1,h_CD1 = RICHARDSON_EXTRAPOLATION(NumPan_list=NumPan_sweep,Result_list=CD_list_1[i])[:2]
        CL_1[i] = F_CL1
        CD_1[i] = F_CD1
    

    CSV_SAVE(CL_lists=[CL_1],CD_lists=[CD_1],file_name='Case_1',AoA_sweep=AoA_sweep,h=h_CL1)"""

    # Liste de tes paires CL/CD
    CL_lists = [CL_list_1]
    CD_lists = [CD_list_1]                               

    # Initialisation du tableau final
    data = []

    # Parcours de tous les cas
    for case_num, (CL_mat, CD_mat) in enumerate(zip(CL_lists, CD_lists), start=1):
        for i, aoa in enumerate(AoA_sweep):
            for j, num_pan in enumerate(NumPan_sweep):
                cl = CL_mat[i, j]
                cd = CD_mat[i, j]
                data.append([case_num, aoa, num_pan, cl, cd])

    # Création du DataFrame
    df = pd.DataFrame(data, columns=['Case', 'AoA', 'NumPan', 'CL', 'CD'])

    # Sauvegarde en CSV
    df.to_csv("resultats_simulations_Airfoil1.csv", index=False)
    

    #%% Case 2 : Porous NACA 0018 with horizontal pores
    #Pore geometry
    type = 'rectangle'
    pore_geometry = [0.157,0.014]
    L = 0.89
    a = 0.014                       #Height of the pores
    n = 1/0.166
    y0 = -0.01
    angle_pore = 0

    CL_list_2 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    CD_list_2 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    j = 0
    for numPan in NumPan_sweep:
        i = 0
        for AoA in AoA_sweep:
            Fluid_characteristics,Pore_characteristics,Airfoil_geometry,Cp,Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_AIRFOIL(numPan,AoA,Vinf,rhoinf,Re,L,pore_geometry,type,NameAirfoil,y0,angle_pore,a,n,power,tol,max_iter)
            print('Case 2 :NumPan : ', numPan, '     AoA : ', AoA, '   Finished! \n')
            CL_list_2[i,j] = CL
            CD_list_2[i,j] = CD
            i+=1
        j+=1

    # Liste de tes paires CL/CD
    CL_lists = [CL_list_2]
    CD_lists = [CD_list_2]                               

    # Initialisation du tableau final
    data = []

    # Parcours de tous les cas
    for case_num, (CL_mat, CD_mat) in enumerate(zip(CL_lists, CD_lists), start=1):
        for i, aoa in enumerate(AoA_sweep):
            for j, num_pan in enumerate(NumPan_sweep):
                cl = CL_mat[i, j]
                cd = CD_mat[i, j]
                data.append([case_num, aoa, num_pan, cl, cd])

    # Création du DataFrame
    df = pd.DataFrame(data, columns=['Case', 'AoA', 'NumPan', 'CL', 'CD'])

    # Sauvegarde en CSV
    df.to_csv("resultats_simulations_Airfoil2.csv", index=False)
    

    #%% Case 3 : Porous NACA 0018 with vertical pores

    #Pore geometry
    type = 'rectangle'
    pore_geometry = [0.157,0.014]
    L = 0.89
    a = 0.014                       #Height of the pores
    n = 1/0.166
    y0 = 0.2
    angle_pore = 90

    CL_list_3 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    CD_list_3 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    j = 0
    for numPan in NumPan_sweep:
        i = 0
        for AoA in AoA_sweep:
            Fluid_characteristics,Pore_characteristics,Airfoil_geometry,Cp,Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_AIRFOIL(numPan,AoA,Vinf,rhoinf,Re,L,pore_geometry,type,NameAirfoil,y0,angle_pore,a,n,power,tol,max_iter)
            print('Case 3 :NumPan : ', numPan, '     AoA : ', AoA, '   Finished! \n')
            CL_list_3[i,j] = CL
            CD_list_3[i,j] = CD
            i+=1
        j+=1

    # Liste de tes paires CL/CD
    CL_lists = [CL_list_3]
    CD_lists = [CD_list_3]                               

    # Initialisation du tableau final
    data = []

    # Parcours de tous les cas
    for case_num, (CL_mat, CD_mat) in enumerate(zip(CL_lists, CD_lists), start=1):
        for i, aoa in enumerate(AoA_sweep):
            for j, num_pan in enumerate(NumPan_sweep):
                cl = CL_mat[i, j]
                cd = CD_mat[i, j]
                data.append([case_num, aoa, num_pan, cl, cd])

    # Création du DataFrame
    df = pd.DataFrame(data, columns=['Case', 'AoA', 'NumPan', 'CL', 'CD'])

    # Sauvegarde en CSV
    df.to_csv("resultats_simulations_Airfoil3.csv", index=False)
    
    
    #%% Case 4 : Solid Ellipse
    CL_list_4 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    CD_list_4 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    j = 0
    for numPan in NumPan_sweep:
        i = 0
        for AoA in AoA_sweep:
            Fluid_characteristics,Airfoil_geometry,Cp,lam,gamma,CL,CM,CD = SOLID_ELLIPSE(numPan,power,AoA,Vinf,rhoinf,Re)
            print('Case 4 :NumPan : ', numPan, '     AoA : ', AoA, '   Finished! \n')
            CL_list_4[i,j] = CL
            CD_list_4[i,j] = CD
            i+=1
        j+=1
    # Liste de tes paires CL/CD
    CL_lists = [CL_list_4]
    CD_lists = [CD_list_4]                               

    # Initialisation du tableau final
    data = []

    # Parcours de tous les cas
    for case_num, (CL_mat, CD_mat) in enumerate(zip(CL_lists, CD_lists), start=1):
        for i, aoa in enumerate(AoA_sweep):
            for j, num_pan in enumerate(NumPan_sweep):
                cl = CL_mat[i, j]
                cd = CD_mat[i, j]
                data.append([case_num, aoa, num_pan, cl, cd])

    # Création du DataFrame
    df = pd.DataFrame(data, columns=['Case', 'AoA', 'NumPan', 'CL', 'CD'])

    # Sauvegarde en CSV
    df.to_csv("resultats_simulations_Airfoil4.csv", index=False)
    
    #%% Case 5 : Porous ellipse with horizontal pores
    #Pore geometry
    type = 'rectangle'
    pore_geometry = [0.157,0.0125]
    L = 0.89
    a = 0.0125                       #Height of the pores
    n = 1/0.166
    y0 = -0.01
    angle_pore = -10

    CL_list_5 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    CD_list_5 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    j = 0
    for numPan in NumPan_sweep:
        i = 0
        for AoA in AoA_sweep:
            Fluid_characteristics,Pore_characteristics,Airfoil_geometry,Cp,Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_ELLIPSE(numPan,AoA,Vinf,rhoinf,Re,L,pore_geometry,type,y0,angle_pore,a,n,power,tol,max_iter)
            print('Case 5 :NumPan : ', numPan, '     AoA : ', AoA, '   Finished! \n')
            CL_list_5[i,j] = CL
            CD_list_5[i,j] = CD
            i+=1
            XB = Airfoil_geometry['XB']
            YB = Airfoil_geometry['YB']
            low_point = Pore_characteristics['low_point']
            high_point = Pore_characteristics['high_point'] 
            PLOT_AIRFOIL(XB,YB,low_point,high_point,alone=1)
        j+=1
    
    # Liste de tes paires CL/CD
    CL_lists = [CL_list_5]
    CD_lists = [CD_list_5]                               

    # Initialisation du tableau final
    data = []

    # Parcours de tous les cas
    for case_num, (CL_mat, CD_mat) in enumerate(zip(CL_lists, CD_lists), start=1):
        for i, aoa in enumerate(AoA_sweep):
            for j, num_pan in enumerate(NumPan_sweep):
                cl = CL_mat[i, j]
                cd = CD_mat[i, j]
                data.append([case_num, aoa, num_pan, cl, cd])

    # Création du DataFrame
    df = pd.DataFrame(data, columns=['Case', 'AoA', 'NumPan', 'CL', 'CD'])

    # Sauvegarde en CSV
    df.to_csv("resultats_simulations_Airfoil5.csv", index=False)

    #%% Case 6 : Porous ellipse  with vertical pores

    #Pore geometry
    type = 'rectangle'
    pore_geometry = [0.157,0.0125]
    L = 0.89
    a = 0.0125                       #Height of the pores
    n = 1/0.166
    y0 = 0.2
    angle_pore = 90

    CL_list_6 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    CD_list_6 = np.zeros([len(AoA_sweep),len(NumPan_sweep)])
    j = 0
    for numPan in NumPan_sweep:
        i = 0
        for AoA in AoA_sweep:
            Fluid_characteristics,Pore_characteristics,Airfoil_geometry,Cp,Cp_Solid, Cp_inter_low,Cp_inter_high, CL, CL_Solid, CD, CD_Solid,lam,gamma = POROUS_ELLIPSE(numPan,AoA,Vinf,rhoinf,Re,L,pore_geometry,type,y0,angle_pore,a,n,power,tol,max_iter)
            print('Case 6 :NumPan : ', numPan, '     AoA : ', AoA, '   Finished! \n')
            CL_list_6[i,j] = CL
            CD_list_6[i,j] = CD
            i+=1
            XB = Airfoil_geometry['XB']
            YB = Airfoil_geometry['YB']
            low_point = Pore_characteristics['low_point']
            high_point = Pore_characteristics['high_point'] 
            PLOT_AIRFOIL(XB,YB,low_point,high_point,alone=1)
        j+=1

    """plt.plot(AoA_sweep,CL_list_1[:, 1],marker = 'd',c='r',label = 'Airfoil Solid')
    plt.plot(AoA_sweep,CL_list_2[:, 1],marker = 'd',c='black',label = 'Airfoil Porous H')
    plt.plot(AoA_sweep,CL_list_3[:, 1],marker = 'd',c='green',label = 'Airfoil Porous V')
    plt.plot(AoA_sweep,CL_list_4[:, 1],marker = '.',c='r',label = 'ELLIPSE Solid')
    plt.plot(AoA_sweep,CL_list_5[:, 1],marker = '.',c='black',label = 'ELLIPSE Porous H')
    plt.plot(AoA_sweep,CL_list_6[:, 1],marker = '.',c='green',label = 'ELLIPSE Porous V')
    plt.legend()
    plt.grid(True)
    plt.show()"""

    # Liste de tes paires CL/CD
    CL_lists = [CL_list_6]
    CD_lists = [CD_list_6]                               

    # Initialisation du tableau final
    data = []

    # Parcours de tous les cas
    for case_num, (CL_mat, CD_mat) in enumerate(zip(CL_lists, CD_lists), start=1):
        for i, aoa in enumerate(AoA_sweep):
            for j, num_pan in enumerate(NumPan_sweep):
                cl = CL_mat[i, j]
                cd = CD_mat[i, j]
                data.append([case_num, aoa, num_pan, cl, cd])

    # Création du DataFrame
    df = pd.DataFrame(data, columns=['Case', 'AoA', 'NumPan', 'CL', 'CD'])

    # Sauvegarde en CSV
    df.to_csv("resultats_simulations_ellipse6.csv", index=False)
# %%
