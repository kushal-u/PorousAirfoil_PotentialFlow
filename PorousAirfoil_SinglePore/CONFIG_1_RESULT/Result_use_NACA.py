import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from COMPUTATION.Richardson_extrapolation import extrapolate_matrices
from PLOT import plot_convergence, plot_extrapolated_vs_aoa

# Chemin complet du fichier
path = "SAVE_FIGURE/GRID_CONVERGENCE/GRID_CONVERGENCE.pdf"
# Créer le dossier si nécessaire
os.makedirs(os.path.dirname(path), exist_ok=True)


#%% Case 1
# Lecture du fichier CSV
df = pd.read_csv("CONFIG_1_RESULT/resultats_simulations_Airfoil1.csv")

# Tri des valeurs uniques de AoA et NumPan
aoa_values = sorted(df['AoA'].unique())
numpan_values = sorted(df['NumPan'].unique())

# Initialisation des matrices
CL_matrix = np.zeros((len(aoa_values), len(numpan_values)))
CD_matrix = np.zeros((len(aoa_values), len(numpan_values)))

# Remplissage des matrices
for i, aoa in enumerate(aoa_values):
    for j, numpan in enumerate(numpan_values):
        subset = df[(df['AoA'] == aoa) & (df['NumPan'] == numpan)]
        if not subset.empty:
            CL_matrix[i, j] = subset['CL'].values[0]
            CD_matrix[i, j] = subset['CD'].values[0]
print('CL_matrix1 = ', CL_matrix)
CL_matrix_extrapolated1, CD_matrix_extrapolated1,h,GCI21_l1,GCI32_l1,AR_l1,GCI21_d1,GCI32_d1,AR_d1, p_l1,p_d1 = extrapolate_matrices(CL_matrix,CD_matrix,numpan_values)
print('Case 1 :', CL_matrix_extrapolated1[7])
# h_list = 1 / NumPan_values
h_list = [max(numpan_values)*2.076 / n for n in numpan_values]

#plot_convergence(h_list, CL_matrix, CD_matrix, CL_matrix_extrapolated1, CD_matrix_extrapolated1, aoa_values)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
axes[0].plot(h_list,CL_matrix[7], 'o-', label='CL(h)', color='black')
axes[0].plot(0, CL_matrix_extrapolated1[7], 'o', color='red', label='Extrapolated')
axes[0].plot([h_list[-1], 0], [CL_matrix[7][-1], CL_matrix_extrapolated1[7]], 'r--')
axes[0].set_xlabel('1000*h',fontsize = 14)
axes[0].set_ylabel('CL',fontsize = 14)
axes[0].set_title('Solid',fontsize = 16)
axes[0].grid(True)
axes[0].legend()
axes[0].tick_params(axis='both', labelsize=12)
ymin, ymax = axes[0].get_ylim()
axes[0].set_ylim(ymin - 0.05*(ymax-ymin), ymax + 0.05*(ymax-ymin))




#%% Case 2
# Lecture du fichier CSV
df = pd.read_csv("CONFIG_1_RESULT/resultats_simulations_Airfoil2.csv")

# Tri des valeurs uniques de AoA et NumPan
aoa_values = sorted(df['AoA'].unique())
numpan_values = sorted(df['NumPan'].unique())

# Initialisation des matrices
CL_matrix = np.zeros((len(aoa_values), len(numpan_values)))
CD_matrix = np.zeros((len(aoa_values), len(numpan_values)))

# Remplissage des matrices
for i, aoa in enumerate(aoa_values):
    for j, numpan in enumerate(numpan_values):
        subset = df[(df['AoA'] == aoa) & (df['NumPan'] == numpan)]
        if not subset.empty:
            CL_matrix[i, j] = subset['CL'].values[0]
            CD_matrix[i, j] = subset['CD'].values[0]
print('CL_matrix2 = ', CL_matrix)
CL_matrix_extrapolated2, CD_matrix_extrapolated2,h,GCI21_l2,GCI32_l2,AR_l2,GCI21_d2,GCI32_d2,AR_d2, p_l2,p_d2 = extrapolate_matrices(CL_matrix,CD_matrix,numpan_values)
print('Case 2 :', CL_matrix_extrapolated2[7])

# h_list = 1 / NumPan_values
h_list = [max(numpan_values)*2.076 / n for n in numpan_values]

#plot_convergence(h_list, CL_matrix, CD_matrix, CL_matrix_extrapolated2, CD_matrix_extrapolated2, aoa_values)

axes[1].plot(h_list,CL_matrix[7], 'o-', label='CL(h)', color='black')
axes[1].plot(0, CL_matrix_extrapolated2[7], 'o', color='red', label='Extrapolated')
axes[1].plot([h_list[-1], 0], [CL_matrix[7][-1], CL_matrix_extrapolated2[7]], 'r--')
axes[1].set_xlabel('1000*h',fontsize = 14)
axes[1].set_title('Horizontal pore',fontsize = 16)
axes[1].grid(True)
axes[1].legend()
axes[1].tick_params(axis='both', labelsize=12)
ymin, ymax = axes[1].get_ylim()
axes[1].set_ylim(ymin - 0.05*(ymax-ymin), ymax + 0.05*(ymax-ymin))





print('Case 3 : ')


#%% Case 3
# Lecture du fichier CSV
df = pd.read_csv("CONFIG_1_RESULT/resultats_simulations_Airfoil3.csv")

# Tri des valeurs uniques de AoA et NumPan
aoa_values = sorted(df['AoA'].unique())
numpan_values = sorted(df['NumPan'].unique())

# Initialisation des matrices
CL_matrix = np.zeros((len(aoa_values), len(numpan_values)))
CD_matrix = np.zeros((len(aoa_values), len(numpan_values)))

# Remplissage des matrices
for i, aoa in enumerate(aoa_values):
    for j, numpan in enumerate(numpan_values):
        subset = df[(df['AoA'] == aoa) & (df['NumPan'] == numpan)]
        if not subset.empty:
            CL_matrix[i, j] = subset['CL'].values[0]
            CD_matrix[i, j] = subset['CD'].values[0]
print('CL_matrix3 = ', CL_matrix)
CL_matrix_extrapolated3, CD_matrix_extrapolated3,h,GCI21_l3,GCI32_l3,AR_l3,GCI21_d3,GCI32_d3,AR_d3, p_l3,p_d3 = extrapolate_matrices(CL_matrix,CD_matrix,numpan_values)
print('Case 3 :', CL_matrix_extrapolated3[7])

# h_list = 1 / NumPan_values
h_list = [max(numpan_values)*2.076 / n for n in numpan_values]

#plot_convergence(h_list, CL_matrix, CD_matrix, CL_matrix_extrapolated3, CD_matrix_extrapolated3, aoa_values)
axes[2].plot(h_list,CL_matrix[7], 'o-', label='CL(h)', color='black')
axes[2].plot(0, CL_matrix_extrapolated3[7], 'o', color='red', label='Extrapolated')
axes[2] .plot([h_list[-1], 0], [CL_matrix[7][-1], CL_matrix_extrapolated3[7]], 'r--')
axes[2].set_xlabel('1000*h',fontsize = 14)
axes[2].set_title('Vertical pore',fontsize = 16)
axes[2].grid(True)
axes[2].legend()
axes[2].tick_params(axis='both', labelsize=12)
ymin, ymax = axes[2].get_ylim()
axes[2].set_ylim(ymin - 0.05*(ymax-ymin), ymax + 0.05*(ymax-ymin))


#plt.suptitle('CL Convergence for AoA = 2°', fontsize=14)
plt.savefig("SAVE_FIGURE/GRID_CONVERGENCE/GRID_CONVERGENCE.pdf", format="pdf")
plt.show()






plot_extrapolated_vs_aoa(aoa_values, CL_matrix_extrapolated1, CD_matrix_extrapolated1, CL_matrix_extrapolated2, CD_matrix_extrapolated2, CL_matrix_extrapolated3, CD_matrix_extrapolated3)