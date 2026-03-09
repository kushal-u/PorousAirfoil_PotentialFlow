import numpy as np
import math as math
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def extrapolate_matrices(CL_matrix, CD_matrix, NumPan_list):
    num_aoa = CL_matrix.shape[0]
    extrapolated_CL = np.zeros(num_aoa)
    extrapolated_CD = np.zeros(num_aoa)
    GCI21_l_list = np.zeros(num_aoa)
    GCI21_d_list = np.zeros(num_aoa)
    GCI32_l_list = np.zeros(num_aoa)
    GCI32_d_list = np.zeros(num_aoa)
    AR_l_list = np.zeros(num_aoa)
    AR_d_list = np.zeros(num_aoa)
    p_l_list = np.zeros(num_aoa)
    p_d_list = np.zeros(num_aoa)

    for i in range(num_aoa):
        cl_values = [CL_matrix[i, NumPan_list.index(n)] for n in NumPan_list]
        cd_values = [CD_matrix[i, NumPan_list.index(n)] for n in NumPan_list]

        # Appliquer l'extrapolation à chaque ligne
        F_cl,h, GCI21_l,GCI32_l,AR_l,p_l = RICHARDSON_EXTRAPOLATION(NumPan_list, cl_values)
        F_cd,h, GCI21_d,GCI32_d,AR_d,p_d = RICHARDSON_EXTRAPOLATION(NumPan_list, cd_values)

        extrapolated_CL[i] = F_cl[0]
        extrapolated_CD[i] = F_cd[0]
        GCI21_l_list[i] = GCI21_l
        GCI21_d_list[i] = GCI21_d
        GCI32_l_list[i] = GCI32_l
        GCI32_d_list[i] = GCI32_d
        AR_l_list[i] = AR_l
        AR_d_list[i] = AR_d
        p_l_list[i] = p_l
        p_d_list[i] = p_d

    return extrapolated_CL, extrapolated_CD,h,GCI21_l_list,GCI32_l_list,AR_l_list,GCI21_d_list,GCI32_d_list,AR_d_list, p_l_list,p_d_list

def RICHARDSON_EXTRAPOLATION(NumPan_list,Result_list):
    # Richardson Extrapolation
    # NumPan_list: list of number of panels
    # Result_list: list of results
    # return: extrapolated result
    h = np.zeros(len(NumPan_list)+1)
    F = np.zeros(len(Result_list)+1)
    F[1:] = Result_list[::-1]
    h[1] = NumPan_list[-1]/NumPan_list[-1]*2.076
    h[2] = NumPan_list[-1]/NumPan_list[-2]*2.076
    h[3] = NumPan_list[-1]/NumPan_list[-3]*2.076

    r32 = h[3]/h[2]
    r21 = h[2]/h[1]
    if abs(F[1]-F[2])<1e-14:
        F[0] = F[1]
        GCI21 = -1
        GCI32 = -1
        AR = -1
        p=-1
    else:
        p = find_p(r21, r32, F[1], F[2], F[3])
        
        print('p = ',p)


        F[0]  = (r21**p*F[1] - F[2])/(r21**p-1)

        eps32 = abs((F[3]-F[2])/F[2])
        eps21 = abs((F[2]-F[1])/F[1])

        GCI21 = 1.25*eps21/(r21**p-1)
        GCI32 = 1.25*eps32/(r32**p-1)

        AR = r21**p*GCI21/GCI32

    return F,h,GCI21,GCI32,AR,p

def equation_p(p, r21, r32, f1, f2, f3):
    delta_f = (f3 - f2) / (f2 - f1)
    sgn = np.sign(delta_f)
    
    q_p_numerator = r21**p - sgn
    q_p_denominator = r32**p - sgn

    if q_p_denominator == 0 or q_p_numerator <= 0 or q_p_denominator <= 0:
        return np.inf  # pour éviter les erreurs log ou divisions nulles
    
    q_p = np.log(q_p_numerator / q_p_denominator)
    
    try:
        main_term = (1 / np.log(r21)) * np.log(abs((f3 - f2) / (f2 - f1)))
    except ZeroDivisionError:
        return np.inf
    
    return p - main_term - q_p

def find_p(r21, r32, f1, f2, f3):
    def func(p):
        return equation_p(p, r21, r32, f1, f2, f3)

    a, b = 0.001, 10
    fa, fb = func(a), func(b)

    if np.sign(fa) == np.sign(fb):
        print(f"⚠️ Warning: equation_p(p) n'a pas de signe opposé entre {a} et {b}.")
        print(f"Les valeurs de F sont f1 = {f1},f2 = {f2} et f3 = {f3} ")
        # Affiche la courbe pour aider le débogage
        
        return 2
        raise ValueError("Pas de changement de signe dans l'intervalle donné pour p.")
    #plot_equation_p(r21, r32, f1, f2, f3)
    #print(fa, '    ', fb)
    sol = root_scalar(equation_p, args=(r21, r32, f1, f2, f3), method='brentq', bracket=[0.001, 10])
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Échec de la résolution pour p")

def plot_equation_p(r21, r32, f1, f2, f3):
    ps = np.linspace(0.1, 10, 1000)
    values = []

    for p in ps:
        try:
            val = equation_p(p, r21, r32, f1, f2, f3)
            if np.isfinite(val):
                values.append(val)
            else:
                values.append(np.nan)
        except:
            values.append(np.nan)

    plt.figure()
    plt.plot(ps, values, label="equation_p(p)")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("p")
    plt.ylabel("equation_p(p)")
    plt.title("Équation à résoudre pour p")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    F,h,GCI21,GCI32,AR = RICHARDSON_EXTRAPOLATION([250,500,1000,2000],[0.5,0.55,0.555,0.5555])

    print(F)
    print(h)
    print(GCI21)
    print(GCI32)
    print(AR)