import numpy as np
import matplotlib.pyplot as plt


def XFOIL_DATA(AirfoilName,AoA):
    filename = "X_FOIL/XFOIL_DATA/"+AirfoilName+"_"+str(AoA)+"deg.dat"
    # Charger les données
    # skiprows=1 pour sauter la ligne d'entête
    data = np.loadtxt(filename, skiprows=1)

    # Séparer X et Cp
    x = data[:, 0]
    cp = data[:, 1]
    return x,cp


if __name__ == '__main__':
    x,cp = XFOIL_DATA('2412',4)
    # Tracer Cp
    plt.figure()
    plt.plot(x, cp)
    plt.gca().invert_yaxis()  # Cp plots ont souvent l'axe Y inversé
    plt.xlabel("x/c")
    plt.ylabel("Cp")
    plt.title("Coefficient de Pression à α = 4°")
    plt.grid(True)
    plt.show()