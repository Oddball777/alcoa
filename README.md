3.2: Résolution de l'équation différentielle

L'équation à résoudre:
$$(pc_p)_{eq} \frac{\partial T}{\partial t} + (pc_p)_f u \frac{\partial T}{\partial x} = k_{eq} \frac{\partial^2 T}{\partial x^2}$$

- $(pc_p)_{eq}$ sont la densité et la chaleur spécifique équivalentes du milieu poreux (moyenne spatiale entre gaz et solide).
- $(pc_p)_f$ sont la densité et la chaleur spécifique du fluide.
- $k_{eq}$ est la conductivité thermique équivalente du milieu poreux (moyenne entre gaz et solide).
- $u$ est la vitesse des gaz.
- $T$ est la température. On suppose l'équilibre thermique local (température du solide = température du fluide en tout point).
- $x$ est la position le long de l'écoulement.

Constantes et autres paramètres:
- Débit massique des gaz: 2000 scfm = 0.9438949 m3/s
- Ratio du volume occupé par le solide dans le garnissage: 50%
- Hauteur du garnissage: 1.6 m
- Densité du garnissage: 2600 kg/m3
- Chaleur spécifique du garnissage: 1000 J/kgK
- Conductivité thermique du garnissage: 2 W/mK
- Température d'entrée des gaz: 150 °C = 423.15 K
- Condition à la sortie: $\frac{\partial T}{\partial x} = 0$ (pas de conduction à la sortie, seulement le déplacement du fluide tranporte la chaleur vers l'extérieur)


Paramètres de la simulation:
- Maillage: linspace(0,1.6,75)
- Temps de changement de régime: 1000
- Temps discrétisé: linspace(0,1000,100)
- Tinitial: 750*ones(100)


Stratégie de résolution:
- On résout l'équation différentielle en utilisant scipy.integrate.odeint

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Paramètres
Q = 0.9438949 # Débit massique des gaz (m3/s)
V = 0.5*1.6 # Volume du garnissage (m3)
rho = 2600 # Densité du garnissage (kg/m3)
cp = 1000 # Chaleur spécifique du garnissage (J/kgK)
k = 2 # Conductivité thermique du garnissage (W/mK)
Tin = 423.15 # Température d'entrée des gaz (K)
Tout = 298.15 # Température de sortie des gaz (K)
L = 1.6 # Hauteur du garnissage (m)
x = np.linspace(0,L,75) # Maillage
t = np.linspace(0,1000,100) # Temps discrétisé
Tinitial = 750*np.ones(100) # Température initiale

# Fonction à résoudre
def f(T,t):
    dTdt = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0:
            dTdt[i] = (k/(rho*cp))*(T[i+1]-T[i])/(x[i+1]-x[i]) - (Q/(rho*cp*V))*(T[i]-Tin)/(x[i+1]-x[i])
        elif i == len(x)-1:
            dTdt[i] = (k/(rho*cp))*(T[i]-T[i-1])/(x[i]-x[i-1]) - (Q/(rho*cp*V))*(T[i]-Tin)/(x[i]-x[i-1])
        else:
            dTdt[i] = (k/(rho*cp))*(T[i+1]-T[i])/(x[i+1]-x[i]) - (k/(rho*cp))*(T[i]-T[i-1])/(x[i]-x[i-1]) - (Q/(rho*cp*V))*(T[i]-Tin)/(x[i+1]-x[i-1])
    return dTdt

# Résolution
T = odeint(f,Tinitial,t)

# Plot
plt.plot(x,T[0,:],label='t=0')
plt.plot(x,T[10,:],label='t=10')
plt.plot(x,T[20,:],label='t=20')
plt.plot(x,T[30,:],label='t=30')
plt.plot(x,T[40,:],label='t=40')
plt.plot(x,T[50,:],label='t=50')
plt.plot(x,T[60,:],label='t=60')
plt.plot(x,T[70,:],label='t=70')
plt.plot(x,T[80,:],label='t=80')
plt.plot(x,T[90,:],label='t=90')
plt.plot(x,T[99,:],label='t=100')
plt.xlabel('x (m)')
plt.ylabel('T (K)')
plt.legend()
plt.show()
```