from calendar import c
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from findiff import PDE, BoundaryConditions, FinDiff
from matplotlib.animation import FuncAnimation
from sympy import Subs


@dataclass
class Substance:
    """
    Classe pour définir les propriétés d'une substance.
    """

    temperature: float
    densite: float
    chaleur_specifique: float
    conductivite_thermique: float

    @property
    def rho_cp(self):
        return self.densite * self.chaleur_specifique


@dataclass
class Garnissage:
    """
    Classe pour simuler le transfert de chaleur dans un garnissage.

    Les paramètres doivent être exprimés dans les unités suivantes:
    - rho_cp_eq: J/m3.K
    - rho_cp_f: J/m3.K
    - k_eq: W/m.K
    - vitesse_fluide: m/s
    - longueur_garnissage: m
    - resolution_temporelle: s (intervalle de temps entre chaque itération)
    - resolution_spatiale: m (intervalle de distance entre chaque point spatial)
    - temperature_initiale_garnissage: K
    """

    substance: Substance
    ratio_volume_solide: float
    vitesse_fluide: float  # m/s
    longueur_garnissage: float  # m
    resolution_temporelle: float  # s
    resolution_spatiale: float  # m
    temperature_initiale_garnissage: float  # K
    phases: list = field(default_factory=list, init=False)

    def demarrer_phase(
        self,
        fluide: Substance,
        direction: int = 1,
        duree: float = 15,
    ):
        """
        Démarre une phase de simulation. Les paramètres sont:
        - fluide: objet Substance représentant le fluide et ses propriétés
        - direction: direction du fluide (1 pour entrant à gauche, -1 pour entrant à droite)
        - duree: durée de la phase (s)
        """
        print(
            f"Phase {len(self.phases)+1} en cours: température du gaz {fluide.temperature} K"
        )
        if len(self.phases) == 0:
            n_points = int(self.longueur_garnissage / self.resolution_spatiale)
            temperature_initiale = (
                np.ones(n_points) * self.temperature_initiale_garnissage
            )
        else:
            temperature_initiale = self.phases[-1][:, -1]
        T = self._simuler_phase(temperature_initiale, fluide, duree, direction)
        self.phases.append(T)

    def _simuler_phase(
        self, initial_condition, fluide: Substance, duree: float, direction: int
    ):
        # Définir le domaine spatial
        x_max = self.longueur_garnissage
        x_min = 0
        n_points = int((x_max - x_min) / self.resolution_spatiale)
        self.x = np.linspace(x_min, x_max, n_points)
        dx = self.x[1] - self.x[0]

        # Définir le domaine temporel
        t_max = duree
        n_t_points = int(t_max / self.resolution_temporelle)
        t = np.linspace(0, t_max, n_t_points)
        dt = t[1] - t[0]

        # Définir les opérateurs de différences finies
        dT_dx = FinDiff(0, dx, 1)  # première dérivée p/r à x
        d2T_dx2 = FinDiff(0, dx, 2)  # deuxième dérivée p/r à x
        dT_dt = FinDiff(1, dt, 1)  # première dérivée p/r à t

        rho_cp_eq = (
            self.substance.rho_cp * self.ratio_volume_solide
            + (1 - self.ratio_volume_solide) * fluide.rho_cp
        )
        k_eq = (
            self.substance.conductivite_thermique * self.ratio_volume_solide
            + (1 - self.ratio_volume_solide) * fluide.conductivite_thermique
        )

        if direction == 1 or direction == -1:
            u = self.vitesse_fluide * direction
        else:
            raise ValueError("direction must be 1 or -1")
        lhs = rho_cp_eq * dT_dt + fluide.rho_cp * u * dT_dx - k_eq * d2T_dx2
        rhs = np.zeros((n_points, n_t_points))

        # Définir les conditions aux limites
        bc = BoundaryConditions(shape=(n_points, n_t_points))
        # Condition initiale
        for x_point in range(n_points):
            bc[x_point, 0] = initial_condition[x_point]
        if direction == 1:
            bc[0, :] = fluide.temperature  # Température d'entrée à gauche
            bc[-1, :] = dT_dx, 0  # Dérivée nulle à la sortie à droite
        elif direction == -1:
            bc[0, :] = dT_dx, 0  # Dérivée nulle à la sortie à gauche
            bc[-1, :] = fluide.temperature  # Température d'entrée à droite

        pde = PDE(lhs, rhs, bc)
        T = pde.solve()
        return T

    def get_resultats(self):
        """
        Retourne les résultats de la simulation sous forme d'un tableau numpy.

        Le tableau a la forme (nombre de points spatiaux, nombre de points temporels).

        Exemples:
        - pour accéder à la température au point spatial 5 à la fin, on fait T[5, -1].
        - pour accéder à la température au point spatial 5 à tous les temps, on fait T[5, :].
        - pour accéder à la température à tous les points spatiaux au temps 10, on fait T[:, 10].

        Note: ici, 5 et 10 sont des indices, pas des valeurs de x ou t.
        """
        T = np.concatenate(self.phases, axis=1)
        return T

    def animer_ligne(self, nom_fichier="ligne", step: int = 1):
        """
        Crée une animation du graphique de la température en fonction de la position.
        """
        print(f"Animation du graphique {nom_fichier}")
        T = self.get_resultats()
        T_min = np.min(T)
        T_max = np.max(T)
        fig, ax = plt.subplots()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Temperature (K)")
        ax.set_xlim(0, self.longueur_garnissage)
        ax.set_ylim(T_min, T_max)
        (line,) = ax.plot([], [], lw=2)

        time_text = ax.text(
            0.02,
            0.95,
            "",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        def init():
            line.set_data([], [])
            return (line,)

        def animate(i):
            line.set_data(self.x, T[:, i * step])
            time_text.set_text(f"t = {i*self.resolution_temporelle*step:.2f} s")
            return (line,)

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=T.shape[1] // step,
            interval=20,
            blit=True,
        )
        anim.save(f"{nom_fichier}.mp4", writer="ffmpeg")

    def animer_heatmap(self, nom_fichier="heatmap", step: int = 1):
        """
        Crée une animation du graphique de la température en fonction de la position et du temps.
        """
        print(f"Animation du graphique {nom_fichier}")
        T = self.get_resultats()
        x_min, x_max = 0, self.longueur_garnissage
        y_min, y_max = 0, 3
        n_y_points = 20
        T_min = np.min(T)
        T_max = np.max(T)

        x = np.linspace(x_min, x_max, T.shape[0])
        y = np.linspace(y_min, y_max, n_y_points)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        heat_map = ax.pcolormesh(X, Y, np.zeros_like(X), shading="auto", cmap="hot")
        time_text = ax.text(
            0.02,
            0.95,
            "",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        def animate2(i):
            T_dup = np.tile(T[:, i * step], (n_y_points, 1)).T
            heat_map.set_array(T_dup.T)
            heat_map.set_clim(T_min, T_max)
            time_text.set_text(f"t = {i*self.resolution_temporelle*step:.2f} s")
            return (heat_map,)

        anim = FuncAnimation(fig, animate2, frames=int(T.shape[1] / step), interval=20)
        anim.save(f"{nom_fichier}.mp4", writer="ffmpeg")

    def get_min_temperature_at_end(self):
        T = self.get_resultats()
        return np.min(T[:, -1])

    def get_max_temperature_at_end(self):
        T = self.get_resultats()
        return np.max(T[:, -1])


if __name__ == "__main__":
    debit = 0.9438949  # m3/s (valeur absolue)
    A_conduite = 1  # m2
    vitesse_fluide = debit / A_conduite  # m/s

    co2 = Substance(
        temperature=1090,
        densite=0.4847,
        chaleur_specifique=1257.4,
        conductivite_thermique=0.076578,
    )

    air = Substance(
        temperature=423,
        densite=0.8345,
        chaleur_specifique=1014.5,
        conductivite_thermique=0.034425,
    )

    subst_garnissage = Substance(
        temperature=500,
        densite=2600,
        chaleur_specifique=1000,
        conductivite_thermique=2,
    )

    # Créer le garnissage
    garni = Garnissage(
        substance=subst_garnissage,
        vitesse_fluide=vitesse_fluide,
        ratio_volume_solide=0.5,
        longueur_garnissage=1.6,
        temperature_initiale_garnissage=450,
        resolution_temporelle=0.3,  # augmenter pour accélérer le calcul (et la vitesse de l'animation)
        resolution_spatiale=0.015,  # augmenter pour accélérer la simulation (mais moins précis)
    )

    # Démarrer les phases
    for i in range(10):
        print(f"Cycle {i+1}")
        garni.demarrer_phase(air, direction=1, duree=100)
        garni.demarrer_phase(air, direction=1, duree=100)
        garni.demarrer_phase(co2, direction=-1, duree=100)
        print(f"\n")

    # Animations
    garni.animer_ligne("ligne", step=10)
    garni.animer_heatmap("heatmap", step=10)
