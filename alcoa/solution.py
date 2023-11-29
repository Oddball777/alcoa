from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from findiff import PDE, BoundaryConditions, FinDiff
from matplotlib.animation import FuncAnimation


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

    # Constantes pour l'équation différentielle
    rho_cp_f: float  # kg/m3.K
    rho_cp_eq: float  # kg/m3.K
    k_eq: float  # W/m.K
    vitesse_fluide: float  # m/s
    # Paramètres de simulation
    longueur_garnissage: float  # m
    resolution_temporelle: float  # s
    resolution_spatiale: float  # m
    temperature_initiale_garnissage: float  # K
    phases: list = field(default_factory=list, init=False)

    def demarrer_phase(
        self,
        temperature_fluide: float,
        direction: int = 1,
        duree: float = 15,
    ):
        """
        Démarre une phase de simulation. Les paramètres sont:
        - temperature_fluide: température du fluide entrant dans le garnissage (K)
        - direction: direction du fluide (1 pour entrant à gauche, -1 pour entrant à droite)
        - duree: durée de la phase (s)
        """
        print(
            f"Phase {len(self.phases)+1} en cours: température du gaz {temperature_fluide} K"
        )
        if len(self.phases) == 0:
            n_points = int(self.longueur_garnissage / self.resolution_spatiale)
            temperature_initiale = (
                np.ones(n_points) * self.temperature_initiale_garnissage
            )
        else:
            temperature_initiale = self.phases[-1][:, -1]
        T = self._simuler_phase(
            temperature_initiale, temperature_fluide, duree, direction
        )
        self.phases.append(T)

    def _simuler_phase(self, initial_condition, fluid_temperature, duree, direction):
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

        if direction == 1 or direction == -1:
            u = self.vitesse_fluide * direction
        else:
            raise ValueError("direction must be 1 or -1")
        lhs = self.rho_cp_eq * dT_dt + self.rho_cp_f * u * dT_dx - self.k_eq * d2T_dx2
        rhs = np.zeros((n_points, n_t_points))

        # Définir les conditions aux limites
        bc = BoundaryConditions(shape=(n_points, n_t_points))
        if direction == 1:
            bc[0, :] = fluid_temperature  # Température d'entrée à gauche
            bc[-1, :] = dT_dx, 0  # Dérivée nulle à la sortie à droite
        elif direction == -1:
            bc[0, :] = dT_dx, 0  # Dérivée nulle à la sortie à gauche
            bc[-1, :] = fluid_temperature  # Température d'entrée à droite
        # Condition initiale
        for x_point in range(n_points):
            bc[x_point, 0] = initial_condition[x_point]

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

    def animer_ligne(self, nom_fichier="ligne"):
        """
        Crée une animation du graphique de la température en fonction de la position.
        """
        print(f"Animation du graphique {nom_fichier}")
        T = self.get_resultats()
        fig, ax = plt.subplots()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Temperature (K)")
        ax.set_xlim(0, self.longueur_garnissage)
        ax.set_ylim(0, 600)
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
            line.set_data(self.x, T[:, i])
            time_text.set_text(f"t = {i*self.resolution_temporelle:.2f} s")
            return (line,)

        anim = FuncAnimation(
            fig, animate, init_func=init, frames=T.shape[1], interval=20, blit=True
        )
        anim.save(f"{nom_fichier}.mp4", writer="ffmpeg")

    def animer_heatmap(self, nom_fichier="heatmap"):
        """
        Crée une animation du graphique de la température en fonction de la position et du temps.
        """
        print(f"Animation du graphique {nom_fichier}")
        T = self.get_resultats()
        x_min, x_max = 0, self.longueur_garnissage
        y_min, y_max = 0, 3
        n_y_points = 20

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
            T_dup = np.tile(T[:, i], (n_y_points, 1)).T
            heat_map.set_array(T_dup.T)
            heat_map.set_clim(0, 600)
            time_text.set_text(f"t = {i*self.resolution_temporelle:.2f} s")
            return (heat_map,)

        anim = FuncAnimation(fig, animate2, frames=T.shape[1], interval=20)
        anim.save(f"{nom_fichier}.mp4", writer="ffmpeg")

    def get_min_temperature_at_end(self):
        T = self.get_resultats()
        return np.min(T[:, -1])

    def get_max_temperature_at_end(self):
        T = self.get_resultats()
        return np.max(T[:, -1])


def get_proprietes_equivalentes(
    ratio_volume_solide,
    densite_solide,
    densite_fluide,
    chaleur_specifique_solide,
    chaleur_specifique_fluide,
    conductivite_thermique_solide,
    conductivite_thermique_fluide,
):
    """
    Retourne les propriétés équivalentes rho_cp_eq, rho_cp_f, k_eq à partir des propriétés des solides et des fluides
    et du ratio entre le volume de solide et le volume de fluide.

    Cette fonction assume que les propriétés des solides et des fluides sont exprimées dans les unités suivantes:
    - densité: kg/m3
    - chaleur spécifique: J/kg.K
    - conductivité thermique: W/m.K
    """
    rho_cp_eq = (
        ratio_volume_solide * densite_solide * chaleur_specifique_solide
        + (1 - ratio_volume_solide) * densite_fluide * chaleur_specifique_fluide
    )
    k_eq = (
        ratio_volume_solide * conductivite_thermique_solide
        + (1 - ratio_volume_solide) * conductivite_thermique_fluide
    )
    rho_cp_f = densite_fluide * chaleur_specifique_fluide
    return rho_cp_eq, rho_cp_f, k_eq


if __name__ == "__main__":
    # Définir les paramètres rho_cp_eq, rho_cp_f, k_eq et vitesse_fluide
    rho_cp_eq, rho_cp_f, k_eq = get_proprietes_equivalentes(
        ratio_volume_solide=0.5,
        densite_solide=2600,
        densite_fluide=1.225,
        chaleur_specifique_solide=1000,
        chaleur_specifique_fluide=1015,
        conductivite_thermique_solide=2,
        conductivite_thermique_fluide=0.035,
    )
    vitesse_fluide = 0.05  # vitesse en valeur absolue

    # Réglage manuel pour les tests (à commenter si on utilise la fonction get_proprietes_equivalentes)
    rho_cp_eq = 1
    rho_cp_f = 1
    k_eq = 1

    # Créer le garnissage
    garni = Garnissage(
        rho_cp_f=rho_cp_f,
        rho_cp_eq=rho_cp_eq,
        k_eq=k_eq,
        vitesse_fluide=vitesse_fluide,
        longueur_garnissage=1.6,
        temperature_initiale_garnissage=300,
        resolution_temporelle=0.01,  # augmenter pour accélérer le calcul (et la vitesse de l'animation)
        resolution_spatiale=0.015,  # augmenter pour accélérer la simulation (mais moins précis)
    )

    # Démarrer les phases
    garni.demarrer_phase(temperature_fluide=500, direction=1, duree=2)
    garni.demarrer_phase(temperature_fluide=100, direction=-1, duree=2)

    # Animations
    garni.animer_ligne("ligne")
    garni.animer_heatmap("heatmap")
