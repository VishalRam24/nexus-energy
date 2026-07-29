"""
EC014 -- Metal Hydride H2 Storage -- F2a Coupled Kinetics + Lumped Thermal Model

First-principles 0D lumped model of a metal-hydride (LaNi5) bed. Two coupled
state variables integrated with scipy.integrate.solve_ivp:

  1. Hydride loading  X = H/M ratio  [atoms_H per formula unit]   (X in [0, X_max])
  2. Bed temperature  T  [K]

------------------------------------------------------------------------------
Equilibrium plateau pressure -- van't Hoff (Sandrock 1999, Lototskyy 2014):

    ln(P_eq / P_ref) = -dH_des/(R*T) + dS_des/R
    P_eq^abs = hysteresis_factor * P_eq^des           (absorption branch higher)

with dH_des = -dH_abs > 0 (endothermic desorption), dS_des = -dS_abs > 0.
d(ln P_eq)/dT > 0 : the plateau pressure rises with temperature.  ✓

------------------------------------------------------------------------------
Reaction kinetics (Jemni & Ben Nasrallah 1995; Mayer, Groll & Supper 1987):

  Absorption (P_supply > P_eq), Arrhenius pre-factor + log-pressure driving force:
    dX/dt = Ca * exp(-Ea_abs/(R*T)) * ln(P_supply / P_eq) * (X_max - X)

  Desorption (P_supply < P_eq):
    dX/dt = Cd * exp(-Ea_des/(R*T)) * ((P_supply - P_eq)/P_eq) * X

The driving force vanishes at equilibrium (P_supply = P_eq) -> dX/dt = 0, which
enforces the equilibrium *plateau*: the bed loads/unloads only while the supply
pressure is off the plateau, and stalls on it. The (X_max - X) and X factors
enforce the mass bounds 0 <= X <= X_max (saturation / depletion).

------------------------------------------------------------------------------
Lumped energy balance ODE (Jemni & Ben Nasrallah 1995; Chung & Ho 2009):

    (m_bed * cp_bed) * dT/dt = Q_rxn - Q_cool

    Q_rxn  =  -dH_abs * dN_H2/dt        exothermic on absorption (dX/dt>0),
                                        endothermic on desorption (dX/dt<0)
    Q_cool =  hA_cool * (T - T_coolant)

    dN_H2/dt = (m_bed / M_alloy) * (1/2) * dX/dt    [mol_H2 / s]
      (1 H2 molecule == 2 H atoms; n_formula = m_bed / M_alloy)

Energy conservation: integral of Q_rxn over a full absorption equals
|dH_abs| * total mol H2 absorbed, to numerical tolerance.

------------------------------------------------------------------------------
References:
    Sandrock, G. (1999). J. Alloys Compd. 293-295, 877-888.  (LaNi5 PCT data)
    Jemni, A. & Ben Nasrallah, S. (1995). Int. J. Hydrogen Energy 20(1), 43-52.
    Mayer, U., Groll, M. & Supper, W. (1987). J. Less-Common Met. 131, 235-244.
    Chung, C.A. & Ho, C.-J. (2009). Int. J. Hydrogen Energy 34, 4351-4364.
    Lototskyy et al. (2014). Prog. Nat. Sci. Mater. Int. 24(2), 97-116.
    van't Hoff, J.H. (1886). Z. Phys. Chem. 1, 481.
"""

import numpy as np
from scipy.integrate import solve_ivp

R_UNIVERSAL = 8.314      # J/(mol.K)
M_H2 = 0.002016          # kg/mol


class MetalHydrideF2a:
    """Metal-hydride bed: coupled absorption/desorption kinetics + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.delta_H_abs = u["delta_H"]["value"]      # J/mol_H2 (negative)
        self.delta_S_abs = u["delta_S"]["value"]      # J/mol_H2/K (negative)
        self.H_max_wt_pct = u["H_max_wt_pct"]["value"]
        self.X_max = u["HM_max"]["value"]             # atoms H / formula
        self.m_bed = u["bed_mass_kg"]["value"]        # kg alloy
        self.M_alloy = u["M_alloy"]["value"]          # kg/mol
        self.Ca = u["Ca"]["value"]                    # 1/s
        self.Ea_abs = u["Ea_abs"]["value"]            # J/mol
        self.Cd = u["Cd"]["value"]                    # 1/s
        self.Ea_des = u["Ea_des"]["value"]            # J/mol
        self.P_ref = u["P_ref"]["value"]              # bar
        self.hysteresis_factor = u["hysteresis_factor"]["value"]
        self.cp_bed = u["cp_bed"]["value"]            # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]          # W/K
        self.T_coolant = u["T_coolant"]["value"]      # K

        # mol of formula units in the bed
        self.n_formula = self.m_bed / self.M_alloy    # mol
        # mol H2 at full loading (X_max atoms H / 2 per H2)
        self.n_H2_max = self.n_formula * self.X_max / 2.0   # mol
        self.m_H2_max = self.n_H2_max * M_H2          # kg

    # ------------------------------------------------------------------
    # Equilibrium plateau pressure (van't Hoff)
    # ------------------------------------------------------------------
    def plateau_pressure(self, T, mode="absorption"):
        """Equilibrium plateau pressure [bar] for given branch."""
        dH_des = -self.delta_H_abs    # > 0
        dS_des = -self.delta_S_abs    # > 0
        ln_P_over_ref = -dH_des / (R_UNIVERSAL * T) + dS_des / R_UNIVERSAL
        P_des = self.P_ref * np.exp(ln_P_over_ref)
        if mode == "absorption":
            return self.hysteresis_factor * P_des
        return P_des

    # ------------------------------------------------------------------
    # Reaction rate dX/dt  [ (atoms H/formula) / s ]
    # ------------------------------------------------------------------
    def reaction_rate(self, X, T, P_supply):
        """Net hydride loading rate dX/dt given loading X, temperature, supply P."""
        X = float(np.clip(X, 0.0, self.X_max))
        P_supply = max(float(P_supply), 1e-9)

        P_eq_abs = self.plateau_pressure(T, "absorption")
        P_eq_des = self.plateau_pressure(T, "desorption")

        if P_supply > P_eq_abs:
            # Absorption: positive, vanishes as X -> X_max
            k = self.Ca * np.exp(-self.Ea_abs / (R_UNIVERSAL * T))
            driving = np.log(P_supply / P_eq_abs)
            return k * driving * (self.X_max - X)
        elif P_supply < P_eq_des:
            # Desorption: negative, vanishes as X -> 0
            k = self.Cd * np.exp(-self.Ea_des / (R_UNIVERSAL * T))
            driving = (P_supply - P_eq_des) / P_eq_des   # negative
            return k * driving * X
        else:
            # On the plateau (between desorption & absorption branches): stalled
            return 0.0

    # ------------------------------------------------------------------
    # H2 molar flow corresponding to dX/dt
    # ------------------------------------------------------------------
    def h2_molar_flow(self, dXdt):
        """mol H2 / s absorbed (>0) or desorbed (<0) for a given dX/dt."""
        return self.n_formula * 0.5 * dXdt

    # ------------------------------------------------------------------
    # Heat-generation rate  [W]
    # ------------------------------------------------------------------
    def heat_generation(self, dXdt):
        """Reaction heat rate [W]: exothermic (+) on absorption, (-) on desorption."""
        n_dot_H2 = self.h2_molar_flow(dXdt)        # mol H2 / s
        # Q = -dH_abs * n_dot : absorption (dH_abs<0, n_dot>0) -> Q>0 (release)
        return -self.delta_H_abs * n_dot_H2        # W

    # ------------------------------------------------------------------
    # Thermal ODE  dT/dt  [K/s]
    # ------------------------------------------------------------------
    def dTdt(self, T, dXdt):
        Q_rxn = self.heat_generation(dXdt)
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_rxn - Q_cool) / (self.m_bed * self.cp_bed)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def stored_mass_kg(self, X):
        """Stored H2 mass [kg] at loading X."""
        X = np.clip(np.asarray(X, dtype=float), 0.0, self.X_max)
        return self.n_formula * 0.5 * X * M_H2

    def soc(self, X):
        """State of charge (0..1) = X / X_max."""
        return np.clip(np.asarray(X, dtype=float) / self.X_max, 0.0, 1.0)

    def gravimetric_wt_pct(self, X):
        """Gravimetric H2 capacity [wt%] at loading X."""
        m_h2 = self.stored_mass_kg(X)
        return m_h2 / (m_h2 + self.m_bed) * 100.0

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, P_supply_bar, T_bed_K, X0=0.0, dt=1.0, duration_s=600.0):
        """
        Integrate coupled (X, T) ODE system.

        Parameters
        ----------
        P_supply_bar : float or callable(t)
            Hydrogen supply (manifold) pressure [bar]. > plateau -> charge.
        T_bed_K : float
            Initial bed temperature [K].
        X0 : float
            Initial H/M loading [atoms H / formula] (0 = empty).
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation time [s].

        Returns
        -------
        dict of time-series arrays:
            t, HM_ratio, soc, stored_mass_kg, temperature,
            P_supply, P_eq_abs, P_eq_des, dXdt, Q_rxn, Q_cool, gravimetric_wt_pct
        """
        _P = P_supply_bar if callable(P_supply_bar) else (lambda t: P_supply_bar)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            X, T = y
            X = min(max(X, 0.0), self.X_max)
            P = _P(t)
            dXdt = self.reaction_rate(X, T, P)
            dTdt = self.dTdt(T, dXdt)
            return [dXdt, dTdt]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [float(X0), float(T_bed_K)],
            t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        X_out = np.clip(sol.y[0], 0.0, self.X_max)
        T_out = sol.y[1]
        N = len(t_out)

        P_sup = np.zeros(N)
        P_abs = np.zeros(N)
        P_des = np.zeros(N)
        dXdt = np.zeros(N)
        Q_rxn = np.zeros(N)
        Q_cool = np.zeros(N)

        for i in range(N):
            P = _P(t_out[i])
            P_sup[i] = P
            P_abs[i] = self.plateau_pressure(T_out[i], "absorption")
            P_des[i] = self.plateau_pressure(T_out[i], "desorption")
            dXdt[i] = self.reaction_rate(X_out[i], T_out[i], P)
            Q_rxn[i] = self.heat_generation(dXdt[i])
            Q_cool[i] = self.hA_cool * (T_out[i] - self.T_coolant)

        return {
            "t": t_out,
            "HM_ratio": X_out,
            "soc": self.soc(X_out),
            "stored_mass_kg": self.stored_mass_kg(X_out),
            "temperature": T_out,
            "P_supply": P_sup,
            "P_eq_abs": P_abs,
            "P_eq_des": P_des,
            "dXdt": dXdt,
            "Q_rxn": Q_rxn,
            "Q_cool": Q_cool,
            "gravimetric_wt_pct": self.gravimetric_wt_pct(X_out),
        }
