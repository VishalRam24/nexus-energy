"""
EC081 -- Thermochemical Energy Storage (CaO/Ca(OH)2) -- F2a Reaction Kinetics

Physics-lumped 0D model: Arrhenius reaction kinetics with an equilibrium driving
force, coupled to a lumped reactor energy balance ODE.

State vector  y = [X, T]
    X : conversion (state-of-charge), X in [0, 1]
        X = 0 -> fully hydrated Ca(OH)2 (discharged)
        X = 1 -> fully dehydrated CaO + H2O (charged)
    T : reactor (bed) temperature [K]

Reactions
    Charge   (endothermic) : Ca(OH)2 + dH  -> CaO + H2O   (X increases)
    Discharge (exothermic) : CaO + H2O      -> Ca(OH)2 + dH (X decreases)

Kinetics -- shrinking-state first-order Arrhenius model (Schaube 2012):
    Charge   : dX/dt = +A_c * exp(-Ea_c/RT) * (1 - X) * f_eq_charge(T)
    Discharge: dX/dt = -A_d * exp(-Ea_d/RT) *  X      * f_eq_discharge(T)
  where f_eq is a thermodynamic driving-force factor (0..1) that vanishes at the
  equilibrium temperature T_eq(P) given by the van't Hoff relation, so the
  reaction halts when the bed reaches equilibrium.

Equilibrium (van't Hoff / Clausius-Clapeyron form, Schaube 2012):
    ln(P_h2o / P_ref) = -dH/(R*T_eq) + dS/R
  ->  T_eq(P) = dH / (dS - R*ln(P_h2o/P_ref))
  Hydration (discharge) is favoured for T < T_eq; dehydration (charge) for
  T > T_eq. At P = 1 bar, T_eq ~ 505 C for CaO/Ca(OH)2 (Schaube 2012).

Lumped reactor energy balance ODE (1st law on the reactor control volume):
    C_th * dT/dt = Q_rxn + Q_ext
    Q_rxn = -dH_per_kg_rate * (dX/dt)   [exothermic discharge releases heat]
          = -(n_mol * dH_rxn) * dX/dt
    Q_ext = hA * (T_source - T)          [coupling to HTF / heat source / loss]
    C_th  = m_active*cp_solid + m_reactor*cp_reactor   [J/K]

Energy conservation (verifiable):
    Stored chemical energy  E_stored(X) = n_mol * dH_rxn * X        [J]
    i.e. stored energy = reaction enthalpy x reaction extent.
    When the reaction is halted (rate -> 0), X is constant => loss-free storage,
    the defining feature of thermochemical storage (N'Tsoukpoe 2009).

References
    Schaube, Worner, Tamme (2012). High temperature thermochemical heat storage
        with CaO/Ca(OH)2 -- equilibrium and kinetics. Thermochim. Acta 538, 9-20.
    N'Tsoukpoe, Liu, Le Pierres, Luo (2009). A review on long-term sorption solar
        energy storage. Renew. Sust. Energy Rev. 13, 2385-2396.
    Wentworth & Chen (1976). Simple thermal decomposition reactions for storage
        of solar thermal energy. Solar Energy 18, 205-214.
    Pardo et al. (2014). A review on high temperature thermochemical heat energy
        storage. Renew. Sust. Energy Rev. 32, 591-610.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ThermochemicalStorageF2a:
    """CaO/Ca(OH)2 thermochemical reactor -- kinetics + lumped energy balance."""

    R = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.m_active = u["m_active"]["value"]          # kg
        self.M_active = u["M_active"]["value"]          # kg/mol
        self.dH = u["dH_rxn"]["value"]                  # J/mol (positive magnitude)
        self.dS = u["dS_rxn"]["value"]                  # J/(mol.K)
        self.A_c = u["A_pre_charge"]["value"]           # 1/s
        self.Ea_c = u["E_act_charge"]["value"]          # J/mol
        self.A_d = u["A_pre_discharge"]["value"]        # 1/s
        self.Ea_d = u["E_act_discharge"]["value"]       # J/mol
        self.cp_solid = u["cp_solid"]["value"]          # J/(kg.K)
        self.cp_reactor = u["cp_reactor"]["value"]      # J/(kg.K)
        self.m_reactor = u["m_reactor"]["value"]        # kg
        self.hA = u["hA_ext"]["value"]                  # W/K
        self.P_h2o = u["P_h2o"]["value"]                # Pa
        self.P_ref = u["P_ref"]["value"]                # Pa

        # Derived quantities
        self.n_mol = self.m_active / self.M_active      # mol of reactive material
        self.E_max = self.n_mol * self.dH               # J, full charge capacity
        self.C_th = self.m_active * self.cp_solid + self.m_reactor * self.cp_reactor  # J/K

    # ------------------------------------------------------------------
    # Thermodynamic equilibrium temperature (van't Hoff)
    # ------------------------------------------------------------------
    def T_eq(self, P_h2o=None):
        """Equilibrium temperature [K] for given water-vapour pressure."""
        if P_h2o is None:
            P_h2o = self.P_h2o
        denom = self.dS - self.R * np.log(P_h2o / self.P_ref)
        return self.dH / denom

    # ------------------------------------------------------------------
    # Stored chemical energy = reaction enthalpy x extent
    # ------------------------------------------------------------------
    def stored_energy(self, X):
        """Stored chemical energy [J] at conversion X (= n_mol*dH*X)."""
        return self.n_mol * self.dH * np.clip(X, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Reaction rate dX/dt [1/s]
    # ------------------------------------------------------------------
    def reaction_rate(self, X, T, mode, P_h2o=None):
        """
        Arrhenius rate with equilibrium driving-force gate.

        mode : 'charge'  -> endothermic dehydration, X increases (needs T > T_eq)
               'discharge'-> exothermic hydration,   X decreases (needs T < T_eq)
               'hold'     -> reaction halted, rate = 0 (loss-free storage)
        """
        if mode == "hold":
            return 0.0
        X = min(max(X, 0.0), 1.0)
        Teq = self.T_eq(P_h2o)
        # Driving-force factor in [0,1]: relative temperature offset from eq.
        # Scaled by a thermodynamic window of ~50 K (smoothly gates the reaction).
        window = 50.0
        if mode == "charge":
            if X >= 1.0:
                return 0.0
            drive = (T - Teq) / window           # >0 when above eq (dehydration)
            if drive <= 0.0:
                return 0.0
            f_eq = min(drive, 1.0)
            k = self.A_c * np.exp(-self.Ea_c / (self.R * T))
            return +k * (1.0 - X) * f_eq
        elif mode == "discharge":
            if X <= 0.0:
                return 0.0
            drive = (Teq - T) / window           # >0 when below eq (hydration)
            if drive <= 0.0:
                return 0.0
            f_eq = min(drive, 1.0)
            k = self.A_d * np.exp(-self.Ea_d / (self.R * T))
            return -k * X * f_eq
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, mode, T_source, P_h2o):
        X, T = y
        dXdt = self.reaction_rate(X, T, mode, P_h2o)
        # Chemical heat release rate: discharge (dX/dt<0) releases heat (+).
        Q_rxn = -(self.n_mol * self.dH) * dXdt          # W
        Q_ext = self.hA * (T_source - T)                # W
        dTdt = (Q_rxn + Q_ext) / self.C_th
        return [dXdt, dTdt]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, mode="discharge", X0=None, T0=600.0 + 273.15,
                 T_source=400.0 + 273.15, P_h2o=None, dt=10.0, duration_s=3600.0):
        """
        Simulate reactor dynamics with scipy.solve_ivp.

        Parameters
        ----------
        mode : 'charge' | 'discharge' | 'hold'
        X0 : initial conversion (default: 0.0 for charge, 1.0 for discharge/hold)
        T0 : initial bed temperature [K]
        T_source : HTF / source coupling temperature [K]
        P_h2o : water-vapour pressure [Pa] (default from params)
        dt : output time step [s]
        duration_s : total simulated time [s]

        Returns
        -------
        dict of time-series arrays: t, X, SOC, temperature, T_eq,
            stored_energy_J, reaction_rate, Q_rxn_W, plus scalar E_max_J.
        """
        if P_h2o is None:
            P_h2o = self.P_h2o
        if X0 is None:
            X0 = 0.0 if mode == "charge" else 1.0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [X0, T0],
            t_eval=t_eval, args=(mode, T_source, P_h2o),
            method="LSODA", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t_out = sol.t
        X_out = np.clip(sol.y[0], 0.0, 1.0)
        T_out = sol.y[1]
        N = len(t_out)

        rate = np.zeros(N)
        Q_rxn = np.zeros(N)
        for i in range(N):
            r = self.reaction_rate(X_out[i], T_out[i], mode, P_h2o)
            rate[i] = r
            Q_rxn[i] = -(self.n_mol * self.dH) * r

        return {
            "t": t_out,
            "X": X_out,
            "SOC": X_out,
            "temperature": T_out,
            "T_eq": np.full(N, self.T_eq(P_h2o)),
            "stored_energy_J": self.n_mol * self.dH * X_out,
            "reaction_rate": rate,
            "Q_rxn_W": Q_rxn,
            "E_max_J": self.E_max,
        }
