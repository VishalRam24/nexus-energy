"""
EC206 -- CO2 Mineralization (Mineral Carbonation) -- F2a Carbonation Kinetics

Physics-lumped 0D batch-reactor model: CO2 reacts with a Mg-silicate
(forsterite olivine) to form a permanent carbonate, coupling a shrinking-core
conversion ODE to a reactor energy-balance ODE, integrated with
scipy.integrate.solve_ivp.

Carbonation reaction (permanent storage):
    Mg2SiO4 + 2 CO2  ->  2 MgCO3 + SiO2          (forsterite)
    -> 2 mol CO2 permanently bound as carbonate per mol mineral.

Shrinking-core kinetics (surface-reaction controlled):
    A spherical grain of unreacted mineral shrinks as a carbonate/silica rim
    grows. For surface-reaction control (Levenspiel 1999, "Chemical Reaction
    Engineering", ch. 25), the unreacted-core radius r_c obeys

        dr_c/dt = - k_s * C_drive / (rho_mol)            (m/s)

    and the conversion of a sphere of initial radius R0 is

        X = 1 - (r_c / R0)^3 .

    Equivalently, the standard SCM surface-reaction conversion law

        1 - (1 - X)^(1/3) = (k_s * C_drive / (rho_mol * R0)) * t

    gives the rate

        dX/dt = 3 * (k_s * C_drive) / (rho_mol * R0) * (1 - X)^(2/3) .

    Smaller particle radius R0 -> faster conversion (matches O'Connor 2005:
    fine -400 mesh grind required). The (1-X)^(2/3) factor makes the reaction
    self-decelerate as the core shrinks (slow late-stage kinetics).

Rate constant (Arrhenius) and driving force:
    k_s = A * exp(-E_act / (R T))                          (m/s)
    C_drive = (P_CO2 / P_ref) ** order                     (-)  (dissolved-CO2
              supply proxy, first order in CO2 partial pressure)
    so the effective rate constant rises with T (Arrhenius) and with CO2
    pressure -- both consistent with the aqueous NETL route operating at
    ~185 C and ~115-150 atm (O'Connor et al. 2005).

Energy balance (lumped, exothermic):
    The carbonation is strongly exothermic (dH < 0). Heat released per unit
    time = (-dH_rxn) * (mol CO2 bound / s). Reactor slurry of mass m_slurry
    and heat capacity cp obeys

        m_slurry * cp * dT/dt = Q_rxn - Q_cool
        Q_rxn  = (-dH_rxn) * stoich * n_mineral0 * dX/dt
        Q_cool = hA * (T - T_coolant)

References:
    Lackner, K.S. et al. (1995). Carbon dioxide disposal in carbonate
        minerals. Energy 20(11):1153-1170.   (thermodynamics, dH, permanence)
    O'Connor, W.K. et al. (2005). Aqueous mineral carbonation. NETL/ARC
        DOE/ARC-TR-04-002.                    (kinetics, T/P/particle-size data)
    Sanna, A. et al. (2014). A review of mineral carbonation technologies to
        sequester CO2. Chem. Soc. Rev. 43:8049-8080.
    Levenspiel, O. (1999). Chemical Reaction Engineering, 3rd ed., Wiley
        (shrinking-core model).
"""

import numpy as np
from scipy.integrate import solve_ivp


class CO2Mineralization_F2a:
    """Mineral carbonation -- shrinking-core conversion + reactor energy ODE."""

    R = 8.314          # J/(mol.K) universal gas constant

    def __init__(self, params: dict):
        u = params["unit"]
        self.stoich = u["stoich_CO2_per_mineral"]["value"]      # mol CO2 / mol mineral
        self.M_mineral = u["M_mineral_kg_mol"]["value"]         # kg/mol
        self.M_CO2 = u["M_CO2_kg_mol"]["value"]                 # kg/mol
        self.rho = u["rho_mineral_kg_m3"]["value"]              # kg/m3
        self.A_pre = u["A_pre_1_s"]["value"]                    # 1/s (see note)
        self.E_act = u["E_act_J_mol"]["value"]                  # J/mol
        self.order = u["reaction_order_P"]["value"]             # -
        self.P_ref = u["P_ref_atm"]["value"]                    # atm
        self.dH = u["dH_rxn_J_molCO2"]["value"]                 # J/mol CO2 (<0)
        self.R0_ref = u["particle_radius_ref_m"]["value"]       # m
        self.m_mineral = u["m_mineral_kg"]["value"]             # kg
        self.m_slurry = u["slurry_mass_kg"]["value"]            # kg
        self.cp = u["cp_slurry_J_kgK"]["value"]                 # J/(kg.K)
        self.hA = u["hA_cool_W_K"]["value"]                     # W/K
        self.T_coolant = u["T_coolant_K"]["value"]              # K
        self.T_ref = u["T_ref_K"]["value"]                      # K

        # molar inventory of mineral in the charge
        self.n_mineral0 = self.m_mineral / self.M_mineral       # mol
        # molar density of mineral (mol/m3) for the SCM core-shrink rate
        self.rho_mol = self.rho / self.M_mineral                # mol/m3

    # --- kinetics ----------------------------------------------------------
    def rate_constant(self, T):
        """Arrhenius surface rate constant pre-factor [1/s]."""
        return self.A_pre * np.exp(-self.E_act / (self.R * T))

    def driving_force(self, P_CO2_atm):
        """Dimensionless CO2 supply driving force (first order in P_CO2)."""
        return (P_CO2_atm / self.P_ref) ** self.order

    def dXdt(self, X, T, P_CO2_atm, R0):
        """Shrinking-core surface-reaction conversion rate dX/dt [1/s].

        dX/dt = 3 * keff * (R0_ref / R0) * (1 - X)^(2/3)
        where keff = k_s(T) * C_drive(P) absorbs the geometric 1/R0_ref scale
        into A_pre, so the explicit (R0_ref/R0) captures particle-size effect:
        halving the radius doubles the rate.
        """
        X = np.clip(X, 0.0, 1.0)
        keff = self.rate_constant(T) * self.driving_force(P_CO2_atm)
        return 3.0 * keff * (self.R0_ref / R0) * (1.0 - X) ** (2.0 / 3.0)

    # --- reactor ODE -------------------------------------------------------
    def _rhs(self, t, y, P_CO2_atm, R0):
        X, T = y
        dX = self.dXdt(X, T, P_CO2_atm, R0)
        # mol CO2 bound per second = stoich * n_mineral0 * dX/dt
        co2_rate_mol_s = self.stoich * self.n_mineral0 * dX
        Q_rxn = (-self.dH) * co2_rate_mol_s          # W (exothermic, dH<0)
        Q_cool = self.hA * (T - self.T_coolant)      # W
        dT = (Q_rxn - Q_cool) / (self.m_slurry * self.cp)
        return [dX, dT]

    def simulate(self, T0=458.15, P_CO2_atm=115.0, R0=None,
                 dt=30.0, duration_s=3600.0):
        """Integrate conversion + temperature ODE over the batch.

        Returns dict of time-series arrays + derived quantities.
        """
        if R0 is None:
            R0 = self.R0_ref
        n_steps = int(round(duration_s / dt))
        t_eval = np.linspace(0.0, duration_s, n_steps + 1)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [0.0, T0],
            t_eval=t_eval, args=(P_CO2_atm, R0),
            method="LSODA", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        X = np.clip(sol.y[0], 0.0, 1.0)
        T = sol.y[1]

        # CO2 permanently bound [kg]: stoich * mol_mineral_reacted * M_CO2
        co2_bound_mol = self.stoich * self.n_mineral0 * X
        co2_bound_kg = co2_bound_mol * self.M_CO2
        # carbonate (MgCO3, M=0.0843 kg/mol) produced [kg] = mol CO2 * M_MgCO3
        M_MgCO3 = 0.08431
        carbonate_kg = co2_bound_mol * M_MgCO3
        # cumulative heat released [J]
        heat_released_J = (-self.dH) * co2_bound_mol

        return {
            "t": t,
            "conversion": X,
            "temperature": T,
            "co2_bound_kg": co2_bound_kg,
            "carbonate_kg": carbonate_kg,
            "heat_released_J": heat_released_J,
            "co2_rate_mol_s": self.stoich * self.n_mineral0
                              * np.array([self.dXdt(xi, Ti, P_CO2_atm, R0)
                                          for xi, Ti in zip(X, T)]),
        }

    # --- closed-form SCM (for cross-check / fast eval) ---------------------
    def conversion_analytic(self, T, P_CO2_atm, R0, t):
        """Closed-form isothermal shrinking-core conversion at time t [s].

        1 - (1-X)^(1/3) = keff * (R0_ref/R0) * t   (capped at X=1).
        """
        keff = self.rate_constant(T) * self.driving_force(P_CO2_atm)
        g = keff * (self.R0_ref / R0) * np.asarray(t, dtype=float)
        g = np.clip(g, 0.0, 1.0)
        return 1.0 - (1.0 - g) ** 3
