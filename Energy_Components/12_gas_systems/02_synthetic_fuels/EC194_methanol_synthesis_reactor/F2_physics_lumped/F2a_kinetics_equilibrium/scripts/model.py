"""
EC194 -- Methanol Synthesis Reactor -- F2a Kinetics + Equilibrium CSTR

Physics-lumped (0D) model of a Cu/ZnO/Al2O3 fixed-bed methanol synthesis
reactor treated as a lumped CSTR with LHHW kinetics, equilibrium-limited
conversion, and an exothermic energy balance with cooling.

Reaction network (CO/CO2 hydrogenation over Cu/ZnO/Al2O3):

    R1  CO2 hydrogenation : CO2 + 3 H2  <-> CH3OH + H2O   dH = -49.5 kJ/mol
    R2  reverse water-gas-shift (RWGS) : CO2 + H2 <-> CO + H2O   dH = +41.2 kJ/mol
    (CO hydrogenation, CO + 2 H2 <-> CH3OH, is the linear combination R1 - R2,
     dH = -90.7 kJ/mol; it is captured implicitly through R1 and R2.)

Kinetics -- Vanden Bussche & Froment (1996) LHHW rate expressions
(the modern Graaf-family low-pressure Cu/ZnO/Al2O3 mechanism). Both rates
are driven by the distance from chemical equilibrium (1 - Q/K), so a
positive forward rate is impossible past equilibrium -- conversion is
intrinsically equilibrium-limited:

    r_MeOH = k1 * P_CO2 * P_H2 * (1 - (P_H2O*P_CH3OH)/(K1 * P_H2^3 * P_CO2))
             / DEN^3
    r_RWGS = k2 * P_CO2 * (1 - K2 * (P_H2O*P_CO)/(P_CO2*P_H2))
             / DEN

    DEN = 1 + Kc*(P_H2O/P_H2) + sqrt(Ka*P_H2) + Kb*P_H2O

with k_i, K_a..c, and equilibrium constants K1, K2 from Arrhenius/van't Hoff
forms (Graaf 1986 thermodynamics, Vanden Bussche & Froment 1996 Table).

Lumped CSTR balances (state = species concentrations + reactor temperature):

    dC_i/dt = (C_i_in - C_i)/tau + sum_j nu_ij * r_j * rho_cat_eff
    (m*cp) dT/dt = sum_j (-dH_j) r_j V rho_cat_eff
                   - UA (T - T_cool) + F_in cp_gas (T_in - T)

Tracked outputs: per-pass CO_x conversion, methanol yield / dry mole
fraction, reactor temperature, equilibrium conversion at the final T.

References:
    Graaf, G.H., Sijtsema, P.J.J.M., Stamhuis, E.J., Joosten, G.E.H. (1986).
        Chem. Eng. Sci. 41(11), 2883-2890.  (chemical equilibria CH3OH synthesis)
    Graaf, G.H., Stamhuis, E.J., Beenackers, A.A.C.M. (1988).
        Chem. Eng. Sci. 43(12), 3185-3195. (kinetics of low-pressure synthesis)
    Vanden Bussche, K.M. & Froment, G.F. (1996). J. Catal. 161(1), 1-10.
        (steady-state kinetic LHHW model for CH3OH synthesis & RWGS on Cu/ZnO/Al2O3)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

R_GAS = 8.314  # J/(mol.K)


class MethanolReactor_F2a:
    """Cu/ZnO/Al2O3 methanol synthesis CSTR with VB&F LHHW kinetics + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V = u["V_reactor"]["value"]            # m3
        self.rho_cat = u["rho_cat"]["value"]         # kg/m3
        self.cat_frac = u["cat_fraction"]["value"]   # -
        self.UA = u["UA"]["value"]                   # W/K
        self.T_cool = u["T_cool"]["value"]           # K
        self.m_th = u["m_thermal"]["value"]          # kg
        self.cp_th = u["cp_thermal"]["value"]        # J/(kg.K)
        self.T_in = u["T_in"]["value"]               # K
        self.P = u["P_total"]["value"]               # bar
        self.GHSV = u["GHSV"]["value"]               # 1/h
        self.cp_gas = u["cp_gas"]["value"]           # J/(mol.K)

        # Feed composition (mole fractions): CO, CO2, H2, CH3OH, H2O, inert
        fc = u["feed_fractions"]["value"]
        self.y_CO_in = fc["CO"]
        self.y_CO2_in = fc["CO2"]
        self.y_H2_in = fc["H2"]
        self.y_inert_in = fc.get("inert", 0.0)

        # Reaction enthalpies [J/mol]
        self.dH_MeOH = u["dH_MeOH"]["value"]   # CO2 hydrogenation (R1), exothermic
        self.dH_RWGS = u["dH_RWGS"]["value"]   # RWGS (R2), endothermic

        # VB&F (1996) kinetic / adsorption parameters: A (pre-exp) + E or dS,dH
        kp = u["kinetics"]["value"]
        self.k1A, self.k1E = kp["k1_A"], kp["k1_E"]        # MeOH rate constant
        self.k2A, self.k2E = kp["k2_A"], kp["k2_E"]        # RWGS rate constant
        self.KaA, self.KaE = kp["Ka_A"], kp["Ka_E"]        # sqrt(K_H2) group
        self.KbA, self.KbE = kp["Kb_A"], kp["Kb_E"]        # K_H2O group
        self.KcA, self.KcE = kp["Kc_A"], kp["Kc_E"]        # K_H2O/K_H2 group

        # Derived
        self.m_cat = self.rho_cat * self.cat_frac * self.V  # kg catalyst
        self.tau = 3600.0 / self.GHSV                        # residence time [s]

    # ------------------------------------------------------------------
    # Arrhenius / van't Hoff helper:  A * exp(B / (R*T))
    # Following VB&F (1996) Table 1 sign convention: B is the listed value,
    # positive B => lumped constant rises with T (k1), negative B => falls (k2).
    # ------------------------------------------------------------------
    @staticmethod
    def _arr(A, B, T):
        return A * np.exp(B / (R_GAS * T))

    # ------------------------------------------------------------------
    # Equilibrium constants (Graaf 1986; Pa-based forms, converted to bar)
    # ------------------------------------------------------------------
    @staticmethod
    def K1_MeOH(T):
        """K_p for CO2 + 3H2 <-> CH3OH + H2O, units bar^-2.

        Combined from the two Graaf (1986) correlations:
          log10 K_p(CO+2H2<->CH3OH)        = 5139/T - 12.621   (bar^-2)
          log10 K_p(CO2+H2<->CO+H2O, RWGS) = -2073/T + 2.029   (dimensionless)
        K1 = K_CO_hydro / K_RWGS.
        """
        logK_CO = 5139.0 / T - 12.621
        logK_RWGS = -2073.0 / T + 2.029
        return 10.0 ** (logK_CO - logK_RWGS)

    @staticmethod
    def K2_RWGS(T):
        """K_p for RWGS  CO2 + H2 <-> CO + H2O (dimensionless), Graaf (1986)."""
        logK_RWGS = -2073.0 / T + 2.029
        return 10.0 ** logK_RWGS

    # ------------------------------------------------------------------
    # LHHW rate expressions -- Vanden Bussche & Froment (1996)
    # Partial pressures in bar; rates in mol/(kg_cat.s)
    # ------------------------------------------------------------------
    def _den(self, T, P_H2, P_H2O):
        Ka = self._arr(self.KaA, self.KaE, T)
        Kb = self._arr(self.KbA, self.KbE, T)
        Kc = self._arr(self.KcA, self.KcE, T)
        P_H2 = max(P_H2, 1e-9)
        P_H2O = max(P_H2O, 0.0)
        return 1.0 + Kc * (P_H2O / P_H2) + np.sqrt(Ka * P_H2) + Kb * P_H2O

    def rate_meoh(self, T, P_CO2, P_H2, P_CH3OH, P_H2O):
        """CO2 hydrogenation -> methanol rate [mol/(kg_cat.s)]."""
        P_CO2 = max(P_CO2, 0.0)
        P_H2 = max(P_H2, 1e-9)
        k1 = self._arr(self.k1A, self.k1E, T)
        K1 = self.K1_MeOH(T)
        beta = (P_H2O * P_CH3OH) / (K1 * P_H2**3 * P_CO2 + 1e-30)  # Q/K
        beta = min(beta, 1.0)  # cannot exceed equilibrium (no negative forward rate)
        den = self._den(T, P_H2, P_H2O)
        return k1 * P_CO2 * P_H2 * (1.0 - beta) / den**3

    def rate_rwgs(self, T, P_CO, P_CO2, P_H2, P_H2O):
        """Reverse water-gas-shift rate [mol/(kg_cat.s)] (may be negative => WGS)."""
        P_CO2 = max(P_CO2, 0.0)
        P_H2 = max(P_H2, 1e-9)
        k2 = self._arr(self.k2A, self.k2E, T)
        K2 = self.K2_RWGS(T)  # equilibrium constant for CO2+H2 <-> CO+H2O
        # reaction quotient Q = (P_CO*P_H2O)/(P_CO2*P_H2); driving force (1 - Q/K2)
        gamma = (P_H2O * P_CO) / (P_CO2 * P_H2 + 1e-30) / (K2 + 1e-30)
        den = self._den(T, P_H2, P_H2O)
        return k2 * P_CO2 * (1.0 - gamma) / den

    # ------------------------------------------------------------------
    # Equilibrium CO_x -> MeOH conversion (single CO2-hydrogenation extent)
    # ------------------------------------------------------------------
    def equilibrium_conversion(self, T, P=None):
        """Equilibrium per-pass conversion of CO2 via CO2 + 3H2 <-> CH3OH + H2O.

        Stoichiometric basis: feed 1 mol CO2 + 3 mol H2 (RWGS neglected for this
        thermodynamic limit -> conservative MeOH-only equilibrium extent).
        """
        P = P if P is not None else self.P
        K1 = self.K1_MeOH(T)  # bar^-2

        def eq_func(X):
            if X <= 0 or X >= 1:
                return 1e10
            # CO2: 1-X, H2: 3-3X, CH3OH: X, H2O: X ; total = 4 - 2X
            n_tot = 4.0 - 2.0 * X
            y_CO2 = (1.0 - X) / n_tot
            y_H2 = (3.0 - 3.0 * X) / n_tot
            y_MeOH = X / n_tot
            y_H2O = X / n_tot
            # K_p = (y_MeOH*y_H2O)/(y_CO2*y_H2^3) * P^(2-4) = ... * P^-2
            Kp_calc = (y_MeOH * y_H2O) / (y_CO2 * y_H2**3 + 1e-30) * P**(-2)
            return Kp_calc - K1

        try:
            return brentq(eq_func, 1e-6, 1.0 - 1e-6)
        except (ValueError, RuntimeError):
            # endpoints same sign: low T => near complete, high T => near zero
            return 0.99 if eq_func(1e-6) < 0 else 1e-4

    # ------------------------------------------------------------------
    # Lumped CSTR dynamics
    # State y = [C_CO, C_CO2, C_H2, C_CH3OH, C_H2O, T]   (C in mol/m3, T in K)
    # ------------------------------------------------------------------
    def simulate(self, T0=None, duration_s=600.0, dt=1.0, P=None, GHSV=None,
                 T_in=None, T_cool=None):
        T0 = T0 if T0 is not None else self.T_in
        P = P if P is not None else self.P
        GHSV = GHSV if GHSV is not None else self.GHSV
        T_in = T_in if T_in is not None else self.T_in
        T_cool = T_cool if T_cool is not None else self.T_cool
        tau = 3600.0 / GHSV

        # Feed concentrations [mol/m3] from ideal gas at inlet T
        C_total_in = (P * 1e5) / (R_GAS * T_in)
        C_CO_in = C_total_in * self.y_CO_in
        C_CO2_in = C_total_in * self.y_CO2_in
        C_H2_in = C_total_in * self.y_H2_in
        C_MeOH_in = 0.0
        C_H2O_in = 0.0
        C_inert = C_total_in * self.y_inert_in  # tracked only in DEN/total

        F_mol_in = C_total_in * self.V / tau     # total molar feed [mol/s]
        rho_cat_eff = self.m_cat / self.V         # kg_cat / m3_reactor

        carbon_in = C_CO_in + C_CO2_in  # for conversion reference

        def rhs(t, y):
            C_CO, C_CO2, C_H2, C_MeOH, C_H2O, T = y
            C_CO = max(C_CO, 0.0)
            C_CO2 = max(C_CO2, 0.0)
            C_H2 = max(C_H2, 1e-12)
            C_MeOH = max(C_MeOH, 0.0)
            C_H2O = max(C_H2O, 0.0)

            C_sum = C_CO + C_CO2 + C_H2 + C_MeOH + C_H2O + C_inert + 1e-12
            P_CO = P * C_CO / C_sum
            P_CO2 = P * C_CO2 / C_sum
            P_H2 = P * C_H2 / C_sum
            P_MeOH = P * C_MeOH / C_sum
            P_H2O = P * C_H2O / C_sum

            r1 = self.rate_meoh(T, P_CO2, P_H2, P_MeOH, P_H2O)         # R1
            r2 = self.rate_rwgs(T, P_CO, P_CO2, P_H2, P_H2O)          # R2

            rc1 = r1 * rho_cat_eff
            rc2 = r2 * rho_cat_eff

            # Stoichiometry
            # R1: CO2 + 3H2 -> CH3OH + H2O
            # R2: CO2 + H2  -> CO + H2O
            dCO   = (C_CO_in   - C_CO)   / tau + rc2
            dCO2  = (C_CO2_in  - C_CO2)  / tau - rc1 - rc2
            dH2   = (C_H2_in   - C_H2)   / tau - 3.0 * rc1 - rc2
            dMeOH = (C_MeOH_in - C_MeOH) / tau + rc1
            dH2O  = (C_H2O_in  - C_H2O)  / tau + rc1 + rc2

            # Energy balance
            Q_rxn = ((-self.dH_MeOH) * r1 + (-self.dH_RWGS) * r2) * self.m_cat  # W
            Q_cool = self.UA * (T - T_cool)
            Q_flow = F_mol_in * self.cp_gas * (T_in - T)
            dT = (Q_rxn - Q_cool + Q_flow) / (self.m_th * self.cp_th)

            return [dCO, dCO2, dH2, dMeOH, dH2O, dT]

        y0 = [C_CO_in, C_CO2_in, C_H2_in, 0.0, 0.0, T0]
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="BDF", rtol=1e-7, atol=1e-9,
            max_step=dt * 10,
        )

        C_CO = np.maximum(sol.y[0], 0.0)
        C_CO2 = np.maximum(sol.y[1], 0.0)
        C_H2 = np.maximum(sol.y[2], 0.0)
        C_MeOH = np.maximum(sol.y[3], 0.0)
        C_H2O = np.maximum(sol.y[4], 0.0)
        T_out = sol.y[5]

        # Per-pass carbon (CO_x) conversion = carbon converted to MeOH / carbon in
        X_C = C_MeOH / (carbon_in + 1e-30)
        X_C = np.clip(X_C, 0.0, 1.0)

        # Methanol yield (mol MeOH per mol carbon fed) == X_C here; plus dry frac
        C_total = C_CO + C_CO2 + C_H2 + C_MeOH + C_H2O + C_inert + 1e-12
        y_MeOH_wet = C_MeOH / C_total
        y_MeOH_dry = C_MeOH / (C_total - C_H2O + 1e-12)

        T_max = float(np.max(T_out))
        thermal_runaway = bool(T_max > T_in + 150.0)
        X_eq_final = self.equilibrium_conversion(T_out[-1], P)

        return {
            "t": sol.t,
            "T": T_out,
            "C_CO": C_CO,
            "C_CO2": C_CO2,
            "C_H2": C_H2,
            "C_CH3OH": C_MeOH,
            "C_H2O": C_H2O,
            "X_C": X_C,
            "meoh_yield": X_C,
            "y_MeOH_wet": y_MeOH_wet,
            "y_MeOH_dry": y_MeOH_dry,
            "T_max": T_max,
            "thermal_runaway": thermal_runaway,
            "X_eq_final": X_eq_final,
            "carbon_in": carbon_in,
            "C_CO_in": C_CO_in,
            "C_CO2_in": C_CO2_in,
            "C_H2_in": C_H2_in,
        }

    # ------------------------------------------------------------------
    # Steady-state conversion vs temperature sweep (kinetic vs equilibrium)
    # ------------------------------------------------------------------
    def conversion_vs_temperature(self, T_range=None, P=None, GHSV=None):
        if T_range is None:
            T_range = np.linspace(473.15, 573.15, 25)
        P = P if P is not None else self.P
        GHSV = GHSV if GHSV is not None else self.GHSV

        X_kin, X_eq = [], []
        for T in T_range:
            r = self.simulate(T0=T, duration_s=3000.0, dt=100.0, P=P, GHSV=GHSV,
                              T_in=T, T_cool=T)
            X_kin.append(r["X_C"][-1])
            X_eq.append(self.equilibrium_conversion(T, P))
        return T_range, np.array(X_kin), np.array(X_eq)
