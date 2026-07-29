"""
EC150 -- Fischer-Tropsch Synthesis (BTL) -- F2a ASF Kinetics + Lumped Thermal ODE

Physics-lumped (0D) model of a low-temperature Fischer-Tropsch reactor (cobalt
catalyst) converting syngas (CO + H2) into hydrocarbons. Three coupled pieces:

1.  CO-conversion kinetics (LHHW / Arrhenius).
    Surface-mechanism rate of CO consumption (Yates & Satterfield 1991, cobalt):

        r_CO = k(T) * P_CO * P_H2 / (1 + K_ads * P_CO)^2        [mol/(kg_cat.s)]
        k(T) = k0 * exp(-Ea / (R T))                             [Arrhenius]

    The reactor is treated as a lumped plug-flow element: CO conversion X is
    obtained by integrating the molar balance along the dimensionless catalyst
    coordinate w in [0, W_cat] (a 1-D quadrature in residence/weight space):

        F_CO0 * dX/dw = r_CO(P_CO(X), P_H2(X), T)

    giving X(T, feed) in (0, 1). This is the 0D/1D first-principles upgrade of
    the F1 fixed-conversion yield model.

2.  Anderson-Schulz-Flory (ASF) product distribution (Anderson 1984).
    Chain growth with probability alpha gives the mole/weight fractions:

        x_n = (1 - alpha) * alpha^(n-1)            (mole fraction, sum = 1)
        W_n = n * (1 - alpha)^2 * alpha^(n-1)      (weight/carbon fraction, sum = 1)

    alpha is made weakly T-dependent (alpha drops as T rises -> lighter
    products), consistent with the empirical LTFT trend (Dry 2002).

3.  Lumped exothermic thermal balance (heat removal critical).
    FT is strongly exothermic (~165 kJ per mol CO). Reactor temperature evolves:

        m * cp * dT/dt = Q_gen - Q_cool
        Q_gen  = dH_rxn * (F_CO0 * X)              [W]   (heat released)
        Q_cool = UA * (T - T_coolant)              [W]   (boiling-water jacket)

    Integrated with scipy.integrate.solve_ivp. Runaway is avoided only when
    cooling can match generation -- the model exposes this stability margin.

Conservation:
  * Carbon: CO converted = carbon entering the hydrocarbon pool = sum_n n*(mol C_n).
  * Mass:   per FT stoichiometry CO + 2 H2 -> (-CH2-) + H2O, the -CH2- mass is
            (14/28) of converted CO mass; O leaves as H2O. Checked in tests.

References:
    Dry, M.E. (2002). "The Fischer-Tropsch process: 1950-2000." Catal. Today 71:227-241.
    Anderson, R.B. (1984). The Fischer-Tropsch Synthesis. Academic Press. (ASF distribution)
    Yates, I.C. & Satterfield, C.N. (1991). Energy & Fuels 5:168-173. (LHHW CO rate, cobalt)
    Steynberg, A. & Dry, M. (2004). Fischer-Tropsch Technology. Elsevier.
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314  # J/(mol.K)

MW_CO = 28.01   # g/mol
MW_H2 = 2.016   # g/mol
MW_CH2 = 14.027  # g/mol  (one -CH2- chain unit)
MW_H2O = 18.015  # g/mol


class FischerTropschF2a:
    """FT synthesis: LHHW CO kinetics + ASF product slate + lumped thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.alpha0   = float(u["alpha_ASF"]["value"])
        self.k0       = float(u["k0_FT"]["value"])           # mol/(kg.s.bar)
        self.Ea       = float(u["Ea_FT"]["value"])           # J/mol
        self.K_ads    = float(u["K_ads"]["value"])           # 1/bar
        self.dH_rxn   = float(u["dH_rxn_per_CO"]["value"])   # J/mol CO
        self.T_nom    = float(u["T_operating_degC"]["value"]) + 273.15
        self.P_nom    = float(u["P_operating_bar"]["value"])  # bar
        self.H2_CO    = float(u["H2_CO_ratio"]["value"])
        self.LHV      = float(u["LHV_FT_liquid_MJ_kg"]["value"])
        self.W_cat    = float(u["W_cat"]["value"])            # kg
        self.m_react  = float(u["m_reactor"]["value"])        # kg
        self.cp_react = float(u["cp_reactor"]["value"])       # J/(kg.K)
        self.UA       = float(u["UA_cool"]["value"])          # W/K
        self.T_cool   = float(u["T_coolant_degC"]["value"]) + 273.15

    # ------------------------------------------------------------------
    # Arrhenius rate constant
    # ------------------------------------------------------------------
    def rate_constant(self, T):
        """Arrhenius rate constant k(T) [mol/(kg_cat.s.bar)]."""
        return self.k0 * np.exp(-self.Ea / (R_GAS * T))

    # ------------------------------------------------------------------
    # LHHW intrinsic CO consumption rate
    # ------------------------------------------------------------------
    def co_rate(self, T, P_CO, P_H2):
        """LHHW CO consumption rate [mol/(kg_cat.s)] (Yates-Satterfield form)."""
        if P_CO <= 0.0 or P_H2 <= 0.0:
            return 0.0
        k = self.rate_constant(T)
        return k * P_CO * P_H2 / (1.0 + self.K_ads * P_CO) ** 2

    # ------------------------------------------------------------------
    # ASF chain-growth probability (mild T dependence)
    # ------------------------------------------------------------------
    def alpha_of_T(self, T):
        """Chain-growth probability alpha(T). Falls ~ as T rises (Dry 2002)."""
        # Linear de-rate around nominal; clamp to a physical FT window.
        a = self.alpha0 - 7.0e-4 * (T - self.T_nom)
        return float(np.clip(a, 0.50, 0.95))

    # ------------------------------------------------------------------
    # ASF distributions
    # ------------------------------------------------------------------
    @staticmethod
    def asf_mole_fraction(alpha, n):
        """Mole fraction of carbon number n: x_n = (1-a) a^(n-1). sum_n = 1."""
        n = np.asarray(n, dtype=float)
        return (1.0 - alpha) * alpha ** (n - 1.0)

    @staticmethod
    def asf_weight_fraction(alpha, n):
        """Weight (carbon) fraction: W_n = n (1-a)^2 a^(n-1). sum_n = 1."""
        n = np.asarray(n, dtype=float)
        return n * (1.0 - alpha) ** 2 * alpha ** (n - 1.0)

    def product_cuts(self, alpha, n_max=80):
        """Lump ASF weight fractions into refinery cuts. Returns dict, sums to ~1."""
        n = np.arange(1, n_max + 1)
        W = self.asf_weight_fraction(alpha, n)
        W = W / W.sum()  # renormalise truncation tail onto resolved cuts
        light = W[(n >= 1) & (n <= 4)].sum()    # C1-C4 light gas
        naphtha = W[(n >= 5) & (n <= 9)].sum()  # C5-C9
        diesel = W[(n >= 10) & (n <= 20)].sum()  # C10-C20
        wax = W[n >= 21].sum()                   # C21+
        return {
            "light_gas_C1_C4": float(light),
            "naphtha_C5_C9": float(naphtha),
            "diesel_C10_C20": float(diesel),
            "wax_C21plus": float(wax),
        }

    @staticmethod
    def c5plus_fraction(alpha, n_max=200):
        """Liquid (C5+) weight fraction from ASF. Analytic-equivalent sum."""
        n = np.arange(1, n_max + 1)
        W = n * (1.0 - alpha) ** 2 * alpha ** (n - 1.0)
        light = W[n <= 4].sum()
        return float(max(0.0, 1.0 - light))

    # ------------------------------------------------------------------
    # CO conversion along lumped plug-flow catalyst coordinate
    # ------------------------------------------------------------------
    def co_conversion(self, T, F_CO0, F_H2_0, P_total):
        """
        Integrate dX/dw = r_CO / F_CO0 over the catalyst weight 0..W_cat.

        Partial pressures track conversion via mole balance:
            CO + 2 H2 -> (-CH2-) + H2O   (consumes 1 CO + 2 H2, makes 1 H2O)

        Returns CO conversion X in (0, 1).
        """
        if F_CO0 <= 0.0:
            return 0.0
        nu_H2 = 2.0  # mol H2 per mol CO

        def dXdw(w, X):
            Xc = float(np.clip(X[0], 0.0, 0.999999))
            F_CO = F_CO0 * (1.0 - Xc)
            F_H2 = F_H2_0 - nu_H2 * F_CO0 * Xc
            F_H2 = max(F_H2, 0.0)
            F_H2O = F_CO0 * Xc
            # Total moles: CO+H2 consumed (3) replaced by H2O (1) -> shrinkage.
            F_tot = F_CO + F_H2 + F_H2O
            if F_tot <= 0.0:
                return [0.0]
            P_CO = P_total * F_CO / F_tot
            P_H2 = P_total * F_H2 / F_tot
            r = self.co_rate(T, P_CO, P_H2)
            return [r / F_CO0]

        # RK45 (explicit) so this inner solve is re-entrant when the outer
        # thermal ODE solver is itself LSODA (LSODA is not re-entrant).
        sol = solve_ivp(
            dXdw, (0.0, self.W_cat), [0.0],
            method="RK45", rtol=1e-7, atol=1e-9, max_step=self.W_cat / 50.0,
        )
        X = float(sol.y[0, -1])
        return float(np.clip(X, 0.0, 0.999))

    # ------------------------------------------------------------------
    # Heat generation and removal
    # ------------------------------------------------------------------
    def heat_generated(self, X, F_CO0):
        """Exothermic heat release [W] = dH_rxn * (mol CO converted / s)."""
        return self.dH_rxn * (F_CO0 * X)

    def heat_removed(self, T):
        """Coolant duty [W] = UA*(T - T_coolant)."""
        return self.UA * (T - self.T_cool)

    def dTdt(self, T, F_CO0, F_H2_0, P_total):
        """Lumped reactor energy balance [K/s]."""
        X = self.co_conversion(T, F_CO0, F_H2_0, P_total)
        Q_gen = self.heat_generated(X, F_CO0)
        Q_cool = self.heat_removed(T)
        return (Q_gen - Q_cool) / (self.m_react * self.cp_react)

    # ------------------------------------------------------------------
    # Steady-state product slate at fixed T (no thermal transient)
    # ------------------------------------------------------------------
    def steady_products(self, T, F_CO0, F_H2_0, P_total):
        """Conversion + ASF product slate + energy at a fixed temperature."""
        X = self.co_conversion(T, F_CO0, F_H2_0, P_total)
        alpha = self.alpha_of_T(T)
        cuts = self.product_cuts(alpha)

        co_conv_mol_s = F_CO0 * X                      # mol CO/s consumed
        # Carbon -> -CH2- mass flow (one C per converted CO).
        ch2_kg_s = co_conv_mol_s * MW_CH2 / 1000.0     # kg/s total HC (-CH2- basis)
        cut_kg_s = {k: v * ch2_kg_s for k, v in cuts.items()}
        liquid_kg_s = cut_kg_s["naphtha_C5_C9"] + cut_kg_s["diesel_C10_C20"] + cut_kg_s["wax_C21plus"]
        energy_MW = liquid_kg_s * self.LHV  # kg/s * MJ/kg = MW

        return {
            "CO_conversion": float(X),
            "alpha": float(alpha),
            "product_cuts": cuts,
            "HC_total_kg_s": float(ch2_kg_s),
            "cut_kg_s": cut_kg_s,
            "liquid_C5plus_kg_s": float(liquid_kg_s),
            "energy_output_MW": float(energy_MW),
            "CO_converted_mol_s": float(co_conv_mol_s),
        }

    # ------------------------------------------------------------------
    # Time-domain simulation with coupled thermal ODE
    # ------------------------------------------------------------------
    def simulate(self, syngas_flow_mol_s, CO_fraction, T0_K, P_total_bar,
                 dt, duration_s):
        """
        Simulate FT reactor thermal transient with conversion + ASF slate.

        Parameters
        ----------
        syngas_flow_mol_s : float or callable(t)  total syngas molar feed [mol/s]
        CO_fraction       : float                 CO mole fraction in syngas [-]
        T0_K              : float                 initial reactor temperature [K]
        P_total_bar       : float                 reactor pressure [bar]
        dt                : float                 output time step [s]
        duration_s        : float                 total duration [s]

        Returns
        -------
        dict of time-series arrays:
            t, temperature, CO_conversion, alpha,
            heat_generated_W, heat_removed_W,
            liquid_C5plus_kg_s, energy_output_MW,
            product_cuts (dict of arrays)
        """
        _Q = (syngas_flow_mol_s if callable(syngas_flow_mol_s)
              else (lambda t: syngas_flow_mol_s))

        def feeds(t):
            Q = max(float(_Q(t)), 0.0)
            F_CO0 = Q * CO_fraction
            F_H2_0 = Q * CO_fraction * self.H2_CO  # H2 fixed by H2:CO ratio vs CO
            return F_CO0, F_H2_0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            F_CO0, F_H2_0 = feeds(t)
            return [self.dTdt(T, F_CO0, F_H2_0, P_total_bar)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T0_K],
            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-7,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        X_arr = np.zeros(N)
        alpha_arr = np.zeros(N)
        Qgen = np.zeros(N)
        Qcool = np.zeros(N)
        liq = np.zeros(N)
        energy = np.zeros(N)
        cuts_arr = {k: np.zeros(N) for k in
                    ["light_gas_C1_C4", "naphtha_C5_C9", "diesel_C10_C20", "wax_C21plus"]}

        for i in range(N):
            T = T_out[i]
            F_CO0, F_H2_0 = feeds(t_out[i])
            res = self.steady_products(T, F_CO0, F_H2_0, P_total_bar)
            X_arr[i] = res["CO_conversion"]
            alpha_arr[i] = res["alpha"]
            Qgen[i] = self.heat_generated(res["CO_conversion"], F_CO0)
            Qcool[i] = self.heat_removed(T)
            liq[i] = res["liquid_C5plus_kg_s"]
            energy[i] = res["energy_output_MW"]
            for k in cuts_arr:
                cuts_arr[k][i] = res["product_cuts"][k]

        return {
            "t": t_out,
            "temperature": T_out,
            "CO_conversion": X_arr,
            "alpha": alpha_arr,
            "heat_generated_W": Qgen,
            "heat_removed_W": Qcool,
            "liquid_C5plus_kg_s": liq,
            "energy_output_MW": energy,
            "product_cuts": cuts_arr,
        }
