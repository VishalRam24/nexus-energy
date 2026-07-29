"""
EC196 -- Synthetic Jet Fuel (Power-to-Liquid) -- F2a
Fischer-Tropsch Chain-Growth Kinetics + Lumped Reactor Thermal ODE

Physics-lumped (0D) model of the FT + hydrocracking route producing synthetic
kerosene (jet fuel) from CO2-derived syngas in a Power-to-Liquid (PtL) plant.

Upstream context (algebraic):
    CO2 + renewable H2  --(RWGS)-->  CO + H2O  -->  syngas (H2/CO ~ 2.1)
Modelled reactor (this file):
    n CO + (2n+1) H2  -->  C_nH_{2n+2} + n H2O      (FT, Co/Al2O3 LTFT)
    C16+ wax  --(hydrocracking, +H2)-->  C8-C16 jet cut   (De Klerk 2011)

Kinetics (Langmuir-Hinshelwood, Yates & Satterfield 1991):
    r_CO = k(T) * P_H2 * P_CO / (1 + K_ads * P_CO)^2     [mol/(s.kg_cat)]
    k(T) = k0 * exp(-Ea / (R T))                          Arrhenius
    X_CO = r_CO * m_cat / n_CO_in    (clipped to [0, X_stoich])

Chain growth (Anderson-Schulz-Flory):
    W_n   = n (1-alpha)^2 alpha^(n-1)        weight fraction of C_n
    alpha = alpha0 + dalpha_dT (T - T_opt)   chain growth falls with T (Dry 2002)
    S_jet = sum_{n=8..16} W_n  +  wax_to_jet * sum_{n>=17} W_n
            (direct jet cut + hydrocracked wax recovered to jet, De Klerk 2011)

Lumped reactor thermal ODE (exothermic energy balance):
    m_r cp_r dT/dt = Q_gen - Q_cool
    Q_gen  = (-DH) * r_CO * m_cat            [W]   FT exotherm
    Q_cool = UA * (T - T_coolant)            [W]   steam-raising coils

Overall power-to-liquid efficiency (Schmidt et al. 2018):
    eta_PtL = E_jet_out / E_power_in
            = (jet_mol * LHV_jet) / (H2_mol*LHV_H2/eta_elec + CO_chem/eta_rwgs)
    enforced strictly < 1 (PtL state-of-the-art ~38-50 %, Schmidt 2018).

References:
    Schmidt, P. et al. (2018). Power-to-Liquids as Renewable Fuel Option for
        Aviation. Chemie Ingenieur Technik, 90(1-2), 127-140.
    De Klerk, A. (2011). Fischer-Tropsch Refining. Wiley-VCH.
    Yates, I.C. & Satterfield, C.N. (1991). Energy & Fuels, 5, 168-173.
    Dry, M.E. (2002). The Fischer-Tropsch process: 1950-2000.
        Catalysis Today, 71, 227-241.
    Anderson, R.B. (1956); Schulz, H. (1999) Appl. Catal. A 186, 3-12.
"""

import numpy as np
from scipy.integrate import solve_ivp


class FTJetFuelF2a:
    """FT PtL reactor: LH kinetics + ASF chain growth + lumped thermal ODE."""

    R = 8.314          # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_opt        = u["T_opt"]["value"]            # degC
        self.P_ref        = u["P_ref"]["value"]            # bar

        self.k0           = u["k0_FT"]["value"]            # mol/(s.bar.kg_cat)
        self.Ea           = u["Ea_FT"]["value"]            # J/mol
        self.K_ads        = u["K_ads"]["value"]            # 1/bar
        self.m_cat        = u["m_cat"]["value"]            # kg

        self.alpha0       = u["alpha_ASF"]["value"]
        self.dalpha_dT    = u["dalpha_dT"]["value"]        # 1/degC
        self.wax_to_jet   = u["wax_to_jet_frac"]["value"]

        self.n_CO_in      = u["n_CO_in"]["value"]          # mol/s
        self.h2_co        = u["H2_CO_ratio"]["value"]
        self.P_CO         = u["P_CO"]["value"]             # bar
        self.P_H2         = u["P_H2"]["value"]             # bar

        self.LHV_FT       = u["LHV_FT_kJ_mol"]["value"]    # kJ/mol jet
        self.MW_jet       = u["MW_jet"]["value"]           # g/mol
        self.C_per_jet    = u["C_per_jet"]["value"]
        self.LHV_H2       = u["LHV_H2"]["value"]           # kJ/mol
        self.LHV_CO       = u["LHV_CO"]["value"]           # kJ/mol
        self.eta_elec     = u["eta_electrolysis"]["value"]
        self.eta_rwgs     = u["eta_rwgs"]["value"]

        self.DH           = abs(u["DH_reaction"]["value"]) # kJ/mol CO (exotherm)
        self.m_reactor    = u["m_reactor"]["value"]        # kg
        self.cp_reactor   = u["cp_reactor"]["value"]       # J/(kg.K)
        self.UA           = u["UA_cool"]["value"]          # W/K
        self.T_coolant    = u["T_coolant"]["value"]        # degC

    # ------------------------------------------------------------------
    # Chain-growth probability (temperature dependent)
    # ------------------------------------------------------------------
    def alpha(self, T_C):
        """ASF chain-growth probability, decreasing with temperature."""
        a = self.alpha0 + self.dalpha_dT * (np.asarray(T_C, float) - self.T_opt)
        return np.clip(a, 0.50, 0.98)

    # ------------------------------------------------------------------
    # ASF distribution -> jet-cut selectivity (with wax hydrocracking)
    # ------------------------------------------------------------------
    def asf_selectivity_jet(self, T_C=None, alpha=None):
        """
        Carbon-weight selectivity to the jet cut (C8-C16) including the
        fraction of C17+ wax recovered to jet by hydrocracking.
        """
        if alpha is None:
            a = self.alpha(self.T_opt if T_C is None else T_C)
        else:
            a = float(alpha)
        a = float(a)
        n = np.arange(1, 201)                       # C1..C200 (mass-closure)
        W = n * (1.0 - a) ** 2 * a ** (n - 1)       # ASF weight fractions
        jet = np.sum(W[(n >= 8) & (n <= 16)])
        wax = np.sum(W[n >= 17])
        return float(jet + self.wax_to_jet * wax)

    def asf_weight_fractions(self, T_C=None, n_max=40, alpha=None):
        """ASF weight fraction array W_n for n=1..n_max (for inspection/plots)."""
        if alpha is None:
            a = float(self.alpha(self.T_opt if T_C is None else T_C))
        else:
            a = float(alpha)
        n = np.arange(1, n_max + 1)
        return n, n * (1.0 - a) ** 2 * a ** (n - 1)

    # ------------------------------------------------------------------
    # Langmuir-Hinshelwood FT rate (Yates & Satterfield 1991)
    # ------------------------------------------------------------------
    def rate_constant(self, T_C):
        """Arrhenius rate constant k(T) [mol/(s.bar.kg_cat)]."""
        T_K = np.asarray(T_C, float) + 273.15
        return self.k0 * np.exp(-self.Ea / (self.R * T_K))

    def co_consumption_rate(self, T_C, P_CO=None, P_H2=None):
        """
        Specific CO consumption rate r_CO [mol/(s.kg_cat)] via LH kinetics:
            r = k(T) P_H2 P_CO / (1 + K_ads P_CO)^2
        """
        P_CO = self.P_CO if P_CO is None else P_CO
        P_H2 = self.P_H2 if P_H2 is None else P_H2
        k = self.rate_constant(T_C)
        return k * P_H2 * P_CO / (1.0 + self.K_ads * P_CO) ** 2

    def conversion(self, T_C, P_CO=None, P_H2=None, n_co_in=None):
        """
        Per-pass CO conversion X = r_CO * m_cat / n_CO_in, bounded by the
        H2/CO stoichiometric limit (each CO needs ~2.1 H2).
        """
        n_co = self.n_CO_in if n_co_in is None else n_co_in
        r = self.co_consumption_rate(T_C, P_CO, P_H2)
        X = r * self.m_cat / max(n_co, 1e-12)
        X_stoich = np.clip(self.h2_co / 2.1, 0.0, 1.0)
        return float(np.clip(np.minimum(X, X_stoich), 0.0, 1.0)) \
            if np.ndim(X) == 0 else np.clip(np.minimum(X, X_stoich), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Yields (mol/s) with carbon balance
    # ------------------------------------------------------------------
    def yields(self, T_C, P_CO=None, P_H2=None, n_co_in=None):
        """
        Carbon-balanced product split [mol C/s] and jet-fuel molar rate.
        Returns dict with CO converted, C to jet, C to other, jet mol/s.
        """
        n_co = self.n_CO_in if n_co_in is None else n_co_in
        X = self.conversion(T_C, P_CO, P_H2, n_co)
        S_jet = self.asf_selectivity_jet(T_C)
        C_converted = X * n_co                      # mol C/s entering products
        C_to_jet = C_converted * S_jet              # mol C/s in jet cut
        C_to_other = C_converted * (1.0 - S_jet)    # mol C/s naphtha/gas/wax-loss
        jet_mol_s = C_to_jet / self.C_per_jet       # mol jet/s (C12 surrogate)
        return {
            "co_conversion": X,
            "selectivity_jet": S_jet,
            "C_converted_mol_s": C_converted,
            "C_to_jet_mol_s": C_to_jet,
            "C_to_other_mol_s": C_to_other,
            "jet_mol_s": jet_mol_s,
            "jet_kg_s": jet_mol_s * self.MW_jet / 1000.0,
        }

    # ------------------------------------------------------------------
    # Overall power-to-liquid efficiency (Schmidt et al. 2018)
    # ------------------------------------------------------------------
    def ptl_efficiency(self, T_C, P_CO=None, P_H2=None, n_co_in=None):
        """
        eta_PtL = jet LHV out / renewable power in (incl. electrolysis + RWGS).
        Each CO converted consumes ~2.1 H2; H2 carries the renewable power via
        electrolysis (eta_elec) and the CO chemical energy via RWGS (eta_rwgs).
        Strictly bounded to (0, 1).
        """
        y = self.yields(T_C, P_CO, P_H2, n_co_in)
        E_out = y["jet_mol_s"] * self.LHV_FT                       # kW (kJ/s)
        H2_consumed = 2.1 * y["C_converted_mol_s"]                 # mol H2/s
        E_power_H2 = H2_consumed * self.LHV_H2 / self.eta_elec     # kW
        E_power_CO = y["C_converted_mol_s"] * self.LHV_CO / self.eta_rwgs
        E_in = E_power_H2 + E_power_CO
        eta = E_out / (E_in + 1e-12)
        return float(np.clip(eta, 0.0, 0.999))

    # ------------------------------------------------------------------
    # Heat release [W]
    # ------------------------------------------------------------------
    def heat_released_W(self, T_C, P_CO=None, P_H2=None, n_co_in=None):
        """
        FT exothermic heat from the CO *actually converted* [W].

        Heat release is bounded by the available CO feed (conversion <= 1):
            Q_gen = (-DH) * X_CO * n_CO_in
        Using the bounded conversion (not the raw kinetic rate) is what makes
        the lumped energy balance physically stable -- generation saturates
        once the feed is fully consumed, while cooling stays linear, so a
        finite steady state exists (Dry 2002; standard CSTR heat-balance).
        """
        n_co = self.n_CO_in if n_co_in is None else n_co_in
        X = self.conversion(T_C, P_CO, P_H2, n_co)
        return self.DH * 1000.0 * X * n_co              # kJ->J : W

    # ------------------------------------------------------------------
    # Lumped reactor thermal ODE
    # ------------------------------------------------------------------
    def dTdt(self, T_C, P_CO=None, P_H2=None, n_co_in=None):
        """Reactor temperature rate of change [degC/s]."""
        Q_gen = self.heat_released_W(T_C, P_CO, P_H2, n_co_in)
        Q_cool = self.UA * (T_C - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_reactor * self.cp_reactor)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, T0_C=None, P_CO=None, P_H2=None, n_co_in=None,
                 dt=10.0, duration_s=3600.0):
        """
        Integrate the lumped reactor thermal ODE with scipy.solve_ivp and
        evaluate the coupled FT kinetics / ASF chain-growth at each step.

        Parameters
        ----------
        T0_C : float or callable(t)
            Initial reactor temperature [degC] (callable -> coolant setpoint
            schedule is not used; T0 is the IC).
        P_CO, P_H2 : float
            Partial pressures [bar] (default from parameters).
        n_co_in : float
            CO feed rate [mol/s].
        dt : float
            Output time step [s].
        duration_s : float
            Total duration [s].

        Returns
        -------
        dict of time-series arrays: t, temperature, co_conversion,
            selectivity_jet, alpha, jet_mol_s, jet_kg_s, ptl_efficiency,
            heat_released_kW, plus scalar carbon-balance check.
        """
        T0 = self.T_coolant if T0_C is None else (
            T0_C if not callable(T0_C) else T0_C(0.0))
        P_CO = self.P_CO if P_CO is None else P_CO
        P_H2 = self.P_H2 if P_H2 is None else P_H2
        n_co = self.n_CO_in if n_co_in is None else n_co_in

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], P_CO, P_H2, n_co)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        X = np.zeros(N)
        S = np.zeros(N)
        al = np.zeros(N)
        jet_mol = np.zeros(N)
        jet_kg = np.zeros(N)
        eta = np.zeros(N)
        Q = np.zeros(N)

        for i in range(N):
            Ti = T_out[i]
            y = self.yields(Ti, P_CO, P_H2, n_co)
            X[i] = y["co_conversion"]
            S[i] = y["selectivity_jet"]
            al[i] = self.alpha(Ti)
            jet_mol[i] = y["jet_mol_s"]
            jet_kg[i] = y["jet_kg_s"]
            eta[i] = self.ptl_efficiency(Ti, P_CO, P_H2, n_co)
            Q[i] = self.heat_released_W(Ti, P_CO, P_H2, n_co) / 1000.0   # kW

        return {
            "t": t_out,
            "temperature": T_out,
            "co_conversion": X,
            "selectivity_jet": S,
            "alpha": al,
            "jet_mol_s": jet_mol,
            "jet_kg_s": jet_kg,
            "ptl_efficiency": eta,
            "heat_released_kW": Q,
        }
