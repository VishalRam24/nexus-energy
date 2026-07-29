"""
EC195 -- Ammonia Synthesis (Haber-Bosch) -- F2a Temkin-Pyzhev Kinetics + CSTR

Physics-lumped model: CSTR with Temkin-Pyzhev kinetics, energy balance, and recycle loop.

Reaction:
    N2 + 3H2 <-> 2NH3    (delta_H = -92 kJ/mol per mol N2)

Rate (simplified Temkin-Pyzhev):
    r = k_f * K_eq^0.5 * (P_N2 * P_H2^1.5 / P_NH3) - k_r / K_eq^0.5 * (P_NH3 / P_H2^1.5)
    where k_f = A_f * exp(-Ea_f/(R*T))
          k_r = k_f / K_eq  (from microscopic reversibility)

K_eq(T) from Gillespie-Beattie:
    ln(K_eq) = -2.691122*ln(T) - 5.519265e-5*T + 1.848863e-7*T^2 + 2001.6/T + 2.6899

Operating: 400-500C, 150-300 bar, promoted iron catalyst.
Single-pass conversion ~15-20%, overall with recycle ~97%.

References:
    Temkin & Pyzhev (1940) Acta Physicochim. URSS
    Gillespie & Beattie (1930) Phys. Rev.
    Appl (1999) Ammonia: Principles and Industrial Practice, Wiley-VCH
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314  # J/(mol.K)


class AmmoniaSynthesis_F2a:
    """Ammonia synthesis (Haber-Bosch) CSTR with Temkin-Pyzhev kinetics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V = u["V_reactor"]["value"]              # m3
        self.rho_cat = u["rho_cat"]["value"]           # kg/m3
        self.cat_frac = u["cat_fraction"]["value"]     # -
        self.A_f = u["A_f"]["value"]                   # mol/(m3.s.atm)
        self.Ea_f = u["Ea_f"]["value"]                 # J/mol
        self.delta_H = u["delta_H"]["value"]           # J/mol (per mol N2)
        self.UA = u["UA"]["value"]                     # W/K
        self.T_cool = u["T_cool"]["value"]             # K
        self.m_th = u["m_thermal"]["value"]             # kg
        self.cp_th = u["cp_thermal"]["value"]           # J/(kg.K)
        self.T_in = u["T_in"]["value"]                 # K
        self.P = u["P_total"]["value"]                 # atm
        self.GHSV = u["GHSV"]["value"]                 # 1/h
        self.H2_N2 = u["feed_H2_N2_ratio"]["value"]   # mol/mol
        self.cp_gas = u["cp_gas"]["value"]             # J/(mol.K)
        self.recycle_ratio = u["recycle_ratio"]["value"]
        self.separator_eff = u["separator_efficiency"]["value"]

        # Derived
        self.m_cat = self.rho_cat * self.cat_frac * self.V  # kg
        self.tau = 3600.0 / self.GHSV                        # s

    # ------------------------------------------------------------------
    # Equilibrium constant K_eq(T)
    # ------------------------------------------------------------------
    @staticmethod
    def K_eq(T):
        """
        Equilibrium constant for N2 + 3H2 <-> 2NH3.
        Gillespie-Beattie correlation. K_eq in atm^(-2).
        """
        ln_K = (-2.691122 * np.log(T)
                - 5.519265e-5 * T
                + 1.848863e-7 * T**2
                + 2001.6 / T
                + 2.6899)
        return np.exp(ln_K)

    # ------------------------------------------------------------------
    # Forward rate constant
    # ------------------------------------------------------------------
    def k_forward(self, T):
        """Forward rate constant."""
        return self.A_f * np.exp(-self.Ea_f / (R_GAS * T))

    # ------------------------------------------------------------------
    # Temkin-Pyzhev reaction rate
    # ------------------------------------------------------------------
    def reaction_rate(self, T, P_N2, P_H2, P_NH3):
        """
        Net reaction rate [mol/(m3.s)] for N2 consumption.
        Temkin-Pyzhev kinetics (simplified).

        r = k_f * K^0.5 * [P_N2 * P_H2^1.5 / P_NH3] - k_f / K^0.5 * [P_NH3 / P_H2^1.5]
        """
        K = self.K_eq(T)
        kf = self.k_forward(T)

        P_N2 = max(P_N2, 1e-6)
        P_H2 = max(P_H2, 1e-6)
        P_NH3 = max(P_NH3, 1e-6)

        K_sqrt = np.sqrt(K)

        # Forward term
        r_fwd = kf * K_sqrt * (P_N2 * P_H2**1.5 / P_NH3)
        # Reverse term
        r_rev = kf / K_sqrt * (P_NH3 / P_H2**1.5)

        return r_fwd - r_rev

    # ------------------------------------------------------------------
    # Equilibrium conversion
    # ------------------------------------------------------------------
    def equilibrium_conversion(self, T, P=None):
        """
        Compute equilibrium conversion of N2 at given T and P.
        Feed: 1 mol N2 + 3 mol H2 = 4 mol total.
        """
        P = P if P is not None else self.P
        K = self.K_eq(T)

        from scipy.optimize import brentq

        def eq_func(X):
            if X <= 0 or X >= 1:
                return 1e10
            n_N2 = 1.0 - X
            n_H2 = 3.0 - 3.0 * X
            n_NH3 = 2.0 * X
            n_total = 4.0 - 2.0 * X
            y_N2 = n_N2 / n_total
            y_H2 = n_H2 / n_total
            y_NH3 = n_NH3 / n_total

            # K_p = y_NH3^2 / (y_N2 * y_H2^3) * P^(-2)
            Kp_calc = (y_NH3**2) / (y_N2 * y_H2**3 + 1e-30) * P**(-2)
            return Kp_calc - K

        try:
            X_eq = brentq(eq_func, 1e-6, 1.0 - 1e-6)
        except (ValueError, RuntimeError):
            X_eq = 0.99
        return X_eq

    # ------------------------------------------------------------------
    # CSTR dynamics
    # ------------------------------------------------------------------
    def simulate(self, T0=None, duration_s=600.0, dt=1.0, P=None, GHSV=None,
                 T_in=None, T_cool=None):
        """
        Simulate single-pass CSTR dynamics.

        State: [C_N2, C_H2, C_NH3, T]
        """
        T0 = T0 if T0 is not None else self.T_in
        P = P if P is not None else self.P
        GHSV = GHSV if GHSV is not None else self.GHSV
        T_in = T_in if T_in is not None else self.T_in
        T_cool = T_cool if T_cool is not None else self.T_cool
        tau = 3600.0 / GHSV

        # Feed concentrations [mol/m3] from ideal gas: P*101325/(R*T_in)
        C_total_in = (P * 101325.0) / (R_GAS * T_in)  # mol/m3
        C_N2_in = C_total_in / (1.0 + self.H2_N2)      # 1/4 of total
        C_H2_in = C_total_in * self.H2_N2 / (1.0 + self.H2_N2)  # 3/4 of total
        C_NH3_in = 0.0

        F_mol_in = C_total_in * self.V / tau  # mol/s

        def rhs(t, y):
            C_N2, C_H2, C_NH3, T = y
            C_N2 = max(C_N2, 1e-6)
            C_H2 = max(C_H2, 1e-6)
            C_NH3 = max(C_NH3, 1e-6)

            # Partial pressures [atm]
            C_total = C_N2 + C_H2 + C_NH3 + 1e-10
            P_N2 = P * C_N2 / C_total
            P_H2 = P * C_H2 / C_total
            P_NH3 = P * C_NH3 / C_total

            # Reaction rate [mol/(m3.s)] (per reactor volume)
            r = self.reaction_rate(T, P_N2, P_H2, P_NH3)
            # Scale by catalyst loading
            r_vol = r * (self.m_cat / self.V) / self.rho_cat  # effective rate

            # Stoichiometry: N2 + 3H2 -> 2NH3
            dN2_dt = (C_N2_in - C_N2) / tau - r_vol
            dH2_dt = (C_H2_in - C_H2) / tau - 3.0 * r_vol
            dNH3_dt = (C_NH3_in - C_NH3) / tau + 2.0 * r_vol

            # Energy balance
            Q_rxn = (-self.delta_H) * r_vol * self.V  # W
            Q_cool = self.UA * (T - T_cool)
            Q_flow = F_mol_in * self.cp_gas * (T_in - T)

            dT_dt = (Q_rxn - Q_cool + Q_flow) / (self.m_th * self.cp_th)

            return [dN2_dt, dH2_dt, dNH3_dt, dT_dt]

        y0 = [C_N2_in, C_H2_in, 1e-3, T0]  # small initial NH3 to avoid division by zero
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="BDF", rtol=1e-8, atol=1e-10,
            max_step=dt * 10,
        )

        t_out = sol.t
        C_N2_out = np.maximum(sol.y[0], 0.0)
        C_H2_out = np.maximum(sol.y[1], 0.0)
        C_NH3_out = np.maximum(sol.y[2], 0.0)
        T_out = sol.y[3]

        X_N2 = 1.0 - C_N2_out / C_N2_in
        C_total_out = C_N2_out + C_H2_out + C_NH3_out + 1e-10
        y_NH3 = C_NH3_out / C_total_out

        X_eq_final = self.equilibrium_conversion(T_out[-1], P)

        return {
            "t": t_out,
            "T": T_out,
            "C_N2": C_N2_out,
            "C_H2": C_H2_out,
            "C_NH3": C_NH3_out,
            "X_N2": X_N2,
            "y_NH3": y_NH3,
            "X_eq_final": X_eq_final,
            "C_N2_in": C_N2_in,
            "C_H2_in": C_H2_in,
        }

    # ------------------------------------------------------------------
    # Recycle loop steady-state
    # ------------------------------------------------------------------
    def simulate_with_recycle(self, n_passes=20, T0=None, P=None, GHSV=None):
        """
        Simulate Haber-Bosch loop with recycle.
        Iteratively: reactor -> separator -> recycle unconverted gas.

        Returns overall conversion and energy per ton NH3.
        """
        T0 = T0 if T0 is not None else self.T_in
        P = P if P is not None else self.P
        GHSV = GHSV if GHSV is not None else self.GHSV

        # Fresh feed: 1 mol N2 + 3 mol H2
        N2_fresh = 1.0
        H2_fresh = 3.0
        NH3_total = 0.0

        N2_recycle = 0.0
        H2_recycle = 0.0

        pass_conversions = []

        for i in range(n_passes):
            # Mix fresh + recycle
            N2_feed = N2_fresh + N2_recycle
            H2_feed = H2_fresh + H2_recycle

            # Simulate single pass (long enough for SS)
            r = self.simulate(T0=T0, duration_s=2000.0, dt=50.0, P=P, GHSV=GHSV,
                              T_in=T0, T_cool=self.T_cool)
            X_sp = r["X_N2"][-1]
            X_sp = min(max(X_sp, 0.0), 0.99)
            pass_conversions.append(X_sp)

            # Products from this pass
            N2_reacted = N2_feed * X_sp
            NH3_produced = 2.0 * N2_reacted
            H2_consumed = 3.0 * N2_reacted

            N2_out = N2_feed - N2_reacted
            H2_out = H2_feed - H2_consumed

            # Separator: remove NH3 with given efficiency
            NH3_removed = NH3_produced * self.separator_eff
            NH3_total += NH3_removed

            # Recycle unconverted gas
            N2_recycle = N2_out * self.recycle_ratio
            H2_recycle = H2_out * self.recycle_ratio

            # Check convergence
            overall_X = NH3_total / (2.0 * N2_fresh)
            if overall_X > 0.99:
                break

        # Overall conversion
        overall_conversion = NH3_total / (2.0 * N2_fresh)
        overall_conversion = min(overall_conversion, 1.0)

        # Energy per ton NH3
        # Compression work: ~0.5 kWh/kg NH3 at 200 atm (typical)
        # Reactor heat balance: exothermic, but need to heat feed
        # Typical: 28-35 GJ/ton NH3 (includes compression + separation)
        MW_NH3 = 17.031  # g/mol
        mol_NH3_per_ton = 1e6 / MW_NH3  # mol
        energy_rxn_GJ = mol_NH3_per_ton * abs(self.delta_H) / 2.0 * 1e-9  # GJ (per mol N2, 2 mol NH3)
        # Add compression estimate (~12 GJ/ton for compression)
        energy_compression_GJ = 12.0
        energy_total_GJ = energy_compression_GJ + 3.0  # separation + utilities
        # Real-world: ~28-35 GJ/ton

        return {
            "overall_conversion": overall_conversion,
            "NH3_total_mol_per_mol_N2_fed": NH3_total / N2_fresh,
            "single_pass_conversions": pass_conversions,
            "n_passes": len(pass_conversions),
            "energy_per_ton_NH3_GJ": energy_total_GJ + energy_rxn_GJ * 0.1,
            "energy_rxn_GJ_per_ton": energy_rxn_GJ,
        }

    # ------------------------------------------------------------------
    # Conversion vs T and P
    # ------------------------------------------------------------------
    def conversion_vs_T_P(self, T_range=None, P_values=None):
        """Equilibrium conversion vs temperature at various pressures."""
        if T_range is None:
            T_range = np.linspace(573.15, 873.15, 40)
        if P_values is None:
            P_values = [100, 150, 200, 250, 300]

        results = {}
        for P in P_values:
            X_arr = [self.equilibrium_conversion(T, P) for T in T_range]
            results[P] = np.array(X_arr)
        return T_range, results
