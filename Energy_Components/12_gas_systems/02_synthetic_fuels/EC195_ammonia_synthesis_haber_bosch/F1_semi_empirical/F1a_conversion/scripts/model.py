"""
EC195 — Ammonia Synthesis (Haber-Bosch) — F1a Per-Pass Conversion Model

N2 + 3H2 → 2NH3  (exothermic, DH = -92 kJ/mol N2)

Per-pass conversion:
    X_calc = X_ref * (P/P_ref)^0.5 * exp(-E_a/R * (1/T - 1/T_ref))
    X_eq   = equilibrium limit (decreases with T, increases with P)
    X      = min(X_calc, X_eq)

Equilibrium: Temkin-Pyzhev approximation
    ln(K_eq) = A/T - B  →  K_eq = exp(A/T - B)
    K_eq defines equilibrium NH3 mole fraction at given T, P

Reference:
    Appl, M. (2011). Ammonia. In Ullmann's Encyclopedia of Industrial Chemistry.
    Wiley-VCH. DOI: 10.1002/14356007.a02_143
"""

import numpy as np


class AmmoniaHaberBoschF1a:
    """
    Haber-Bosch ammonia synthesis — per-pass conversion model with equilibrium limit.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_ref      = u["T_ref"]["value"] + 273.15    # K
        self.P_ref      = u["P_ref"]["value"]              # bar
        self.X_ref      = u["X_ref"]["value"]              # dimensionless
        self.E_a_R      = u["E_a_R"]["value"]              # K (Ea/R)
        self.P_exp      = u["P_exp"]["value"]
        self.E_specific = u["E_specific"]["value"]         # GJ/tNH3
        self.recycle    = u["recycle_ratio"]["value"]
        self.MW_NH3     = u["MW_NH3"]["value"] / 1000.0    # kg/mol
        self.n_N2       = u["n_N2_in"]["value"]            # mol/s

    def equilibrium_conversion(self, temperature_c, pressure_bar):
        """
        Equilibrium NH3 conversion at given T, P.

        Uses Dyson & Simon (1968) correlation for K_p (T in K, P in atm):
            log10(K_p) = -2.691122*log10(T) - 5.519265e-5*T + 1.848863e-7*T^2 + 2001.6/T + 2.6899

        For stoichiometric N2:H2=1:3 feed, equilibrium is solved analytically from:
            K_p = [y_NH3^2 * P^(-2)] / [y_N2 * y_H2^3]
        where y_i are mole fractions at equilibrium (outlet).
        For x = per-pass conversion fraction:
            y_NH3 = 2x/(4-2x), y_N2 = (1-x)/(4-2x), y_H2 = 3(1-x)/(4-2x)
        Solved via Newton-Raphson iteration.
        """
        T = np.asarray(temperature_c, dtype=float) + 273.15  # K
        P_atm = np.asarray(pressure_bar, dtype=float) / 1.01325  # bar → atm

        # Dyson & Simon (1968) log10(Kp) — accurate to ±0.5% over 300–600°C
        log_Kp = (-2.691122 * np.log10(T)
                  - 5.519265e-5 * T
                  + 1.848863e-7 * T ** 2
                  + 2001.6 / T
                  + 2.6899)
        Kp = 10.0 ** log_Kp

        # Solve equilibrium conversion numerically (vectorized via Newton method)
        # f(x) = Kp_calc(x) - Kp = 0
        # Initial guess x=0.5
        T_arr = np.atleast_1d(T)
        P_arr = np.atleast_1d(P_atm)
        Kp_arr = np.atleast_1d(Kp)
        x = np.full_like(Kp_arr, 0.5)

        for _ in range(50):
            denom = 4.0 - 2.0 * x
            y_NH3 = 2.0 * x / denom
            y_N2  = (1.0 - x) / denom
            y_H2  = 3.0 * (1.0 - x) / denom
            P_tot = P_arr
            Kp_calc = (y_NH3 * P_tot) ** 2 / (y_N2 * P_tot * (y_H2 * P_tot) ** 3 + 1e-30)
            # Residual
            res = np.log(Kp_calc + 1e-30) - np.log(Kp_arr + 1e-30)
            # Jacobian: d(log Kp_calc)/dx  ≈ (Kp_calc(x+h)-Kp_calc(x))/h / Kp_calc
            h = 1e-5
            x_h = x + h
            denom_h = 4.0 - 2.0 * x_h
            y_NH3_h = 2.0 * x_h / denom_h
            y_N2_h  = (1.0 - x_h) / denom_h
            y_H2_h  = 3.0 * (1.0 - x_h) / denom_h
            Kp_h = (y_NH3_h * P_tot) ** 2 / (y_N2_h * P_tot * (y_H2_h * P_tot) ** 3 + 1e-30)
            dlog_dx = (np.log(Kp_h + 1e-30) - np.log(Kp_calc + 1e-30)) / h
            # Newton step
            dx = -res / (dlog_dx + 1e-10)
            x = np.clip(x + dx, 1e-6, 0.9999)
            if np.all(np.abs(res) < 1e-8):
                break

        result = np.clip(x, 0.0, 0.999)
        # Return scalar if scalar input
        if result.size == 1:
            return float(result.flat[0])
        return result

    def per_pass_conversion(self, temperature_c, pressure_bar):
        """
        Per-pass conversion from Arrhenius-pressure expression.
        X_calc = X_ref * (P/P_ref)^0.5 * exp(-E_a/R * (1/T - 1/T_ref))
        """
        T = np.asarray(temperature_c, dtype=float) + 273.15
        P = np.asarray(pressure_bar, dtype=float)

        X_calc = (self.X_ref
                  * (P / self.P_ref) ** self.P_exp
                  * np.exp(-self.E_a_R * (1.0 / T - 1.0 / self.T_ref)))

        X_eq = self.equilibrium_conversion(temperature_c, pressure_bar)
        X    = np.minimum(X_calc, X_eq)
        return np.clip(X, 0.0, 1.0)

    def nh3_rate(self, temperature_c, pressure_bar, n_n2_in=None):
        """
        NH3 production rate (kg/s) for given N2 feed rate.
        NH3_rate = 2 * X * n_N2 * MW_NH3  [kg/s]
        """
        if n_n2_in is None:
            n_n2_in = self.n_N2
        X = self.per_pass_conversion(temperature_c, pressure_bar)
        n_nh3 = 2.0 * X * np.asarray(n_n2_in, dtype=float)  # mol/s
        return n_nh3 * self.MW_NH3  # kg/s

    def energy_per_ton(self, temperature_c, pressure_bar):
        """
        Specific energy consumption (GJ/tNH3).
        Increases at low conversion (more recycle, more compression).
        E = E_specific * (X_ref / X)^0.2
        """
        X = self.per_pass_conversion(temperature_c, pressure_bar)
        # At low X, more recycle required → higher energy
        E = self.E_specific * (self.X_ref / np.clip(X, 0.01, 1.0)) ** 0.2
        return E

    def efficiency(self, temperature_c, pressure_bar):
        """
        Synthesis efficiency: fraction of input H2 energy captured as NH3.
        Based on LHV: LHV_NH3 = 18.6 GJ/t, LHV_H2 = 120 GJ/t
        H2 input = 3 * n_N2 * X mol/s → H2_mass = 3X * 2/1000 kg/s
        NH3_LHV/H2_LHV ≈ 0.56 (ideal)
        Actual efficiency = E_NH3_out / E_H2_in
        """
        X = self.per_pass_conversion(temperature_c, pressure_bar)
        # LHV_NH3 = 18.6 MJ/kg, MW_NH3 = 17.031, MW_H2 = 2.016
        # NH3 produced per N2: 2X mol → 2X*17.031 g
        # H2 consumed: 3 mol → 3*2.016 g  (per mol N2 at 100% conversion)
        # At conversion X: H2 consumed per cycle = 3X mol, per mol N2
        lhv_NH3_kJ_mol = 316.0    # kJ/mol (18.6 MJ/kg * 0.017031 kg/mol * 1000)
        lhv_H2_kJ_mol  = 241.8    # kJ/mol
        eta = (2.0 * X * lhv_NH3_kJ_mol) / (3.0 * X * 4.0 * lhv_H2_kJ_mol + 1e-12)
        return np.clip(eta, 0.0, 1.0)
