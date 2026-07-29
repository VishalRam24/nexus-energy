"""
EC193 -- Methanation Reactor (Power-to-Gas) -- F2a Kinetics + Equilibrium CSTR

Physics-lumped model: CSTR with Sabatier reaction kinetics and energy balance.

Sabatier reaction:
    CO2 + 4H2 -> CH4 + 2H2O    (delta_H = -165 kJ/mol)

Rate law (power law):
    r = k * P_CO2^0.5 * P_H2^0.5   [mol/(kg_cat.s)]
    k = A * exp(-Ea/(R*T))

CSTR mole balance (per species):
    dC_i/dt = (C_i_in - C_i)/tau + nu_i * r * rho_cat_eff

Energy balance:
    (m*cp) * dT/dt = (-delta_H)*r*V*rho_cat_eff - UA*(T-T_cool) + F_in*cp_gas*(T_in-T)

Includes:
    - Equilibrium conversion limit check via K_eq(T)
    - Thermal runaway detection
    - CO2 conversion vs temperature analysis

References:
    Koschany et al. (2016) Applied Catalysis B, 181, 504-516
    Roensch et al. (2016) Fuel, 166, 276-296
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Constants
R_GAS = 8.314  # J/(mol.K)


class MethanationReactor_F2a:
    """Methanation CSTR with Sabatier kinetics and thermal dynamics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V = u["V_reactor"]["value"]             # m3
        self.rho_cat = u["rho_cat"]["value"]          # kg/m3
        self.cat_frac = u["cat_fraction"]["value"]    # -
        self.A_pre = u["A_pre"]["value"]              # mol/(kg_cat.s.bar)
        self.Ea = u["Ea"]["value"]                    # J/mol
        self.delta_H = u["delta_H"]["value"]          # J/mol (negative = exothermic)
        self.UA = u["UA"]["value"]                    # W/K
        self.T_cool = u["T_cool"]["value"]            # K
        self.m_th = u["m_thermal"]["value"]            # kg
        self.cp_th = u["cp_thermal"]["value"]          # J/(kg.K)
        self.T_in = u["T_in"]["value"]                # K
        self.P = u["P_total"]["value"]                # bar
        self.GHSV = u["GHSV"]["value"]                # 1/h
        self.H2_CO2 = u["feed_H2_CO2_ratio"]["value"] # mol/mol
        self.cp_gas = u["cp_gas"]["value"]             # J/(mol.K)

        # Derived
        self.m_cat = self.rho_cat * self.cat_frac * self.V  # kg catalyst
        self.tau = 3600.0 / self.GHSV                        # residence time [s]

    # ------------------------------------------------------------------
    # Rate constant
    # ------------------------------------------------------------------
    def rate_constant(self, T):
        """Arrhenius rate constant k(T)."""
        return self.A_pre * np.exp(-self.Ea / (R_GAS * T))

    # ------------------------------------------------------------------
    # Reaction rate
    # ------------------------------------------------------------------
    def reaction_rate(self, T, P_CO2, P_H2):
        """Sabatier reaction rate [mol/(kg_cat.s)]."""
        k = self.rate_constant(T)
        P_CO2 = max(P_CO2, 0.0)
        P_H2 = max(P_H2, 0.0)
        return k * P_CO2**0.5 * P_H2**0.5

    # ------------------------------------------------------------------
    # Equilibrium constant for Sabatier reaction
    # ------------------------------------------------------------------
    @staticmethod
    def K_eq_sabatier(T):
        """
        Equilibrium constant for CO2 + 4H2 <-> CH4 + 2H2O.
        Based on Gibbs free energy: K = exp(-delta_G / (R*T)).
        delta_G(T) from van't Hoff with heat capacity correction.

        At 298K: delta_G0 = -130.8 kJ/mol, delta_H0 = -165 kJ/mol
        Simplified van't Hoff: ln(K) = ln(K_298) + (delta_H/R)*(1/298 - 1/T)
        K_298 = exp(130800/(8.314*298)) ~ 1.4e22
        """
        # van't Hoff: d(ln K)/d(1/T) = -delta_H/R
        # delta_H = -165000 J/mol (exothermic), so d(ln K)/d(1/T) > 0
        # => K decreases with increasing T
        delta_H_rxn = -165000.0  # J/mol (negative = exothermic)
        delta_G_298 = -130800.0  # J/mol (negative = spontaneous at 298K)
        ln_K_298 = -delta_G_298 / (R_GAS * 298.15)
        ln_K = ln_K_298 - (delta_H_rxn / R_GAS) * (1.0 / T - 1.0 / 298.15)
        return np.exp(ln_K)

    # ------------------------------------------------------------------
    # Equilibrium conversion
    # ------------------------------------------------------------------
    def equilibrium_conversion(self, T, P=None):
        """
        Compute equilibrium CO2 conversion at given T and P.
        Uses stoichiometric feed (H2:CO2 = 4:1).
        """
        P = P if P is not None else self.P
        K = self.K_eq_sabatier(T)

        # For CO2 + 4H2 -> CH4 + 2H2O, starting with 1 mol CO2, 4 mol H2
        # At conversion X: CO2=1-X, H2=4-4X, CH4=X, H2O=2X
        # total = 1-X + 4-4X + X + 2X = 5-2X
        # K = (X * (2X)^2 * (5-2X)^2) / ((1-X) * (4-4X)^4) * P^(-2)
        # Simplify: K * P^2 = 4*X^3 * (5-2X)^2 / ((1-X) * 256*(1-X)^4)
        # = X^3 * (5-2X)^2 / (64 * (1-X)^5)

        def eq_func(X):
            if X <= 0 or X >= 1:
                return 1e10
            n_total = 5.0 - 2.0 * X
            y_CO2 = (1.0 - X) / n_total
            y_H2 = (4.0 - 4.0 * X) / n_total
            y_CH4 = X / n_total
            y_H2O = 2.0 * X / n_total
            # K_p = (y_CH4 * y_H2O^2) / (y_CO2 * y_H2^4) * P^(1+2-1-4) = ... * P^(-2)
            Kp_calc = (y_CH4 * y_H2O**2) / (y_CO2 * y_H2**4 + 1e-30) * P**(-2)
            return Kp_calc - K

        try:
            X_eq = brentq(eq_func, 1e-6, 1.0 - 1e-6)
        except (ValueError, RuntimeError):
            X_eq = 0.99  # nearly complete at low T
        return X_eq

    # ------------------------------------------------------------------
    # CSTR dynamics ODE
    # ------------------------------------------------------------------
    def simulate(self, T0=None, duration_s=600.0, dt=1.0, P=None, GHSV=None,
                 T_in=None, T_cool=None):
        """
        Simulate CSTR dynamics with coupled mole + energy balance.

        State: [C_CO2, C_H2, C_CH4, C_H2O, T]
        C_i in mol/m3 (gas phase concentrations)

        Parameters
        ----------
        T0 : float
            Initial reactor temperature [K]
        duration_s : float
            Simulation duration [s]
        dt : float
            Output time step [s]
        P, GHSV, T_in, T_cool : float, optional
            Override operating conditions

        Returns
        -------
        dict with time series
        """
        T0 = T0 if T0 is not None else self.T_in
        P = P if P is not None else self.P
        GHSV = GHSV if GHSV is not None else self.GHSV
        T_in = T_in if T_in is not None else self.T_in
        T_cool = T_cool if T_cool is not None else self.T_cool
        tau = 3600.0 / GHSV  # s

        # Feed concentrations [mol/m3] from ideal gas law: C = P/(R*T)
        # Total feed: 1 mol CO2 + 4 mol H2 = 5 mol total
        C_total_in = (P * 1e5) / (R_GAS * T_in)  # mol/m3
        C_CO2_in = C_total_in / (1.0 + self.H2_CO2)      # 1/5 of total
        C_H2_in = C_total_in * self.H2_CO2 / (1.0 + self.H2_CO2)  # 4/5 of total
        C_CH4_in = 0.0
        C_H2O_in = 0.0

        # Molar flow rate
        F_mol_in = C_total_in * self.V / tau  # mol/s total molar flow

        # Catalyst loading per unit reactor volume
        rho_cat_eff = self.m_cat / self.V  # kg_cat/m3_reactor

        def rhs(t, y):
            C_CO2, C_H2, C_CH4, C_H2O, T = y

            # Ensure non-negative
            C_CO2 = max(C_CO2, 0.0)
            C_H2 = max(C_H2, 0.0)
            C_CH4 = max(C_CH4, 0.0)
            C_H2O = max(C_H2O, 0.0)

            # Partial pressures [bar] from ideal gas
            C_total = C_CO2 + C_H2 + C_CH4 + C_H2O + 1e-10
            P_CO2 = P * C_CO2 / C_total
            P_H2 = P * C_H2 / C_total

            # Reaction rate [mol/(kg_cat.s)]
            r = self.reaction_rate(T, P_CO2, P_H2)

            # Stoichiometry: CO2 + 4H2 -> CH4 + 2H2O
            # nu = [-1, -4, +1, +2]
            dCO2_dt = (C_CO2_in - C_CO2) / tau - r * rho_cat_eff
            dH2_dt = (C_H2_in - C_H2) / tau - 4.0 * r * rho_cat_eff
            dCH4_dt = (C_CH4_in - C_CH4) / tau + r * rho_cat_eff
            dH2O_dt = (C_H2O_in - C_H2O) / tau + 2.0 * r * rho_cat_eff

            # Energy balance
            Q_rxn = (-self.delta_H) * r * self.m_cat  # W (positive = heat gen)
            Q_cool = self.UA * (T - T_cool)             # W
            Q_flow = F_mol_in * self.cp_gas * (T_in - T)  # W

            dT_dt = (Q_rxn - Q_cool + Q_flow) / (self.m_th * self.cp_th)

            return [dCO2_dt, dH2_dt, dCH4_dt, dH2O_dt, dT_dt]

        # Initial conditions
        y0 = [C_CO2_in, C_H2_in, 0.0, 0.0, T0]
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="BDF", rtol=1e-8, atol=1e-10,
            max_step=dt * 10,
        )

        t_out = sol.t
        C_CO2_out = np.maximum(sol.y[0], 0.0)
        C_H2_out = np.maximum(sol.y[1], 0.0)
        C_CH4_out = np.maximum(sol.y[2], 0.0)
        C_H2O_out = np.maximum(sol.y[3], 0.0)
        T_out = sol.y[4]

        # Compute derived quantities
        X_CO2 = 1.0 - C_CO2_out / C_CO2_in  # CO2 conversion
        C_total_out = C_CO2_out + C_H2_out + C_CH4_out + C_H2O_out + 1e-10
        y_CH4_dry = C_CH4_out / (C_total_out - C_H2O_out + 1e-10)  # dry mole fraction

        # Thermal runaway detection
        T_max = np.max(T_out)
        thermal_runaway = bool(T_max > T_in + 300)

        # Equilibrium conversion at final temperature
        X_eq_final = self.equilibrium_conversion(T_out[-1], P)

        return {
            "t": t_out,
            "T": T_out,
            "C_CO2": C_CO2_out,
            "C_H2": C_H2_out,
            "C_CH4": C_CH4_out,
            "C_H2O": C_H2O_out,
            "X_CO2": X_CO2,
            "y_CH4_dry": y_CH4_dry,
            "X_eq_final": X_eq_final,
            "T_max": T_max,
            "thermal_runaway": thermal_runaway,
            "C_CO2_in": C_CO2_in,
            "C_H2_in": C_H2_in,
        }

    # ------------------------------------------------------------------
    # Conversion vs temperature (steady-state sweep)
    # ------------------------------------------------------------------
    def conversion_vs_temperature(self, T_range=None, P=None, GHSV=None):
        """
        Compute steady-state CO2 conversion across temperature range.
        Uses long simulation at each T to approximate steady state.
        """
        if T_range is None:
            T_range = np.linspace(473.15, 773.15, 40)
        P = P if P is not None else self.P
        GHSV = GHSV if GHSV is not None else self.GHSV

        X_kinetic = []
        X_equil = []
        for T in T_range:
            r = self.simulate(T0=T, duration_s=2000.0, dt=50.0, P=P, GHSV=GHSV,
                              T_in=T, T_cool=T)
            X_kinetic.append(r["X_CO2"][-1])
            X_equil.append(self.equilibrium_conversion(T, P))

        return T_range, np.array(X_kinetic), np.array(X_equil)
