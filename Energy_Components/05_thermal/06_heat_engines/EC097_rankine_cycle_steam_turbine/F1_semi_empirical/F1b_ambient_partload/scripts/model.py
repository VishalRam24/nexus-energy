"""
EC097 — Rankine Cycle Steam Turbine — F1b Part-Load + Condenser Ambient Effect

Extends F1a (pure part-load curve) with:
  1. Condenser temperature (ambient) correction to efficiency
  2. Auxiliary power deduction (boiler feed pump, cooling water pump, fans)

Efficiency model:
    eta_cycle  = eta_rated * (1 - a_partload*(1-PLR)^2)    [part-load]
    f_cond     = 1 - k_cond * (T_cond - T_cond_design)     [condenser correction]
    eta_gross  = min(eta_cycle * f_cond, eta_carnot)
    eta_net    = eta_gross * (1 - aux_fraction)             [auxiliary deduction]

Condenser pressure (saturation):
    Using Antoine equation approximation for steam:
    P_sat_kPa = exp(16.3872 - 3885.70/(T_sat_K - 42.98)) / 1000   [roughly]
    A simpler linear fit for 15-55 degC range is used here.

Heat balance:
    Q_in  = P_net / eta_net      [W]
    Q_rej = Q_in - P_gross       [W]

References:
    Cotton, K.C. (1998) Evaluating and Improving Steam Turbine Performance.
    Moran, M.J. & Shapiro, H.N. (2010) Fundamentals of Engineering Thermodynamics (7e).
    Spencer, R.C., Cotton, K.C. & Cannon, C.N. (1963) ASME Paper 63-AHGT-4.
    EPRI TR-107274 (1997) Turbine Steam Path Damage: Theory and Practice.
"""

import numpy as np


class RankineCycleF1b:
    """Rankine steam turbine with condenser ambient correction and part-load model."""

    def __init__(self, params: dict):
        t = params["turbine"]
        self.P_rated       = float(t["P_rated_w"]["value"])         # W
        self.eta_rated     = float(t["eta_rated"]["value"])
        self.a_partload    = float(t["a_partload"]["value"])
        self.T_steam       = float(t["T_steam_c"]["value"])         # degC
        self.T_cond_design = float(t["T_cond_design_c"]["value"])   # degC
        self.T_cond_min    = float(t["T_cond_min_c"]["value"])
        self.T_cond_max    = float(t["T_cond_max_c"]["value"])
        self.k_cond        = float(t["k_cond"]["value"])            # 1/K
        self.PLR_min       = float(t["PLR_min"]["value"])
        self.aux_fraction  = float(t["aux_fraction"]["value"])

    # ------------------------------------------------------------------
    # Correction factors
    # ------------------------------------------------------------------

    def eta_carnot(self, T_cond_c):
        """Carnot efficiency upper bound."""
        T_hot = np.asarray(self.T_steam, dtype=float) + 273.15
        T_cold = np.asarray(T_cond_c, dtype=float) + 273.15
        return np.clip(1.0 - T_cold / T_hot, 0.0, 1.0)

    def f_partload(self, PLR):
        """Part-load correction factor: Cotton (1998) quadratic deviation."""
        PLR = np.asarray(PLR, dtype=float)
        return 1.0 - self.a_partload * (1.0 - PLR) ** 2

    def f_condenser(self, T_cond_c):
        """
        Condenser temperature correction.
        f = 1 - k_cond * (T_cond - T_cond_design)
        Higher T_cond (hot ambient) reduces cycle efficiency.
        """
        T = np.asarray(T_cond_c, dtype=float)
        f = 1.0 - self.k_cond * (T - self.T_cond_design)
        return np.clip(f, 0.4, 1.2)

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def efficiency_gross(self, PLR, T_cond_c=None):
        """
        Gross cycle thermal efficiency.
        eta_gross = eta_rated * f_PLR * f_cond, capped at Carnot.
        """
        PLR = np.asarray(PLR, dtype=float)
        T_c = self.T_cond_design if T_cond_c is None else T_cond_c
        T_c = np.asarray(T_c, dtype=float)

        eta = self.eta_rated * self.f_partload(PLR) * self.f_condenser(T_c)
        eta_c = self.eta_carnot(T_c)

        # Zero below minimum load
        eta = np.where(PLR < self.PLR_min, 0.0, eta)
        return np.clip(np.minimum(eta, eta_c), 0.0, 1.0)

    def efficiency_net(self, PLR, T_cond_c=None):
        """Net efficiency after auxiliary deduction."""
        eta_g = self.efficiency_gross(PLR, T_cond_c)
        return eta_g * (1.0 - self.aux_fraction)

    # ------------------------------------------------------------------
    # Power and heat flows
    # ------------------------------------------------------------------

    def power_output_w(self, PLR, T_cond_c=None):
        """Net electrical output [W]."""
        PLR = np.asarray(PLR, dtype=float)
        return np.maximum(PLR, 0.0) * self.P_rated

    def heat_input_w(self, PLR, T_cond_c=None):
        """Thermal heat input to boiler [W]."""
        P_net = self.power_output_w(PLR, T_cond_c)
        eta = self.efficiency_net(PLR, T_cond_c)
        safe = np.where(eta > 0.001, eta, np.inf)
        return P_net / safe

    def heat_rejection_w(self, PLR, T_cond_c=None):
        """Heat rejected to condenser [W]."""
        Q_in = self.heat_input_w(PLR, T_cond_c)
        P_net = self.power_output_w(PLR, T_cond_c)
        return np.maximum(Q_in - P_net, 0.0)

    def condenser_pressure_kpa(self, T_cond_c):
        """
        Approximate saturation pressure at condenser temperature [kPa].
        Linear fit valid 15-55 degC: P_sat ~ 1.7 * (T - 10) kPa
        (Antoine equation gives ~1.7 kPa at 15 degC, ~12.4 kPa at 52 degC)
        A more accurate expression: P_sat = exp(a - b/(T_K - c)) (Antoine)
        Here use simple linear model for quick estimation.
        """
        T = np.asarray(T_cond_c, dtype=float)
        # Antoine-like approx: ln(P) = A - B/(T+C) for steam (T in K)
        # For steam: A=18.3036, B=3816.44, C=-46.13 (Prausnitz)
        T_K = T + 273.15
        ln_P = 18.3036 - 3816.44 / (T_K - 46.13)
        return np.exp(ln_P) / 1000.0  # convert Pa to kPa

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, T_cond_c=None):
        PLR = np.asarray(PLR, dtype=float)
        T_c = self.T_cond_design if T_cond_c is None else T_cond_c
        T_c = np.asarray(T_c, dtype=float)
        return {
            "efficiency_gross":     self.efficiency_gross(PLR, T_c),
            "efficiency_net":       self.efficiency_net(PLR, T_c),
            "power_output_mw":      self.power_output_w(PLR, T_c) / 1e6,
            "heat_input_mw":        self.heat_input_w(PLR, T_c) / 1e6,
            "heat_rejection_mw":    self.heat_rejection_w(PLR, T_c) / 1e6,
            "condenser_pressure_kpa": self.condenser_pressure_kpa(T_c),
            "f_condenser":          self.f_condenser(T_c),
            "f_partload":           self.f_partload(PLR),
        }
