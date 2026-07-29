"""
EC099 — Stirling Engine — F1b T_h/T_c Range Effects + Ambient T_c Dependence + Part-Load

Extends F1a (efficiency curve) with:
  1. Explicit hot-side temperature effect: T_h affects Carnot directly
  2. Ambient temperature determines cold-side temperature:
       T_c = T_ambient + T_approach
     Warmer ambient => higher T_c => lower Carnot efficiency
  3. Part-load curve: eta = eta_design * (1 - a*(1-PLR)^2)
  4. Auxiliary power deduction

Efficiency model:
    eta_carnot = 1 - T_c_K / T_h_K
    eta_design = f_carnot * eta_carnot
    eta_partload = eta_design * (1 - a_partload*(1-PLR)^2)
    eta_net = eta_partload * (1 - aux_fraction)

Stirling physics notes:
- The Stirling cycle is externally heated: any heat source works (combustion, solar, etc.)
- T_h refers to heater head temperature (gas temperature in hot space)
- T_c refers to cooler temperature (gas temperature in cold space)
- Real Stirling engines achieve ~40-60% of Carnot (Kongtragool & Wongwises 2003)
- Part-load: power is reduced by adjusting mean charge pressure
  (variable charge pressure Stirling), which also reduces efficiency

References:
    Kongtragool, B. & Wongwises, S. (2003) Ren. Sustain. Energy Rev. 7, 131-154.
    Cinar, C. et al. (2005) Appl. Therm. Eng. 25, 1845-1854.
    Thombare, D.G. & Verma, S.K. (2008) Ren. Sustain. Energy Rev. 12, 1-38.
"""

import numpy as np


class StirlingEngineF1b:
    """Stirling engine with ambient T_c dependence and part-load curve."""

    def __init__(self, params: dict):
        e = params["engine"]
        self.P_rated    = float(e["P_rated_w"]["value"])
        self.f_carnot   = float(e["f_carnot"]["value"])
        self.a_partload = float(e["a_partload"]["value"])
        self.T_h_design = float(e["T_h_design_c"]["value"])
        self.T_c_design = float(e["T_c_design_c"]["value"])
        self.T_c_offset = float(e["T_c_ambient_offset_c"]["value"])
        self.PLR_min    = float(e["PLR_min"]["value"])
        self.aux_frac   = float(e["aux_fraction"]["value"])

    # ------------------------------------------------------------------
    # Temperature handling
    # ------------------------------------------------------------------

    def cold_side_temp(self, T_ambient_c):
        """
        Cold-side (cooler) temperature [degC].
        T_c = T_ambient + approach offset.
        """
        T_a = np.asarray(T_ambient_c, dtype=float)
        return T_a + self.T_c_offset

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def eta_carnot(self, T_h_c, T_c_c):
        """Carnot efficiency (upper bound) for given T_h and T_c."""
        T_h = np.asarray(T_h_c, dtype=float) + 273.15
        T_c = np.asarray(T_c_c, dtype=float) + 273.15
        dT = T_h - T_c
        return np.where(dT > 0.0, 1.0 - T_c / T_h, 0.0)

    def efficiency_design(self, T_h_c, T_c_c):
        """Design-point efficiency = f_carnot * eta_Carnot."""
        return self.f_carnot * self.eta_carnot(T_h_c, T_c_c)

    def f_plr(self, PLR):
        """Part-load correction factor."""
        PLR = np.asarray(PLR, dtype=float)
        return 1.0 - self.a_partload * (1.0 - PLR) ** 2

    def efficiency_gross(self, PLR, T_h_c=None, T_ambient_c=None):
        """
        Gross efficiency = eta_design * f_PLR.
        Must not exceed Carnot.
        """
        PLR = np.asarray(PLR, dtype=float)
        T_h = self.T_h_design if T_h_c is None else T_h_c
        T_a = self.T_c_design - self.T_c_offset if T_ambient_c is None else T_ambient_c
        T_c = self.cold_side_temp(T_a)

        eta_d = self.efficiency_design(T_h, T_c)
        eta_c = self.eta_carnot(T_h, T_c)
        eta = eta_d * self.f_plr(PLR)

        # Zero below minimum load
        eta = np.where(PLR < self.PLR_min, 0.0, eta)
        return np.clip(np.minimum(eta, eta_c), 0.0, 1.0)

    def efficiency_net(self, PLR, T_h_c=None, T_ambient_c=None):
        """Net efficiency after auxiliary power deduction."""
        eta_g = self.efficiency_gross(PLR, T_h_c, T_ambient_c)
        return eta_g * (1.0 - self.aux_frac)

    # ------------------------------------------------------------------
    # Power and heat flows
    # ------------------------------------------------------------------

    def power_output_w(self, PLR):
        """Electrical output [W]."""
        PLR = np.asarray(PLR, dtype=float)
        return np.where(PLR >= self.PLR_min, PLR * self.P_rated, 0.0)

    def heat_input_w(self, PLR, T_h_c=None, T_ambient_c=None):
        """Thermal input to heater head [W]."""
        P_net = self.power_output_w(PLR)
        eta = self.efficiency_net(PLR, T_h_c, T_ambient_c)
        safe = np.where(eta > 0.001, eta, np.inf)
        return P_net / safe

    def heat_rejection_w(self, PLR, T_h_c=None, T_ambient_c=None):
        """Heat rejected to cooler [W]."""
        Q_in = self.heat_input_w(PLR, T_h_c, T_ambient_c)
        P_net = self.power_output_w(PLR)
        return np.maximum(Q_in - P_net, 0.0)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, T_h_c=None, T_ambient_c=None):
        PLR = np.asarray(PLR, dtype=float)
        T_h = self.T_h_design if T_h_c is None else T_h_c
        T_a = self.T_c_design - self.T_c_offset if T_ambient_c is None else T_ambient_c
        T_c = self.cold_side_temp(T_a)

        return {
            "efficiency_gross":   self.efficiency_gross(PLR, T_h, T_ambient_c),
            "efficiency_net":     self.efficiency_net(PLR, T_h, T_ambient_c),
            "power_output_w":     self.power_output_w(PLR),
            "heat_input_w":       self.heat_input_w(PLR, T_h, T_ambient_c),
            "heat_rejection_w":   self.heat_rejection_w(PLR, T_h, T_ambient_c),
            "eta_carnot":         self.eta_carnot(np.asarray(T_h), np.asarray(T_c)),
            "T_cold_side_c":      np.asarray(T_c, dtype=float) * np.ones_like(PLR),
            "f_partload":         self.f_plr(PLR),
        }
