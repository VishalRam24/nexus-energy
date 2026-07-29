"""
EC086 — Electric Boiler / Resistance Heater — F1b Standby Loss + Ambient Dependence

Extends F1a (constant eta) with:
  1. Standby heat loss from thermal mass:
       Q_standby = UA * (T_fluid - T_ambient)     [W]
     This is active whenever the boiler holds temperature (always-on hot-water
     applications). Physically it represents jacket conduction/convection.
  2. Ambient-temperature dependence:
     Colder ambient => larger standby loss => lower effective seasonal efficiency.
  3. Explicit controls parasitic (constant P_controls_kw).

There is NO flue-gas loss for an electric boiler — the name "F1b_flue_standby"
refers to the sub-fidelity branch pattern shared across the boiler family;
the "flue" term simply evaluates to zero for electric units.

Energy balance (steady-state, firing):
    P_in      = PLR * P_rated + P_controls              [kW electrical]
    Q_useful  = eta_nom * (P_in - P_controls)           [kW thermal]
    Q_standby = UA * (T_fluid - T_amb) / 1000           [kW]
    Net Q_out = Q_useful - Q_standby  (clamped >= 0)

References:
    ASHRAE Handbook HVAC Systems & Equipment (2020), Ch. 32 'Boilers'.
    BS EN 12828:2012 — Heating systems in buildings (standby heat loss testing).
    IEA Task 44 (2013), 'Solar and Heat Pump Systems', reference electric heater.
"""

import numpy as np


class ElectricBoilerF1b:
    """Electric resistance boiler with standby loss and ambient dependence."""

    def __init__(self, params: dict):
        self.P_rated = float(params["P_rated_kw"])           # kW
        self.eta_nom = float(params["eta_nom"])              # -
        self.PLR_min = float(params["PLR_min"])              # -
        self.C_thermal = float(params["thermal_mass_kj_per_k"])  # kJ/K
        self.UA = float(params["standby_ua_w_per_k"])        # W/K
        self.T_amb_design = float(params["T_ambient_design"])  # degC
        self.T_fluid = float(params["T_fluid_design"])       # degC
        self.P_controls = float(params["P_controls_kw"])     # kW

        if not (0.0 < self.eta_nom <= 1.0):
            raise ValueError(f"eta_nom must be in (0, 1], got {self.eta_nom}")
        if self.P_rated <= 0:
            raise ValueError(f"P_rated_kw must be > 0, got {self.P_rated}")

    # ------------------------------------------------------------------
    # Core sub-models
    # ------------------------------------------------------------------

    def standby_loss_kw(self, T_ambient=None, T_fluid=None):
        """
        Standby (jacket) heat loss [kW].

        Q_standby = UA [W/K] * (T_fluid - T_ambient) / 1000 [kW]

        Parameters
        ----------
        T_ambient : float or array or None
            Ambient air temperature [degC]. Default: design value.
        T_fluid : float or array or None
            Mean boiler fluid temperature [degC]. Default: design value.
        """
        T_a = self.T_amb_design if T_ambient is None else np.asarray(T_ambient, dtype=float)
        T_f = self.T_fluid if T_fluid is None else np.asarray(T_fluid, dtype=float)
        return self.UA * (T_f - T_a) / 1000.0  # kW

    def flue_loss_kw(self, PLR=None, T_ambient=None):
        """
        Flue gas loss — zero for electric boilers (no combustion).

        Signature retained for interface compatibility with the boiler family.
        """
        if PLR is not None:
            return np.zeros_like(np.asarray(PLR, dtype=float))
        return 0.0

    def electrical_input_kw(self, PLR):
        """Total electrical input including controls [kW]."""
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        return PLR_eff * self.P_rated + self.P_controls

    def heat_output_kw(self, PLR, T_ambient=None, T_fluid=None):
        """
        Net useful thermal output [kW].

        Q_net = eta_nom * (P_in - P_controls) - Q_standby
        Clamped to >= 0.
        """
        PLR = np.asarray(PLR, dtype=float)
        P_in = self.electrical_input_kw(PLR)
        Q_gross = self.eta_nom * (P_in - self.P_controls)
        Q_sb = self.standby_loss_kw(T_ambient, T_fluid)
        return np.maximum(Q_gross - Q_sb, 0.0)

    def efficiency(self, PLR, T_ambient=None, T_fluid=None):
        """
        Effective electrical-to-useful-thermal efficiency [-].

        eta_eff = Q_net / P_in
        """
        PLR = np.asarray(PLR, dtype=float)
        P_in = self.electrical_input_kw(PLR)
        Q_out = self.heat_output_kw(PLR, T_ambient, T_fluid)
        safe_P = np.where(P_in > 0.01, P_in, 0.01)
        return np.clip(Q_out / safe_P, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, T_ambient=None, T_fluid=None):
        """
        Full operating-point evaluation.

        Parameters
        ----------
        PLR        : float or array, part-load ratio [0, 1]
        T_ambient  : float or array or None, ambient temperature [degC]
        T_fluid    : float or array or None, boiler fluid temperature [degC]

        Returns
        -------
        dict with: efficiency, electrical_input_kw, heat_output_kw,
                   standby_loss_kw, flue_loss_kw, controls_kw
        """
        PLR = np.asarray(PLR, dtype=float)
        return {
            "efficiency":          self.efficiency(PLR, T_ambient, T_fluid),
            "electrical_input_kw": self.electrical_input_kw(PLR),
            "heat_output_kw":      self.heat_output_kw(PLR, T_ambient, T_fluid),
            "standby_loss_kw":     np.full_like(PLR,
                                       float(self.standby_loss_kw(T_ambient, T_fluid))),
            "flue_loss_kw":        self.flue_loss_kw(PLR),
            "controls_kw":         np.full_like(PLR, self.P_controls),
        }
