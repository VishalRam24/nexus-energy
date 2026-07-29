"""
EC204 — Calcium Looping — F1a Capture Rate & Energy

Two-reactor looping: carbonator (650 C) absorbs CO2, calciner (900 C) regenerates CaO.
Sorbent activity decays with cycling number.

Model:
    capture_rate(N) = capture_rate_0 * max(1 - decay * N, capture_rate_min)
    CO2_captured = capacity_fraction * capacity_tCO2_h * capture_rate(N)
    Q_thermal    = CO2_captured * SEC_thermal
    W_elec       = CO2_captured * SEC_elec

References:
    Abanades, J.C. et al. (2004). Capture of CO2 from combustion gases in a fluidized bed
      of CaO. AIChE J. 50(7):1614-1622.
    Dean, C.C. et al. (2011). The calcium looping cycle for CO2 capture from power
      generation, cement manufacture and hydrogen production. Chem. Eng. Res. Des. 89(6):836-855.
"""

import numpy as np


class CalciumLoopingF1a:
    """Calcium looping capture rate and energy model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capture_rate_0 = u["capture_rate"]["value"]
        self.SEC_thermal = u["SEC_thermal_GJ_tCO2"]["value"]
        self.SEC_elec = u["SEC_elec_GJ_tCO2"]["value"]
        self.T_calciner = u["T_calciner_C"]["value"]
        self.T_carbonator = u["T_carbonator_C"]["value"]
        self.decay = u["sorbent_activity_decay"]["value"]
        self.capacity_tCO2_h = 1.0
        self.capture_rate_min = 0.50  # floor: sorbent always has residual activity

    def capture_rate(self, cycle_number=1):
        """Effective capture rate accounting for sorbent decay [-]."""
        N = np.asarray(cycle_number, dtype=float)
        cr = self.capture_rate_0 * (1.0 - self.decay * N)
        return np.clip(cr, self.capture_rate_min, self.capture_rate_0)

    def co2_captured(self, capacity_fraction, cycle_number=1):
        """CO2 captured [tCO2/h]."""
        cf = np.asarray(capacity_fraction, dtype=float)
        cr = self.capture_rate(cycle_number)
        return cf * self.capacity_tCO2_h * cr

    def thermal_energy(self, capacity_fraction, cycle_number=1):
        """Thermal energy [GJ/h]."""
        return self.co2_captured(capacity_fraction, cycle_number) * self.SEC_thermal

    def electric_energy(self, capacity_fraction, cycle_number=1):
        """Electrical energy [GJ/h]."""
        return self.co2_captured(capacity_fraction, cycle_number) * self.SEC_elec

    def sec_total(self):
        """Total SEC [GJ/tCO2] — constant."""
        return self.SEC_thermal + self.SEC_elec
