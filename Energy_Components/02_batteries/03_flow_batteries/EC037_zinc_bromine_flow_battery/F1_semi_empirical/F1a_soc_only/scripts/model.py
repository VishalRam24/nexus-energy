"""
EC037 — Zinc-Bromine Flow Battery (ZBFB) — F1a Nernst + Ohmic Model

Cell voltage model:
    E_Nernst = E0 + (R_gas * T) / (n * F) * ln(SOC^2 / (1 - SOC)^2)
             = E0 + 2 * (R_gas * T) / (n * F) * ln(SOC / (1 - SOC))
    V_cell   = E_Nernst - I_cell * R_cell
    V_stack  = N_cells * V_cell

The two redox couples are:
    Negative: Zn       <-> Zn2+ + 2 e-     E0 = -0.763 V
    Positive: Br2 + 2 e- <-> 2 Br-          E0 = +1.087 V
    => E0_cell ~= 1.85 V

Both half-cells contribute symmetric ln(SOC/(1-SOC)) terms, so the
combined Nernst expression has the same form used for VRFB but with
n = 2 instead of n = 1.

References:
    Lim, H. S., Lackner, A. M., Knechtli, R. C. (1977). "Zinc-Bromine
    Secondary Battery." J. Electrochem. Soc., 124, 1154.
    Skyllas-Kazacos, M., et al. (2011). "Progress in Flow Battery
    Research and Development." J. Electrochem. Soc., 158, R55.
"""

import numpy as np

# Physical constants
R_GAS = 8.314    # J / (mol*K)
F     = 96485.0  # C / mol


class ZnBrFlowF1a:
    """Zinc-bromine flow battery stack — voltage as f(SOC, I) using Nernst + ohmic."""

    SOC_MIN = 0.01
    SOC_MAX = 0.99

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2   = u["electrode_area_cm2"]["value"]
        self.E0      = u["E0"]["value"]
        self.R_ohm_cm2 = u["R_cell_ohm_cm2"]["value"]
        self.T       = u["T_K"]["value"]
        self.n       = int(u["n"]["value"])
        self.R_cell  = self.R_ohm_cm2 / self.A_cm2  # Ohm per cell

    def _thermal_factor(self):
        """(R_gas * T) / (n * F) in Volts."""
        return R_GAS * self.T / (self.n * F)

    def e_nernst(self, soc):
        """Nernst cell potential [V]. SOC clamped to [0.01, 0.99] to avoid
        ln(0) singularity; values outside [0, 1] raise ValueError."""
        soc = np.asarray(soc, dtype=float)
        if np.any(soc < 0.0) or np.any(soc > 1.0):
            raise ValueError(
                f"SOC must be in [0, 1]. Got min={float(np.min(soc)):.6g}, "
                f"max={float(np.max(soc)):.6g}."
            )
        soc = np.clip(soc, self.SOC_MIN, self.SOC_MAX)
        return self.E0 + 2.0 * self._thermal_factor() * np.log(soc / (1.0 - soc))

    def cell_voltage(self, soc, current):
        """Terminal cell voltage [V]. current > 0 = discharge."""
        soc     = np.asarray(soc,     dtype=float)
        current = np.asarray(current, dtype=float)
        return self.e_nernst(soc) - current * self.R_cell

    def stack_voltage(self, soc, current):
        """Stack terminal voltage [V] = N_cells * V_cell."""
        return self.N_cells * self.cell_voltage(soc, current)

    def power_w(self, soc, current):
        """Stack power [W]. Positive = discharge."""
        return self.stack_voltage(soc, current) * np.asarray(current, dtype=float)

    def efficiency(self, soc, current):
        """
        Voltage efficiency at given SOC and (discharge) current magnitude.
        eta = V_discharge(I) / V_charge(I), clipped to [0, 1].
        """
        current = np.asarray(current, dtype=float)
        V_dis = self.cell_voltage(soc,  np.abs(current))
        V_chg = self.cell_voltage(soc, -np.abs(current))
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(V_chg > 0, V_dis / V_chg, 0.0)
        return np.clip(eta, 0.0, 1.0)
