"""
EC036 — Vanadium Redox Flow Battery (VRFB) — F1a Nernst + Ohmic Model

Cell voltage model:
    E_Nernst = E0 + (R_gas * T) / (n * F) * ln(SOC^2 / (1 - SOC)^2)
    V_cell   = E_Nernst - I_cell * R_cell
    V_stack  = N_cells * V_cell

where:
    SOC appears squared because both the positive (VO2+/VO2+) and
    negative (V2+/V3+) couples each contribute one log term with the
    same SOC argument, yielding 2 * (R*T/nF) * ln(SOC/(1-SOC)).

    I_cell  = I_stack (series stack — all cells carry same current)
    R_cell  = R_area_specific / A_electrode   [Ohm per cell]

Efficiency (DC round-trip, single pass):
    eta = V_discharge / V_charge  (for equal |I|)

References:
    Blanc, C., Rufer, A. (2010). Multiphysics and Energetic Modeling of a VRFB.
    Paths to Sustainable Energy, InTech, ch. 16.
"""

import numpy as np

# Physical constants
R_GAS = 8.314    # J / (mol·K)
F     = 96485.0  # C / mol


class VRFBF1a:
    """VRFB stack voltage as f(SOC, I) using Nernst + ohmic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2   = u["electrode_area_cm2"]["value"]    # cm2
        self.E0      = u["E0"]["value"]                    # V
        self.R_ohm_cm2 = u["R_cell_ohm_cm2"]["value"]     # Ohm·cm2
        self.T       = u["T_K"]["value"]                   # K
        self.n       = int(u["n"]["value"])
        # Per-cell resistance
        self.R_cell  = self.R_ohm_cm2 / self.A_cm2        # Ohm

    def _thermal_factor(self):
        """(R_gas * T) / (n * F) in Volts."""
        return R_GAS * self.T / (self.n * F)

    # Physically reasonable SOC operating limits for a flow battery.
    SOC_MIN = 0.01
    SOC_MAX = 0.99

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
        # Both couples: total = 2 * (RT/nF) * ln(SOC / (1-SOC))
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
        Voltage efficiency at given SOC and discharge current I.
        eta = V_discharge(I) / V_charge(I)
        Returned as a fraction [0, 1]. Only valid for I > 0.
        """
        current = np.asarray(current, dtype=float)
        V_dis = self.cell_voltage(soc,  current)         # drops with +I
        V_chg = self.cell_voltage(soc, -np.abs(current)) # rises with -I
        # Protect against divide-by-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(V_chg > 0, V_dis / V_chg, 0.0)
        return np.clip(eta, 0.0, 1.0)
