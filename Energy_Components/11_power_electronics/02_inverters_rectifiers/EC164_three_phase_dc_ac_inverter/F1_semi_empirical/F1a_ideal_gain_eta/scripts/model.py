"""
EC164 — Three-Phase DC-AC Inverter — F1a Ideal Gain + Efficiency
Physics equations class.

Model:
  V_ac_rms = m * V_dc / sqrt(2)           (space-vector modulation peak-to-RMS)
  eta(PLR)  = eta_rated - k1*(1-PLR) - k2*(1-PLR)^2
  P_out     = P_in * eta(PLR)
  P_loss    = P_in - P_out

Source: Mohan, Undeland & Robbins (2003), "Power Electronics," 3rd ed. Wiley.
        Part-load efficiency model: IEC 61683 / EN 50530 curve fitting.
"""

import numpy as np


class ThreePhaseInverterModel:
    """
    Semi-empirical model for a three-phase DC/AC voltage-source inverter.

    Gain model
    ----------
    For sinusoidal PWM with modulation index m (0 ≤ m ≤ 1):
        V_ac_peak = m * V_dc / 2            (phase-to-neutral peak)
        V_ac_rms  = V_ac_peak / sqrt(2)     (RMS of sinusoid)
        => V_ac_rms = m * V_dc / (2*sqrt(2))

    For space-vector PWM the DC utilisation is sqrt(3)/sqrt(2) higher:
        V_ac_rms  = m * V_dc / sqrt(2)      (line-to-line RMS, SVPWM)

    This model uses the SVPWM convention (line-to-line RMS).

    Efficiency model
    ----------------
    Part-load ratio: PLR = P_load / P_rated  [0, 1]
    eta(PLR) = eta_rated - k1*(1 - PLR) - k2*(1 - PLR)^2
    Clipped to [0, 1].
    """

    SQRT2 = np.sqrt(2.0)

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict
            V_dc_rated  : rated DC bus voltage [V]
            P_rated     : rated output power [W]
            eta_rated   : efficiency at rated load [-]
            k1          : first-order part-load loss coefficient [-]
            k2          : second-order part-load loss coefficient [-]
            f_sw        : switching frequency [Hz] (metadata only)
        """
        self.V_dc_rated = float(params["V_dc_rated"])
        self.P_rated    = float(params["P_rated"])
        self.eta_rated  = float(params["eta_rated"])
        self.k1         = float(params["k1"])
        self.k2         = float(params["k2"])
        self.f_sw       = float(params.get("f_sw", 10000.0))

    # ------------------------------------------------------------------
    # Gain model
    # ------------------------------------------------------------------

    def ac_rms_voltage(self, v_dc: float, m: float) -> float:
        """
        Line-to-line AC RMS output voltage [V] (SVPWM convention).
        V_ac_rms = m * V_dc / sqrt(2)
        """
        if not (0.0 <= m <= 1.0):
            raise ValueError(f"Modulation index m must be in [0,1], got {m}")
        return m * v_dc / self.SQRT2

    # ------------------------------------------------------------------
    # Efficiency model
    # ------------------------------------------------------------------

    def part_load_ratio(self, p_load: float) -> float:
        """Part-load ratio (PLR) = P_load / P_rated, clipped to [0,1]."""
        return float(np.clip(p_load / self.P_rated, 0.0, 1.0))

    def efficiency(self, p_load: float) -> float:
        """
        Part-load efficiency [-].
        eta = eta_rated - k1*(1-PLR) - k2*(1-PLR)^2
        Clipped to [0, 1].
        """
        if p_load <= 0.0:
            return 0.0
        PLR = self.part_load_ratio(p_load)
        eta = self.eta_rated - self.k1 * (1.0 - PLR) - self.k2 * (1.0 - PLR) ** 2
        return float(np.clip(eta, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Power quantities
    # ------------------------------------------------------------------

    def power_input(self, p_load: float) -> float:
        """DC power input [W] = P_load / eta."""
        eta = self.efficiency(p_load)
        if eta <= 0.0:
            return float("inf")
        return p_load / eta

    def power_loss(self, p_load: float) -> float:
        """Total inverter losses [W] = P_in - P_out."""
        return self.power_input(p_load) - p_load

    # ------------------------------------------------------------------
    # AC current
    # ------------------------------------------------------------------

    def ac_rms_current(self, v_dc: float, m: float, p_load: float,
                       power_factor: float = 1.0) -> float:
        """
        Line RMS current [A].
        I_ac = P_out / (sqrt(3) * V_ac_LL * PF)
        """
        V_ac = self.ac_rms_voltage(v_dc, m)
        if V_ac <= 0.0:
            return 0.0
        return p_load / (np.sqrt(3.0) * V_ac * power_factor)

    # ------------------------------------------------------------------
    # Full operating-point evaluation
    # ------------------------------------------------------------------

    def evaluate(self, v_dc: float, p_load: float, m: float,
                 power_factor: float = 1.0) -> dict:
        """
        Return all outputs for a given DC voltage, load power, and modulation.

        Parameters
        ----------
        v_dc         : DC bus voltage [V]
        p_load       : requested output power [W]
        m            : modulation index [0, 1]
        power_factor : load power factor [-]
        """
        if v_dc <= 0:
            raise ValueError(f"v_dc must be > 0, got {v_dc}")
        if p_load < 0:
            raise ValueError(f"p_load must be >= 0, got {p_load}")

        V_ac_rms = self.ac_rms_voltage(v_dc, m)
        eta      = self.efficiency(p_load)
        P_in     = self.power_input(p_load)
        P_loss   = self.power_loss(p_load)
        I_ac_rms = self.ac_rms_current(v_dc, m, p_load, power_factor)
        PLR      = self.part_load_ratio(p_load)

        return {
            "v_dc_V":       v_dc,
            "p_load_W":     p_load,
            "modulation_index": m,
            "PLR":          PLR,
            "v_ac_rms_V":   V_ac_rms,
            "i_ac_rms_A":   I_ac_rms,
            "efficiency":   eta,
            "p_in_W":       P_in,
            "p_loss_W":     P_loss,
        }
