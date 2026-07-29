"""
EC180 — Doubly-Fed Induction Generator (DFIG) — F1b Efficiency + Thermal

DFIG has stator directly connected to grid; rotor connected via power converter.
Slip determines rotor power flow direction:
  - Sub-synchronous (s > 0): rotor absorbs power from grid (motor-like)
  - Super-synchronous (s < 0): rotor delivers power to grid (typical wind)

Loss breakdown:
1. Stator copper loss (temperature-dependent):
     I_s ~ P_rated * PLR / (sqrt(3) * V_stator * pf)
     P_stator_cu = 3 * I_s^2 * R_s(T_s)
     R_s(T_s) = R_s_ref * (1 + alpha_Cu * (T_s - T_ref))

2. Rotor copper loss (temperature-dependent, scales with |slip|):
     Rotor current I_r ~ I_s (simplification for loss model)
     P_rotor_cu = 3 * I_r^2 * R_r(T_r)
     R_r(T_r) = R_r_ref * (1 + alpha_Cu * (T_r - T_ref))

3. Rotor power (slip power):
     P_rotor_slip = slip * P_airgap          [negative = super-sync: rotor delivers]
     P_airgap ~ P_out + P_stator_cu + P_iron   (simplified air-gap power)

4. Converter loss (rotor circuit converter):
     P_conv = (1 - eta_converter) * |P_rotor_slip|

5. Iron loss (constant, frequency ~fixed):
     P_iron = k_iron

6. Mechanical loss:
     P_mech = k_mech

7. Stray:
     P_stray = k_stray * P_rated

Total generator efficiency:
     P_in_shaft = P_out + P_stator_cu + P_rotor_cu + P_iron + P_mech + P_stray
     + converter losses on rotor circuit
     eta = P_out / P_in_shaft    (for wind generator: net power to grid)

Note on sign convention: P_out is the net electrical power delivered to grid
(stator + rotor converter output). For the loss model we focus on the overall
input-output relationship: mechanical shaft to net electrical output.

References:
    Muller, S. et al. (2002). Doubly Fed Induction Generator Systems for Wind Turbines.
      IEEE Industry Applications Magazine 8(3):26–33.
    Boldea, I. (2006). Variable Speed Generators. CRC Press.
    IEC 60034-30-1:2014.
"""

import numpy as np


class DFIGF1b:
    """DFIG efficiency model with slip and thermal effects."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_MW"]["value"] * 1e6   # W
        self.V_stator = u["V_stator_kV"]["value"] * 1000.0  # V
        self.pf_rated = u["pf_rated"]["value"]
        self.omega_sync = u["omega_sync_rpm"]["value"]
        self.slip_rated = u["slip_rated"]["value"]
        self.R_s_ref = u["R_s_ref"]["value"]
        self.R_r_ref = u["R_r_ref"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.alpha_Cu = u["alpha_Cu"]["value"]
        self.k_iron = u["k_iron"]["value"]
        self.k_mech = u["k_mech"]["value"]
        self.k_stray = u["k_stray"]["value"]
        self.eta_conv = u["eta_converter"]["value"]
        self.ambient_threshold = u["ambient_derating_threshold"]["value"]
        self.derating_slope = u["derating_slope"]["value"]

    def stator_resistance(self, stator_temperature):
        T_s = np.asarray(stator_temperature, dtype=float)
        return self.R_s_ref * (1.0 + self.alpha_Cu * (T_s - self.T_ref))

    def rotor_resistance(self, rotor_temperature):
        T_r = np.asarray(rotor_temperature, dtype=float)
        return self.R_r_ref * (1.0 + self.alpha_Cu * (T_r - self.T_ref))

    def stator_current(self, load_fraction):
        """Approximate stator current [A]."""
        plr = np.asarray(load_fraction, dtype=float)
        denom = np.sqrt(3.0) * self.V_stator * self.pf_rated
        return plr * self.P_rated / denom

    def rotor_speed_rpm(self, slip):
        """Actual rotor speed [rpm] = omega_sync * (1 - slip)."""
        s = np.asarray(slip, dtype=float)
        return self.omega_sync * (1.0 - s)

    def losses(self, load_fraction, slip=None, stator_temperature=75.0,
               rotor_temperature=75.0):
        """
        Full loss breakdown [W].

        Parameters
        ----------
        load_fraction : PLR (0–1.2)
        slip          : generator slip; defaults to rated slip if None
        stator_temperature, rotor_temperature : winding temperatures [degC]

        Returns dict: p_stator_cu_w, p_rotor_cu_w, p_iron_w, p_mech_w,
                      p_stray_w, p_converter_w, p_total_w
        """
        plr = np.asarray(load_fraction, dtype=float)
        s = np.asarray(slip if slip is not None else self.slip_rated, dtype=float)

        R_s = self.stator_resistance(stator_temperature)
        R_r = self.rotor_resistance(rotor_temperature)
        I_s = self.stator_current(plr)

        P_stator_cu = 3.0 * I_s**2 * R_s

        # Rotor current approximated equal to stator current magnitude (simplified)
        I_r = I_s
        P_rotor_cu = 3.0 * I_r**2 * R_r

        # Air gap power (simplified: P_out + fixed losses)
        P_out = plr * self.P_rated
        P_airgap = P_out + P_stator_cu + self.k_iron

        # Rotor slip power (positive slip = sub-sync, rotor absorbs)
        P_slip = s * P_airgap

        # Converter loss on rotor circuit
        P_conv = (1.0 - self.eta_conv) * np.abs(P_slip)

        P_iron = np.full_like(plr, self.k_iron, dtype=float)
        P_mech = np.full_like(plr, self.k_mech, dtype=float)
        P_stray = self.k_stray * self.P_rated * np.ones_like(plr)

        P_total = P_stator_cu + P_rotor_cu + P_iron + P_mech + P_stray + P_conv

        return {
            "p_stator_cu_w": P_stator_cu,
            "p_rotor_cu_w": P_rotor_cu,
            "p_iron_w": P_iron,
            "p_mech_w": P_mech,
            "p_stray_w": P_stray,
            "p_converter_w": P_conv,
            "p_total_w": P_total,
        }

    def output_power(self, load_fraction):
        """Net electrical power output [W]."""
        plr = np.asarray(load_fraction, dtype=float)
        return plr * self.P_rated

    def input_power(self, load_fraction, slip=None, stator_temperature=75.0,
                    rotor_temperature=75.0):
        """Mechanical shaft input [W]."""
        P_out = self.output_power(load_fraction)
        loss = self.losses(load_fraction, slip, stator_temperature, rotor_temperature)
        return P_out + loss["p_total_w"]

    def efficiency(self, load_fraction, slip=None, stator_temperature=75.0,
                   rotor_temperature=75.0):
        """Overall DFIG efficiency = P_electrical_out / P_shaft."""
        P_out = self.output_power(load_fraction)
        P_in = self.input_power(load_fraction, slip, stator_temperature, rotor_temperature)
        eta = np.where(P_in > 1.0, P_out / P_in, 0.0)
        return np.clip(eta, 0.0, 1.0)

    def derating_factor(self, ambient_temperature):
        """IEC 60034-1 ambient derating above 40 C."""
        T_a = np.asarray(ambient_temperature, dtype=float)
        d = 1.0 - self.derating_slope * np.maximum(T_a - self.ambient_threshold, 0.0)
        return np.clip(d, 0.0, 1.0)
