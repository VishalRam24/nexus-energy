"""
EC091 — Vapor Compression Chiller — F2a Thermodynamic Cycle

Same vapor compression cycle as heat pump but COP defined for cooling:
    COP_cooling = Q_evap / W_comp = (h1 - h4) / (h2 - h1)

Uses R134a refrigerant.

Reference:
    ASHRAE Handbook — HVAC Systems and Equipment (2020)
    Stoecker & Jones, Refrigeration and Air Conditioning, 2nd ed.
"""

import numpy as np

_HAS_COOLPROP = False
try:
    from CoolProp.CoolProp import PropsSI
    _HAS_COOLPROP = True
except ImportError:
    pass


def _get_prop(output, in1_name, in1_val, in2_name, in2_val, fluid="R134a"):
    if _HAS_COOLPROP:
        return PropsSI(output, in1_name, in1_val, in2_name, in2_val, fluid)
    # Simplified R134a fallback
    A, B, C = 23.80, 3200.0, -20.0
    cp_l, cp_v = 1430.0, 1010.0
    h_fg_ref, T_ref = 198e3, 273.15
    h_l_ref, s_l_ref = 200e3, 1000.0

    def p_sat(T): return np.exp(A - B / (T + C))
    def t_sat(P): return B / (A - np.log(P)) - C

    if in1_name == "T" and in2_name == "Q":
        T, Q = in1_val, in2_val
        P = p_sat(T)
        h_l = h_l_ref + cp_l * (T - T_ref)
        h_fg = h_fg_ref * (1 - 0.0008 * (T - T_ref))
        s_l = s_l_ref + cp_l * np.log(T / T_ref)
        if output == "P": return P
        if output == "H": return h_l + Q * h_fg
        if output == "S": return s_l + Q * h_fg / T
    if in1_name == "T" and in2_name == "P":
        T, P = in1_val, in2_val
        Ts = t_sat(P)
        h_l = h_l_ref + cp_l * (Ts - T_ref)
        h_fg = h_fg_ref * (1 - 0.0008 * (Ts - T_ref))
        if T > Ts:
            h = h_l + h_fg + cp_v * (T - Ts)
            s = s_l_ref + cp_l * np.log(Ts / T_ref) + h_fg / Ts + cp_v * np.log(T / Ts)
        else:
            h = h_l + cp_l * (T - Ts)
            s = s_l_ref + cp_l * np.log(T / T_ref)
        if output == "H": return h
        if output == "S": return s
    if in1_name == "P" and in2_name == "S":
        P, s = in1_val, in2_val
        Ts = t_sat(P)
        h_l = h_l_ref + cp_l * (Ts - T_ref)
        h_fg = h_fg_ref * (1 - 0.0008 * (Ts - T_ref))
        s_v = s_l_ref + cp_l * np.log(Ts / T_ref) + h_fg / Ts
        dT = (s - s_v) / cp_v * Ts
        if output == "H": return h_l + h_fg + cp_v * dT
    if in1_name == "P" and in2_name == "H":
        P, h = in1_val, in2_val
        Ts = t_sat(P)
        h_l = h_l_ref + cp_l * (Ts - T_ref)
        h_fg = h_fg_ref * (1 - 0.0008 * (Ts - T_ref))
        h_v = h_l + h_fg
        if h > h_v: T = Ts + (h - h_v) / cp_v
        elif h < h_l: T = Ts + (h - h_l) / cp_l
        else: T = Ts
        if output == "T": return T
    raise ValueError(f"Unsupported: {output}({in1_name},{in2_name})")


class ChillerF2a:
    """Vapor compression chiller — cooling mode thermodynamic cycle."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_nominal = u["capacity_nominal_kw"]["value"]
        self.eta_is = u["eta_isentropic"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.superheat_K = u["superheat_K"]["value"]
        self.subcool_K = u["subcool_K"]["value"]
        self.refrigerant = params.get("refrigerant", "R134a")

    def solve_cycle(self, T_evap_degC, T_cond_degC):
        """Solve chiller cycle. COP_cooling = Q_evap / W_comp."""
        fluid = self.refrigerant
        T_evap_K = T_evap_degC + 273.15
        T_cond_K = T_cond_degC + 273.15

        P_evap = _get_prop("P", "T", T_evap_K, "Q", 1.0, fluid)
        P_cond = _get_prop("P", "T", T_cond_K, "Q", 0.0, fluid)

        T1 = T_evap_K + self.superheat_K
        h1 = _get_prop("H", "T", T1, "P", P_evap, fluid)
        s1 = _get_prop("S", "T", T1, "P", P_evap, fluid)

        h2s = _get_prop("H", "P", P_cond, "S", s1, fluid)
        h2 = h1 + (h2s - h1) / self.eta_is

        T3 = T_cond_K - self.subcool_K
        h3 = _get_prop("H", "T", T3, "P", P_cond, fluid)
        h4 = h3  # isenthalpic expansion

        w_comp = h2 - h1
        q_evap = h1 - h4
        q_cond = h2 - h3

        cop_cooling = q_evap / w_comp if w_comp > 0 else 0.0

        # Size to nominal cooling capacity
        m_dot = self.capacity_nominal * 1000 / q_evap if q_evap > 0 else 0.0
        cooling_kw = self.capacity_nominal
        compressor_kw = cooling_kw / cop_cooling / self.eta_motor if cop_cooling > 0 else 0.0
        heat_rejection_kw = cooling_kw + compressor_kw * self.eta_motor

        T2 = _get_prop("T", "P", P_cond, "H", h2, fluid)

        state_points = {
            "1_evap_out": {"T_K": T1, "P_Pa": P_evap, "h_J_kg": h1},
            "2_comp_out": {"T_K": float(T2), "P_Pa": P_cond, "h_J_kg": h2},
            "3_cond_out": {"T_K": T3, "P_Pa": P_cond, "h_J_kg": h3},
            "4_exp_out":  {"T_K": T_evap_K, "P_Pa": P_evap, "h_J_kg": h4},
        }

        return {
            "cop_cooling": float(cop_cooling),
            "cooling_capacity_kw": float(cooling_kw),
            "compressor_kw": float(compressor_kw),
            "heat_rejection_kw": float(heat_rejection_kw),
            "mass_flow_kg_s": float(m_dot),
            "pressure_ratio": float(P_cond / P_evap),
            "state_points": state_points,
        }
