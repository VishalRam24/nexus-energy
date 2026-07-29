"""
EC069 — Ground-Source Heat Pump — F2a Steady-State Vapor Compression Cycle

Same vapor compression cycle as EC068 but with ground-loop coupling.
The evaporator temperature is determined by the borehole thermal interaction:
    T_source = T_ground - Q_evap / (m_brine * cp_brine * eta_bhe)
    T_evap ~ T_source - dT_approach

This creates a coupled problem: Q_evap depends on the cycle, which depends on T_evap,
which depends on Q_evap. Solved iteratively.

Reference:
    Kavanaugh & Rafferty (2014). Ground-Source Heat Pumps.
    ASHRAE Handbook — HVAC Applications, Ch. 34.
"""

import numpy as np
from scipy.optimize import brentq

_HAS_COOLPROP = False
try:
    from CoolProp.CoolProp import PropsSI
    _HAS_COOLPROP = True
except ImportError:
    pass


def _get_prop(output, in1_name, in1_val, in2_name, in2_val, fluid="R410A"):
    if _HAS_COOLPROP:
        return PropsSI(output, in1_name, in1_val, in2_name, in2_val, fluid)
    # Minimal fallback for R410A
    A, B, C = 24.50, 3800.0, -15.0
    cp_l, cp_v = 1600.0, 1100.0
    h_fg_ref, T_ref = 220e3, 273.15
    h_l_ref, s_l_ref = 200e3, 1000.0

    def p_sat(T_K):
        return np.exp(A - B / (T_K + C))

    def t_sat(P):
        return B / (A - np.log(P)) - C

    if in1_name == "T" and in2_name == "Q":
        T, Q = in1_val, in2_val
        P = p_sat(T)
        h_l = h_l_ref + cp_l * (T - T_ref)
        h_fg = h_fg_ref * (1 - 0.001 * (T - T_ref))
        s_l = s_l_ref + cp_l * np.log(T / T_ref)
        s_v = s_l + h_fg / T
        if output == "P": return P
        if output == "H": return h_l + Q * h_fg
        if output == "S": return s_l + Q * (s_v - s_l)
    if in1_name == "T" and in2_name == "P":
        T, P = in1_val, in2_val
        Ts = t_sat(P)
        h_l = h_l_ref + cp_l * (Ts - T_ref)
        h_fg = h_fg_ref * (1 - 0.001 * (Ts - T_ref))
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
        h_fg = h_fg_ref * (1 - 0.001 * (Ts - T_ref))
        s_v = s_l_ref + cp_l * np.log(Ts / T_ref) + h_fg / Ts
        dT = (s - s_v) / cp_v * Ts
        if output == "H": return h_l + h_fg + cp_v * dT
    if in1_name == "P" and in2_name == "H":
        P, h = in1_val, in2_val
        Ts = t_sat(P)
        h_l = h_l_ref + cp_l * (Ts - T_ref)
        h_fg = h_fg_ref * (1 - 0.001 * (Ts - T_ref))
        h_v = h_l + h_fg
        if h > h_v:
            T = Ts + (h - h_v) / cp_v
        elif h < h_l:
            T = Ts + (h - h_l) / cp_l
        else:
            T = Ts
        if output == "T": return T
    raise ValueError(f"Unsupported: {output}({in1_name},{in2_name})")


class GSHPF2a:
    """Ground-source heat pump with vapor compression cycle + borehole coupling."""

    def __init__(self, params: dict):
        u = params["unit"]
        gl = params["ground_loop"]
        self.capacity_nominal = u["capacity_nominal_kw"]["value"]
        self.eta_is = u["eta_isentropic"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.superheat_K = u["superheat_K"]["value"]
        self.subcool_K = u["subcool_K"]["value"]
        self.refrigerant = params.get("refrigerant", "R410A")

        self.T_ground = gl["T_ground_degC"]["value"]
        self.borehole_depth = gl["borehole_depth_m"]["value"]
        self.R_borehole = gl["borehole_resistance_mKW"]["value"]
        self.m_brine = gl["m_brine_kg_s"]["value"]
        self.cp_brine = gl["cp_brine_J_kgK"]["value"]
        self.eta_bhe = gl["eta_bhe"]["value"]

    def _solve_cycle_at_tevap(self, T_evap_degC, T_cond_degC):
        """Solve vapor compression cycle for given evap/cond temperatures."""
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
        h4 = h3

        w_comp = h2 - h1
        q_cond = h2 - h3
        q_evap = h1 - h4

        cop = q_cond / w_comp if w_comp > 0 else 0.0
        m_dot = self.capacity_nominal * 1000 / q_cond if q_cond > 0 else 0.0
        Q_evap_kw = m_dot * q_evap / 1000.0

        T2 = _get_prop("T", "P", P_cond, "H", h2, fluid)

        return {
            "cop": cop, "q_evap_kw": Q_evap_kw, "q_cond_kw": self.capacity_nominal,
            "w_comp_kw": self.capacity_nominal / cop / self.eta_motor if cop > 0 else 0,
            "m_dot": m_dot, "P_evap": P_evap, "P_cond": P_cond,
            "h1": h1, "h2": h2, "h2s": h2s, "h3": h3, "h4": h4,
            "T1": T1, "T2": float(T2), "T3": T3, "T_evap_K": T_evap_K,
            "q_evap_specific": q_evap, "q_cond_specific": q_cond, "w_comp_specific": w_comp,
        }

    def solve_cycle(self, T_cond_degC, Q_demand_kw=None):
        """Solve coupled ground-loop + vapor compression cycle.

        The borehole source temperature depends on heat extraction:
            T_source = T_ground - Q_evap / (m_brine * cp_brine * eta_bhe)
            T_evap = T_source - dT_approach (typically 3-5 K)
        """
        dT_approach = 4.0  # K between brine and refrigerant

        if Q_demand_kw is None:
            Q_demand_kw = self.capacity_nominal

        # Iterative solution: guess T_evap, solve cycle, check consistency
        def residual(T_evap_degC):
            cyc = self._solve_cycle_at_tevap(T_evap_degC, T_cond_degC)
            Q_evap = cyc["q_evap_kw"]
            # Brine temperature drop
            dT_brine = Q_evap * 1000 / (self.m_brine * self.cp_brine * self.eta_bhe)
            T_source = self.T_ground - dT_brine
            # Borehole thermal resistance effect
            Q_borehole = Q_evap * 1000  # W
            dT_borehole = Q_borehole * self.R_borehole / self.borehole_depth
            T_source_corrected = T_source - dT_borehole
            T_evap_calc = T_source_corrected - dT_approach
            return T_evap_calc - T_evap_degC

        # Solve for consistent T_evap
        try:
            T_evap_sol = brentq(residual, -15.0, self.T_ground - 1.0, xtol=0.01)
        except ValueError:
            T_evap_sol = self.T_ground - 8.0  # fallback

        cyc = self._solve_cycle_at_tevap(T_evap_sol, T_cond_degC)

        Q_evap = cyc["q_evap_kw"]
        dT_brine = Q_evap * 1000 / (self.m_brine * self.cp_brine * self.eta_bhe)
        T_source = self.T_ground - dT_brine
        dT_borehole = Q_evap * 1000 * self.R_borehole / self.borehole_depth
        T_brine_out = T_source - dT_borehole

        state_points = {
            "1_evap_out":    {"T_K": cyc["T1"], "P_Pa": cyc["P_evap"], "h_J_kg": cyc["h1"]},
            "2_comp_out":    {"T_K": cyc["T2"], "P_Pa": cyc["P_cond"], "h_J_kg": cyc["h2"]},
            "3_cond_out":    {"T_K": cyc["T3"], "P_Pa": cyc["P_cond"], "h_J_kg": cyc["h3"]},
            "4_exp_out":     {"T_K": cyc["T_evap_K"], "P_Pa": cyc["P_evap"], "h_J_kg": cyc["h4"]},
        }

        return {
            "cop": float(cyc["cop"]),
            "heating_capacity_kw": float(cyc["q_cond_kw"]),
            "compressor_power_kw": float(cyc["w_comp_kw"]),
            "mass_flow_kg_s": float(cyc["m_dot"]),
            "T_evap_degC": float(T_evap_sol),
            "T_source_degC": float(T_source),
            "T_brine_out_degC": float(T_brine_out),
            "Q_evap_kw": float(Q_evap),
            "pressure_ratio": float(cyc["P_cond"] / cyc["P_evap"]),
            "state_points": state_points,
        }
