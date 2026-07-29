"""
EC068 — Air-Source Heat Pump — F2a Steady-State Vapor Compression Cycle

Models the 4-component vapor compression cycle:
    Compressor (isentropic + eta_is) -> Condenser -> Expansion valve (isenthalpic) -> Evaporator

State points:
    1: Evaporator outlet (superheated vapor)
    2s: Isentropic compressor outlet
    2: Actual compressor outlet
    3: Condenser outlet (subcooled liquid)
    4: Expansion valve outlet (two-phase)

Uses CoolProp for R410A refrigerant properties when available,
falls back to simplified empirical model otherwise.

Reference:
    ASHRAE Handbook — HVAC Systems and Equipment (2020)
    Cengel & Boles, Thermodynamics: An Engineering Approach, 9th ed.
"""

import numpy as np

# ---------- CoolProp wrapper with fallback ----------
_HAS_COOLPROP = False
try:
    from CoolProp.CoolProp import PropsSI
    _HAS_COOLPROP = True
except ImportError:
    pass


def _props_si_fallback(output, in1_name, in1_val, in2_name, in2_val, fluid):
    """Simplified R410A property fallback using Antoine-style correlations.

    Only supports the subset of queries needed by the cycle model.
    """
    # Approximate R410A saturation curve (valid ~-30 to 70 degC)
    # Antoine-style: ln(P) = A - B/(T+C) where T in K, P in Pa
    A, B, C = 24.50, 3800.0, -15.0  # rough fit for R410A

    def p_sat(T_K):
        return np.exp(A - B / (T_K + C))

    def t_sat(P_Pa):
        return B / (A - np.log(P_Pa)) - C

    # Approximate properties
    cp_liquid = 1600.0   # J/kgK
    cp_vapor = 1100.0    # J/kgK
    h_fg_ref = 220e3     # J/kg at ~0 degC
    T_ref = 273.15       # K
    h_liquid_ref = 200e3  # J/kg
    s_liquid_ref = 1000.0  # J/kgK
    s_vapor_ref = 1800.0  # J/kgK
    gamma = 1.18  # ratio of specific heats for R410A vapor

    if in1_name == "T" and in2_name == "Q":
        T_K = in1_val
        Q = in2_val
        P = p_sat(T_K)
        h_l = h_liquid_ref + cp_liquid * (T_K - T_ref)
        h_fg = h_fg_ref * (1 - 0.001 * (T_K - T_ref))
        h_v = h_l + h_fg
        s_l = s_liquid_ref + cp_liquid * np.log(T_K / T_ref)
        s_v = s_l + h_fg / T_K

        if output == "P":
            return P
        elif output == "H":
            return h_l + Q * h_fg
        elif output == "S":
            return s_l + Q * (s_v - s_l)
        elif output == "D":
            if Q < 0.5:
                return 1100.0
            return P / (200.0 * T_K)

    if in1_name == "P" and in2_name == "S":
        P = in1_val
        s = in2_val
        T_K = t_sat(P)
        h_l = h_liquid_ref + cp_liquid * (T_K - T_ref)
        h_fg = h_fg_ref * (1 - 0.001 * (T_K - T_ref))
        h_v = h_l + h_fg
        s_v = s_liquid_ref + cp_liquid * np.log(T_K / T_ref) + h_fg / T_K
        # Superheated: isentropic compression
        dT_super = (s - s_v) / cp_vapor * T_K
        T_out = T_K + dT_super
        h_out = h_v + cp_vapor * (T_out - T_K)
        if output == "H":
            return h_out
        elif output == "T":
            return T_out

    if in1_name == "T" and in2_name == "P":
        T_K = in1_val
        P = in2_val
        T_sat = t_sat(P)
        h_l = h_liquid_ref + cp_liquid * (T_sat - T_ref)
        h_fg = h_fg_ref * (1 - 0.001 * (T_sat - T_ref))
        if T_K > T_sat:
            h = h_l + h_fg + cp_vapor * (T_K - T_sat)
            s = s_liquid_ref + cp_liquid * np.log(T_sat / T_ref) + h_fg / T_sat + cp_vapor * np.log(T_K / T_sat)
        else:
            h = h_l + cp_liquid * (T_K - T_sat)
            s = s_liquid_ref + cp_liquid * np.log(T_K / T_ref)
        if output == "H":
            return h
        elif output == "S":
            return s
        elif output == "D":
            if T_K <= T_sat:
                return 1100.0
            return P / (200.0 * T_K)

    if in1_name == "P" and in2_name == "H":
        P = in1_val
        h = in2_val
        T_sat = t_sat(P)
        h_l = h_liquid_ref + cp_liquid * (T_sat - T_ref)
        h_fg = h_fg_ref * (1 - 0.001 * (T_sat - T_ref))
        h_v = h_l + h_fg
        if h < h_l:
            T = T_sat + (h - h_l) / cp_liquid
        elif h > h_v:
            T = T_sat + (h - h_v) / cp_vapor
        else:
            T = T_sat
        if output == "T":
            return T
        elif output == "S":
            if h <= h_l:
                return s_liquid_ref + cp_liquid * np.log(T / T_ref)
            elif h >= h_v:
                s_v = s_liquid_ref + cp_liquid * np.log(T_sat / T_ref) + h_fg / T_sat
                return s_v + cp_vapor * np.log(T / T_sat)
            else:
                Q = (h - h_l) / h_fg
                s_l = s_liquid_ref + cp_liquid * np.log(T_sat / T_ref)
                s_v = s_l + h_fg / T_sat
                return s_l + Q * (s_v - s_l)

    raise ValueError(f"Unsupported property query: {output}({in1_name},{in2_name}) for fallback")


def _get_prop(output, in1_name, in1_val, in2_name, in2_val, fluid="R410A"):
    """Get fluid property, trying CoolProp first then fallback."""
    if _HAS_COOLPROP:
        return PropsSI(output, in1_name, in1_val, in2_name, in2_val, fluid)
    return _props_si_fallback(output, in1_name, in1_val, in2_name, in2_val, fluid)


class ASHPF2a:
    """Steady-state vapor compression cycle model for ASHP with R410A."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_nominal = u["capacity_nominal_kw"]["value"]  # kW
        self.eta_is = u["eta_isentropic"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.superheat_default = u["superheat_K"]["value"]
        self.subcool_default = u["subcool_K"]["value"]
        self.refrigerant = params.get("refrigerant", "R410A")

    def solve_cycle(self, T_evap_degC, T_cond_degC, superheat_K=None, subcool_K=None):
        """Solve vapor compression cycle and return full state + performance.

        Parameters
        ----------
        T_evap_degC : float
            Evaporating temperature (deg C).
        T_cond_degC : float
            Condensing temperature (deg C).
        superheat_K : float, optional
            Evaporator outlet superheat (K). Default from params.
        subcool_K : float, optional
            Condenser outlet subcooling (K). Default from params.

        Returns
        -------
        dict with keys:
            cop, heating_capacity_kw, compressor_power_kw, mass_flow_kg_s,
            state_points (dict of dicts for states 1-4), pressure_ratio
        """
        if superheat_K is None:
            superheat_K = self.superheat_default
        if subcool_K is None:
            subcool_K = self.subcool_default

        fluid = self.refrigerant
        T_evap_K = T_evap_degC + 273.15
        T_cond_K = T_cond_degC + 273.15

        # Saturation pressures
        P_evap = _get_prop("P", "T", T_evap_K, "Q", 1.0, fluid)
        P_cond = _get_prop("P", "T", T_cond_K, "Q", 0.0, fluid)

        # State 1: Evaporator outlet — superheated vapor
        T1 = T_evap_K + superheat_K
        h1 = _get_prop("H", "T", T1, "P", P_evap, fluid)
        s1 = _get_prop("S", "T", T1, "P", P_evap, fluid)

        # State 2s: Isentropic compression to condenser pressure
        h2s = _get_prop("H", "P", P_cond, "S", s1, fluid)

        # State 2: Actual compressor outlet
        h2 = h1 + (h2s - h1) / self.eta_is

        # State 3: Condenser outlet — subcooled liquid
        T3 = T_cond_K - subcool_K
        h3 = _get_prop("H", "T", T3, "P", P_cond, fluid)

        # State 4: Expansion valve outlet — isenthalpic
        h4 = h3

        # Performance
        w_comp_specific = (h2 - h1)  # J/kg — compressor work
        q_cond_specific = (h2 - h3)  # J/kg — condenser heat rejection (heating)
        q_evap_specific = (h1 - h4)  # J/kg — evaporator heat absorption

        # COP for heating
        cop_heating = q_cond_specific / w_comp_specific if w_comp_specific > 0 else 0.0

        # Mass flow rate to achieve nominal capacity
        # Q_heating = m_dot * q_cond_specific
        q_cond_kj = q_cond_specific / 1000.0
        m_dot = self.capacity_nominal / q_cond_kj if q_cond_kj > 0 else 0.0

        # Powers
        heating_kw = self.capacity_nominal
        compressor_kw = heating_kw / cop_heating if cop_heating > 0 else 0.0
        shaft_power = compressor_kw / self.eta_motor

        # Get state 2 and 4 temperatures for reporting
        T2 = _get_prop("T", "P", P_cond, "H", h2, fluid)
        T4 = T_evap_K  # approximately at evaporator pressure, two-phase

        state_points = {
            "1_evap_out":  {"T_K": T1, "P_Pa": P_evap, "h_J_kg": h1, "s_J_kgK": s1, "phase": "superheated_vapor"},
            "2s_isentropic": {"T_K": None, "P_Pa": P_cond, "h_J_kg": h2s, "phase": "superheated_vapor"},
            "2_comp_out":  {"T_K": float(T2), "P_Pa": P_cond, "h_J_kg": h2, "phase": "superheated_vapor"},
            "3_cond_out":  {"T_K": T3, "P_Pa": P_cond, "h_J_kg": h3, "phase": "subcooled_liquid"},
            "4_exp_out":   {"T_K": float(T4), "P_Pa": P_evap, "h_J_kg": h4, "phase": "two_phase"},
        }

        return {
            "cop": float(cop_heating),
            "heating_capacity_kw": float(heating_kw),
            "compressor_power_kw": float(shaft_power),
            "mass_flow_kg_s": float(m_dot),
            "pressure_ratio": float(P_cond / P_evap),
            "q_evap_specific_kj_kg": float(q_evap_specific / 1000),
            "q_cond_specific_kj_kg": float(q_cond_kj),
            "w_comp_specific_kj_kg": float(w_comp_specific / 1000),
            "state_points": state_points,
        }
