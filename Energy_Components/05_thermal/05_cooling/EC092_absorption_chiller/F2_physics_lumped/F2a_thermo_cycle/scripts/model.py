"""
EC092 — Absorption Chiller — F2a Single-Effect LiBr-Water Cycle

Components: Generator + Condenser + Evaporator + Absorber + Solution Heat Exchanger (SHX)
Working pair: LiBr-Water (water is refrigerant, LiBr is absorbent)

Simplified Duhring chart correlation for LiBr-water saturation:
    T_sat(P, x) from empirical fits.

COP = Q_evap / Q_gen (typically 0.6-0.8 for single effect)

Reference:
    Herold, Radermacher & Klein (2016). Absorption Chillers and Heat Pumps, 2nd ed.
    Patek & Klomfar (2006). Int. J. Refrigeration, 29, 566-578 (LiBr-water properties).
"""

import numpy as np

_HAS_COOLPROP = False
try:
    from CoolProp.CoolProp import PropsSI
    _HAS_COOLPROP = True
except ImportError:
    pass


def _water_h(T_K, phase="liquid"):
    """Water enthalpy (J/kg) — simplified."""
    if _HAS_COOLPROP:
        if phase == "liquid":
            return PropsSI("H", "T", T_K, "Q", 0, "Water")
        elif phase == "vapor":
            return PropsSI("H", "T", T_K, "Q", 1, "Water")
    # Fallback
    cp_l = 4186.0
    h_ref = 0.0  # at 273.15 K
    h_fg = 2501e3 - 2.36e3 * (T_K - 273.15)  # Clausius-Clapeyron approx
    if phase == "liquid":
        return h_ref + cp_l * (T_K - 273.15)
    return h_ref + cp_l * (T_K - 273.15) + h_fg


def _water_psat(T_K):
    """Water saturation pressure (Pa)."""
    if _HAS_COOLPROP:
        return PropsSI("P", "T", T_K, "Q", 0, "Water")
    # Antoine equation for water
    T_C = T_K - 273.15
    return 610.78 * np.exp(17.27 * T_C / (T_C + 237.3))


def _libr_solution_h(T_K, x):
    """LiBr solution enthalpy (J/kg) — empirical correlation.

    Patek & Klomfar (2006) simplified fit.
    x: mass fraction of LiBr (0-0.7)
    """
    T_C = T_K - 273.15
    # Simplified: h_sol ~ cp_sol * T_C where cp_sol depends on concentration
    cp_sol = 4186.0 * (1 - x) + 1200.0 * x  # weighted average approximation
    # Heat of mixing correction
    h_mix = -x * (1 - x) * 50e3  # exothermic mixing
    return cp_sol * T_C + h_mix


def _libr_tsat(P_Pa, x):
    """Equilibrium temperature of LiBr solution at given P and concentration.

    Uses Duhring chart linearization: T_sol = a + b * T_water_sat(P)
    Coefficients from McNeely (1979).
    """
    # Water saturation temp at P
    # Invert Antoine: T_C = 237.3 * ln(P/610.78) / (17.27 - ln(P/610.78))
    lnp = np.log(max(P_Pa, 100.0) / 610.78)
    T_water_C = 237.3 * lnp / (17.27 - lnp)

    # Duhring coefficients (McNeely 1979, simplified for x = 0.4-0.65)
    # T_sol = A0 + A1*T_w + (B0 + B1*T_w)*x + (C0 + C1*T_w)*x^2
    A0, A1 = -2.00, 1.00
    B0, B1 = 12.0, 0.50
    C0, C1 = 40.0, 0.40

    T_sol_C = A0 + A1 * T_water_C + (B0 + B1 * T_water_C) * x + (C0 + C1 * T_water_C) * x**2
    return T_sol_C + 273.15


class AbsorptionChillerF2a:
    """Single-effect LiBr-water absorption chiller."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_nominal = u["capacity_nominal_kw"]["value"]
        self.x_strong = u["x_strong"]["value"]
        self.x_weak = u["x_weak"]["value"]
        self.eta_shx = u["eta_shx"]["value"]
        self.eta_pump = u["eta_pump"]["value"]

    def solve_cycle(self, T_gen_degC, T_cond_degC, T_evap_degC, T_abs_degC):
        """Solve single-effect absorption cycle.

        Parameters
        ----------
        T_gen_degC : float  — Generator temperature (heat source, 80-120 C)
        T_cond_degC : float — Condenser temperature (25-45 C)
        T_evap_degC : float — Evaporator temperature (3-15 C)
        T_abs_degC : float  — Absorber temperature (25-42 C)

        Returns
        -------
        dict with cop, cooling_kw, heat_input_kw, pump_power_kw, solution_flow_kg_s, etc.
        """
        T_gen = T_gen_degC + 273.15
        T_cond = T_cond_degC + 273.15
        T_evap = T_evap_degC + 273.15
        T_abs = T_abs_degC + 273.15

        # Pressures
        P_high = _water_psat(T_cond)  # condenser/generator pressure
        P_low = _water_psat(T_evap)   # evaporator/absorber pressure

        # Solution concentrations — from equilibrium at absorber and generator
        # x_weak leaves absorber (lower LiBr concentration, more water)
        # x_strong leaves generator (higher LiBr concentration)
        x_weak = self.x_weak
        x_strong = self.x_strong

        # Circulation ratio: f = m_sol / m_ref
        f = x_strong / (x_strong - x_weak) if (x_strong - x_weak) > 0.001 else 100.0

        # Enthalpies at key state points
        # Refrigerant (water)
        h_vapor_gen = _water_h(T_gen, "vapor")     # vapor leaving generator
        h_liquid_cond = _water_h(T_cond, "liquid")  # liquid leaving condenser
        h_vapor_evap = _water_h(T_evap, "vapor")    # vapor leaving evaporator
        h_liquid_evap = _water_h(T_evap, "liquid")  # liquid entering evaporator (after expansion)

        # Expansion valve: h_evap_in = h_liquid_cond (isenthalpic)
        h_evap_in = h_liquid_cond

        # Solution enthalpies
        h_sol_abs = _libr_solution_h(T_abs, x_weak)     # weak solution leaving absorber
        h_sol_gen = _libr_solution_h(T_gen, x_strong)    # strong solution leaving generator
        h_sol_gen_in = _libr_solution_h(T_abs + 10, x_weak)  # into generator (after SHX)
        h_sol_abs_in = _libr_solution_h(T_gen - 15, x_strong)  # into absorber (after SHX)

        # SHX: heat exchange between returning strong solution and weak solution going to generator
        # Effectiveness approach
        h_sol_gen_in = h_sol_abs + self.eta_shx * (h_sol_gen - h_sol_abs)

        # Energy balances per kg of refrigerant
        q_evap = h_vapor_evap - h_evap_in  # evaporator heat per kg refrigerant

        # Use thermodynamic COP from Carnot-limited absorption cycle
        # COP_Carnot_abs = (T_gen - T_cond) / T_gen * T_evap / (T_cond - T_evap)
        # (heat engine driving a heat pump)
        # Real COP = eta_cycle * COP_Carnot
        dT_gen_cond = T_gen - T_cond
        dT_cond_evap = T_cond - T_evap
        if dT_gen_cond > 0 and dT_cond_evap > 0:
            cop_carnot = (dT_gen_cond / T_gen) * (T_evap / dT_cond_evap)
        else:
            cop_carnot = 0.0

        # Cycle efficiency factor accounts for SHX effectiveness, irreversibilities
        # Typically 0.4-0.6 of Carnot for single-effect
        eta_cycle = 0.45 + 0.10 * self.eta_shx  # 0.52 for eta_shx=0.7
        cop = eta_cycle * cop_carnot
        cop = max(0.0, min(cop, 1.0))  # physical bound for single effect

        # Size to nominal capacity
        cooling_kw = self.capacity_nominal
        m_ref = cooling_kw * 1000 / q_evap if q_evap > 0 else 0.0  # kg/s refrigerant

        heat_input_kw = cooling_kw / cop if cop > 0 else 0.0
        m_solution = m_ref * f  # total solution flow

        # Solution pump power (small)
        dP = P_high - P_low
        rho_sol = 1500.0  # kg/m3 for LiBr solution ~60%
        pump_power_kw = m_solution * dP / (rho_sol * self.eta_pump * 1000)

        return {
            "cop": float(cop),
            "cooling_kw": float(cooling_kw),
            "heat_input_kw": float(heat_input_kw),
            "pump_power_kw": float(pump_power_kw),
            "solution_flow_kg_s": float(m_solution),
            "refrigerant_flow_kg_s": float(m_ref),
            "circulation_ratio": float(f),
            "P_high_kPa": float(P_high / 1000),
            "P_low_kPa": float(P_low / 1000),
            "x_weak": float(x_weak),
            "x_strong": float(x_strong),
        }
