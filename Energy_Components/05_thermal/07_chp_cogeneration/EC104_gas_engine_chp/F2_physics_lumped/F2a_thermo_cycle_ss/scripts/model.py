"""
EC104 -- Gas Engine CHP -- F2a Otto Cycle Thermodynamic Model

Air-standard Otto cycle analysis:
    1->2: Isentropic compression     T2 = T1 * r^(gamma-1)
    2->3: Constant volume heat add   T3 = T2 + Q_in / (m_air * cv)
    3->4: Isentropic expansion       T4 = T3 / r^(gamma-1)
    4->1: Constant volume rejection  Q_out = m_air * cv * (T4 - T1)

    eta_Otto = 1 - 1/r^(gamma-1)

Heat recovery:
    - Exhaust heat: Q_exhaust = m_exhaust * cp_exhaust * (T_exhaust - T_stack)
      with recovery efficiency eta_exhaust_recovery
    - Jacket water: Q_jacket = fraction of fuel input (typically 25-30%)

Net outputs:
    P_electrical = Q_in * eta_Otto * eta_mechanical * eta_generator
    Q_recovered  = Q_exhaust_recovered + Q_jacket

Reference:
    Cengel, Y.A. & Boles, M.A. (2019). Thermodynamics: An Engineering Approach, 9th ed.
    US EPA CHP Technology Fact Sheets (2017).
"""

import numpy as np


class GasEngineCHPF2a:
    """Gas engine CHP -- Otto cycle first-principles model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.r = u["compression_ratio"]["value"]
        self.gamma = u["gamma"]["value"]
        self.eta_mech = u["eta_mechanical"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.cp_air = u["cp_air"]["value"]              # J/(kg*K)
        self.cv_air = u["cv_air"]["value"]              # J/(kg*K)
        self.cp_exhaust = u["cp_exhaust"]["value"]      # J/(kg*K)
        self.m_air_per_kw = u["m_air_per_kw_fuel"]["value"]  # kg/s per kW fuel
        self.T_stack = u["T_stack_K"]["value"]           # exhaust stack T limit
        self.eta_exhaust_recovery = u["eta_exhaust_recovery"]["value"]
        self.f_jacket = u["f_jacket"]["value"]           # jacket heat fraction of fuel

    def otto_efficiency(self, compression_ratio=None):
        """Ideal Otto cycle thermal efficiency."""
        r = compression_ratio if compression_ratio is not None else self.r
        r = np.asarray(r, dtype=float)
        return 1.0 - 1.0 / r**(self.gamma - 1.0)

    def cycle_temperatures(self, fuel_input_kw, compression_ratio=None, T_ambient_K=298.15):
        """
        Compute Otto cycle state temperatures.

        Args:
            fuel_input_kw:      Fuel input [kW]
            compression_ratio:  Override default r
            T_ambient_K:        Ambient/intake temperature [K]

        Returns:
            dict with T1, T2, T3, T4 [K]
        """
        r = compression_ratio if compression_ratio is not None else self.r
        r = np.asarray(r, dtype=float)
        T1 = np.asarray(T_ambient_K, dtype=float)
        Q_in = np.asarray(fuel_input_kw, dtype=float) * 1000.0  # W

        # State 1->2: Isentropic compression
        T2 = T1 * r**(self.gamma - 1.0)

        # State 2->3: Constant volume heat addition
        m_air = self.m_air_per_kw * np.asarray(fuel_input_kw, dtype=float)  # kg/s
        m_air = np.maximum(m_air, 1e-6)
        T3 = T2 + Q_in / (m_air * self.cv_air)

        # State 3->4: Isentropic expansion
        T4 = T3 / r**(self.gamma - 1.0)

        return {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "m_air": m_air}

    def compute(self, fuel_input_kw, compression_ratio=None, T_ambient_K=298.15):
        """
        Full CHP computation.

        Args:
            fuel_input_kw:      Fuel input [kW]
            compression_ratio:  Override default
            T_ambient_K:        Ambient temperature [K]

        Returns:
            dict with all outputs
        """
        fuel_input_kw = np.asarray(fuel_input_kw, dtype=float)
        r = compression_ratio if compression_ratio is not None else self.r

        temps = self.cycle_temperatures(fuel_input_kw, r, T_ambient_K)
        T1, T2, T3, T4 = temps["T1"], temps["T2"], temps["T3"], temps["T4"]
        m_air = temps["m_air"]

        # Ideal Otto efficiency
        eta_otto = self.otto_efficiency(r)

        # Electrical output
        P_shaft = fuel_input_kw * eta_otto * self.eta_mech
        P_electrical = P_shaft * self.eta_gen

        # Exhaust heat recovery
        T_exhaust = T4  # exhaust temperature after expansion
        Q_exhaust_total = m_air * self.cp_exhaust * (T_exhaust - self.T_stack) / 1000.0  # kW
        Q_exhaust_total = np.maximum(Q_exhaust_total, 0.0)
        Q_exhaust_recovered = Q_exhaust_total * self.eta_exhaust_recovery

        # Jacket water heat
        Q_jacket = fuel_input_kw * self.f_jacket

        # Total thermal output
        Q_thermal_total = Q_exhaust_recovered + Q_jacket

        # Unrecoverable losses (radiation, unburned fuel, oil cooling) ~5%
        Q_loss = fuel_input_kw * 0.05

        # Cap total recovered energy to fuel input minus losses
        total_out = P_electrical + Q_thermal_total
        max_out = fuel_input_kw - Q_loss
        scale = np.where(total_out > max_out,
                         np.where(total_out > 0, max_out / total_out, 1.0), 1.0)
        P_electrical = P_electrical * scale
        Q_exhaust_recovered = Q_exhaust_recovered * scale
        Q_jacket = Q_jacket * scale
        Q_thermal_total = Q_exhaust_recovered + Q_jacket

        # Efficiencies
        eta_electrical = np.where(fuel_input_kw > 0,
                                  P_electrical / fuel_input_kw, 0.0)
        eta_thermal = np.where(fuel_input_kw > 0,
                               Q_thermal_total / fuel_input_kw, 0.0)

        return {
            "power_electrical_kw": P_electrical,
            "heat_exhaust_kw": Q_exhaust_recovered,
            "heat_jacket_kw": Q_jacket,
            "eta_electrical": np.clip(eta_electrical, 0.0, 0.55),
            "eta_thermal": np.clip(eta_thermal, 0.0, 0.60),
            "eta_total": np.clip(eta_electrical + eta_thermal, 0.0, 0.95),
            "T_exhaust_K": T_exhaust,
            "T2_K": T2,
            "T3_K": T3,
            "eta_otto_ideal": eta_otto,
        }
