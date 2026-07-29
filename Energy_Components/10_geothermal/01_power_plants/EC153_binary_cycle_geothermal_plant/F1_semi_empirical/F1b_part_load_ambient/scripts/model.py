"""
EC153 -- Binary Cycle Geothermal Plant -- F1b Part-Load & Ambient Derating Model

Builds on F1a (exergy-based efficiency) by adding:
  - Part-load ORC efficiency curve (eta_PL = eta_design * f_PLR(PLR))
  - Air-cooled condenser ambient derating (higher T_amb -> lower efficiency)
  - Resource temperature decline over plant lifetime (1.5%/yr)
  - Combined derating: P = P_rated * PLR * f_PLR * f_ambient * f_resource

Part-load curve (empirical, from binary ORC data):
    f_PLR: interpolated from lookup table

Ambient derating:
    f_ambient = 1 - k_amb * max(0, T_amb - T_cond_design)
    where k_amb ~ 0.005 /degC (air-cooled condenser performance)

Resource decline:
    f_resource = (1 - decline_rate)^years
    Applied to brine temperature: T_brine_eff = T_brine_initial - dT_decline

References:
    DiPippo, R. (2015). Geothermal Power Plants, 4th ed. Butterworth-Heinemann.
    Lukawski, M.Z. et al. (2014). Geofluid temperature changes during
        geothermal energy production. PROCEEDINGS, 39th Workshop on
        Geothermal Reservoir Engineering, Stanford University.
"""

import numpy as np


class BinaryGeothermalF1b:
    """Binary cycle geothermal plant with part-load, ambient derating, and resource decline."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_brine_design = u["T_brine_design_degC"]["value"]   # degC
        self.T_cond_design = u["T_cond_design_degC"]["value"]     # degC
        self.eta_design = u["eta_design"]["value"]                 # -
        self.P_rated = u["P_rated_kw"]["value"]                    # kW
        self.brine_flow_design = u["brine_flow_design_kg_s"]["value"]  # kg/s
        self.cp_brine = u["cp_brine_J_kgK"]["value"]              # J/(kg*K)
        self.decline_rate = u["decline_rate_per_yr"]["value"]      # 1/yr
        self.T_reinject_offset = u["T_reinject_offset_degC"]["value"]  # degC
        self.PLR_min = u["PLR_min"]["value"]
        self.k_amb = u["ambient_derating_coeff"]["value"]          # 1/degC

        # Part-load curve
        plc = params["part_load_curve"]
        self.plr_points = np.array(plc["PLR_points"])
        self.eta_ratio_points = np.array(plc["eta_ratio"])

    def f_plr(self, PLR):
        """Part-load efficiency ratio: eta_actual / eta_design."""
        PLR = np.clip(float(PLR), self.PLR_min, 1.0)
        return float(np.interp(PLR, self.plr_points, self.eta_ratio_points))

    def condenser_factor(self, T_ambient_degC):
        """
        Air-cooled condenser derating factor.
        Higher ambient -> higher condensing pressure -> lower efficiency.
        f_cond = 1 - k_amb * max(0, T_amb - T_cond_design)
        Minimum factor: 0.5
        """
        T_amb = float(T_ambient_degC)
        dT = max(0.0, T_amb - self.T_cond_design)
        f = 1.0 - self.k_amb * dT
        # Bonus for cold ambient (below design)
        if T_amb < self.T_cond_design:
            f = 1.0 + 0.002 * (self.T_cond_design - T_amb)
        return float(np.clip(f, 0.5, 1.15))

    def resource_factor(self, years_operation):
        """
        Resource temperature decline factor.
        f_resource = (1 - decline_rate)^years
        """
        years = max(0.0, float(years_operation))
        return float((1.0 - self.decline_rate) ** years)

    def effective_brine_temp(self, T_brine_degC, years_operation):
        """Effective brine temperature after resource decline."""
        f_res = self.resource_factor(years_operation)
        # Temperature decline applied as fraction of (T_brine - T_ambient_ground)
        T_ground = 15.0  # approximate ground temperature
        T_eff = T_ground + (float(T_brine_degC) - T_ground) * f_res
        return T_eff

    def predict(self, T_brine_degC, brine_flow_kg_s, T_ambient_degC,
                PLR=1.0, years_operation=0.0):
        """
        Compute binary plant performance with all derating factors.

        Args:
            T_brine_degC:     Wellhead brine temperature [degC]
            brine_flow_kg_s:  Brine mass flow rate [kg/s]
            T_ambient_degC:   Ambient air temperature [degC]
            PLR:              Part-load ratio [0.3-1.0]
            years_operation:  Years since start of production [years]

        Returns:
            dict with:
                power_output_kw   : Net electrical output [kW]
                efficiency        : Overall net efficiency [-]
                resource_factor   : Resource decline factor [-]
                condenser_factor  : Ambient derating factor [-]
        """
        PLR = np.clip(float(PLR), self.PLR_min, 1.0)
        T_brine = float(T_brine_degC)
        m_dot = float(brine_flow_kg_s)
        T_amb = float(T_ambient_degC)
        years = float(years_operation)

        # Derating factors
        f_pl = self.f_plr(PLR)
        f_cond = self.condenser_factor(T_amb)
        f_res = self.resource_factor(years)

        # Effective brine temperature
        T_eff = self.effective_brine_temp(T_brine, years)

        # Reinjection temperature
        T_reinject = T_amb + self.T_reinject_offset

        # Heat input from brine [kW]
        dT = max(0.0, T_eff - T_reinject)
        Q_in = m_dot * self.cp_brine * dT / 1000.0  # kW

        # Effective efficiency
        # Base Carnot-like scaling for brine temperature different from design
        T_eff_K = T_eff + 273.15
        T_design_K = self.T_brine_design + 273.15
        T_cond_K = T_amb + 273.15
        T_cond_design_K = self.T_cond_design + 273.15

        eta_carnot_actual = 1.0 - T_cond_K / T_eff_K
        eta_carnot_design = 1.0 - T_cond_design_K / T_design_K
        carnot_ratio = eta_carnot_actual / eta_carnot_design if eta_carnot_design > 0 else 0.0
        carnot_ratio = np.clip(carnot_ratio, 0.0, 1.5)

        eta_eff = self.eta_design * f_pl * f_cond * float(carnot_ratio)
        eta_eff = np.clip(eta_eff, 0.0, 0.20)

        # Power output [kW]
        P_gross = Q_in * eta_eff
        P_output = min(P_gross * PLR / max(PLR, 0.01), self.P_rated)
        P_output = max(0.0, P_output)

        return {
            "power_output_kw": float(P_output),
            "efficiency": float(eta_eff),
            "resource_factor": float(f_res),
            "condenser_factor": float(f_cond),
        }
