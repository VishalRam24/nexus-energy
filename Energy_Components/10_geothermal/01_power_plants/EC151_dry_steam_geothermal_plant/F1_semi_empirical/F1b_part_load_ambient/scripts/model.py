"""
EC151 -- Dry Steam Geothermal Plant -- F1b Part-Load & Ambient Derating Model

Builds on F1a (exergy-based efficiency) by adding:
  - Part-load steam turbine efficiency curve (f_PLR)
  - Condensing temperature sensitivity to ambient (wet/dry cooling)
  - Resource pressure/steam-flow decline over plant lifetime
  - NCG (non-condensable gas) content effect on turbine/condenser performance
  - Combined derating: P = P_rated * PLR * f_PLR * f_cond * f_resource * f_ncg

Part-load turbine curve:
    Dry steam turbines maintain good part-load performance; typical f_PLR
    follows an empirical curve (Willan's line adapted for steam turbines).
    DiPippo (2015) Chap 7: turbine isentropic efficiency ~0.80-0.85 at full load,
    drops ~5-10% at 50% load.

Ambient / condenser derating:
    Condenser saturation T rises with ambient. For wet cooling towers:
    dT_cond ~ 0.5 * dT_amb. For air-cooled condensers: dT_cond ~ dT_amb.
    Efficiency loss: proportional to reduction in Carnot efficiency.

NCG effect:
    NCG (CO2, H2S, N2) dilutes steam in condenser, raising back-pressure.
    Typical NCG content 0.5-5 wt%. Each 1% NCG raises condenser pressure ~3 kPa,
    reducing net power ~0.5-1.0%.
    Sutton, F.M. (1976). Geothermics 4, 121-128.

Resource decline:
    Steam well pressure declines over time, reducing mass flow rate.
    Decline rate ~0.5-1.5%/yr for pressure, translated to ~0.5-1%/yr in flow.

References:
    DiPippo, R. (2015). Geothermal Power Plants, 4th ed. Butterworth-Heinemann.
    Sutton, F.M. (1976). Calculation of production decline in geothermal hot-water
        reservoirs. Geothermics, 4, 1-4, 121-128.
    Zarrouk, S.J. & Moon, H. (2014). Efficiency of geothermal power plants: A
        worldwide review. Geothermics, 51, 142-153.
"""

import numpy as np


class DrySteamGeothermalF1b:
    """
    Dry steam geothermal plant with part-load turbine, ambient derating,
    NCG penalty, and resource decline.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_steam          = u["cp_steam"]["value"]                # J/(kg*K)
        self.eta_util          = u["eta_utilization"]["value"]         # -
        self.T_cond_offset     = u["T_condenser_offset"]["value"]      # degC above T_reject
        self.P_rated           = u["P_rated_kw"]["value"]              # kW
        self.T_geo_design      = u["T_geo_design_degC"]["value"]       # degC
        self.T_reject_design   = u["T_reject_design_degC"]["value"]    # degC
        self.m_dot_design      = u["m_dot_steam_design_kg_s"]["value"] # kg/s
        self.decline_rate      = u["decline_rate_per_yr"]["value"]     # 1/yr
        self.k_amb             = u["ambient_derating_coeff"]["value"]  # 1/degC
        self.cooling_mode      = u["cooling_mode"]["value"]            # "wet" or "dry"
        self.ncg_base_pct      = u["ncg_content_base_pct"]["value"]    # wt% NCG at design
        self.k_ncg             = u["ncg_penalty_per_pct"]["value"]     # fractional loss per wt% NCG
        self.PLR_min           = u["PLR_min"]["value"]

        plc = params["part_load_curve"]
        self.plr_points     = np.array(plc["PLR_points"])
        self.eta_ratio_pts  = np.array(plc["eta_ratio"])

    # ------------------------------------------------------------------
    # Part-load efficiency ratio
    # ------------------------------------------------------------------
    def f_plr(self, PLR: float) -> float:
        """
        Turbine isentropic efficiency ratio at part load.
        Interpolated from empirical lookup. At PLR=1: ratio=1; degrades at lower PLR.
        """
        PLR_c = np.clip(float(PLR), self.PLR_min, 1.0)
        return float(np.interp(PLR_c, self.plr_points, self.eta_ratio_pts))

    # ------------------------------------------------------------------
    # Condenser (ambient) derating
    # ------------------------------------------------------------------
    def condenser_factor(self, T_ambient_degC: float) -> float:
        """
        Derating factor for elevated ambient temperature.

        Wet cooling tower: condenser T rises ~0.5 degC per degC of ambient rise.
        Dry/air cooling:   condenser T rises ~1.0 degC per degC of ambient rise.

        f_cond = (eta_Carnot_actual) / (eta_Carnot_design)
        Approximated linearly around design point, bounded [0.5, 1.2].
        """
        T_amb = float(T_ambient_degC)
        dT_amb = T_amb - self.T_reject_design  # deviation from design reject T

        # Condenser temperature rise factor (wet vs dry cooling)
        cond_sensitivity = 0.5 if self.cooling_mode == "wet" else 1.0

        # Design Carnot terms (K)
        T_hot_d  = self.T_geo_design + 273.15
        T_cond_d = (self.T_reject_design + self.T_cond_offset) + 273.15
        eta_carnot_d = 1.0 - T_cond_d / T_hot_d

        # Actual Carnot terms
        dT_cond = cond_sensitivity * dT_amb
        T_cond_a = T_cond_d + dT_cond  # K
        eta_carnot_a = 1.0 - T_cond_a / T_hot_d

        f = (eta_carnot_a / eta_carnot_d) if eta_carnot_d > 0 else 1.0
        return float(np.clip(f, 0.5, 1.2))

    # ------------------------------------------------------------------
    # Resource decline (steam well pressure/flow decline)
    # ------------------------------------------------------------------
    def resource_factor(self, years_operation: float) -> float:
        """
        Steam resource decline factor.
        Models pressure/flow decline as:
          f_resource = (1 - decline_rate)^years

        The Geysers (CA) field: ~2-3%/yr pressure decline historically.
        Modern managed fields: 0.5-1.5%/yr with reinjection.
        DiPippo (2015) p.238; Zarrouk & Moon (2014).
        """
        years = max(0.0, float(years_operation))
        return float((1.0 - self.decline_rate) ** years)

    # ------------------------------------------------------------------
    # NCG (non-condensable gas) penalty
    # ------------------------------------------------------------------
    def ncg_factor(self, ncg_content_pct: float) -> float:
        """
        NCG derating factor.
        Excess NCG above baseline raises condenser back-pressure, reducing output.
        f_ncg = 1 - k_ncg * max(0, NCG - NCG_base)

        Sutton (1976): each 1 wt% NCG above baseline reduces net output ~0.5-1%.
        Typical range: 0.5-5 wt% NCG.
        """
        ncg = float(ncg_content_pct)
        excess = max(0.0, ncg - self.ncg_base_pct)
        f = 1.0 - self.k_ncg * excess
        return float(np.clip(f, 0.5, 1.0))

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------
    def predict(self, T_geo_degC: float, m_dot_steam_kg_s: float,
                T_reject_degC: float, PLR: float = 1.0,
                years_operation: float = 0.0,
                ncg_content_pct: float = None) -> dict:
        """
        Compute dry steam plant performance with all derating factors.

        Args:
            T_geo_degC:         Steam wellhead temperature [degC]
            m_dot_steam_kg_s:   Steam mass flow rate [kg/s] (at year 0 conditions)
            T_reject_degC:      Cooling rejection temperature [degC]
            PLR:                Part-load ratio [PLR_min - 1.0]
            years_operation:    Years since commissioning [years]
            ncg_content_pct:    NCG weight fraction [wt%]; uses base value if None

        Returns:
            dict with:
                power_output_kw   : Net electrical output [kW]
                efficiency        : Overall net efficiency [-]
                resource_factor   : Steam resource decline factor [-]
                condenser_factor  : Ambient derating factor [-]
                ncg_factor        : NCG penalty factor [-]
                plr_factor        : Part-load efficiency ratio [-]
        """
        PLR     = np.clip(float(PLR), self.PLR_min, 1.0)
        T_geo   = float(T_geo_degC)
        m_dot   = float(m_dot_steam_kg_s)
        T_rej   = float(T_reject_degC)
        years   = max(0.0, float(years_operation))
        ncg_pct = float(ncg_content_pct) if ncg_content_pct is not None else self.ncg_base_pct

        # Derating factors
        f_pl   = self.f_plr(PLR)
        f_cond = self.condenser_factor(T_rej)
        f_res  = self.resource_factor(years)
        f_ncg  = self.ncg_factor(ncg_pct)

        # Effective steam flow after resource decline
        m_eff = m_dot * f_res

        # Condenser temperature
        T_cond = T_rej + self.T_cond_offset

        # Heat input from steam to condenser [kW]
        dT = max(0.0, T_geo - T_cond)
        Q_in = m_eff * self.cp_steam * dT / 1000.0  # kW

        # Carnot efficiency at actual conditions
        T_geo_K  = T_geo  + 273.15
        T_cond_K = T_cond + 273.15
        eta_carnot = max(0.0, 1.0 - T_cond_K / T_geo_K)

        # Effective plant efficiency (utilization * Carnot * part-load * ambient * NCG)
        eta_eff = self.eta_util * eta_carnot * f_pl * f_cond * f_ncg
        eta_eff = float(np.clip(eta_eff, 0.0, 0.30))

        # Gross power
        P_gross = Q_in * eta_eff

        # Apply PLR throttling
        P_output = min(P_gross * PLR, self.P_rated)
        P_output = max(0.0, float(P_output))

        return {
            "power_output_kw": P_output,
            "efficiency":      eta_eff,
            "resource_factor": float(f_res),
            "condenser_factor": float(f_cond),
            "ncg_factor":      float(f_ncg),
            "plr_factor":      float(f_pl),
        }
