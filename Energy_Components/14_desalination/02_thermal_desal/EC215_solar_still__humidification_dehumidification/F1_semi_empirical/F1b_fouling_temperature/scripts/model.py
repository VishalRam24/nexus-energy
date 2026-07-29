"""
EC215 — Solar Still / Humidification-Dehumidification (HDH) — F1b GOR + Solar + Temperature Model

HDH process: air is humidified by warm saline water (humidifier) then condensed
to produce fresh water (dehumidifier). Solar collector provides heat input.

Extends F1a with:
  1. GOR (Gain Output Ratio) as function of top water temperature:
     GOR = L_v * m_distillate / Q_solar_input
     Higher T_top → more evaporation into air → higher GOR up to ~1.5.
  2. Solar irradiance and collector efficiency effect:
     Q_avail = eta_solar * G * A_collector
     eta_solar = eta_0 - a1*(T_mean - T_amb)/G  [Hottel-Whillier collector model]
  3. Air flow rate optimization: GOR has optimal air-to-water ratio (Lambda = m_air/m_water).
     GOR_max at Lambda_opt ≈ 1.2-2.0.
  4. Temperature correction on humidity ratio: w_sat(T) from psychrometrics.

References:
    Narayan, G.P. et al. (2010). Renewable and Sustainable Energy Reviews, 14(6), 1840-1850.
    Hermosillo, J.J. et al. (2012). Sol. Energy, 86(4), 1217-1228.
    Müller-Holst, H. et al. (1998). Desalination, 122(2-3), 255-262.
"""

import numpy as np

# Psychrometric constants
P_ATM_PA = 101325.0   # Pa atmospheric pressure


def _saturation_pressure(T_degC):
    """Saturation vapor pressure [Pa] from Antoine equation (simplified).
    log10(Psat/Pa) = A - B/(C+T)  [T in degC, Antoine for water 1-100 degC]
    A=8.07131, B=1730.63, C=233.426 (Antoine, T in degC, P in mmHg)
    """
    T = np.asarray(T_degC, dtype=float)
    log_P_mmHg = 8.07131 - 1730.63 / (233.426 + T)
    P_mmHg = 10.0 ** log_P_mmHg
    return P_mmHg * 133.322  # mmHg → Pa


def _humidity_ratio(T_degC, RH=1.0):
    """Saturation humidity ratio [kg_water/kg_dry_air] at temperature T.
    w = 0.622 * Psat / (Patm - Psat)
    """
    P_sat = _saturation_pressure(T_degC) * RH
    P_sat = np.clip(P_sat, 0.0, P_ATM_PA * 0.99)
    return 0.622 * P_sat / (P_ATM_PA - P_sat)


def _latent_heat(T_degC):
    T = np.asarray(T_degC, dtype=float)
    return np.clip(2501.0 - 2.37 * T, 1500.0, 2600.0)


class HDHF1b:
    """Solar Still / HDH — GOR + solar irradiance + temperature model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.eta_solar_0   = u["eta_solar_0"]["value"]      # collector optical efficiency
        self.a1            = u["a1"]["value"]               # W/(m2*K) first-order loss coeff
        self.A_coll        = u["A_collector_m2"]["value"]   # m2 solar collector area
        self.T_top_ref     = u["T_top_ref"]["value"]        # degC reference top temperature
        self.T_cond        = u["T_cond"]["value"]           # degC condenser (dehumidifier out)
        self.Lambda_opt    = u["Lambda_opt"]["value"]       # optimal air-to-water mass ratio
        self.eta_HX        = u["eta_HX"]["value"]           # humidifier/dehumidifier effectiveness
        self.m_water_design = u["m_water_design_kg_h"]["value"]  # kg/h design water flow
        self.scale_resist  = u["scale_resistance"]["value"] # fouling resistance factor

    # ------------------------------------------------------------------ #
    #  Solar energy available
    # ------------------------------------------------------------------ #

    def solar_heat_kw(self, G_Wm2, T_mean_degC, T_amb_degC):
        """Useful solar heat [kW].
        Q = eta_solar * G * A_coll
        eta = eta_0 - a1 * (T_mean - T_amb) / G  [Hottel-Whillier]
        """
        G    = np.asarray(G_Wm2, dtype=float)
        G    = np.clip(G, 1.0, None)
        T_m  = np.asarray(T_mean_degC, dtype=float)
        T_a  = np.asarray(T_amb_degC, dtype=float)
        eta  = np.clip(self.eta_solar_0 - self.a1 * (T_m - T_a) / G, 0.0, 1.0)
        return eta * G * self.A_coll / 1000.0  # W → kW

    # ------------------------------------------------------------------ #
    #  GOR
    # ------------------------------------------------------------------ #

    def gor(self, T_top_degC, T_cond_degC=None, Lambda=None):
        """GOR (Gain Output Ratio) = m_distillate * L_v / Q_in.
        Based on Narayan 2010 simplified model:
        GOR ~ eta_HX^2 * (w_sat(T_top) - w_sat(T_cond)) * L_v(T_avg) / (c_pa * dT_cycle)
        where c_pa ~ 1.0 kJ/(kg*K) for moist air.
        """
        T_top  = np.asarray(T_top_degC, dtype=float)
        T_cond = np.asarray(T_cond_degC if T_cond_degC is not None else self.T_cond,
                            dtype=float)
        Lam    = Lambda if Lambda is not None else self.Lambda_opt

        # Humidity differences
        w_top  = _humidity_ratio(T_top)
        w_cond = _humidity_ratio(T_cond)
        dw     = np.clip(w_top - w_cond, 0.0, None)

        # Latent heat at mean temperature
        T_avg  = (T_top + T_cond) / 2.0
        L_v    = _latent_heat(T_avg)

        # Sensible heat input per kg air ≈ c_pa * (T_top - T_cond) [kJ/(kg air)]
        c_pa   = 1.006  # kJ/(kg*K) dry air
        dT_air = np.clip(T_top - T_cond, 1.0, None)

        # Yield per kg dry air = dw [kg water/kg air]
        # GOR = Lam * dw * L_v / (c_pa * dT_air)  [dimensionless]
        # Scale by eta_HX^2 (effectiveness of both HX)
        GOR = self.eta_HX ** 2 * Lam * dw * L_v / (c_pa * dT_air)
        return np.clip(GOR, 0.1, 4.0)

    # ------------------------------------------------------------------ #
    #  Production rate
    # ------------------------------------------------------------------ #

    def distillate_kg_h(self, T_top_degC, G_Wm2, T_amb_degC, T_cond_degC=None):
        """Distillate production rate [kg/h].
        m_distillate = GOR * Q_solar / L_v
        """
        T_top = np.asarray(T_top_degC, dtype=float)
        T_c   = T_cond_degC if T_cond_degC is not None else self.T_cond
        Q_kw  = self.solar_heat_kw(G_Wm2, (T_top + T_c) / 2.0, T_amb_degC)
        GOR_v = self.gor(T_top, T_c)
        T_avg = (T_top + np.asarray(T_c, dtype=float)) / 2.0
        L_v   = _latent_heat(T_avg)  # kJ/kg
        # m = GOR * Q_kW * 3600 / L_v  [kg/h]
        m_dist = GOR_v * Q_kw * 3600.0 / L_v
        return np.clip(m_dist, 0.0, None)

    def sec_kwh_m3(self, T_top_degC, G_Wm2, T_amb_degC):
        """SEC [kWh/m3] — for HDH requires auxiliary pump power.
        Thermal component: Q_solar / m_dist
        Electrical: pump power ≈ 0.2 kWh/m3 (small fan + pump).
        """
        T_top  = np.asarray(T_top_degC, dtype=float)
        Q_kw   = self.solar_heat_kw(G_Wm2, (T_top + self.T_cond) / 2.0, T_amb_degC)
        m_dist = self.distillate_kg_h(T_top, G_Wm2, T_amb_degC)
        m_dist_safe = np.clip(m_dist, 0.001, None)
        # Thermal SEC [kWh_th/m3]: Q_kw * 1 h / m_dist [m3/h] (density≈1)
        sec_th = Q_kw / (m_dist_safe / 1000.0)  # kWh/m3
        sec_el = 0.2  # kWh/m3 auxiliary electrical
        return np.clip(sec_th + sec_el, 50.0, 2000.0)

    def lambda_factor(self, Lambda):
        """GOR sensitivity to air-water ratio Lambda.
        Optimal Lambda ≈ 2.0; deviation reduces GOR.
        Factor = 1 - k_Lambda * (Lambda - Lambda_opt)^2 / Lambda_opt^2
        """
        Lam = np.asarray(Lambda, dtype=float)
        f   = 1.0 - 0.4 * ((Lam - self.Lambda_opt) / self.Lambda_opt) ** 2
        return np.clip(f, 0.3, 1.0)

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, T_top_degC, G_Wm2, T_amb_degC, Lambda=None, T_cond_degC=None):
        """Full computation.

        Parameters
        ----------
        T_top_degC  : degC   — top (humidifier outlet) water temperature
        G_Wm2       : W/m2   — solar irradiance on collector
        T_amb_degC  : degC   — ambient temperature
        Lambda      : kg/kg  — air-to-water mass ratio (default: optimal)
        T_cond_degC : degC   — condenser temperature (default: parameter value)

        Returns
        -------
        dict with gor, distillate_kg_h, sec_kwh_m3, solar_heat_kw, humidity_diff
        """
        T_top = np.asarray(T_top_degC, dtype=float)
        G     = np.asarray(G_Wm2, dtype=float)
        T_a   = np.asarray(T_amb_degC, dtype=float)
        T_c   = T_cond_degC if T_cond_degC is not None else self.T_cond
        Lam   = Lambda if Lambda is not None else self.Lambda_opt

        GOR_v  = self.gor(T_top, T_c, Lam)
        Q_kw   = self.solar_heat_kw(G, (T_top + np.asarray(T_c, dtype=float)) / 2.0, T_a)
        m_dist = self.distillate_kg_h(T_top, G, T_a, T_c)
        sec    = self.sec_kwh_m3(T_top, G, T_a)
        dw     = _humidity_ratio(T_top) - _humidity_ratio(np.asarray(T_c, dtype=float))

        return {
            "gor":             GOR_v,
            "distillate_kg_h": m_dist,
            "sec_kwh_m3":      sec,
            "solar_heat_kw":   Q_kw,
            "humidity_diff":   dw,
        }
