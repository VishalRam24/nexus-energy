"""
EC212 — Multi-Stage Flash (MSF) Distillation — F1b GOR + Top Brine Temperature + Scaling Model

Extends F1a (basic SEC/GOR) with:
  1. GOR (Gain Output Ratio) curve as function of N_stages and TBT:
     GOR ≈ 0.9 * N_stages * dT_stage / L_v  — classic MSF approximation.
     More precisely: GOR = eta_HX * (TBT - T_last_stage) / (L_v/c_p)
     where T_last_stage ≈ 40 degC (condenser).
  2. Top Brine Temperature (TBT) effect on GOR and scaling risk:
     - Higher TBT → higher GOR (more evaporation per stage) BUT
     - TBT > 110 degC → CaSO4 scaling occurs (carbonate scales: TBT > 90 degC).
     - Scaling penalty: SEC_penalty = 1 + k_scale * max(TBT - T_scale_limit, 0)
  3. Brine heater duty: Q_BH = m_feed * c_p * (TBT - T_recycle)
  4. Part-load: pumping power scales with flow, brine heater scales linearly with capacity.

References:
    El-Dessouky, H.T. & Ettouney, H.M. (2002). Fundamentals of Salt Water Desalination.
    Elseviers Science, Amsterdam. Chapter 5 (MSF).
    Al-Rawajfeh, A.E. et al. (2004). Desalination, 166, 213-222.
    Darwish, M.A. & Alsairafi, A. (2004). Desalination, 170(3), 223-239.
"""

import numpy as np

CP_WATER  = 4.18    # kJ/(kg*K)
CP_BRINE  = 3.90    # kJ/(kg*K) — slightly lower than pure water


def _latent_heat(T_degC):
    """Latent heat of vaporization [kJ/kg] at temperature T.
    Approximation: L_v(T) = 2501 - 2.37*T  [kJ/kg] (valid 0-200 degC).
    """
    T = np.asarray(T_degC, dtype=float)
    return np.clip(2501.0 - 2.37 * T, 1500.0, 2600.0)


class MSFF1b:
    """MSF distillation — GOR curve + TBT effects + scaling model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_stages      = u["N_stages"]["value"]
        self.TBT_ref       = u["TBT_ref"]["value"]        # degC reference TBT
        self.T_condenser   = u["T_condenser"]["value"]    # degC last-stage condenser
        self.T_scale_lim   = u["T_scale_limit"]["value"]  # degC scaling onset (carbonate)
        self.k_scale       = u["k_scale"]["value"]        # SEC penalty per degC above limit
        self.eta_HX        = u["eta_HX"]["value"]         # heat recovery section effectiveness
        self.PR_ref        = u["PR_ref"]["value"]          # performance ratio at design
        self.m_ratio       = u["m_ratio"]["value"]         # recycle brine/distillate ratio
        self.pump_sec_ref  = u["pump_SEC_ref"]["value"]   # kWh/m3 pumping at design
        self.eta_pump      = u["eta_pump"]["value"]

    # ------------------------------------------------------------------ #
    #  GOR and thermal performance
    # ------------------------------------------------------------------ #

    def gor(self, TBT_degC, N_stages=None):
        """Gain Output Ratio (kg distillate / kg steam input).
        GOR ≈ N_stages * eta_HX * dT_effective / (L_v / cp_brine)
        dT_effective = (TBT - T_condenser) / N_stages  [flash range per stage]
        This gives: GOR = eta_HX * (TBT - T_condenser) * cp_brine / L_v(TBT_avg)
        """
        TBT = np.asarray(TBT_degC, dtype=float)
        N   = N_stages if N_stages is not None else self.N_stages
        T_avg = (TBT + self.T_condenser) / 2.0
        L_v   = _latent_heat(T_avg)
        dT_total = np.clip(TBT - self.T_condenser, 1.0, None)
        # Approximate: GOR = eta_HX * cp_brine * dT_total * N / L_v / N = eta_HX * cp_brine * dT/L_v
        # More stages don't change GOR if dT is fixed — but in practice N_stages sets the
        # granularity. With more stages, same total dT is split more finely → slightly higher GOR.
        N_factor = np.sqrt(N / 20.0)  # normalized by typical 20-stage plant; sqrt captures diminishing returns
        GOR = self.eta_HX * CP_BRINE * dT_total * N_factor / L_v
        return np.clip(GOR, 2.0, 15.0)

    def thermal_sec_kwh_m3(self, TBT_degC, steam_temperature_degC=None):
        """Thermal specific energy consumption [kWh_th/m3 distillate].
        SEC_th = L_v(T_steam) / GOR / 3600 * 1000   [kJ/kg → kWh/m3 (density=1)]
        Penalty applied above scaling temperature.
        """
        TBT     = np.asarray(TBT_degC, dtype=float)
        T_steam = steam_temperature_degC if steam_temperature_degC is not None else TBT + 10.0
        GOR_val = self.gor(TBT)
        L_v     = _latent_heat(T_steam)
        SEC_th  = L_v / GOR_val / 3.6  # kJ/kg / 3.6 = kWh/m3

        # Scaling penalty above limit
        excess = np.clip(TBT - self.T_scale_lim, 0.0, None)
        scale_penalty = 1.0 + self.k_scale * excess
        return np.clip(SEC_th * scale_penalty, 30.0, 400.0)

    def pump_sec_kwh_m3(self, plr):
        """Pumping SEC [kWh_e/m3].
        Scales roughly linearly with flow (fixed head, variable flow).
        At part-load: efficiency drops; SEC ∝ 1/PLR^0.5 (affinity laws partially apply).
        """
        plr = np.asarray(plr, dtype=float)
        sec = self.pump_sec_ref * (0.8 + 0.2 / np.clip(plr, 0.1, 1.0))
        return np.clip(sec, 0.5, 10.0)

    def scaling_risk(self, TBT_degC):
        """Scaling risk index [0-1].
        0 = no risk (T < carbonate limit ~90C)
        1 = high sulfate scale risk (T > 110C)
        Linear interpolation between 90-110C.
        """
        TBT = np.asarray(TBT_degC, dtype=float)
        risk = np.clip((TBT - self.T_scale_lim) / 20.0, 0.0, 1.0)
        return risk

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, TBT_degC, plr, steam_temperature_degC=None):
        """Full computation.

        Parameters
        ----------
        TBT_degC                : degC  — top brine temperature (70-120 degC)
        plr                     : 0-1   — plant load ratio
        steam_temperature_degC  : degC  — steam supply temperature (default: TBT + 10)

        Returns
        -------
        dict with gor, thermal_sec_kwh_m3, pump_sec_kwh_m3, total_sec_kwh_m3,
                  scaling_risk_index
        """
        TBT = np.asarray(TBT_degC, dtype=float)
        plr = np.asarray(plr, dtype=float)

        GOR_val    = self.gor(TBT)
        th_sec     = self.thermal_sec_kwh_m3(TBT, steam_temperature_degC)
        pump_sec   = self.pump_sec_kwh_m3(plr)
        total_sec  = th_sec + pump_sec
        scale_risk = self.scaling_risk(TBT)

        return {
            "gor":                  GOR_val,
            "thermal_sec_kwh_m3":   th_sec,
            "pump_sec_kwh_m3":      pump_sec,
            "total_sec_kwh_m3":     total_sec,
            "scaling_risk_index":   scale_risk,
        }
