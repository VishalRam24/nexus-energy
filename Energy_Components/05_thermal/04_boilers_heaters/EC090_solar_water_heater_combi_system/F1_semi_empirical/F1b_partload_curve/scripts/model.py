"""
EC090 — Solar Water Heater (Combi System) — F1b Part-Load with Auxiliary Backup

A solar water heater combi-system uses solar collectors to preheat domestic hot water
and space heating, with an auxiliary boiler (gas/electric) for top-up.

Physics additions over F1a (constant efficiency):
  1. Solar fraction calculation: SF = Q_solar / Q_demand
     Q_solar = eta_collector * A_coll * I_solar  [kW]
  2. Part-load penalty for auxiliary backup: eta_aux(PLR) = a0 + a1*PLR + a2*PLR^2
  3. Standby losses from storage tank: Q_standby = UA_tank * (T_tank - T_amb)
  4. Collector efficiency curve: eta_coll(DeltaT, I) = eta0 - a1_coll*dT/I - a2_coll*(dT/I)^2
     where dT = T_mean_collector - T_ambient  (Hottel-Whillier-Bliss)

Energy balance:
    Q_solar  = eta_coll(dT, I) * A_coll * I  [kW]
    Q_needed = max(Q_demand - Q_solar, 0)
    PLR_aux  = min(Q_needed / Q_aux_rated, 1.0)
    Q_aux    = eta_aux(PLR_aux) * fuel_input   [via eta_aux(PLR)]
    Q_standby = UA_tank * (T_tank - T_amb) / 1000

References:
    EN 12975:2006 — Thermal solar systems and components (collector efficiency).
    Hottel, H.C. & Whillier, A. (1958). Trans. ASME 80, 773-786.
    Duffie, J.A. & Beckman, W.A. (2013). Solar Engineering of Thermal Processes. Wiley.
    Haller, M. et al. (2013). Solar Energy 95, 281-293.
"""

import numpy as np


class SolarWaterHeaterF1b:
    """Solar combi-system with collector efficiency curve and auxiliary part-load backup."""

    def __init__(self, params: dict):
        # Collector
        self.A_coll    = float(params["A_collector_m2"])
        self.eta0      = float(params["eta0"])           # zero-loss efficiency
        self.a1_coll   = float(params["a1_collector"])   # W/(m2·K) linear loss
        self.a2_coll   = float(params["a2_collector"])   # W/(m2·K2) quadratic loss
        self.T_coll    = float(params["T_collector_mean_c"])  # degC, mean collector temp

        # Auxiliary boiler
        self.Q_aux_rated = float(params["Q_aux_rated_kw"])
        self.a0_aux      = float(params["a0_aux"])
        self.a1_aux      = float(params["a1_aux"])
        self.a2_aux      = float(params["a2_aux"])
        self.PLR_min_aux = float(params["PLR_min_aux"])

        # Storage tank
        self.UA_tank    = float(params["UA_tank_w_per_k"])  # W/K
        self.T_tank     = float(params["T_tank_c"])
        self.T_amb_design = float(params["T_ambient_design"])

    # ------------------------------------------------------------------
    # Collector efficiency (Hottel-Whillier-Bliss)
    # ------------------------------------------------------------------

    def collector_efficiency(self, I_solar_w_m2, T_ambient):
        """
        eta_coll = eta0 - a1*(dT/I) - a2*(dT/I)^2
        dT = T_collector_mean - T_ambient
        I_solar in W/m2.
        Returns 0 when I_solar <= 0.
        """
        I     = np.asarray(I_solar_w_m2, dtype=float)
        T_amb = np.asarray(T_ambient, dtype=float)
        dT    = self.T_coll - T_amb
        eta   = np.where(
            I > 1.0,
            self.eta0 - self.a1_coll * dT / I - self.a2_coll * (dT / I) ** 2,
            0.0,
        )
        return np.clip(eta, 0.0, self.eta0)

    # ------------------------------------------------------------------
    # Solar yield
    # ------------------------------------------------------------------

    def solar_yield_kw(self, I_solar_w_m2, T_ambient):
        """Q_solar = eta_coll * A_coll * I_solar  [kW]"""
        eta = self.collector_efficiency(I_solar_w_m2, T_ambient)
        return eta * self.A_coll * np.asarray(I_solar_w_m2, dtype=float) / 1000.0

    # ------------------------------------------------------------------
    # Auxiliary boiler part-load
    # ------------------------------------------------------------------

    def aux_efficiency(self, PLR):
        """eta_aux(PLR) = a0 + a1*PLR + a2*PLR^2"""
        PLR_eff = np.maximum(np.asarray(PLR, dtype=float), self.PLR_min_aux)
        return np.clip(self.a0_aux + self.a1_aux * PLR_eff + self.a2_aux * PLR_eff ** 2,
                       0.0, 1.0)

    def aux_fuel_kw(self, PLR):
        """Auxiliary fuel input [kW]."""
        PLR_eff = np.maximum(np.asarray(PLR, dtype=float), self.PLR_min_aux)
        Q_heat  = PLR_eff * self.Q_aux_rated
        eta     = self.aux_efficiency(PLR)
        return Q_heat / np.where(eta > 0.01, eta, 0.01)

    # ------------------------------------------------------------------
    # Standby tank loss
    # ------------------------------------------------------------------

    def standby_loss_kw(self, T_ambient=None):
        T_amb = self.T_amb_design if T_ambient is None else float(T_ambient)
        return self.UA_tank * (self.T_tank - T_amb) / 1000.0

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, I_solar_w_m2, Q_demand_kw, T_ambient=None):
        """
        Full system evaluation.

        Parameters
        ----------
        I_solar_w_m2 : float or array — solar irradiance on collector plane [W/m2]
        Q_demand_kw  : float or array — thermal demand (DHW + space heating) [kW]
        T_ambient    : float or array or None — ambient temperature [degC]

        Returns
        -------
        dict with Q_solar_kw, Q_aux_kw, Q_aux_fuel_kw, Q_standby_kw,
                  solar_fraction, eta_collector, eta_aux, PLR_aux
        """
        I     = np.asarray(I_solar_w_m2, dtype=float)
        Q_dem = np.asarray(Q_demand_kw,  dtype=float)
        T_amb = self.T_amb_design if T_ambient is None \
                else np.asarray(T_ambient, dtype=float)

        Q_solar  = self.solar_yield_kw(I, T_amb)
        Q_solar  = np.minimum(Q_solar, Q_dem)  # can't deliver more than demand

        Q_needed = np.maximum(Q_dem - Q_solar, 0.0)
        PLR_aux  = np.clip(Q_needed / self.Q_aux_rated, 0.0, 1.0)

        # Auxiliary only fires if needed
        Q_aux      = np.where(PLR_aux > self.PLR_min_aux,
                               PLR_aux * self.Q_aux_rated, 0.0)
        PLR_aux_eff = np.where(PLR_aux > self.PLR_min_aux, PLR_aux, 0.0)
        Q_aux_fuel  = np.where(PLR_aux > self.PLR_min_aux,
                                self.aux_fuel_kw(PLR_aux), 0.0)

        SF = np.where(Q_dem > 0.01, Q_solar / Q_dem, 0.0)
        SF = np.clip(SF, 0.0, 1.0)

        Q_standby  = np.full_like(I, self.standby_loss_kw(T_amb))
        eta_coll   = self.collector_efficiency(I, T_amb)
        eta_aux    = self.aux_efficiency(PLR_aux_eff)

        return {
            "Q_solar_kw":       Q_solar,
            "Q_aux_kw":         Q_aux,
            "Q_aux_fuel_kw":    Q_aux_fuel,
            "Q_standby_kw":     Q_standby,
            "solar_fraction":   SF,
            "eta_collector":    eta_coll,
            "eta_aux":          eta_aux,
            "PLR_aux":          PLR_aux,
        }
