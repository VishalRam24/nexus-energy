"""
EC129 — Run-of-River Hydropower — F1b Head-Flow Model

Extends F1a with:
1. Variable head support: H_net = H_gross * (1 - f_loss)
   Penstock head losses scale with Q^2 (Darcy-Weisbach):
   f_loss(Q) = f_loss_design * (Q / Q_design)^2

2. 2D efficiency hill chart: eta(q, h) = eta_peak * (1 - k_q*(q-1)^2) * (1 - k_h*(h-1)^2)
   where q = Q/Q_design, h = H_net/H_design

3. Cavitation limit (Thoma sigma):
   sigma_plant = (H_atm - H_vapor - H_draft) / H_net
   Cavitation occurs if sigma_plant < sigma_critical
   -> efficiency derated by (1 - k_cav * max(sigma_critical/sigma_plant - 1, 0))

4. Water temperature effect on density:
   rho(T) ≈ 999.8 - 0.04 * (T - 4)^2 / (T + 283)  [simplified polynomial, kg/m3]

Key differences from EC128 (large dam):
  - Low head (2-20 m), high flow, Kaplan/propeller turbines dominant
  - Short penstock → head loss fraction significant
  - Variable seasonal river flow → efficiency varies with Q
  - Environmental (ecological) flow constraint (min Q_eco)

References:
    Penche (1998), "Layman's Guidebook on How to Develop a Small Hydro Site", EC DGXVII.
    Gordon (2001), "Hydraulic turbine efficiency", Can. J. Civ. Eng., 28, 238-247.
    Papantonis (2001), Small Hydro Plants. Simeon Publishers.
    IEC 60041:1991, Field acceptance tests for hydraulic turbines.
"""

import numpy as np


def _water_density(T_water_C):
    """Approximate water density [kg/m3] as function of temperature [°C]."""
    T = np.asarray(T_water_C, dtype=float)
    # Simplified polynomial: rho peaks ~999.97 kg/m3 at 4°C
    return 999.97 - 0.05 * (T - 4.0) ** 2 / (T + 300.0)


class RunOfRiverF1b:
    """Run-of-river hydropower — variable head, hill chart, cavitation, water temperature."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_design   = u["H_design"]["value"]           # m (net at design)
        self.Q_design   = u["Q_design"]["value"]           # m3/s
        self.P_rated    = u["P_rated"]["value"]            # kW
        self.eta_peak   = u["eta_peak"]["value"]
        self.eta_gen    = u["eta_generator"]["value"]
        self.k_q        = u["k_efficiency_q"]["value"]     # flow curvature
        self.k_h        = u["k_efficiency_h"]["value"]     # head curvature
        self.q_min      = u["q_min"]["value"]
        self.q_max      = u["q_max"]["value"]
        self.h_min      = u["h_min"]["value"]              # min head ratio
        self.h_max      = u["h_max"]["value"]
        self.f_loss_ref = u["head_loss_fraction"]["value"] # at Q_design
        self.rho_ref    = u["rho_ref"]["value"]            # kg/m3 (at T_ref)
        self.T_ref      = u["T_water_ref"]["value"]        # °C
        self.g          = u["g"]["value"]
        self.Q_eco      = u["Q_eco"]["value"]              # m3/s ecological min flow
        # Cavitation
        self.sigma_c    = u["sigma_critical"]["value"]     # Thoma sigma limit
        self.H_draft    = u["H_draft"]["value"]            # m (draft tube head)
        self.k_cav      = u["k_cavitation"]["value"]       # derate coefficient

    # ------------------------------------------------------------------
    # Head losses (variable with Q^2)
    # ------------------------------------------------------------------

    def penstock_head_loss_fraction(self, Q_m3s):
        """Penstock head loss fraction, scales as Q^2."""
        Q = np.asarray(Q_m3s, dtype=float)
        q = np.clip(Q / self.Q_design, 0.0, 2.0)
        return self.f_loss_ref * q ** 2

    def net_head(self, gross_head_m, Q_m3s):
        """Net head after penstock losses [m]."""
        H_gross = np.asarray(gross_head_m, dtype=float)
        f = self.penstock_head_loss_fraction(Q_m3s)
        return H_gross * (1.0 - f)

    # ------------------------------------------------------------------
    # 2D hill chart efficiency
    # ------------------------------------------------------------------

    def turbine_efficiency(self, Q_m3s, H_net_m):
        """
        2D efficiency hill chart: eta(q,h) = eta_peak*(1-k_q*(q-1)^2)*(1-k_h*(h-1)^2).
        Returns 0 outside operating range.
        """
        Q = np.asarray(Q_m3s, dtype=float)
        H = np.asarray(H_net_m, dtype=float)
        q = Q / self.Q_design
        h = H / self.H_design

        eta_q = 1.0 - self.k_q * (q - 1.0) ** 2
        eta_h = 1.0 - self.k_h * (h - 1.0) ** 2
        eta_t = self.eta_peak * eta_q * eta_h

        out_of_range = (
            (q < self.q_min) | (q > self.q_max) |
            (h < self.h_min) | (h > self.h_max)
        )
        eta_t = np.where(out_of_range, 0.0, eta_t)
        return np.clip(eta_t, 0.0, self.eta_peak)

    # ------------------------------------------------------------------
    # Cavitation derate
    # ------------------------------------------------------------------

    def cavitation_derate(self, H_net_m):
        """
        Cavitation derating factor [dimensionless].

        sigma_plant = (H_atm - H_vapor - H_draft) / H_net
        Rated H_atm - H_vapor ≈ 10.0 m (at sea level, ~20°C)

        If sigma_plant < sigma_critical, efficiency is derated by:
          derate = 1 - k_cav * (sigma_critical/sigma_plant - 1)

        Returns value in (0, 1].
        """
        H = np.asarray(H_net_m, dtype=float)
        H_safe = np.maximum(H, 0.1)
        # Net positive suction head available
        H_atm_minus_vapor = 10.0   # m (approximate, sea level 20°C)
        sigma_plant = (H_atm_minus_vapor - self.H_draft) / H_safe
        sigma_ratio = np.where(sigma_plant > 0, self.sigma_c / sigma_plant, 10.0)
        excess = np.maximum(sigma_ratio - 1.0, 0.0)
        derate = np.maximum(1.0 - self.k_cav * excess, 0.0)
        return np.where(H <= 0, 0.0, derate)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------

    def overall_efficiency(self, Q_m3s, H_net_m):
        """Overall plant efficiency = eta_turbine * eta_generator * cavitation_derate."""
        eta_t = self.turbine_efficiency(Q_m3s, H_net_m)
        cav = self.cavitation_derate(H_net_m)
        return np.where(eta_t > 0, eta_t * self.eta_gen * cav, 0.0)

    def power_kw(self, Q_m3s, gross_head_m, T_water_C=None):
        """
        Electrical output power [kW].

        P = eta_overall * rho(T) * g * Q_avail * H_net / 1000

        Q_avail = max(Q - Q_eco, 0) — ecological flow constraint.
        """
        Q_total = np.asarray(Q_m3s, dtype=float)
        Q_avail = np.maximum(Q_total - self.Q_eco, 0.0)

        H_net = self.net_head(gross_head_m, Q_avail)
        rho = self.rho_ref if T_water_C is None else _water_density(T_water_C)
        eta = self.overall_efficiency(Q_avail, H_net)
        P = eta * rho * self.g * Q_avail * H_net / 1000.0
        return np.clip(P, 0.0, self.P_rated)

    def capacity_factor(self, Q_m3s, gross_head_m, T_water_C=None):
        """Capacity factor = P / P_rated."""
        return self.power_kw(Q_m3s, gross_head_m, T_water_C) / self.P_rated

    def flow_ratio(self, Q_m3s):
        """q = Q / Q_design."""
        return np.asarray(Q_m3s, dtype=float) / self.Q_design
