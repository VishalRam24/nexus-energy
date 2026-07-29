"""
EC168 -- MPPT Controller -- F1b P&O Algorithm Loss Model

Extends F1a (exponential tracking efficiency curve) with physics-based loss mechanisms:

1. Convergence oscillation loss (steady-state):
   The P&O algorithm oscillates around the MPP by +/- V_step.
   Power loss due to oscillation:
       P_osc = 0.5 * |dP/dV|_mpp * V_step^2 / V_mpp
   This gives eta_static = 1 - P_osc / P_mpp

2. Dynamic tracking loss (transient):
   During irradiance ramps, the algorithm needs time to track.
   Dynamic efficiency drop:
       delta_eta_dyn = |dG/dt| * T_mppt / G
   Represents the fraction of power lost during one perturbation cycle
   while the MPP is shifting.

3. Converter losses:
   The DC-DC converter (buck or boost) has its own efficiency:
       P_out = P_tracked * eta_converter

4. Total MPPT system efficiency:
   eta_total = eta_static * eta_dynamic * eta_converter

Where:
   eta_static  = 1 - 0.5 * |dP/dV| * V_step^2 / (P_mpp * V_mpp)
   eta_dynamic = 1 - |dG/dt| * T_mppt / max(G, G_min)
   eta_converter = constant (from buck/boost F1b model)

Reference:
    Hohm, D.P. & Ropp, M.E. (2003). Comparative study of maximum power point
    tracking algorithms. Progress in Photovoltaics, 11, 47-62.
    Femia, N. et al. (2005). Optimization of perturb and observe maximum power
    point tracking method. IEEE Trans. Power Electron., 20(4), 963-973.
"""

import numpy as np


class MPPTF1b:
    """MPPT controller -- P&O algorithm with detailed loss breakdown."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_step = u["V_step"]["value"]               # V
        self.T_mppt = u["T_mppt"]["value"]               # s
        self.P_max = u["P_max"]["value"]                  # W
        self.eta_static_nom = u["eta_static"]["value"]    # dimensionless
        self.eta_converter = u["eta_converter"]["value"]  # dimensionless
        self.V_mpp_ref = u["V_mpp_ref"]["value"]          # V
        self.dPdV_mpp = u["dPdV_mpp"]["value"]            # W/V (negative)
        self.G_ref = u["G_ref"]["value"]                  # W/m2
        self.tau_track = u["tau_track"]["value"]           # s

    def static_tracking_efficiency(self, irradiance, p_mpp_available):
        """
        Static (steady-state) tracking efficiency.
        Oscillation loss from P&O perturbation around MPP.

        eta_static = 1 - 0.5 * |dP/dV| * V_step^2 / (P_mpp * V_mpp)

        At low irradiance, P_mpp is small -> oscillation loss fraction grows.
        """
        G = np.asarray(irradiance, dtype=float)
        P_mpp = np.asarray(p_mpp_available, dtype=float)

        # Scale |dP/dV| with irradiance (P-V curve flattens at low G)
        dPdV_scaled = np.abs(self.dPdV_mpp) * G / self.G_ref

        # Oscillation power loss
        P_osc = 0.5 * dPdV_scaled * self.V_step ** 2 / self.V_mpp_ref

        # Efficiency
        safe_P = np.where(P_mpp > 0, P_mpp, 1.0)
        eta = np.where(P_mpp > 0, 1.0 - P_osc / safe_P, 0.0)
        return np.clip(eta, 0.0, self.eta_static_nom)

    def dynamic_tracking_efficiency(self, irradiance, dG_dt):
        """
        Dynamic tracking efficiency during irradiance transients.

        eta_dynamic = 1 - |dG/dt| * T_mppt / max(G, G_min)

        Fast irradiance changes cause the MPP to shift faster than the
        algorithm can track, proportional to the ratio of ramp rate to
        irradiance and the perturbation period.
        """
        G = np.asarray(irradiance, dtype=float)
        dG = np.asarray(dG_dt, dtype=float)

        # Minimum irradiance to avoid division by zero
        G_safe = np.where(G > 10.0, G, 10.0)

        # Dynamic loss fraction
        loss_frac = np.abs(dG) * self.T_mppt / G_safe

        eta = np.where(G > 0, 1.0 - loss_frac, 0.0)
        return np.clip(eta, 0.5, 1.0)  # floor at 50% -- below this tracking is lost

    def total_efficiency(self, irradiance, p_mpp_available, dG_dt=0.0):
        """
        Total MPPT system efficiency:
        eta_total = eta_static * eta_dynamic * eta_converter
        """
        G = np.asarray(irradiance, dtype=float)
        eta_s = self.static_tracking_efficiency(irradiance, p_mpp_available)
        eta_d = self.dynamic_tracking_efficiency(irradiance, dG_dt)
        eta_total = eta_s * eta_d * self.eta_converter
        return np.where(G > 0, eta_total, 0.0)

    def output_power(self, irradiance, p_mpp_available, dG_dt=0.0):
        """Output power [W] after all MPPT losses."""
        P_mpp = np.asarray(p_mpp_available, dtype=float)
        eta = self.total_efficiency(irradiance, p_mpp_available, dG_dt)
        return P_mpp * eta

    def loss_breakdown(self, irradiance, p_mpp_available, dG_dt=0.0):
        """
        Breakdown of losses [W]:
        - Oscillation loss (static tracking)
        - Dynamic tracking loss
        - Converter loss
        """
        P_mpp = np.asarray(p_mpp_available, dtype=float)
        G = np.asarray(irradiance, dtype=float)

        eta_s = self.static_tracking_efficiency(irradiance, p_mpp_available)
        eta_d = self.dynamic_tracking_efficiency(irradiance, dG_dt)

        p_after_static = P_mpp * eta_s
        p_osc_loss = P_mpp - p_after_static

        p_after_dynamic = p_after_static * eta_d
        p_dynamic_loss = p_after_static - p_after_dynamic

        p_after_converter = p_after_dynamic * self.eta_converter
        p_converter_loss = p_after_dynamic - p_after_converter

        return {
            "p_oscillation_loss_w": np.where(G > 0, p_osc_loss, 0.0),
            "p_dynamic_loss_w": np.where(G > 0, p_dynamic_loss, 0.0),
            "p_converter_loss_w": np.where(G > 0, p_converter_loss, 0.0),
        }

    def total_losses(self, irradiance, p_mpp_available, dG_dt=0.0):
        """Total losses [W]."""
        P_mpp = np.asarray(p_mpp_available, dtype=float)
        p_out = self.output_power(irradiance, p_mpp_available, dG_dt)
        return P_mpp - p_out
