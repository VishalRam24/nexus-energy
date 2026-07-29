"""
EC138 — Ocean Thermal Energy Conversion (OTEC) — F1a Efficiency Curve

OTEC is a low-ΔT Rankine cycle (closed-cycle, ammonia working fluid):

    eta_Carnot = 1 - T_cold_K / T_warm_K

    eta_gross  = eta_Carnot * eta_cycle_fraction      (gross cycle efficiency)

    eta_net    = eta_gross * (1 - parasitic_fraction)  (net after pumping loads)

    P_net = P_gross * (1 - parasitic_fraction)

Typical values (ΔT ≈ 20°C):
    eta_Carnot ~ 6.7%, eta_gross ~ 3-5%, eta_net ~ 2-3%

References:
    Vega, L.A. (2002). Ocean Thermal Energy Conversion Primer. Mar. Technol. Soc. J., 36(4), 25-35.
    Nihous, G.C. (2007). J. Energy Resour. Technol., 129(1), 10-17.
    Faizal, M. & Ahmed, M.R. (2011). Int. J. Low-Carbon Tech., 6, 215-226.
"""

import numpy as np


class OTECF1a:
    """OTEC closed-cycle ammonia ORC — efficiency as f(T_warm, T_cold)."""

    def __init__(self, params: dict):
        c = params["cycle"]
        self.eta_cycle_frac    = c["eta_cycle_fraction"]["value"]
        self.parasitic_frac    = c["parasitic_fraction"]["value"]
        self.P_gross           = c["P_gross_kw"]["value"]           # kW
        self.q_warm_per_kw     = c["Q_warm_kg_per_s_per_kw"]["value"]
        self.q_cold_per_kw     = c["Q_cold_kg_per_s_per_kw"]["value"]

    # ------------------------------------------------------------------
    def eta_carnot(self, T_warm_c, T_cold_c):
        """Carnot efficiency for given water temperatures."""
        T_warm = np.asarray(T_warm_c, dtype=float) + 273.15
        T_cold = np.asarray(T_cold_c, dtype=float) + 273.15
        dT = T_warm - T_cold
        return np.where(dT > 0.0, 1.0 - T_cold / T_warm, 0.0)

    def eta_gross(self, T_warm_c, T_cold_c):
        """Gross cycle efficiency (before parasitic pumping losses)."""
        return self.eta_carnot(T_warm_c, T_cold_c) * self.eta_cycle_frac

    def eta_net(self, T_warm_c, T_cold_c):
        """Net efficiency (after parasitic pumping losses)."""
        return self.eta_gross(T_warm_c, T_cold_c) * (1.0 - self.parasitic_frac)

    # ------------------------------------------------------------------
    def power_flows(self, T_warm_c, T_cold_c, Q_thermal_kw=None):
        """
        Compute power and heat flows.

        Parameters
        ----------
        T_warm_c     : warm surface water temperature [degC]
        T_cold_c     : cold deep water temperature [degC]
        Q_thermal_kw : thermal input available [kW]; if None, uses rated P_gross

        Returns
        -------
        dict: eta_carnot, eta_gross, eta_net, P_gross_kw, P_net_kw, P_parasitic_kw
        """
        e_c   = self.eta_carnot(T_warm_c, T_cold_c)
        e_g   = self.eta_gross(T_warm_c, T_cold_c)
        e_n   = self.eta_net(T_warm_c, T_cold_c)

        if Q_thermal_kw is None:
            # Back-calculate from rated gross power at rated conditions
            e_g_scalar = float(np.atleast_1d(e_g).flat[0])
            Q_th = self.P_gross / max(e_g_scalar, 1e-9) if e_g_scalar > 0 else 0.0
            Q_thermal_kw = np.broadcast_to(Q_th, np.shape(e_g))

        Q_th = np.asarray(Q_thermal_kw, dtype=float)
        P_gross   = Q_th * e_g
        P_net     = P_gross * (1.0 - self.parasitic_frac)
        P_par     = P_gross - P_net

        return {
            "eta_carnot":    e_c,
            "eta_gross":     e_g,
            "eta_net":       e_n,
            "P_gross_kw":    P_gross,
            "P_net_kw":      P_net,
            "P_parasitic_kw": P_par,
        }

    def water_flows(self, P_gross_kw):
        """Required seawater flow rates for given gross power."""
        P_gross_kw = np.asarray(P_gross_kw, dtype=float)
        return {
            "Q_warm_kg_s": P_gross_kw * self.q_warm_per_kw,
            "Q_cold_kg_s": P_gross_kw * self.q_cold_per_kw,
        }
