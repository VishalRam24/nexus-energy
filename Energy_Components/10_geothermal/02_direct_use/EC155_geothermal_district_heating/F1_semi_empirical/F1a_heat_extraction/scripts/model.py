"""
EC155 — Geothermal District Heating — F1a Heat Extraction Model

Direct use of geothermal water (50–150°C) for district heating — no power generation.
Geothermal fluid passes through a heat exchanger, transferring heat to the network.

Equations:
    Q_extracted  = m_dot_geo * cp * (T_source - T_return)         [kW]
    Q_transferred = Q_extracted * eta_HX                            [kW]  (HX losses)
    Q_delivered  = Q_transferred * (1 - f_dist_loss)               [kW]  (pipe losses)
    coefficient  = Q_delivered / Q_extracted                        [-]   (total coefficient, ~0.85-0.95)
    W_pumps      = f_pump * Q_extracted                             [kW]  (parasitic)
    COP_heat     = Q_delivered / W_pumps                            [-]   (for completeness)

No power conversion — this is a direct-use thermal system.

Reference:
    Lund, J.W. & Toth, A.N. (2021). Direct Utilization of Geothermal Energy 2020
    Worldwide Review. Geothermics, 90, 101915.
    Rybach, L. (2003). Geothermal energy: sustainability and the environment.
    Geothermics, 32(4-6), 463-470.
"""

import numpy as np


class GeothermalDistrictHeatingF1a:
    """
    Geothermal district heating system — heat extraction and delivery model.
    Direct-use application: no power generation.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_geo           = u["cp_geo"]["value"]                    # J/(kg·K)
        self.eta_hx           = u["heat_transfer_efficiency"]["value"]  # dimensionless
        self.f_dist_loss      = u["distribution_losses"]["value"]       # dimensionless
        self.f_pump           = u["pump_power_fraction"]["value"]       # dimensionless

    def heat_extracted(self, T_source_c, T_return_c, m_dot_kgs):
        """
        Thermal power extracted from geothermal fluid (kW).
        Q_ext = m_dot * cp * (T_source - T_return)

        Parameters
        ----------
        T_source_c : float or array — geothermal supply temperature (degC)
        T_return_c : float or array — return temperature to reinjection well (degC)
        m_dot_kgs  : float or array — geothermal fluid flow rate (kg/s)

        Returns
        -------
        Q_ext : kW
        """
        T_source = np.asarray(T_source_c, dtype=float)
        T_return = np.asarray(T_return_c, dtype=float)
        m_dot    = np.asarray(m_dot_kgs,  dtype=float)
        dT = np.clip(T_source - T_return, 0.0, None)
        return m_dot * self.cp_geo * dT / 1000.0  # W → kW

    def heat_transferred(self, T_source_c, T_return_c, m_dot_kgs):
        """
        Heat transferred across HX to network side (kW).
        Q_trans = Q_ext * eta_HX
        """
        Q_ext = self.heat_extracted(T_source_c, T_return_c, m_dot_kgs)
        return Q_ext * self.eta_hx

    def heat_delivered(self, T_source_c, T_return_c, m_dot_kgs):
        """
        Heat delivered to end users after pipeline distribution losses (kW).
        Q_del = Q_trans * (1 - f_dist_loss)
        """
        Q_trans = self.heat_transferred(T_source_c, T_return_c, m_dot_kgs)
        return Q_trans * (1.0 - self.f_dist_loss)

    def heat_coefficient(self, T_source_c, T_return_c, m_dot_kgs):
        """
        Overall heat delivery coefficient = Q_delivered / Q_extracted.
        Typical: 0.85–0.95 (includes HX + pipeline losses).
        """
        Q_ext = self.heat_extracted(T_source_c, T_return_c, m_dot_kgs)
        Q_del = self.heat_delivered(T_source_c, T_return_c, m_dot_kgs)
        # avoid division by zero
        return np.where(Q_ext > 0.0, Q_del / Q_ext, 0.0)

    def pump_power(self, T_source_c, T_return_c, m_dot_kgs):
        """
        Circulation pump electrical consumption (kW).
        W_pump = f_pump * Q_extracted
        """
        Q_ext = self.heat_extracted(T_source_c, T_return_c, m_dot_kgs)
        return self.f_pump * Q_ext

    def system_cop(self, T_source_c, T_return_c, m_dot_kgs):
        """
        System COP = Q_delivered / W_pump.
        Very high (30–50) since pump work is small compared to Q.
        """
        Q_del = self.heat_delivered(T_source_c, T_return_c, m_dot_kgs)
        W_p   = self.pump_power(T_source_c, T_return_c, m_dot_kgs)
        W_p   = np.where(W_p > 0.0, W_p, 1e-6)
        return Q_del / W_p
