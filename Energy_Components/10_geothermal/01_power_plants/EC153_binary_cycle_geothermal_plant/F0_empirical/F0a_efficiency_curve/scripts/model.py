"""F0a empirical efficiency-curve model for EC153 Binary Cycle Geothermal Plant.

Simplest fidelity: a 1-D lookup of net plant efficiency vs geothermal
resource temperature, eta_net(T_geo), tabulated from the F1a relation
    eta = eta_utilization * eta_Carnot,  eta_Carnot = 1 - T_reject/T_geo
at the F1a design rejection temperature. Net electric power is then
    P_net = eta_net(T_geo) * Q_resource,
    Q_resource = m_dot * cp * (T_geo - T_reject).

Data source: DiPippo, R. (2015), Geothermal Power Plants, 4th ed.
Breakpoints reuse the EC153 F1a exergy-model design numbers.

Pure NumPy. No scipy / ODEs / AI.
"""
import numpy as np


class EfficiencyCurve:
    def __init__(self, T_geo_degC, eta_net, T_reject_ref, cp, eta_utilization):
        self.T_geo = np.asarray(T_geo_degC, dtype=float)
        self.eta = np.asarray(eta_net, dtype=float)
        self.T_reject_ref = float(T_reject_ref)
        self.cp = float(cp)
        self.eta_utilization = float(eta_utilization)

    def eta_net(self, T_geo):
        """Net plant efficiency at resource temperature (degC), clamped to table ends."""
        return np.interp(T_geo, self.T_geo, self.eta)

    def net_power_kW(self, T_geo, m_dot_kgs, T_reject=None):
        """Net electric power (kW) for a brine/steam stream."""
        if T_reject is None:
            T_reject = self.T_reject_ref
        q_kW = m_dot_kgs * self.cp * (np.asarray(T_geo, float) - T_reject) / 1000.0
        q_kW = np.maximum(q_kW, 0.0)
        return self.eta_net(T_geo) * q_kW
