"""F0a empirical delivered-heat curve for EC155 Geothermal District Heating.

Simplest fidelity: a 1-D lookup of delivered heat per unit geothermal flow
(kW per kg/s) vs source temperature, tabulated from the F1a relation
    q_specific(T_source) = cp*(T_source - T_return)*eta_HX*(1 - dist_loss)
at the F1a design return temperature. Delivered heat is then
    Q_delivered = q_specific(T_source) * m_dot
and circulation pump power Q*pump_power_fraction.

Data source: Lund & Toth (2021), Direct Utilization of Geothermal Energy
2020 Worldwide Review. Breakpoints reuse the EC155 F1a design numbers.

Pure NumPy. No scipy / ODEs / AI.
"""
import numpy as np


class DeliveredHeatCurve:
    def __init__(self, T_source_degC, q_specific, T_return_ref, cp,
                 eta_hx, dist_loss, pump_frac):
        self.T_src = np.asarray(T_source_degC, dtype=float)
        self.q = np.asarray(q_specific, dtype=float)
        self.T_return_ref = float(T_return_ref)
        self.cp = float(cp)
        self.eta_hx = float(eta_hx)
        self.dist_loss = float(dist_loss)
        self.pump_frac = float(pump_frac)

    def q_specific(self, T_source, T_return=None):
        """Delivered heat per kg/s (kW per kg/s) at a source temperature."""
        if T_return is None:
            return np.interp(T_source, self.T_src, self.q)
        # explicit return temperature: evaluate the underlying relation directly
        dT = np.maximum(np.asarray(T_source, float) - T_return, 0.0)
        return self.cp * dT / 1000.0 * self.eta_hx * (1.0 - self.dist_loss)

    def delivered_heat_kW(self, T_source, m_dot_kgs, T_return=None):
        return self.q_specific(T_source, T_return) * m_dot_kgs

    def pump_power_kW(self, Q_delivered_kW):
        return self.pump_frac * np.asarray(Q_delivered_kW, float)
