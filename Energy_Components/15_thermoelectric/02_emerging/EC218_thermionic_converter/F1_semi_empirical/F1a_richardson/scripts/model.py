"""
EC218 — Thermionic Converter — F1a Richardson-Dushman Model

Richardson-Dushman thermionic emission:
    J = A * T^2 * exp(-phi / (k_B * T))   [A/m^2]

Net current density (emitter minus back-emission from collector):
    J_net = J_emitter - J_collector

Output voltage (ideal, no space-charge or plasma):
    V_out = (phi_emitter - phi_collector) / q - V_loss
    simplified: V_out = (phi_e - phi_c) / q

Power density:
    P_density = J_net * V_out   [W/m^2]
    P_total = P_density * emitter_area

Efficiency:
    Q_in = J_emitter * (phi_emitter/q + 2*k_B*T_e/q) * emitter_area  (approximate)
    eta = P_total / Q_in

References:
    Hatsopoulos, G.N. & Gyftopoulos, E.P. (1979). Thermionic Energy Conversion.
    Angrist, S.W. (1982). Direct Energy Conversion. Allyn & Bacon.
"""

import numpy as np

# Physical constants
k_B = 1.380649e-23   # J/K
q_e = 1.602176634e-19  # C


class ThermionicF1a:
    """Thermionic converter — Richardson-Dushman emission model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.phi_e = u["phi_emitter"]["value"]        # eV
        self.phi_c = u["phi_collector"]["value"]      # eV
        self.A_r = u["A_richardson"]["value"]         # A/(m^2*K^2)
        self.area = u["emitter_area"]["value"]        # m^2

    def _emission_density(self, phi_ev, T_K):
        """Richardson-Dushman current density [A/m^2]."""
        T = np.asarray(T_K, dtype=float)
        phi_J = phi_ev * q_e   # convert eV -> J
        return self.A_r * T**2 * np.exp(-phi_J / (k_B * T))

    def compute(self, T_emitter_K, T_collector_K):
        """
        Parameters
        ----------
        T_emitter_K  : float or array — emitter temperature [K]
        T_collector_K: float or array — collector temperature [K]

        Returns
        -------
        dict: J_emitter, J_collector, J_net, V_out, power_w, heat_input_w, efficiency
        """
        T_e = np.asarray(T_emitter_K, dtype=float)
        T_c = np.asarray(T_collector_K, dtype=float)

        J_e = self._emission_density(self.phi_e, T_e)
        J_c = self._emission_density(self.phi_c, T_c)
        J_net = np.maximum(J_e - J_c, 0.0)

        # Output voltage: work function difference (ideal Cs-diode)
        V_out = np.maximum((self.phi_e - self.phi_c), 0.0)  # eV

        P_density = J_net * V_out  # W/m^2  (eV * A/m^2 = W/m^2 since 1 eV * 1 A/m^2 = 1 W/m^2 by convention when V in volts)
        P_total = P_density * self.area

        # Heat input approximation: emitted electrons carry kinetic energy
        # Q_in ~ J_e * (phi_e + 2*k_B*T_e/q_e) * area  [W]
        Q_in = J_e * (self.phi_e + 2.0 * k_B * T_e / q_e) * self.area
        Q_in = np.maximum(Q_in, 1e-12)

        eta = np.where(Q_in > 0, P_total / Q_in, 0.0)
        eta = np.clip(eta, 0.0, 0.5)

        return {
            "J_emitter_Am2": J_e,
            "J_collector_Am2": J_c,
            "J_net_Am2": J_net,
            "V_out_V": V_out * np.ones_like(T_e),
            "power_w": P_total,
            "heat_input_w": Q_in,
            "efficiency": eta,
        }
