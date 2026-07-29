"""
EC218 — Thermionic Converter — F1b Space-Charge Corrected Model

Extends F1a with:
1. Temperature-dependent work functions: phi(T) = phi0 + dphi/dT * (T - T0)
2. Space-charge correction (Schottky-Langmuir regime):
     J_corrected = J_RD * f_sc   where f_sc is a phenomenological reduction factor
     In Cs-vapor diodes, Cs ions partially neutralize space charge.
     Full motive calculation requires PDE — here we use a fitted factor.
3. Back-emission from collector at T_collector
4. Lead resistance voltage drop: V_out = V_open - J_net * area * R_lead
5. Heat input: Q_in = emitted electron enthalpy (Nottingham correction not included)

References:
    Hatsopoulos, G.N. & Gyftopoulos, E.P. (1979). Thermionic Energy Conversion. MIT Press.
    Houston, J.M. (1959). J. Appl. Phys. 30(4), 481-487.
    Angrist, S.W. (1982). Direct Energy Conversion. Allyn & Bacon.
    Rasor, N.S. (1991). IEEE Trans. Plasma Sci. 19(6), 1191-1208.
"""

import numpy as np

k_B = 1.380649e-23      # J/K
q_e = 1.602176634e-19   # C


class ThermionicF1b:
    """Thermionic converter with space-charge correction and T-dependent work functions."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.phi_e0 = u["phi_emitter_0"]["value"]          # eV
        self.phi_c0 = u["phi_collector_0"]["value"]        # eV
        self.A_r = u["A_richardson"]["value"]              # A/(m^2*K^2)
        self.area = u["emitter_area"]["value"]             # m^2
        self.gap = u["interelectrode_gap"]["value"]        # m
        self.T0 = u["T0_K"]["value"]                       # K
        self.dphi_e_dT = u["dphi_dT_emitter"]["value"]     # eV/K
        self.dphi_c_dT = u["dphi_dT_collector"]["value"]   # eV/K
        self.f_sc = u["space_charge_factor"]["value"]      # [-]
        self.R_lead = u["lead_resistance_ohm"]["value"]    # ohm

    def _phi_emitter(self, T_K):
        """Temperature-dependent emitter work function [eV]."""
        T = np.asarray(T_K, dtype=float)
        return self.phi_e0 + self.dphi_e_dT * (T - self.T0)

    def _phi_collector(self, T_K):
        """Temperature-dependent collector work function [eV]."""
        T = np.asarray(T_K, dtype=float)
        return self.phi_c0 + self.dphi_c_dT * (T - self.T0)

    def _emission_density(self, phi_ev, T_K, space_charge=True):
        """Richardson-Dushman current density [A/m^2] with optional space-charge."""
        T = np.asarray(T_K, dtype=float)
        phi_J = np.asarray(phi_ev, dtype=float) * q_e
        J_rd = self.A_r * T ** 2 * np.exp(-phi_J / (k_B * T))
        if space_charge:
            return J_rd * self.f_sc
        return J_rd

    def compute(self, T_emitter_K, T_collector_K):
        """
        Parameters
        ----------
        T_emitter_K  : float or array — emitter temperature [K]
        T_collector_K: float or array — collector temperature [K]

        Returns
        -------
        dict: phi_e_eV, phi_c_eV, J_emitter, J_collector, J_net,
              V_open_V, V_terminal_V, power_w, heat_input_w, efficiency
        """
        T_e = np.asarray(T_emitter_K, dtype=float)
        T_c = np.asarray(T_collector_K, dtype=float)

        # Temperature-dependent work functions
        phi_e = self._phi_emitter(T_e)  # eV
        phi_c = self._phi_collector(T_c)  # eV

        # Emission current densities (emitter with space-charge, collector back-emission)
        J_e = self._emission_density(phi_e, T_e, space_charge=True)
        J_c = self._emission_density(phi_c, T_c, space_charge=False)  # back-emission smaller

        J_net = np.maximum(J_e - J_c, 0.0)

        # Open-circuit output voltage: work function difference
        V_open = np.maximum(phi_e - phi_c, 0.0)  # eV = V (since charge is 1 electron)

        # Terminal voltage accounting for lead resistance
        # V_terminal = V_open - J_net * area * R_lead
        V_terminal = V_open - J_net * self.area * self.R_lead
        V_terminal = np.maximum(V_terminal, 0.0)

        # Electrical power output
        P_total = J_net * V_terminal * self.area

        # Heat input: emitted electron enthalpy flux
        # Q_in = J_e * (phi_e + 2*k_B*T_e/q_e) * area
        # The 2*k_B*T/q factor accounts for kinetic energy of emitted electrons
        Q_in = J_e * (phi_e + 2.0 * k_B * T_e / q_e) * self.area
        Q_in = np.maximum(Q_in, 1e-12)

        eta = np.where(Q_in > 0, P_total / Q_in, 0.0)
        eta = np.clip(eta, 0.0, 0.5)

        # Current-voltage characteristic (power density)
        P_density_w_cm2 = J_net * V_terminal * 1e-4  # W/cm^2

        return {
            "phi_e_eV": phi_e,
            "phi_c_eV": phi_c,
            "J_emitter_Am2": J_e,
            "J_collector_Am2": J_c,
            "J_net_Am2": J_net,
            "V_open_V": V_open * np.ones_like(T_e),
            "V_terminal_V": V_terminal * np.ones_like(T_e),
            "power_w": P_total,
            "power_density_w_cm2": P_density_w_cm2,
            "heat_input_w": Q_in,
            "efficiency": eta,
        }
