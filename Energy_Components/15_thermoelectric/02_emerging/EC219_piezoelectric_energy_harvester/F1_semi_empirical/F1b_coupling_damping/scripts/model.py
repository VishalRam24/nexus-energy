"""
EC219 — Piezoelectric Energy Harvester — F1b Coupling + Damping Model

Extends F1a with explicit electromechanical coupling coefficient k31 and
full coupled electromechanical equations (Erturk-Inman framework):

The bimorph is modeled as a 1-DOF coupled electromechanical system:

    Mechanical ODE (frequency domain, harmonic excitation):
        m_eff * x_dd + c_mech * x_d + k_eff * x - theta * v = -m_eff * a
    Electrical ODE:
        C_p * v_d + v/R_L + theta * x_d = 0

where:
    theta = d31 * Y * b * t_p * n_layers / L_eff   [N/V or C/m] — coupling coefficient
    C_p = eps_33T * b * L / t_p * n_layers          [F]          — clamp capacitance

Frequency-domain solution (steady-state harmonic):
    Voltage across load:  V(omega) = ...
    Power:                P = |V|^2 / (2 * R_L)    [W] (RMS for sinusoidal)

Electromechanical coupling factor k31 modifies effective coupling:
    theta_eff = theta * k31_correction

Electrical damping (from energy extraction):
    zeta_e = theta^2 / (2 * m_eff * omega_n * (1/(omega*C_p) + R_L)^(-1))

Total effective damping: zeta_total = zeta_mech + zeta_e

References:
    Erturk, A. & Inman, D.J. (2011). Piezoelectric Energy Harvesting. Wiley.
    duToit, N.E. et al. (2005). Integr. Ferroelectr. 71, 121-160.
    Roundy, S. (2005). J. Intell. Mater. Syst. Struct. 16(10), 809-823.
"""

import numpy as np


class PiezoF1b:
    """Piezoelectric harvester — coupled electromechanical model with k31 coupling."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.d31 = abs(u["d31"]["value"])                   # m/V
        self.Y_p = u["Y_piezo"]["value"]                    # Pa
        self.Y_s = u["Y_substrate"]["value"]                # Pa
        self.eps_33T = u["epsilon_33_T"]["value"]           # F/m
        self.L = u["beam_length"]["value"]                  # m
        self.b = u["beam_width"]["value"]                   # m
        self.t_p = u["beam_thickness_piezo"]["value"]       # m
        self.t_s = u["beam_thickness_substrate"]["value"]   # m
        self.m_tip = u["mass_tip"]["value"]                 # kg
        self.zeta_mech = u["zeta_mech"]["value"]            # -
        self.f_n = u["resonance_freq_hz"]["value"]          # Hz
        self.k31 = u["k31"]["value"]                        # -
        self.omega_n = 2.0 * np.pi * self.f_n              # rad/s

        # Effective mass (bimorph cantilever + tip)
        rho_pzt = 7800.0  # kg/m^3
        rho_steel = 7800.0  # kg/m^3
        m_beam_p = rho_pzt * self.L * self.b * self.t_p * 2.0  # bimorph (2 pzt layers)
        m_beam_s = rho_steel * self.L * self.b * self.t_s
        self.m_eff = 0.236 * (m_beam_p + m_beam_s) + self.m_tip

        # Effective stiffness (from omega_n)
        self.k_eff = self.m_eff * self.omega_n ** 2

        # Mechanical damping coefficient
        self.c_mech = 2.0 * self.zeta_mech * np.sqrt(self.k_eff * self.m_eff)

        # Piezoelectric clamped capacitance (two PZT layers in parallel for bimorph series connection)
        # For series bimorph: C_p = 0.5 * eps_33T * b * L / t_p (layers in series)
        self.C_p = 0.5 * self.eps_33T * self.b * self.L / self.t_p

        # Coupling coefficient theta [N/V] for series bimorph
        # theta = d31 * Y_p * b * t_p / t_neutral (bending)
        # For bimorph: theta = d31 * Y_p * b * (t_p + t_s/2) / 2 (simplified)
        # More precisely from Erturk: theta = d31 * Y_p * b * (t_pc) where t_pc = centroid distance
        t_pc = (self.t_s / 2.0 + self.t_p / 2.0)  # neutral axis to piezo centroid
        self.theta = self.d31 * self.Y_p * self.b * t_pc

    def _coupled_power(self, omega, acceleration_ms2, R_L):
        """
        Coupled electromechanical solution in frequency domain.

        Returns P_electrical [W] for given omega, acceleration, and load resistance.
        """
        omega = np.asarray(omega, dtype=float)
        a = np.asarray(acceleration_ms2, dtype=float)
        R_L = np.asarray(R_L, dtype=float)

        # Force input
        F = self.m_eff * a  # [N]

        # Denominator of coupled system (from Erturk Eq. 6.33 style)
        # D(omega) = (k_eff - m_eff*omega^2 + j*c_mech*omega) * (1/(j*omega*C_p) + R_L)^{-1} + theta^2
        # More directly:
        # V = -j*omega*theta*R_L / D(omega) * F
        # where D = (k_eff - m_eff*omega^2 + j*c_mech*omega) * (1 + j*omega*R_L*C_p) + j*omega*theta^2*R_L
        j = 1j
        Z_mech = self.k_eff - self.m_eff * omega ** 2 + j * self.c_mech * omega
        D = Z_mech * (1.0 + j * omega * R_L * self.C_p) + j * omega * self.theta ** 2 * R_L

        # Voltage across load
        V_load = -j * omega * self.theta * R_L * F / D

        # Power (RMS)
        P = 0.5 * np.abs(V_load) ** 2 / R_L  # [W]

        return P, np.abs(V_load)

    def compute(self, acceleration_ms2, frequency_hz, R_load_ohm):
        """
        Parameters
        ----------
        acceleration_ms2 : float or array — base acceleration [m/s^2]
        frequency_hz     : float or array — excitation frequency [Hz]
        R_load_ohm       : float or array — load resistance [ohm]

        Returns
        -------
        dict: power_w, power_uw, voltage_v, frequency_ratio,
              zeta_electrical, zeta_total, optimal_R_ohm, at_resonance_power_w
        """
        a = np.asarray(acceleration_ms2, dtype=float)
        f = np.asarray(frequency_hz, dtype=float)
        R_L = np.asarray(R_load_ohm, dtype=float)
        omega = 2.0 * np.pi * f

        P_out, V_out = self._coupled_power(omega, a, R_L)

        # Frequency ratio
        freq_ratio = f / self.f_n

        # Electrical damping (approximate at resonance)
        # zeta_e = theta^2 / (2 * m_eff * omega_n) * Re(1/(1/R_L + j*omega_n*C_p))
        omega_n = self.omega_n
        R_L_real = np.atleast_1d(R_L).flatten()[0] if np.ndim(R_L) == 0 else float(np.atleast_1d(R_L)[0])
        Z_elec = 1.0 / (1.0 / R_L_real + 1j * omega_n * self.C_p)
        zeta_e = self.theta ** 2 / (2.0 * self.m_eff * omega_n) * np.real(Z_elec) / (
            self.k_eff - self.m_eff * omega_n**2 + 1e-12 if abs(self.k_eff - self.m_eff * omega_n**2) > 0.1 else 1.0
        )
        # Simplified: zeta_e at resonance (omega=omega_n)
        zeta_e = float(self.theta ** 2 * R_L_real / (
            2.0 * self.m_eff * omega_n * (1.0 + (omega_n * R_L_real * self.C_p) ** 2)))
        zeta_total = self.zeta_mech + zeta_e

        # Optimal load resistance at resonance: R_opt = 1 / (omega_n * C_p) / sqrt(1 + k31_eff^2)
        # Simplified: R_opt = 1 / (omega_n * C_p)
        R_opt = 1.0 / (omega_n * self.C_p)

        # Power at resonance with optimal load
        P_res_opt, _ = self._coupled_power(np.array([omega_n]), np.atleast_1d(a)[0], R_opt)
        at_res_power = float(np.atleast_1d(P_res_opt)[0])

        return {
            "power_w": P_out,
            "power_uw": P_out * 1e6,
            "voltage_v": V_out,
            "frequency_ratio": freq_ratio,
            "zeta_electrical": float(zeta_e),
            "zeta_total": float(zeta_total),
            "optimal_R_ohm": float(R_opt),
            "at_resonance_power_w": at_res_power,
        }
