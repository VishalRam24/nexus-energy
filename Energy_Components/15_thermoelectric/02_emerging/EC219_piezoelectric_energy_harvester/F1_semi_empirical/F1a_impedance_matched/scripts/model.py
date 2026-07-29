"""
EC219 — Piezoelectric Energy Harvester — F1a Impedance-Matched Model

Simplified lumped-parameter model for cantilever PZT bimorph at resonance:

    omega_n = sqrt(k_eff / m_eff)   — natural frequency [rad/s]
    Q_mech  = 1 / (2*zeta)          — mechanical Q factor

At mechanical resonance with impedance-matched load:
    P_out = m_eff^2 * a^2 / (4 * c_mech)  [W]
    where c_mech = 2 * zeta * sqrt(k_eff * m_eff) is the mechanical damping coefficient

Frequency response (off-resonance):
    H(omega) = omega_n^2 / sqrt((omega_n^2 - omega^2)^2 + (2*zeta*omega_n*omega)^2)
    P_out(omega) = P_peak * H(omega)^2 * (effective_factor)

Power scales as P ~ a^2 (fundamental piezoelectric energy harvesting relation).

References:
    Roundy, S. et al. (2003). Smart Mater. Struct. 12(6).
    Erturk, A. & Inman, D.J. (2011). Piezoelectric Energy Harvesting. Wiley.
"""

import numpy as np


class PiezoF1a:
    """Piezoelectric cantilever harvester — impedance-matched lumped model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.d31 = abs(u["d31"]["value"])              # m/V (use magnitude)
        self.Y = u["Y_piezo"]["value"]                 # Pa
        self.L = u["beam_length"]["value"]             # m
        self.w = u["beam_width"]["value"]              # m
        self.t_p = u["beam_thickness_piezo"]["value"]  # m
        self.m_tip = u["mass_tip"]["value"]            # kg
        self.zeta = u["damping_ratio"]["value"]        # dimensionless
        self.f_n = u["resonance_freq_hz"]["value"]     # Hz
        self.omega_n = 2.0 * np.pi * self.f_n         # rad/s

        # Effective mass (cantilever + tip; lumped approx)
        # Distributed beam mass + tip mass
        rho_pzt = 7800.0  # kg/m^3 typical
        m_beam = rho_pzt * self.L * self.w * self.t_p * 2.0  # bimorph
        self.m_eff = 0.236 * m_beam + self.m_tip  # 0.236 = cantilever distribution factor

        # Effective stiffness
        self.k_eff = self.m_eff * self.omega_n**2

        # Mechanical damping coefficient
        self.c_mech = 2.0 * self.zeta * np.sqrt(self.k_eff * self.m_eff)

    def _frequency_response(self, omega):
        """Amplitude ratio at given frequency relative to resonance."""
        # |H(omega)| = omega_n^2 / sqrt((omega_n^2-omega^2)^2 + (2*zeta*omega_n*omega)^2)
        num = self.omega_n**2
        denom = np.sqrt((self.omega_n**2 - omega**2)**2 + (2*self.zeta*self.omega_n*omega)**2)
        return num / (denom + 1e-30)

    def compute(self, acceleration_ms2, frequency_hz):
        """
        Parameters
        ----------
        acceleration_ms2 : float or array — base acceleration [m/s^2]
        frequency_hz     : float or array — excitation frequency [Hz]

        Returns
        -------
        dict: power_w, power_uw, voltage_v, frequency_ratio, at_resonance_power_w
        """
        a = np.asarray(acceleration_ms2, dtype=float)
        f = np.asarray(frequency_hz, dtype=float)
        omega = 2.0 * np.pi * f

        # Power at resonance (matched load): P = m_eff^2 * a^2 / (4 * c_mech)
        P_resonance = self.m_eff**2 * a**2 / (4.0 * self.c_mech)

        # Off-resonance correction: power ~ |H|^2 (relative to resonance peak = 1/2*zeta^2)
        H = self._frequency_response(omega)
        H_resonance = 1.0 / (2.0 * self.zeta)  # |H| at omega = omega_n
        H_norm = H / H_resonance  # normalized response

        P_out = P_resonance * H_norm**2
        P_out = np.maximum(P_out, 0.0)

        # Open-circuit voltage ~ alpha*x (strain-related), approximate
        # V_oc ~ d31 * Y * t_p * L / (epsilon * w) * relative_displacement
        # Simplified: V ~ sqrt(P_out * R_opt), R_opt = 1/(omega*C_p)
        eps0 = 8.854e-12
        eps_r = 1700.0
        C_p = eps_r * eps0 * self.L * self.w / self.t_p
        R_opt = 1.0 / (omega * C_p + 1e-30)
        V_out = np.sqrt(np.maximum(P_out * R_opt, 0.0))

        freq_ratio = f / self.f_n

        return {
            "power_w": P_out,
            "power_uw": P_out * 1e6,
            "voltage_v": V_out,
            "frequency_ratio": freq_ratio,
            "at_resonance_power_w": P_resonance,
        }
