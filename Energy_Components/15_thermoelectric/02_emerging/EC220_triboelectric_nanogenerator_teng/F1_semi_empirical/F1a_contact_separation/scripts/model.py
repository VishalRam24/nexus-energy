"""
EC220 — Triboelectric Nanogenerator (TENG) — F1a Contact-Separation Model

Contact-separation mode TENG equivalent circuit:
    V_oc(x) = sigma * x / epsilon_0              [V]   open-circuit voltage vs gap x
    C_TENG(x) = epsilon_0 * A / x               [F]   capacitance at gap x
    Q_sc(x)   = sigma * A * (1 - d0/(x + d0))  [C]   short-circuit charge (with dielectric)

    More complete: V_oc(x) = sigma * x / (epsilon_0) * (1 + d/(epsilon_r*x))^(-1) — simplified below

Output with resistive load R:
    Time constant: tau = R * C_TENG
    At steady sinusoidal operation with gap x(t) = x_max/2 * (1 - cos(omega*t)):
    Peak power: P_peak = V_oc_peak^2 / (4 * R_internal)
    Average power: P_avg = P_peak / 2 (for sinusoidal gap motion)

    R_internal ~ 1 / (omega * C_avg) — TENG has very high internal impedance

Efficiency:
    eta = P_out / (P_mech_in)
    P_mech_in = sigma * A * V_oc * f * x_max  (mechanical work per cycle)

References:
    Wang, Z.L. (2013). ACS Nano, 7(11), 9533.
    Niu, S. & Wang, Z.L. (2015). Nano Energy, 14, 161-192.
    Fan, F-R. et al. (2012). Nano Lett. 12(6), 3109.
"""

import numpy as np

eps_0 = 8.854187817e-12   # F/m


class TENGF1a:
    """TENG contact-separation mode — capacitive equivalent circuit model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.sigma = u["sigma"]["value"]           # C/m^2
        self.A = u["electrode_area"]["value"]       # m^2
        self.d = u["dielectric_thickness"]["value"] # m
        self.eps_r = u["epsilon_r_dielectric"]["value"]
        self.x_max = u["gap_max"]["value"]          # m

    def _voc_at_gap(self, x):
        """Open-circuit voltage at separation gap x [m]."""
        # V_oc = sigma * x / (eps_0 * (1 + eps_0*x / (eps_r*d)))
        # = sigma * x / (eps_0 + eps_0^2 * x / (eps_r * d))
        # Simplified lumped: V_oc = sigma / eps_0 * x / (1 + x*eps_0/(eps_r*d))
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 1e-9)
        return self.sigma * x / (eps_0 * (1.0 + eps_0 * x / (self.eps_r * self.d)))

    def _capacitance_at_gap(self, x):
        """TENG capacitance at gap x [F]."""
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 1e-9)
        return eps_0 * self.A / x

    def compute(self, frequency_hz, R_load_ohm):
        """
        Parameters
        ----------
        frequency_hz : float or array — contact-separation frequency [Hz]
        R_load_ohm   : float or array — load resistance [ohm]

        Returns
        -------
        dict: V_oc_peak_V, C_avg_F, R_internal_ohm, power_avg_w, power_density_mwcm2, efficiency
        """
        f = np.asarray(frequency_hz, dtype=float)
        R = np.asarray(R_load_ohm, dtype=float)
        omega = 2.0 * np.pi * f

        # Peak V_oc at maximum gap
        V_oc_peak = self._voc_at_gap(self.x_max)

        # Average capacitance (over gap 0 to x_max)
        x_avg = self.x_max / 2.0
        C_avg = self._capacitance_at_gap(x_avg)

        # TENG internal impedance (capacitive)
        R_int = 1.0 / (omega * C_avg + 1e-30)

        # Power at load (voltage divider with internal impedance)
        # V_load = V_oc * R / (R + R_int)
        # P = V_load^2 / R = V_oc^2 * R / (R + R_int)^2 / 2 (RMS for sinusoidal)
        V_load_peak = V_oc_peak * R / (R + R_int)
        P_peak = 0.5 * V_load_peak**2 / R   # average of sinusoidal signal
        P_avg = P_peak

        # Power density [mW/cm^2]
        area_cm2 = self.A * 1e4
        P_density = P_avg * 1000.0 / area_cm2

        # Mechanical input power (approximate)
        # W_mech/cycle = sigma * V_oc_peak * A * x_max (work against electrostatic field)
        P_mech = 0.5 * self.sigma * self.A * V_oc_peak * self.x_max * f
        P_mech = np.maximum(P_mech, 1e-20)
        eta = np.minimum(P_avg / P_mech, 1.0)

        return {
            "V_oc_peak_V": V_oc_peak * np.ones_like(f * R),
            "C_avg_F": C_avg * np.ones_like(f * R),
            "R_internal_ohm": R_int,
            "power_avg_w": P_avg,
            "power_density_mwcm2": P_density,
            "efficiency": eta,
        }
