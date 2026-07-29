"""
EC220 — Triboelectric Nanogenerator (TENG) — F1b Surface Charge Dynamics Model

Extends F1a with:
1. Surface charge density decay: sigma(t) = sigma0 * exp(-t/tau_decay)
   Accounts for leakage current dissipating stored charge over time.

2. Two-layer dielectric stack (PTFE + Nylon or any two dielectrics):
   V_oc(x) = sigma_eff / eps_0 * x / (1 + x*eps_0/(d1/eps_r1 + d2/eps_r2))
   This is the Niu et al. standard TENG equation for two dielectric layers.

3. Dielectric loss: dissipation factor modifies effective output power
   P_loss = P_out * tan_delta (first order correction)

4. Frequency-dependent output: complete V_oc(f) and P(f) curves
   including RC time constant effects with the dielectric stack capacitance.

References:
    Niu, S. & Wang, Z.L. (2015). Nano Energy, 14, 161-192.
    Niu, S. et al. (2013). Energy Environ. Sci. 6(12), 3576.
    Zi, Y. et al. (2015). ACS Nano, 9(7), 7455-7463.
"""

import numpy as np

eps_0 = 8.854187817e-12   # F/m


class TENGF1b:
    """TENG with surface charge dynamics: decay, dual-dielectric stack, and dielectric loss."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.sigma0 = u["sigma0"]["value"]             # C/m^2
        self.A = u["electrode_area"]["value"]           # m^2
        self.x_max = u["gap_max"]["value"]              # m
        self.eps_r1 = u["eps_r1"]["value"]
        self.eps_r2 = u["eps_r2"]["value"]
        self.d1 = u["d1"]["value"]                     # m  (PTFE)
        self.d2 = u["d2"]["value"]                     # m  (Nylon)
        self.tau_decay = u["tau_decay_s"]["value"]      # s
        self.tan_delta = u["tan_delta"]["value"]        # -

        # Effective dielectric thickness: d_eff = d1/eps_r1 + d2/eps_r2
        self.d_eff = self.d1 / self.eps_r1 + self.d2 / self.eps_r2

        # Average capacitance at mid-gap
        x_avg = self.x_max / 2.0
        self.C_avg = eps_0 * self.A / (x_avg + self.d_eff)

    def sigma(self, t_s):
        """Surface charge density [C/m^2] at time t [s]."""
        t = np.asarray(t_s, dtype=float)
        return self.sigma0 * np.exp(-t / self.tau_decay)

    def _voc_at_gap(self, x, sigma):
        """Open-circuit voltage at separation gap x [m] and charge density sigma [C/m^2].
        V_oc = sigma * x / (eps_0 * (1 + x / d_eff))
        This is the standard two-dielectric-layer TENG equation (Niu et al. 2015).
        """
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 1e-9)
        sigma = np.asarray(sigma, dtype=float)
        return sigma * x / (eps_0 * (1.0 + x / self.d_eff))

    def _capacitance_at_gap(self, x):
        """TENG capacitance (two dielectric + air gap) at x [F]."""
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 1e-9)
        return eps_0 * self.A / (x + self.d_eff)

    def compute(self, frequency_hz, R_load_ohm, t_s=0.0):
        """
        Compute TENG output at given frequency, load, and time (surface charge decay).

        Parameters
        ----------
        frequency_hz : float or array — contact-separation frequency [Hz]
        R_load_ohm   : float or array — load resistance [ohm]
        t_s          : float          — elapsed time since start [s] (charge decay)

        Returns
        -------
        dict: sigma_Cm2, V_oc_peak_V, C_avg_F, R_internal_ohm,
              power_avg_w, power_density_mwcm2, efficiency, dielectric_loss_w
        """
        f = np.asarray(frequency_hz, dtype=float)
        R = np.asarray(R_load_ohm, dtype=float)
        t = float(t_s)
        omega = 2.0 * np.pi * f

        # Current charge density (with decay)
        sigma_t = self.sigma(t)

        # Peak V_oc at maximum gap
        V_oc_peak = self._voc_at_gap(self.x_max, sigma_t)

        # Average capacitance at mid-gap
        x_avg = self.x_max / 2.0
        C_avg = self._capacitance_at_gap(x_avg)

        # Internal impedance (capacitive, accounts for dielectric stack)
        R_int = 1.0 / (omega * C_avg + 1e-30)

        # Voltage divider output
        # P = V_oc^2 * R / (R + R_int)^2 * 0.5 (RMS sinusoidal)
        V_load_peak = V_oc_peak * R / (R + R_int)
        P_avg = 0.5 * V_load_peak ** 2 / (R + 1e-30)

        # Dielectric loss correction
        # P_loss = P_dielectric = V^2 * omega * C * tan_delta
        P_dielectric_loss = 0.5 * V_oc_peak ** 2 * omega * C_avg * self.tan_delta
        P_net = np.maximum(P_avg - P_dielectric_loss, 0.0)

        # Power density [mW/cm^2]
        area_cm2 = self.A * 1e4
        P_density = P_net * 1000.0 / area_cm2

        # Mechanical input power
        P_mech = 0.5 * sigma_t * self.A * V_oc_peak * self.x_max * f
        P_mech = np.maximum(P_mech, 1e-20)
        eta = np.minimum(P_net / P_mech, 1.0)

        return {
            "sigma_Cm2": float(sigma_t),
            "V_oc_peak_V": V_oc_peak * np.ones_like(f * R),
            "C_avg_F": C_avg * np.ones_like(f * R),
            "R_internal_ohm": R_int,
            "power_avg_w": P_avg,
            "power_net_w": P_net,
            "power_density_mwcm2": P_density,
            "efficiency": eta,
            "dielectric_loss_w": P_dielectric_loss * np.ones_like(f * R),
        }

    def power_vs_time(self, frequency_hz, R_load_ohm, t_array_s):
        """Compute power output as surface charge decays over time."""
        t_arr = np.asarray(t_array_s, dtype=float)
        powers = []
        for t in t_arr:
            r = self.compute(frequency_hz, R_load_ohm, t)
            powers.append(float(np.atleast_1d(r["power_net_w"])[0]))
        return np.array(powers)
