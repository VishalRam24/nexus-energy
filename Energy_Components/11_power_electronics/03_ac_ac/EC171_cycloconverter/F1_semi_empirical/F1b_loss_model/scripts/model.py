"""
EC171 -- Cycloconverter -- F1b Detailed SCR/Thyristor Loss Model

A cycloconverter is a direct AC-to-AC frequency converter using SCR (thyristor)
groups. For a three-phase to three-phase cycloconverter (most common), each
output phase requires two thyristor groups (positive and negative converters),
each group containing three SCRs — total 18 SCRs for a 3-phase output.

Per-thyristor current (continuous conduction, sinusoidal output):
    At output frequency f_out and firing angle alpha:
    I_T_avg = I_out_peak / π * (1 + cos(alpha)) / 2
        (simplified for sinusoidal input and output)
    For a 3-pulse group: I_T_avg ≈ I_out_peak / π
    For a 6-pulse group: I_T_avg ≈ I_out_peak / (2π) * (1 + cos(alpha))

    Simplified approach (natural commutation, alpha ~ 60° at 50% output voltage):
    I_T_avg = I_out_peak / (2π) * (1 + cos(alpha))
    I_T_rms = I_out_peak * sqrt((pi - alpha + sin(2*alpha)/2) / (2*pi))

Per-thyristor conduction loss:
    P_T_cond = V_T0 * I_T_avg + r_T * I_T_rms^2

Snubber loss (R-C snubber dissipation):
    P_snubber = C_snubber * V_in_peak^2 * f_line
    (energy stored in snubber cap discharged each half-cycle)

Total (n_scr thyristors):
    P_total = n_scr * P_T_cond + n_snubber * P_snubber

Efficiency:
    eta = P_out / (P_out + P_total)

Key characteristic: cycloconverters have NO intermediate DC link, so switching
losses from commutation are determined by the supply frequency (50/60 Hz), not
a high PWM frequency → very low switching losses but high conduction losses
because many SCRs must carry full current.

Temperature:
    T_j = T_a + P_T_device * R_th_ja

References:
    Mohan, Undeland & Robbins (2003). Power Electronics, 3rd ed. Wiley, ch. 11.
    Rashid, M.H. (2011). Power Electronics Handbook, 3rd ed. Elsevier.
    Lipo, T.A. (2017). Pulse Width Modulation for Power Converters. Wiley.
"""

import numpy as np


class CycloconverterF1b:
    """Cycloconverter -- per-thyristor conduction + snubber loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_phase_out = int(u["n_phase_out"]["value"])   # output phases (3)
        self.V_T0 = u["V_T0"]["value"]                      # V  SCR threshold voltage
        self.r_T = u["r_T"]["value"]                        # Ohm  SCR slope resistance
        self.C_snubber = u["C_snubber"]["value"]            # F   snubber capacitance per device
        self.f_line = u["f_line"]["value"]                  # Hz
        self.V_in_ll = u["V_in_ll"]["value"]                # V  input L-L RMS
        self.P_rated = u["P_rated"]["value"]                # W
        self.R_th_ja = u["R_th_ja"]["value"]                # K/W
        self.T_a = u["T_a"]["value"]                        # degC

        # 3-phase output: 2 groups × 3 SCRs × 3 phases = 18 SCRs
        # For single-phase output: 2 × 3 = 6 SCRs
        self.n_scr = 2 * 3 * self.n_phase_out
        self.n_snubber = self.n_scr

    def firing_angle(self, v_out_ll_rms, v_in_ll_rms=None):
        """
        Firing angle alpha [rad] from output/input voltage ratio.
        For 3-pulse group: V_out = V_in * cos(alpha)
        => alpha = arccos(V_out / V_in)
        """
        if v_in_ll_rms is None:
            v_in_ll_rms = self.V_in_ll
        v_out = np.asarray(v_out_ll_rms, dtype=float)
        v_in = np.asarray(v_in_ll_rms, dtype=float)
        ratio = np.clip(v_out / np.where(v_in > 0, v_in, 1.0), 0.0, 0.95)
        return np.arccos(ratio)

    def thyristor_currents(self, i_out_rms, alpha):
        """
        Average and RMS current per thyristor.
        I_out_peak = I_out_rms * sqrt(2)
        I_T_avg = I_peak / (2π) * (1 + cos(alpha))   [3-pulse group avg per SCR]
        I_T_rms = I_peak * sqrt((π - alpha + sin(2α)/2) / (2π))
        """
        i_pk = np.asarray(i_out_rms, dtype=float) * np.sqrt(2.0)
        alpha = np.asarray(alpha, dtype=float)

        i_t_avg = i_pk / (2.0 * np.pi) * (1.0 + np.cos(alpha))
        # RMS integral of sin^2 waveform from alpha to pi:
        term = (np.pi - alpha + np.sin(2.0 * alpha) / 2.0) / (2.0 * np.pi)
        term = np.maximum(term, 0.0)
        i_t_rms = i_pk * np.sqrt(term)

        return i_t_avg, i_t_rms

    def thyristor_conduction_loss(self, i_out_rms, alpha):
        """Total thyristor conduction losses [W] (all n_scr devices)."""
        i_avg, i_rms = self.thyristor_currents(i_out_rms, alpha)
        p_per_scr = self.V_T0 * i_avg + self.r_T * i_rms ** 2
        return p_per_scr, self.n_scr * p_per_scr

    def snubber_loss(self, v_in_ll_rms=None):
        """
        Total snubber R-C dissipation [W] (all devices).
        P = n * C * V_peak^2 * f_line
        V_peak = V_in_ll * sqrt(2/3)  (line-to-neutral peak)
        """
        if v_in_ll_rms is None:
            v_in_ll_rms = self.V_in_ll
        v_in = np.asarray(v_in_ll_rms, dtype=float)
        v_ln_peak = v_in * np.sqrt(2.0 / 3.0)
        return self.n_snubber * self.C_snubber * v_ln_peak ** 2 * self.f_line

    def output_current_rms(self, p_out, v_out_ll_rms, power_factor=0.85):
        """Output RMS current [A] from output power."""
        p = np.asarray(p_out, dtype=float)
        v = np.asarray(v_out_ll_rms, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        n = float(self.n_phase_out)
        denom = np.sqrt(n) * v * pf
        denom = np.where(denom > 0, denom, 1.0)
        return np.where(p > 0, p / denom, 0.0)

    def loss_breakdown(self, p_out, v_out_ll_rms, power_factor=0.85):
        """Full loss breakdown [W]."""
        i_rms = self.output_current_rms(p_out, v_out_ll_rms, power_factor)
        alpha = self.firing_angle(v_out_ll_rms)
        _, p_cond = self.thyristor_conduction_loss(i_rms, alpha)
        p_snub = self.snubber_loss()
        return {
            "p_conduction_w": p_cond,
            "p_snubber_w": p_snub,
        }

    def total_losses(self, p_out, v_out_ll_rms, power_factor=0.85):
        """Total cycloconverter losses [W]."""
        bd = self.loss_breakdown(p_out, v_out_ll_rms, power_factor)
        return bd["p_conduction_w"] + bd["p_snubber_w"]

    def efficiency(self, p_out, v_out_ll_rms, power_factor=0.85):
        """Efficiency = P_out / (P_out + P_loss)."""
        p = np.asarray(p_out, dtype=float)
        p_loss = self.total_losses(p_out, v_out_ll_rms, power_factor)
        p_in = p + p_loss
        safe = p_in > 0
        return np.where(safe, p / np.where(safe, p_in, 1.0), 0.0)

    def junction_temperature(self, p_out, v_out_ll_rms, power_factor=0.85):
        """Hottest SCR junction temperature [degC]."""
        i_rms = self.output_current_rms(p_out, v_out_ll_rms, power_factor)
        alpha = self.firing_angle(v_out_ll_rms)
        p_per_scr, _ = self.thyristor_conduction_loss(i_rms, alpha)
        return self.T_a + p_per_scr * self.R_th_ja
