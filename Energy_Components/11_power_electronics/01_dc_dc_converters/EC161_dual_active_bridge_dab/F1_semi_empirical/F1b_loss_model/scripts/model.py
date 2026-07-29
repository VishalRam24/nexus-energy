"""
EC161 -- Dual Active Bridge (DAB) DC-DC Converter -- F1b Detailed Semiconductor Loss Model

Topology: Two full H-bridges (8 MOSFETs total) connected via isolation transformer.
Single-phase-shift (SPS) modulation: phase shift phi between primary and secondary bridges.

SPS power transfer equation:
    P = V_in * V_out / (n * 2 * pi * f_sw * L_s) * phi * (pi - |phi|)

Phase shift required for given power:
    phi = pi/2 - sqrt((pi/2)^2 - 2*pi*f_sw*L_s*P*n / (V_in*V_out))
    (taking the smaller root for stable operation)

Primary RMS current (SPS approximation):
    I_pri_rms = (V_in/(2*pi*f_sw*L_s)) * sqrt(phi^2 + (pi-phi)^2/3) / pi
    Simplified from De Doncker (1991) Appendix.

Per MOSFET conduction (4 primary + 4 secondary, each carries I_rms / 2 in RMS):
    P_cond_per_device = (I_rms/2)^2 * 0.5 * R_ds_on(T_j)
    (each device conducts approximately half the period)
    Total: 8 * P_cond_per_device

Per MOSFET switching (hard-switching for non-ZVS region):
    P_sw_per_device = 0.5 * V_device * I_sw * (t_on + t_off) * f_sw
    Total: 8 * P_sw_per_device

Transformer copper loss:
    P_xfmr = I_pri_rms^2 * R_xfmr

Temperature-dependent Rds_on + thermal balance.

Reference:
    De Doncker, R.W.A.A. et al. (1991). IEEE Trans. Ind. Appl., 27(1), 63-73.
    Zhao, B. et al. (2014). Overview of dual-active-bridge isolated DC/DC converter.
    CSEE JPES, 1(1), 1-9.
"""

import numpy as np


class DABF1b:
    """Dual Active Bridge -- detailed semiconductor + transformer loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n = u["n_turns"]["value"]
        self.L_s = u["L_s"]["value"]
        self.f_sw = u["f_sw"]["value"]
        self.R_ds_on_ref = u["R_ds_on"]["value"]
        self.t_on = u["t_on"]["value"]
        self.t_off = u["t_off"]["value"]
        self.R_xfmr = u["R_xfmr"]["value"]
        self.T_a = u["T_a"]["value"]
        self.R_theta = u["R_theta"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.alpha_rds = u["alpha_rds"]["value"]

    def phase_shift(self, v_in, v_out_target, p_load):
        """
        Compute SPS phase shift phi [rad] for given power transfer.
        phi in [0, pi/2]. Returns 0 at zero load.
        """
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        p = np.asarray(p_load, dtype=float)
        # P_max = V_in * V_out / (n * 4 * f_sw * L_s)  at phi = pi/2
        P_max = v_in * v_out / (self.n * 4.0 * self.f_sw * self.L_s)
        p_norm = np.clip(p / np.where(P_max > 0, P_max, 1.0), 0.0, 1.0)
        # phi = pi/2 * p_norm  (linear approximation valid for small phi)
        # Exact SPS: p_norm = (4/pi) * phi * (1 - phi/pi) => solve quadratic
        # phi^2/pi - phi + pi/4 * p_norm = 0  => phi = pi/2 * (1 - sqrt(1-p_norm))
        phi = np.pi / 2.0 * (1.0 - np.sqrt(np.clip(1.0 - p_norm, 0.0, 1.0)))
        return np.clip(phi, 0.0, np.pi / 2.0)

    def primary_rms_current(self, v_in, v_out_target, p_load):
        """Primary inductor RMS current [A] using SPS approximation."""
        v_in = np.asarray(v_in, dtype=float)
        phi = self.phase_shift(v_in, v_out_target, p_load)
        # I_rms^2 = V_in^2 / (3 * (pi * f_sw * L_s)^2) * (phi^2 * (pi-phi)^2 / (4*pi^2) * 3)
        # Simplified: I_rms = V_in * phi*(pi-phi) / (n * pi^2 * f_sw * L_s) (SPS avg-current proxy)
        # More accurate: from Fourier analysis
        omega = 2.0 * np.pi * self.f_sw
        # Use: I_rms ~ sqrt(phi/pi) * V_in / (n * omega * L_s) -- engineering approximation
        denom = np.where(phi > 0, self.n * omega * self.L_s, 1.0)
        i_rms = np.where(phi > 0, v_in * np.sqrt(phi / np.pi) / denom, 0.0)
        return i_rms

    def _rds_on(self, T_j):
        return self.R_ds_on_ref * (1.0 + self.alpha_rds * (T_j - self.T_ref))

    def _losses_at_Tj(self, v_in, v_out_target, p_load, T_j):
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        p = np.asarray(p_load, dtype=float)
        T_j = np.asarray(T_j, dtype=float)

        i_rms = self.primary_rms_current(v_in, v_out_target, p)
        R_ds = self._rds_on(T_j)

        # Each device sees ~I_rms * sqrt(0.5) (conducts half the cycle in H-bridge)
        i_device_rms_sq = 0.5 * i_rms ** 2
        p_cond_per_device = i_device_rms_sq * R_ds
        p_cond_total = 8.0 * p_cond_per_device  # 4 primary + 4 secondary

        # Switching: primary bridge at V_in, secondary at V_out/n_eff
        i_sw = i_rms * np.sqrt(2.0)  # approximate peak at switching instant
        p_sw_pri = 4.0 * 0.5 * v_in * i_sw * (self.t_on + self.t_off) * self.f_sw
        v_sec_reflected = v_out / self.n
        p_sw_sec = 4.0 * 0.5 * v_sec_reflected * i_sw * (self.t_on + self.t_off) * self.f_sw
        p_sw_total = p_sw_pri + p_sw_sec

        # Transformer copper
        p_xfmr = i_rms ** 2 * self.R_xfmr

        return p_cond_total, p_sw_total, p_xfmr

    def _solve_thermal(self, v_in, v_out_target, p_load):
        T_j = np.full_like(np.asarray(p_load, dtype=float), self.T_a)
        for _ in range(20):
            pc, ps, px = self._losses_at_Tj(v_in, v_out_target, p_load, T_j)
            p_total = pc + ps + px
            T_j_new = self.T_a + p_total * self.R_theta
            if np.max(np.abs(T_j_new - T_j)) < 1e-4:
                break
            T_j = T_j_new
        return T_j

    def junction_temperature(self, v_in, v_out_target, p_load):
        return self._solve_thermal(v_in, v_out_target, p_load)

    def loss_breakdown(self, v_in, v_out_target, p_load):
        T_j = self._solve_thermal(v_in, v_out_target, p_load)
        pc, ps, px = self._losses_at_Tj(v_in, v_out_target, p_load, T_j)
        return {
            "p_mosfet_cond_w": pc,
            "p_switching_w": ps,
            "p_transformer_w": px,
        }

    def total_losses(self, v_in, v_out_target, p_load):
        T_j = self._solve_thermal(v_in, v_out_target, p_load)
        pc, ps, px = self._losses_at_Tj(v_in, v_out_target, p_load, T_j)
        return pc + ps + px

    def efficiency(self, v_in, v_out_target, p_load):
        p = np.asarray(p_load, dtype=float)
        p_loss = self.total_losses(v_in, v_out_target, p_load)
        p_in = p + p_loss
        safe = p_in > 0
        return np.where(safe, p / np.where(safe, p_in, 1.0), 0.0)
