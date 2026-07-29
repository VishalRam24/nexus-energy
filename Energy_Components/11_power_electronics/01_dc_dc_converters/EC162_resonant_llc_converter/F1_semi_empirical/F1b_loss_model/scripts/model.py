"""
EC162 -- Resonant LLC Converter -- F1b Detailed Semiconductor Loss Model

Half-bridge LLC topology at/near resonance. Achieves ZVS for primary MOSFETs,
ZCS for secondary diodes. Switching losses are greatly reduced vs hard-switching,
but residual turn-off losses remain.

At resonant frequency (f_sw = f_r), voltage gain M = n * V_out / (V_in/2) ~ 1.
Primary current is approximately sinusoidal with amplitude I_pk.

Peak resonant current:
    I_pk = pi * I_out / (2 * n)   [from power balance at resonance]

Primary MOSFET RMS current (each of 2 devices in half-bridge):
    I_rms_mosfet = I_pk / sqrt(2)

Conduction loss:
    P_cond = 2 * I_rms_mosfet^2 * R_ds_on(T_j)

Residual switching loss (ZVS turn-on is zero; turn-off has small loss):
    P_sw = 2 * 0.5 * V_in/2 * I_pk * t_off * f_sw   (turn-off only, ZVS on)

Secondary diode conduction:
    I_rms_sec = I_out * sqrt(pi^2/8)  [sinusoidal rectification]
    P_diode = 2 * V_f * I_out / 2     [two diodes, each carries I_out/2 avg]

Transformer + inductor copper:
    P_Lr = I_rms_mosfet^2 * R_Lr
    P_pri = I_rms_mosfet^2 * R_pri
    P_sec = I_rms_sec^2 * R_sec

Temperature-dependent Rds_on + thermal balance.

Reference:
    Yang, B. et al. (2002). LLC resonant converter for front end DC/DC conversion.
    IEEE Applied Power Electronics Conference (APEC), 1108-1112.
"""

import numpy as np


class LLCConverterF1b:
    """LLC Resonant Converter -- detailed semiconductor + magnetic loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n = u["n_turns"]["value"]
        self.f_sw = u["f_sw"]["value"]
        self.R_ds_on_ref = u["R_ds_on"]["value"]
        self.V_f = u["V_f"]["value"]
        self.t_off = u["t_off"]["value"]
        self.R_Lr = u["R_Lr"]["value"]
        self.R_pri = u["R_pri"]["value"]
        self.R_sec = u["R_sec"]["value"]
        self.T_a = u["T_a"]["value"]
        self.R_theta = u["R_theta"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.alpha_rds = u["alpha_rds"]["value"]

    def _rds_on(self, T_j):
        return self.R_ds_on_ref * (1.0 + self.alpha_rds * (T_j - self.T_ref))

    def _peak_current(self, i_load):
        """Primary peak resonant current [A] from power balance."""
        i = np.asarray(i_load, dtype=float)
        return np.pi * i / (2.0 * self.n)

    def _losses_at_Tj(self, v_in, i_load, T_j):
        v_in = np.asarray(v_in, dtype=float)
        i = np.asarray(i_load, dtype=float)
        T_j = np.asarray(T_j, dtype=float)

        I_pk = self._peak_current(i)
        # Primary MOSFET RMS (half-bridge: 2 devices, each sees I_pk/sqrt(2) in RMS over half period)
        I_rms_mosfet = I_pk / np.sqrt(2.0)
        R_ds = self._rds_on(T_j)
        p_cond = 2.0 * I_rms_mosfet ** 2 * R_ds

        # Residual turn-off switching (ZVS turn-on is lossless)
        V_sw = v_in / 2.0
        p_sw = 2.0 * 0.5 * V_sw * I_pk * self.t_off * self.f_sw

        # Secondary diode (2 diodes in full-wave rect, each carries I_out/2 avg)
        p_diode = 2.0 * self.V_f * (i / 2.0)

        # Inductor + transformer copper
        p_Lr = I_rms_mosfet ** 2 * self.R_Lr
        p_pri = I_rms_mosfet ** 2 * self.R_pri
        i_rms_sec = i * np.sqrt(np.pi ** 2 / 8.0)  # sinusoidal rectification
        p_sec = i_rms_sec ** 2 * self.R_sec

        return p_cond, p_sw, p_diode, p_Lr, p_pri, p_sec

    def _solve_thermal(self, v_in, i_load):
        T_j = np.full_like(np.asarray(i_load, dtype=float), self.T_a)
        for _ in range(20):
            pc, ps, pd, pl, pp, psc = self._losses_at_Tj(v_in, i_load, T_j)
            p_total = pc + ps + pd + pl + pp + psc
            T_j_new = self.T_a + p_total * self.R_theta
            if np.max(np.abs(T_j_new - T_j)) < 1e-4:
                break
            T_j = T_j_new
        return T_j

    def junction_temperature(self, v_in, i_load):
        return self._solve_thermal(v_in, i_load)

    def loss_breakdown(self, v_in, i_load):
        T_j = self._solve_thermal(v_in, i_load)
        pc, ps, pd, pl, pp, psc = self._losses_at_Tj(v_in, i_load, T_j)
        return {
            "p_mosfet_cond_w": pc,
            "p_switching_w": ps,
            "p_diode_cond_w": pd,
            "p_resonant_inductor_w": pl,
            "p_transformer_pri_w": pp,
            "p_transformer_sec_w": psc,
        }

    def total_losses(self, v_in, i_load):
        T_j = self._solve_thermal(v_in, i_load)
        pc, ps, pd, pl, pp, psc = self._losses_at_Tj(v_in, i_load, T_j)
        return pc + ps + pd + pl + pp + psc

    def efficiency(self, v_in, v_out_target, i_load):
        """eta = P_out / (P_out + P_loss). v_out_target used for P_out."""
        v_out = np.asarray(v_out_target, dtype=float)
        i = np.asarray(i_load, dtype=float)
        p_out = v_out * i
        p_loss = self.total_losses(v_in, i_load)
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)
