"""
EC160 -- Isolated DC-DC Converter (Flyback/Forward) -- F1b Detailed Semiconductor Loss Model

Flyback topology (single-switch, galvanic isolation via transformer turns ratio n = N1/N2):

Duty cycle:
    D = V_out * n / (V_in + V_out * n)    [flyback DCM approximation / CCM ideal]

Primary-side MOSFET:
    I_pri_rms = I_out / (n * sqrt(1-D))   [primary carries input current during D]
    P_cond_mosfet = I_pri_rms^2 * R_ds_on(T_j)

Output diode:
    P_cond_diode = I_out * V_f             [diode conducts during (1-D)]

Switching loss (primary MOSFET, hard-switching):
    P_sw = 0.5 * V_in_eff * I_pri_pk * (t_on + t_off) * f_sw
    V_in_eff = V_in + V_out/n              [drain voltage includes reflected output]
    I_pri_pk = I_out / (n * (1-D))

Transformer copper losses:
    P_pri = I_pri_rms^2 * R_pri
    P_sec = I_sec_rms^2 * R_sec
    I_sec_rms = I_out / sqrt(1-D)

Temperature-dependent Rds_on:
    R_ds_on(T) = R_ds_on_ref * (1 + alpha * (T_j - T_ref))

Thermal balance (iterative):
    T_j = T_a + P_loss * R_theta

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
"""

import numpy as np


class IsolatedDCDCF1b:
    """Isolated DC-DC (Flyback/Forward) -- detailed semiconductor + transformer loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n = u["n_turns"]["value"]            # turns ratio N1/N2
        self.R_ds_on_ref = u["R_ds_on"]["value"]  # Ohm at T_ref
        self.V_f = u["V_f"]["value"]              # V
        self.t_on = u["t_on"]["value"]            # s
        self.t_off = u["t_off"]["value"]          # s
        self.f_sw = u["f_sw"]["value"]            # Hz
        self.R_pri = u["R_pri"]["value"]          # Ohm
        self.R_sec = u["R_sec"]["value"]          # Ohm
        self.T_a = u["T_a"]["value"]              # degC
        self.R_theta = u["R_theta"]["value"]       # degC/W
        self.T_ref = u["T_ref"]["value"]           # degC
        self.alpha_rds = u["alpha_rds"]["value"]   # 1/degC

    def duty_cycle(self, v_in, v_out_target):
        """Flyback ideal duty cycle D = V_out*n / (V_in + V_out*n). Clipped to [0.05, 0.90]."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        num = v_out * self.n
        denom = v_in + num
        D = np.where(denom > 0, num / denom, 0.5)
        return np.clip(D, 0.05, 0.90)

    def _rds_on(self, T_j):
        """Temperature-dependent Rds_on [Ohm]."""
        return self.R_ds_on_ref * (1.0 + self.alpha_rds * (T_j - self.T_ref))

    def _losses_at_Tj(self, v_in, v_out_target, i_load, T_j):
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.asarray(v_out_target, dtype=float)
        i = np.asarray(i_load, dtype=float)
        T_j = np.asarray(T_j, dtype=float)
        D = self.duty_cycle(v_in, v_out_target)
        one_minus_D = np.clip(1.0 - D, 0.1, 1.0)

        # Primary RMS current
        i_pri_rms = i / (self.n * np.sqrt(one_minus_D))
        # Secondary RMS current
        i_sec_rms = i / np.sqrt(one_minus_D)
        # Primary peak current (for switching loss)
        i_pri_pk = i / (self.n * one_minus_D)
        # Effective drain voltage (including reflected voltage spike)
        v_drain = v_in + v_out / self.n

        R_ds = self._rds_on(T_j)
        p_mosfet = i_pri_rms ** 2 * R_ds
        p_diode = i * self.V_f
        p_sw = 0.5 * v_drain * i_pri_pk * (self.t_on + self.t_off) * self.f_sw
        p_pri = i_pri_rms ** 2 * self.R_pri
        p_sec = i_sec_rms ** 2 * self.R_sec

        return p_mosfet, p_diode, p_sw, p_pri, p_sec

    def _solve_thermal(self, v_in, v_out_target, i_load):
        """Iteratively solve T_j = T_a + P_total(T_j) * R_theta."""
        T_j = np.full_like(np.asarray(i_load, dtype=float), self.T_a)
        for _ in range(20):
            pm, pd, ps, pp, psc = self._losses_at_Tj(v_in, v_out_target, i_load, T_j)
            p_total = pm + pd + ps + pp + psc
            T_j_new = self.T_a + p_total * self.R_theta
            if np.max(np.abs(T_j_new - T_j)) < 1e-4:
                break
            T_j = T_j_new
        return T_j

    def junction_temperature(self, v_in, v_out_target, i_load):
        return self._solve_thermal(v_in, v_out_target, i_load)

    def loss_breakdown(self, v_in, v_out_target, i_load):
        T_j = self._solve_thermal(v_in, v_out_target, i_load)
        pm, pd, ps, pp, psc = self._losses_at_Tj(v_in, v_out_target, i_load, T_j)
        return {
            "p_mosfet_cond_w": pm,
            "p_diode_cond_w": pd,
            "p_switching_w": ps,
            "p_transformer_pri_w": pp,
            "p_transformer_sec_w": psc,
        }

    def total_losses(self, v_in, v_out_target, i_load):
        T_j = self._solve_thermal(v_in, v_out_target, i_load)
        pm, pd, ps, pp, psc = self._losses_at_Tj(v_in, v_out_target, i_load, T_j)
        return pm + pd + ps + pp + psc

    def efficiency(self, v_in, v_out_target, i_load):
        v_out = np.asarray(v_out_target, dtype=float)
        i = np.asarray(i_load, dtype=float)
        p_out = v_out * i
        p_loss = self.total_losses(v_in, v_out_target, i_load)
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)
