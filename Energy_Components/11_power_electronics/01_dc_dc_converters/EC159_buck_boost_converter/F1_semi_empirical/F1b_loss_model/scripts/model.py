"""
EC159 -- Buck-Boost Converter (Inverting) -- F1b Detailed Semiconductor Loss Model

Topology: single MOSFET + single diode, inverting output (V_out = -D*V_in/(1-D)).
For loss modelling we use magnitudes (absolute values of voltages/currents).

Duty cycle:
    D = |V_out| / (V_in + |V_out|)

MOSFET conduction loss (temperature-dependent Rds_on):
    R_ds_on(T) = R_ds_on_ref * (1 + alpha * (T_j - T_ref))
    I_rms_mosfet = I_out * sqrt(D / (1 - D))   [inductor peak current scales with 1/(1-D)]
    P_cond_mosfet = I_rms_mosfet^2 * R_ds_on(T_j)

Diode conduction loss:
    P_cond_diode = I_D_avg * V_f
    I_D_avg = I_out                              [diode carries average output current]

Switching loss:
    P_sw = 0.5 * V_in * I_in * (t_on + t_off) * f_sw
    I_in = I_out * |V_out| / V_in              [power balance: P_in = P_out]

Inductor DCR loss:
    I_rms_L ~ I_out / (1 - D)                  [inductor carries ripple over full cycle]
    P_L = I_rms_L^2 * R_L

Thermal balance (iterative, solved to convergence):
    T_j = T_a + P_loss(T_j) * R_theta

Total losses:
    P_loss = P_cond_mosfet + P_cond_diode + P_sw + P_L

Efficiency:
    eta = P_out / (P_out + P_loss)

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
"""

import numpy as np


class BuckBoostConverterF1b:
    """Buck-boost converter (inverting) -- detailed semiconductor loss model with thermal."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.R_ds_on_ref = u["R_ds_on"]["value"]   # Ohm at T_ref
        self.V_f = u["V_f"]["value"]               # V
        self.t_on = u["t_on"]["value"]             # s
        self.t_off = u["t_off"]["value"]           # s
        self.f_sw = u["f_sw"]["value"]             # Hz
        self.R_L = u["R_L"]["value"]               # Ohm
        self.T_a = u["T_a"]["value"]               # degC
        self.R_theta = u["R_theta"]["value"]        # degC/W
        self.T_ref = u["T_ref"]["value"]            # degC
        self.alpha_rds = u["alpha_rds"]["value"]    # 1/degC

    def duty_cycle(self, v_in, v_out_target):
        """Ideal duty cycle D = |V_out| / (V_in + |V_out|). Clipped to [0.05, 0.95]."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.abs(np.asarray(v_out_target, dtype=float))
        denom = v_in + v_out
        D = np.where(denom > 0, v_out / denom, 0.5)
        return np.clip(D, 0.05, 0.95)

    def _rds_on(self, T_j):
        """Temperature-dependent Rds_on [Ohm] with positive MOSFET tempco."""
        return self.R_ds_on_ref * (1.0 + self.alpha_rds * (T_j - self.T_ref))

    def _losses_at_Tj(self, v_in, v_out_target, i_load, T_j):
        """Compute all loss terms at a given junction temperature."""
        v_in = np.asarray(v_in, dtype=float)
        v_out = np.abs(np.asarray(v_out_target, dtype=float))
        i = np.asarray(i_load, dtype=float)
        T_j = np.asarray(T_j, dtype=float)
        D = self.duty_cycle(v_in, v_out_target)
        one_minus_D = np.clip(1.0 - D, 0.05, 1.0)

        # MOSFET: carries inductor current during D fraction
        # Inductor current = I_out / (1-D) in steady state
        i_L = i / one_minus_D
        i_rms_sq_mosfet = i_L ** 2 * D
        R_ds = self._rds_on(T_j)
        p_mosfet = i_rms_sq_mosfet * R_ds

        # Diode: carries I_out average current
        p_diode = i * self.V_f

        # Switching: 0.5 * V_in * I_L * (t_on + t_off) * f_sw
        p_sw = 0.5 * v_in * i_L * (self.t_on + self.t_off) * self.f_sw

        # Inductor DCR: full cycle
        p_L = i_L ** 2 * self.R_L

        return p_mosfet, p_diode, p_sw, p_L

    def _solve_thermal(self, v_in, v_out_target, i_load):
        """Iteratively solve T_j = T_a + P_loss(T_j) * R_theta."""
        T_j = np.full_like(np.asarray(i_load, dtype=float), self.T_a)
        for _ in range(20):  # converges in <5 iterations
            pm, pd, ps, pl = self._losses_at_Tj(v_in, v_out_target, i_load, T_j)
            p_total = pm + pd + ps + pl
            T_j_new = self.T_a + p_total * self.R_theta
            if np.max(np.abs(T_j_new - T_j)) < 1e-4:
                break
            T_j = T_j_new
        return T_j

    def junction_temperature(self, v_in, v_out_target, i_load):
        """Steady-state junction temperature [degC]."""
        return self._solve_thermal(v_in, v_out_target, i_load)

    def mosfet_conduction_loss(self, v_in, v_out_target, i_load):
        """MOSFET conduction loss [W] at steady-state T_j."""
        T_j = self._solve_thermal(v_in, v_out_target, i_load)
        p, _, _, _ = self._losses_at_Tj(v_in, v_out_target, i_load, T_j)
        return p

    def diode_conduction_loss(self, v_in, v_out_target, i_load):
        """Diode conduction loss [W]: I_out * V_f."""
        i = np.asarray(i_load, dtype=float)
        return i * self.V_f

    def switching_loss(self, v_in, v_out_target, i_load):
        """Switching loss [W]."""
        v_in = np.asarray(v_in, dtype=float)
        D = self.duty_cycle(v_in, v_out_target)
        one_minus_D = np.clip(1.0 - D, 0.05, 1.0)
        i_L = np.asarray(i_load, dtype=float) / one_minus_D
        return 0.5 * v_in * i_L * (self.t_on + self.t_off) * self.f_sw

    def inductor_loss(self, v_in, v_out_target, i_load):
        """Inductor DCR loss [W]."""
        D = self.duty_cycle(v_in, v_out_target)
        one_minus_D = np.clip(1.0 - D, 0.05, 1.0)
        i_L = np.asarray(i_load, dtype=float) / one_minus_D
        return i_L ** 2 * self.R_L

    def total_losses(self, v_in, v_out_target, i_load):
        """Total losses [W] at thermally converged T_j."""
        T_j = self._solve_thermal(v_in, v_out_target, i_load)
        pm, pd, ps, pl = self._losses_at_Tj(v_in, v_out_target, i_load, T_j)
        return pm + pd + ps + pl

    def loss_breakdown(self, v_in, v_out_target, i_load):
        """Return dict of individual loss components [W]."""
        T_j = self._solve_thermal(v_in, v_out_target, i_load)
        pm, pd, ps, pl = self._losses_at_Tj(v_in, v_out_target, i_load, T_j)
        return {
            "p_mosfet_cond_w": pm,
            "p_diode_cond_w": pd,
            "p_switching_w": ps,
            "p_inductor_w": pl,
        }

    def efficiency(self, v_in, v_out_target, i_load):
        """Overall efficiency eta = P_out / (P_out + P_loss)."""
        v_out = np.abs(np.asarray(v_out_target, dtype=float))
        i = np.asarray(i_load, dtype=float)
        p_out = v_out * i
        p_loss = self.total_losses(v_in, v_out_target, i_load)
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)
