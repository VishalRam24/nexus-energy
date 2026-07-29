"""
EC074 -- Plate Heat Exchanger -- F2a Lumped Dynamic Model

Two-fluid lumped capacitance model with thermal mass and transient response.

Governing ODEs:
    dT_h/dt = (m_dot_h*cp_h*(T_h_in - T_h) - UA*(T_h - T_c) - UA_loss*(T_h - T_amb)) / (M_h*cp_h)
    dT_c/dt = (m_dot_c*cp_c*(T_c_in - T_c) + UA*(T_h - T_c) - UA_loss*(T_c - T_amb)) / (M_c*cp_c)

UA estimation via epsilon-NTU method for counterflow arrangement.

References:
    Incropera & DeWitt (2011), Fundamentals of Heat and Mass Transfer
    Shah & Sekulic (2003), Fundamentals of Heat Exchanger Design
"""

import numpy as np
from scipy.integrate import solve_ivp


class PlateHeatExchanger_F2a:
    """Plate Heat Exchanger -- lumped dynamic two-fluid capacitance model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.UA_nominal = u["UA_nominal"]["value"]
        self.M_hot = u["M_hot"]["value"]
        self.M_cold = u["M_cold"]["value"]
        self.cp_hot = u["cp_hot"]["value"]
        self.cp_cold = u["cp_cold"]["value"]
        self.m_dot_hot_design = u["m_dot_hot_design"]["value"]
        self.m_dot_cold_design = u["m_dot_cold_design"]["value"]
        self.n_plates = u["n_plates"]["value"]
        self.UA_loss = u["UA_loss"]["value"]
        self.T_ambient = u["T_ambient"]["value"]

    # ------------------------------------------------------------------
    # UA estimation using epsilon-NTU (counterflow)
    # ------------------------------------------------------------------
    def compute_UA(self, m_dot_h, m_dot_c):
        """
        Compute effective UA [W/K] scaled from nominal.

        UA scales approximately with (m_dot)^0.7 for turbulent plate flow.
        """
        if m_dot_h <= 0 or m_dot_c <= 0:
            return 0.0
        ratio_h = m_dot_h / max(self.m_dot_hot_design, 1e-6)
        ratio_c = m_dot_c / max(self.m_dot_cold_design, 1e-6)
        # Geometric mean scaling with 0.7 exponent (turbulent convection)
        scale = ((ratio_h * ratio_c) ** 0.5) ** 0.7
        return self.UA_nominal * scale

    def epsilon_ntu(self, m_dot_h, m_dot_c, UA):
        """
        Compute effectiveness for counterflow HX using epsilon-NTU method.

        Returns effectiveness epsilon in [0, 1].
        """
        if UA <= 0 or m_dot_h <= 0 or m_dot_c <= 0:
            return 0.0
        C_h = m_dot_h * self.cp_hot
        C_c = m_dot_c * self.cp_cold
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        C_r = C_min / max(C_max, 1e-10)
        NTU = UA / max(C_min, 1e-10)

        if abs(C_r - 1.0) < 1e-10:
            eps = NTU / (1.0 + NTU)
        else:
            exp_term = np.exp(-NTU * (1.0 - C_r))
            eps = (1.0 - exp_term) / (1.0 - C_r * exp_term)
        return np.clip(eps, 0.0, 1.0)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def ode_rhs(self, t, y, m_dot_h_func, m_dot_c_func, T_h_in_func, T_c_in_func):
        """
        ODE system for two-fluid lumped capacitance model.

        y = [T_h, T_c]
        """
        T_h, T_c = y
        m_dot_h = m_dot_h_func(t) if callable(m_dot_h_func) else m_dot_h_func
        m_dot_c = m_dot_c_func(t) if callable(m_dot_c_func) else m_dot_c_func
        T_h_in = T_h_in_func(t) if callable(T_h_in_func) else T_h_in_func
        T_c_in = T_c_in_func(t) if callable(T_c_in_func) else T_c_in_func

        UA = self.compute_UA(m_dot_h, m_dot_c)

        # Heat transfer from hot to cold
        Q_transfer = UA * (T_h - T_c)

        # Heat loss to ambient
        Q_loss_h = self.UA_loss * (T_h - self.T_ambient)
        Q_loss_c = self.UA_loss * (T_c - self.T_ambient)

        # Hot side
        dTh_dt = (m_dot_h * self.cp_hot * (T_h_in - T_h) - Q_transfer - Q_loss_h) / (self.M_hot * self.cp_hot)
        # Cold side
        dTc_dt = (m_dot_c * self.cp_cold * (T_c_in - T_c) + Q_transfer - Q_loss_c) / (self.M_cold * self.cp_cold)

        return [dTh_dt, dTc_dt]

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, m_dot_h, m_dot_c, T_h_in, T_c_in, T_h_init, T_c_init,
                 dt, duration_s):
        """
        Simulate plate heat exchanger dynamics.

        Parameters
        ----------
        m_dot_h : float or callable(t) -- hot-side mass flow rate [kg/s]
        m_dot_c : float or callable(t) -- cold-side mass flow rate [kg/s]
        T_h_in  : float or callable(t) -- hot inlet temperature [K]
        T_c_in  : float or callable(t) -- cold inlet temperature [K]
        T_h_init : float -- initial hot-side temperature [K]
        T_c_init : float -- initial cold-side temperature [K]
        dt : float -- output time step [s]
        duration_s : float -- total simulation duration [s]

        Returns
        -------
        dict with time-series arrays
        """
        _mh = m_dot_h if callable(m_dot_h) else lambda t: m_dot_h
        _mc = m_dot_c if callable(m_dot_c) else lambda t: m_dot_c
        _Thi = T_h_in if callable(T_h_in) else lambda t: T_h_in
        _Tci = T_c_in if callable(T_c_in) else lambda t: T_c_in

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return self.ode_rhs(t, y, _mh, _mc, _Thi, _Tci)

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_h_init, T_c_init],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt
        )

        t_out = sol.t
        T_h_out = sol.y[0]
        T_c_out = sol.y[1]
        N = len(t_out)

        # Post-process
        Q_transfer = np.zeros(N)
        effectiveness = np.zeros(N)
        UA_arr = np.zeros(N)

        for i in range(N):
            t = t_out[i]
            mh = _mh(t)
            mc = _mc(t)
            Thi = _Thi(t)
            Tci = _Tci(t)
            UA = self.compute_UA(mh, mc)
            UA_arr[i] = UA
            Q_transfer[i] = UA * (T_h_out[i] - T_c_out[i])
            # Effectiveness
            C_h = mh * self.cp_hot
            C_c = mc * self.cp_cold
            C_min = min(C_h, C_c) if min(mh, mc) > 0 else 1e-10
            Q_max = C_min * (Thi - Tci)
            effectiveness[i] = Q_transfer[i] / Q_max if Q_max > 0 else 0.0

        return {
            "t": t_out,
            "T_hot_out": T_h_out,
            "T_cold_out": T_c_out,
            "Q_transfer": Q_transfer,
            "effectiveness": effectiveness,
            "UA": UA_arr,
        }
