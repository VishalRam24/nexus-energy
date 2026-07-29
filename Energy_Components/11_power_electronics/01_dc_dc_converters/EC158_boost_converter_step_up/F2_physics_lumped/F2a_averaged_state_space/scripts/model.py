"""
EC158 -- Boost Converter (Step-Up) -- F2a Averaged State-Space Model

Averaged continuous-time state-space model over one switching cycle.
States:  x = [i_L, v_C]

State equations (CCM averaged):
    di_L/dt = (V_in - (1-D)*v_C - i_L*R_L) / L
    dv_C/dt = ((1-D)*i_L - v_C/R_load) / C

Steady-state:
    V_out = V_in / (1-D) * R_load / (R_load + R_L/(1-D)^2)
    I_L = V_in / (R_L + (1-D)^2 * R_load)

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BoostConverterF2a:
    """Boost converter -- averaged state-space dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.L = u["L"]["value"]
        self.C = u["C"]["value"]
        self.R_L = u["R_L"]["value"]
        self.f_sw = u["f_sw"]["value"]

    def derivatives(self, t, x, v_in, duty, R_load):
        """
        State derivatives for averaged boost converter.

        Args:
            t:      time
            x:      state [i_L, v_C]
            v_in:   input voltage [V]
            duty:   duty cycle [0, 1]
            R_load: load resistance [Ohm]

        Returns:
            [di_L/dt, dv_C/dt]
        """
        i_L, v_C = x
        D = np.clip(duty, 0.0, 0.95)
        D_prime = 1.0 - D

        di_L_dt = (v_in - D_prime * v_C - i_L * self.R_L) / self.L
        dv_C_dt = (D_prime * i_L - v_C / R_load) / self.C

        return [di_L_dt, dv_C_dt]

    def steady_state(self, v_in, duty, R_load):
        """
        Analytic steady-state solution.

        From di_L/dt = 0: V_in = (1-D)*v_C + i_L*R_L
        From dv_C/dt = 0: (1-D)*i_L = v_C/R_load  =>  i_L = v_C / ((1-D)*R_load)
        Substituting: v_C = V_in * (1-D) * R_load / ((1-D)^2 * R_load + R_L)
        """
        D = np.clip(duty, 0.0, 0.95)
        D_prime = 1.0 - D
        v_out = v_in * D_prime * R_load / (D_prime**2 * R_load + self.R_L)
        i_L = v_out / (D_prime * R_load)
        i_out = v_out / R_load
        power = v_out * i_out

        return {
            "v_out_ss": v_out,
            "i_L_ss": i_L,
            "i_out_ss": i_out,
            "power_ss": power,
        }

    def simulate(self, v_in, duty, R_load, dt, duration_s, x0=None):
        """
        Simulate the averaged boost converter dynamics.

        Args:
            v_in:       input voltage [V] (scalar or callable(t))
            duty:       duty cycle (scalar or callable(t))
            R_load:     load resistance [Ohm] (scalar or callable(t))
            dt:         output time step [s]
            duration_s: total simulation duration [s]
            x0:         initial state [i_L_0, v_C_0] (default: [0, v_in])

        Returns:
            dict with time-series: t, v_out, i_L, i_out, power
        """
        _v_in = v_in if callable(v_in) else lambda t: v_in
        _duty = duty if callable(duty) else lambda t: duty
        _R_load = R_load if callable(R_load) else lambda t: R_load

        if x0 is None:
            x0 = [0.0, _v_in(0.0)]  # Start with output cap at V_in

        t_span = (0.0, duration_s)
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, x):
            return self.derivatives(t, x, _v_in(t), _duty(t), _R_load(t))

        sol = solve_ivp(
            rhs, t_span, x0, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        i_L = sol.y[0]
        v_C = sol.y[1]
        t = sol.t

        R_load_arr = np.array([_R_load(ti) for ti in t])
        i_out = v_C / R_load_arr
        power = v_C * i_out

        return {
            "t": t,
            "v_out": v_C,
            "i_L": i_L,
            "i_out": i_out,
            "power": power,
        }
