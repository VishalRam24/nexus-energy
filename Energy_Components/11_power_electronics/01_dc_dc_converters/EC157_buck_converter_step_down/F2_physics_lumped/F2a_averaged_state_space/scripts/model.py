"""
EC157 -- Buck Converter (Step-Down) -- F2a Averaged State-Space Model

Averaged continuous-time state-space model over one switching cycle.
States:  x = [i_L, v_C]
Inputs:  u = [V_in, D]  (D = duty cycle)

State equations (CCM averaged):
    di_L/dt = (D * V_in - v_C - i_L * R_L) / L
    dv_C/dt = (i_L - v_C / R_load) / C

Outputs:
    v_out = v_C
    i_out = v_C / R_load
    power = v_out * i_out

In matrix form:
    dx/dt = A * x + B * u
    A = [[-R_L/L,  -1/L],
         [ 1/C,    -1/(R_load*C)]]
    B = [[D/L],
         [ 0 ]]

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
    Chapter 7: AC Equivalent Circuit Modeling.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BuckConverterF2a:
    """Buck converter -- averaged state-space dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.L = u["L"]["value"]            # H
        self.C = u["C"]["value"]            # F
        self.R_L = u["R_L"]["value"]        # Ohm (inductor ESR)
        self.f_sw = u["f_sw"]["value"]      # Hz

    def derivatives(self, t, x, v_in, duty, R_load):
        """
        State derivatives for averaged buck converter.

        Args:
            t:      time (unused, autonomous system for constant inputs)
            x:      state vector [i_L, v_C]
            v_in:   input voltage [V]
            duty:   duty cycle [0, 1]
            R_load: load resistance [Ohm]

        Returns:
            [di_L/dt, dv_C/dt]
        """
        i_L, v_C = x
        D = np.clip(duty, 0.0, 1.0)

        di_L_dt = (D * v_in - v_C - i_L * self.R_L) / self.L
        dv_C_dt = (i_L - v_C / R_load) / self.C

        return [di_L_dt, dv_C_dt]

    def steady_state(self, v_in, duty, R_load):
        """
        Analytic steady-state solution (dx/dt = 0).

        Returns:
            dict with i_L_ss, v_out_ss, i_out_ss, power_ss
        """
        D = np.clip(duty, 0.0, 1.0)
        # From di_L/dt = 0: D*V_in = v_C + i_L*R_L
        # From dv_C/dt = 0: i_L = v_C / R_load
        # Substituting: D*V_in = v_C + (v_C/R_load)*R_L
        #             : v_C = D*V_in * R_load / (R_load + R_L)
        v_out = D * v_in * R_load / (R_load + self.R_L)
        i_L = v_out / R_load
        i_out = i_L
        power = v_out * i_out

        return {
            "v_out_ss": v_out,
            "i_L_ss": i_L,
            "i_out_ss": i_out,
            "power_ss": power,
        }

    def simulate(self, v_in, duty, R_load, dt, duration_s, x0=None):
        """
        Simulate the averaged buck converter dynamics.

        Args:
            v_in:       input voltage [V] (scalar or callable(t))
            duty:       duty cycle (scalar or callable(t))
            R_load:     load resistance [Ohm] (scalar or callable(t))
            dt:         output time step [s]
            duration_s: total simulation duration [s]
            x0:         initial state [i_L_0, v_C_0] (default: [0, 0])

        Returns:
            dict with time-series arrays: t, v_out, i_L, i_out, power
        """
        if x0 is None:
            x0 = [0.0, 0.0]

        # Make inputs callable
        _v_in = v_in if callable(v_in) else lambda t: v_in
        _duty = duty if callable(duty) else lambda t: duty
        _R_load = R_load if callable(R_load) else lambda t: R_load

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

        # Compute R_load at each time for output current
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
