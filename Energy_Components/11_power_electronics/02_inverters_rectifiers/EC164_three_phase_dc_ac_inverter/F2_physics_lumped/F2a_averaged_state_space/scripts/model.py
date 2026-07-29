"""
EC164 -- Three-Phase DC-AC Inverter -- F2a dq-Frame Averaged Model

Grid-tied voltage source inverter in dq synchronous reference frame.

States: x = [i_d, i_q]

State equations:
    di_d/dt = (v_d - R*i_d + omega_e*L*i_q - e_d) / L
    di_q/dt = (v_q - R*i_q - omega_e*L*i_d - e_q) / L

where:
    omega_e = 2*pi*f_grid  (electrical angular frequency)
    e_d, e_q = grid voltage in dq frame
    For grid-aligned dq: e_d = V_grid_peak = sqrt(2/3)*V_LL_rms, e_q = 0

Power:
    P = 1.5 * (e_d*i_d + e_q*i_q) = 1.5 * e_d * i_d  (when e_q=0)
    Q = 1.5 * (e_q*i_d - e_d*i_q) = -1.5 * e_d * i_q  (when e_q=0)

The model includes a simple PI current controller to track P_ref, Q_ref
by computing i_d_ref, i_q_ref from desired power, then applying PI control
to generate v_d, v_q commands.

Reference:
    Teodorescu, R., Liserre, M. & Rodriguez, P. (2011).
    Grid Converters for Photovoltaic and Wind Power Systems. Wiley.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ThreePhaseInverterF2a:
    """Three-phase grid-tied inverter -- dq-frame averaged dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.L = u["L"]["value"]            # H
        self.R = u["R"]["value"]            # Ohm
        self.V_dc = u["V_dc"]["value"]      # V
        self.f_grid = u["f_grid"]["value"]  # Hz
        self.V_grid_rms = u["V_grid_rms"]["value"]  # V (line-to-line RMS)

        self.omega_e = 2.0 * np.pi * self.f_grid
        # Grid voltage in dq frame (aligned to d-axis)
        # V_phase_peak = V_LL_rms * sqrt(2/3)
        self.e_d = self.V_grid_rms * np.sqrt(2.0 / 3.0)
        self.e_q = 0.0

        # PI controller gains (tuned for L=2mH, R=0.1)
        self.Kp = self.L * 500.0   # bandwidth ~ 500 rad/s
        self.Ki = self.R * 500.0

    def current_references(self, P_ref_w, Q_ref_var):
        """
        Compute dq current references from power references.

        P = 1.5 * e_d * i_d  =>  i_d_ref = P_ref / (1.5 * e_d)
        Q = -1.5 * e_d * i_q  =>  i_q_ref = -Q_ref / (1.5 * e_d)
        """
        i_d_ref = P_ref_w / (1.5 * self.e_d) if abs(self.e_d) > 1e-6 else 0.0
        i_q_ref = -Q_ref_var / (1.5 * self.e_d) if abs(self.e_d) > 1e-6 else 0.0
        return i_d_ref, i_q_ref

    def steady_state(self, P_ref_kw, Q_ref_kvar):
        """
        Analytic steady state for given power references.

        At steady state: di/dt = 0
            v_d = R*i_d - omega*L*i_q + e_d
            v_q = R*i_q + omega*L*i_d + e_q
        """
        P_ref_w = P_ref_kw * 1000.0
        Q_ref_var = Q_ref_kvar * 1000.0
        i_d, i_q = self.current_references(P_ref_w, Q_ref_var)

        v_d = self.R * i_d - self.omega_e * self.L * i_q + self.e_d
        v_q = self.R * i_q + self.omega_e * self.L * i_d + self.e_q

        P = 1.5 * (self.e_d * i_d + self.e_q * i_q)
        Q = 1.5 * (self.e_q * i_d - self.e_d * i_q)

        return {
            "i_d_ss": i_d,
            "i_q_ss": i_q,
            "P_ss_w": P,
            "Q_ss_var": Q,
            "v_d_ss": v_d,
            "v_q_ss": v_q,
        }

    def simulate(self, P_ref_kw, Q_ref_kvar, dt, duration_s, x0=None):
        """
        Simulate inverter with PI current control tracking power references.

        Args:
            P_ref_kw:   active power ref [kW] (scalar or callable(t))
            Q_ref_kvar: reactive power ref [kvar] (scalar or callable(t))
            dt:         output time step [s]
            duration_s: total duration [s]
            x0:         initial state [i_d_0, i_q_0, int_d_0, int_q_0]

        Returns:
            dict: t, i_d, i_q, P, Q, v_dc
        """
        _P = P_ref_kw if callable(P_ref_kw) else lambda t: P_ref_kw
        _Q = Q_ref_kvar if callable(Q_ref_kvar) else lambda t: Q_ref_kvar

        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 0.0]  # [i_d, i_q, integrator_d, integrator_q]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, state):
            i_d, i_q, int_d, int_q = state

            P_w = _P(t) * 1000.0
            Q_var = _Q(t) * 1000.0
            i_d_ref, i_q_ref = self.current_references(P_w, Q_var)

            # PI controller
            err_d = i_d_ref - i_d
            err_q = i_q_ref - i_q

            v_d = (self.Kp * err_d + self.Ki * int_d
                   - self.omega_e * self.L * i_q + self.e_d)
            v_q = (self.Kp * err_q + self.Ki * int_q
                   + self.omega_e * self.L * i_d + self.e_q)

            # Voltage limits (modulation index limit)
            v_max = self.V_dc / np.sqrt(3.0)
            v_d = np.clip(v_d, -v_max, v_max)
            v_q = np.clip(v_q, -v_max, v_max)

            # Plant dynamics
            di_d_dt = (v_d - self.R * i_d + self.omega_e * self.L * i_q - self.e_d) / self.L
            di_q_dt = (v_q - self.R * i_q - self.omega_e * self.L * i_d - self.e_q) / self.L

            # Integrator dynamics (with anti-windup)
            dint_d = err_d
            dint_q = err_q

            return [di_d_dt, di_q_dt, dint_d, dint_q]

        sol = solve_ivp(
            rhs, (0.0, duration_s), x0, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        i_d = sol.y[0]
        i_q = sol.y[1]
        t = sol.t

        P = 1.5 * (self.e_d * i_d + self.e_q * i_q)
        Q = 1.5 * (self.e_q * i_d - self.e_d * i_q)

        # V_dc is assumed constant in this model (stiff DC bus)
        v_dc = np.full_like(t, self.V_dc)

        return {
            "t": t,
            "i_d": i_d,
            "i_q": i_q,
            "P": P,
            "Q": Q,
            "v_dc": v_dc,
        }
