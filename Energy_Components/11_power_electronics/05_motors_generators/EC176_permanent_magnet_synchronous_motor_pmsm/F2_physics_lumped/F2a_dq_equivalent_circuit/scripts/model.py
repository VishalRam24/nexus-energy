"""
EC176 -- PMSM -- F2a dq-Frame Dynamic Model

PMSM dq-frame model: 2 electrical ODEs + 1 mechanical ODE.

States: x = [i_d, i_q, omega_m]

Electrical equations:
    di_d/dt = (v_d - Rs*i_d + omega_e*Lq*i_q) / Ld
    di_q/dt = (v_q - Rs*i_q - omega_e*Ld*i_d - omega_e*Phi_m) / Lq

where omega_e = P * omega_m (electrical frequency from mechanical speed)

Electromagnetic torque:
    T_e = 1.5 * P * (Phi_m*i_q + (Ld - Lq)*i_d*i_q)
    For surface-mount (Ld = Lq): T_e = 1.5 * P * Phi_m * i_q

Mechanical equation:
    d(omega_m)/dt = (T_e - T_load - B*omega_m) / J

The model includes a simple speed PI controller that generates v_d, v_q
from a speed reference. Alternatively, direct v_d, v_q can be applied.

Reference:
    Gieras, J.F. (2010). Permanent Magnet Motor Technology, 3rd ed. CRC Press.
    Krause, P.C. et al. (2013). Analysis of Electric Machinery and Drive Systems, 3rd ed. Wiley.
"""

import numpy as np
from scipy.integrate import solve_ivp


class PMSMF2a:
    """PMSM -- dq-frame dynamic model (2 electrical + 1 mechanical ODE)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Ld = u["Ld"]["value"]
        self.Lq = u["Lq"]["value"]
        self.Rs = u["Rs"]["value"]
        self.Phi_m = u["Phi_m"]["value"]
        self.P = u["P"]["value"]       # pole pairs
        self.J = u["J"]["value"]
        self.B = u["B"]["value"]
        self.V_dc = u["V_dc"]["value"]

        # Speed PI controller gains
        self.Kp_speed = 2.0   # Nm per rad/s error
        self.Ki_speed = 10.0
        # Current PI gains
        self.Kp_id = self.Ld * 500.0
        self.Ki_id = self.Rs * 500.0
        self.Kp_iq = self.Lq * 500.0
        self.Ki_iq = self.Rs * 500.0

    def torque(self, i_d, i_q):
        """Electromagnetic torque [Nm]."""
        return 1.5 * self.P * (self.Phi_m * i_q + (self.Ld - self.Lq) * i_d * i_q)

    def simulate_direct(self, v_d, v_q, T_load_Nm, dt, duration_s, x0=None):
        """
        Simulate with direct voltage inputs (open-loop).

        Args:
            v_d, v_q: dq voltages [V] (scalar or callable(t))
            T_load_Nm: load torque [Nm] (scalar or callable(t))
        """
        _vd = v_d if callable(v_d) else lambda t: v_d
        _vq = v_q if callable(v_q) else lambda t: v_q
        _T = T_load_Nm if callable(T_load_Nm) else lambda t: T_load_Nm

        if x0 is None:
            x0 = [0.0, 0.0, 0.0]  # [i_d, i_q, omega_m]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, x):
            i_d, i_q, omega_m = x
            omega_e = self.P * omega_m

            di_d = (_vd(t) - self.Rs * i_d + omega_e * self.Lq * i_q) / self.Ld
            di_q = (_vq(t) - self.Rs * i_q - omega_e * self.Ld * i_d
                    - omega_e * self.Phi_m) / self.Lq

            T_e = self.torque(i_d, i_q)
            domega_m = (T_e - _T(t) - self.B * omega_m) / self.J

            return [di_d, di_q, domega_m]

        sol = solve_ivp(
            rhs, (0.0, duration_s), x0, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        return self._format_output(sol)

    def simulate_speed_control(self, speed_ref_rpm, T_load_Nm, dt, duration_s, x0=None):
        """
        Simulate with cascaded speed + current PI control.

        Args:
            speed_ref_rpm: speed reference [rpm] (scalar or callable(t))
            T_load_Nm: load torque [Nm] (scalar or callable(t))
        """
        _ref = speed_ref_rpm if callable(speed_ref_rpm) else lambda t: speed_ref_rpm
        _T = T_load_Nm if callable(T_load_Nm) else lambda t: T_load_Nm

        if x0 is None:
            # [i_d, i_q, omega_m, int_speed, int_id, int_iq]
            x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        v_max = self.V_dc / np.sqrt(3.0)

        def rhs(t, x):
            i_d, i_q, omega_m, int_speed, int_id, int_iq = x
            omega_e = self.P * omega_m

            # Speed controller -> i_q reference (i_d_ref = 0 for surface mount)
            omega_ref = _ref(t) * np.pi / 30.0  # rpm to rad/s
            err_speed = omega_ref - omega_m
            i_q_ref = np.clip(
                self.Kp_speed * err_speed + self.Ki_speed * int_speed,
                -50.0, 50.0,
            )
            i_d_ref = 0.0  # MTPA for surface-mount PMSM

            # Current controllers
            err_id = i_d_ref - i_d
            err_iq = i_q_ref - i_q

            v_d = (self.Kp_id * err_id + self.Ki_id * int_id
                   - omega_e * self.Lq * i_q)
            v_q = (self.Kp_iq * err_iq + self.Ki_iq * int_iq
                   + omega_e * self.Ld * i_d + omega_e * self.Phi_m)

            # Voltage limiting
            v_d = np.clip(v_d, -v_max, v_max)
            v_q = np.clip(v_q, -v_max, v_max)

            # Plant dynamics
            di_d = (v_d - self.Rs * i_d + omega_e * self.Lq * i_q) / self.Ld
            di_q = (v_q - self.Rs * i_q - omega_e * self.Ld * i_d
                    - omega_e * self.Phi_m) / self.Lq

            T_e = self.torque(i_d, i_q)
            domega_m = (T_e - _T(t) - self.B * omega_m) / self.J

            # Integrator dynamics
            dint_speed = err_speed
            dint_id = err_id
            dint_iq = err_iq

            return [di_d, di_q, domega_m, dint_speed, dint_id, dint_iq]

        sol = solve_ivp(
            rhs, (0.0, duration_s), x0, t_eval=t_eval,
            method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        return self._format_output(sol)

    def _format_output(self, sol):
        """Format ODE solution into output dict."""
        i_d = sol.y[0]
        i_q = sol.y[1]
        omega_m = sol.y[2]
        t = sol.t

        T_e = self.torque(i_d, i_q)
        speed_rpm = omega_m * 30.0 / np.pi
        power = T_e * omega_m

        return {
            "t": t,
            "speed_rpm": speed_rpm,
            "torque": T_e,
            "i_d": i_d,
            "i_q": i_q,
            "power": power,
            "omega_m": omega_m,
        }
