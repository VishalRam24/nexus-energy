"""
EC175 -- Induction Motor/Generator -- F2a dq-Frame Dynamic Model

Full dq-frame induction motor model with 4 electrical + 1 mechanical ODE.

States: x = [i_ds, i_qs, i_dr, i_qr, omega_r]

Electrical equations (stator-flux-oriented dq frame):
    di_ds/dt = (v_ds - Rs*i_ds + omega_s*Ls*i_qs + omega_s*Lm*i_qr) / Ls
    di_qs/dt = (v_qs - Rs*i_qs - omega_s*Ls*i_ds - omega_s*Lm*i_dr) / Ls
    di_dr/dt = (-Rr*i_dr + (omega_s - omega_r)*Lr*i_qr + (omega_s - omega_r)*Lm*i_qs) / Lr
    di_qr/dt = (-Rr*i_qr - (omega_s - omega_r)*Lr*i_dr - (omega_s - omega_r)*Lm*i_ds) / Lr

Electromagnetic torque:
    T_e = 1.5 * P * Lm * (i_qs * i_dr - i_ds * i_qr)

Mechanical equation:
    d(omega_r)/dt = (T_e - T_load - B * omega_r) / J

where omega_r = electrical rotor angular velocity [rad/s]
      omega_s = 2*pi*f_supply (synchronous electrical angular velocity)
      P = number of pole pairs

Speed in RPM:
    n = omega_r / P * 30/pi

Slip:
    s = (omega_s - omega_r) / omega_s

Reference:
    Boldea, I. & Nasar, S.A. (2010).
    The Induction Machine Handbook, 2nd ed. CRC Press.
"""

import numpy as np
from scipy.integrate import solve_ivp


class InductionMotorF2a:
    """Induction motor -- dq-frame dynamic model (4 electrical + 1 mechanical ODE)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Rs = u["Rs"]["value"]
        self.Rr = u["Rr"]["value"]
        self.Ls = u["Ls"]["value"]
        self.Lr = u["Lr"]["value"]
        self.Lm = u["Lm"]["value"]
        self.P = u["P"]["value"]       # pole pairs
        self.J = u["J"]["value"]
        self.B = u["B"]["value"]

    def derivatives(self, t, x, v_ds, v_qs, omega_s, T_load):
        """
        State derivatives.

        Args:
            x: [i_ds, i_qs, i_dr, i_qr, omega_r]
            v_ds, v_qs: stator dq voltages
            omega_s: synchronous electrical frequency [rad/s]
            T_load: load torque [Nm]
        """
        i_ds, i_qs, i_dr, i_qr, omega_r = x
        slip_omega = omega_s - omega_r

        di_ds = (v_ds - self.Rs * i_ds + omega_s * self.Ls * i_qs
                 + omega_s * self.Lm * i_qr) / self.Ls
        di_qs = (v_qs - self.Rs * i_qs - omega_s * self.Ls * i_ds
                 - omega_s * self.Lm * i_dr) / self.Ls
        di_dr = (-self.Rr * i_dr + slip_omega * self.Lr * i_qr
                 + slip_omega * self.Lm * i_qs) / self.Lr
        di_qr = (-self.Rr * i_qr - slip_omega * self.Lr * i_dr
                 - slip_omega * self.Lm * i_ds) / self.Lr

        T_e = 1.5 * self.P * self.Lm * (i_qs * i_dr - i_ds * i_qr)
        domega_r = (T_e - T_load - self.B * omega_r) / self.J

        return [di_ds, di_qs, di_dr, di_qr, domega_r]

    def torque(self, i_ds, i_qs, i_dr, i_qr):
        """Electromagnetic torque [Nm]."""
        return 1.5 * self.P * self.Lm * (i_qs * i_dr - i_ds * i_qr)

    def simulate(self, v_supply_rms, frequency_hz, T_load_Nm, dt, duration_s, x0=None):
        """
        Simulate induction motor dynamics.

        The supply voltage is applied in dq frame:
            v_ds = V_phase_peak (d-axis aligned with stator flux)
            v_qs = 0

        Args:
            v_supply_rms: line-to-line RMS voltage [V] (scalar or callable(t))
            frequency_hz: supply frequency [Hz] (scalar or callable(t))
            T_load_Nm:    load torque [Nm] (scalar or callable(t))
            dt:           output time step [s]
            duration_s:   total duration [s]
            x0:           initial state [i_ds, i_qs, i_dr, i_qr, omega_r]
        """
        _v = v_supply_rms if callable(v_supply_rms) else lambda t: v_supply_rms
        _f = frequency_hz if callable(frequency_hz) else lambda t: frequency_hz
        _T = T_load_Nm if callable(T_load_Nm) else lambda t: T_load_Nm

        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 0.0, 0.0]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, x):
            v_rms = _v(t)
            f = _f(t)
            T_l = _T(t)
            omega_s = 2.0 * np.pi * f
            # Phase peak voltage from line-to-line RMS
            v_phase_peak = v_rms * np.sqrt(2.0 / 3.0)
            # In dq frame aligned with stator voltage
            v_ds = v_phase_peak
            v_qs = 0.0
            return self.derivatives(t, x, v_ds, v_qs, omega_s, T_l)

        sol = solve_ivp(
            rhs, (0.0, duration_s), x0, t_eval=t_eval,
            method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        i_ds = sol.y[0]
        i_qs = sol.y[1]
        i_dr = sol.y[2]
        i_qr = sol.y[3]
        omega_r = sol.y[4]
        t = sol.t

        # Derived quantities
        T_e = self.torque(i_ds, i_qs, i_dr, i_qr)
        speed_rpm = omega_r / self.P * 30.0 / np.pi

        # Stator current magnitude
        i_s = np.sqrt(i_ds**2 + i_qs**2)

        # Slip
        omega_s_arr = np.array([2.0 * np.pi * _f(ti) for ti in t])
        slip = np.where(omega_s_arr > 0,
                        (omega_s_arr - omega_r) / omega_s_arr, 0.0)

        # Mechanical power
        omega_mech = omega_r / self.P  # mechanical rad/s
        power_mech = T_e * omega_mech

        return {
            "t": t,
            "speed_rpm": speed_rpm,
            "torque": T_e,
            "current": i_s,
            "power": power_mech,
            "slip": slip,
            "i_ds": i_ds,
            "i_qs": i_qs,
            "i_dr": i_dr,
            "i_qr": i_qr,
            "omega_r": omega_r,
        }
