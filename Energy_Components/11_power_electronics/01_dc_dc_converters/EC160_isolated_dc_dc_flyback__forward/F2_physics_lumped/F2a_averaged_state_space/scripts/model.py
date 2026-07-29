"""
EC160 -- Isolated DC-DC Converter (Flyback) -- F2a Averaged State-Space Model

Continuous-time state-space averaged model of an isolated flyback converter,
over one switching cycle (CCM). Galvanic isolation is provided by the coupled
inductor / transformer with turns ratio n = N1/N2 (primary/secondary). All
states are referred to the primary side; the secondary quantities follow from
the turns ratio (no direct electrical path between primary and secondary --
isolation is preserved at the model level: V_in never appears additively in the
secondary loop, only the magnetizing energy is transferred via 1/n scaling).

States:  x = [i_m, v_C]
    i_m : primary-referred magnetizing current [A]
    v_C : output capacitor voltage [V]
Inputs:  u = [V_in, d, R_load]
    d   : duty cycle of the primary switch [0, 1]

Switch network averaging (Erickson & Maksimovic, Ch. 7):
    During sub-interval d*Ts   : switch ON  -> V_in is applied across Lm,
                                 diode OFF, load supplied by C.
    During sub-interval (1-d)*Ts: switch OFF -> magnetizing energy flows to the
                                 secondary; the reflected output voltage v_C/n
                                 appears across Lm (with the diode drop), and the
                                 secondary current i_m/n charges C / feeds load.

Averaged state equations (CCM). The secondary voltage is reflected to the
primary by the turns ratio n (v_primary = n*v_secondary), and the magnetizing
current is reflected to the secondary by n (amp-turns N1*i_m = N2*i_sec):
    Lm * di_m/dt = d * V_in
                   - (1-d) * n * (v_C + V_f)          (reflected output + diode)
                   - i_m * R_series(d)                (winding + switch loss)
     C * dv_C/dt = (1-d) * n * i_m  -  v_C / R_load

with the duty-weighted series parasitic resistance referred to the primary
(secondary winding reflected by n**2):
    R_series(d) = R_pri + d * R_ds_on + (1-d) * R_sec * n**2

Ideal (lossless) DC conversion ratio of the flyback:
    V_out = (d / (1-d)) * (V_in / n)          [E&M, flyback gain]

Matrix form (for a frozen d):
    dx/dt = A(d) * x + B(d) * u
    A = [[-R_series/Lm,   -(1-d)*n/Lm],
         [ (1-d)*n/C,     -1/(R_load*C)]]
    B (V_in term) = [[ d/Lm], [0]]
    (the constant diode term -(1-d)*n*V_f/Lm enters as an affine offset)

Efficiency from first-principles power balance:
    P_out = v_out * i_out
    P_loss = i_pri_rms^2 * (R_pri + R_ds_on*... )  + i_sec_rms^2 * R_sec
             + I_out * V_f                          (diode conduction)
    eta = P_out / (P_out + P_loss)   in (0, 1)

Reference:
    Erickson, R.W. & Maksimovic, D. (2020).
    Fundamentals of Power Electronics, 3rd ed. Springer.
    Ch. 6 (Converter circuits, flyback) and Ch. 7 (AC equivalent circuit
    modeling -- state-space averaging).
"""

import numpy as np
from scipy.integrate import solve_ivp


class FlybackConverterF2a:
    """Isolated flyback converter -- averaged state-space dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n = u["n_turns"]["value"]        # turns ratio N1/N2
        self.Lm = u["Lm"]["value"]            # H (primary magnetizing inductance)
        self.C = u["C"]["value"]              # F (output capacitance)
        self.R_ds_on = u["R_ds_on"]["value"]  # Ohm (switch on-resistance)
        self.V_f = u["V_f"]["value"]          # V (output diode forward drop)
        self.R_pri = u["R_pri"]["value"]      # Ohm (primary winding)
        self.R_sec = u["R_sec"]["value"]      # Ohm (secondary winding)
        self.R_C = u.get("R_C", {"value": 0.0})["value"]  # Ohm (cap ESR)
        self.f_sw = u["f_sw"]["value"]        # Hz

    # ------------------------------------------------------------------
    # Parasitics
    # ------------------------------------------------------------------
    def _r_series(self, d):
        """Duty-weighted series resistance referred to the primary [Ohm].

        Switch conducts during d (R_ds_on + R_pri), secondary winding conducts
        during (1-d) and is reflected to the primary by n**2 (impedance
        reflection from secondary to primary).
        """
        d = np.clip(d, 0.0, 1.0)
        return self.R_pri + d * self.R_ds_on + (1.0 - d) * self.R_sec * self.n**2

    # ------------------------------------------------------------------
    # Averaged state derivatives
    # ------------------------------------------------------------------
    def derivatives(self, t, x, v_in, duty, R_load):
        """State derivatives [di_m/dt, dv_C/dt] for the averaged flyback.

        Args:
            t:      time (unused; autonomous for constant inputs)
            x:      [i_m, v_C]
            v_in:   input voltage [V]
            duty:   duty cycle [0, 1]
            R_load: load resistance [Ohm]
        """
        i_m, v_C = x
        d = np.clip(duty, 0.0, 1.0)
        one_minus_d = 1.0 - d
        n = self.n
        Rs = self._r_series(d)

        # Magnetizing-current dynamics (primary side). During (1-d) the
        # secondary voltage (v_C + diode drop) is reflected to the primary by
        # the turns ratio n: v_primary = n * v_secondary  (E&M flyback model).
        di_m_dt = (
            d * v_in
            - one_minus_d * n * (v_C + self.V_f)
            - i_m * Rs
        ) / self.Lm

        # Output capacitor dynamics. The magnetizing current i_m is reflected to
        # the secondary as n * i_m (amp-turns N1*i_m = N2*i_sec) during (1-d).
        dv_C_dt = (one_minus_d * n * i_m - v_C / R_load) / self.C

        return [di_m_dt, dv_C_dt]

    # ------------------------------------------------------------------
    # Steady state
    # ------------------------------------------------------------------
    def ideal_gain(self, v_in, duty):
        """Ideal (lossless) flyback output voltage V_out = d/(1-d) * V_in/n."""
        d = np.clip(duty, 0.0, 1.0 - 1e-9)
        return (d / (1.0 - d)) * (v_in / self.n)

    def steady_state(self, v_in, duty, R_load):
        """Analytic averaged steady-state solution (dx/dt = 0).

        From dv_C/dt = 0:  i_m = v_C / ((1-d) * n * R_load)
        From di_m/dt = 0:  d*V_in - (1-d)*n*(v_C+V_f) - i_m*Rs = 0
        Substitute i_m and solve the linear equation for v_C.
        """
        d = np.clip(duty, 0.0, 1.0 - 1e-9)
        one_minus_d = 1.0 - d
        n = self.n
        Rs = self._r_series(d)

        # d*V_in = (1-d)*n*(v_C+V_f) + Rs * v_C/((1-d)*n*R_load)
        a = one_minus_d * n + Rs / (one_minus_d * n * R_load)
        b = d * v_in - one_minus_d * n * self.V_f
        v_C = b / a
        v_C = max(v_C, 0.0)

        i_m = v_C / (one_minus_d * n * R_load)
        i_out = v_C / R_load
        power = v_C * i_out

        return {
            "v_out_ss": v_C,
            "i_m_ss": i_m,
            "i_out_ss": i_out,
            "power_ss": power,
            "v_out_ideal": self.ideal_gain(v_in, duty),
        }

    # ------------------------------------------------------------------
    # Efficiency (first-principles power balance)
    # ------------------------------------------------------------------
    def efficiency(self, v_in, duty, R_load):
        """Steady-state efficiency in (0, 1) from conduction + diode losses."""
        ss = self.steady_state(v_in, duty, R_load)
        d = float(np.clip(duty, 0.05, 0.90))
        one_minus_d = 1.0 - d
        i_out = ss["i_out_ss"]

        # RMS currents (CCM, approximating flat-topped trapezoids by their
        # conduction-interval RMS; E&M Sec. on RMS of pulsating waveforms).
        # Magnetizing current level i_m = i_out / ((1-d)*n). The primary switch
        # conducts i_m during d; the secondary conducts n*i_m = i_out/(1-d)
        # during (1-d).
        i_m_level = i_out / (one_minus_d * self.n)
        i_pri_rms = i_m_level * np.sqrt(d)          # primary conducts during d
        i_sec_level = i_out / one_minus_d           # secondary conducts during (1-d)
        i_sec_rms = i_sec_level * np.sqrt(one_minus_d)

        p_pri = i_pri_rms**2 * (self.R_pri + self.R_ds_on)
        p_sec = i_sec_rms**2 * self.R_sec
        p_diode = i_out * self.V_f
        p_loss = p_pri + p_sec + p_diode

        p_out = ss["v_out_ss"] * i_out
        p_in = p_out + p_loss
        if p_in <= 0:
            return 0.0
        return float(p_out / p_in)

    # ------------------------------------------------------------------
    # Dynamic simulation
    # ------------------------------------------------------------------
    def simulate(self, v_in, duty, R_load, dt, duration_s, x0=None):
        """Simulate averaged flyback dynamics via scipy.solve_ivp.

        Args:
            v_in:       input voltage [V] (scalar or callable(t))
            duty:       duty cycle (scalar or callable(t))
            R_load:     load resistance [Ohm] (scalar or callable(t))
            dt:         output time step [s]
            duration_s: total simulation duration [s]
            x0:         initial state [i_m_0, v_C_0] (default [0, 0])

        Returns:
            dict of arrays: t, v_out, i_m, i_out, power
        """
        if x0 is None:
            x0 = [0.0, 0.0]

        _v_in = v_in if callable(v_in) else (lambda t: v_in)
        _duty = duty if callable(duty) else (lambda t: duty)
        _R_load = R_load if callable(R_load) else (lambda t: R_load)

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

        i_m = sol.y[0]
        v_C = sol.y[1]
        t = sol.t

        R_load_arr = np.array([_R_load(ti) for ti in t])
        i_out = v_C / R_load_arr
        power = v_C * i_out

        return {
            "t": t,
            "v_out": v_C,
            "i_m": i_m,
            "i_out": i_out,
            "power": power,
        }
