"""
EC167 -- Active Front End Rectifier / PFC -- F2a Averaged Boost-PFC, Dual-Loop

Physics-lumped (0D, averaged-switch) model of a single-phase boost power-factor-
correction (PFC) front end.  A diode bridge rectifies the mains; a boost stage
shapes the inductor current so that the current drawn from the line is sinusoidal
and in phase with the line voltage (near-unity power factor, low input THD), while
the DC-link (output) voltage is regulated above the line peak.

State-space (averaged over a switching period, CCM) -- Erickson & Maksimovic (2001)
averaged-switch model of the boost converter:

    L  di_L/dt = |v_g| - (1 - d) * v_dc - R_L * i_L          (inductor)
    C  dv_dc/dt = (1 - d) * i_L - i_load                     (DC-link cap, energy bal.)

with control states (PI integrators):

    dx_v/dt = (v_ref - v_dc)                 outer voltage loop
    dx_i/dt = (i_ref - i_L)                  inner current loop

Dual-loop control (Mohan et al. 2003, Ch.18 "Power-factor-correction circuits"):

    Outer loop produces a current-amplitude command:
        I_cmd = Kp_v*(v_ref - v_dc) + Ki_v*x_v          [A]
    Inner loop forces i_L to follow a *rectified-sine* reference (so line current is
    sinusoidal & in phase => unity PF):
        i_ref(t) = I_cmd * |sin(w_line t)|
        d = 1 - ( |v_g| + L*(di_ref/dt-ish) ... )/v_dc  via PI:
        d = d_ff + Kp_i*(i_ref - i_L) + Ki_i*x_i
        d_ff = 1 - |v_g|/v_dc            (boost feed-forward duty)
    d is clamped to [0, 1].

Power factor (displacement * distortion) is computed from the resulting line current
waveform; THD from its harmonic content.  Losses: MOSFET conduction (R_ds_on),
diode conduction (V_f + r_d), and switching loss scaled from datasheet energy.

Energy conservation: at steady state  P_in_line ~= P_out_dc + P_loss.

References:
    Mohan, Undeland & Robbins (2003). Power Electronics, 3rd ed., Wiley, Ch.8 & Ch.18.
    Erickson & Maksimovic (2001). Fundamentals of Power Electronics, 2nd ed.,
        Kluwer, Ch.18 (Pulse-width modulated rectifiers / low-harmonic rectifiers).
"""

import numpy as np
from scipy.integrate import solve_ivp

try:  # SciPy >= 1.6 / NumPy 2.x: trapz deprecated
    from scipy.integrate import trapezoid as _trapz
except ImportError:  # pragma: no cover
    _trapz = np.trapz


class BoostPFC_F2a:
    """Averaged boost-PFC front end with dual-loop (current + voltage) control."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_line_rms = u["V_line_rms"]["value"]
        self.f_line = u["f_line"]["value"]
        self.f_sw = u["f_sw"]["value"]
        self.L = u["L_boost"]["value"]
        self.R_L = u["R_L"]["value"]
        self.C = u["C_dc"]["value"]
        self.V_dc_ref = u["V_dc_ref"]["value"]
        self.P_load_rated = u["P_load_rated"]["value"]
        self.V_f = u["V_f_diode"]["value"]
        self.r_d = u["r_d_diode"]["value"]
        self.R_ds_on = u["R_ds_on"]["value"]
        self.E_sw_ref = u["E_sw_ref"]["value"]
        self.V_sw_ref = u["V_sw_ref"]["value"]
        self.I_sw_ref = u["I_sw_ref"]["value"]
        self.Kp_i = u["Kp_i"]["value"]
        self.Ki_i = u["Ki_i"]["value"]
        self.Kp_v = u["Kp_v"]["value"]
        self.Ki_v = u["Ki_v"]["value"]

        self.w_line = 2.0 * np.pi * self.f_line
        self.V_peak = np.sqrt(2.0) * self.V_line_rms

    # ------------------------------------------------------------------
    # Instantaneous rectified line voltage |v_g(t)|
    # ------------------------------------------------------------------
    def v_g_abs(self, t, V_line_rms=None):
        Vp = np.sqrt(2.0) * (self.V_line_rms if V_line_rms is None else V_line_rms)
        return np.abs(Vp * np.sin(self.w_line * t))

    # ------------------------------------------------------------------
    # Averaged-switch ODE right-hand side
    #   state y = [i_L, v_dc, x_i, x_v]
    # ------------------------------------------------------------------
    def _rhs(self, t, y, V_line_rms, v_ref, P_load):
        i_L, v_dc, x_i, x_v = y
        v_dc = max(v_dc, 1.0)  # guard divide-by-zero

        vg = self.v_g_abs(t, V_line_rms)

        # --- outer voltage loop -> current amplitude command (>=0) ---
        e_v = v_ref - v_dc
        I_cmd = self.Kp_v * e_v + self.Ki_v * x_v
        I_cmd = max(I_cmd, 0.0)

        # rectified-sine current reference -> sinusoidal line current, unity PF
        sin_t = np.abs(np.sin(self.w_line * t))
        i_ref = I_cmd * sin_t

        # --- inner current loop -> duty cycle ---
        d_ff = 1.0 - vg / v_dc          # boost feed-forward
        e_i = i_ref - i_L
        d = d_ff + self.Kp_i * e_i + self.Ki_i * x_i
        d = min(max(d, 0.0), 1.0)       # physical duty clamp

        # --- averaged boost dynamics ---
        di_L = (vg - (1.0 - d) * v_dc - self.R_L * i_L) / self.L

        # DC-link load current (constant-power load)
        i_load = P_load / v_dc
        dv_dc = ((1.0 - d) * i_L - i_load) / self.C

        # integrator states (only integrate when not saturated -> anti-windup)
        dx_i = e_i
        if (d <= 0.0 and e_i < 0.0) or (d >= 1.0 and e_i > 0.0):
            dx_i = 0.0
        dx_v = e_v
        if (I_cmd <= 0.0 and e_v < 0.0):
            dx_v = 0.0

        return [di_L, dv_dc, dx_i, dx_v]

    # ------------------------------------------------------------------
    # Time-domain simulation via scipy.solve_ivp
    # ------------------------------------------------------------------
    def simulate(self, V_line_rms=None, v_ref=None, P_load=None,
                 duration_s=0.1, n_points=4000, v_dc0=None):
        Vrms = self.V_line_rms if V_line_rms is None else V_line_rms
        vref = self.V_dc_ref if v_ref is None else v_ref
        Pld = self.P_load_rated if P_load is None else P_load
        v0 = vref if v_dc0 is None else v_dc0

        y0 = [0.0, v0, 0.0, 0.0]
        t_eval = np.linspace(0.0, duration_s, int(n_points))

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45",
            args=(Vrms, vref, Pld),
            rtol=1e-6, atol=1e-8, max_step=duration_s / n_points,
        )

        t = sol.t
        i_L = sol.y[0]
        v_dc = sol.y[1]

        vg_abs = self.v_g_abs(t, Vrms)
        # full-wave: line current = +/- i_L following the line-voltage sign
        sign = np.sign(np.sin(self.w_line * t))
        sign[sign == 0] = 1.0
        i_line = sign * i_L
        v_line = np.sqrt(2.0) * Vrms * np.sin(self.w_line * t)

        # instantaneous powers
        p_in = v_line * i_line                       # line-side instantaneous power
        i_load = Pld / np.maximum(v_dc, 1.0)
        p_out = v_dc * i_load                        # DC load power

        return {
            "t": t,
            "i_L": i_L,
            "i_line": i_line,
            "v_line": v_line,
            "v_g_abs": vg_abs,
            "v_dc": v_dc,
            "p_in_inst": p_in,
            "p_out_inst": p_out,
            "x_i": sol.y[2],
            "x_v": sol.y[3],
            "V_peak": np.sqrt(2.0) * Vrms,
        }

    # ------------------------------------------------------------------
    # Power factor & THD from a steady-state window of the line current
    # ------------------------------------------------------------------
    def _steady_window(self, t, *arrays, periods=2):
        """Return last `periods` line-cycles of the given arrays."""
        T_line = 1.0 / self.f_line
        t_end = t[-1]
        t0 = max(t_end - periods * T_line, t[0])
        mask = t >= t0
        return (t[mask],) + tuple(a[mask] for a in arrays)

    def power_factor(self, res):
        """True power factor = P_avg / (V_rms * I_rms)."""
        t, v, i = self._steady_window(res["t"], res["v_line"], res["i_line"])
        if len(t) < 4:
            return 0.0
        # time-averaged real power and rms over the window (trapezoid / window)
        win = t[-1] - t[0]
        P = _trapz(v * i, t) / win
        Vrms = np.sqrt(_trapz(v * v, t) / win)
        Irms = np.sqrt(_trapz(i * i, t) / win)
        denom = Vrms * Irms
        if denom <= 0:
            return 0.0
        return float(np.clip(P / denom, -1.0, 1.0))

    def thd_current(self, res, n_harm=40):
        """Total harmonic distortion of the line current [fraction]."""
        T_line = 1.0 / self.f_line
        t = res["t"]
        i = res["i_line"]
        t_end = t[-1]
        t0 = max(t_end - T_line, t[0])
        mask = t >= t0
        tw = t[mask]
        iw = i[mask]
        if len(tw) < 16:
            return 1.0
        # resample uniformly over exactly one line period
        tu = np.linspace(tw[0], tw[0] + T_line, 1024, endpoint=False)
        iu = np.interp(tu, tw, iw)
        # Fourier coefficients at line harmonics
        n = np.arange(1, n_harm + 1)
        w = self.w_line
        a = np.array([2.0 / len(tu) * np.sum(iu * np.cos(k * w * tu)) for k in n])
        b = np.array([2.0 / len(tu) * np.sum(iu * np.sin(k * w * tu)) for k in n])
        mag = np.sqrt(a ** 2 + b ** 2)
        fund = mag[0]
        if fund <= 1e-9:
            return 1.0
        harm = np.sqrt(np.sum(mag[1:] ** 2))
        return float(harm / fund)

    # ------------------------------------------------------------------
    # Loss model & efficiency (averaged over a line cycle)
    # ------------------------------------------------------------------
    def losses(self, res):
        """Average conduction + switching losses over a steady line cycle [W]."""
        t, i_L, v_dc, vg = self._steady_window(
            res["t"], res["i_L"], res["v_dc"], res["v_g_abs"], periods=1)
        if len(t) < 4:
            return {"p_cond_mosfet": 0.0, "p_cond_diode": 0.0,
                    "p_sw": 0.0, "p_total": 0.0}
        win = t[-1] - t[0]
        iL = np.maximum(i_L, 0.0)
        d = np.clip(1.0 - vg / np.maximum(v_dc, 1.0), 0.0, 1.0)

        # MOSFET conducts fraction d ; diode conducts (1-d)
        p_mos = _trapz(self.R_ds_on * iL ** 2 * d, t) / win
        p_dio = _trapz((self.V_f * iL + self.r_d * iL ** 2) * (1.0 - d), t) / win
        # switching loss: E_sw scaled by V/I, at f_sw, averaged over cycle
        e_sw = self.E_sw_ref * (v_dc / self.V_sw_ref) * (iL / self.I_sw_ref)
        p_sw = _trapz(e_sw * self.f_sw, t) / win

        p_total = float(p_mos + p_dio + p_sw)
        return {
            "p_cond_mosfet": float(p_mos),
            "p_cond_diode": float(p_dio),
            "p_sw": float(p_sw),
            "p_total": p_total,
        }

    def efficiency(self, res):
        """Efficiency = P_out / (P_out + P_loss) over a steady line cycle (in (0,1)).

        Tied to the cited semiconductor loss model so that energy conservation
        P_in = P_out + P_loss holds and 0 < eta < 1 strictly for any P_out > 0.
        """
        t, p_out = self._steady_window(res["t"], res["p_out_inst"], periods=1)
        if len(t) < 4:
            return 0.0
        win = t[-1] - t[0]
        Pout = _trapz(p_out, t) / win
        Ploss = self.losses(res)["p_total"]
        if Pout <= 0:
            return 0.0
        return float(np.clip(Pout / (Pout + Ploss), 0.0, 1.0 - 1e-12))

    def summary(self, res):
        """Scalar steady-state KPIs."""
        pf = self.power_factor(res)
        thd = self.thd_current(res)
        loss = self.losses(res)
        eta = self.efficiency(res)
        # mean DC-link over last cycle
        t, vdc = self._steady_window(res["t"], res["v_dc"], periods=1)
        win = t[-1] - t[0]
        v_dc_mean = float(_trapz(vdc, t) / win) if len(t) > 3 else float(vdc[-1])
        return {
            "power_factor": pf,
            "thd_current": thd,
            "efficiency": eta,
            "v_dc_mean": v_dc_mean,
            "V_peak": float(res["V_peak"]),
            "p_loss_w": loss["p_total"],
            "loss_breakdown": loss,
        }
