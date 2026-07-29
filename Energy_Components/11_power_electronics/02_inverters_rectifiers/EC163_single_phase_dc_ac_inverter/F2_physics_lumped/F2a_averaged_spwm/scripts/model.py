"""
EC163 -- Single-Phase DC-AC Inverter -- F2a Averaged SPWM + LC Filter Model

Physics-lumped 0D model of a single-phase full H-bridge inverter driven by
sinusoidal pulse-width modulation (SPWM) with a second-order LC output filter.

--------------------------------------------------------------------------
1. Averaged switch (SPWM) model
--------------------------------------------------------------------------
For bipolar/unipolar SPWM the duty ratio of the bridge varies sinusoidally
with modulation index m_a (0..1, linear region). Averaging the switching over
one carrier period, the bridge-leg (pre-filter) output voltage is

    v_inv(t) = m_a * V_dc * sin(w*t),     w = 2*pi*f_grid          (full-bridge)

so the fundamental peak is m_a*V_dc and, decomposing per half-bridge, the
per-leg averaged voltage is m_a*(V_dc/2)*sin(w*t) about the mid-rail. The
fundamental rms therefore scales as V1_rms = m_a*V_dc/sqrt(2)  (full-bridge,
linear region, m_a <= 1). This is the classic SPWM result of Mohan et al.
(2003), Ch. 8, Eq. 8-30..8-34.

--------------------------------------------------------------------------
2. LC output filter dynamics (averaged state-space ODE)
--------------------------------------------------------------------------
The averaged inverter voltage v_inv drives a series L_f (with parasitic R_L)
into a shunt C_f across a resistive load R_load. State variables are the
inductor current i_L and capacitor (=output) voltage v_C:

    L_f * di_L/dt = v_inv(t) - R_L*i_L - v_C
    C_f * dv_C/dt = i_L - v_C / R_load

Integrated with scipy.integrate.solve_ivp. The LC corner frequency
f_LC = 1/(2*pi*sqrt(L_f*C_f)) is placed between f_grid and f_sw so the filter
passes the fundamental and attenuates switching harmonics (Mohan Ch. 8;
Holmes & Lipo 2003).

--------------------------------------------------------------------------
3. Harmonics / THD
--------------------------------------------------------------------------
Naturally-sampled SPWM places dominant harmonics around the carrier f_sw and
its sidebands (Holmes & Lipo 2003, Ch. 3). The second-order LC filter
attenuates a harmonic at frequency f_h by approximately

    |H(f_h)| ~ (f_LC / f_h)^2      for f_h >> f_LC

The pre-filter voltage THD for linear SPWM is estimated from the standard
distortion factor; the post-filter THD is the pre-filter harmonic content
scaled by the LC attenuation, yielding the low output THD characteristic of
filtered SPWM inverters.

--------------------------------------------------------------------------
4. Losses & efficiency
--------------------------------------------------------------------------
Conduction (IGBT + freewheeling diode) and switching (E_on+E_off, E_rr)
losses for the 4 devices of the H-bridge are evaluated from the fundamental
output current, following the analytic SPWM loss integrals of the Semikron
Application Manual (2015) / Mohan Ch. 8. Efficiency eta = P_out/(P_out+P_loss),
strictly in (0,1).

References:
    Mohan, Undeland & Robbins (2003), Power Electronics: Converters,
        Applications, and Design, 3rd ed., Wiley. (Ch. 8: PWM inverters)
    Holmes & Lipo (2003), Pulse Width Modulation for Power Converters,
        Wiley-IEEE Press. (SPWM harmonic spectra, Ch. 3)
    Semikron Application Manual (2015), Power Semiconductors. (loss integrals)
"""

import numpy as np
from scipy.integrate import solve_ivp


class SinglePhaseInverterF2a:
    """Single-phase H-bridge SPWM inverter -- averaged model + LC filter ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_dc = u["V_dc"]["value"]
        self.P_rated = u["P_rated"]["value"]
        self.f_grid = u["f_grid"]["value"]
        self.f_sw = u["f_sw"]["value"]
        self.L_f = u["L_f"]["value"]
        self.R_L = u["R_L"]["value"]
        self.C_f = u["C_f"]["value"]
        self.R_load = u["R_load"]["value"]
        # device loss parameters
        self.V_ce0 = u["V_ce0"]["value"]
        self.r_ce = u["r_ce"]["value"]
        self.V_f = u["V_f"]["value"]
        self.r_d = u["r_d"]["value"]
        self.E_on = u["E_on"]["value"]
        self.E_off = u["E_off"]["value"]
        self.E_rr = u["E_rr"]["value"]
        self.V_ref = u["V_ref"]["value"]
        self.I_ref = u["I_ref"]["value"]

    # ------------------------------------------------------------------
    # Averaged SPWM bridge voltage
    # ------------------------------------------------------------------
    def inverter_voltage(self, t, v_dc, m_a, f_grid=None):
        """Averaged full-bridge SPWM output voltage [V]: m_a*V_dc*sin(w t)."""
        f = self.f_grid if f_grid is None else f_grid
        w = 2.0 * np.pi * f
        return m_a * v_dc * np.sin(w * np.asarray(t, dtype=float))

    def fundamental_peak(self, v_dc, m_a):
        """Peak of fundamental component of the averaged bridge voltage [V]."""
        return m_a * v_dc

    def fundamental_rms(self, v_dc, m_a):
        """RMS of fundamental SPWM output voltage [V] = m_a*V_dc/sqrt(2)."""
        return m_a * v_dc / np.sqrt(2.0)

    def lc_corner_frequency(self):
        """LC filter corner (resonant) frequency [Hz]."""
        return 1.0 / (2.0 * np.pi * np.sqrt(self.L_f * self.C_f))

    # ------------------------------------------------------------------
    # LC filter state-space ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, x, v_dc, m_a, f_grid, R_load):
        i_L, v_C = x
        v_inv = self.inverter_voltage(t, v_dc, m_a, f_grid)
        di_L = (v_inv - self.R_L * i_L - v_C) / self.L_f
        dv_C = (i_L - v_C / R_load) / self.C_f
        return [di_L, dv_C]

    def simulate(self, v_dc=None, m_a=0.85, f_grid=None, R_load=None,
                 duration_s=None, dt=None, x0=(0.0, 0.0)):
        """
        Integrate the averaged LC-filter ODE over `duration_s`.

        Returns dict with time series t, v_inv (pre-filter), i_L, v_C (output),
        plus steady-state RMS metrics and the modulation index.
        """
        v_dc = self.V_dc if v_dc is None else v_dc
        f_grid = self.f_grid if f_grid is None else f_grid
        R_load = self.R_load if R_load is None else R_load
        T_grid = 1.0 / f_grid
        if duration_s is None:
            duration_s = 6.0 * T_grid  # several fundamental cycles to settle
        if dt is None:
            dt = T_grid / 400.0        # ~400 pts/cycle for clean RMS

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), list(x0),
            t_eval=t_eval, args=(v_dc, m_a, f_grid, R_load),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )
        t = sol.t
        i_L = sol.y[0]
        v_C = sol.y[1]
        v_inv = self.inverter_voltage(t, v_dc, m_a, f_grid)

        # Steady-state window = last full fundamental cycle
        mask = t >= (t[-1] - T_grid)
        v_out_rms = float(np.sqrt(np.mean(v_C[mask] ** 2)))
        i_out_rms = float(np.sqrt(np.mean((v_C[mask] / R_load) ** 2)))
        p_out = float(np.mean(v_C[mask] * (v_C[mask] / R_load)))

        return {
            "t": t,
            "v_inv": v_inv,
            "i_L": i_L,
            "v_C": v_C,
            "v_out": v_C,
            "m_a": m_a,
            "v_out_rms": v_out_rms,
            "i_out_rms": i_out_rms,
            "p_out_w": p_out,
            "v_fund_rms_ideal": self.fundamental_rms(v_dc, m_a),
            "f_lc_hz": self.lc_corner_frequency(),
        }

    # ------------------------------------------------------------------
    # Harmonics / THD
    # ------------------------------------------------------------------
    def filter_attenuation(self, f_h):
        """Magnitude of 2nd-order LC transfer function at frequency f_h [Hz].

        |H(jw)| = w0^2 / sqrt((w0^2 - w^2)^2 + (w*w0/Q)^2), undamped-asymptote
        ~ (f_lc/f_h)^2 for f_h >> f_lc.
        """
        f_h = np.asarray(f_h, dtype=float)
        w = 2.0 * np.pi * f_h
        w0 = 2.0 * np.pi * self.lc_corner_frequency()
        # Q from load damping: zeta = (1/2R)*sqrt(L/C)
        zeta = (1.0 / (2.0 * self.R_load)) * np.sqrt(self.L_f / self.C_f)
        Q = 1.0 / (2.0 * zeta) if zeta > 0 else 1e6
        denom = np.sqrt((w0**2 - w**2) ** 2 + (w * w0 / Q) ** 2)
        return w0**2 / np.where(denom > 0, denom, 1e-30)

    def thd_prefilter(self, m_a):
        """Approximate voltage THD [-] of unfiltered naturally-sampled SPWM.

        Linear-region SPWM distortion factor: dominant harmonic energy sits at
        the carrier sidebands; total harmonic content relative to fundamental
        is approximated by the standard SPWM distortion expression. Decreasing
        with m_a in the linear region (Holmes & Lipo 2003, Mohan 2003).
        """
        m_a = float(np.clip(m_a, 1e-3, 1.0))
        # Empirical fit to naturally-sampled SPWM normalized distortion:
        # ~ sqrt(1 - (3/4)*m_a^2)/m_a behaviour scaled to ~50-100% region.
        return float(np.sqrt(max(1.0 - 0.75 * m_a**2, 0.0)) / m_a)

    def thd_postfilter(self, m_a):
        """Output (post-LC-filter) voltage THD [-].

        Dominant SPWM harmonics cluster around the carrier f_sw; the LC filter
        attenuates them by ~(f_lc/f_sw)^2. Output THD = prefilter THD * atten.
        """
        atten = float(self.filter_attenuation(self.f_sw))
        return self.thd_prefilter(m_a) * atten

    # ------------------------------------------------------------------
    # Losses & efficiency (H-bridge, 4 devices)
    # ------------------------------------------------------------------
    def _device_currents(self, i_peak, m_a, pf=1.0):
        i_avg_igbt = i_peak / (2.0 * np.pi) + m_a * i_peak * pf / 8.0
        i_rms2_igbt = i_peak**2 * (1.0 / 8.0 + m_a * pf / (3.0 * np.pi))
        i_avg_diode = max(i_peak / (2.0 * np.pi) - m_a * i_peak * pf / 8.0, 0.0)
        i_rms2_diode = i_peak**2 * max(1.0 / 8.0 - m_a * pf / (3.0 * np.pi), 0.0)
        return i_avg_igbt, i_rms2_igbt, i_avg_diode, i_rms2_diode

    def losses(self, v_dc, i_out_rms, m_a, pf=1.0):
        """Total H-bridge losses [W] split into conduction + switching."""
        i_peak = float(i_out_rms) * np.sqrt(2.0)
        ia_t, ir2_t, ia_d, ir2_d = self._device_currents(i_peak, m_a, pf)
        p_cond_igbt = self.V_ce0 * ia_t + self.r_ce * ir2_t
        p_cond_diode = self.V_f * ia_d + self.r_d * ir2_d
        i_sw_avg = i_peak / np.pi
        p_sw_igbt = (self.E_on + self.E_off) * self.f_sw * (v_dc / self.V_ref) * (i_sw_avg / self.I_ref)
        p_sw_diode = self.E_rr * self.f_sw * (v_dc / self.V_ref) * (i_sw_avg / self.I_ref)
        p_cond = 4.0 * (p_cond_igbt + p_cond_diode)
        p_sw = 4.0 * (p_sw_igbt + p_sw_diode)
        return {
            "p_conduction_w": float(p_cond),
            "p_switching_w": float(p_sw),
            "p_loss_total_w": float(p_cond + p_sw),
        }

    def efficiency(self, v_dc, p_out, i_out_rms, m_a, pf=1.0):
        """Inverter efficiency [-] in (0,1)."""
        p_loss = self.losses(v_dc, i_out_rms, m_a, pf)["p_loss_total_w"]
        p_out = float(p_out)
        if p_out <= 0:
            return 0.0
        p_in = p_out + p_loss
        return p_out / p_in if p_in > 0 else 0.0

    # ------------------------------------------------------------------
    # Full operating-point evaluation
    # ------------------------------------------------------------------
    def operating_point(self, v_dc=None, m_a=0.85, f_grid=None, R_load=None,
                        duration_s=None, dt=None):
        """Simulate filter ODE and return a full performance summary dict."""
        v_dc = self.V_dc if v_dc is None else v_dc
        sim = self.simulate(v_dc, m_a, f_grid, R_load, duration_s, dt)
        loss = self.losses(v_dc, sim["i_out_rms"], m_a)
        eta = self.efficiency(v_dc, sim["p_out_w"], sim["i_out_rms"], m_a)
        out = dict(sim)
        out.update(loss)
        out["efficiency"] = eta
        out["thd_prefilter"] = self.thd_prefilter(m_a)
        out["thd_postfilter"] = self.thd_postfilter(m_a)
        out["p_in_w"] = sim["p_out_w"] + loss["p_loss_total_w"]
        return out
