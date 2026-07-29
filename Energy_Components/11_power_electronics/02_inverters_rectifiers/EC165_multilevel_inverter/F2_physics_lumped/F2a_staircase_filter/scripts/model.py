"""
EC165 -- Multilevel Inverter -- F2a Physics-Lumped (Staircase + Averaged Filter ODE)

A first-principles 0D/1D lumped model of an N-level multilevel inverter
(Neutral-Point-Clamped or Cascaded H-Bridge topology). The inverter synthesises
a stepped (staircase) sine of N discrete voltage levels from a DC bus; the more
levels, the closer the waveform approaches a pure sinusoid and the lower the
harmonic distortion (THD).

Physics
-------
1. Staircase synthesis (level-shifted / nearest-level modulation)
   A phase voltage reference  v_ref(t) = m * (V_dc/2) * sin(w t)  is quantised to
   the nearest of N levels spaced  dV = V_dc / (N-1)  apart (NPC pole voltage in
   [0, V_dc], referred to the bus mid-point gives [-V_dc/2, +V_dc/2]).
   Nearest-level control (NLC) is the canonical multilevel modulator for high
   level counts (Rodriguez 2002).

2. THD of the synthesised waveform
   THD = sqrt(V_rms^2 - V_1_rms^2) / V_1_rms ,  with V_1 the fundamental obtained
   by Fourier projection. As N -> infinity the staircase -> sinusoid and THD -> 0.
   This monotone reduction of THD with level count is the defining benefit of
   multilevel converters (Rodriguez, Lai & Peng 2002; Lai & Peng 1996).

3. Averaged output-filter dynamics (ODE, solve_ivp)
   The switched pole voltage v_pole(t) drives a series-R-L / shunt-C output
   filter feeding a resistive load:
       L di_L/dt = v_pole(t) - R_f i_L - v_C
       C dv_C/dt = i_L - v_C / R_load
   Integrated with scipy.integrate.solve_ivp. The filter attenuates the
   switching ripple; the capacitor voltage v_C is the clean AC output. This is
   the standard second-order LC-filter state-space model (Holmes & Lipo 2003).

4. Losses and efficiency
   Conduction:  P_cond = n_dev_conducting * (V_ce0 * I_avg + r_ce * I_rms^2)
   Switching:   P_sw   = N_sw_events * E_sw_ref * (V_cell/V_ref) * (I/I_ref) * f_sw
   Multilevel cells block only V_dc/(N-1), so per-event switching energy falls
   with level count -- another multilevel benefit. Efficiency = P_out/(P_out+P_loss),
   strictly in (0,1) by construction (losses > 0, P_out > 0).

References
----------
Rodriguez, J., Lai, J.S., Peng, F.Z. (2002). "Multilevel inverters: a survey of
    topologies, controls, and applications." IEEE Trans. Ind. Electron. 49(4) 724-738.
Lai, J.S. & Peng, F.Z. (1996). "Multilevel converters - a new breed of power
    converters." IEEE Trans. Ind. Appl. 32(3) 509-517.
Holmes, D.G. & Lipo, T.A. (2003). Pulse Width Modulation for Power Converters. Wiley.
Nabae, Takahashi, Akagi (1981). IEEE Trans. Ind. Appl. IA-17(5) 518-523 (NPC).
"""

import numpy as np
from scipy.integrate import solve_ivp


class MultilevelInverterF2a:
    """N-level multilevel inverter: staircase synthesis + averaged LC filter ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_dc = float(u["V_dc"]["value"])          # V (total bus)
        self.P_rated = float(u["P_rated"]["value"])    # W
        self.n_levels = int(u["n_levels"]["value"])    # default level count
        self.f_out = float(u["f_out"]["value"])        # Hz
        self.f_sw = float(u["f_sw"]["value"])          # Hz
        self.L = float(u["L_filter"]["value"])         # H
        self.C = float(u["C_filter"]["value"])         # F
        self.R_load = float(u["R_load"]["value"])      # Ohm
        self.R_f = float(u["R_filter"]["value"])       # Ohm
        self.V_ce0 = float(u["V_ce0"]["value"])        # V
        self.r_ce = float(u["r_ce"]["value"])          # Ohm
        self.E_sw_ref = float(u["E_sw_ref"]["value"])  # J
        self.V_ref = float(u["V_ref"]["value"])        # V
        self.I_ref = float(u["I_ref"]["value"])        # A
        self.power_factor = float(u["power_factor"]["value"])

    # ------------------------------------------------------------------
    # 1. Staircase voltage synthesis (nearest-level modulation)
    # ------------------------------------------------------------------
    def pole_voltage(self, t, m, n_levels=None):
        """
        Staircase phase pole voltage [V] referenced to bus mid-point,
        in [-V_dc/2, +V_dc/2], via nearest-level quantisation.

        v_ref = m*(V_dc/2)*sin(2 pi f t) quantised to N levels.
        """
        N = self.n_levels if n_levels is None else int(n_levels)
        N = max(N, 2)
        t = np.asarray(t, dtype=float)
        v_ref = m * (self.V_dc / 2.0) * np.sin(2.0 * np.pi * self.f_out * t)
        dV = self.V_dc / (N - 1)                     # level spacing
        v_max = self.V_dc / 2.0                       # outermost pole level
        # Nearest-level quantisation onto the symmetric grid of N levels.
        # Odd N  -> levels include 0 (integer multiples of dV).
        # Even N -> levels are half-integer offsets of dV (no zero level),
        #           e.g. N=2 gives a square wave at +/- V_dc/2.
        if N % 2 == 1:
            k = np.round(v_ref / dV)
            k = np.clip(k, -(N - 1) / 2.0, (N - 1) / 2.0)
            return k * dV
        else:
            k = np.floor(v_ref / dV) + 0.5            # half-integer grid
            k = np.clip(k, -(N - 1) / 2.0, (N - 1) / 2.0)
            v = k * dV
            return np.clip(v, -v_max, v_max)

    # ------------------------------------------------------------------
    # 2. Harmonic analysis of the synthesised waveform
    # ------------------------------------------------------------------
    def waveform(self, m, n_levels=None, n_samples=4000):
        """One fundamental period of the staircase pole voltage."""
        N = self.n_levels if n_levels is None else int(n_levels)
        T = 1.0 / self.f_out
        t = np.linspace(0.0, T, n_samples, endpoint=False)
        v = self.pole_voltage(t, m, N)
        return t, v

    def fundamental_rms(self, m, n_levels=None, n_samples=4000):
        """RMS of the fundamental (50/60 Hz) component via Fourier projection [V]."""
        t, v = self.waveform(m, n_levels, n_samples)
        w = 2.0 * np.pi * self.f_out
        # a1, b1 Fourier coefficients (orthogonal projection over one period)
        a1 = 2.0 * np.mean(v * np.cos(w * t))
        b1 = 2.0 * np.mean(v * np.sin(w * t))
        v1_peak = np.hypot(a1, b1)
        return v1_peak / np.sqrt(2.0)

    def thd(self, m, n_levels=None, n_samples=4000):
        """
        Total Harmonic Distortion of the staircase waveform [fraction].
        THD = sqrt(V_rms^2 - V1_rms^2) / V1_rms.
        Decreases monotonically as the level count increases.
        """
        t, v = self.waveform(m, n_levels, n_samples)
        v_rms = np.sqrt(np.mean(v ** 2))
        v1_rms = self.fundamental_rms(m, n_levels, n_samples)
        harm = max(v_rms ** 2 - v1_rms ** 2, 0.0)
        return np.sqrt(harm) / v1_rms if v1_rms > 1e-9 else 0.0

    def ac_rms_voltage(self, m, n_levels=None):
        """Fundamental line-to-neutral RMS output voltage [V] (scales with V_dc)."""
        return self.fundamental_rms(m, n_levels)

    # ------------------------------------------------------------------
    # 3. Averaged output-filter ODE (solve_ivp)
    # ------------------------------------------------------------------
    def _rhs(self, t, x, m, n_levels):
        """State: x=[i_L, v_C].  L di/dt = v_pole - R_f i - v_C ; C dv/dt = i - v_C/R_load."""
        i_L, v_C = x
        v_pole = float(self.pole_voltage(t, m, n_levels))
        di = (v_pole - self.R_f * i_L - v_C) / self.L
        dv = (i_L - v_C / self.R_load) / self.C
        return [di, dv]

    def simulate(self, m=1.0, n_levels=None, n_periods=6, dt=None,
                 x0=(0.0, 0.0), rtol=1e-6, atol=1e-8):
        """
        Integrate the averaged LC-filter ODE driven by the staircase pole voltage.

        Returns dict with time series (t, v_pole, i_L, v_out=v_C) plus scalar
        steady-state metrics (thd_pole, v_ac_rms, p_out, p_loss, efficiency).
        Steady-state metrics are evaluated over the LAST fundamental period.
        """
        N = self.n_levels if n_levels is None else int(n_levels)
        N = max(N, 2)
        T = 1.0 / self.f_out
        t_end = n_periods * T
        if dt is None:
            dt = T / 400.0
        t_eval = np.arange(0.0, t_end, dt)

        sol = solve_ivp(
            self._rhs, (0.0, t_end), list(x0), t_eval=t_eval,
            args=(m, N), method="RK45", rtol=rtol, atol=atol, max_step=dt,
        )
        t = sol.t
        i_L = sol.y[0]
        v_out = sol.y[1]
        v_pole = self.pole_voltage(t, m, N)

        # last-period mask for steady-state averaging
        last = t >= (t_end - T)

        # Output power delivered to the load (resistive): P = mean(v_C^2/R_load) * 3 phases
        p_out_1ph = np.mean(v_out[last] ** 2) / self.R_load
        p_out = 3.0 * p_out_1ph

        # RMS load current for loss model
        i_load = v_out[last] / self.R_load
        i_rms = np.sqrt(np.mean(i_load ** 2))
        i_avg = np.mean(np.abs(i_load))
        i_pk = np.max(np.abs(i_load)) if i_load.size else 0.0

        p_loss = self.total_losses(i_avg, i_rms, i_pk, N)
        p_in = p_out + p_loss
        eff = p_out / p_in if p_in > 1e-9 else 0.0

        return {
            "t": t,
            "v_pole": v_pole,
            "i_L": i_L,
            "v_out": v_out,
            "n_levels": N,
            "modulation_index": m,
            "thd_pole": self.thd(m, N),
            "thd_output": self._thd_timeseries(v_out[last], t[last]),
            "v_ac_rms": self.ac_rms_voltage(m, N),
            "v_out_rms": np.sqrt(np.mean(v_out[last] ** 2)),
            "p_out": p_out,
            "p_loss": p_loss,
            "efficiency": eff,
        }

    def _thd_timeseries(self, v, t):
        """THD of an arbitrary sampled steady-state waveform over one period."""
        if v.size < 8:
            return 0.0
        w = 2.0 * np.pi * self.f_out
        tt = t - t[0]
        a1 = 2.0 * np.mean(v * np.cos(w * tt))
        b1 = 2.0 * np.mean(v * np.sin(w * tt))
        v1_rms = np.hypot(a1, b1) / np.sqrt(2.0)
        v_rms = np.sqrt(np.mean(v ** 2))
        harm = max(v_rms ** 2 - v1_rms ** 2, 0.0)
        return np.sqrt(harm) / v1_rms if v1_rms > 1e-9 else 0.0

    # ------------------------------------------------------------------
    # 4. Conduction + switching losses, efficiency
    # ------------------------------------------------------------------
    def conduction_loss(self, i_avg, i_rms, n_levels):
        """
        Conduction loss [W] for the 3-phase inverter.
        At any instant a series string of conducting devices carries the load
        current; an N-level leg has ~(N-1) series devices conducting per phase.
        """
        n_series = max(n_levels - 1, 1)
        per_device = self.V_ce0 * i_avg + self.r_ce * i_rms ** 2
        return 3.0 * n_series * per_device

    def switching_loss(self, i_pk, n_levels):
        """
        Switching loss [W] for the 3-phase inverter.
        Each cell blocks V_cell = V_dc/(N-1); switching energy scales with that
        blocking voltage, so higher level counts cut per-event loss. Total event
        count scales with the number of devices (N-1 per phase) and f_sw.
        """
        N = max(n_levels, 2)
        v_cell = self.V_dc / (N - 1)
        n_dev = 2 * (N - 1)                 # devices per phase that switch
        e_event = self.E_sw_ref * (v_cell / self.V_ref) * (i_pk / self.I_ref)
        return 3.0 * n_dev * e_event * self.f_sw

    def total_losses(self, i_avg, i_rms, i_pk, n_levels):
        """Total inverter loss [W] (conduction + switching), always > 0 for I>0."""
        return (self.conduction_loss(i_avg, i_rms, n_levels)
                + self.switching_loss(i_pk, n_levels))

    def efficiency(self, p_out, i_avg, i_rms, i_pk, n_levels):
        """Efficiency in (0,1): P_out / (P_out + P_loss)."""
        p_loss = self.total_losses(i_avg, i_rms, i_pk, n_levels)
        p_in = p_out + p_loss
        return p_out / p_in if p_in > 1e-9 else 0.0
