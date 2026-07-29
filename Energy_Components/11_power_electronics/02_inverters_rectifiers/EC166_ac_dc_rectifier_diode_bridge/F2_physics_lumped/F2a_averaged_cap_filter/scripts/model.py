"""
EC166 -- AC-DC Rectifier (Diode Bridge) -- F2a Physics-Lumped Averaged Model

Physics-lumped diode-bridge rectifier with capacitive output filter. A single-
or three-phase full-bridge uncontrolled (diode) rectifier feeds a smoothing
capacitor C_out and a DC load. The lumped state is the capacitor (DC bus)
voltage v_C, integrated with scipy.integrate.solve_ivp.

----------------------------------------------------------------------------
1. Rectified-envelope source voltage (the |.| of the AC, peak-following)
----------------------------------------------------------------------------
The rectifier's bridge output (before the cap) tracks the instantaneous peak
of the rectified line voltage:

    3-phase 6-pulse:  v_rect(t) = Vpk * max_k |cos(w t - k*60deg)|   (6 humps/cycle)
                      Vpk = sqrt(2) * V_LL                           (line-line peak)
    1-phase 2-pulse:  v_rect(t) = Vpk * |sin(w t)|                   (2 humps/cycle)
                      Vpk = sqrt(2) * V_rms

Averaged (ideal, no overlap) DC value  --  Mohan eq. (5-68), (5-9):
    3-phase:  V_do = (3/pi)*sqrt(2)*V_LL = (3*sqrt(2)/pi)*V_LL = 1.3505 * V_LL
    1-phase:  V_do = (2/pi)*sqrt(2)*V_rms = 0.9003 * V_rms

----------------------------------------------------------------------------
2. Commutation overlap (source inductance Ls)  --  Mohan eq. (5-75)
----------------------------------------------------------------------------
Finite line inductance Ls delays current transfer between diodes, dropping the
mean DC voltage:

    3-phase:  dV = (3 * w * Ls / pi) * I_dc
    1-phase:  dV = (2 * w * Ls / pi) * I_dc

This is folded into the conducting-source model so the no-load envelope is
reduced by the overlap drop at the present load current.

----------------------------------------------------------------------------
3. Conduction drop (diode Vf + series resistance)  --  Rashid ch.3
----------------------------------------------------------------------------
Two diodes conduct in series in the path at any instant:

    v_avail(t) = max(0, v_rect(t) - overlap_drop - 2*Vf - 2*r_d*i_path)

----------------------------------------------------------------------------
4. Lumped output-cap voltage ODE  (charge from rectified peaks, discharge to load)
----------------------------------------------------------------------------
Diodes conduct only while the rectified envelope exceeds the cap voltage; then
the cap charges through the small conduction resistance. Otherwise the diodes
block and the cap discharges into the load alone:

    C dv_C/dt = i_diode(t) - i_load(v_C)
    i_diode   = (v_avail - v_C)/R_path   if v_avail > v_C  else 0      (peak charging)
    i_load    = v_C / R_load                                          (discharge)

R_path is the small conduction-loop resistance (diode r_d + a thin source
resistance proxy) that sets the charging pulse magnitude. This reproduces the
classic peak-charging / capacitor-discharge ripple waveform, the highly
non-sinusoidal pulsed line current (poor power factor / high THD), and the DC
ripple that shrinks as C grows.

----------------------------------------------------------------------------
Power factor / harmonics of the pulsed line current
----------------------------------------------------------------------------
For the capacitor-input (peak-charging) bridge the line current is a narrow
pulse near each voltage peak. Distortion power factor is dominated by the
displacement-free harmonic content; a representative capacitor-input bridge has
PF ~ 0.5-0.7 (Mohan Fig 5-19; Rashid ch.3). We report a conduction-angle-based
PF estimate that is well below unity, decreasing as the charging pulse narrows.

References:
    Mohan, N., Undeland, T.M., Robbins, W.P. (2003). Power Electronics:
        Converters, Applications, and Design, 3rd ed. Wiley. Ch. 5 (5-9, 5-68, 5-75).
    Rashid, M.H. (2014). Power Electronics: Circuits, Devices and Applications,
        4th ed. Pearson. Ch. 3 (uncontrolled rectifiers, capacitor filter).
"""

import numpy as np
from scipy.integrate import solve_ivp


class DiodeBridgeRectifierF2a:
    """Averaged diode-bridge rectifier with capacitive-filter voltage ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_phases = int(u["n_phases"]["value"])
        self.f_line = float(u["f_line"]["value"])
        self.C_out = float(u["C_out"]["value"])
        self.L_source = float(u["L_source"]["value"])
        self.V_f = float(u["V_f"]["value"])
        self.r_d = float(u["r_d"]["value"])
        self.R_load_default = float(u["R_load"]["value"])

        self.w = 2.0 * np.pi * self.f_line          # rad/s

        if self.n_phases == 3:
            self._vdc_coeff = 3.0 * np.sqrt(2.0) / np.pi   # 1.3505 * V_LL
            self._pulses = 6                                # humps per cycle
            self._overlap_k = 3.0                           # dV = 3 w Ls/pi * Idc
        else:
            self._vdc_coeff = 2.0 * np.sqrt(2.0) / np.pi   # 0.9003 * V_rms
            self._pulses = 2
            self._overlap_k = 2.0

        # Conduction-loop resistance for the charging pulse (two diodes + thin
        # source/ESR proxy). Kept small so charging pulses are sharp.
        self.R_path = 2.0 * self.r_d + 0.05

    # ------------------------------------------------------------------
    # Static / averaged quantities
    # ------------------------------------------------------------------
    def peak_voltage(self, v_ac_rms):
        """Peak of the rectified envelope [V]. V_LL_peak (3ph) or V_phase_peak (1ph)."""
        return np.sqrt(2.0) * np.asarray(v_ac_rms, dtype=float)

    def ideal_dc_voltage(self, v_ac_rms):
        """Ideal averaged (no-load, no-overlap) DC voltage [V]. Mohan 5-9 / 5-68."""
        return self._vdc_coeff * np.asarray(v_ac_rms, dtype=float)

    def overlap_drop(self, i_dc):
        """Commutation-overlap voltage drop [V]. Mohan eq. (5-75)."""
        return self._overlap_k * self.w * self.L_source / np.pi * np.asarray(i_dc, float)

    def rectified_envelope(self, t, v_ac_rms):
        """Instantaneous peak-following rectified bridge voltage [V] (no cap)."""
        Vpk = np.sqrt(2.0) * v_ac_rms
        t = np.asarray(t, dtype=float)
        if self.n_phases == 3:
            # max over the three line-line cosines, 6 humps per cycle
            ph = self.w * t
            env = np.maximum.reduce([
                np.abs(np.cos(ph)),
                np.abs(np.cos(ph - 2.0 * np.pi / 3.0)),
                np.abs(np.cos(ph + 2.0 * np.pi / 3.0)),
            ])
            return Vpk * env
        else:
            return Vpk * np.abs(np.sin(self.w * t))

    # ------------------------------------------------------------------
    # Lumped capacitor voltage ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, v_ac_rms, R_load):
        v_C = y[0]
        v_rect = self.rectified_envelope(t, v_ac_rms)
        i_load = v_C / R_load
        # overlap drop scales with present load current; subtract conduction
        v_avail = v_rect - self.overlap_drop(i_load) - 2.0 * self.V_f
        if v_avail > v_C:
            i_diode = (v_avail - v_C) / self.R_path
        else:
            i_diode = 0.0
        dvC = (i_diode - i_load) / self.C_out
        return [dvC]

    def simulate(self, v_ac_rms, R_load=None, dt=2e-5, duration_s=0.1, v_C0=None):
        """
        Integrate the output-cap voltage ODE.

        Returns dict with t, v_dc (capacitor voltage), i_load, i_diode,
        plus scalar averaged metrics over the last few cycles.
        """
        if R_load is None:
            R_load = self.R_load_default
        if v_C0 is None:
            # start near the ideal averaged DC so we settle in few cycles
            v_C0 = self.ideal_dc_voltage(v_ac_rms) * 0.95

        n_steps = int(round(duration_s / dt))
        t_eval = np.linspace(0.0, duration_s, n_steps + 1)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [v_C0],
            t_eval=t_eval, args=(v_ac_rms, R_load),
            method="RK45", rtol=1e-6, atol=1e-4, max_step=dt,
        )
        t = sol.t
        v_C = sol.y[0]
        v_rect = self.rectified_envelope(t, v_ac_rms)
        i_load = v_C / R_load
        v_avail = v_rect - self.overlap_drop(i_load) - 2.0 * self.V_f
        i_diode = np.where(v_avail > v_C, (v_avail - v_C) / self.R_path, 0.0)

        # --- averaged metrics over the last whole line cycle (steady state) ---
        T_line = 1.0 / self.f_line
        mask = t >= (t[-1] - T_line) if t[-1] >= T_line else np.ones_like(t, bool)
        v_ss = v_C[mask]
        i_ss = i_load[mask]
        v_dc_mean = float(np.mean(v_ss))
        v_ripple_pp = float(np.max(v_ss) - np.min(v_ss))
        i_dc_mean = float(np.mean(i_ss))

        p_out = v_dc_mean * i_dc_mean
        # conduction loss: two diodes carry the pulsed diode current
        id_ss = i_diode[mask]
        p_cond = float(np.mean(2.0 * (self.V_f * id_ss + self.r_d * id_ss ** 2)))
        p_in = p_out + p_cond
        eff = p_out / p_in if p_in > 0 else 0.0

        return {
            "t": t,
            "v_dc": v_C,
            "v_rect": v_rect,
            "i_load": i_load,
            "i_diode": i_diode,
            "v_dc_mean": v_dc_mean,
            "v_ripple_pp": v_ripple_pp,
            "i_dc_mean": i_dc_mean,
            "p_out_w": p_out,
            "p_cond_w": p_cond,
            "efficiency": eff,
            "v_dc_ideal": float(self.ideal_dc_voltage(v_ac_rms)),
            "power_factor": self.power_factor(v_ac_rms, R_load, i_diode[mask], t[mask]),
        }

    # ------------------------------------------------------------------
    # Power factor of the pulsed line current
    # ------------------------------------------------------------------
    def power_factor(self, v_ac_rms, R_load, i_diode_ss, t_ss):
        """
        Estimate input power factor of the capacitor-input bridge.

        The line current is the (folded) diode current pulse. PF = P / (Vrms*Irms)
        with the line voltage sinusoidal. For the peak-charging bridge the current
        pulse is narrow, so distortion PF is well below 1 (Mohan Fig 5-19).
        We approximate via the conduction duty of the diode pulses: a narrower
        pulse (smaller conduction angle) gives lower PF.
        """
        if len(i_diode_ss) < 3 or np.max(i_diode_ss) <= 0:
            return 0.0
        # conduction fraction = duty where diode current flows
        conducting = i_diode_ss > 1e-6 * np.max(i_diode_ss)
        duty = float(np.mean(conducting))
        # For a current pulse of relative width 'duty', form factor I_rms/I_avg
        # grows as ~1/sqrt(duty); PF (distortion) ~ sqrt(duty)-like, capped < 1.
        # Use mean/rms of the actual pulse train as the distortion factor.
        i_avg = float(np.mean(i_diode_ss))
        i_rms = float(np.sqrt(np.mean(i_diode_ss ** 2)))
        if i_rms <= 0:
            return 0.0
        # distortion power factor (assuming ~unity displacement for diode bridge)
        pf = i_avg / i_rms
        return float(np.clip(pf, 0.0, 0.99))
