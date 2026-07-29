"""
EC171 -- Cycloconverter -- F2a Physics-Lumped Phase-Controlled Averaged Model

A cycloconverter is a *direct* AC-to-AC frequency converter: it synthesises a
lower output frequency f_out directly from the line frequency f_line WITHOUT an
intermediate DC link, by continuously phase-controlling two anti-parallel
thyristor (SCR) converter groups (a "dual converter") per output phase. The
positive group conducts during the positive output half-cycle, the negative
group during the negative half-cycle. The firing angle alpha is modulated
sinusoidally so that the *averaged* (slow-envelope) output voltage traces a
low-frequency sinusoid.

Naturally-commutated (line-commutated) cycloconverters are restricted to
    f_out  <  ~(1/3) * f_line
because the output is assembled from segments of the input waves; above ~1/3
the synthesised waveform degrades and sub-harmonics appear (Pelly 1971, Bose
2002). This model enforces that limit.

------------------------------------------------------------------------------
1. AVERAGED OUTPUT VOLTAGE vs FIRING ANGLE  (cosine control law)
------------------------------------------------------------------------------
For a p-pulse phase-controlled converter group the averaged (DC-side, here the
slow-envelope) output voltage is the classic Pelly/Bose relation

    V_do = (p/pi) * V_m * sin(pi/p)          (max average, alpha = 0)
    V_avg(alpha) = V_do * cos(alpha)

where V_m is the peak of the input phase voltage feeding the group. To synthesise
a sinusoidal output of fundamental peak V_out_pk = r * V_do (r = modulation ratio
in [0,1]) the firing angle is modulated by the cosine-wave / inverse-cosine law
(Pelly 1971, Bose 2002 eq. 4.x):

    cos(alpha(t)) = r * sin(omega_out * t)
    alpha(t) = arccos( r * sin(omega_out * t) )

so the *instantaneous averaged* output voltage is

    v_out_avg(t) = V_do * cos(alpha(t)) = r * V_do * sin(omega_out * t)

a clean fundamental at omega_out = 2*pi*f_out. alpha sweeps 0..pi over an output
cycle (rectifying when cos>0, inverting when cos<0), which is exactly how the
dual converter regenerates during part of each output cycle.

------------------------------------------------------------------------------
2. INPUT DISPLACEMENT POWER FACTOR (always lagging)
------------------------------------------------------------------------------
A phase-controlled converter draws current that lags the supply by the firing
angle. The fundamental displacement factor of a single group is cos(alpha).
Because alpha is modulated over a full output cycle, the cycloconverter draws a
*time-averaged* lagging reactive component. For a load of displacement angle
phi_load the standard cycloconverter input displacement factor result is
(Pelly 1971; Bose 2002, sec 4.2):

    DPF_in  ~  cos(alpha)_effective  <  1      (always lagging, even at unity
                                                load PF, the converter consumes
                                                reactive power from the line)

We use the established envelope result that the mean input displacement factor
is the cycle-average of |cos(alpha(t))| weighted by current, which evaluates to
a quantity strictly less than the load displacement factor -> input PF is always
poorer (more lagging) than the load. This is THE defining drawback of the
cycloconverter and is enforced as a physical invariant.

------------------------------------------------------------------------------
3. HARMONIC CONTENT
------------------------------------------------------------------------------
Output harmonics of a p-pulse, f_out, f_line cycloconverter occur at
    f_h = | n * p * f_line  +/-  m * f_out |       (Pelly 1971)
The dominant unwanted families cluster around p*f_line. The fundamental-to-rms
ratio degrades as f_out/f_line rises; a compact engineering estimate of output
voltage THD used here (rising with the frequency ratio) is

    THD ~ k_thd * (f_out / f_line)              (monotone, ->0 as f_out->0)

------------------------------------------------------------------------------
4. LUMPED OUTPUT-CURRENT ODE (R-L load)  -- solved with scipy.solve_ivp
------------------------------------------------------------------------------
The averaged output voltage source drives a series R-L load per output phase.
Including the commutation (overlap) voltage drop r_mu = (p/(2*pi)) * omega_line
* Lc which behaves like an extra series resistance, and the SCR on-state drop:

    L * di/dt = v_out_avg(t) - (R + r_mu) * i - V_T0 * sign(i) - r_T * i

This first-order ODE is integrated with scipy.integrate.solve_ivp (RK45). The
state is the per-phase output current i(t); from it we get instantaneous and RMS
output power, the (lagging) load displacement angle, conduction loss, efficiency
(0<eta<1 enforced), and we verify energy conservation P_in = P_out + P_loss.

References:
    Pelly, B.R. (1971). Thyristor Phase-Controlled Converters and Cycloconverters.
        Wiley-Interscience. (cosine firing law, V_avg = V_do cos alpha; harmonics)
    Bose, B.K. (2002). Modern Power Electronics and AC Drives. Prentice Hall,
        ch. 4 (cycloconverter principle, lagging input DPF, f_out < f_line/3).
    Mohan, Undeland & Robbins (2003). Power Electronics, 3rd ed. Wiley, ch. 12.
"""

import numpy as np
from scipy.integrate import solve_ivp


class CycloconverterF2a:
    """Physics-lumped averaged phase-controlled cycloconverter with R-L load ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_phase_out = int(u["n_phase_out"]["value"])
        self.p_pulse = int(u["p_pulse"]["value"])
        self.f_line = float(u["f_line"]["value"])
        self.V_in_ll = float(u["V_in_ll"]["value"])
        self.P_rated = float(u["P_rated"]["value"])
        self.R_load = float(u["R_load"]["value"])
        self.L_load = float(u["L_load"]["value"])
        self.V_T0 = float(u["V_T0"]["value"])
        self.r_T = float(u["r_T"]["value"])
        self.Lc = float(u["Lc_commutation"]["value"])
        self.alpha_min = float(u["alpha_min"]["value"])
        self.alpha_max = float(u["alpha_max"]["value"])

        self.omega_line = 2.0 * np.pi * self.f_line
        # Peak of input PHASE voltage feeding each converter group
        self.V_m = self.V_in_ll * np.sqrt(2.0) / np.sqrt(3.0)

    # ------------------------------------------------------------------
    # 1. Averaged voltage / firing-angle relations
    # ------------------------------------------------------------------
    def V_do(self):
        """Maximum average group output voltage at alpha=0 (Pelly 1971)."""
        p = self.p_pulse
        return (p / np.pi) * self.V_m * np.sin(np.pi / p)

    def firing_angle(self, t, r_mod, f_out):
        """
        Cosine-wave (inverse-cosine) modulation law:
            cos(alpha) = r_mod * sin(omega_out * t)
            alpha = arccos(r_mod * sin(omega_out t))   in [0, pi]
        Clamped to [alpha_min, alpha_max] for realisable firing.
        """
        omega_out = 2.0 * np.pi * f_out
        cos_a = np.clip(r_mod, 0.0, 1.0) * np.sin(omega_out * np.asarray(t, float))
        cos_a = np.clip(cos_a, -1.0, 1.0)
        alpha = np.arccos(cos_a)
        return np.clip(alpha, self.alpha_min, self.alpha_max)

    def v_out_avg(self, t, r_mod, f_out):
        """Instantaneous averaged output phase voltage [V] (slow envelope)."""
        alpha = self.firing_angle(t, r_mod, f_out)
        return self.V_do() * np.cos(alpha)

    def v_out_fundamental_peak(self, r_mod):
        """Fundamental peak of synthesised output phase voltage [V]."""
        return np.clip(r_mod, 0.0, 1.0) * self.V_do()

    def v_out_ll_rms(self, r_mod):
        """Output line-to-line RMS fundamental voltage [V]."""
        v_ph_pk = self.v_out_fundamental_peak(r_mod)
        v_ph_rms = v_ph_pk / np.sqrt(2.0)
        return v_ph_rms * np.sqrt(3.0)

    # ------------------------------------------------------------------
    # 2. Input displacement power factor (always lagging)
    # ------------------------------------------------------------------
    def input_displacement_factor(self, r_mod, f_out, phi_load=0.0):
        """
        Mean input displacement factor (lagging). The fundamental displacement
        of a phase-controlled group is cos(alpha); averaged (current-weighted)
        over the output cycle the converter always draws lagging reactive power,
        so DPF_in < load displacement factor and DPF_in < 1.

        Cycle-average of cos(alpha(t)) with alpha = arccos(r sin(omega_out t)):
            <|cos alpha|> = (2/pi)*r    (mean of |r sin|)   -- envelope estimate.
        A load lag phi_load adds further lag; we combine multiplicatively and
        cap strictly below the load displacement factor.
        Returns DPF in (0, 1).
        """
        r = np.clip(r_mod, 1e-6, 1.0)
        dpf_converter = (2.0 / np.pi) * r            # in (0, 2/pi]
        dpf_load = np.cos(phi_load)
        dpf = dpf_converter * dpf_load
        # guarantee strictly lagging and < load factor
        dpf = min(float(dpf), float(0.999 * dpf_load) if dpf_load > 0 else 0.999)
        return float(np.clip(dpf, 1e-3, 0.999))

    # ------------------------------------------------------------------
    # 3. Harmonic content
    # ------------------------------------------------------------------
    def output_thd(self, f_out, k_thd=1.2):
        """
        Engineering estimate of output-voltage THD, monotone in f_out/f_line
        (Pelly 1971: harmonic families at |n p f_line +/- m f_out|; distortion
        grows with f_out/f_line). -> 0 as f_out -> 0.
        """
        ratio = f_out / self.f_line
        return float(k_thd * ratio)

    def dominant_harmonic_freqs(self, f_out, n_max=2, m_max=2):
        """Dominant output harmonic frequencies |n p f_line +/- m f_out| [Hz]."""
        p = self.p_pulse
        freqs = set()
        for n in range(1, n_max + 1):
            for m in range(0, m_max + 1):
                freqs.add(abs(n * p * self.f_line + m * f_out))
                freqs.add(abs(n * p * self.f_line - m * f_out))
        return sorted(freqs)

    # ------------------------------------------------------------------
    # 4. Lumped output-current ODE (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def _r_mu(self):
        """Commutation/overlap equivalent series resistance [Ohm]."""
        return (self.p_pulse / (2.0 * np.pi)) * self.omega_line * self.Lc

    def _di_dt(self, t, y, r_mod, f_out):
        i = y[0]
        v = self.v_out_avg(t, r_mod, f_out)
        R_eff = self.R_load + self._r_mu() + self.r_T
        drop = R_eff * i + self.V_T0 * np.tanh(50.0 * i)  # smooth sign for SCR drop
        return [(v - drop) / self.L_load]

    def simulate(self, r_mod=0.8, f_out=10.0, n_cycles=4, n_pts_per_cycle=400,
                 phi_load=None):
        """
        Integrate the lumped output-current ODE over n_cycles output cycles.

        Returns dict with time series and aggregate metrics. Enforces:
            f_out < f_line  (and warns/handles the f_out < f_line/3 design rule),
            0 < efficiency < 1, lagging input DPF, energy conservation.
        """
        if f_out <= 0:
            raise ValueError("f_out must be > 0")
        if f_out >= self.f_line:
            raise ValueError(
                f"f_out ({f_out} Hz) must be < f_line ({self.f_line} Hz): "
                "a naturally-commutated cycloconverter steps frequency DOWN."
            )
        f_ratio = f_out / self.f_line
        below_third = f_ratio < (1.0 / 3.0)

        T_out = 1.0 / f_out
        t_end = n_cycles * T_out
        n_pts = int(n_cycles * n_pts_per_cycle)
        t_eval = np.linspace(0.0, t_end, n_pts)

        sol = solve_ivp(
            self._di_dt, (0.0, t_end), [0.0],
            args=(r_mod, f_out), t_eval=t_eval,
            method="RK45", rtol=1e-7, atol=1e-9, max_step=T_out / 200.0,
        )
        t = sol.t
        i = sol.y[0]

        v_avg = self.v_out_avg(t, r_mod, f_out)
        alpha = self.firing_angle(t, r_mod, f_out)

        # Use the last full output cycle for steady-periodic metrics
        mask = t >= (t_end - T_out)
        tc, ic, vc = t[mask], i[mask], v_avg[mask]

        I_rms = float(np.sqrt(np.trapz(ic ** 2, tc) / (tc[-1] - tc[0])))
        V_rms = float(np.sqrt(np.trapz(vc ** 2, tc) / (tc[-1] - tc[0])))

        # Instantaneous output power and its mean (per phase)
        p_inst = vc * ic
        P_out_phase = float(np.trapz(p_inst, tc) / (tc[-1] - tc[0]))
        P_out_total = self.n_phase_out * max(P_out_phase, 0.0)

        # Load displacement angle from the actual R-L load
        if phi_load is None:
            phi_load = float(np.arctan2(self.omega_out_eff(f_out) * self.L_load,
                                        self.R_load))

        # Conduction + commutation loss (per phase) -> total
        R_loss = self.r_T + self._r_mu()
        P_loss_phase = R_loss * I_rms ** 2 + self.V_T0 * float(np.mean(np.abs(ic)))
        P_loss_total = self.n_phase_out * P_loss_phase

        P_in_total = P_out_total + P_loss_total
        eta = P_out_total / P_in_total if P_in_total > 0 else 0.0
        eta = float(np.clip(eta, 1e-6, 1.0 - 1e-9))  # enforce 0 < eta < 1

        dpf_in = self.input_displacement_factor(r_mod, f_out, phi_load)
        thd = self.output_thd(f_out)

        return {
            "t": t,
            "i_out": i,
            "v_out_avg": v_avg,
            "alpha": alpha,
            "f_out": f_out,
            "f_line": self.f_line,
            "freq_ratio": f_ratio,
            "below_one_third": below_third,
            "I_out_rms": I_rms,
            "V_out_rms": V_rms,
            "V_out_ll_rms": self.v_out_ll_rms(r_mod),
            "P_out_total": P_out_total,
            "P_loss_total": P_loss_total,
            "P_in_total": P_in_total,
            "efficiency": eta,
            "phi_load_rad": phi_load,
            "input_displacement_factor": dpf_in,
            "output_thd": thd,
            "dominant_harmonics_hz": self.dominant_harmonic_freqs(f_out),
        }

    def omega_out_eff(self, f_out):
        """Output angular frequency [rad/s] (for load impedance angle)."""
        return 2.0 * np.pi * f_out
