"""
EC162 -- Resonant LLC Converter -- F2a Physics-Lumped (First Harmonic Approximation)

Physics-lumped 0D model of a half-bridge LLC resonant DC-DC converter using the
First Harmonic Approximation (FHA / fundamental-mode approximation). The
square-wave half-bridge excitation and the rectified load are replaced by their
fundamental sinusoidal components, so the resonant tank (L_r, C_r, L_m) is solved
as a linear AC network and the DC voltage conversion ratio follows in closed form.

------------------------------------------------------------------------------
1. Normalised quantities
------------------------------------------------------------------------------
    f_r  = 1 / (2*pi*sqrt(L_r * C_r))        series resonant frequency
    Z_0  = sqrt(L_r / C_r)                    characteristic impedance
    fn   = f_sw / f_r                          normalised switching frequency
    k    = L_m / L_r                           inductance ratio
    R_ac = (8 / pi^2) * n^2 * R_load           FHA-reflected AC load resistance
    Q    = Z_0 / R_ac                          quality factor (loaded)

------------------------------------------------------------------------------
2. DC voltage gain  M(fn, Q, k)   (Steigerwald 1988; Yang 2003, FHA)
------------------------------------------------------------------------------
    M(fn,Q,k) = n * V_out / (V_in/2)
              = 1 / sqrt( ( 1 + 1/k - 1/(k*fn^2) )^2 + Q^2 * ( fn - 1/fn )^2 )

    - At fn = 1 (resonance) the tank term ( fn - 1/fn ) = 0  ->  M = 1
      (load-independent unity gain point), the hallmark of the LLC.
    - For fn < 1 the converter boosts (M > 1, up to a Q-dependent peak);
      for fn > 1 it bucks (M < 1).  The gain therefore peaks at/below
      resonance, giving the characteristic LLC gain curve.

------------------------------------------------------------------------------
3. Soft switching (ZVS)
------------------------------------------------------------------------------
The primary MOSFETs achieve Zero-Voltage-Switching when the tank input impedance
is inductive, i.e. when f_sw lies above the boundary where the magnetizing
current is sufficient to fully discharge the device output capacitance. A robust
FHA proxy used here: ZVS holds when fn >= fn_zvs where fn_zvs is the frequency at
which the tank input phase becomes inductive (input reactance > 0). Operating in
the inductive region (above the gain peak) guarantees ZVS and hence low turn-on
loss. (Steigerwald 1988; Yang 2003.)

------------------------------------------------------------------------------
4. Efficiency (conduction + core losses)
------------------------------------------------------------------------------
    I_pri_rms ~ (pi/(2*sqrt(2))) * I_out / n     fundamental tank current (FHA)
    P_cond_pri = 2 * I_pri_rms^2 * R_ds_on  + I_pri_rms^2 * R_Lr
    P_diode    = 2 * V_f * (I_out/2)              two rectifier diodes
    P_sec      = I_sec_rms^2 * R_sec
    P_core     = K_core * fn^2                    lumped Steinmetz core loss
    eta        = P_out / (P_out + P_loss),     0 < eta < 1

------------------------------------------------------------------------------
5. Lumped output-filter transient ODE  (solve_ivp)
------------------------------------------------------------------------------
The averaged output stage is a current source i_rect (set by the steady-state
tank/gain solution) feeding the output capacitor and load:

    C_out * dV_out/dt = i_rect - V_out / R_load

integrated with scipy.integrate.solve_ivp to give the start-up / load-step
transient toward the FHA steady-state operating point.

References:
    Steigerwald, R.L. (1988). "A comparison of half-bridge resonant converter
        topologies." IEEE Trans. Power Electronics, 3(2), 174-182.
    Yang, B. (2003). "Topology investigation for front end DC/DC power
        conversion for distributed power system." Ph.D. dissertation,
        Virginia Tech (LLC FHA gain derivation).
"""

import numpy as np
from scipy.integrate import solve_ivp


class LLCConverterF2a:
    """LLC resonant converter -- FHA physics-lumped model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_in_nom = u["v_in_nominal"]["value"]
        self.V_out_nom = u["v_out_nominal"]["value"]
        self.n = u["n_turns"]["value"]
        self.L_r = u["L_r"]["value"]
        self.C_r = u["C_r"]["value"]
        self.L_m = u["L_m"]["value"]
        self.C_out = u["C_out"]["value"]
        self.R_ds_on = u["R_ds_on"]["value"]
        self.V_f = u["V_f"]["value"]
        self.R_Lr = u["R_Lr"]["value"]
        self.R_sec = u["R_sec"]["value"]
        self.K_core = u["K_core"]["value"]
        self.P_rated = u["P_rated"]["value"]

        # Derived tank constants
        self.f_r = 1.0 / (2.0 * np.pi * np.sqrt(self.L_r * self.C_r))
        self.Z_0 = np.sqrt(self.L_r / self.C_r)
        self.k = self.L_m / self.L_r

    # ------------------------------------------------------------------ tank
    def f_resonant(self):
        """Series resonant frequency [Hz]."""
        return self.f_r

    def ac_load_resistance(self, r_load):
        """FHA-reflected AC equivalent load resistance seen by the tank [Ohm]."""
        return (8.0 / np.pi ** 2) * self.n ** 2 * np.asarray(r_load, dtype=float)

    def quality_factor(self, r_load):
        """Loaded quality factor Q = Z_0 / R_ac."""
        R_ac = self.ac_load_resistance(r_load)
        return self.Z_0 / R_ac

    def gain(self, fn, Q):
        """
        DC voltage gain M(fn, Q, k) = n*V_out / (V_in/2)  [FHA, Steigerwald 1988].
        """
        fn = np.asarray(fn, dtype=float)
        Q = np.asarray(Q, dtype=float)
        k = self.k
        real = 1.0 + 1.0 / k - 1.0 / (k * fn ** 2)
        imag = Q * (fn - 1.0 / fn)
        return 1.0 / np.sqrt(real ** 2 + imag ** 2)

    def gain_from_load(self, fn, r_load):
        """Convenience: gain from normalised freq and physical load resistance."""
        return self.gain(fn, self.quality_factor(r_load))

    # ----------------------------------------------------------- soft switching
    def tank_input_reactance(self, fn, r_load):
        """
        Imaginary part of the tank input impedance (series L_r,C_r then L_m||R_ac),
        normalised by Z_0. Positive => inductive => ZVS region.
        """
        fn = np.asarray(fn, dtype=float)
        w = fn  # normalised angular frequency (in units where w_r = 1)
        R_ac = self.ac_load_resistance(r_load) / self.Z_0  # normalised
        k = self.k
        # Magnetizing branch jX_Lm = j*k*w in parallel with R_ac
        XLm = k * w
        # Z_par = (jXLm * R)/(R + jXLm)
        denom = R_ac ** 2 + XLm ** 2
        Z_par_re = R_ac * XLm ** 2 / denom
        Z_par_im = R_ac ** 2 * XLm / denom
        # Series L_r (jw) and C_r (-j/w)
        Z_in_im = w - 1.0 / w + Z_par_im
        Z_in_re = Z_par_re
        return Z_in_re, Z_in_im

    def is_zvs(self, fn, r_load):
        """ZVS achieved when tank input impedance is inductive (X_in > 0)."""
        _, X = self.tank_input_reactance(fn, r_load)
        return np.asarray(X) > 0.0

    # ---------------------------------------------------------------- outputs
    def output_voltage(self, fn, r_load, v_in=None):
        """Steady-state output voltage V_out = M * (V_in/2) / n [V]."""
        v_in = self.V_in_nom if v_in is None else v_in
        M = self.gain_from_load(fn, r_load)
        return M * (v_in / 2.0) / self.n

    def operating_point(self, fn, r_load, v_in=None):
        """Steady-state solution: V_out, I_out, P_out, gain, Q, ZVS."""
        v_in = self.V_in_nom if v_in is None else v_in
        r_load = float(r_load)
        Q = self.quality_factor(r_load)
        M = float(self.gain(fn, Q))
        v_out = M * (v_in / 2.0) / self.n
        i_out = v_out / r_load
        return {
            "v_out": v_out,
            "i_out": i_out,
            "p_out": v_out * i_out,
            "gain": M,
            "Q": float(Q),
            "fn": float(fn),
            "zvs": bool(self.is_zvs(fn, r_load)),
        }

    # ------------------------------------------------------------ efficiency
    def loss_breakdown(self, fn, r_load, v_in=None):
        v_in = self.V_in_nom if v_in is None else v_in
        op = self.operating_point(fn, r_load, v_in)
        i_out = op["i_out"]
        # Fundamental primary tank RMS current (FHA): reflect rectified output
        i_pri_rms = (np.pi / (2.0 * np.sqrt(2.0))) * i_out / self.n
        i_sec_rms = i_out * np.sqrt(np.pi ** 2 / 8.0)
        p_cond_pri = 2.0 * i_pri_rms ** 2 * self.R_ds_on + i_pri_rms ** 2 * self.R_Lr
        p_diode = 2.0 * self.V_f * (i_out / 2.0)
        p_sec = i_sec_rms ** 2 * self.R_sec
        p_core = self.K_core * float(np.asarray(fn)) ** 2
        return {
            "p_mosfet_cond_w": p_cond_pri,
            "p_diode_cond_w": p_diode,
            "p_secondary_w": p_sec,
            "p_core_w": p_core,
            "p_total_w": p_cond_pri + p_diode + p_sec + p_core,
        }

    def efficiency(self, fn, r_load, v_in=None):
        """eta = P_out / (P_out + P_loss),  0 < eta < 1."""
        op = self.operating_point(fn, r_load, v_in)
        p_out = op["p_out"]
        p_loss = self.loss_breakdown(fn, r_load, v_in)["p_total_w"]
        p_in = p_out + p_loss
        if p_in <= 0:
            return 0.0
        return p_out / p_in

    # ----------------------------------------------------- transient ODE
    def simulate(self, fn, r_load, v_in=None, v_out0=0.0, dt=1.0e-6, duration_s=2.0e-3):
        """
        Lumped output-filter transient via scipy.integrate.solve_ivp.

            C_out * dV_out/dt = i_rect - V_out / R_load

        i_rect is the rectified tank current that, at steady state, delivers the
        FHA target output voltage into R_load. Modelled as a (slightly stiff)
        averaged source proportional to the gap from the FHA steady-state point,
        capturing the dominant RC settling of the output stage.
        """
        v_in = self.V_in_nom if v_in is None else v_in
        r_load = float(r_load)
        V_ss = self.output_voltage(fn, r_load, v_in)  # FHA steady-state target
        i_ss = V_ss / r_load                          # steady-state load current

        # Effective source conductance of the averaged rectifier (drives V->V_ss).
        # g_src sets the closed-loop pole; chosen >> 1/R_load for fast tank dynamics.
        g_src = 5.0 / r_load

        def rhs(t, y):
            v = y[0]
            i_rect = i_ss + g_src * (V_ss - v)   # averaged rectified current
            dv = (i_rect - v / r_load) / self.C_out
            return [dv]

        t_eval = np.arange(0.0, duration_s + dt / 2.0, dt)
        sol = solve_ivp(
            rhs, (0.0, duration_s), [v_out0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9, max_step=duration_s / 50.0,
        )
        v_out = sol.y[0]
        i_load = v_out / r_load
        p_out = v_out * i_load
        eta_ss = self.efficiency(fn, r_load, v_in)
        return {
            "t": sol.t,
            "v_out": v_out,
            "i_load": i_load,
            "p_out": p_out,
            "v_out_ss": V_ss,
            "efficiency": eta_ss,
            "gain": self.gain_from_load(fn, r_load),
            "zvs": bool(self.is_zvs(fn, r_load)),
        }
