"""
EC161 -- Dual Active Bridge (DAB) DC-DC Converter -- F2a Phase-Shift Averaged Model

Physics-lumped (0D) model of an isolated bidirectional DAB DC-DC converter.

Topology
--------
Two voltage-fed full H-bridges (8 active switches total) linked by an
isolation transformer (turns ratio n = N1/N2) and a series inductor L_s.
Each bridge applies a +-V square wave across the transformer; the secondary
bridge square wave is phase-shifted by phi relative to the primary. The
voltage difference across L_s drives a (quasi-trapezoidal) inductor current,
which is the sole mechanism of power transfer. Reversing the sign of phi
reverses the direction of power flow -> inherently bidirectional.

Single-Phase-Shift (SPS) power transfer  [De Doncker 1991, Eq. 6]
-----------------------------------------------------------------
    P(phi) = (n * V1 * V2) / (2 * pi * f_sw * L_s) * phi * (1 - |phi| / pi)

with phi in [-pi, pi]. (Here V1 = primary bus, V2 = secondary bus reflected
through the model with explicit turns ratio n.) Power is maximum in magnitude
at phi = +-pi/2:
    P_max = (n * V1 * V2) / (8 * f_sw * L_s).

Averaged output-voltage ODE (charge balance on the output capacitor)
--------------------------------------------------------------------
    C_out * dV2/dt = I_out_avg - I_load
    I_out_avg = P_transfer / V2        (average current delivered to bus)
    I_load    = V2 / R_load            (resistive load)
This is a genuine lumped ODE integrated with scipy.solve_ivp.

Soft switching (ZVS)
--------------------
Conventional SPS DAB achieves Zero-Voltage-Switching on both bridges when the
inductor current has the correct polarity at each switching instant. For the
voltage-mismatch case the classic boundary [Kheraluwala 1992] is:
    ZVS on the bridge with the *lower* applied voltage requires
        d = (n * V2) / V1  ->  primary ZVS when d >= 1, secondary ZVS when d <= 1,
    and full-range (both bridges) ZVS at d = 1 for any phi > 0.
A simplified region check is implemented from the inductor-current polarity.

Losses & efficiency
--------------------
  - Conduction:   8 * (I_rms_device)^2 * R_ds_on(T_j)   + transformer copper.
  - Switching:    hard-switching energy only outside the ZVS region; ~0 inside.
Efficiency eta = P_out / (P_out + P_loss) is bounded strictly in (0, 1).

Energy conservation: P_in = P_out + P_loss is enforced by construction
(P_loss >= 0, so 0 < eta < 1 for P_out > 0).

References
----------
De Doncker, R.W.A.A., Divan, D.M., Kheraluwala, M.H. (1991).
    IEEE Trans. Ind. Appl., 27(1), 63-73.
Kheraluwala, M.H., Gascoigne, R.W., Divan, D.M., Baumann, E.D. (1992).
    IEEE Trans. Ind. Appl., 28(6), 1294-1301.
Zhao, B. et al. (2014). CSEE JPES, 1(1), 1-9 (SPS RMS-current expressions).
"""

import numpy as np
from scipy.integrate import solve_ivp


class DAB_F2a:
    """Dual Active Bridge -- phase-shift averaged physics-lumped model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V1_nom = u["v_in_nominal"]["value"]
        self.V2_nom = u["v_out_nominal"]["value"]
        self.n = u["n_turns"]["value"]
        self.L_s = u["L_s"]["value"]
        self.f_sw = u["f_sw"]["value"]
        self.C_out = u["C_out"]["value"]
        self.R_load = u["R_load"]["value"]
        self.R_ds_on_ref = u["R_ds_on"]["value"]
        self.t_on = u["t_on"]["value"]
        self.t_off = u["t_off"]["value"]
        self.R_xfmr = u["R_xfmr"]["value"]
        self.alpha_rds = u["alpha_rds"]["value"]
        self.T_ref = u["T_ref"]["value"]
        self.T_j = u["T_j"]["value"]

    # ------------------------------------------------------------------
    # Core power-transfer physics  (De Doncker 1991, Eq. 6)
    # ------------------------------------------------------------------
    def power_transfer(self, v1, v2, phi):
        """
        SPS power transfer [W], primary -> secondary positive.

        P(phi) = n*V1*V2/(2*pi*f_sw*L_s) * phi * (1 - |phi|/pi)

        Sign of phi sets direction (bidirectional). phi clipped to [-pi, pi].
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        phi = np.clip(np.asarray(phi, dtype=float), -np.pi, np.pi)
        k = self.n * v1 * v2 / (2.0 * np.pi * self.f_sw * self.L_s)
        return k * phi * (1.0 - np.abs(phi) / np.pi)

    def power_max(self, v1, v2):
        """Maximum transferable power [W], attained at phi = pi/2."""
        return self.power_transfer(v1, v2, np.pi / 2.0)

    def phase_for_power(self, v1, v2, p_target):
        """
        Invert the SPS relation for the phase shift phi (small/stable root)
        that delivers p_target. Sign of p_target sets direction.
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        p = np.asarray(p_target, dtype=float)
        pmax = self.power_max(v1, v2)
        pmax = np.where(pmax > 0, pmax, 1.0)
        p_norm = np.clip(p / pmax, -1.0, 1.0)
        # p_norm = 4/pi * phi * (1 - phi/pi) for phi>0; stable root:
        mag = (np.pi / 2.0) * (1.0 - np.sqrt(np.clip(1.0 - np.abs(p_norm), 0.0, 1.0)))
        return np.sign(p_norm) * mag

    # ------------------------------------------------------------------
    # Inductor RMS current (SPS, Zhao 2014)
    # ------------------------------------------------------------------
    def inductor_rms_current(self, v1, v2, phi):
        """
        Transformer / inductor primary-referred RMS current [A] for SPS.

        Computed directly from the first-principles piecewise-linear inductor
        current. Over a half switching period (angle theta in [0, pi]) the
        primary bridge applies +V1 and the secondary applies +/- n*V2 phase-
        shifted by |phi|, so the voltage across L_s is:
            v_L = V1 + n*V2   for theta in [0, |phi|)      (secondary still -)
            v_L = V1 - n*V2   for theta in [|phi|, pi)     (secondary now +)
        di/dtheta = v_L / (omega * L_s). Half-wave symmetry fixes the DC offset
        so that i(pi) = -i(0). We integrate analytically over the two segments
        and form the RMS of the piecewise-linear waveform. This is exact for
        SPS (De Doncker 1991 / Krismer 2010) and monotone in |phi|.
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        ph = np.clip(np.abs(np.asarray(phi, dtype=float)), 0.0, np.pi)

        omega = 2.0 * np.pi * self.f_sw
        s = 1.0 / (omega * self.L_s)          # current slope per unit (V*angle)
        nv2 = self.n * v2

        # Slopes (A per radian) on the two segments:
        m1 = (v1 + nv2) * s                    # theta in [0, ph]
        m2 = (v1 - nv2) * s                    # theta in [ph, pi]
        # Half-wave symmetry: i(pi) = -i(0). With i(ph)=i0+m1*ph and
        # i(pi)=i(ph)+m2*(pi-ph) = -i0  -> solve for i0:
        i0 = -(m1 * ph + m2 * (np.pi - ph) + 0.0) / 2.0
        i_ph = i0 + m1 * ph                    # current at the breakpoint

        # RMS^2 = (1/pi) * [ integral_0^ph (i0+m1 t)^2 dt
        #                  + integral_ph^pi (i_ph+m2 (t-ph))^2 dt ]
        def seg_int(a, slope, length):
            # integral_0^length (a + slope*t)^2 dt
            return (a * a) * length + a * slope * length ** 2 + (slope ** 2) * length ** 3 / 3.0

        integ = seg_int(i0, m1, ph) + seg_int(i_ph, m2, np.pi - ph)
        rms_sq = np.clip(integ / np.pi, 0.0, None)
        return np.sqrt(rms_sq)

    # ------------------------------------------------------------------
    # ZVS region
    # ------------------------------------------------------------------
    def zvs_region(self, v1, v2, phi):
        """
        Return dict with ZVS boundary check [Kheraluwala 1992].

        d = n*V2/V1. For phi>0:
          - primary bridge ZVS when d >= 1 (or phi large enough),
          - secondary bridge ZVS when d <= 1 (or phi large enough).
        Both bridges soft-switch when d == 1 for any phi != 0.
        We return whether each bridge satisfies the inductor-current-polarity
        condition i_L < 0 at the primary switching instant (necessary for ZVS).
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        phi = np.asarray(phi, dtype=float)
        d = self.n * v2 / np.where(v1 > 0, v1, 1.0)
        aphi = np.abs(phi)
        # Classic SPS ZVS conditions (Kheraluwala 1992 / Krismer 2010):
        #   primary ZVS:   d >= 1 - 2*phi/pi
        #   secondary ZVS: d <= 1/(1 - 2*phi/pi)   (i.e. (1/d) >= 1 - 2*phi/pi)
        thr = 1.0 - 2.0 * aphi / np.pi
        primary_zvs = d >= thr
        secondary_zvs = (1.0 / np.where(d > 0, d, 1e-9)) >= thr
        both = primary_zvs & secondary_zvs
        return {
            "d": d,
            "primary_zvs": primary_zvs,
            "secondary_zvs": secondary_zvs,
            "full_zvs": both,
        }

    # ------------------------------------------------------------------
    # Losses & efficiency
    # ------------------------------------------------------------------
    def _rds_on(self):
        return self.R_ds_on_ref * (1.0 + self.alpha_rds * (self.T_j - self.T_ref))

    def losses(self, v1, v2, phi):
        """
        Total averaged power loss [W]: conduction (8 switches + transformer)
        plus hard-switching loss applied only outside the ZVS region.
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        phi = np.asarray(phi, dtype=float)

        i_rms = self.inductor_rms_current(v1, v2, phi)
        R_ds = self._rds_on()

        # Each of the 8 devices carries i_rms in RMS for ~half the period:
        p_cond_sw = 8.0 * 0.5 * (i_rms ** 2) * R_ds
        p_cond_xfmr = (i_rms ** 2) * self.R_xfmr
        p_cond = p_cond_sw + p_cond_xfmr

        # Hard-switching loss only where ZVS is lost:
        zvs = self.zvs_region(v1, v2, phi)
        i_peak = i_rms * np.sqrt(2.0)
        e_sw_pri = 0.5 * v1 * i_peak * (self.t_on + self.t_off)
        e_sw_sec = 0.5 * (self.n * v2) * i_peak * (self.t_on + self.t_off)
        p_sw_pri = np.where(zvs["primary_zvs"], 0.0, 4.0 * e_sw_pri * self.f_sw)
        p_sw_sec = np.where(zvs["secondary_zvs"], 0.0, 4.0 * e_sw_sec * self.f_sw)
        p_sw = p_sw_pri + p_sw_sec

        return p_cond + p_sw

    def efficiency(self, v1, v2, phi):
        """Efficiency in (0,1): eta = P_out / (P_out + P_loss)."""
        p_t = np.abs(self.power_transfer(v1, v2, phi))
        p_loss = self.losses(v1, v2, phi)
        p_in = p_t + p_loss
        eta = np.where(p_in > 0, p_t / np.where(p_in > 0, p_in, 1.0), 0.0)
        return eta

    # ------------------------------------------------------------------
    # Averaged output-voltage ODE  (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def _dvdt(self, t, y, phi_fn, v1_fn, r_load):
        v2 = max(float(y[0]), 1e-3)
        phi = phi_fn(t)
        v1 = v1_fn(t)
        p_t = self.power_transfer(v1, v2, phi)         # primary -> secondary
        p_loss = float(self.losses(v1, v2, phi))
        # Power delivered to the output bus = transferred power minus losses
        # (losses are shared but for the averaged bus we charge with net out):
        p_out = p_t - np.sign(p_t) * p_loss
        i_out_avg = p_out / v2
        i_load = v2 / r_load
        return [(i_out_avg - i_load) / self.C_out]

    def simulate(self, phi, v1=None, v2_0=None, r_load=None,
                 dt=2e-5, duration_s=2e-3):
        """
        Integrate the averaged output-voltage ODE.

        phi      : float OR callable(t)->phi [rad], the control input.
        v1       : float OR callable(t)->V1 [V] (primary bus). Default nominal.
        v2_0     : initial output-bus voltage [V]. Default nominal.
        r_load   : output resistive load [Ohm]. Default nominal.
        dt       : output sample spacing [s].
        duration_s : total simulated time [s].

        Returns dict of time-series arrays.
        """
        v1 = self.V1_nom if v1 is None else v1
        v2_0 = self.V2_nom if v2_0 is None else v2_0
        r_load = self.R_load if r_load is None else r_load

        phi_fn = phi if callable(phi) else (lambda t, _p=float(phi): _p)
        v1_fn = v1 if callable(v1) else (lambda t, _v=float(v1): _v)

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        sol = solve_ivp(
            self._dvdt, (0.0, duration_s), [v2_0],
            t_eval=t_eval, args=(phi_fn, v1_fn, r_load),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )
        t = sol.t
        v2 = np.maximum(sol.y[0], 1e-3)
        phi_arr = np.array([phi_fn(tt) for tt in t])
        v1_arr = np.array([v1_fn(tt) for tt in t])

        p_transfer = self.power_transfer(v1_arr, v2, phi_arr)
        p_loss = self.losses(v1_arr, v2, phi_arr)
        eta = self.efficiency(v1_arr, v2, phi_arr)
        i_rms = self.inductor_rms_current(v1_arr, v2, phi_arr)
        zvs = self.zvs_region(v1_arr, v2, phi_arr)

        return {
            "t": t,
            "v_out": v2,
            "phi": phi_arr,
            "power_transfer": p_transfer,
            "power_loss": p_loss,
            "efficiency": eta,
            "i_rms": i_rms,
            "full_zvs": zvs["full_zvs"],
        }
