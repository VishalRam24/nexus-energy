"""
EC184 -- Power Factor Correction Unit -- F2a Physics-Lumped (Switching Transient)

Physics-lumped model of a shunt capacitor bank PFC unit. Combines the
steady-state reactive-power compensation algebra with a lumped first-order /
second-order ODE for the capacitor-energization (inrush) transient.

Governing physics
------------------
1. Reactive compensation (Grainger & Stevenson 1994, Ch.6; IEEE 1036-2010):
       phi1 = arccos(pf1),  phi2 = arccos(pf2)
       Q_load     = P * tan(phi1)
       Qc_req     = P * (tan(phi1) - tan(phi2))          [reactive demanded]
       Q_residual = Q_load - Qc                          [Qc = actual delivered]
       pf_achieved = P / sqrt(P^2 + Q_residual^2)
   Apparent power before/after:
       S1 = P/pf1,  S2 = sqrt(P^2 + Q_residual^2)
       released_capacity = S1 - S2     [kVA freed upstream]

2. Capacitor sizing:
       C = Qc / (2*pi*f*V_LL^2)         (single-phase-equivalent farads)

3. Detuned-reactor / harmonic resonance (IEEE 1036-2010 Annex; Grainger&Stevenson):
   A shunt capacitor C in parallel with system inductance Lsys forms a parallel
   resonance at
       h_res = f_res/f = sqrt(Ssc / Qc) = sqrt(Xc / Xsys)
   With a series detuning reactor of relative reactance p (p = X_L/X_C at f),
   the bank+reactor leg is series-resonant at
       h_tune = 1 / sqrt(p)
   chosen below the lowest troublesome harmonic (e.g. p=7% -> h=3.78, below 5th).

4. Voltage rise from capacitor injection (IEEE 1036-2010):
       dV/V ~ Qc / Ssc        (Ssc = short-circuit MVA at the bus)

5. Energization inrush transient -- lumped series RLC ODE
   (Grainger & Stevenson 1994; IEEE 1036-2010 inrush current):
   Switching the bank onto the source closes a loop L_eff-R_eff-C driven by the
   instantaneous source voltage v_s(t) = Vm*sin(w t + theta). Per-phase KVL:
       L_eff i'  + R_eff i + v_C = v_s(t)
       C v_C'    = i
   State x = [i, v_C]. This is integrated with scipy.solve_ivp. The natural
   ringing frequency is w0 = 1/sqrt(L_eff*C), damped by R_eff; the peak inrush
   current is the classic IEEE-1036 first-peak overshoot.

References
----------
- IEEE Std 1036-2010, IEEE Guide for the Application of Shunt Power Capacitors.
- J.J. Grainger & W.D. Stevenson (1994), Power System Analysis, McGraw-Hill,
  Chapters 5-6 (reactive power, capacitor compensation, resonance).
- IEC 60871-1 shunt capacitor standard (detuning reactors).
"""

import numpy as np
from scipy.integrate import solve_ivp


class PFCUnit_F2a:
    """Physics-lumped capacitor-bank PFC: compensation + resonance + inrush ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated_kVAR = float(u["Q_rated_kVAR"]["value"])
        self.V_rated_kV = float(u["V_rated_kV"]["value"])
        self.f = float(u["f_system_Hz"]["value"])
        self.w = 2.0 * np.pi * self.f
        self.n_stages = int(u["n_stages"]["value"])
        self.detuning_pct = float(u["detuning_pct"]["value"])
        self.loss_factor = float(u["loss_factor"]["value"])
        self.esr = float(u["esr_ohm"]["value"])
        self.Lsys = float(u["Lsys_mH"]["value"]) * 1e-3      # H
        self.pf_target_default = float(u["pf_target"]["value"])

        # Single-phase-equivalent capacitance of the full rated bank [F]
        V_ll = self.V_rated_kV * 1e3
        self.C_rated = (self.Q_rated_kVAR * 1e3) / (self.w * V_ll ** 2)

    # ------------------------------------------------------------------
    # 1. Steady-state reactive compensation
    # ------------------------------------------------------------------
    def compensate(self, P_kW, pf_initial, pf_target=None,
                   Q_comp_override_kVAR=None):
        """Reactive-power compensation algebra. Returns a dict of scalars."""
        if pf_target is None:
            pf_target = self.pf_target_default

        P = float(P_kW)
        pf1 = float(np.clip(pf_initial, 1e-6, 1.0 - 1e-9))
        pf2 = float(np.clip(pf_target, 1e-6, 1.0 - 1e-9))

        phi1 = np.arccos(pf1)
        phi2 = np.arccos(pf2)

        Q_load = P * np.tan(phi1)
        Qc_req = P * (np.tan(phi1) - np.tan(phi2))

        if Q_comp_override_kVAR is not None:
            Qc = float(np.clip(Q_comp_override_kVAR, 0.0, self.Q_rated_kVAR))
        else:
            Qc = float(np.clip(Qc_req, 0.0, self.Q_rated_kVAR))

        Q_residual = Q_load - Qc

        S1 = np.sqrt(P ** 2 + Q_load ** 2)               # before (= P/pf1)
        S2 = np.sqrt(P ** 2 + Q_residual ** 2)           # after
        pf_achieved = P / S2 if S2 > 0 else 1.0
        released_capacity = S1 - S2                      # kVA freed upstream

        P_loss = self.loss_factor * Qc                   # bank losses [kW]

        # number of switched stages required (round up)
        stage_kVAR = self.Q_rated_kVAR / self.n_stages
        stages_on = int(min(self.n_stages, np.ceil(Qc / stage_kVAR))) if Qc > 0 else 0

        return {
            "Q_load_kVAR": Q_load,
            "Q_required_kVAR": Qc_req,
            "Q_compensated_kVAR": Qc,
            "Q_residual_kVAR": Q_residual,
            "S_before_kVA": S1,
            "S_after_kVA": S2,
            "pf_initial": pf1,
            "pf_achieved": pf_achieved,
            "released_capacity_kVA": released_capacity,
            "P_loss_kW": P_loss,
            "bank_utilization": Qc / self.Q_rated_kVAR,
            "stages_on": stages_on,
        }

    # ------------------------------------------------------------------
    # 2. Capacitor sizing for a given Qc
    # ------------------------------------------------------------------
    def capacitance_for_Q(self, Qc_kVAR):
        """Single-phase-equivalent capacitance [F] for reactive power Qc."""
        V_ll = self.V_rated_kV * 1e3
        Qc = max(float(Qc_kVAR), 0.0)
        return (Qc * 1e3) / (self.w * V_ll ** 2)

    # ------------------------------------------------------------------
    # 3. Harmonic resonance
    # ------------------------------------------------------------------
    def resonance(self, Qc_kVAR, Lsys_H=None, detuning_pct=None):
        """Parallel-resonance harmonic order and detuned series-tuning order.

        h_par  = sqrt(Xc / Xsys)  parallel resonance (cap || system L)
        h_tune = 1/sqrt(p)        series resonance of cap + detuning reactor
        """
        if Lsys_H is None:
            Lsys_H = self.Lsys
        if detuning_pct is None:
            detuning_pct = self.detuning_pct

        Qc = max(float(Qc_kVAR), 1e-9)
        V_ll = self.V_rated_kV * 1e3
        Xc = V_ll ** 2 / (Qc * 1e3)            # capacitive reactance at f [Ohm]
        Xsys = self.w * Lsys_H                  # system reactance at f [Ohm]
        h_par = np.sqrt(Xc / Xsys)              # parallel resonance order
        f_par = h_par * self.f

        p = detuning_pct / 100.0
        h_tune = 1.0 / np.sqrt(p) if p > 0 else float("inf")
        f_tune = h_tune * self.f

        # short-circuit MVA at the bus from Lsys
        Ssc_MVA = (V_ll ** 2 / Xsys) / 1e6
        return {
            "Xc_ohm": Xc,
            "Xsys_ohm": Xsys,
            "h_parallel": h_par,
            "f_parallel_Hz": f_par,
            "h_tune": h_tune,
            "f_tune_Hz": f_tune,
            "Ssc_MVA": Ssc_MVA,
        }

    def voltage_rise(self, Qc_kVAR, Lsys_H=None):
        """Per-unit voltage rise from capacitor injection dV/V ~ Qc/Ssc."""
        res = self.resonance(Qc_kVAR, Lsys_H)
        Ssc_kVA = res["Ssc_MVA"] * 1e3
        return float(Qc_kVAR) / Ssc_kVA if Ssc_kVA > 0 else 0.0

    # ------------------------------------------------------------------
    # 4. Energization inrush transient -- lumped RLC ODE
    # ------------------------------------------------------------------
    def _rlc_rhs(self, t, x, L, R, C, Vm, theta):
        i, vC = x
        vs = Vm * np.sin(self.w * t + theta)
        di = (vs - R * i - vC) / L
        dvC = i / C
        return [di, dvC]

    def energize(self, Qc_kVAR=None, Lsys_H=None, R_eff=None,
                 detuning_pct=0.0, switch_angle_deg=90.0,
                 duration_s=0.06, n_points=2000, v0_cap=0.0):
        """Simulate capacitor-bank energization inrush via scipy.solve_ivp.

        Parameters
        ----------
        Qc_kVAR       : reactive size of the energized bank (default rated)
        Lsys_H        : source inductance in the loop (default param value)
        R_eff         : loop resistance (default bank ESR)
        detuning_pct  : add a series detuning reactor of p% (adds inductance)
        switch_angle_deg : point-on-wave closing angle (90 deg = voltage peak)
        duration_s    : sim length (a few cycles)
        v0_cap        : pre-charge voltage on the cap [V]

        Returns dict with time series and inrush metrics.
        """
        if Qc_kVAR is None:
            Qc_kVAR = self.Q_rated_kVAR
        if Lsys_H is None:
            Lsys_H = self.Lsys
        if R_eff is None:
            R_eff = self.esr

        C = self.capacitance_for_Q(Qc_kVAR)
        Xc = 1.0 / (self.w * C)
        # detuning reactor adds inductance L_d = p * Xc / w
        p = detuning_pct / 100.0
        L_d = p * Xc / self.w
        L_eff = Lsys_H + L_d
        L_eff = max(L_eff, 1e-9)

        Vm = np.sqrt(2.0 / 3.0) * (self.V_rated_kV * 1e3)   # phase peak
        theta = np.deg2rad(switch_angle_deg)

        t_eval = np.linspace(0.0, duration_s, n_points)
        sol = solve_ivp(
            self._rlc_rhs, (0.0, duration_s), [0.0, v0_cap],
            t_eval=t_eval, args=(L_eff, R_eff, C, Vm, theta),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=duration_s / n_points,
        )

        i = sol.y[0]
        vC = sol.y[1]
        # steady-state rms current the bank draws: I_ss = V_phase / Xc
        I_ss_rms = (Vm / np.sqrt(2.0)) / Xc
        I_ss_peak = Vm / Xc
        I_peak = float(np.max(np.abs(i)))
        inrush_factor = I_peak / I_ss_peak if I_ss_peak > 0 else 0.0

        w0 = 1.0 / np.sqrt(L_eff * C)               # natural ang. freq
        f0 = w0 / (2.0 * np.pi)
        zeta = (R_eff / 2.0) * np.sqrt(C / L_eff)    # damping ratio

        return {
            "t": sol.t,
            "i": i,
            "v_cap": vC,
            "C_F": C,
            "L_eff_H": L_eff,
            "R_eff_ohm": R_eff,
            "I_peak_A": I_peak,
            "I_ss_peak_A": I_ss_peak,
            "I_ss_rms_A": I_ss_rms,
            "inrush_factor": inrush_factor,
            "f_natural_Hz": f0,
            "damping_ratio": zeta,
            "v_cap_final_V": float(vC[-1]),
            "v_cap_max_V": float(np.max(np.abs(vC))),
        }
