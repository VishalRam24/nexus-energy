"""
EC183 -- Circuit Breaker -- F2a Cassie-Mayr Arc Interruption Model

Physics-lumped (0D) model of AC fault interruption. Two coupled ODE systems
are integrated together with scipy.integrate.solve_ivp:

1) Arc conductance g(t) -- black-box arc model.
   The arc column is described by an energy-balance ODE for ln(g):

       d(ln g)/dt = (1/tau) * ( u*i / P(...) - 1 )

   Two classical closure laws for the cooling power P and time constant tau:

     * Cassie (1939):   high-current regime, arc cooled by convection,
       constant arc voltage U_c so that  P = U_c^2 * g   and tau = tau_C.
       =>  dg/dt = (g/tau_C) * ( u^2 / U_c^2 - 1 )

     * Mayr (1943):     near current-zero regime, arc cooled by thermal
       conduction, constant loss power P0 so that  P = P0  and tau = tau_M.
       =>  d(ln g)/dt = (1/tau_M) * ( u*i / P0 - 1 )

   We use a Cassie-Mayr series/blended model (Habedank 1988 style): the
   effective P and tau transition from the Cassie law at high current to the
   Mayr law near current zero via a smooth current-dependent weight, which
   captures both the high-current arc voltage plateau and the current-zero
   thermal-reignition behaviour relevant to interruption.

2) Circuit -- a single-phase test loop (CIGRE first-pole-to-clear equivalent):

       source emf  e(t) = Vpk * sin(w t + phi)
       series  R_s, L_s   (set the prospective fault current)
       breaker arc conductance g(t)  in parallel with TRV capacitance C

   KCL/KVL with arc voltage u across the gap:

       L_s di/dt = e(t) - R_s*i - u
       C  du/dt  = i - g*u            (i = inductor/line current)

   At current zero, if the arc fails to reignite (g collapses toward g_min),
   the capacitor C and the source ring up the Transient Recovery Voltage
   (TRV). Interruption succeeds iff the post-arc conductance stays low AND
   the TRV peak does not exceed the gap dielectric withstand.

Enforced physics:
   * Arc extinguishes at a current zero only if TRV is withstood.
   * Arc energy E_arc = integral(u*i dt) is accounted (Joule integral).
   * Breaking-capacity limit: I_fault > I_interrupt_kA => thermal failure flag.
   * Interruption time measured from contact separation to final current zero.

References:
   Cassie, A.M. (1939). "Arc rupture and circuit severity: a new theory."
       CIGRE Report 102, Paris.
   Mayr, O. (1943). "Beitraege zur Theorie des statischen und des
       dynamischen Lichtbogens." Archiv f. Elektrotechnik 37, 588-608.
   Habedank, U. (1988). "Application of a new arc model for the evaluation
       of short-circuit breaking tests." ETZ Archiv 10, 339-343.
   CIGRE WG 13.01 (1998). "Applications of black box modelling to circuit
       breakers." Electra 149.
   IEC 62271-100: High-voltage switchgear -- AC circuit-breakers.
"""

import numpy as np
from scipy.integrate import solve_ivp


class CircuitBreakerArc_F2a:
    """Cassie-Mayr arc interruption model with TRV circuit coupling."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_rated = u["V_rated_kV"]["value"] * 1e3       # V (RMS ph-ph)
        self.I_rated = u["I_rated_A"]["value"]              # A
        self.I_interrupt = u["I_interrupt_kA"]["value"] * 1e3  # A
        self.f = u["f_Hz"]["value"]                         # Hz

        # Arc model constants
        self.U_c = u["U_c_Cassie_V"]["value"]               # V
        self.tau_C = u["tau_Cassie_us"]["value"] * 1e-6     # s
        self.P0 = u["P_loss_Mayr_W"]["value"]               # W
        self.tau_M = u["tau_Mayr_us"]["value"] * 1e-6       # s
        self.g_min = u["g_min_S"]["value"]                  # S
        self.g0 = u["g0_S"]["value"]                        # S

        # Circuit
        self.L_s = u["L_source_mH"]["value"] * 1e-3         # H
        self.R_s = u["R_source_ohm"]["value"]               # Ohm
        self.C = u["C_trv_nF"]["value"] * 1e-9              # F
        self.k_pp = u["k_pp"]["value"]
        self.k_af = u["k_af"]["value"]

        self.t_open = u["t_open_ms"]["value"] * 1e-3        # s
        self.u_diel = u["u_dielectric_kV"]["value"] * 1e3   # V (peak withstand)

        self.w = 2.0 * np.pi * self.f

        # per-run loop impedance (sized in simulate(); defaults = nameplate)
        self._L_run = self.L_s
        self._R_run = self.R_s

    # -- arc closure laws ---------------------------------------------------
    def _arc_P_tau(self, i):
        """Blended Cassie-Mayr cooling power P and time constant tau.

        Weight w_hi in [0,1] selects Cassie (high current) vs Mayr (low
        current). Transition near a characteristic current i_t = P0/U_c
        (the current at which the two loss laws are comparable).
        """
        i_abs = np.abs(i)
        i_t = self.P0 / self.U_c                     # ~ crossover current
        w_hi = i_abs**2 / (i_abs**2 + i_t**2)        # ->1 high I, ->0 at zero
        return w_hi, i_t

    def _arc_dlng(self, ln_g, u, i, w_diel):
        """d(ln g)/dt for the blended Cassie-Mayr arc with dielectric gate.

        Cassie term:  (1/tau_C) * ( u^2/U_c^2 - 1 )      [P_C = U_c^2 g]
        Mayr   term:  (1/tau_M) * ( u*i / P0   - 1 )      [P_M = P0]
        Blend by current weight w_hi.

        Dielectric gate (w_diel in [0,1]): once the contacts have opened and
        the arc has extinguished at a current zero, reconduction is a
        *dielectric* breakdown event, not continuous thermal regrowth. The
        gate multiplies any *positive* (regrowth) part of d(ln g)/dt by
        w_diel, which collapses to 0 while |u| < dielectric withstand and
        rises to 1 only if the recovery voltage exceeds the gap strength
        (thermal/dielectric reignition). This is the standard hybrid
        thermal+dielectric black-box criterion (CIGRE WG 13.01, 1998).
        """
        w_hi, _ = self._arc_P_tau(i)
        dlng_cassie = (1.0 / self.tau_C) * (u * u / (self.U_c**2) - 1.0)
        dlng_mayr = (1.0 / self.tau_M) * (u * i / self.P0 - 1.0)
        dlng = w_hi * dlng_cassie + (1.0 - w_hi) * dlng_mayr
        # suppress regrowth while the open gap holds off the recovery voltage
        if dlng > 0.0:
            dlng *= w_diel
        return dlng

    def _dielectric_gate(self, t, u):
        """Reconduction weight: 0 = gap holds, 1 = gap breaks down.

        Before contacts open (t < t_open) the gate is fully on (arc conducts
        freely). After opening, the gap withstand rises; the gate is a smooth
        function of (|u| / u_withstand) so the arc can only re-strike if the
        recovery voltage exceeds the dielectric strength.
        """
        if t < self.t_open:
            return 1.0
        ratio = abs(u) / self.u_diel
        # sharp threshold near ratio = 1 (breakdown)
        return 1.0 / (1.0 + np.exp(-(ratio - 1.0) * 20.0))

    # -- coupled ODE rhs ----------------------------------------------------
    def _rhs(self, t, y, Vpk, phi):
        """State y = [i, u, ln_g, E_arc].

        i     : line current (through L_s)            [A]
        u     : arc / gap voltage (across C)          [V]
        ln_g  : log arc conductance                   [-]
        E_arc : accumulated arc energy                [J]
        """
        i, u, ln_g, _ = y
        g = np.exp(ln_g)

        e = Vpk * np.sin(self.w * t + phi)
        di = (e - self._R_run * i - u) / self._L_run
        # capacitor node: i_line = i_cap + i_arc ; i_arc = g*u
        du = (i - g * u) / self.C
        w_diel = self._dielectric_gate(t, u)
        dlng = self._arc_dlng(ln_g, u, i, w_diel)
        dE = u * i                          # instantaneous arc power
        return [di, du, dlng, dE]

    # -- simulation ---------------------------------------------------------
    def simulate(self, I_fault_kA=None, phi=None, dt_us=0.01,
                 duration_ms=20.0, Vpk=None):
        """Integrate the arc + circuit through contact separation.

        Parameters
        ----------
        I_fault_kA : prospective symmetrical fault current [kA]. The source
                     EMF peak is fixed at the system phase voltage; the loop
                     inductance L_s is adjusted so this RMS fault current
                     flows. Hence the TRV is bounded by system voltage
                     (physical) while di/dt at current zero scales with the
                     fault current -- the true breaking-capacity driver.
        phi        : source phase at t=0 [rad]. Default puts a current zero
                     a few ms into the window.
        dt_us      : output sample step [us].
        duration_ms: total simulated time [ms].
        Vpk        : override peak source emf [V] (else system phase voltage).

        Returns dict with time series + interruption verdict.
        """
        # --- source EMF peak = system phase voltage (fixed by grid) ---
        Vph_pk = self.V_rated * np.sqrt(2.0 / 3.0)        # phase peak [V]
        if Vpk is None:
            Vpk = Vph_pk
        if phi is None:
            phi = 0.0

        # --- size the loop impedance for the requested fault current ---
        if I_fault_kA is not None:
            Irms_target = I_fault_kA * 1e3
        else:
            Irms_target = self.I_interrupt                # rate at breaking cap.
        # Z = Vrms / Irms ; keep base R/X ratio, solve L_s for this Z
        Z = (Vpk / np.sqrt(2.0)) / Irms_target
        rx = self.R_s / (self.w * self.L_s)               # base R/X ratio
        X = Z / np.hypot(rx, 1.0)
        L_s = X / self.w
        R_s = rx * X
        # store for the rhs closure
        self._L_run, self._R_run = L_s, R_s

        Irms_actual = (Vpk / np.sqrt(2.0)) / Z
        I_fault_actual_kA = Irms_actual / 1e3

        T = duration_ms * 1e-3
        dt = dt_us * 1e-6
        n = int(round(T / dt)) + 1
        t_eval = np.linspace(0.0, T, n)

        # initial current = steady fault current at t=0 (arc fully on)
        # i(0) from phasor of the L-R loop driven by e(t):
        theta_z = np.arctan2(self.w * L_s, R_s)
        i0 = (Vpk / Z) * np.sin(phi - theta_z)
        u0 = 0.0
        y0 = [i0, u0, np.log(self.g0), 0.0]

        sol = solve_ivp(
            self._rhs, (0.0, T), y0, t_eval=t_eval,
            args=(Vpk, phi), method="LSODA",
            rtol=1e-6, atol=[1e-3, 1e-2, 1e-6, 1e-3],
            max_step=20 * dt,
        )

        t = sol.t
        i = sol.y[0]
        u = sol.y[1]
        # clamp to the physical residual (leakage) conductance floor g_min so
        # the post-arc gap retains a tiny non-zero conductance (no underflow)
        g = np.maximum(np.exp(sol.y[2]), self.g_min)
        E_arc = sol.y[3]

        # --- detect current zeros (sign changes of i) ---
        sign = np.sign(i)
        zero_idx = np.where(np.diff(sign) != 0)[0]

        # --- TRV: peak gap voltage after the last current zero ---
        if len(zero_idx) > 0:
            last_zi = zero_idx[-1]
            trv_peak = float(np.max(np.abs(u[last_zi:])))
            t_interrupt = float(t[last_zi])
        else:
            trv_peak = float(np.max(np.abs(u)))
            t_interrupt = float("nan")

        # --- post-arc conductance: did the arc actually go out? ---
        g_final = float(g[-1])
        arc_extinguished = bool(g_final < 1e-3)   # collapsed to residual

        # --- CIGRE reference TRV peak (first-pole-to-clear) ---
        Vph_pk = self.V_rated * np.sqrt(2.0 / 3.0)        # phase peak
        trv_ref_peak = self.k_pp * self.k_af * Vph_pk

        # --- breaking-capacity limit ---
        within_capacity = bool(I_fault_actual_kA <= (self.I_interrupt / 1e3))

        # --- dielectric withstand of the (cold) open gap ---
        trv_withstood = bool(trv_peak <= self.u_diel)

        # --- overall interruption verdict ---
        success = bool(arc_extinguished and trv_withstood and within_capacity
                       and len(zero_idx) > 0)

        return {
            "t": t,
            "current": i,
            "arc_voltage": u,
            "conductance": g,
            "arc_energy": E_arc,
            "arc_energy_total_J": float(E_arc[-1]),
            "I_fault_kA": I_fault_actual_kA,
            "current_zeros": zero_idx,
            "n_current_zeros": int(len(zero_idx)),
            "trv_peak_V": trv_peak,
            "trv_ref_peak_V": float(trv_ref_peak),
            "g_final_S": g_final,
            "arc_extinguished": arc_extinguished,
            "trv_withstood": trv_withstood,
            "within_capacity": within_capacity,
            "interruption_success": success,
            "interruption_time_s": t_interrupt,
            "Vpk": float(Vpk),
            "phi": float(phi),
        }

    # -- convenience: static arc voltage / characteristic --------------------
    def static_arc_voltage(self, i):
        """Quasi-static Cassie arc voltage U_c (column voltage plateau)."""
        return self.U_c

    def crossover_current(self):
        """Current at which Cassie and Mayr cooling are comparable [A]."""
        return self.P0 / self.U_c
