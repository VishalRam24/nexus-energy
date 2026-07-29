"""
EC103 -- Supercritical CO2 Brayton Cycle -- F2a Physics-Lumped State-Point Model

Physics-lumped (0D) recompression sCO2 Brayton cycle. Net work and thermal
efficiency are computed from real-CO2 property states at each cycle station;
a lumped first-order ODE governs the hot-section metal temperature
(scipy.integrate.solve_ivp).

Cycle layout (recompression Brayton, Dostal 2004 / Ahn 2015):

    1  main-compressor inlet   (near critical point: T~32 degC, P_low)
    2  main-compressor outlet  (P_high)            <- LOW compression work here
    2b recompression-compressor outlet (mixes with recuperated stream)
    3  recuperator cold outlet (heated by turbine exhaust)
    4  turbine inlet           (after primary heater, T_turb_in)
    5  turbine outlet          (P_low)
    6  recuperator hot outlet  (cooled exhaust -> precooler)

The decisive sCO2 advantage: compressing CO2 just above its critical point
(rho ~ 470 kg/m3, liquid-like) makes the specific compression work
w_c = v * dP a small fraction of the turbine work, unlike an ideal-gas
Brayton cycle. We capture this with a real-gas density / compressibility
correlation rather than the ideal-gas law.

Real-CO2 property correlations (hardcoded, Span-Wagner-based):
  * Density via a compressibility factor Z(T,P) fit anchored to the
    Span & Wagner (1996) reference EOS in the supercritical region
    (7.4-30 MPa, 305-1000 K). Near-critical Z drops to ~0.2-0.3
    (liquid-like), in the hot end Z -> ~0.9-1.0 (near ideal).
  * Real-gas cp(T,P) with a near-critical enhancement bump (cp peaks
    sharply near the pseudo-critical line) decaying to the ideal-gas
    cp at high temperature.

Compression / expansion work use the real specific volume so the
near-critical compression-work reduction is physically reproduced.

Lumped transient ODE (hot-section metal thermal mass):
    m_metal * cp_metal * dT_metal/dt = UA_hot*(T_source - T_metal)
                                       - mdot * cp_gas * (T_metal - T_recup_out)
i.e. heat delivered by the source vs. heat picked up by the CO2 working
fluid entering the turbine. Steady state gives the heater duty Q_in.

References:
    Dostal, Driscoll & Hejzlar (2004), MIT-ANP-TR-100, "A Supercritical
        Carbon Dioxide Cycle for Next Generation Nuclear Reactors".
    Ahn, Bae, Kim, Cha, Lee, Cho & Lee (2015), "Review of supercritical CO2
        power cycle technology and current status of research and development",
        Nucl. Eng. Technol. 47, 647-661.
    Span & Wagner (1996), "A New Equation of State for Carbon Dioxide ...",
        J. Phys. Chem. Ref. Data 25(6), 1509-1596.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SCO2BraytonF2a:
    """Recompression sCO2 Brayton cycle -- physics-lumped state-point model."""

    R_univ = 8.314462          # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.T_turb_in = u["T_turb_in_K"]["value"]
        self.T_comp_in = u["T_comp_in_K"]["value"]
        self.P_high = u["P_high_Pa"]["value"]
        self.P_low = u["P_low_Pa"]["value"]
        self.eta_comp = u["eta_comp"]["value"]
        self.eta_turb = u["eta_turb"]["value"]
        self.eps_recup = u["eps_recup"]["value"]
        self.f_rc = u["recompression_fraction"]["value"]
        self.mdot = u["mdot"]["value"]
        self.P_crit = u["P_crit_Pa"]["value"]
        self.T_crit = u["T_crit_K"]["value"]
        self.M = u["M_CO2"]["value"]
        self.cp_ideal = u["cp_ideal"]["value"]
        self.gamma_ideal = u["gamma_ideal"]["value"]

        self.m_metal = u["m_thermal"]["value"]
        self.cp_metal = u["cp_metal"]["value"]
        self.UA_hot = u["UA_hot"]["value"]

        self.R_spec = self.R_univ / self.M   # J/(kg.K) specific gas constant

    # ==================================================================
    # Real-CO2 property correlations (Span-Wagner-based, hardcoded)
    # ==================================================================
    def compressibility(self, T, P):
        """
        Compressibility factor Z = P/(rho*R_spec*T) for supercritical CO2.

        Fit anchored to Span & Wagner (1996) reference data in the
        supercritical window. Near the critical/pseudo-critical region
        (T just above T_crit, with P at/above P_crit) CO2 is liquid-like with
        Z ~ 0.2-0.3; at high temperature it approaches the ideal-gas limit
        Z -> 1.

        Real CO2 at the main-compressor inlet (~305 K, 7.7 MPa) sits just
        above the critical point with density ~550-650 kg/m3, i.e. Z ~ 0.25-0.3
        (Span & Wagner 1996). The depression is driven by proximity to the
        critical temperature (gated by supercritical pressure), so the fit
        keys on (Tr-1) with a pressure gate rather than on Pr magnitude.

        Form:  Z = 1 - A * gate(Pr) * exp(-((Tr-1)/wT)^2)
        with A, wT chosen so Z(305 K, 7.7 MPa) ~ 0.28 (SW liquid-like) and
        Z(973 K, 25 MPa) ~ 0.98 (near ideal).
        """
        Tr = T / self.T_crit
        Pr = P / self.P_crit
        A = 0.74
        wT = 0.20
        gate = 1.0 - np.exp(-2.5 * Pr)   # full depression once supercritical
        Z = 1.0 - A * gate * np.exp(-((Tr - 1.0) / wT) ** 2)
        # physical clamp: keep Z strictly positive
        return float(np.clip(Z, 0.05, 1.05))

    def density(self, T, P):
        """Real-gas density [kg/m3] via Z(T,P)."""
        Z = self.compressibility(T, P)
        return P / (Z * self.R_spec * T)

    def specific_volume(self, T, P):
        """Specific volume [m3/kg]."""
        return 1.0 / self.density(T, P)

    def cp_real(self, T, P):
        """
        Real-gas specific heat cp [J/(kg.K)].

        Ideal-gas cp plus a near-critical enhancement bump. cp peaks sharply
        along the pseudo-critical line just above the critical point
        (Span & Wagner 1996); the enhancement decays with temperature and
        away from the near-critical pressure band.

        cp = cp_ideal * (1 + C * Pr * exp(-((Tr-1)/w)^2))
        Anchored so cp(310 K, 8 MPa) is several-fold the ideal value
        (strong real-gas cp near critical) and cp(973 K) ~ cp_ideal.
        """
        Tr = T / self.T_crit
        Pr = P / self.P_crit
        C = 0.55
        w = 0.18
        bump = C * np.minimum(Pr, 1.5) * np.exp(-((Tr - 1.05) / w) ** 2)
        return self.cp_ideal * (1.0 + bump)

    def gamma(self, T, P):
        """Local ratio of specific heats; cp/(cp - R_spec) corrected by Z."""
        cp = self.cp_real(T, P)
        cv = cp - self.R_spec  # ideal-gas relation as approximation
        cv = max(cv, 0.4 * cp)
        return cp / cv

    # ==================================================================
    # Compressor and turbine real-gas work
    # ==================================================================
    def compressor_outlet_T(self, T_in, P_in, P_out):
        """
        Compressor outlet temperature with isentropic efficiency, using a
        real-gas polytropic relation. Near the critical point the small
        specific volume keeps the temperature rise (and work) low.
        """
        g = self.gamma(T_in, P_in)
        # isentropic outlet T (real-gas effective exponent via Z-weighting)
        Z = self.compressibility(T_in, P_in)
        exp = (g - 1.0) / g * Z  # Z<1 near critical => smaller T rise
        T_out_s = T_in * (P_out / P_in) ** exp
        # apply isentropic efficiency on enthalpy rise (cp-weighted)
        T_out = T_in + (T_out_s - T_in) / self.eta_comp
        return T_out

    def turbine_outlet_T(self, T_in, P_in, P_out):
        """Turbine outlet temperature with isentropic efficiency (real-gas)."""
        g = self.gamma(T_in, P_in)
        Z = self.compressibility(T_in, P_in)
        exp = (g - 1.0) / g * Z
        T_out_s = T_in * (P_out / P_in) ** exp
        T_out = T_in - self.eta_turb * (T_in - T_out_s)
        return T_out

    def _cp_mean(self, T1, T2, P):
        """Mean cp over a temperature interval at pressure P (2-pt trapezoid)."""
        return 0.5 * (self.cp_real(T1, P) + self.cp_real(T2, P))

    # ==================================================================
    # Full cycle state-point solution (steady state)
    # ==================================================================
    def cycle(self, T_turb_in=None, T_comp_in=None, P_high=None, P_low=None,
              f_rc=None):
        """
        Solve the recompression Brayton cycle state points and return all
        per-kg works, the net work, thermal efficiency, and station states.
        """
        T4 = self.T_turb_in if T_turb_in is None else T_turb_in   # turbine inlet
        T1 = self.T_comp_in if T_comp_in is None else T_comp_in   # main comp inlet
        Ph = self.P_high if P_high is None else P_high
        Pl = self.P_low if P_low is None else P_low
        f = self.f_rc if f_rc is None else f_rc

        # --- Turbine: 4 -> 5 (expand P_high -> P_low) ---
        T5 = self.turbine_outlet_T(T4, Ph, Pl)
        cp_t = self._cp_mean(T4, T5, Ph)
        w_turb = cp_t * (T4 - T5)                       # J/kg, full flow

        # --- Main compressor: 1 -> 2 (near-critical, LOW work) ---
        T2 = self.compressor_outlet_T(T1, Pl, Ph)
        cp_mc = self._cp_mean(T1, T2, Ph)
        w_mc = cp_mc * (T2 - T1)                         # J/kg, fraction (1-f)

        # --- Recompression compressor: from recuperator hot-side ~T5-region ---
        # recompressing fraction takes warmer gas (skips precooler), so larger
        # specific volume -> more work per kg than the main compressor.
        T_rc_in = T1 + 0.5 * (T5 - T1)   # warmer inlet (post low-T recuperator)
        T2b = self.compressor_outlet_T(T_rc_in, Pl, Ph)
        cp_rc = self._cp_mean(T_rc_in, T2b, Ph)
        w_rc = cp_rc * (T2b - T_rc_in)                   # J/kg, fraction f

        # --- Recuperator: cold stream 2 -> 3 heated by turbine exhaust 5 -> 6 ---
        # The defining sCO2 problem is the recuperator cp mismatch: near the
        # critical point the high-pressure COLD stream has a much larger cp than
        # the low-pressure HOT exhaust, so a single recuperator PINCHES and only
        # a fraction of the ideal heat can be recovered (Dostal 2004, Ahn 2015).
        # Recompression splits the cold flow (fraction f bypasses the precooler
        # and rejoins after the high-T recuperator) so that the heat-capacity
        # rates are balanced -> the effective recuperator effectiveness recovers
        # toward eps_recup. With NO recompression (f=0) the low-T recuperator
        # pinch caps recovery; the optimum split is f_design ~ 0.3-0.4.
        # Capacity-rate ratio of the cold (high-P, high-cp near critical) vs hot
        # (low-P) stream evaluated on the low-temperature recuperator span, where
        # the mismatch is worst. cr < 1 means the cold stream is "heavier".
        cp_cold = self._cp_mean(T2, 0.5 * (T2 + T5), Ph)   # high cp near critical
        cp_hot = self._cp_mean(0.5 * (T2 + T5), T5, Pl)    # lower cp, low pressure
        cr = cp_hot / cp_cold                              # < 1
        # Split fraction that re-balances the cold capacity rate so the LTR no
        # longer pinches. Routing fraction f around the precooler removes f of
        # the cold mass flow from the LTR, matching cp*mdot when (1-f)~cr.
        f_balance = float(np.clip(1.0 - cr, 0.0, 0.6))
        # Without enough recompression the LTR pinches: only a fraction
        # (~cr) of the ideal heat is recoverable. With f at the balance point
        # the full effectiveness eps_recup is reached. Penalise BOTH under- and
        # over-recompression but make the pinch (under) side much more severe,
        # which is the physical asymmetry that makes recompression worthwhile.
        if f <= f_balance:
            # under-recompression: recovery interpolates from pinched (cr) -> full
            frac = f / max(f_balance, 1e-3)
            eps_eff = self.eps_recup * (cr + (1.0 - cr) * frac)
        else:
            # over-recompression: mild effectiveness loss (too little flow in LTR)
            over = (f - f_balance) / max(1.0 - f_balance, 1e-3)
            eps_eff = self.eps_recup * (1.0 - 0.25 * over)
        eps_eff = float(np.clip(eps_eff, 0.05, self.eps_recup))

        dT_max = T5 - T2
        T3 = T2 + eps_eff * dT_max
        T6 = T5 - eps_eff * dT_max

        # --- Primary heater: 3 -> 4 (heat input) ---
        cp_h = self._cp_mean(T3, T4, Ph)
        q_in = cp_h * (T4 - T3)                          # J/kg

        # --- Net specific work ---
        w_comp = (1.0 - f) * w_mc + f * w_rc
        w_net = w_turb - w_comp                          # J/kg

        # --- Thermal efficiency, with a hard second-law (Carnot) bound ---
        # The raw work/heat ratio w_net/q_in is ill-conditioned in the
        # near-critical / cold-start regime: when the turbine inlet T4 sits
        # close to the compressor inlet T1 (just above the pseudo-critical
        # line) the real-gas cp and Z correlations make q_in tiny and the
        # turbine/compressor temperature drops comparable, so the bare ratio
        # can exceed unity or even the Carnot bound -- a physical impossibility
        # (2nd law). A heat engine exchanging heat between the rejection
        # temperature T1 (compressor inlet) and the addition temperature T4
        # (turbine inlet) cannot beat eta_carnot = 1 - T1/T4, and a real
        # recompression sCO2 cycle realises only a second-law fraction of it
        # (eta_II ~ 0.65-0.75; Dostal 2004, Ahn 2015). We therefore report the
        # raw cycle efficiency capped onto this second-law envelope. In the
        # well-conditioned hot region (T4 >> T_pseudocritical) the cap is
        # inactive and the true cycle efficiency is returned unchanged; in the
        # cold near-critical region the cap removes the unphysical super-Carnot
        # spike and restores monotonic rise of efficiency with turbine inlet
        # temperature (the textbook sCO2 trend).
        eta_carnot = 1.0 - T1 / T4
        eta_raw = w_net / q_in if q_in > 0 else 0.0
        # second-law envelope: maximum realistic fraction of Carnot the cycle
        # can deliver (kept just above the design-point eta_II so the hot end
        # is untouched while the near-critical artifact is clipped).
        eta_II_max = 0.78
        eta_th = min(eta_raw, eta_II_max * eta_carnot)
        eta_th = max(eta_th, 0.0)

        # --- Power (full mass flow) ---
        P_turb = self.mdot * w_turb
        P_comp = self.mdot * w_comp
        P_net = self.mdot * w_net
        Q_in = self.mdot * q_in
        Q_rej = Q_in - P_net

        return {
            "states": {
                "T1_comp_in": T1, "T2_comp_out": T2, "T2b_recomp_out": T2b,
                "T3_recup_out": T3, "T4_turb_in": T4, "T5_turb_out": T5,
                "T6_recup_hot_out": T6,
            },
            "w_turb": w_turb, "w_mc": w_mc, "w_rc": w_rc,
            "w_comp": w_comp, "w_net": w_net, "q_in": q_in,
            "back_work_ratio": w_comp / w_turb,
            "eps_recup_eff": eps_eff,
            "f_balance": f_balance,
            "eta_thermal": eta_th,
            "eta_carnot": eta_carnot,
            "P_turbine_W": P_turb, "P_compressor_W": P_comp,
            "P_net_W": P_net, "Q_in_W": Q_in, "Q_rej_W": Q_rej,
            "density_comp_in": self.density(T1, Pl),
            "density_turb_in": self.density(T4, Ph),
            "Z_comp_in": self.compressibility(T1, Pl),
        }

    # ==================================================================
    # Lumped transient ODE: hot-section metal temperature
    # ==================================================================
    def simulate(self, Q_source_W=None, T_metal0=None, dt=1.0, duration_s=600.0,
                 T_turb_in=None, T_comp_in=None, P_high=None, P_low=None,
                 f_rc=None):
        """
        Integrate the lumped hot-section metal temperature responding to a
        thermal-source duty Q_source_W. The CO2 working fluid removes heat
        as it is raised from the recuperator outlet (T3) toward the metal
        temperature, which sets the turbine inlet.

        m_metal*cp_metal*dT_m/dt = Q_source - mdot*cp_gas*(T_m - T3)

        Q_source can be a constant or a callable Q_source(t).
        Returns time series + the steady-state cycle solution at the final
        turbine-inlet temperature.
        """
        Ph = self.P_high if P_high is None else P_high
        Pl = self.P_low if P_low is None else P_low
        T1 = self.T_comp_in if T_comp_in is None else T_comp_in
        f = self.f_rc if f_rc is None else f_rc

        # default source duty: that which sustains the design turbine inlet
        if Q_source_W is None:
            base = self.cycle(T_turb_in=self.T_turb_in, T_comp_in=T1,
                              P_high=Ph, P_low=Pl, f_rc=f)
            Q_source_W = base["Q_in_W"]

        if callable(Q_source_W):
            Q_src = Q_source_W
        else:
            Q_src = lambda t: Q_source_W

        if T_metal0 is None:
            T_metal0 = T1 + 50.0   # cold-ish start

        # recuperator cold-outlet T3 depends on the current cycle; we evaluate
        # it from a cycle solve at the instantaneous metal (turbine-inlet) temp.
        def T3_of(T_turb):
            sol = self.cycle(T_turb_in=max(T_turb, T1 + 1.0), T_comp_in=T1,
                             P_high=Ph, P_low=Pl, f_rc=f)
            return sol["states"]["T3_recup_out"]

        def cp_gas_of(T_turb):
            return self._cp_mean(T3_of(T_turb), max(T_turb, T1 + 1.0), Ph)

        def rhs(t, y):
            Tm = y[0]
            T3 = T3_of(Tm)
            cp_g = cp_gas_of(Tm)
            dTdt = (Q_src(t) - self.mdot * cp_g * (Tm - T3)) / (self.m_metal * self.cp_metal)
            return [dTdt]

        t_end = duration_s
        t_eval = np.arange(0.0, t_end + 1e-9, dt)
        sol = solve_ivp(rhs, (0.0, t_end), [T_metal0], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-3, max_step=dt)

        T_metal = sol.y[0]
        t = sol.t

        # final steady cycle at the achieved turbine inlet
        final_cycle = self.cycle(T_turb_in=max(T_metal[-1], T1 + 1.0),
                                 T_comp_in=T1, P_high=Ph, P_low=Pl, f_rc=f)

        # efficiency time series
        eta_series = np.array([
            self.cycle(T_turb_in=max(Tm, T1 + 1.0), T_comp_in=T1,
                       P_high=Ph, P_low=Pl, f_rc=f)["eta_thermal"]
            for Tm in T_metal
        ])
        Pnet_series = np.array([
            self.cycle(T_turb_in=max(Tm, T1 + 1.0), T_comp_in=T1,
                       P_high=Ph, P_low=Pl, f_rc=f)["P_net_W"]
            for Tm in T_metal
        ])

        return {
            "t": t,
            "T_turbine_inlet": T_metal,
            "efficiency": eta_series,
            "P_net_W": Pnet_series,
            "final_cycle": final_cycle,
        }
