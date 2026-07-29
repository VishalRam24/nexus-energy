"""
EC119 -- Molten Salt Reactor (MSR) -- F2a Point Reactor Kinetics

Physics-lumped (0D) model of a liquid-fuel MSR coupling:

  1. Point reactor kinetics with 6 delayed-neutron precursor groups.
  2. The MSR-defining feature -- FLOWING FUEL.  Because the fissile material
     is dissolved in the circulating salt, delayed-neutron precursors are
     PHYSICALLY CARRIED out of the core before they decay.  A precursor born
     in the core may emit its neutron in the external loop (where it is lost
     to the chain) and a fraction returns to the core after one loop transit.
     This drift REDUCES the EFFECTIVE delayed-neutron fraction beta_eff < beta,
     making the reactor more "prompt" and requiring a smaller reactivity to
     reach prompt-critical.  (Haubenreich & Engel 1970, MSRE; ORNL-TM design.)
  3. Strong NEGATIVE temperature/density reactivity feedback of the liquid fuel
     (Doppler + thermal expansion) -- the dominant inherent-safety mechanism.
  4. A lumped two-node core+loop thermal model (energy balance).

Governing equations
--------------------
Flowing-fuel point kinetics (precursor advection model, Lapenta 2001;
Dulla 2002; classic MSRE treatment in Haubenreich & Engel 1970):

    dn/dt   = (rho(t,T) - beta) / Lambda * n
              + sum_i lambda_i * C_i

    dC_i/dt = beta_i / Lambda * n
              - lambda_i * C_i
              - C_i / tau_core                 (precursors swept OUT of core)
              + C_i_return / tau_core          (precursors returning from loop)

The out-of-core decay of precursors during loop transit is captured by the
return fraction  f_ret = exp(-lambda_i * tau_loop):  precursors that survive
the loop transit re-enter the core, the rest decayed (emitting neutrons
uselessly outside the core).  This is the standard lumped-loop reduction of
the full advection problem and yields the well-known loss of effective beta.

Reactivity (pcm -> absolute, 1 pcm = 1e-5):

    rho(t,T) = rho_ext(t)
               + alpha_fuel     * (T_core - T_core0)
               + alpha_graphite * (T_core - T_core0)

Two-node lumped thermal model (energy conservation):

    M_core*cp_core dT_core/dt = P(t)
                                - hA_cl  * (T_core - T_loop)
    M_loop*cp_loop dT_loop/dt = hA_cl  * (T_core - T_loop)
                                - hA_ls * (T_loop - T_sink)

    P(t) = P0 * n / n0      (fission power proportional to neutron population)

References
----------
    Haubenreich, P.N. & Engel, J.R. (1970). "Experience with the Molten-Salt
        Reactor Experiment." Nucl. Appl. Technol. 8(2):118-136.
    Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
        (point kinetics, Ch. 6; reactivity feedback, Ch. 7)
    Keepin, G.R. (1965). Physics of Nuclear Kinetics. Addison-Wesley.
        (6-group U-235 delayed-neutron data)
    MacPherson, H.G. (1985). Nucl. Sci. Eng. 90:374-380.
"""

import numpy as np
from scipy.integrate import solve_ivp

PCM = 1.0e-5  # 1 pcm = 1e-5 absolute reactivity (dk/k)


class MSR_F2a:
    """Molten Salt Reactor -- flowing-fuel point kinetics + lumped thermal."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P0 = float(u["P0_w"]["value"])                       # W
        self.Lambda = float(u["Lambda_s"]["value"])              # s
        self.beta_i = np.asarray(u["beta_i"]["value"], float)    # -
        self.lambda_i = np.asarray(u["lambda_i"]["value"], float)  # 1/s
        self.beta = float(u["beta"]["value"])
        self.n_groups = len(self.beta_i)

        self.tau_core = float(u["tau_core_s"]["value"])          # s
        self.tau_loop = float(u["tau_loop_s"]["value"])          # s

        self.alpha_fuel = float(u["alpha_fuel_pcm_per_K"]["value"]) * PCM     # 1/K
        self.alpha_graph = float(u["alpha_graphite_pcm_per_K"]["value"]) * PCM  # 1/K

        self.M_core = float(u["M_fuel_kg"]["value"])
        self.cp_core = float(u["cp_fuel_J_per_kgK"]["value"])
        self.M_loop = float(u["M_loop_kg"]["value"])
        self.cp_loop = float(u["cp_loop_J_per_kgK"]["value"])
        self.hA_cl = float(u["hA_core_loop_W_per_K"]["value"])
        self.hA_ls = float(u["hA_loop_sink_W_per_K"]["value"])
        self.T_sink = float(u["T_sink_K"]["value"])

        self.T_core0 = float(u["T_core0_K"]["value"])
        self.T_loop0 = float(u["T_loop0_K"]["value"])

    # ------------------------------------------------------------------
    # Effective delayed-neutron fraction with flowing fuel
    # ------------------------------------------------------------------
    def beta_eff(self, flow_fraction=1.0):
        """
        Effective delayed-neutron fraction beta_eff(flow).

        For a flowing fuel, a fraction of each precursor group is swept out of
        the core and decays elsewhere, contributing nothing to the in-core
        chain.  Using the lumped circulating-fuel reduction, the effective
        per-group delayed fraction is scaled by the fraction of precursors
        that decay IN the core during one core transit:

            f_i = lambda_i*tau_core / (lambda_i*tau_core + 1 - exp(-lambda_i*tau_loop))

        With zero flow (flow_fraction -> 0, tau_core/tau_loop -> inf) f_i -> 1
        and beta_eff -> beta.  With fast flow precursors are flushed out and
        beta_eff < beta.  (Haubenreich & Engel 1970; Duderstadt 1976.)
        """
        if flow_fraction <= 1e-9:
            return self.beta  # stationary fuel: no loss
        tau_c = self.tau_core / flow_fraction
        tau_l = self.tau_loop / flow_fraction
        lt_c = self.lambda_i * tau_c
        loss = 1.0 - np.exp(-self.lambda_i * tau_l)   # decayed-out-of-core
        f_i = lt_c / (lt_c + loss)
        return float(np.sum(self.beta_i * f_i))

    def _group_scaling(self, flow_fraction):
        """Per-group in-core retention factor f_i (vector), see beta_eff()."""
        if flow_fraction <= 1e-9:
            return np.ones(self.n_groups)
        tau_c = self.tau_core / flow_fraction
        tau_l = self.tau_loop / flow_fraction
        lt_c = self.lambda_i * tau_c
        loss = 1.0 - np.exp(-self.lambda_i * tau_l)
        return lt_c / (lt_c + loss)

    def _drift(self, flow_fraction):
        """
        Per-group effective in-core precursor LOSS rate [1/s] from circulation.

        Defined so that the steady-state delayed source reproduces beta_eff
        exactly:  C_i,ss = (beta_i/Lambda) n /(lambda_i + drift_i) gives
        sum lambda_i C_i,ss = (beta_eff/Lambda) n  iff
            lambda_i/(lambda_i + drift_i) = f_i   ->   drift_i = lambda_i (1-f_i)/f_i.
        This keeps kinetics, beta_eff, and the circulation reactivity penalty
        mutually consistent (no spurious initial transient).
        """
        if flow_fraction <= 1e-9:
            return np.zeros(self.n_groups)
        f_i = self._group_scaling(flow_fraction)
        return self.lambda_i * (1.0 - f_i) / f_i

    # ------------------------------------------------------------------
    # Circulation reactivity penalty (critical-holding bias)
    # ------------------------------------------------------------------
    def rho_circulation(self, flow_fraction=1.0):
        """
        Static reactivity that the operator must hold to keep a FLOWING-fuel
        core critical, rho_circ = beta - beta_eff(flow) (absolute dk/k).

        Because circulating precursors emit delayed neutrons partly outside
        the core, the in-core delayed source is reduced; criticality requires
        compensating control reactivity equal to the lost delayed fraction.
        MSRE measured ~0.2-0.3 %dk/k from fuel circulation (Haubenreich &
        Engel 1970).  This bias makes the rho_ext=0, T=T0 state a true
        equilibrium, so the model is self-consistent (no spurious drift).
        """
        return self.beta - self.beta_eff(flow_fraction)

    # ------------------------------------------------------------------
    # Reactivity with temperature feedback
    # ------------------------------------------------------------------
    def reactivity(self, T_core, rho_ext, flow_fraction=1.0):
        """Total reactivity (absolute dk/k): circulation-hold bias + external
        insertion + temperature feedback."""
        dT = T_core - self.T_core0
        return (self.rho_circulation(flow_fraction) + rho_ext
                + (self.alpha_fuel + self.alpha_graph) * dT)

    # ------------------------------------------------------------------
    # Initial equilibrium precursor concentrations (steady state)
    # ------------------------------------------------------------------
    def equilibrium_precursors(self, n0, flow_fraction=1.0):
        """
        Steady-state precursor concentrations C_i for given neutron level n0.

        At steady state dC_i/dt = 0 with the lumped flowing-fuel sink/source,
        the net effective loss rate is (lambda_i + (1-f_i_ret)/tau_core).
        We use the in-core retention scaling f_i so that the production is
        balanced:  C_i = (beta_i/Lambda) n0 / (lambda_i + drift_i).
        """
        drift = self._drift(flow_fraction)
        denom = self.lambda_i + drift
        return (self.beta_i / self.Lambda) * n0 / denom

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, rho_ext_fn, flow_fraction):
        n = y[0]
        C = y[1:1 + self.n_groups]
        T_core = y[1 + self.n_groups]
        T_loop = y[2 + self.n_groups]

        rho_ext = rho_ext_fn(t)
        rho = self.reactivity(T_core, rho_ext, flow_fraction)

        # flowing-fuel precursor drift: net in-core precursor loss to circulation
        drift = self._drift(flow_fraction)

        # point kinetics (n normalised so n0 = 1 at rated)
        dndt = (rho - self.beta) / self.Lambda * n + np.sum(self.lambda_i * C)
        dCdt = (self.beta_i / self.Lambda) * n - (self.lambda_i + drift) * C

        # fission power proportional to neutron population
        P = self.P0 * n

        dTc = (P - self.hA_cl * (T_core - T_loop)) / (self.M_core * self.cp_core)
        dTl = (self.hA_cl * (T_core - T_loop)
               - self.hA_ls * (T_loop - self.T_sink)) / (self.M_loop * self.cp_loop)

        return np.concatenate(([dndt], dCdt, [dTc, dTl]))

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, rho_ext_pcm=0.0, flow_fraction=1.0,
                 dt=0.5, duration_s=200.0,
                 T_core0=None, T_loop0=None, n0=1.0):
        """
        Simulate MSR transient.

        Parameters
        ----------
        rho_ext_pcm : float or callable(t)->float
            External (control) reactivity insertion [pcm].
        flow_fraction : float
            Fuel-salt pump flow as a fraction of nominal (1.0 = rated flow).
            0.0 = stagnant fuel (beta_eff = beta, no precursor loss).
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation time [s].
        T_core0, T_loop0 : float
            Optional initial temperatures [K].
        n0 : float
            Initial normalised neutron population (1.0 = rated).

        Returns
        -------
        dict with arrays: t, power_w, power_fraction, n, T_core_K, T_loop_K,
             reactivity_pcm, and scalars beta_eff, beta_static.
        """
        if callable(rho_ext_pcm):
            rho_fn = lambda t: rho_ext_pcm(t) * PCM
        else:
            rho_fn = lambda t: rho_ext_pcm * PCM

        Tc0 = self.T_core0 if T_core0 is None else T_core0
        Tl0 = self.T_loop0 if T_loop0 is None else T_loop0

        C0 = self.equilibrium_precursors(n0, flow_fraction)
        y0 = np.concatenate(([n0], C0, [Tc0, Tl0]))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(rho_fn, flow_fraction),
            method="LSODA",          # stiff: prompt-jump + slow thermal
            rtol=1e-8, atol=1e-12, max_step=dt,
        )

        t = sol.t
        n = sol.y[0]
        T_core = sol.y[1 + self.n_groups]
        T_loop = sol.y[2 + self.n_groups]

        rho_pcm = np.array([
            self.reactivity(T_core[k], rho_fn(t[k]), flow_fraction) / PCM
            for k in range(len(t))
        ])

        return {
            "t": t,
            "n": n,
            "power_w": self.P0 * n,
            "power_fraction": n / n0,
            "T_core_K": T_core,
            "T_loop_K": T_loop,
            "reactivity_pcm": rho_pcm,
            "beta_eff": self.beta_eff(flow_fraction),
            "beta_static": self.beta,
        }
