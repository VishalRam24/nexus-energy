"""
EC118 -- Small Modular Reactor (SMR) -- F2a Point Kinetics + Lumped Thermal

Physics-lumped dynamic model of an integral PWR-type SMR (NuScale VOYGR-class).
Six-group delayed-neutron point reactor kinetics coupled to a lumped fuel +
coolant thermal model, with Doppler (fuel) and moderator-temperature reactivity
feedback, natural-circulation coolant flow, and a load-following controller.

This is a STIFF ODE system (prompt generation time Lambda ~ 2e-5 s vs thermal
time constants of minutes) -> integrated with an implicit solver (Radau/BDF).

------------------------------------------------------------------------------
Neutron kinetics (point reactor, 6 delayed groups) -- Duderstadt & Hamilton (1976):

    dn/dt      = (rho - beta_tot)/Lambda * n + sum_i lambda_i * C_i
    dC_i/dt    = beta_i/Lambda * n - lambda_i * C_i        (i = 1..6)

Reactivity with temperature feedback:

    rho = rho_ext + alpha_f*(T_f - T_f0) + alpha_m*(T_m - T_m0)

    alpha_f < 0 : Doppler broadening of U-238 resonances (prompt, stabilizing)
    alpha_m < 0 : moderator-density / spectral feedback (delayed, stabilizing)

Lumped thermal-hydraulics (fuel node + coolant node):

    m_f cp_f dT_f/dt = P - hA*(T_f - T_m)
    m_m cp_m dT_m/dt = hA*(T_f - T_m) - W(P)*(T_m - T_in)

    P    = n * P0                          (fission power, n normalized to rated)
    W(P) = mdot*cp * flow_fraction(P)      (natural circulation: mdot ~ P^0.5)

State vector (9 states): [n, C1..C6, T_f, T_m]

The reference temperatures T_f0, T_m0 are chosen as the rated-power equilibrium
so that the feedback reactivity is exactly zero at full power (consistent
linearization point, Stacey 2007).

------------------------------------------------------------------------------
SMR-specific physics vs large PWR:
  * Natural-circulation primary flow (no pumps): W scales with sqrt(power),
    Boussinesq buoyancy balance (Todreas & Kazimi 2012). This couples flow to
    power and is what makes the integral SMR strongly self-regulating.
  * Large primary coolant inventory relative to core power -> long thermal time
    constant (~8 min) -> slow, smooth load-following.
  * Large-magnitude negative MTC -> deep load-following (20-100%) via coolant
    temperature feedback alone ("reactor-follows-turbine").

References:
    Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
    Keepin, G.R. (1965). Physics of Nuclear Kinetics. Addison-Wesley.
    Stacey, W.M. (2007). Nuclear Reactor Physics, 2nd ed. Wiley-VCH.
    Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems, 2nd ed. CRC Press.
    IAEA (2022). Advances in Small Modular Reactor Technology Developments.
    NuScale VOYGR Design Certification Application (2020).
    Ingersoll, D.T. & Carelli, M.D. (2020). Handbook of Small Modular Reactors.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SMRPointKineticsThermalF2a:
    """SMR point kinetics (6-group) + lumped fuel/coolant thermal with feedback."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_th = u["P_thermal_rated"]["value"]      # W
        self.P_el = u["P_elec_rated"]["value"]         # W
        self.eta_th = u["eta_thermal"]["value"]

        self.Lambda = u["Lambda"]["value"]             # s
        self.beta = np.array(u["beta"]["value"], dtype=float)
        self.lam = np.array(u["lambda_precursor"]["value"], dtype=float)
        self.beta_total = float(np.sum(self.beta))
        self.n_groups = len(self.beta)

        self.alpha_f = u["alpha_f"]["value"]           # dk/k per K (<0)
        self.alpha_m = u["alpha_m"]["value"]           # dk/k per K (<0)

        self.T_in = u["T_inlet"]["value"]              # K

        self.mf_cpf = u["m_f_cp_f"]["value"]           # J/K
        self.mm_cpm = u["m_m_cp_m"]["value"]           # J/K
        self.hA_fg = u["hA_fg"]["value"]               # W/K
        self.mdot_cp = u["m_dot_cp_c"]["value"]        # W/K (full-power rated)

        self.n0 = u["n0"]["value"]

        self.nat_circ = bool(u.get("natural_circulation", {}).get("value", True))
        self.nat_exp = u.get("nat_circ_exponent", {}).get("value", 0.5)
        self.nat_min = u.get("nat_circ_min_frac", {}).get("value", 0.25)

        # Rated-power equilibrium temperatures -> reference (zero feedback at full power)
        P0 = self.n0 * self.P_th
        W0 = self._flow_capacity(self.n0)
        T_m_eq = self.T_in + P0 / W0
        T_f_eq = T_m_eq + P0 / self.hA_fg
        self.T_f0 = T_f_eq
        self.T_m0 = T_m_eq

    # ------------------------------------------------------------------
    # Natural-circulation coolant flow capacity W = mdot*cp [W/K]
    # ------------------------------------------------------------------
    def _flow_capacity(self, n):
        """Effective coolant heat-removal capacity mdot*cp at power level n.

        Natural circulation: buoyancy head ~ rho*g*beta_th*dT*H balances friction
        ~ mdot^2, and core dT ~ P/mdot, giving mdot ~ P^(1/3..1/2). We use the
        common engineering exponent 0.5 (Todreas & Kazimi 2012) with a floor.
        """
        if not self.nat_circ:
            return self.mdot_cp
        frac = max(self.nat_min, abs(n) ** self.nat_exp)
        return self.mdot_cp * frac

    # ------------------------------------------------------------------
    # Equilibrium initial conditions
    # ------------------------------------------------------------------
    def initial_conditions(self, n0=None):
        """Steady-state initial state at power level n0.

        dC_i/dt = 0 => C_i = beta_i * n0 / (lambda_i * Lambda)
        Thermal equilibrium consistent with natural-circulation flow at n0.
        """
        if n0 is None:
            n0 = self.n0
        C0 = self.beta * n0 / (self.lam * self.Lambda)
        P0 = n0 * self.P_th
        W0 = self._flow_capacity(n0)
        T_m_eq = self.T_in + P0 / W0
        T_f_eq = T_m_eq + P0 / self.hA_fg

        state0 = np.zeros(1 + self.n_groups + 2)
        state0[0] = n0
        state0[1:1 + self.n_groups] = C0
        state0[-2] = T_f_eq
        state0[-1] = T_m_eq
        return state0

    # ------------------------------------------------------------------
    # Reactivity with temperature feedback
    # ------------------------------------------------------------------
    def reactivity(self, T_f, T_m, rho_ext):
        """Total reactivity [dk/k] = external + Doppler + moderator feedback."""
        return (rho_ext
                + self.alpha_f * (T_f - self.T_f0)
                + self.alpha_m * (T_m - self.T_m0))

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def derivatives(self, t, state, rho_ext_func):
        n = state[0]
        C = state[1:1 + self.n_groups]
        T_f = state[-2]
        T_m = state[-1]

        rho_ext = rho_ext_func(t)
        rho = self.reactivity(T_f, T_m, rho_ext)

        dn_dt = (rho - self.beta_total) / self.Lambda * n + np.dot(self.lam, C)
        dC_dt = self.beta / self.Lambda * n - self.lam * C

        P = max(n, 0.0) * self.P_th
        W = self._flow_capacity(n)

        dTf_dt = (P - self.hA_fg * (T_f - T_m)) / self.mf_cpf
        dTm_dt = (self.hA_fg * (T_f - T_m) - W * (T_m - self.T_in)) / self.mm_cpm

        dstate = np.empty_like(state)
        dstate[0] = dn_dt
        dstate[1:1 + self.n_groups] = dC_dt
        dstate[-2] = dTf_dt
        dstate[-1] = dTm_dt
        return dstate

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------
    def simulate(self, rho_ext, dt, duration_s, x0=None, method="BDF",
                 rtol=1e-7, atol=1e-9):
        """Integrate the coupled kinetics + thermal ODEs.

        Args:
            rho_ext:    external reactivity [dk/k], scalar or callable(t)
            dt:         output time step [s]
            duration_s: total duration [s]
            x0:         initial state (default rated equilibrium)
            method:     stiff solver ('BDF' or 'Radau')

        Returns dict of time series:
            t, n, C(6xN), T_f, T_m, P_thermal_W, P_elec_W, rho, rho_ext,
            flow_fraction, power_fraction
        """
        if x0 is None:
            x0 = self.initial_conditions()
        _rho = rho_ext if callable(rho_ext) else (lambda t: rho_ext)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            lambda t, y: self.derivatives(t, y, _rho),
            (0.0, duration_s), x0, t_eval=t_eval,
            method=method, rtol=rtol, atol=atol, max_step=dt,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t = sol.t
        n = sol.y[0]
        C = sol.y[1:1 + self.n_groups]
        T_f = sol.y[-2]
        T_m = sol.y[-1]

        P_thermal = np.maximum(n, 0.0) * self.P_th
        P_elec = P_thermal * self.eta_th
        rho_arr = np.array([self.reactivity(T_f[i], T_m[i], _rho(t[i]))
                            for i in range(len(t))])
        flow_frac = np.array([self._flow_capacity(n[i]) / self.mdot_cp
                              for i in range(len(t))])

        return {
            "t": t,
            "n": n,
            "C": C,
            "T_f": T_f,
            "T_m": T_m,
            "P_thermal_W": P_thermal,
            "P_elec_W": P_elec,
            "rho": rho_arr,
            "rho_ext": np.array([_rho(ti) for ti in t]),
            "flow_fraction": flow_frac,
            "power_fraction": np.maximum(n, 0.0),
        }

    # ------------------------------------------------------------------
    # Transients
    # ------------------------------------------------------------------
    def step_reactivity_insertion(self, rho_step, dt=0.01, duration_s=100.0,
                                  t_insert=1.0, **kw):
        """Step external reactivity insertion at t_insert."""
        def rho_ext(t):
            return rho_step if t >= t_insert else 0.0
        return self.simulate(rho_ext, dt, duration_s, **kw)

    def equilibrium_power_fraction(self, rho_ext):
        """Steady neutron power fraction reached for a constant external rho.

        At steady state n=const and feedback exactly cancels rho_ext:
            rho_ext + alpha_f*(T_f-T_f0) + alpha_m*(T_m-T_m0) = 0,
        with equilibrium temperatures a function of power. Solved by a short
        fixed-point on the lumped thermal balance.
        """
        nf = self.n0
        for _ in range(200):
            P = nf * self.P_th
            W = self._flow_capacity(nf)
            T_m = self.T_in + P / W
            T_f = T_m + P / self.hA_fg
            fb = self.alpha_f * (T_f - self.T_f0) + self.alpha_m * (T_m - self.T_m0)
            # Want rho_ext + fb(n) = 0. fb decreases (more negative) as n rises.
            resid = rho_ext + fb
            # Newton-ish update: d(fb)/dn approximated numerically
            dn = 1e-4
            P2 = (nf + dn) * self.P_th
            W2 = self._flow_capacity(nf + dn)
            T_m2 = self.T_in + P2 / W2
            T_f2 = T_m2 + P2 / self.hA_fg
            fb2 = self.alpha_f * (T_f2 - self.T_f0) + self.alpha_m * (T_m2 - self.T_m0)
            dfb = (fb2 - fb) / dn
            if abs(dfb) < 1e-12:
                break
            nf_new = nf - resid / dfb
            nf_new = min(max(nf_new, 1e-4), 2.0)
            if abs(nf_new - nf) < 1e-9:
                nf = nf_new
                break
            nf = nf_new
        return nf

    def required_external_reactivity(self, power_fraction):
        """Inverse of equilibrium_power_fraction: external (rod) reactivity that
        holds the reactor at a given steady power fraction.

        From the steady-state feedback balance rho_ext + fb(n) = 0:
            rho_ext = -[alpha_f*(T_f(n)-T_f0) + alpha_m*(T_m(n)-T_m0)].
        """
        nf = float(power_fraction)
        P = nf * self.P_th
        W = self._flow_capacity(nf)
        T_m = self.T_in + P / W
        T_f = T_m + P / self.hA_fg
        fb = self.alpha_f * (T_f - self.T_f0) + self.alpha_m * (T_m - self.T_m0)
        return -fb

    def load_follow(self, power_schedule, dt=0.5, duration_s=7200.0,
                    kp=0.02, ki=2e-4, **kw):
        """Load-following ("reactor-follows-demand") via control-rod reactivity.

        The external (control-rod) reactivity is recomputed once per output step
        and held piecewise-constant across the segment, so the stiff implicit
        solver sees a smooth RHS within each segment (robust integration). The
        controller combines a feed-forward term -- the steady rod reactivity
        required for the demanded power (required_external_reactivity) -- with a
        proportional-integral trim on the neutron-power error. Reactivity is
        clamped to a physical control-rod band.

        Args:
            power_schedule: callable(t)->demanded power fraction in [0.2, 1.0]
            kp, ki:         PI trim gains [dk/k per unit power error]
            dt, duration_s: output step / duration [s]

        Returns the simulate() dict plus 'power_demand' and 'rho_ext'.
        """
        rho_band = 0.01  # +/-1000 pcm control-rod authority (deep load-following)
        x0 = kw.pop("x0", None)
        if x0 is None:
            x0 = self.initial_conditions()
        method = kw.get("method", "BDF")
        rtol = kw.get("rtol", 1e-7)
        atol = kw.get("atol", 1e-9)

        t_grid = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_grid = t_grid[t_grid <= duration_s]
        Ns = len(self.beta)

        t_all = [0.0]
        y_all = [x0.copy()]
        rho_all = [0.0]
        integ = 0.0
        y = x0.copy()

        integ_max = rho_band / max(ki, 1e-12)  # anti-windup bound on integral
        for k in range(len(t_grid) - 1):
            t0, t1 = t_grid[k], t_grid[k + 1]
            demand = float(power_schedule(t0))
            n_now = y[0]
            err = demand - n_now
            rho_ff = self.required_external_reactivity(demand)
            rho_unsat = rho_ff + kp * err + ki * integ
            rho = float(np.clip(rho_unsat, -rho_band, rho_band))
            # Conditional integration (anti-windup): only accumulate when not
            # saturated, or when the error would drive the controller back into
            # the linear band. Keeps the integral from latching at a limit.
            if abs(rho_unsat - rho) < 1e-12 or (err * rho_unsat < 0):
                integ += err * (t1 - t0)
                integ = float(np.clip(integ, -integ_max, integ_max))

            sol = solve_ivp(
                lambda t, yy: self.derivatives(t, yy, lambda _t: rho),
                (t0, t1), y, t_eval=[t1],
                method=method, rtol=rtol, atol=atol, max_step=(t1 - t0),
            )
            if not sol.success:
                raise RuntimeError(f"Load-follow solver failed: {sol.message}")
            y = sol.y[:, -1]
            t_all.append(t1)
            y_all.append(y.copy())
            rho_all.append(rho)

        Y = np.array(y_all).T
        t = np.array(t_all)
        n = Y[0]
        C = Y[1:1 + Ns]
        T_f = Y[-2]
        T_m = Y[-1]
        P_thermal = np.maximum(n, 0.0) * self.P_th
        return {
            "t": t,
            "n": n,
            "C": C,
            "T_f": T_f,
            "T_m": T_m,
            "rho_ext": np.array(rho_all),
            "P_thermal_W": P_thermal,
            "P_elec_W": P_thermal * self.eta_th,
            "power_fraction": np.maximum(n, 0.0),
            "power_demand": np.array([power_schedule(ti) for ti in t]),
            "flow_fraction": np.array([self._flow_capacity(n[i]) / self.mdot_cp
                                       for i in range(len(t))]),
        }
