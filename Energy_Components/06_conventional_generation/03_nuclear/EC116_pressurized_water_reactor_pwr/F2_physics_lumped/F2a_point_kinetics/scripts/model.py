"""
EC116 -- Pressurized Water Reactor (PWR) -- F2a Point Kinetics Model

Six-group delayed neutron point kinetics equations coupled with
lumped fuel and moderator thermal-hydraulic feedback.

THIS IS A STIFF ODE SYSTEM -- uses Radau (implicit Runge-Kutta) or BDF solver.

Neutron kinetics:
    dn/dt = (rho - beta_total) / Lambda * n + sum_i(lambda_i * C_i)
    dC_i/dt = beta_i / Lambda * n - lambda_i * C_i   (i = 1..6)

Temperature feedback:
    rho = rho_ext + alpha_f * (T_f - T_f0) + alpha_m * (T_m - T_m0)

Thermal-hydraulics (lumped):
    dT_f/dt = (P - hA_fg * (T_f - T_m)) / (m_f * cp_f)
    dT_m/dt = (hA_fg * (T_f - T_m) - m_dot * cp * (T_m - T_in)) / (m_m * cp_m)

where:
    P = n * P0  (thermal power proportional to neutron population)
    n is normalized to initial condition (n0 = 1.0 at rated power)

State vector (9 states):
    [n, C1, C2, C3, C4, C5, C6, T_f, T_m]

Reference:
    Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
    Stacey, W.M. (2007). Nuclear Reactor Physics, 2nd ed. Wiley-VCH.
    Keepin, G.R. (1965). Physics of Nuclear Kinetics. Addison-Wesley.
"""

import numpy as np
from scipy.integrate import solve_ivp


class PWRPointKineticsF2a:
    """PWR point kinetics with 6-group delayed neutrons and thermal feedback."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_th = u["P_thermal_rated"]["value"]
        self.P_el = u["P_elec_rated"]["value"]
        self.eta_th = u["eta_thermal"]["value"]

        self.Lambda = u["Lambda"]["value"]
        self.beta = np.array(u["beta"]["value"])
        self.lam = np.array(u["lambda_precursor"]["value"])
        self.beta_total = np.sum(self.beta)
        self.n_groups = len(self.beta)

        self.alpha_f = u["alpha_f"]["value"]
        self.alpha_m = u["alpha_m"]["value"]

        self.T_in = u["T_inlet"]["value"]

        self.mf_cpf = u["m_f_cp_f"]["value"]
        self.mm_cpm = u["m_m_cp_m"]["value"]
        self.hA_fg = u["hA_fg"]["value"]
        self.mdot_cp = u["m_dot_cp_c"]["value"]

        self.n0 = u["n0"]["value"]

        # Compute equilibrium temperatures at rated power so that rho = 0
        # at steady state (reference temperatures = equilibrium temperatures)
        P0 = self.n0 * self.P_th
        T_m_eq = self.T_in + P0 / self.mdot_cp
        T_f_eq = T_m_eq + P0 / self.hA_fg
        self.T_f0 = T_f_eq
        self.T_m0 = T_m_eq

    def initial_conditions(self, n0=None):
        """
        Compute equilibrium initial conditions.

        At steady state: dC_i/dt = 0 => C_i = beta_i * n / (lambda_i * Lambda)
        Thermal equilibrium from rated power.
        """
        if n0 is None:
            n0 = self.n0

        # Precursor equilibrium concentrations
        C0 = self.beta * n0 / (self.lam * self.Lambda)

        # Thermal equilibrium: P = n0 * P_th
        P0 = n0 * self.P_th
        # From dT_m/dt = 0: hA*(T_f - T_m) = mdot_cp*(T_m - T_in)
        # From dT_f/dt = 0: P = hA*(T_f - T_m)
        delta_T_fm = P0 / self.hA_fg
        T_m_eq = self.T_in + P0 / self.mdot_cp
        T_f_eq = T_m_eq + delta_T_fm

        state0 = np.zeros(1 + self.n_groups + 2)  # n + 6 precursors + T_f + T_m = 9
        state0[0] = n0
        state0[1:7] = C0
        state0[7] = T_f_eq
        state0[8] = T_m_eq

        return state0

    def reactivity(self, T_f, T_m, rho_ext):
        """
        Total reactivity including temperature feedback.

        rho = rho_ext + alpha_f * (T_f - T_f0) + alpha_m * (T_m - T_m0)
        """
        return rho_ext + self.alpha_f * (T_f - self.T_f0) + self.alpha_m * (T_m - self.T_m0)

    def derivatives(self, t, state, rho_ext_func):
        """
        Point kinetics + thermal ODEs.

        Args:
            t:             time [s]
            state:         [n, C1..C6, T_f, T_m]
            rho_ext_func:  callable(t) -> external reactivity [dk/k]

        Returns:
            d(state)/dt
        """
        n = state[0]
        C = state[1:7]
        T_f = state[7]
        T_m = state[8]

        # External reactivity
        rho_ext = rho_ext_func(t)
        rho = self.reactivity(T_f, T_m, rho_ext)

        # Neutron kinetics
        dn_dt = (rho - self.beta_total) / self.Lambda * n + np.dot(self.lam, C)

        # Precursor equations
        dC_dt = self.beta / self.Lambda * n - self.lam * C

        # Thermal power
        P = max(n, 0.0) * self.P_th

        # Fuel temperature
        dTf_dt = (P - self.hA_fg * (T_f - T_m)) / self.mf_cpf

        # Moderator temperature
        dTm_dt = (self.hA_fg * (T_f - T_m) - self.mdot_cp * (T_m - self.T_in)) / self.mm_cpm

        dstate = np.zeros_like(state)
        dstate[0] = dn_dt
        dstate[1:7] = dC_dt
        dstate[7] = dTf_dt
        dstate[8] = dTm_dt

        return dstate

    def simulate(self, rho_ext, dt, duration_s, x0=None, method="Radau",
                 rtol=1e-8, atol=1e-10):
        """
        Simulate PWR point kinetics transient.

        Args:
            rho_ext:     external reactivity [dk/k] (scalar or callable(t))
            dt:          output time step [s]
            duration_s:  simulation duration [s]
            x0:          initial state (default: equilibrium at n0=1)
            method:      ODE solver ('Radau' or 'BDF' for stiff system)
            rtol:        relative tolerance (default 1e-8)
            atol:        absolute tolerance (default 1e-10)

        Returns:
            dict with time-series: t, n, C (6 groups), T_f, T_m, P_thermal, P_elec, rho
        """
        if x0 is None:
            x0 = self.initial_conditions()

        _rho_ext = rho_ext if callable(rho_ext) else lambda t: rho_ext

        t_span = (0.0, duration_s)
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, state):
            return self.derivatives(t, state, _rho_ext)

        sol = solve_ivp(
            rhs, t_span, x0, t_eval=t_eval,
            method=method, rtol=rtol, atol=atol,
            max_step=dt,  # Prevent solver from stepping over discontinuities
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t = sol.t
        n = sol.y[0]
        C = sol.y[1:7]
        T_f = sol.y[7]
        T_m = sol.y[8]

        P_thermal = np.maximum(n, 0.0) * self.P_th
        P_elec = P_thermal * self.eta_th

        rho_arr = np.array([
            self.reactivity(T_f[i], T_m[i], _rho_ext(t[i]))
            for i in range(len(t))
        ])

        return {
            "t": t,
            "n": n,
            "C": C,
            "T_f": T_f,
            "T_m": T_m,
            "P_thermal_W": P_thermal,
            "P_elec_W": P_elec,
            "rho": rho_arr,
            "rho_ext": np.array([_rho_ext(ti) for ti in t]),
        }

    def step_reactivity_insertion(self, rho_step, dt=0.01, duration_s=100.0,
                                   t_insert=1.0):
        """
        Simulate a step reactivity insertion at t = t_insert.

        Args:
            rho_step:  magnitude of step reactivity [dk/k]
            dt:        output time step
            duration_s: total duration
            t_insert:  time of insertion

        Returns:
            simulation result dict
        """
        def rho_ext(t):
            return rho_step if t >= t_insert else 0.0
        return self.simulate(rho_ext, dt, duration_s)

    def ramp_reactivity_insertion(self, rho_rate, rho_max, dt=0.01,
                                   duration_s=100.0, t_start=1.0):
        """
        Simulate a ramp reactivity insertion starting at t_start.

        Args:
            rho_rate:  rate of reactivity insertion [dk/k/s]
            rho_max:   maximum reactivity [dk/k]
            dt:        output time step
            duration_s: total duration
            t_start:   ramp start time

        Returns:
            simulation result dict
        """
        def rho_ext(t):
            if t < t_start:
                return 0.0
            return min(rho_rate * (t - t_start), rho_max)
        return self.simulate(rho_ext, dt, duration_s)
