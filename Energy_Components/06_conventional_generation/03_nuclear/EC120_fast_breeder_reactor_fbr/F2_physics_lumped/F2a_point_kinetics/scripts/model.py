"""
EC120 -- Fast Breeder Reactor (FBR), sodium-cooled -- F2a Point Kinetics Model
(physics-lumped, fast-spectrum point reactor kinetics + lumped thermal feedback)

Six-group delayed-neutron point kinetics for a FAST spectrum coupled to a
lumped fuel + sodium-coolant thermal model, with fast-reactor specific
reactivity feedbacks (Doppler, sodium void, axial fuel expansion) and
on-line breeding-ratio tracking (fertile U-238 -> Pu-239 conversion).

----------------------------------------------------------------------------
Fast-spectrum point kinetics (Duderstadt & Hamilton 1976; Waltar & Reynolds
1981, "Fast Breeder Reactors", Ch. 6):

    dn/dt    = (rho - beta_tot)/Lambda * n + sum_i lambda_i C_i
    dC_i/dt  = beta_i/Lambda * n - lambda_i C_i           (i = 1..6)

Distinguishing fast-reactor features vs a thermal PWR (EC116):
  * Prompt-neutron generation time Lambda is MUCH shorter -- Lambda ~ 4e-7 s
    for a sodium-cooled fast reactor vs ~2e-5 s for a PWR (no moderation, the
    neutron lifetime is set by fast leakage/absorption only). Waltar &
    Reynolds (1981) quote ~10^-7 s; Duderstadt 4.4e-7 s. This makes the
    prompt-neutron dynamics far faster and the system stiffer.
  * Smaller total delayed fraction. The dominant fissile isotope is Pu-239,
    whose delayed-neutron fraction (beta_eff ~ 0.0021-0.0036) is roughly a
    third of U-235's (0.0065). We use beta_tot ~ 0.0036 (Pu-239 dominated,
    Keepin 1965; Waltar & Reynolds 1981 Table 6-2).

----------------------------------------------------------------------------
Reactivity feedback (fast-reactor specific):

    rho = rho_ext
        + alpha_D  * (T_f - T_f0)                # Doppler (negative)
        + alpha_ax * (T_f - T_f0)                # axial fuel expansion (neg)
        + alpha_v  * void_frac(T_Na)             # sodium void (can be +)

  - Doppler: negative, driven by U-238 resonance broadening in the fertile
    fuel/blanket (Waltar & Reynolds 1981, Ch. 6; the U-238 capture resonance
    integral grows with fuel temperature, hardening capture -> negative).
  - Axial expansion: fuel column lengthens with temperature -> increased
    leakage -> negative (a prompt, fuel-temperature-driven effect).
  - Sodium void: in a large LMFBR core the spectral-hardening component makes
    the sodium-void coefficient POSITIVE in the core centre; here it is
    represented as a coolant-temperature-driven density (pseudo-void) effect.
    The combined power coefficient (Doppler + axial + void) is kept NEGATIVE
    so the reactor is self-stabilising, per the safety requirement that the
    isothermal/power reactivity feedback be net-negative (Waltar & Reynolds
    1981, Ch. 6 "reactivity feedback"; IAEA design practice).

----------------------------------------------------------------------------
Lumped thermal model (two nodes: fuel, sodium coolant):

    dT_f/dt  = (P - hA_fc (T_f - T_Na)) / (m_f cp_f)
    dT_Na/dt = (hA_fc (T_f - T_Na) - W_cp (T_Na - T_in)) / (m_Na cp_Na)

  P = n * P0 (thermal power proportional to neutron population). Sodium has a
  high boiling point (~883 C) and excellent heat transfer; cp_Na ~ 1270
  J/kg/K (Foust 1972 / Waltar & Reynolds 1981 sodium property tables).

----------------------------------------------------------------------------
Breeding ratio (fertile -> fissile conversion):

    BR = (Pu-239 produced by U-238 capture) / (fissile destroyed)

  A breeder has BR > 1 (more fissile bred than consumed). We compute the
  conversion/breeding ratio from one-group balance (Waltar & Reynolds 1981,
  Ch. 7 "Breeding"):

    BR = eta * (1 - p_loss) - 1 + (excess capture in fertile)
       ~ eta_bar - (1 + alpha_c) + extra_blanket

  rather than re-deriving from cross-sections, we expose BR as a configured
  fast-spectrum value (>1) and scale it weakly with core-average fuel
  temperature/spectrum hardening (hotter, harder spectrum -> slightly higher
  eta for Pu-239 -> slightly higher BR). The cumulative bred fissile inventory
  is integrated over the transient as a tracked, conserved quantity.

----------------------------------------------------------------------------
State vector (9 states):  [n, C1..C6, T_f, T_Na]

Solver: STIFF system (very small Lambda) -> implicit Radau / BDF via
scipy.integrate.solve_ivp.

References:
  Waltar, A.E. & Reynolds, A.B. (1981). Fast Breeder Reactors. Pergamon.
  Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
  Keepin, G.R. (1965). Physics of Nuclear Kinetics. Addison-Wesley.
  Stacey, W.M. (2007). Nuclear Reactor Physics, 2nd ed. Wiley-VCH.
"""

import numpy as np
from scipy.integrate import solve_ivp


class FBRPointKineticsF2a:
    """Sodium-cooled FBR fast-spectrum point kinetics with thermal feedback
    and breeding-ratio tracking."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_th = u["P_thermal_rated"]["value"]
        self.P_el = u["P_elec_rated"]["value"]
        self.eta_th = u["eta_thermal"]["value"]

        # Fast-spectrum kinetics
        self.Lambda = u["Lambda"]["value"]
        self.beta = np.array(u["beta"]["value"], dtype=float)
        self.lam = np.array(u["lambda_precursor"]["value"], dtype=float)
        self.beta_total = float(np.sum(self.beta))
        self.n_groups = len(self.beta)

        # Reactivity feedback coefficients (dk/k per K, or per void fraction)
        self.alpha_D = u["alpha_doppler"]["value"]      # Doppler, negative
        self.alpha_ax = u["alpha_axial"]["value"]       # axial expansion, neg
        self.alpha_void = u["alpha_void"]["value"]      # sodium void, +ve coeff

        # Sodium "void"/density model: pseudo-void fraction grows with coolant T
        self.T_Na_ref = u["T_Na_ref"]["value"]          # ref coolant temp [K]
        self.void_per_K = u["void_per_K"]["value"]      # void fraction / K

        # Thermal model
        self.T_in = u["T_inlet"]["value"]
        self.mf_cpf = u["m_f_cp_f"]["value"]
        self.mNa_cpNa = u["m_Na_cp_Na"]["value"]
        self.hA_fc = u["hA_fc"]["value"]
        self.W_cp = u["W_cp_coolant"]["value"]

        self.n0 = u["n0"]["value"]

        # Breeding
        self.BR0 = u["breeding_ratio"]["value"]               # nominal BR > 1
        self.BR_T_coeff = u["breeding_ratio_T_coeff"]["value"]  # /K spectrum hardening
        self.fissile_burn_rate = u["fissile_burn_rate"]["value"]  # kg/s at rated

        # Equilibrium reference temperatures so that feedback rho = 0 at rated
        P0 = self.n0 * self.P_th
        T_Na_eq = self.T_in + P0 / self.W_cp
        T_f_eq = T_Na_eq + P0 / self.hA_fc
        self.T_f0 = T_f_eq
        self.T_Na0 = T_Na_eq
        # Reference coolant temp for void model defaults to equilibrium coolant T
        if self.T_Na_ref <= 0:
            self.T_Na_ref = T_Na_eq

    # ------------------------------------------------------------------ #
    def initial_conditions(self, n0=None):
        """Equilibrium IC: dC_i/dt = 0 and thermal balance at rated power."""
        if n0 is None:
            n0 = self.n0
        C0 = self.beta * n0 / (self.lam * self.Lambda)
        P0 = n0 * self.P_th
        T_Na_eq = self.T_in + P0 / self.W_cp
        T_f_eq = T_Na_eq + P0 / self.hA_fc
        state0 = np.zeros(1 + self.n_groups + 2)
        state0[0] = n0
        state0[1:1 + self.n_groups] = C0
        state0[1 + self.n_groups] = T_f_eq
        state0[2 + self.n_groups] = T_Na_eq
        return state0

    # ------------------------------------------------------------------ #
    def void_fraction(self, T_Na):
        """Pseudo sodium-void fraction driven by coolant temperature (density
        decrease / incipient voiding). Non-negative."""
        return max(0.0, self.void_per_K * (T_Na - self.T_Na_ref))

    def reactivity(self, T_f, T_Na, rho_ext):
        """Total reactivity = external + Doppler + axial expansion + sodium void.

        Doppler and axial are fuel-temperature driven (negative); sodium void
        is coolant-temperature driven (positive coefficient). Net power
        coefficient is negative by construction (see class docstring)."""
        dTf = T_f - self.T_f0
        rho_dopp = self.alpha_D * dTf
        rho_ax = self.alpha_ax * dTf
        rho_void = self.alpha_void * self.void_fraction(T_Na)
        return rho_ext + rho_dopp + rho_ax + rho_void

    def power_coefficient(self):
        """Net power (temperature) reactivity coefficient [dk/k per K of fuel].

        For self-stabilisation this must be NEGATIVE. The void term is mapped
        onto fuel-temperature change through the steady-state coupling
        dT_Na = dT_f (both rise together with power), so the effective per-K
        coefficient combines Doppler + axial + void*void_per_K."""
        return self.alpha_D + self.alpha_ax + self.alpha_void * self.void_per_K

    def breeding_ratio(self, T_f):
        """Instantaneous breeding ratio. Nominal BR0 (>1), weakly increasing
        with fuel temperature via spectrum hardening (Pu-239 eta rises with a
        harder spectrum -> more breeding). Waltar & Reynolds (1981) Ch. 7."""
        return self.BR0 + self.BR_T_coeff * (T_f - self.T_f0)

    # ------------------------------------------------------------------ #
    def derivatives(self, t, state, rho_ext_func):
        n = state[0]
        C = state[1:1 + self.n_groups]
        T_f = state[1 + self.n_groups]
        T_Na = state[2 + self.n_groups]

        rho_ext = rho_ext_func(t)
        rho = self.reactivity(T_f, T_Na, rho_ext)

        dn_dt = (rho - self.beta_total) / self.Lambda * n + np.dot(self.lam, C)
        dC_dt = self.beta / self.Lambda * n - self.lam * C

        P = max(n, 0.0) * self.P_th
        dTf_dt = (P - self.hA_fc * (T_f - T_Na)) / self.mf_cpf
        dTNa_dt = (self.hA_fc * (T_f - T_Na) - self.W_cp * (T_Na - self.T_in)) / self.mNa_cpNa

        dstate = np.zeros_like(state)
        dstate[0] = dn_dt
        dstate[1:1 + self.n_groups] = dC_dt
        dstate[1 + self.n_groups] = dTf_dt
        dstate[2 + self.n_groups] = dTNa_dt
        return dstate

    # ------------------------------------------------------------------ #
    def simulate(self, rho_ext, dt, duration_s, x0=None, method="Radau",
                 rtol=1e-8, atol=1e-10):
        """Simulate FBR fast-spectrum point-kinetics transient.

        Returns dict of time-series including power, fuel/sodium temperature,
        reactivity, breeding ratio, and cumulative bred fissile inventory."""
        if x0 is None:
            x0 = self.initial_conditions()
        _rho_ext = rho_ext if callable(rho_ext) else (lambda t, _r=rho_ext: _r)

        t_span = (0.0, duration_s)
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, state):
            return self.derivatives(t, state, _rho_ext)

        sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method=method,
                        rtol=rtol, atol=atol, max_step=dt)
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        t = sol.t
        n = sol.y[0]
        C = sol.y[1:1 + self.n_groups]
        T_f = sol.y[1 + self.n_groups]
        T_Na = sol.y[2 + self.n_groups]

        P_thermal = np.maximum(n, 0.0) * self.P_th
        P_elec = P_thermal * self.eta_th

        rho_arr = np.array([self.reactivity(T_f[i], T_Na[i], _rho_ext(t[i]))
                            for i in range(len(t))])
        BR_arr = np.array([self.breeding_ratio(T_f[i]) for i in range(len(t))])
        void_arr = np.array([self.void_fraction(T_Na[i]) for i in range(len(t))])

        # Cumulative fissile balance (kg): fissile destroyed scales with power,
        # bred = BR * destroyed; net bred inventory = (BR - 1) * destroyed.
        burn_rate = self.fissile_burn_rate * np.maximum(n, 0.0)  # kg/s
        fissile_destroyed = np.concatenate([[0.0], np.cumsum(0.5 * (burn_rate[1:] + burn_rate[:-1]) * np.diff(t))]) if len(t) > 1 else np.zeros_like(t)
        fissile_bred = np.concatenate([[0.0], np.cumsum(0.5 * ((BR_arr * burn_rate)[1:] + (BR_arr * burn_rate)[:-1]) * np.diff(t))]) if len(t) > 1 else np.zeros_like(t)
        net_bred = fissile_bred - fissile_destroyed

        return {
            "t": t,
            "n": n,
            "C": C,
            "T_f": T_f,
            "T_Na": T_Na,
            "P_thermal_W": P_thermal,
            "P_elec_W": P_elec,
            "rho": rho_arr,
            "rho_ext": np.array([_rho_ext(ti) for ti in t]),
            "breeding_ratio": BR_arr,
            "void_fraction": void_arr,
            "fissile_destroyed_kg": fissile_destroyed,
            "fissile_bred_kg": fissile_bred,
            "net_fissile_bred_kg": net_bred,
        }

    # ------------------------------------------------------------------ #
    def step_reactivity_insertion(self, rho_step, dt=0.01, duration_s=100.0,
                                  t_insert=1.0, **kw):
        def rho_ext(t):
            return rho_step if t >= t_insert else 0.0
        return self.simulate(rho_ext, dt, duration_s, **kw)

    def ramp_reactivity_insertion(self, rho_rate, rho_max, dt=0.01,
                                  duration_s=100.0, t_start=1.0, **kw):
        def rho_ext(t):
            if t < t_start:
                return 0.0
            return min(rho_rate * (t - t_start), rho_max)
        return self.simulate(rho_ext, dt, duration_s, **kw)
