"""
EC117 -- Boiling Water Reactor (BWR) -- F2a Point Kinetics + Thermal-Hydraulics

Physics-lumped 0D model: the point reactor kinetics equations (6 delayed-neutron
groups) coupled to a lumped fuel + coolant thermal-hydraulic model, closed by
reactivity feedback. In a BWR the coolant boils inside the core, so the dominant
self-stabilising feedback is the (strongly negative) VOID coefficient, together
with the (negative) Doppler fuel-temperature coefficient and a moderator-
temperature term.

State vector y = [n, C1..C6, T_fuel, T_coolant, void]:
    n        -- normalised neutron population / power  (P = n * P_rated)
    C_i      -- normalised delayed-neutron precursor concentration, group i
    T_fuel   -- lumped fuel temperature                [K]
    T_cool   -- lumped coolant/moderator temperature   [K]
    void     -- core-average void fraction             [-]

Point reactor kinetics (Duderstadt & Hamilton 1976, Ch. 6; Lamarsh & Baratta 2001):
    dn/dt   = (rho - beta)/Lambda * n + sum_i lambda_i * C_i
    dC_i/dt = beta_i/Lambda * n - lambda_i * C_i

Reactivity with feedback (rho measured relative to the steady full-power state):
    rho = rho_ext
          + alpha_D    * (T_fuel  - T_fuel_ref)        Doppler  (negative)
          + alpha_void * (void    - void_ref)           void     (negative, BWR key)
          + alpha_cool * (T_cool  - T_cool_ref)         moderator (negative)

Lumped thermal-hydraulics (Todreas & Kazimi 2012; Lamarsh & Baratta 2001):
    m_f cp_f dT_fuel/dt = P_th         - hA (T_fuel - T_cool)
    m_c cp_c dT_cool/dt = hA (T_fuel - T_cool) - W_cool (T_cool - T_sink)
    Boiling power Q_boil drives vapour generation -> core-average void:
    void_eq = void_ref * void_gain * (Q_boil / Q_boil_ref)
    tau_v dvoid/dt = void_eq - void          (first-order void relaxation)

Energy conservation: at steady state P_th = hA dT_fc = W_cool dT_cs, i.e. all
fission power is removed by the coolant sink (balance verified in tests).

References:
    Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
    Lamarsh, J.R. & Baratta, A.J. (2001). Introduction to Nuclear Engineering,
        3rd ed. Prentice Hall.
    Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems, 2nd ed. CRC Press.
    Keepin, G.R. (1965). Physics of Nuclear Kinetics (6-group U-235 data).
"""

import numpy as np
from scipy.integrate import solve_ivp


class BWR_F2a:
    """BWR point-kinetics + lumped thermal-hydraulics with reactivity feedback."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_thermal_rated_mw"]["value"] * 1e6          # W

        # --- point kinetics ---
        self.Lambda = u["Lambda"]["value"]
        self.beta_i = np.asarray(u["beta_fractions"]["value"], dtype=float)
        self.lam_i = np.asarray(u["lambda_decay"]["value"], dtype=float)
        self.beta = float(u["beta_total"]["value"])
        self.n_groups = len(self.beta_i)

        # --- reactivity feedback coefficients ---
        self.alpha_D = u["alpha_doppler"]["value"]
        self.alpha_void = u["alpha_void"]["value"]
        self.alpha_cool = u["alpha_coolant"]["value"]
        self.T_fuel_ref = u["T_fuel_ref_k"]["value"]
        self.T_cool_ref = u["T_coolant_ref_k"]["value"]
        self.void_ref = u["alpha_void_ref"]["value"]

        # --- thermal masses ---
        self.m_fuel = u["m_fuel_kg"]["value"]
        self.cp_fuel = u["cp_fuel"]["value"]
        self.m_cool = u["m_coolant_kg"]["value"]
        self.cp_cool = u["cp_coolant"]["value"]

        # --- heat transfer ---
        self.hA = u["hA_fuel_coolant"]["value"]
        self.W_cool = u["W_cool"]["value"]
        self.T_sink = u["T_sink_k"]["value"]

        # --- void / boiling ---
        self.h_fg = u["h_fg_jkg"]["value"]
        self.void_gain = u["void_gain"]["value"]
        self.tau_void = u["tau_void_s"]["value"]

        # Reference boiling power (full-power coolant heat throughput) used to
        # normalise the void map. Equals steady heat removed by the sink.
        self.Q_boil_ref = self.W_cool * (self.T_cool_ref - self.T_sink)
        if self.Q_boil_ref <= 0.0:
            # Fall back to rated thermal power if sink == coolant ref.
            self.Q_boil_ref = self.P_rated

        # The reactivity feedbacks are referenced to the SELF-CONSISTENT full-power
        # steady state (n=1) so that with no external reactivity the reactor sits
        # exactly at equilibrium (rho == 0). The datasheet alpha_* coefficients and
        # nominal void are retained; the temperature reference points are recomputed
        # from the thermal balance so kinetics and thermal-hydraulics are consistent.
        T_cool_ss = self.T_sink + self.P_rated / self.W_cool
        T_fuel_ss = T_cool_ss + self.P_rated / self.hA
        Q_cs_ss = self.W_cool * (T_cool_ss - self.T_sink)
        void_ss = self.void_ref * self.void_gain * Q_cs_ss / self.Q_boil_ref
        self.T_fuel_ref = T_fuel_ss
        self.T_cool_ref = T_cool_ss
        self.void_ref = void_ss

    # ------------------------------------------------------------------ kinetics
    def reactivity(self, rho_ext, T_fuel, T_cool, void):
        """Total reactivity [dimensionless dk/k] including feedbacks."""
        return (rho_ext
                + self.alpha_D * (T_fuel - self.T_fuel_ref)
                + self.alpha_void * (void - self.void_ref)
                + self.alpha_cool * (T_cool - self.T_cool_ref))

    def dollars_to_reactivity(self, dollars):
        """Convert reactivity in dollars to absolute dk/k ( 1$ = beta )."""
        return dollars * self.beta

    # --------------------------------------------------------------- derivatives
    def _rhs(self, t, y, rho_ext_func):
        n = y[0]
        C = y[1:1 + self.n_groups]
        T_fuel = y[1 + self.n_groups]
        T_cool = y[2 + self.n_groups]
        void = y[3 + self.n_groups]

        rho_ext = rho_ext_func(t)
        rho = self.reactivity(rho_ext, T_fuel, T_cool, void)

        # Point kinetics
        dn = (rho - self.beta) / self.Lambda * n + np.dot(self.lam_i, C)
        dC = self.beta_i / self.Lambda * n - self.lam_i * C

        # Thermal-hydraulics
        P_th = n * self.P_rated
        Q_fc = self.hA * (T_fuel - T_cool)                  # fuel -> coolant
        Q_cs = self.W_cool * (T_cool - self.T_sink)         # coolant -> sink (boiling/extraction)

        dT_fuel = (P_th - Q_fc) / (self.m_fuel * self.cp_fuel)
        dT_cool = (Q_fc - Q_cs) / (self.m_cool * self.cp_cool)

        # Void: boiling power sets equilibrium core-average void, first-order lag
        void_eq = self.void_ref * self.void_gain * max(Q_cs, 0.0) / self.Q_boil_ref
        void_eq = min(void_eq, 0.95)
        dvoid = (void_eq - void) / self.tau_void

        dydt = np.empty_like(y)
        dydt[0] = dn
        dydt[1:1 + self.n_groups] = dC
        dydt[1 + self.n_groups] = dT_fuel
        dydt[2 + self.n_groups] = dT_cool
        dydt[3 + self.n_groups] = dvoid
        return dydt

    # --------------------------------------------------------------- equilibrium
    def steady_state(self, n0=1.0):
        """
        Self-consistent initial condition for power level n0 (fraction of rated).
        Precursors at equilibrium: C_i = beta_i/(Lambda lambda_i) * n0.
        Temperatures and void from the steady thermal balance at P = n0 P_rated.
        """
        C0 = self.beta_i / (self.Lambda * self.lam_i) * n0
        P_th = n0 * self.P_rated
        T_cool = self.T_sink + P_th / self.W_cool
        T_fuel = T_cool + P_th / self.hA
        Q_cs = self.W_cool * (T_cool - self.T_sink)
        void = self.void_ref * self.void_gain * Q_cs / self.Q_boil_ref
        y0 = np.concatenate(([n0], C0, [T_fuel, T_cool, void]))
        return y0

    # ----------------------------------------------------------------- simulate
    def simulate(self, rho_ext, duration_s=100.0, dt=0.05, n0=1.0, y0=None):
        """
        Integrate the coupled kinetics + thermal-hydraulic ODEs.

        rho_ext     : float (constant) or callable rho_ext(t) -- EXTERNAL reactivity
                      in absolute dk/k, applied on top of the reference steady state.
        duration_s  : total simulated time [s]
        dt          : output sampling step [s]
        n0          : initial power fraction (used if y0 is None)
        y0          : optional explicit initial state vector

        Returns dict of time-series arrays.
        """
        if callable(rho_ext):
            rho_func = rho_ext
        else:
            rho_val = float(rho_ext)
            rho_func = lambda t: rho_val

        if y0 is None:
            y0 = self.steady_state(n0)

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            method="LSODA", t_eval=t_eval,
            args=(rho_func,), rtol=1e-7, atol=1e-9, max_step=dt,
        )

        n = sol.y[0]
        C = sol.y[1:1 + self.n_groups]
        T_fuel = sol.y[1 + self.n_groups]
        T_cool = sol.y[2 + self.n_groups]
        void = sol.y[3 + self.n_groups]

        rho_ext_arr = np.array([rho_func(tt) for tt in sol.t])
        rho_total = self.reactivity(rho_ext_arr, T_fuel, T_cool, void)

        return {
            "t": sol.t,
            "n": n,                                  # power fraction
            "power_mw": n * self.P_rated / 1e6,      # MW_th
            "precursors": C,                         # (6, N)
            "T_fuel": T_fuel,                        # K
            "T_coolant": T_cool,                     # K
            "void_fraction": void,                   # -
            "reactivity": rho_total,                 # dk/k
            "reactivity_ext": rho_ext_arr,           # dk/k
            "reactivity_dollars": rho_total / self.beta,
            "success": sol.success,
        }
