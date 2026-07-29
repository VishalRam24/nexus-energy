"""
EC121 -- High Temperature Gas Reactor (HTGR) -- F2a Point Kinetics + Lumped Thermal

Physics-lumped (0D) dynamic model of a TRISO-fueled, helium-cooled, graphite-moderated
HTGR (pebble-bed / HTR-PM class). Couples the point reactor kinetics equations
(6 delayed-neutron groups) to a 3-node lumped thermal model (fuel pebbles, graphite
moderator/reflector, helium coolant) through strong NEGATIVE temperature reactivity
feedback (prompt Doppler in the fuel + slow graphite feedback).

State vector y = [ n, C_1..C_6, T_fuel, T_graphite, T_helium ]
  n          : neutron population (= normalized reactor power, n = P/P_rated)
  C_i        : delayed-neutron precursor concentrations (same normalization as n)
  T_fuel     : lumped fuel (pebble) temperature           [K]
  T_graphite : lumped graphite moderator temperature      [K]
  T_helium   : lumped (mean) helium coolant temperature   [K]

----------------------------------------------------------------------------
1. Point reactor kinetics (Duderstadt & Hamilton 1976, Ch. 6; Hetrick 1971):

    dn/dt   = (rho(t) - beta)/Lambda * n + sum_i lambda_i * C_i
    dC_i/dt = beta_i/Lambda * n - lambda_i * C_i

   rho(t) = rho_ext(t) + alpha_fuel*(T_fuel - T_fuel_0)
                        + alpha_graphite*(T_graphite - T_graphite_0)

   alpha_fuel, alpha_graphite < 0  =>  inherent negative feedback (self-stabilizing,
   characteristic of TRISO/graphite HTGR; large negative Doppler gives passive safety).

2. Lumped thermal model (energy balance per node, Reitsma 2004; Strydom 2013):

   Fission power deposited (mostly) in the fuel:
     P_fission = n * P_rated

   C_fuel  dT_fuel/dt = P_fission
                        - hA_fg (T_fuel - T_graphite)
                        - hA_fHe (T_fuel - T_helium)

   C_grph  dT_grph/dt = + hA_fg (T_fuel - T_graphite)
                        - hA_gHe (T_graphite - T_helium)

   C_He    dT_He/dt   = + hA_fHe (T_fuel - T_helium)
                        + hA_gHe (T_graphite - T_helium)
                        - 2*mdot*cp_He (T_He - T_He_in)      [coolant advection]

   The graphite node carries an ENORMOUS heat capacity (~1400 t x 1.3 kJ/kgK ~ 1.8 GJ/K),
   giving HTGRs their very large thermal inertia: power transients are heavily damped and
   slow. Helium (monatomic, cp ~ 5193 J/kgK) is essentially constant-property.

   Mean-coolant model: T_out = 2*T_He_mean - T_He_in (log-mean approx, lumped form).

References:
    Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley. Ch. 6.
    Hetrick, D.L. (1971). Dynamics of Nuclear Reactors. Univ. Chicago Press.
    Reitsma, F. (2004). The PBMR steady state and pebble flow. IAEA TRS-424.
    Strydom, G. (2013). Nucl. Eng. Des. 263:516 (PBMR/HTGR neutronics & feedback).
    Zhang, Z. et al. (2009). Nucl. Eng. Des. 239:2265 (HTR-PM).
    Dong, Y. (2011). Nucl. Eng. Des. 241:4755.
"""

import numpy as np
from scipy.integrate import solve_ivp

PCM = 1.0e-5  # 1 pcm = 1e-5 in reactivity (dk/k)


class HTGR_F2a:
    """HTGR point-kinetics + lumped 3-node thermal model."""

    def __init__(self, params: dict):
        u = params["unit"]

        # --- power / efficiency ---
        self.P_rated = float(u["P_thermal_rated_MW"]["value"]) * 1e6   # W_th
        self.eta = float(u["eta_thermal"]["value"])

        # --- neutronics ---
        self.Lambda = float(u["Lambda_neutron_s"]["value"])            # s
        self.beta_i = np.asarray(u["beta_i"]["value"], dtype=float)
        self.lambda_i = np.asarray(u["lambda_i_per_s"]["value"], dtype=float)
        self.beta = float(np.sum(self.beta_i))
        self.n_groups = len(self.beta_i)

        # --- reactivity feedback (pcm/K -> dk/k per K) ---
        self.alpha_fuel = float(u["alpha_fuel_pcm_per_K"]["value"]) * PCM
        self.alpha_grph = float(u["alpha_graphite_pcm_per_K"]["value"]) * PCM

        # --- thermal capacitances (J/K) ---
        self.C_fuel = (float(u["M_fuel_t"]["value"]) * 1e3
                       * float(u["cp_fuel_J_per_kgK"]["value"]))
        self.C_grph = (float(u["M_graphite_t"]["value"]) * 1e3
                       * float(u["cp_graphite_J_per_kgK"]["value"]))
        self.cp_He = float(u["cp_helium_J_per_kgK"]["value"])
        self.C_He = float(u["M_helium_kg"]["value"]) * self.cp_He

        # --- thermal conductances (MW/K -> W/K) ---
        self.hA_fg = float(u["hA_fuel_graphite_MW_per_K"]["value"]) * 1e6
        self.hA_fHe = float(u["hA_fuel_helium_MW_per_K"]["value"]) * 1e6
        self.hA_gHe = float(u["hA_graphite_helium_MW_per_K"]["value"]) * 1e6

        # --- coolant ---
        self.mdot = float(u["mdot_helium_kg_per_s"]["value"])
        self.T_He_in = float(u["T_helium_inlet_C"]["value"]) + 273.15   # K
        self.T_out_rated = float(u["T_helium_outlet_C"]["value"]) + 273.15
        self.decay_frac = float(u["decay_heat_frac"]["value"])

        # advective coolant removal coefficient (W/K), mean-coolant lumped form
        self.k_adv = 2.0 * self.mdot * self.cp_He

        # reference (steady-state) temperatures, set by steady_state()
        self.T_fuel_0 = None
        self.T_grph_0 = None
        self.T_He_0 = None
        self._steady_state()

    # ------------------------------------------------------------------
    # Steady-state reference at rated power (n = 1)
    # ------------------------------------------------------------------
    def _steady_state(self):
        """
        Solve the linear thermal network at n=1 for the reference node
        temperatures. Reactivity feedback is referenced to these so that at
        rated power the feedback contribution is exactly zero (rho_ext=0 holds
        the reactor critical at full power).
        """
        P = self.P_rated

        # Helium mean from overall balance: P removed by coolant advection
        # k_adv * (T_He - T_He_in) = P  (all fission power leaves with coolant)
        T_He = self.T_He_in + P / self.k_adv

        # Graphite: hA_fg(T_f - T_g) = hA_gHe(T_g - T_He)
        # Fuel: P = hA_fg(T_f - T_g) + hA_fHe(T_f - T_He)
        # Solve 2x2 for T_f, T_g given T_He.
        a11 = self.hA_fg + self.hA_fHe
        a12 = -self.hA_fg
        b1 = P + self.hA_fHe * T_He
        a21 = -self.hA_fg
        a22 = self.hA_fg + self.hA_gHe
        b2 = self.hA_gHe * T_He
        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        T_f, T_g = np.linalg.solve(A, b)

        self.T_fuel_0 = float(T_f)
        self.T_grph_0 = float(T_g)
        self.T_He_0 = float(T_He)

    # ------------------------------------------------------------------
    # Reactivity feedback
    # ------------------------------------------------------------------
    def feedback_reactivity(self, T_fuel, T_grph):
        """Temperature reactivity feedback [dk/k]. Negative when hotter."""
        return (self.alpha_fuel * (T_fuel - self.T_fuel_0)
                + self.alpha_grph * (T_grph - self.T_grph_0))

    def outlet_temperature(self, T_He_mean):
        """Helium core OUTLET temperature from lumped mean (K)."""
        return 2.0 * T_He_mean - self.T_He_in

    # ------------------------------------------------------------------
    # RHS of the coupled ODE system
    # ------------------------------------------------------------------
    def _rhs(self, t, y, rho_ext_fn):
        n = y[0]
        C = y[1:1 + self.n_groups]
        T_fuel = y[1 + self.n_groups]
        T_grph = y[2 + self.n_groups]
        T_He = y[3 + self.n_groups]

        rho_ext = rho_ext_fn(t)
        rho = rho_ext + self.feedback_reactivity(T_fuel, T_grph)

        # Point kinetics
        dn = (rho - self.beta) / self.Lambda * n + np.dot(self.lambda_i, C)
        dC = self.beta_i / self.Lambda * n - self.lambda_i * C

        # Thermal power deposited in fuel (n is normalized power)
        P_fission = n * self.P_rated

        q_fg = self.hA_fg * (T_fuel - T_grph)
        q_fHe = self.hA_fHe * (T_fuel - T_He)
        q_gHe = self.hA_gHe * (T_grph - T_He)
        q_adv = self.k_adv * (T_He - self.T_He_in)

        dT_fuel = (P_fission - q_fg - q_fHe) / self.C_fuel
        dT_grph = (q_fg - q_gHe) / self.C_grph
        dT_He = (q_fHe + q_gHe - q_adv) / self.C_He

        out = np.empty_like(y)
        out[0] = dn
        out[1:1 + self.n_groups] = dC
        out[1 + self.n_groups] = dT_fuel
        out[2 + self.n_groups] = dT_grph
        out[3 + self.n_groups] = dT_He
        return out

    # ------------------------------------------------------------------
    # Initial condition consistent with a given power fraction
    # ------------------------------------------------------------------
    def initial_state(self, P0_fraction=1.0):
        """
        Equilibrium IC at power fraction P0. Precursors at equilibrium:
        C_i = beta_i/(Lambda*lambda_i) * n. Temperatures interpolated from
        the n=1 steady state down toward the inlet at zero power.
        """
        n0 = float(P0_fraction)
        C0 = self.beta_i / (self.Lambda * self.lambda_i) * n0
        # Temperatures scale ~linearly with power above the inlet temperature.
        T_fuel = self.T_He_in + n0 * (self.T_fuel_0 - self.T_He_in)
        T_grph = self.T_He_in + n0 * (self.T_grph_0 - self.T_He_in)
        T_He = self.T_He_in + n0 * (self.T_He_0 - self.T_He_in)
        y0 = np.concatenate([[n0], C0, [T_fuel, T_grph, T_He]])
        return y0

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, rho_ext=0.0, duration_s=1000.0, P0_fraction=1.0,
                 n_eval=400, method="LSODA"):
        """
        Integrate the coupled kinetics+thermal system.

        Parameters
        ----------
        rho_ext   : float or callable(t)->float
                    External reactivity [dk/k] (e.g. control-rod insertion).
                    Negative = rods in. A constant or a function of time.
        duration_s : total simulated time [s]
        P0_fraction: initial power fraction (1.0 = rated)
        n_eval     : number of output sample points
        method     : scipy solver ('LSODA' is stiff-aware, recommended)

        Returns dict of NumPy arrays (interface mirrors EC001 F2a).
        """
        if callable(rho_ext):
            rho_fn = rho_ext
        else:
            _rho = float(rho_ext)
            rho_fn = lambda t: _rho

        y0 = self.initial_state(P0_fraction)
        t_eval = np.linspace(0.0, duration_s, n_eval)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method=method, args=(rho_fn,),
            rtol=1e-6, atol=1e-9, max_step=duration_s / 50.0,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        n = sol.y[0]
        T_fuel = sol.y[1 + self.n_groups]
        T_grph = sol.y[2 + self.n_groups]
        T_He = sol.y[3 + self.n_groups]
        T_out = self.outlet_temperature(T_He)

        P_th = n * self.P_rated
        P_el = self.eta * P_th
        rho_total = np.array([rho_fn(tt) for tt in sol.t]) \
            + self.feedback_reactivity(T_fuel, T_grph)

        return {
            "t": sol.t,
            "power_fraction": n,
            "P_thermal_W": P_th,
            "P_thermal_MW": P_th / 1e6,
            "P_electric_MW": P_el / 1e6,
            "T_fuel_K": T_fuel,
            "T_graphite_K": T_grph,
            "T_helium_mean_K": T_He,
            "T_helium_outlet_K": T_out,
            "T_helium_outlet_C": T_out - 273.15,
            "reactivity_total": rho_total,
            "reactivity_feedback": self.feedback_reactivity(T_fuel, T_grph),
            "precursors": sol.y[1:1 + self.n_groups],
        }

    # ------------------------------------------------------------------
    # Passive decay-heat removal (post-shutdown, no forced flow)
    # ------------------------------------------------------------------
    def decay_heat_simulate(self, duration_s=20000.0, n_eval=400,
                            hA_passive_MW_per_K=0.30):
        """
        After scram, fission stops (n->0) but decay heat persists. With helium
        circulators off, heat is removed passively by conduction/radiation to
        the reactor cavity cooling system (RCCS). HTGRs are designed so the
        fuel stays below the ~1600C TRISO failure limit on passive cooling.

        Decay heat (Way-Wigner approx, Glasstone & Sesonske 1981):
            P_decay(t)/P0 ~ 0.066 * (t^-0.2 - (t+t_op)^-0.2)
        Here simplified to an exponential-free Way-Wigner curve with t_op large.
        """
        P0 = self.P_rated
        hA_p = hA_passive_MW_per_K * 1e6
        C_tot = self.C_fuel + self.C_grph  # solid thermal mass dominates

        def decay_frac(t):
            t = max(t, 1.0)
            return 0.066 * (t ** -0.2)  # fraction of rated power

        def rhs(t, y):
            T = y[0]
            P_dec = decay_frac(t) * P0
            q_out = hA_p * (T - self.T_He_in)
            return [(P_dec - q_out) / C_tot]

        T0 = self.T_grph_0
        t_eval = np.linspace(1.0, duration_s, n_eval)
        sol = solve_ivp(rhs, (1.0, duration_s), [T0], t_eval=t_eval,
                        method="LSODA", rtol=1e-6, atol=1e-6)
        return {
            "t": sol.t,
            "T_core_K": sol.y[0],
            "T_core_C": sol.y[0] - 273.15,
            "P_decay_MW": np.array([decay_frac(tt) for tt in sol.t]) * P0 / 1e6,
        }
