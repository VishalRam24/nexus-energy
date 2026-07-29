"""
EC112 -- Micro Gas Turbine -- F2a Recuperated Brayton Cycle (Physics-Lumped)

First-principles 0D thermodynamic model of a recuperated, single-shaft,
high-speed micro gas turbine, with a lumped shaft + recuperator-metal
transient solved by scipy.integrate.solve_ivp.

Cycle (air-standard with real, temperature-dependent cp):
    1 -> 2   compressor      (isentropic eff eta_c, pressure ratio rp)
    2 -> 2r  recuperator cold side: compressed air pre-heated by exhaust
    2r-> 3   combustor       (fuel added to reach turbine-inlet T, TIT)
    3 -> 4   turbine         (isentropic eff eta_t)
    4 -> 5   recuperator hot side: exhaust gives heat back to air
The recuperator (a counterflow gas-gas heat exchanger, effectiveness epsilon)
is the dominant efficiency driver for micro gas turbines: without it the low
pressure ratio (~3-4) gives only ~17% efficiency; with epsilon~0.85 it reaches
~30% (EPA CHP Catalog 2017; Capstone C200).

Component work (per unit air mass flow), real-cp form:
    w_c  = cp_air(Tbar_12) * (T2  - T1)          / eta_mech   [compressor draw]
    w_t  = cp_gas(Tbar_34) * (T3  - T4)           * eta_mech   [turbine output]
    w_net = w_t - w_c
    q_in  = cp_gas(Tbar)   * (T3  - T2r) / eta_comb           [fuel heat]
    eta_th = w_net / q_in
    eta_el = eta_th * eta_gen

Recuperator (counterflow, effectiveness definition, Moran 2018 / Kays-London):
    T2r = T2 + epsilon * (T4 - T2)               cold-air outlet
    energy balance gives hot-side outlet T5.

Lumped transients (state vector y = [omega, T_recup]):
  Shaft (rotational Newton, single spool):
      I * d(omega)/dt = (P_turb - P_comp - P_load) / omega
  Recuperator metal (thermal mass sets warm-up time constant ~minutes):
      m_r*cp_r * dT_recup/dt = Qhot_in - Qcold_out
  These give the start-up / load-change dynamics; the steady cycle is the
  algebraic fixed point.

Hard-coded gas properties (cited):
  Air / combustion-gas cp(T) use the NASA-style polynomial-equivalent linear
  fits of Cengel & Boles (2015) "Thermodynamics: An Engineering Approach" 8e,
  Table A-2: cp_air rises from ~1.005 kJ/kgK at 300 K to ~1.14 kJ/kgK at
  1200 K; combustion gas ~1.15 kJ/kgK at TIT. gamma from Moran (2018).

References:
    Moran, Shapiro, Boettner, Bailey (2018), Fundamentals of Engineering
        Thermodynamics, 9th ed., Wiley -- Ch. 9 (Brayton + regeneration).
    Cengel & Boles (2015), Thermodynamics: An Engineering Approach, 8th ed.,
        McGraw-Hill -- air/gas cp(T), Table A-2.
    Capstone C200 product data sheet (2023).
    US EPA Combined Heat and Power Catalog (2017), Section 5: Microturbines.
    Kays & London (1984), Compact Heat Exchangers, 3rd ed. (effectiveness-NTU).
"""

import numpy as np
from scipy.integrate import solve_ivp


class MicroGasTurbineF2a:
    """Recuperated Brayton micro gas turbine -- physics-lumped with ODE dynamics."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated = u["P_el_rated_kw"]["value"] * 1e3        # W_e
        self.eta_el_rated = u["eta_el_rated"]["value"]
        self.rp_rated = u["pressure_ratio"]["value"]
        self.T_amb = u["T_amb_k"]["value"]                          # K
        self.P_amb = u["P_amb_kpa"]["value"] * 1e3                  # Pa
        self.TIT = u["TIT_k"]["value"]                              # K
        self.eta_c = u["eta_comp_isen"]["value"]
        self.eta_t = u["eta_turb_isen"]["value"]
        self.eta_comb = u["eta_comb"]["value"]
        self.dp_comb = u["dp_comb_frac"]["value"]
        self.dp_rec_h = u["dp_recup_hot_frac"]["value"]
        self.dp_rec_c = u["dp_recup_cold_frac"]["value"]
        self.eps_rec = u["recup_effectiveness"]["value"]
        self.eta_gen = u["eta_gen"]["value"]
        self.eta_mech = u["eta_mech"]["value"]
        self.LHV = u["LHV_gas_mjkg"]["value"] * 1e6                 # J/kg
        self.rho_gas = u["rho_gas_kgm3"]["value"]
        self.mdot_air_rated = u["mdot_air_rated_kgs"]["value"]      # kg/s
        self.I_shaft = u["shaft_inertia_J"]["value"]               # kg.m^2
        self.omega_rated = u["shaft_speed_rated_rpm"]["value"] * 2.0 * np.pi / 60.0  # rad/s
        self.m_rec = u["recup_mass_kg"]["value"]
        self.cp_rec = u["recup_cp_jkgK"]["value"]
        self.gamma_air = u["gamma_air"]["value"]
        self.gamma_gas = u["gamma_gas"]["value"]
        self.R_air = u["R_air_jkgK"]["value"]

    # ------------------------------------------------------------------
    # Temperature-dependent specific heats (Cengel & Boles 2015, Table A-2)
    # Linear fits valid 250-1300 K; reproduce tabulated values within ~1%.
    # ------------------------------------------------------------------
    @staticmethod
    def cp_air(T):
        """Air cp [J/(kg.K)] vs T [K]. 1.005 at 300 K -> ~1.14 at 1200 K."""
        T = np.asarray(T, dtype=float)
        return 1005.0 + 0.156 * (T - 300.0)

    @staticmethod
    def cp_gas(T):
        """Combustion-gas cp [J/(kg.K)] vs T [K]. ~1.09 at 600 K -> ~1.20 at 1300 K."""
        T = np.asarray(T, dtype=float)
        return 1080.0 + 0.115 * (T - 600.0)

    # ------------------------------------------------------------------
    # Carnot bound (for sanity / enforcement)
    # ------------------------------------------------------------------
    def carnot_efficiency(self, T_hot=None, T_cold=None):
        """Carnot efficiency between TIT and ambient -- the thermodynamic ceiling."""
        Th = self.TIT if T_hot is None else T_hot
        Tc = self.T_amb if T_cold is None else T_cold
        return 1.0 - Tc / Th

    # ------------------------------------------------------------------
    # Isentropic temperature relations with finite efficiency
    # ------------------------------------------------------------------
    def compressor_outlet_T(self, T1, rp):
        """Compressor exit T2 [K] with isentropic efficiency (Moran 2018)."""
        # ideal isentropic exit temperature
        T2s = T1 * rp ** ((self.gamma_air - 1.0) / self.gamma_air)
        # real exit (eff defined on temperature rise)
        return T1 + (T2s - T1) / self.eta_c

    def turbine_outlet_T(self, T3, expansion_ratio):
        """Turbine exit T4 [K] with isentropic efficiency (Moran 2018)."""
        T4s = T3 / expansion_ratio ** ((self.gamma_gas - 1.0) / self.gamma_gas)
        return T3 - self.eta_t * (T3 - T4s)

    # ------------------------------------------------------------------
    # Steady-state cycle solve at a given pressure ratio / TIT / ambient
    # ------------------------------------------------------------------
    def cycle(self, rp=None, TIT=None, T_amb=None, mdot_air=None):
        """
        Solve the recuperated Brayton cycle, return a dict of station states,
        specific works, heat input, efficiencies and power (real cp).
        """
        rp = self.rp_rated if rp is None else rp
        TIT = self.TIT if TIT is None else TIT
        T1 = self.T_amb if T_amb is None else T_amb
        mdot = self.mdot_air_rated if mdot_air is None else mdot_air

        # --- Compressor 1->2 ---
        T2 = self.compressor_outlet_T(T1, rp)
        cp_c = self.cp_air(0.5 * (T1 + T2))
        w_c = cp_c * (T2 - T1) / self.eta_mech            # J/kg air, includes mech loss

        # --- Recuperator cold side 2->2r (preheat by exhaust) ---
        # T4 needed for recuperator; couple via fixed-point on T4.
        # Pressure path: combustor + recuperator-cold loss reduce turbine inlet pressure;
        # recuperator-hot loss reduces available expansion ratio.
        P2 = self.P_amb * rp
        P3 = P2 * (1.0 - self.dp_rec_c) * (1.0 - self.dp_comb)     # turbine inlet pressure
        P4 = self.P_amb / (1.0 - self.dp_rec_h)                    # turbine exit must overcome hot-side loss to ambient
        expansion_ratio = P3 / P4

        T3 = TIT
        T4 = self.turbine_outlet_T(T3, expansion_ratio)

        # recuperator (effectiveness): cold air heated toward hot exhaust temp
        T2r = T2 + self.eps_rec * (T4 - T2)

        # --- Combustor 2r->3 (fuel heat) ---
        cp_b = self.cp_gas(0.5 * (T2r + T3))
        q_in = cp_b * (T3 - T2r) / self.eta_comb          # J/kg air (fuel chemical heat)

        # --- Turbine 3->4 work ---
        cp_t = self.cp_gas(0.5 * (T3 + T4))
        w_t = cp_t * (T3 - T4) * self.eta_mech            # J/kg air

        # --- Recuperator hot side 4->5 (gives heat to air) ---
        # energy balance on the recuperator: heat given by gas = heat taken by air
        cp_air_rec = self.cp_air(0.5 * (T2 + T2r))
        q_rec = cp_air_rec * (T2r - T2)
        cp_gas_hot = self.cp_gas(0.5 * (T4 + T2r))
        T5 = T4 - q_rec / cp_gas_hot                       # exhaust stack temperature

        # --- specific net work & efficiencies ---
        w_net = w_t - w_c
        eta_th = w_net / q_in if q_in > 0 else 0.0
        eta_el = eta_th * self.eta_gen

        # --- power (multiply by air mass flow) ---
        P_shaft = w_net * mdot                             # W mechanical
        P_el = P_shaft * self.eta_gen                      # W electrical
        Q_fuel = q_in * mdot                               # W fuel (LHV)
        mdot_fuel = Q_fuel / self.LHV                      # kg/s

        return {
            "T1": T1, "T2": T2, "T2r": T2r, "T3": T3, "T4": T4, "T5": T5,
            "P2_Pa": P2, "P3_Pa": P3, "P4_Pa": P4,
            "expansion_ratio": expansion_ratio,
            "w_comp": w_c, "w_turb": w_t, "w_net": w_net, "q_in": q_in,
            "eta_thermal": eta_th, "eta_electrical": eta_el,
            "P_shaft_W": P_shaft, "P_el_W": P_el,
            "Q_fuel_W": Q_fuel, "mdot_fuel_kgs": mdot_fuel,
            "T_exhaust_K": T5,
            "eta_carnot": self.carnot_efficiency(TIT, T1),
        }

    def cycle_no_recuperator(self, rp=None, TIT=None, T_amb=None, mdot_air=None):
        """Same cycle with recuperator disabled (epsilon=0) -- for comparison."""
        eps_save = self.eps_rec
        self.eps_rec = 0.0
        try:
            return self.cycle(rp, TIT, T_amb, mdot_air)
        finally:
            self.eps_rec = eps_save

    # ------------------------------------------------------------------
    # Part-load: micro turbines run constant-TIT, reduce speed & airflow.
    # Lower spool speed -> lower pressure ratio & airflow -> lower power.
    # Simple compressor map: rp scales ~ (N/N_rated)^k, mdot ~ N/N_rated.
    # ------------------------------------------------------------------
    def partload_state(self, PLR, T_amb=None):
        """
        Map a target part-load ratio to (rp, mdot_air) and solve the cycle.
        Returns the cycle dict augmented with PLR.
        Iterates spool-speed fraction until P_el matches PLR * rated.
        """
        T1 = self.T_amb if T_amb is None else T_amb
        target = PLR * self.P_el_rated
        # bisection on speed fraction n in [0.3, 1.0]
        lo, hi = 0.30, 1.05
        for _ in range(60):
            n = 0.5 * (lo + hi)
            rp = 1.0 + (self.rp_rated - 1.0) * n ** 1.3
            mdot = self.mdot_air_rated * n
            c = self.cycle(rp=rp, TIT=self.TIT, T_amb=T1, mdot_air=mdot)
            if c["P_el_W"] < target:
                lo = n
            else:
                hi = n
        c["PLR"] = c["P_el_W"] / self.P_el_rated
        c["speed_fraction"] = n
        return c

    # ------------------------------------------------------------------
    # Lumped dynamic ODE: shaft speed + recuperator-metal temperature
    # ------------------------------------------------------------------
    def _rhs(self, t, y, fuel_frac_fn, P_load_fn, T_amb):
        omega, T_rec = y
        omega = max(omega, 1.0)                          # avoid divide-by-zero
        n = omega / self.omega_rated
        n = np.clip(n, 0.05, 1.2)

        # quasi-steady aero map at current speed
        rp = 1.0 + (self.rp_rated - 1.0) * n ** 1.3
        mdot = self.mdot_air_rated * n

        # fuel command sets effective TIT (combustor heat release)
        ff = float(fuel_frac_fn(t))                       # fuel fraction of rated
        # TIT rises with fuel-air ratio; clamp to material limit
        TIT_eff = self.T_amb + (self.TIT - self.T_amb) * np.clip(ff / max(n, 0.1), 0.0, 1.15)
        TIT_eff = np.clip(TIT_eff, self.T_amb + 50.0, self.TIT * 1.05)

        # compressor / turbine powers at this operating point
        T1 = T_amb
        T2 = self.compressor_outlet_T(T1, rp)
        cp_c = self.cp_air(0.5 * (T1 + T2))
        P_comp = cp_c * (T2 - T1) * mdot / self.eta_mech

        P2 = self.P_amb * rp
        P3 = P2 * (1.0 - self.dp_rec_c) * (1.0 - self.dp_comb)
        P4 = self.P_amb / (1.0 - self.dp_rec_h)
        er = P3 / P4
        T4 = self.turbine_outlet_T(TIT_eff, er)
        cp_t = self.cp_gas(0.5 * (TIT_eff + T4))
        P_turb = cp_t * (TIT_eff - T4) * mdot * self.eta_mech

        P_load = float(P_load_fn(t))                      # electrical load -> shaft drag
        P_drag = P_load / self.eta_gen

        # shaft rotational dynamics: I*omega*domega/dt = P_turb - P_comp - P_drag
        domega = (P_turb - P_comp - P_drag) / (self.I_shaft * omega)

        # recuperator metal thermal balance: hot exhaust in, cold air out
        # gas enters recuperator at T4, metal at T_rec; air enters at T2.
        UA_hot = mdot * self.cp_gas(0.5 * (T4 + T_rec)) * self.eps_rec
        UA_cold = mdot * self.cp_air(0.5 * (T2 + T_rec)) * self.eps_rec
        Q_hot_in = UA_hot * (T4 - T_rec)                  # gas heats metal
        Q_cold_out = UA_cold * (T_rec - T2)               # metal heats air
        dT_rec = (Q_hot_in - Q_cold_out) / (self.m_rec * self.cp_rec)

        return [domega, dT_rec]

    def simulate(self, fuel_fraction=1.0, P_load_kw=None, T_amb=None,
                 omega0_frac=1.0, T_rec0=None, dt=0.5, duration_s=120.0):
        """
        Transient simulation of shaft speed + recuperator temperature.

        Parameters
        ----------
        fuel_fraction : float or callable(t)
            Commanded fuel fraction of rated (0..~1.1). Sets TIT.
        P_load_kw : float or callable(t) or None
            Electrical load [kW]. Default = PLR-tracking rated * fuel_fraction.
        T_amb : float
            Ambient (compressor inlet) temperature [K].
        omega0_frac : float
            Initial spool speed as fraction of rated.
        T_rec0 : float
            Initial recuperator metal temperature [K] (default ambient -> cold start).
        dt, duration_s : float
            Output step and total time.

        Returns
        -------
        dict of time-series arrays: t, omega, speed_rpm, T_recup, P_el_kw,
            eta_electrical, eta_carnot, T_exhaust_K, fuel_kw.
        """
        T1 = self.T_amb if T_amb is None else T_amb
        ff_fn = fuel_fraction if callable(fuel_fraction) else (lambda t: fuel_fraction)
        if P_load_kw is None:
            load_fn = lambda t: float(ff_fn(t)) * self.P_el_rated / 1e3 * 0.9
        else:
            load_fn = P_load_kw if callable(P_load_kw) else (lambda t: P_load_kw)
        Pload_W_fn = lambda t: float(load_fn(t)) * 1e3

        omega0 = omega0_frac * self.omega_rated
        Trec0 = T1 if T_rec0 is None else T_rec0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [omega0, Trec0],
            t_eval=t_eval, args=(ff_fn, Pload_W_fn, T1),
            method="RK45", rtol=1e-6, atol=1e-6, max_step=dt,
        )

        t_out = sol.t
        omega = sol.y[0]
        T_rec = sol.y[1]
        N = len(t_out)

        speed_rpm = omega * 60.0 / (2.0 * np.pi)
        P_el = np.zeros(N)
        eta_el = np.zeros(N)
        eta_carnot = np.zeros(N)
        T_exh = np.zeros(N)
        fuel_kw = np.zeros(N)

        for i in range(N):
            n = np.clip(omega[i] / self.omega_rated, 0.05, 1.2)
            rp = 1.0 + (self.rp_rated - 1.0) * n ** 1.3
            mdot = self.mdot_air_rated * n
            ff = float(ff_fn(t_out[i]))
            TIT_eff = self.T_amb + (self.TIT - self.T_amb) * np.clip(ff / max(n, 0.1), 0.0, 1.15)
            TIT_eff = np.clip(TIT_eff, self.T_amb + 50.0, self.TIT * 1.05)
            c = self.cycle(rp=rp, TIT=TIT_eff, T_amb=T1, mdot_air=mdot)
            P_el[i] = c["P_el_W"] / 1e3
            eta_el[i] = c["eta_electrical"]
            eta_carnot[i] = c["eta_carnot"]
            T_exh[i] = c["T_exhaust_K"]
            fuel_kw[i] = c["Q_fuel_W"] / 1e3

        return {
            "t": t_out,
            "omega": omega,
            "speed_rpm": speed_rpm,
            "T_recup": T_rec,
            "P_el_kw": P_el,
            "eta_electrical": eta_el,
            "eta_carnot": eta_carnot,
            "T_exhaust_K": T_exh,
            "fuel_kw": fuel_kw,
        }
