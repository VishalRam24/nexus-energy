"""
EC072 -- CO2 Transcritical Heat Pump (R744) -- F2a Transcritical Cycle Model

Physics-lumped first-principles model of the transcritical CO2 (R744) heat-pump
cycle. The defining feature of R744 is that the high side operates ABOVE the
critical point (P_high > P_crit = 73.77 bar), so heat is rejected in a
SUPERCRITICAL GAS COOLER -- there is no condensation plateau. Instead the CO2
temperature GLIDES continuously from the compressor-discharge temperature down
to near the water-inlet temperature, which is an excellent thermodynamic match
to sensible water heating and the reason CO2 heat pumps reach high water-outlet
temperatures efficiently (Lorentzen 1994; Kim, Pettersen & Bullard 2004).

Cycle state points (no CoolProp -- hardcoded R744 correlations, see below):
    1  Compressor inlet (suction): saturated vapor at P_low + superheat
    2  Compressor outlet (discharge): supercritical, P_high
    3  Gas-cooler outlet: supercritical, P_high, T_gc_out = T_water_in + pinch
    4  Expansion-valve outlet: two-phase at P_low (isenthalpic, h4 = h3)

Subcritical evaporator (state 4 -> 1): refrigerant evaporates at P_low set by
the source temperature; latent-heat correlation declines toward the critical
point. Supercritical gas cooler (state 2 -> 3): sensible cooling of dense
supercritical fluid with a strongly temperature-dependent real-gas cp that
peaks at the pseudocritical temperature -- this cp(T) hump is what produces the
characteristic temperature glide and the optimum high-side pressure.

Thermodynamics:
    w_comp_isentropic = cp_super(T_avg,P_high) * (T2s - T1)   [enthalpy rise]
    w_comp_actual     = w_comp_isentropic / eta_isentropic
    q_gc (heat out)   = h2 - h3   (integrated cp_super along the gas cooler)
    COP_heating       = q_gc / w_comp_actual
There is a clear optimum P_high (Liao et al. 2000): too low -> small enthalpy
difference across the gas cooler; too high -> excessive compressor work. The
model finds it by 1-D search and also exposes the Liao correlation.

Lumped transient: a single ODE integrates the heated-water (gas-cooler water
side / buffer) temperature using scipy.integrate.solve_ivp:
    m_w * cp_w * dT_w/dt = Q_gc(T_w) - Q_load
where Q_gc gliding heat input falls as the water warms (the approach to the
gas-cooler outlet tightens), giving a physically self-limiting charge curve.

References:
    Lorentzen, G. (1994). Revival of carbon dioxide as a refrigerant.
        Int. J. Refrigeration 17(5), 292-301.
    Kim, M.-H., Pettersen, J., Bullard, C.W. (2004). Fundamental process and
        system design issues in CO2 vapor compression systems. Prog. Energy
        Combust. Sci. 30(2), 119-174.
    Span, R., Wagner, W. (1996). A New Equation of State for Carbon Dioxide.
        J. Phys. Chem. Ref. Data 25(6), 1509-1596.  (property correlations)
    Sarkar, J., Bhattacharyya, S., Ram Gopal, M. (2004). Int. J. Refrig. 27,
        830-838.  (optimum-pressure / COP behavior)
    Liao, S.M., Zhao, T.S., Jakobsen, A. (2000). Appl. Thermal Eng. 20, 831-841.
        (optimum high-side pressure correlation)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


class CO2TranscriticalHPF2a:
    """Transcritical R744 heat pump -- thermodynamic cycle + lumped water ODE."""

    R = 8.314          # J/(mol.K)
    M_CO2 = 44.01e-3   # kg/mol

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["co2_property_correlations"]

        self.Q_rated = u["rated_heating_capacity"]["value"]   # kW_th
        self.eta_is = u["eta_isentropic"]["value"]
        self.eta_me = u["eta_mech_elec"]["value"]
        self.pinch_gc = u["pinch_gc"]["value"]                # K
        self.pinch_evap = u["pinch_evap"]["value"]            # K
        self.superheat = u["superheat"]["value"]              # K
        self.T_crit = u["T_crit"]["value"]                    # K
        self.P_crit = u["P_crit"]["value"]                    # bar
        self.cp_gas = u["cp_co2_gas"]["value"]                # kJ/(kg.K)
        self.cp_water = u["cp_water"]["value"]                # kJ/(kg.K)
        self.a = u["P_high_opt_a"]["value"]
        self.b = u["P_high_opt_b"]["value"]
        self.c = u["P_high_opt_c"]["value"]
        self.m_water = u["m_water_tank"]["value"]             # kg
        self.UA_gc = u["UA_gc"]["value"]                      # kW/K
        self.aux_power = u["aux_power"]["value"]              # kW
        self.h_fg_ref = u["h_fg_evap_ref"]["value"]          # kJ/kg

        # Supercritical real-gas cp(T,P) correlation parameters (Span-Wagner trends)
        self.cp_base = c["cp_super_base"]["value"]            # kJ/(kg.K)
        self.cp_peak = c["cp_super_peak"]["value"]            # kJ/(kg.K)
        self.T_pc_ref = c["T_pseudocrit_ref"]["value"]       # K (at ~90 bar)
        self.pc_width = c["pseudocrit_width"]["value"]        # K
        self.dTpc_dP = c["dTpc_dP"]["value"]                 # K/bar
        self._P_pc_ref = 90.0                                 # bar reference for T_pc_ref

    # ------------------------------------------------------------------
    # R744 property correlations (replace CoolProp)
    # ------------------------------------------------------------------
    def T_pseudocritical(self, P_high_bar):
        """Pseudocritical temperature: cp(T) peaks here on a supercritical isobar.
        Rises with pressure above the critical point (Span & Wagner 1996)."""
        return self.T_pc_ref + self.dTpc_dP * (P_high_bar - self._P_pc_ref)

    def cp_super(self, T_K, P_high_bar):
        """Supercritical CO2 specific heat [kJ/(kg.K)].

        Real-gas cp has a sharp hump at the pseudocritical temperature and
        relaxes to a baseline far from it. Modeled as a Gaussian on (T - T_pc),
        reproducing the Span & Wagner (1996) EOS behavior responsible for the
        gas-cooler temperature glide. Vectorized in T."""
        T = np.asarray(T_K, dtype=float)
        T_pc = self.T_pseudocritical(P_high_bar)
        hump = self.cp_peak * np.exp(-((T - T_pc) ** 2) / (2.0 * self.pc_width ** 2))
        return self.cp_base + hump

    def enthalpy_super(self, T_K, P_high_bar, n=60):
        """Supercritical specific enthalpy [kJ/kg] relative to T_crit, via
        trapezoidal integration of cp_super(T) along the isobar. The integral
        over the variable cp captures the real-gas glide that a constant-cp
        model would miss."""
        T = float(T_K)
        Tg = np.linspace(self.T_crit, T, n)
        cpg = self.cp_super(Tg, P_high_bar)
        return float(np.trapz(cpg, Tg))

    def latent_heat_evap(self, T_evap_K):
        """CO2 latent heat of vaporization [kJ/kg] in the subcritical evaporator.
        Declines toward zero at the critical point (Watson-type correlation
        anchored to NIST ~230 kJ/kg near 0 C); Span & Wagner (1996)."""
        Tr = np.clip(T_evap_K / self.T_crit, 0.0, 0.999)
        Tr_ref = 273.15 / self.T_crit
        return self.h_fg_ref * ((1.0 - Tr) / (1.0 - Tr_ref)) ** 0.38

    def saturation_pressure(self, T_K):
        """CO2 saturation pressure [bar] -- Antoine-form fit to Span & Wagner
        (1996) saturation line, valid roughly -30..+30 C."""
        T = np.asarray(T_K, dtype=float)
        # ln(P/Pc) = (Tc/T) * sum(a_i * (1-T/Tc)^p_i)  (Span-Wagner ancillary eq.)
        tau = 1.0 - T / self.T_crit
        a1, a2, a3 = -7.0602087, 1.9391218, -1.6463597
        lnPr = (self.T_crit / T) * (a1 * tau + a2 * tau ** 1.5 + a3 * tau ** 2.5)
        return self.P_crit * np.exp(lnPr)

    # ------------------------------------------------------------------
    # Cycle thermodynamics
    # ------------------------------------------------------------------
    def cycle_states(self, T_source_c, T_water_in_c, P_high_bar):
        """Compute the four cycle state points and per-kg work/heat.

        Returns dict with temperatures [K], pressures [bar], specific
        compressor work and gas-cooler heat [kJ/kg], and COP_heating."""
        T_source = T_source_c + 273.15
        T_water_in = T_water_in_c + 273.15

        # --- Low side (subcritical evaporator) ---
        T_evap = T_source - self.pinch_evap
        P_low = float(self.saturation_pressure(T_evap))
        T1 = T_evap + self.superheat              # suction (superheated vapor)

        # --- Gas-cooler outlet (state 3): glides down to near water inlet ---
        T3 = T_water_in + self.pinch_gc           # supercritical outlet temp

        # --- Compression 1 -> 2 (isentropic ideal then efficiency) ---
        # Isentropic discharge T via ideal-gas relation with pressure ratio.
        # gamma for CO2 vapor ~1.28-1.30 near suction (Kim 2004).
        gamma = 1.29
        PR = P_high_bar / P_low
        T2s = T1 * PR ** ((gamma - 1.0) / gamma)
        # Isentropic specific work uses suction vapor cp; actual via efficiency.
        w_is = self.cp_gas * (T2s - T1)
        w_act = w_is / self.eta_is
        # Actual discharge temperature from actual enthalpy rise.
        T2 = T1 + w_act / self.cp_gas

        # --- Gas cooler 2 -> 3: real-gas enthalpy drop (variable cp) ---
        h2 = self.enthalpy_super(T2, P_high_bar)
        h3 = self.enthalpy_super(T3, P_high_bar)
        q_gc = h2 - h3                             # kJ/kg rejected to water

        # Electrical work includes mechanical/electric losses.
        w_elec = w_act / self.eta_me

        cop = q_gc / w_elec if w_elec > 0 else 0.0

        return {
            "T1": T1, "T2": T2, "T2s": T2s, "T3": T3, "T_evap": T_evap,
            "P_low": P_low, "P_high": P_high_bar,
            "w_is": w_is, "w_act": w_act, "w_elec": w_elec,
            "q_gc": q_gc, "cop": cop, "PR": PR,
        }

    def optimum_high_pressure_search(self, T_source_c, T_water_in_c,
                                     P_lo=74.5, P_hi=130.0):
        """Find P_high that MAXIMIZES heating COP by 1-D search (Sarkar 2004).
        This is the physical optimum from the cp(T) glide, distinct from the
        Liao algebraic correlation."""
        def neg_cop(P):
            return -self.cycle_states(T_source_c, T_water_in_c, P)["cop"]
        res = minimize_scalar(neg_cop, bounds=(P_lo, P_hi), method="bounded")
        return float(res.x), -float(res.fun)

    def optimum_high_pressure_liao(self, T_gc_out_c):
        """Liao et al. (2000) algebraic optimum high-side pressure [bar]."""
        T = np.asarray(T_gc_out_c, dtype=float)
        return self.a * T + self.b * T ** 2 + self.c

    def cop(self, T_source_c, T_water_in_c, P_high_bar=None):
        """Heating COP. If P_high is None, use the optimum."""
        if P_high_bar is None:
            _, cop = self.optimum_high_pressure_search(T_source_c, T_water_in_c)
            return cop
        return self.cycle_states(T_source_c, T_water_in_c, P_high_bar)["cop"]

    def glide(self, T_source_c, T_water_in_c, P_high_bar, n=20):
        """Gas-cooler CO2 temperature glide profile [degC] vs normalized
        enthalpy position. Demonstrates the continuous (non-isothermal) heat
        rejection -- the signature CO2 feature absent in subcritical cycles."""
        st = self.cycle_states(T_source_c, T_water_in_c, P_high_bar)
        T2, T3 = st["T2"], st["T3"]
        Tprof = np.linspace(T2, T3, n)
        return Tprof - 273.15

    # ------------------------------------------------------------------
    # Lumped transient ODE (gas-cooler water side)
    # ------------------------------------------------------------------
    def simulate(self, T_source_c, T_water_in_c, T_water_target_c,
                 P_high_bar=None, Q_load_kW=0.0, duration_s=1800.0,
                 m_dot_co2=None, n_eval=200):
        """Integrate the lumped heated-water-tank temperature with solve_ivp.

        State: T_tank [degC] -- temperature of the heated-water buffer/store.

            m_w cp_w dT_tank/dt = Q_gc - Q_load

        This models the CO2 sanitary-hot-water configuration (Kim, Pettersen &
        Bullard 2004): cold mains water at a FIXED inlet T_water_in_c is heated
        once-through in the gas cooler, where the supercritical CO2 outlet
        glides down to T_water_in_c + pinch. Because the CO2 outlet chases the
        cold inlet (not the hot store), the cycle COP stays high -- this is
        exactly why transcritical CO2 excels at producing high-temperature
        water. The delivered heat charges the store from T_water_in_c up to the
        target, with the charge self-limiting smoothly as T_tank -> target.

        The high-side pressure is held at the optimum for the cold-inlet
        condition (Sarkar 2004 / Liao 2000). Returns a time series dict."""
        # Optimum (or supplied) high-side pressure for the gas-cooler outlet
        # condition, which is set by the fixed cold-water inlet.
        if P_high_bar is None:
            P_high_bar, _ = self.optimum_high_pressure_search(
                T_source_c, T_water_in_c)

        # Cycle is evaluated at the fixed cold inlet -> constant per-kg work/heat.
        st0 = self.cycle_states(T_source_c, T_water_in_c, P_high_bar)
        if m_dot_co2 is None:
            m_dot_co2 = self.Q_rated / st0["q_gc"]   # kg/s
        Q_gc_full = m_dot_co2 * st0["q_gc"]          # kW delivered to water
        W_elec = m_dot_co2 * st0["w_elec"] + self.aux_power
        cop_cycle = st0["cop"]

        mc = self.m_water * self.cp_water            # kJ/K

        def _taper(T_tank):
            # Charge self-limits smoothly as the store approaches the target
            # (thermostat / reduced flow). Avoids solver overshoot, stays
            # physical: deliverable heat -> 0 as T_tank -> target.
            gap = T_water_target_c - T_tank
            if gap <= 0.0:
                return 0.0
            return min(1.0, gap / max(self.pinch_gc, 1e-6))

        def rhs(t, y):
            T_tank = y[0]
            Q_gc = Q_gc_full * _taper(T_tank)        # kW available
            dT = (Q_gc - Q_load_kW) / mc
            return [dT]

        t_eval = np.linspace(0.0, duration_s, n_eval)
        sol = solve_ivp(rhs, (0.0, duration_s), [T_water_in_c],
                        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

        T_tank = sol.y[0]
        taper = np.array([_taper(tt) for tt in T_tank])
        Qgc_t = Q_gc_full * taper
        # Instantaneous COP equals the (constant) cycle COP while charging; it
        # is reported as the cycle COP throughout (store warming does not change
        # the gas-cooler-outlet COP because the cold inlet is fixed).
        cop_t = np.full_like(T_tank, cop_cycle)
        Welec_t = np.where(taper > 0.0, W_elec, self.aux_power)

        return {
            "t": sol.t,
            "T_water": T_tank,
            "cop": cop_t,
            "Q_heat_kW": Qgc_t,
            "P_elec_kW": Welec_t,
            "P_high_bar": P_high_bar,
            "cop_cycle": cop_cycle,
            "m_dot_co2": m_dot_co2,
            "states0": st0,
        }
