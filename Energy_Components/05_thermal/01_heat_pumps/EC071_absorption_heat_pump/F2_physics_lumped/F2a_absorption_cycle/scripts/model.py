"""
EC071 -- Absorption Heat Pump (LiBr-H2O, single-effect) -- F2a Physics-Lumped

First-principles single-effect Type-I absorption heat pump.  Four heat-and-mass
exchangers -- GENERATOR (desorber), CONDENSER, EVAPORATOR, ABSORBER -- linked by
a solution circuit with a solution heat exchanger (SHX).  The cycle is HEAT
DRIVEN: the generator receives high-grade heat Q_gen, the evaporator lifts
low-grade heat Q_evap, and the useful heat Q_heat = Q_cond + Q_abs is delivered
at intermediate temperature.  Thermal COP (heating) = Q_heat / Q_gen lies in the
physical band ~1.5-1.8 for a Type-I single-effect machine (equivalently the
cooling COP Q_evap/Q_gen ~ 0.6-0.8, Herold et al. Table form).

Solution circuit (mass + species balance, Herold/Radermacher/Klein 2016 Ch.6):
    Mass:    m_weak  = m_strong + m_ref          (refrigerant vapour split off)
    LiBr:    m_weak*x_weak = m_strong*x_strong    (salt is conserved, non-volatile)
    => circulation ratio  f = m_weak / m_ref = x_strong / (x_strong - x_weak)

Component energy balances (per kg of refrigerant, steady state):
    Generator:  Q_gen  = h_ref_vap + (f-1)*h_strong - f*h_weak_hot
    Condenser:  Q_cond = h_ref_vap - h_ref_liq
    Evaporator: Q_evap = h_ref_vap_evap - h_ref_liq        (h_ref_liq throttled in)
    Absorber:   Q_abs  = f*h_weak + ... (closes 1st law: Q_gen+Q_evap = Q_cond+Q_abs)

Overall first law (must close):  Q_gen + Q_evap + W_pump = Q_cond + Q_abs.

Lumped TRANSIENT ODE (generator solution node, key loop temperature):
    m_sol * cp_sol * dT_gen/dt = UA_gen*(T_drive - T_gen) - Q_desorb(T_gen)
integrated with scipy.integrate.solve_ivp.  Q_desorb is the heat consumed to
boil refrigerant out of solution, rising with T_gen (more desorption).

LiBr-H2O property correlations are HARDCODED (no CoolProp):
  * Equilibrium / Duhring vapour pressure  -- Patek & Klomfar (2006),
    Int. J. Refrigeration 29, 566-578, Eq.(2)-(3) (saturation T of solution).
  * Solution enthalpy h(T,x)               -- ASHRAE / Kaita (2001) polynomial
    fit, also reproduced in Herold et al. (2016) App.
  * Water/steam enthalpies                 -- linear cp fits valid 0-120 C.

References
----------
Herold, K.E., Radermacher, R., Klein, S.A. (2016). "Absorption Chillers and
    Heat Pumps", 2nd ed., CRC Press.  (cycle model, circulation ratio, COP band)
Patek, J., Klomfar, J. (2006). Int. J. Refrigeration 29, 566-578.
    (LiBr-H2O equilibrium pressure / temperature correlation)
Kaita, Y. (2001). Int. J. Refrigeration 24, 374-390. (enthalpy correlation)
"""

import numpy as np
from scipy.integrate import solve_ivp


class AbsorptionHeatPumpF2a:
    """Single-effect LiBr-H2O absorption heat pump -- physics-lumped cycle."""

    # ---- water / steam reference property fits (0-150 C, 1 atm-ish) ----
    CP_WATER_LIQ = 4.18      # kJ/(kg.K)
    CP_VAPOR = 1.88          # kJ/(kg.K) superheated steam approx
    H_FG_0 = 2501.0          # kJ/kg latent heat of water at 0 C
    DHFG_DT = -2.37          # kJ/(kg.K) slope of h_fg with T (so h_fg(40C)~2406)

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_evap_design = u["Q_evap_design_kW"]["value"]   # kW
        self.x_weak = u["x_weak"]["value"]                    # mass frac LiBr
        self.x_strong = u["x_strong"]["value"]
        self.eps_shx = u["eps_shx"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.UA_gen = u["UA_gen"]["value"]                    # kW/K
        self.m_sol = u["m_gen_solution"]["value"]             # kg
        self.cp_sol = u["cp_solution"]["value"]               # kJ/(kg.K)
        self.T_drive = u["T_gen_drive_C"]["value"]            # degC
        self.T_evap = u["T_evap_C"]["value"]
        self.T_cond = u["T_cond_C"]["value"]
        self.T_abs = u["T_abs_C"]["value"]
        self.aux_power = u["aux_power_kW"]["value"]           # kW_e

    # ==================================================================
    # LiBr-H2O property correlations (HARDCODED, cited)
    # ==================================================================
    @staticmethod
    def solution_enthalpy(T_c, x):
        """
        Specific enthalpy of LiBr-H2O solution [kJ/kg].
        Polynomial fit (Kaita 2001; reproduced Herold et al. 2016, App.).
        Valid x in 0.40-0.70 (mass frac LiBr), T in 0-200 C.
        Reference state: liquid water and solid LiBr at 0 C.
        """
        X = np.asarray(x, dtype=float) * 100.0   # weight-percent LiBr
        T = np.asarray(T_c, dtype=float)
        # A,B,C coefficients (Kaita 2001 form: h = A + B*T + C*T^2)
        A = (-2024.33 + 163.309 * X - 4.88161 * X**2
             + 6.302948e-2 * X**3 - 2.913705e-4 * X**4)
        B = (18.2829 - 1.1691757 * X + 3.248041e-2 * X**2
             - 4.034184e-4 * X**3 + 1.8520569e-6 * X**4)
        C = (-3.7008214e-2 + 2.8877666e-3 * X - 8.1313015e-5 * X**2
             + 9.9116628e-7 * X**3 - 4.4441207e-9 * X**4)
        return A + B * T + C * T**2

    @classmethod
    def water_liquid_enthalpy(cls, T_c):
        """Saturated/subcooled liquid water enthalpy [kJ/kg], ref 0 C."""
        return cls.CP_WATER_LIQ * np.asarray(T_c, dtype=float)

    @classmethod
    def water_vapor_enthalpy(cls, T_c):
        """Water vapour enthalpy [kJ/kg], ref liquid water at 0 C."""
        T = np.asarray(T_c, dtype=float)
        h_fg = cls.H_FG_0 + cls.DHFG_DT * T
        return cls.CP_WATER_LIQ * T + h_fg

    @staticmethod
    def equilibrium_temperature(T_ref_c, x):
        """
        Solution temperature [C] in equilibrium with water vapour whose
        saturation (dew) temperature is T_ref_c, at LiBr mass fraction x.
        Duhring-line form, Patek & Klomfar (2006) Eq.(2):
            T_sol = sum_i a_i * x_pc^m_i * (T_ref)^n_i ... -> simplified linear
        We use the classic Duhring linearisation (McNeely 1979 / Herold 2016):
            T_sol = B(x)*T_ref + A(x)
        with A,B polynomials in weight-percent LiBr.
        """
        X = np.asarray(x, dtype=float) * 100.0
        Tref = np.asarray(T_ref_c, dtype=float)
        # McNeely (1979) Duhring coefficients (Herold et al. 2016, Ch.6;
        # ASHRAE Fundamentals validated fit):
        A = (124.937 - 7.71649 * X + 0.152286 * X**2
             - 7.95090e-4 * X**3)
        Bc = (-2.00755 + 0.16976 * X - 3.13339e-3 * X**2
              + 1.97668e-5 * X**3)
        return Bc * Tref + A

    # ==================================================================
    # Solution circuit: mass + species balance
    # ==================================================================
    def circulation_ratio(self, x_weak=None, x_strong=None):
        """
        f = m_weak / m_refrigerant  (mass of weak solution circulated per
        unit mass of refrigerant desorbed).  From LiBr conservation:
            f = x_strong / (x_strong - x_weak)
        Herold/Radermacher/Klein (2016) Eq.(6.x).
        """
        xw = self.x_weak if x_weak is None else x_weak
        xs = self.x_strong if x_strong is None else x_strong
        dx = xs - xw
        if dx <= 1e-6:
            return 1e6
        return xs / dx

    def check_mass_balance(self, x_weak=None, x_strong=None):
        """Return (salt_in, salt_out) per kg refrigerant -- must be equal."""
        xw = self.x_weak if x_weak is None else x_weak
        xs = self.x_strong if x_strong is None else x_strong
        f = self.circulation_ratio(xw, xs)
        salt_weak = f * xw          # LiBr entering generator (weak stream)
        salt_strong = (f - 1.0) * xs  # LiBr leaving generator (strong stream)
        return salt_weak, salt_strong

    # ==================================================================
    # Steady-state cycle (energy balances per kg refrigerant)
    # ==================================================================
    def steady_cycle(self, T_gen_c=None, T_evap_c=None, T_cond_c=None,
                     T_abs_c=None, x_weak=None, x_strong=None):
        """
        Solve the four-component single-effect cycle per kg of refrigerant.

        Returns dict of specific duties [kJ/kg ref] and COPs.  All four
        component balances enforced; overall 1st law closed to W_pump.
        """
        Tg = self.T_drive if T_gen_c is None else T_gen_c
        Te = self.T_evap if T_evap_c is None else T_evap_c
        Tc = self.T_cond if T_cond_c is None else T_cond_c
        Ta = self.T_abs if T_abs_c is None else T_abs_c
        xw = self.x_weak if x_weak is None else x_weak
        xs = self.x_strong if x_strong is None else x_strong

        f = self.circulation_ratio(xw, xs)

        # --- refrigerant (pure water) states ---
        # Vapour leaves generator at ~T_gen (superheated relative to T_cond)
        h_ref_gen_vap = self.water_vapor_enthalpy(Tg)
        # Condensate leaves condenser as sat. liquid at T_cond
        h_ref_cond_liq = self.water_liquid_enthalpy(Tc)
        # Throttled to evaporator pressure, evaporates at T_evap
        h_ref_evap_vap = self.water_vapor_enthalpy(Te)

        # --- solution states ---
        # Weak solution leaves absorber at T_abs, x_weak
        h_weak_abs = self.solution_enthalpy(Ta, xw)
        # Strong solution leaves generator at T_gen, x_strong
        h_strong_gen = self.solution_enthalpy(Tg, xs)

        # --- solution heat exchanger (SHX) ---
        # Strong (hot) solution preheats weak (cold) solution going to generator.
        # eps based on the smaller stream heat-capacity rate; strong has (f-1) flow.
        cp_w = self._solution_cp(Ta, xw)
        cp_s = self._solution_cp(Tg, xs)
        C_weak = f * cp_w
        C_strong = (f - 1.0) * cp_s
        Cmin = min(C_weak, C_strong)
        Q_shx_max = Cmin * (Tg - Ta)
        Q_shx = self.eps_shx * Q_shx_max          # kJ/kg ref recovered
        # Weak solution enters generator preheated:
        T_weak_hot = Ta + Q_shx / max(C_weak, 1e-9)
        h_weak_hot = self.solution_enthalpy(T_weak_hot, xw)
        # Strong solution cooled before absorber:
        T_strong_cold = Tg - Q_shx / max(C_strong, 1e-9)
        h_strong_cold = self.solution_enthalpy(T_strong_cold, xs)

        # --- pump work (liquid, small) ---
        # negligible enthalpy rise; track for closure only
        w_pump = 0.001 * f / max(self.eta_pump, 1e-3)   # kJ/kg ref (tiny)

        # --- GENERATOR balance (per kg ref) ---
        # in: weak sol (f, preheated); out: vapour (1) + strong sol (f-1)
        Q_gen = (h_ref_gen_vap + (f - 1.0) * h_strong_gen
                 - f * h_weak_hot)

        # --- CONDENSER ---
        Q_cond = h_ref_gen_vap - h_ref_cond_liq

        # --- EVAPORATOR ---
        # refrigerant throttled from sat liquid @T_cond to T_evap, evaporates
        Q_evap = h_ref_evap_vap - h_ref_cond_liq

        # --- ABSORBER (close by 1st law) ---
        # in: vapour from evap (1) + cooled strong sol (f-1); out: weak sol (f)
        Q_abs = (h_ref_evap_vap + (f - 1.0) * h_strong_cold
                 - f * h_weak_abs)

        # Useful heating output (Type-I AHP delivers cond + abs heat)
        Q_heat = Q_cond + Q_abs

        # COPs
        cop_cool = Q_evap / Q_gen if Q_gen > 1e-9 else 0.0
        cop_heat = Q_heat / Q_gen if Q_gen > 1e-9 else 0.0

        # 1st-law residual (should be ~0): Q_gen+Q_evap+w_pump - Q_cond - Q_abs
        residual = Q_gen + Q_evap + w_pump - Q_cond - Q_abs

        return {
            "f_circulation": f,
            "q_gen": Q_gen,
            "q_cond": Q_cond,
            "q_evap": Q_evap,
            "q_abs": Q_abs,
            "q_heat": Q_heat,
            "w_pump": w_pump,
            "cop_cooling": cop_cool,
            "cop_heating": cop_heat,
            "energy_residual": residual,
            "T_weak_preheat_C": float(T_weak_hot),
            "T_strong_cooled_C": float(T_strong_cold),
        }

    def _solution_cp(self, T_c, x):
        """Numerical cp of solution [kJ/(kg.K)] from enthalpy derivative."""
        h1 = self.solution_enthalpy(T_c - 0.5, x)
        h2 = self.solution_enthalpy(T_c + 0.5, x)
        return float(h2 - h1)

    # ==================================================================
    # Absolute (kW) duties scaled to design evaporator load
    # ==================================================================
    def rate_duties(self, plr=1.0, **kw):
        """
        Scale the per-kg cycle to the design evaporator duty * part-load ratio.
        Returns absolute heat rates [kW] and electrical input.
        """
        cyc = self.steady_cycle(**kw)
        Q_evap_target = self.Q_evap_design * float(plr)
        # mass flow of refrigerant [kg/s] to hit the target evaporator duty
        m_ref = Q_evap_target / max(cyc["q_evap"], 1e-9)
        out = {
            "m_ref_kg_s": m_ref,
            "Q_gen_kW": m_ref * cyc["q_gen"],
            "Q_cond_kW": m_ref * cyc["q_cond"],
            "Q_evap_kW": m_ref * cyc["q_evap"],
            "Q_abs_kW": m_ref * cyc["q_abs"],
            "Q_heat_kW": m_ref * cyc["q_heat"],
            "W_pump_kW": m_ref * cyc["w_pump"],
            "P_aux_kW": self.aux_power * (1.0 if plr > 0 else 0.0),
            "cop_cooling": cyc["cop_cooling"],
            "cop_heating": cyc["cop_heating"],
            "f_circulation": cyc["f_circulation"],
        }
        return out

    # ==================================================================
    # Lumped transient ODE: generator solution temperature
    # ==================================================================
    def _q_desorb_kw(self, T_gen_c):
        """
        Desorption heat sink [kW] consumed boiling refrigerant out of solution
        at generator temperature T_gen.  Rises with T_gen (Duhring: more
        superheat above the equilibrium boiling point -> faster desorption).
        Saturates at the design generator duty.
        """
        # Desorption onset = equilibrium boiling temperature of the WEAK
        # (incoming) solution at condenser pressure.  Desorption proceeds as the
        # solution is heated from this onset up toward the strong-solution
        # boiling point; using the weak point gives the generator real headroom
        # (Herold et al. 2016, generator inlet condition).
        T_boil = self.equilibrium_temperature(self.T_cond, self.x_weak)
        drive = T_gen_c - T_boil
        if drive <= 0:
            return 0.0
        # design generator duty (per the steady cycle at design point)
        Q_gen_design = self.rate_duties()["Q_gen_kW"]
        # first-order approach: desorption proportional to superheat, capped
        k = Q_gen_design / max(self.T_drive - T_boil, 1.0)
        return min(k * drive, 1.5 * Q_gen_design)

    def dTdt(self, T_gen_c, T_drive_c=None):
        """Generator solution-node temperature rate [C/s]."""
        Td = self.T_drive if T_drive_c is None else T_drive_c
        Q_in = self.UA_gen * (Td - T_gen_c)          # kW from driving fluid
        Q_out = self._q_desorb_kw(T_gen_c)            # kW into desorption
        C = self.m_sol * self.cp_sol                  # kJ/K
        return (Q_in - Q_out) / max(C, 1e-9)

    def simulate(self, T_drive_c=None, T_gen0_c=None, dt=5.0, duration_s=1800.0):
        """
        Transient warm-up of the generator solution loop via solve_ivp.

        Parameters
        ----------
        T_drive_c : float or callable(t)  driving heat-source temperature [C]
        T_gen0_c  : float                 initial generator solution temp [C]
        dt        : float                 output time step [s]
        duration_s: float                 total duration [s]

        Returns dict of time-series: t, T_gen_C, Q_gen_kW, Q_evap_kW,
        Q_heat_kW, cop_heating, cop_cooling.
        """
        Td_fun = (T_drive_c if callable(T_drive_c)
                  else (lambda t: (self.T_drive if T_drive_c is None else T_drive_c)))
        T0 = (self.T_abs if T_gen0_c is None else T_gen0_c)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], Td_fun(t))]

        sol = solve_ivp(rhs, (0.0, duration_s), [T0], t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=dt)

        t_out = sol.t
        Tg = sol.y[0]
        N = len(t_out)
        Q_gen = np.zeros(N); Q_evap = np.zeros(N); Q_heat = np.zeros(N)
        cop_h = np.zeros(N); cop_c = np.zeros(N)
        for i in range(N):
            d = self.rate_duties(T_gen_c=float(Tg[i]))
            Q_gen[i] = d["Q_gen_kW"]
            Q_evap[i] = d["Q_evap_kW"]
            Q_heat[i] = d["Q_heat_kW"]
            cop_h[i] = d["cop_heating"]
            cop_c[i] = d["cop_cooling"]

        return {
            "t": t_out,
            "T_gen_C": Tg,
            "Q_gen_kW": Q_gen,
            "Q_evap_kW": Q_evap,
            "Q_heat_kW": Q_heat,
            "cop_heating": cop_h,
            "cop_cooling": cop_c,
        }
