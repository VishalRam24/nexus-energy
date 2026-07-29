"""
EC138 -- Ocean Thermal Energy Conversion (OTEC) -- F2a Physics-Lumped Thermal Cycle

Closed-cycle (ammonia Rankine / Anderson cycle) heat engine driven by the small
temperature difference between warm tropical surface seawater (~25 degC) and cold
deep seawater (~5 degC) pumped from ~1000 m depth.

Cycle (closed loop, NH3 working fluid):
    1->2  feed pump          : saturated liquid pressurised p_cond -> p_evap
    2->3  evaporator (UA)     : warm seawater boils NH3 (heat in,  Q_evap)
    3->4  turbine            : saturated vapour expands p_evap -> p_cond (work out)
    4->1  condenser (UA)      : cold seawater condenses NH3 (heat out, Q_cond)

Because the warm/cold reservoir gap is only ~20 K, the Carnot ceiling is tiny:
    eta_Carnot = 1 - T_cold_K / T_warm_K   ~  0.06-0.07
and the *usable* gap across the working fluid is smaller still after the HX pinch,
so net efficiency after the (very large) seawater pumping parasitic is ~2-3 %.

LUMPED THERMAL ODEs (state = working-fluid saturation temperatures in the two HX):
Each heat exchanger is a single lumped node with a thermal capacitance C and an
effectiveness-NTU coupling to its seawater stream and to the cycle heat duty.

    C_evap * dT_evap/dt = Q_warm_to_evap(T_evap) - Q_evap_duty
    C_cond * dT_cond/dt = Q_cond_duty            - Q_cond_to_cold(T_cond)

with seawater-side heat picked up via effectiveness-NTU (cross/counterflow lumped):
    Q_warm_to_evap = eps_e * Cmin_warm * (T_warm_in - T_evap)
    Q_cond_to_cold = eps_c * Cmin_cold * (T_cond   - T_cold_in)
    eps = 1 - exp(-NTU),  NTU = UA / (mdot*cp)         (single-stream limit, Cr->0
          because the boiling/condensing NH3 side is isothermal => infinite C)

The cycle duties are tied to the gross power through the first law on the turbine:
    eta_cycle  = eta_carnot_fraction * (1 - T_cond / T_evap)     (in Kelvin)
    Q_evap_duty = P_gross / eta_cycle
    Q_cond_duty = Q_evap_duty - P_gross          (energy conservation, ignoring
                  the small pump enthalpy rise which is added back explicitly)

At steady state dT/dt -> 0 so the seawater heat delivered equals the cycle duty;
the transient ODE captures HX thermal-mass lag after a step in seawater inlet T or
flow. Integrated with scipy.integrate.solve_ivp (LSODA).

Parasitic loads (Vega 2002; Avery & Wu 1994):
    P_warm_pump = mdot_warm/rho * dP_warm / eta_sw_pump      (hydraulic power)
    P_cold_pump = mdot_cold/rho * dP_cold / eta_sw_pump
    P_wf_pump   = mdot_nh3 * v_liq * (p_evap - p_cond) / eta_wf_pump
    P_net       = P_gross - P_warm_pump - P_cold_pump - P_wf_pump

References:
    Avery, W.H. & Wu, C. (1994), Renewable Energy from the Ocean: A Guide to OTEC,
        Oxford University Press (closed-cycle ammonia plant design, Ch. 3-6).
    Vega, L.A. (2002/2012), Ocean Thermal Energy Conversion Primer,
        Marine Technology Society Journal, 36(4), 25-35 (parasitic budget).
    Nihous, G.C. (2007), J. Energy Resour. Technol., 129(1), 10-17 (Carnot fraction).
    Faizal, M. & Ahmed, M.R. (2011), Int. J. Low-Carbon Tech., 6, 215-226.
    Sharqawy, Lienhard & Zubair (2010), Desal. & Water Treat., 16, 354 (seawater cp).
"""

import numpy as np
from scipy.integrate import solve_ivp


class OTEC_F2a:
    """OTEC closed-cycle ammonia Rankine -- physics-lumped HX thermal model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_gross_design = u["P_gross_design_kw"]["value"]        # kW

        self.UA_evap = u["UA_evap_kw_per_k"]["value"]               # kW/K
        self.UA_cond = u["UA_cond_kw_per_k"]["value"]               # kW/K
        self.C_evap  = u["C_evap_kj_per_k"]["value"]                # kJ/K
        self.C_cond  = u["C_cond_kj_per_k"]["value"]                # kJ/K

        self.T_warm_in = u["T_warm_in_c"]["value"]                  # degC
        self.T_cold_in = u["T_cold_in_c"]["value"]                  # degC

        self.mdot_warm = u["mdot_warm_kg_s"]["value"]               # kg/s
        self.mdot_cold = u["mdot_cold_kg_s"]["value"]               # kg/s

        self.cp_sw  = u["cp_seawater_kj_per_kg_k"]["value"]         # kJ/(kg.K)
        self.rho_sw = u["rho_seawater_kg_per_m3"]["value"]          # kg/m3

        self.eta_turb     = u["eta_turbine"]["value"]
        self.eta_wf_pump  = u["eta_wf_pump"]["value"]
        self.eta_sw_pump  = u["eta_sw_pump"]["value"]
        self.eta_carnot_f = u["eta_carnot_fraction"]["value"]

        self.dP_warm = u["dP_warm_pump_kpa"]["value"]               # kPa
        self.dP_cold = u["dP_cold_pump_kpa"]["value"]               # kPa

        self.p_evap = u["p_evap_kpa"]["value"]                      # kPa
        self.p_cond = u["p_cond_kpa"]["value"]                      # kPa
        self.h_fg   = u["h_fg_nh3_kj_per_kg"]["value"]              # kJ/kg
        self.v_liq  = u["v_liq_nh3_m3_per_kg"]["value"]             # m3/kg

    # ------------------------------------------------------------------
    # Thermodynamic limits
    # ------------------------------------------------------------------
    @staticmethod
    def carnot_efficiency(T_warm_c, T_cold_c):
        """Carnot ceiling for the reservoir temperatures [-]."""
        Tw = np.asarray(T_warm_c, float) + 273.15
        Tc = np.asarray(T_cold_c, float) + 273.15
        return np.where(Tw > Tc, 1.0 - Tc / Tw, 0.0)

    def cycle_efficiency(self, T_evap_c, T_cond_c):
        """Real ammonia-Rankine thermal efficiency across the HX saturation gap.

        Uses the working-fluid saturation temperatures (after HX pinch), so it is
        strictly below the reservoir Carnot ceiling. eta = f_Carnot * (1 - Tc/Te).
        """
        Te = np.asarray(T_evap_c, float) + 273.15
        Tc = np.asarray(T_cond_c, float) + 273.15
        eta_c_internal = np.where(Te > Tc, 1.0 - Tc / Te, 0.0)
        return self.eta_carnot_f * eta_c_internal

    # ------------------------------------------------------------------
    # Effectiveness-NTU (single isothermal-WF-side stream, Cr -> 0)
    # ------------------------------------------------------------------
    @staticmethod
    def effectiveness(UA, mdot, cp):
        """Effectiveness of an HX whose other side (boiling/condensing) is
        isothermal => Cr=0 => eps = 1 - exp(-NTU), NTU = UA/(mdot*cp).
        Incropera & DeWitt (2007), Fundamentals of Heat & Mass Transfer, Tab 11.4."""
        C = max(mdot * cp, 1e-9)            # kW/K
        NTU = UA / C
        return 1.0 - np.exp(-NTU)

    # ------------------------------------------------------------------
    # Parasitic pumping
    # ------------------------------------------------------------------
    def seawater_pumping_kw(self, mdot_warm=None, mdot_cold=None):
        """Hydraulic seawater pumping power [kW]. P = Vdot*dP/eta."""
        mw = self.mdot_warm if mdot_warm is None else mdot_warm
        mc = self.mdot_cold if mdot_cold is None else mdot_cold
        Vdot_w = mw / self.rho_sw                      # m3/s
        Vdot_c = mc / self.rho_sw
        P_warm = Vdot_w * self.dP_warm / self.eta_sw_pump   # m3/s * kPa = kW
        P_cold = Vdot_c * self.dP_cold / self.eta_sw_pump
        return P_warm, P_cold

    def wf_pump_kw(self, Q_evap_kw):
        """Ammonia feed-pump power [kW] from incompressible pump work.
        mdot_nh3 = Q_evap / h_fg ; w_pump = v_liq*(p_evap-p_cond)/eta."""
        mdot_nh3 = Q_evap_kw / self.h_fg                    # kg/s (Q in kW=kJ/s)
        w_pump = self.v_liq * (self.p_evap - self.p_cond) / self.eta_wf_pump  # kJ/kg
        return mdot_nh3 * w_pump, mdot_nh3                  # kW, kg/s

    # ------------------------------------------------------------------
    # Steady-state operating point (used as ODE target & for predict)
    # ------------------------------------------------------------------
    def steady_state(self, T_warm_c=None, T_cold_c=None,
                     mdot_warm=None, mdot_cold=None, pinch_c=2.5):
        """Solve the steady cycle.

        The working-fluid saturation temperatures sit inside the reservoir gap by
        the HX pinch:  T_evap = T_warm_out_side - pinch,  T_cond = T_cold_out_side
        + pinch.  We approximate the WF sat temps from the seawater inlet temps and
        the effectiveness-derived outlet temps.
        """
        Tw = self.T_warm_in if T_warm_c is None else T_warm_c
        Tc = self.T_cold_in if T_cold_c is None else T_cold_c
        mw = self.mdot_warm if mdot_warm is None else mdot_warm
        mc = self.mdot_cold if mdot_cold is None else mdot_cold

        eps_e = self.effectiveness(self.UA_evap, mw, self.cp_sw)
        eps_c = self.effectiveness(self.UA_cond, mc, self.cp_sw)

        # WF saturation temps (degC): inside the gap by the pinch
        T_evap = Tw - pinch_c
        T_cond = Tc + pinch_c

        eta_cycle = float(self.cycle_efficiency(T_evap, T_cond))

        # Available heat the warm stream can give down to T_evap (kW)
        Q_evap_avail = eps_e * (mw * self.cp_sw) * max(Tw - T_evap, 0.0)
        # Available heat the cold stream can absorb up to T_cond (kW)
        Q_cond_avail = eps_c * (mc * self.cp_sw) * max(T_cond - Tc, 0.0)

        # Gross power limited by evaporator heat and cycle efficiency
        P_gross = Q_evap_avail * eta_cycle if eta_cycle > 0 else 0.0
        # Cap by design rating (turbine size)
        P_gross = min(P_gross, self.P_gross_design)
        Q_evap = P_gross / eta_cycle if eta_cycle > 0 else 0.0
        Q_cond = Q_evap - P_gross                          # energy balance

        P_warm_pump, P_cold_pump = self.seawater_pumping_kw(mw, mc)
        P_wf_pump, mdot_nh3 = self.wf_pump_kw(Q_evap)
        P_parasitic = P_warm_pump + P_cold_pump + P_wf_pump
        P_net = P_gross - P_parasitic

        eta_net = P_net / Q_evap if Q_evap > 0 else 0.0

        return {
            "eta_carnot":   float(self.carnot_efficiency(Tw, Tc)),
            "eta_cycle":    eta_cycle,
            "eta_net":      eta_net,
            "T_evap_c":     T_evap,
            "T_cond_c":     T_cond,
            "Q_evap_kw":    Q_evap,
            "Q_cond_kw":    Q_cond,
            "P_gross_kw":   P_gross,
            "P_net_kw":     P_net,
            "P_parasitic_kw": P_parasitic,
            "P_warm_pump_kw": P_warm_pump,
            "P_cold_pump_kw": P_cold_pump,
            "P_wf_pump_kw":   P_wf_pump,
            "mdot_nh3_kg_s":  mdot_nh3,
            "eps_evap":     float(eps_e),
            "eps_cond":     float(eps_c),
        }

    # ------------------------------------------------------------------
    # Transient lumped HX thermal ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, Tw_func, Tc_func, mw, mc):
        """ODE right-hand side. y = [T_evap_c, T_cond_c]."""
        T_evap, T_cond = y
        Tw = Tw_func(t)
        Tc = Tc_func(t)

        Cmin_w = mw * self.cp_sw                            # kW/K
        Cmin_c = mc * self.cp_sw
        eps_e = self.effectiveness(self.UA_evap, mw, self.cp_sw)
        eps_c = self.effectiveness(self.UA_cond, mc, self.cp_sw)

        # Heat the seawater streams exchange with the WF nodes (kW)
        Q_warm_to_evap = eps_e * Cmin_w * (Tw - T_evap)
        Q_cond_to_cold = eps_c * Cmin_c * (T_cond - Tc)

        # Instantaneous cycle duties from current WF sat temps
        eta_cycle = float(self.cycle_efficiency(T_evap, T_cond))
        if eta_cycle > 1e-6 and Q_warm_to_evap > 0:
            P_gross = min(Q_warm_to_evap * eta_cycle, self.P_gross_design)
            Q_evap_duty = P_gross / eta_cycle
            Q_cond_duty = Q_evap_duty - P_gross
        else:
            Q_evap_duty = 0.0
            Q_cond_duty = 0.0

        # Lumped energy balance on each HX node (C in kJ/K, Q in kW=kJ/s)
        dT_evap = (Q_warm_to_evap - Q_evap_duty) / self.C_evap
        dT_cond = (Q_cond_duty - Q_cond_to_cold) / self.C_cond
        return [dT_evap, dT_cond]

    def simulate(self, T_warm_c=None, T_cold_c=None,
                 mdot_warm=None, mdot_cold=None,
                 dt=10.0, duration_s=3600.0, y0=None):
        """Integrate the lumped HX thermal ODE with scipy.solve_ivp.

        T_warm_c / T_cold_c may be scalars or callables of time (degC).
        Returns time series of WF sat temps, efficiencies and net power.
        """
        Tw0 = self.T_warm_in if T_warm_c is None else T_warm_c
        Tc0 = self.T_cold_in if T_cold_c is None else T_cold_c
        mw = self.mdot_warm if mdot_warm is None else mdot_warm
        mc = self.mdot_cold if mdot_cold is None else mdot_cold

        Tw_func = Tw0 if callable(Tw0) else (lambda t, v=Tw0: v)
        Tc_func = Tc0 if callable(Tc0) else (lambda t, v=Tc0: v)

        if y0 is None:
            # init WF temps at a reasonable point inside the gap
            T_evap0 = Tw_func(0.0) - 4.0
            T_cond0 = Tc_func(0.0) + 4.0
            y0 = [T_evap0, T_cond0]

        n = max(int(duration_s / dt) + 1, 2)
        t_eval = np.linspace(0.0, duration_s, n)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="LSODA",
            args=(Tw_func, Tc_func, mw, mc),
            rtol=1e-6, atol=1e-8, max_step=dt,
        )

        T_evap = sol.y[0]
        T_cond = sol.y[1]
        Tw_arr = np.array([Tw_func(t) for t in sol.t])
        Tc_arr = np.array([Tc_func(t) for t in sol.t])

        eta_carnot = self.carnot_efficiency(Tw_arr, Tc_arr)
        eta_cycle = self.cycle_efficiency(T_evap, T_cond)

        Cmin_w = mw * self.cp_sw
        eps_e = self.effectiveness(self.UA_evap, mw, self.cp_sw)
        Q_warm_to_evap = eps_e * Cmin_w * (Tw_arr - T_evap)
        Q_warm_to_evap = np.maximum(Q_warm_to_evap, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            P_gross = np.where(eta_cycle > 1e-6,
                               np.minimum(Q_warm_to_evap * eta_cycle,
                                          self.P_gross_design),
                               0.0)
            Q_evap = np.where(eta_cycle > 1e-6, P_gross / eta_cycle, 0.0)

        P_warm_pump, P_cold_pump = self.seawater_pumping_kw(mw, mc)
        P_wf_pump = np.array([self.wf_pump_kw(q)[0] for q in Q_evap])
        P_parasitic = P_warm_pump + P_cold_pump + P_wf_pump
        P_net = P_gross - P_parasitic
        with np.errstate(divide="ignore", invalid="ignore"):
            eta_net = np.where(Q_evap > 0, P_net / Q_evap, 0.0)

        return {
            "t": sol.t,
            "T_warm_in_c": Tw_arr,
            "T_cold_in_c": Tc_arr,
            "T_evap_c": T_evap,
            "T_cond_c": T_cond,
            "eta_carnot": eta_carnot,
            "eta_cycle": eta_cycle,
            "eta_net": eta_net,
            "Q_evap_kw": Q_evap,
            "P_gross_kw": P_gross,
            "P_parasitic_kw": np.full_like(P_gross, 0.0) + P_parasitic,
            "P_net_kw": P_net,
            "success": sol.success,
        }
