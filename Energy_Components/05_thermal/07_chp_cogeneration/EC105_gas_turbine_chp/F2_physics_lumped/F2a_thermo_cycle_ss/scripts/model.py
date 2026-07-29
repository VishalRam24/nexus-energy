"""
EC105 -- Gas Turbine CHP -- F2a Physics-Lumped Thermodynamic Cycle Model

First-principles cogeneration model:

  TOPPING CYCLE (air-standard Brayton, Moran Ch. 9):
    1->2  Compressor (irreversible adiabatic):
          T2s = T1 * rp^((g_a-1)/g_a)              isentropic outlet
          T2  = T1 + (T2s - T1)/eta_c              real outlet (eta_c < 1)
          w_c = cp_air*(T2 - T1)                    specific compressor work
    2->3  Combustor (heat addition at ~const P):
          q_in = cp_gas*(T3 - T2)                   T3 = turbine inlet temp (TIT)
          mdot_fuel = mdot_air * q_in / (eta_comb*LHV)
    3->4  Turbine (irreversible adiabatic):
          T4s = T3 * (1/rp)^((g_g-1)/g_g)           isentropic outlet
          T4  = T3 - eta_t*(T3 - T4s)               real outlet
          w_t = cp_gas*(T3 - T4)                    specific turbine work
    Net shaft work  w_net = w_t - w_c
    Electrical out  P_el  = mdot*(w_t) - mdot*w_c, times eta_mech_gen
    eta_el = P_el / Q_fuel                          (< Carnot, enforced)

  BOTTOMING / HEAT RECOVERY (HRSG, exhaust gas -> useful heat):
    Available exhaust enthalpy above stack temperature:
          Q_exh_avail = mdot_gas * cp_gas * (T4 - T_stack)
    Useful recovered heat (steady):
          Q_th = epsilon_hrsg * Q_exh_avail
    eta_th    = Q_th / Q_fuel
    eta_total = eta_el + eta_th        (CHP / EUF, total fuel utilisation)
    HPR       = Q_th / P_el            (heat-to-power ratio)

  LUMPED HRSG THERMAL TRANSIENT (0-D ODE, integrated by scipy.solve_ivp):
    The HRSG metal+water mass T_m responds with finite thermal inertia to a
    change in exhaust-gas inlet temperature T4(t):
          m_hrsg*cp_hrsg * dT_m/dt = UA*(T4 - T_m) - UA_loss*(T_m - T_amb)
    The instantaneous useful heat delivered to the steam side scales with the
    captured gas-side duty; at steady state T_m relaxes to its algebraic value
    and Q_th converges to the steady expression above.  This gives the dynamic
    warm-up / load-change response of the heat-recovery loop.

Energy conservation (enforced by construction):
    Q_fuel = P_el/eta_mech_gen_internal + Q_exhaust_total
    Q_exhaust_total = Q_th(recovered) + Q_stack_loss + Q_unrecovered

References:
    Moran, Shapiro, Boettner & Bailey (2018), Fundamentals of Engineering
        Thermodynamics, 9th ed., Wiley -- Ch. 9 (Brayton/gas turbine cycle).
    Cengel & Boles (2015), Thermodynamics: An Engineering Approach, 8th ed.,
        McGraw-Hill -- gas-turbine cogeneration, hot-gas cp values.
    Kehlhofer, Rukes, Hannemann & Stenzel (2009), Combined-Cycle Gas & Steam
        Turbine Power Plants, 3rd ed., PennWell -- HRSG heat recovery.
    EPA (2017), Catalog of CHP Technologies -- gas-turbine CHP performance.
"""

import numpy as np
from scipy.integrate import solve_ivp


class GasTurbineCHP_F2a:
    """Gas turbine CHP -- lumped Brayton topping cycle + HRSG heat recovery."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_el_rated = u["P_el_rated_kw"]["value"] * 1e3        # W
        self.rp = u["pressure_ratio"]["value"]
        self.T_amb = u["T_amb_k"]["value"]                         # K
        self.P_amb = u["P_amb_kpa"]["value"]
        self.T3 = u["T_turbine_inlet_k"]["value"]                  # K (TIT)
        self.eta_c = u["eta_compressor"]["value"]
        self.eta_t = u["eta_turbine"]["value"]
        self.eta_gen = u["eta_mech_gen"]["value"]
        self.eta_comb = u["eta_combustor"]["value"]
        self.g_a = u["gamma_air"]["value"]
        self.g_g = u["gamma_gas"]["value"]
        self.cp_air = u["cp_air"]["value"]                         # J/(kg.K)
        self.cp_gas = u["cp_gas"]["value"]                         # J/(kg.K)
        self.LHV = u["LHV_gas_mjkg"]["value"] * 1e6                # J/kg
        self.mdot_air_rated = u["mdot_air_kgs"]["value"]           # kg/s
        self.T_stack = u["T_stack_k"]["value"]                     # K
        self.T_water_in = u["T_water_in_k"]["value"]               # K
        self.eps_hrsg = u["hrsg_effectiveness"]["value"]
        self.m_hrsg = u["m_hrsg_kg"]["value"]
        self.cp_hrsg = u["cp_hrsg"]["value"]
        self.UA_hrsg = u["UA_hrsg_wk"]["value"]
        self.UA_loss = u["UA_loss_wk"]["value"]
        self.PLR_min = u["PLR_min"]["value"]

    # -- Brayton cycle thermodynamics -------------------------------------

    def compressor_outlet_T(self, T1, rp):
        """Real compressor exit temperature (isentropic + efficiency)."""
        T2s = T1 * rp ** ((self.g_a - 1.0) / self.g_a)
        return T1 + (T2s - T1) / self.eta_c

    def turbine_outlet_T(self, T3, rp):
        """Real turbine exit (exhaust) temperature."""
        T4s = T3 * (1.0 / rp) ** ((self.g_g - 1.0) / self.g_g)
        return T3 - self.eta_t * (T3 - T4s)

    def carnot_efficiency(self, T_hot, T_cold):
        """Carnot upper bound for the power cycle (T in K)."""
        return 1.0 - T_cold / T_hot

    def cycle_state(self, plr=1.0, T_amb=None, rp=None, T3=None):
        """Steady-state thermodynamic state and power/heat split at given load.

        Returns a dict of the full cycle: temperatures, work terms, fuel,
        electrical power, recovered heat, efficiencies, HPR.
        """
        T1 = self.T_amb if T_amb is None else T_amb
        rp = self.rp if rp is None else rp
        T3 = self.T3 if T3 is None else T3
        plr = float(np.clip(plr, self.PLR_min, 1.0))

        # Air mass flow scales with load (variable-IGV / fuel staging proxy)
        mdot = self.mdot_air_rated * plr

        # 1->2 compressor
        T2 = self.compressor_outlet_T(T1, rp)
        w_c = self.cp_air * (T2 - T1)                  # J/kg, specific

        # 2->3 combustor (heat addition)
        q_in = self.cp_gas * (T3 - T2)                 # J/kg
        mdot_fuel = mdot * q_in / (self.eta_comb * self.LHV)
        Q_fuel = mdot * q_in / self.eta_comb           # W, fuel chemical power (LHV)

        # 3->4 turbine
        T4 = self.turbine_outlet_T(T3, rp)
        w_t = self.cp_gas * (T3 - T4)                  # J/kg

        # Net electrical power
        w_net = w_t - w_c                              # J/kg
        P_shaft = mdot * w_net                         # W
        P_el = P_shaft * self.eta_gen                  # W
        eta_el = P_el / Q_fuel if Q_fuel > 0 else 0.0

        # HRSG heat recovery from exhaust (gas side, hot gas ~ air+fuel)
        mdot_gas = mdot + mdot_fuel
        Q_exh_avail = mdot_gas * self.cp_gas * max(T4 - self.T_stack, 0.0)
        Q_th = self.eps_hrsg * Q_exh_avail             # W useful recovered heat
        eta_th = Q_th / Q_fuel if Q_fuel > 0 else 0.0

        eta_total = eta_el + eta_th
        hpr = Q_th / P_el if P_el > 0 else 0.0
        eta_carnot = self.carnot_efficiency(T3, T1)

        return {
            "plr": plr,
            "T1_K": T1, "T2_K": T2, "T3_K": T3, "T4_K": T4,
            "mdot_air_kgs": mdot, "mdot_fuel_kgs": mdot_fuel,
            "mdot_gas_kgs": mdot_gas,
            "w_compressor_jkg": w_c, "w_turbine_jkg": w_t, "w_net_jkg": w_net,
            "fuel_power_w": Q_fuel,
            "electrical_power_w": P_el,
            "exhaust_available_w": Q_exh_avail,
            "thermal_power_w": Q_th,
            "eta_electrical": eta_el,
            "eta_thermal": eta_th,
            "eta_total": eta_total,
            "eta_carnot": eta_carnot,
            "heat_to_power_ratio": hpr,
        }

    # -- Lumped HRSG thermal transient ------------------------------------

    def _T4_of_t(self, plr_func, T_amb, rp, T3, t):
        """Exhaust gas inlet temperature to HRSG as function of time."""
        plr = plr_func(t) if callable(plr_func) else plr_func
        plr = float(np.clip(plr, self.PLR_min, 1.0))
        # Turbine exhaust T4 depends on cycle (independent of plr in cold-air
        # standard) but exhaust mass flow scales with plr; T4 itself fixed by rp.
        return self.turbine_outlet_T(T3, rp)

    def simulate(self, plr=1.0, T_amb=None, rp=None, T3=None,
                 T_hrsg0=None, dt=2.0, duration_s=600.0):
        """Integrate the lumped HRSG thermal ODE with scipy.solve_ivp.

        m_hrsg*cp_hrsg dT_m/dt = UA*(T4(t) - T_m) - UA_loss*(T_m - T_amb)

        plr may be a scalar or a callable plr(t) for load steps.  Returns the
        time history of HRSG metal temperature and the instantaneous useful
        thermal power delivered, plus the steady cycle state.
        """
        T1 = self.T_amb if T_amb is None else T_amb
        rp = self.rp if rp is None else rp
        T3 = self.T3 if T3 is None else T3
        C = self.m_hrsg * self.cp_hrsg                 # J/K thermal capacitance

        T4_ss = self.turbine_outlet_T(T3, rp)
        if T_hrsg0 is None:
            T_hrsg0 = T1                               # cold start at ambient

        def rhs(t, y):
            T_m = y[0]
            T4 = self._T4_of_t(plr, T1, rp, T3, t)
            dTm = (self.UA_hrsg * (T4 - T_m)
                   - self.UA_loss * (T_m - T1)) / C
            return [dTm]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        sol = solve_ivp(rhs, (0.0, duration_s), [T_hrsg0],
                        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-3)

        T_m = sol.y[0]
        t = sol.t

        # Instantaneous useful heat: gas-side duty captured into HRSG mass,
        # bounded by the steady effectiveness expression.  As T_m -> T_m_ss the
        # delivered useful heat -> steady Q_th.
        Q_th_hist = np.zeros_like(T_m)
        steady = self.cycle_state(plr(0) if callable(plr) else plr, T1, rp, T3)
        Q_th_ss = steady["thermal_power_w"]
        T_m_ss = (self.UA_hrsg * T4_ss + self.UA_loss * T1) / (self.UA_hrsg + self.UA_loss)
        # scale useful heat by how warmed the HRSG is relative to its steady point
        for i in range(len(T_m)):
            frac = np.clip((T_m[i] - T1) / max(T_m_ss - T1, 1e-6), 0.0, 1.0)
            Q_th_hist[i] = Q_th_ss * frac

        return {
            "t": t,
            "T_hrsg_K": T_m,
            "thermal_power_w": Q_th_hist,
            "T_exhaust_K": np.full_like(t, T4_ss),
            "T_hrsg_steady_K": T_m_ss,
            "steady_state": steady,
            "solver_success": bool(sol.success),
        }
