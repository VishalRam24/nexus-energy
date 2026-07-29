"""
EC098 -- Organic Rankine Cycle (ORC) -- F2a Thermodynamic Cycle Steady-State

Physics-lumped model: Rankine cycle with R245fa working fluid.
4 state points: pump -> evaporator -> expander -> condenser.

Polynomial property correlations for R245fa (no CoolProp dependency).
Includes part-load with expander efficiency correction.

State points:
    1: Pump inlet (subcooled liquid after condenser)
    2: Pump outlet / Evaporator inlet (compressed liquid)
    3: Expander inlet (superheated vapor after evaporator)
    4: Expander outlet / Condenser inlet (low-pressure vapor)

References:
    Quoilin et al. (2013), Renewable and Sustainable Energy Reviews, 22, 168-186
    Lemort et al. (2009), Applied Thermal Engineering, 29, 1684-1694
    NIST WebBook for R245fa property data
"""

import numpy as np
from scipy.integrate import solve_ivp


class R245faProperties:
    """
    Polynomial correlations for R245fa (1,1,1,3,3-pentafluoropropane).

    Properties fitted from NIST data over 280-430 K range.
    All pressures in Pa, temperatures in K, enthalpies in J/kg, entropies in J/(kg.K).
    """

    # Critical point
    T_crit = 427.16  # K
    P_crit = 3651000.0  # Pa
    M = 0.13404  # kg/mol (molar mass)

    # Saturation pressure via Clausius-Clapeyron-derived correlation.
    # Fitted to NIST R245fa data points:
    #   T=280K -> P~72 kPa, T=300K -> P~153 kPa, T=320K -> P~298 kPa,
    #   T=340K -> P~535 kPa, T=360K -> P~893 kPa, T=380K -> P~1403 kPa,
    #   T=400K -> P~2103 kPa, T=420K -> P~3027 kPa
    # Using ln(P) = A - B/(T+C) form (Antoine), P in Pa, T in K
    _sat_A = 21.943731
    _sat_B = 2823.286134
    _sat_C = -17.612907

    @classmethod
    def P_sat(cls, T):
        """Saturation pressure [Pa] from temperature [K]."""
        T = np.clip(T, 260.0, cls.T_crit - 1.0)
        ln_P = cls._sat_A - cls._sat_B / (T + cls._sat_C)
        return np.exp(ln_P)

    @classmethod
    def T_sat(cls, P):
        """Saturation temperature [K] from pressure [Pa] via inverse Antoine."""
        P = np.clip(P, 10000.0, cls.P_crit - 1000.0)
        T = cls._sat_B / (cls._sat_A - np.log(P)) + cls._sat_C
        # Refine with Newton
        for _ in range(20):
            P_calc = cls.P_sat(T)
            dT = 0.01
            dPdT = (cls.P_sat(T + dT) - cls.P_sat(T - dT)) / (2 * dT)
            if abs(dPdT) < 1e-10:
                break
            T_new = T + (P - P_calc) / dPdT
            T_new = np.clip(T_new, 260.0, cls.T_crit - 1.0)
            if abs(T_new - T) < 1e-6:
                break
            T = T_new
        return T

    @classmethod
    def h_liquid(cls, T):
        """Saturated liquid enthalpy [J/kg]. Fitted from NIST R245fa data.
        Reference: h_f(298K)~232 kJ/kg, h_f(370K)~350 kJ/kg, h_f(400K)~420 kJ/kg."""
        # Polynomial fit: h_f in J/kg as function of T
        # Anchored to NIST values: 298K->232kJ, 320K->266kJ, 350K->316kJ,
        # 370K->350kJ, 400K->420kJ
        T_ref = 298.15
        cp0 = 1340.0  # J/(kg.K) at 298K
        cp1 = 3.5     # linear term
        dT = T - T_ref
        h = 232000.0 + cp0 * dT + 0.5 * cp1 * dT ** 2
        return h

    @classmethod
    def h_vapor(cls, T):
        """Saturated vapor enthalpy [J/kg]. Fitted from NIST R245fa data.
        Reference: h_g(298K)~412 kJ/kg, h_g(370K)~440 kJ/kg, h_g(400K)~435 kJ/kg."""
        T_ref = 298.15
        dT = T - T_ref
        # h_g rises slowly then turns over near critical
        h = 412000.0 + 500.0 * dT - 3.0 * dT ** 2
        return h

    @classmethod
    def h_fg(cls, T):
        """Latent heat of vaporization [J/kg]."""
        return cls.h_vapor(T) - cls.h_liquid(T)

    @classmethod
    def s_liquid(cls, T):
        """Saturated liquid entropy [J/(kg.K)]. Fitted from NIST.
        Reference: s_f(298K)~1100 J/(kg.K), s_f(370K)~1350 J/(kg.K)."""
        T_ref = 298.15
        dT = T - T_ref
        return 1100.0 + 3.5 * dT + 0.005 * dT ** 2

    @classmethod
    def s_vapor(cls, T):
        """Saturated vapor entropy [J/(kg.K)]. Fitted from NIST.
        Reference: s_g(298K)~1710 J/(kg.K), s_g(370K)~1720 J/(kg.K)."""
        T_ref = 298.15
        dT = T - T_ref
        return 1710.0 + 0.5 * dT - 0.005 * dT ** 2

    @classmethod
    def cp_liquid(cls, T):
        """Liquid specific heat [J/(kg.K)]. ~1340 at 298K, rising with T."""
        return 1340.0 + 3.5 * (T - 298.15)

    @classmethod
    def cp_vapor(cls, T, P=None):
        """Vapor specific heat [J/(kg.K)] at low pressure. ~900 at 300K."""
        return 880.0 + 1.0 * (T - 298.15)

    @classmethod
    def v_liquid(cls, T):
        """Liquid specific volume [m3/kg]. rho_f ~ 1340 kg/m3 at 298K."""
        rho = 1340.0 - 2.5 * (T - 298.15)
        return 1.0 / max(rho, 300.0)

    @classmethod
    def gamma_vapor(cls, T):
        """Ratio of specific heats for vapor (approximate)."""
        return 1.08 + 0.02 * (T / cls.T_crit - 0.7)

    @classmethod
    def h_superheated(cls, T_sat, T, P):
        """Superheated vapor enthalpy [J/kg]."""
        h_g = cls.h_vapor(T_sat)
        cp_avg = cls.cp_vapor((T + T_sat) / 2.0)
        return h_g + cp_avg * (T - T_sat)

    @classmethod
    def s_superheated(cls, T_sat, T, P):
        """Superheated vapor entropy [J/(kg.K)]."""
        s_g = cls.s_vapor(T_sat)
        cp_avg = cls.cp_vapor((T + T_sat) / 2.0)
        return s_g + cp_avg * np.log(T / T_sat)

    @classmethod
    def h_subcooled(cls, T_sat, T, P):
        """Subcooled liquid enthalpy [J/kg]."""
        h_f = cls.h_liquid(T_sat)
        cp = cls.cp_liquid(T)
        return h_f - cp * (T_sat - T)

    @classmethod
    def s_subcooled(cls, T_sat, T, P):
        """Subcooled liquid entropy [J/(kg.K)]."""
        s_f = cls.s_liquid(T_sat)
        cp = cls.cp_liquid(T)
        return s_f - cp * np.log(T_sat / max(T, 200.0))


class ORC_F2a:
    """Organic Rankine Cycle -- F2a thermodynamic cycle model with part-load."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_evap = u["P_evap"]["value"]
        self.P_cond = u["P_cond"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_expander = u["eta_expander"]["value"]
        self.T_source_in = u["T_source_in"]["value"]
        self.T_source_out = u["T_source_out"]["value"]
        self.T_sink_in = u["T_sink_in"]["value"]
        self.superheat = u["superheat"]["value"]
        self.subcool = u["subcool"]["value"]
        self.m_thermal = u["m_thermal"]["value"]
        self.cp_system = u["cp_system"]["value"]
        self.hA_loss = u["hA_loss"]["value"]

        self.fluid = R245faProperties

    def _expander_efficiency_partload(self, load_frac):
        """Part-load expander isentropic efficiency correction."""
        # Typical scroll/screw expander: eta drops at off-design
        # Polynomial fit: eta/eta_design = a0 + a1*L + a2*L^2
        L = np.clip(load_frac, 0.1, 1.0)
        correction = 0.1 + 1.45 * L - 0.55 * L ** 2
        return self.eta_expander * min(correction, 1.0)

    def _pump_efficiency_partload(self, load_frac):
        """Part-load pump efficiency correction."""
        L = np.clip(load_frac, 0.1, 1.0)
        correction = 0.2 + 1.2 * L - 0.4 * L ** 2
        return self.eta_pump * min(correction, 1.0)

    def compute_cycle(self, P_evap=None, P_cond=None, superheat=None,
                      load_fraction=1.0):
        """
        Compute ORC thermodynamic cycle at given conditions.

        Returns dict with state points, performance metrics.
        """
        P_e = P_evap if P_evap is not None else self.P_evap
        P_c = P_cond if P_cond is not None else self.P_cond
        dT_sh = superheat if superheat is not None else self.superheat

        fluid = self.fluid

        # Saturation temperatures
        T_sat_evap = fluid.T_sat(P_e)
        T_sat_cond = fluid.T_sat(P_c)

        # Part-load efficiencies
        eta_exp = self._expander_efficiency_partload(load_fraction)
        eta_pmp = self._pump_efficiency_partload(load_fraction)

        # --- State 1: Pump inlet (subcooled liquid) ---
        T1 = T_sat_cond - self.subcool
        h1 = fluid.h_subcooled(T_sat_cond, T1, P_c)
        s1 = fluid.s_subcooled(T_sat_cond, T1, P_c)
        P1 = P_c

        # --- State 2: Pump outlet (compressed liquid) ---
        P2 = P_e
        v1 = fluid.v_liquid(T1)
        w_pump_ideal = v1 * (P2 - P1)  # isentropic pump work (J/kg)
        w_pump = w_pump_ideal / eta_pmp
        h2 = h1 + w_pump
        T2 = T1 + w_pump / fluid.cp_liquid(T1)
        s2 = s1 + fluid.cp_liquid(T1) * np.log(T2 / T1) if T2 > T1 else s1

        # --- State 3: Expander inlet (superheated vapor) ---
        T3 = T_sat_evap + dT_sh
        h3 = fluid.h_superheated(T_sat_evap, T3, P_e)
        s3 = fluid.s_superheated(T_sat_evap, T3, P_e)
        P3 = P_e

        # --- State 4: Expander outlet (low-pressure, ideally isentropic to P_cond) ---
        P4 = P_c
        # Isentropic expansion: find T4s where s(T4s, P4) = s3
        # For dry fluid like R245fa, expansion lands in superheated region
        s_g_cond = fluid.s_vapor(T_sat_cond)
        if s3 > s_g_cond:
            # Superheated at exit (dry expansion -- typical for R245fa)
            cp_v = fluid.cp_vapor(T_sat_cond)
            T4s = T_sat_cond * np.exp((s3 - s_g_cond) / cp_v)
            h4s = fluid.h_superheated(T_sat_cond, T4s, P4)
        else:
            # Wet expansion (unusual for R245fa)
            h4s = fluid.h_vapor(T_sat_cond)
            T4s = T_sat_cond

        h4 = h3 - eta_exp * (h3 - h4s)
        # Estimate T4 from h4
        h_g_cond = fluid.h_vapor(T_sat_cond)
        if h4 > h_g_cond:
            cp_v = fluid.cp_vapor(T_sat_cond)
            T4 = T_sat_cond + (h4 - h_g_cond) / cp_v
        else:
            T4 = T_sat_cond
        s4 = fluid.s_superheated(T_sat_cond, max(T4, T_sat_cond + 0.1), P4)

        # --- Performance calculations ---
        w_expander = h3 - h4  # specific work from expander (J/kg)
        w_net = w_expander - w_pump  # net specific work (J/kg)
        q_in = h3 - h2  # heat input in evaporator (J/kg)
        q_out = h4 - h1  # heat rejection in condenser (J/kg)

        # Thermal efficiency
        eta_thermal = w_net / q_in if q_in > 0 else 0.0

        # Carnot efficiency for reference
        eta_carnot = 1.0 - T_sat_cond / T_sat_evap if T_sat_evap > 0 else 0.0

        # Mass flow rate for design power
        # Adjust for part-load
        W_net_target = load_fraction * 100000.0  # W (design = 100 kW)
        m_dot = W_net_target / w_net if w_net > 0 else 0.0

        Q_in = m_dot * q_in  # total heat input (W)
        Q_out = m_dot * q_out  # total heat rejection (W)
        W_net_total = m_dot * w_net  # total net power (W)
        W_pump_total = m_dot * w_pump  # total pump power (W)
        W_exp_total = m_dot * w_expander  # total expander power (W)

        state_points = {
            "T": [T1, T2, T3, T4],
            "P": [P1, P2, P3, P4],
            "h": [h1, h2, h3, h4],
            "s": [s1, s2, s3, s4],
        }

        return {
            "state_points": state_points,
            "T_sat_evap": T_sat_evap,
            "T_sat_cond": T_sat_cond,
            "w_net_specific": w_net,
            "w_pump_specific": w_pump,
            "w_expander_specific": w_expander,
            "q_in_specific": q_in,
            "q_out_specific": q_out,
            "eta_thermal": eta_thermal,
            "eta_carnot": eta_carnot,
            "eta_expander": eta_exp,
            "eta_pump": eta_pmp,
            "m_dot": m_dot,
            "W_net": W_net_total,
            "W_pump": W_pump_total,
            "W_expander": W_exp_total,
            "Q_in": Q_in,
            "Q_out": Q_out,
            "load_fraction": load_fraction,
        }

    def simulate(self, load_profile, T_ambient_K, P_evap=None, P_cond=None,
                 dt=1.0, duration_s=3600.0):
        """
        Dynamic simulation with thermal inertia.

        Parameters
        ----------
        load_profile : float or callable(t)
            Load fraction [0.1, 1.0]
        T_ambient_K : float
            Ambient temperature [K]
        P_evap : float, optional
            Evaporator pressure [Pa]
        P_cond : float, optional
            Condenser pressure [Pa]
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation duration [s]
        """
        _load = load_profile if callable(load_profile) else lambda t: load_profile

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        T_init = T_ambient_K + 50.0  # initial system temperature estimate

        def rhs(t, y):
            T_sys = y[0]
            lf = _load(t)
            cycle = self.compute_cycle(P_evap, P_cond, load_fraction=lf)
            Q_gen = cycle["Q_in"] - cycle["W_net"] - cycle["Q_out"]  # residual heat
            Q_loss = self.hA_loss * (T_sys - T_ambient_K)
            dTdt = (Q_gen - Q_loss) / (self.m_thermal * self.cp_system)
            return [dTdt]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_init],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=max(dt, 1.0),
        )

        t_out = sol.t
        T_sys = sol.y[0]
        N = len(t_out)

        results = {
            "t": t_out,
            "T_system": T_sys,
            "W_net": np.zeros(N),
            "eta_thermal": np.zeros(N),
            "Q_in": np.zeros(N),
            "Q_out": np.zeros(N),
            "m_dot": np.zeros(N),
            "load_fraction": np.zeros(N),
            "eta_expander": np.zeros(N),
        }

        for i in range(N):
            lf = _load(t_out[i])
            cycle = self.compute_cycle(P_evap, P_cond, load_fraction=lf)
            results["W_net"][i] = cycle["W_net"]
            results["eta_thermal"][i] = cycle["eta_thermal"]
            results["Q_in"][i] = cycle["Q_in"]
            results["Q_out"][i] = cycle["Q_out"]
            results["m_dot"][i] = cycle["m_dot"]
            results["load_fraction"][i] = lf
            results["eta_expander"][i] = cycle["eta_expander"]

        return results
