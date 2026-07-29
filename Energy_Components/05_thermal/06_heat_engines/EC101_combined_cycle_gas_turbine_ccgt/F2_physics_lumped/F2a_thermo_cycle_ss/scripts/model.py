"""
EC101 -- Combined Cycle Gas Turbine (CCGT) -- F2a Thermo Cycle SS

Steady-state thermodynamic analysis of combined Brayton (gas turbine) +
Rankine (steam turbine) cycle.

Brayton cycle:
    1 -> 2: Compressor (isentropic efficiency)
    2 -> 3: Combustor (fuel LHV)
    3 -> 4: Gas turbine (isentropic efficiency)

HRSG:
    Exhaust gas (state 4) heats water/steam for the Rankine bottoming cycle.

Rankine cycle:
    a -> b: Feed pump (isentropic efficiency)
    b -> c: HRSG (heat from exhaust)
    c -> d: Steam turbine (isentropic efficiency)
    d -> a: Condenser (heat rejection)

Air: ideal gas with cp(T) polynomial.
Steam/water: simple polynomial properties (no CoolProp).

Reference:
    Boyce (2012), Gas Turbine Engineering Handbook
    Kehlhofer et al. (2009), Combined-Cycle Gas & Steam Turbine Power Plants
"""

import numpy as np
from scipy.optimize import brentq


class CCGT_F2a:
    """Combined Cycle Gas Turbine -- steady-state thermodynamic cycle model."""

    R_air = 287.0  # J/(kg.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.m_dot_air = u["m_dot_air"]["value"]
        self.P_amb = u["P_amb"]["value"]
        self.T_amb = u["T_amb"]["value"]
        self.PR = u["PR_comp"]["value"]
        self.eta_c = u["eta_comp"]["value"]
        self.eta_gt = u["eta_turb_gas"]["value"]
        self.TIT = u["TIT"]["value"]
        self.LHV = u["LHV_fuel"]["value"]
        self.eta_comb = u["eta_combustor"]["value"]
        self.dp_comb_frac = u["dp_combustor_frac"]["value"]
        self.eta_mech_gt = u["eta_mech_gt"]["value"]
        self.HRSG_eff = u["HRSG_effectiveness"]["value"]
        self.T_exhaust_min = u["T_exhaust_min"]["value"]
        self.P_steam = u["P_steam_HP"]["value"]
        self.T_steam = u["T_steam_HP"]["value"]
        self.P_cond = u["P_condenser"]["value"]
        self.eta_st = u["eta_turb_steam"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_mech_st = u["eta_mech_st"]["value"]
        self.eta_gen = u["eta_gen"]["value"]

    # ------------------------------------------------------------------
    # Air/gas properties (polynomial cp)
    # ------------------------------------------------------------------
    @staticmethod
    def cp_air(T):
        """Specific heat of air [J/(kg.K)] as function of T [K].
        Polynomial fit valid 200-2000 K."""
        # NASA polynomial simplified
        return 1005.0 + 0.1 * (T - 300.0) + 3.5e-5 * (T - 300.0)**2

    @staticmethod
    def cp_gas(T):
        """Specific heat of combustion gas [J/(kg.K)]."""
        # Slightly higher cp due to CO2/H2O content
        return 1050.0 + 0.12 * (T - 300.0) + 4.0e-5 * (T - 300.0)**2

    def mean_cp_air(self, T1, T2):
        """Mean cp of air between T1 and T2."""
        n = 20
        T_arr = np.linspace(T1, T2, n)
        return np.mean([self.cp_air(T) for T in T_arr])

    def mean_cp_gas(self, T1, T2):
        """Mean cp of gas between T1 and T2."""
        n = 20
        T_arr = np.linspace(T1, T2, n)
        return np.mean([self.cp_gas(T) for T in T_arr])

    def gamma_air(self, T):
        """Specific heat ratio for air."""
        cp = self.cp_air(T)
        return cp / (cp - self.R_air)

    def gamma_gas(self, T):
        """Specific heat ratio for combustion gas."""
        cp = self.cp_gas(T)
        R_gas = 290.0  # slightly lower molecular weight mix
        return cp / (cp - R_gas)

    # ------------------------------------------------------------------
    # Steam/water properties (simple polynomial — no CoolProp)
    # ------------------------------------------------------------------
    @staticmethod
    def water_sat_temp(P):
        """Saturation temperature [K] from pressure [Pa]. Simple correlation."""
        # Antoine-like: T_sat = A + B / (C + log10(P))
        # Fitted to steam tables at key points
        lp = np.log(P)
        return 280.0 + 20.0 * lp - 0.8 * lp**2 + 0.012 * lp**3

    @staticmethod
    def water_enthalpy_f(T):
        """Saturated liquid enthalpy [J/kg] at temperature T [K]."""
        T_c = T - 273.15
        return 4186.0 * T_c  # Approximate: h_f ~ cp_water * T_celsius

    @staticmethod
    def water_enthalpy_fg(T):
        """Latent heat of vaporization [J/kg] at temperature T [K]."""
        T_c = T - 273.15
        # Decreases toward critical point
        return max(2500000.0 - 2300.0 * T_c, 500000.0)

    @staticmethod
    def steam_enthalpy(T, P):
        """Superheated steam enthalpy [J/kg]. Simple polynomial."""
        T_c = T - 273.15
        # h ~ h_g + cp_steam * (T - T_sat)
        h_g = 2700000.0  # approximate at moderate pressures
        cp_steam = 2100.0 + 0.4 * T_c  # J/(kg.K)
        return h_g + cp_steam * max(T_c - 100.0, 0.0)

    @staticmethod
    def steam_entropy_approx(T, P):
        """Approximate steam entropy for isentropic expansion."""
        return 6.5 + 2.0 * np.log(T / 373.15) - 0.3 * np.log(P / 101325.0)

    # ------------------------------------------------------------------
    # Brayton cycle analysis
    # ------------------------------------------------------------------
    def brayton_cycle(self, TIT=None, PR=None, m_dot_air=None):
        """
        Compute Brayton (gas turbine) cycle state points.

        Returns dict with state points and performance.
        """
        TIT = TIT or self.TIT
        PR = PR or self.PR
        m_dot = m_dot_air or self.m_dot_air

        # State 1: Compressor inlet
        T1 = self.T_amb
        P1 = self.P_amb

        # State 2: Compressor outlet (isentropic with efficiency)
        gamma_12 = self.gamma_air(T1)
        T2s = T1 * PR ** ((gamma_12 - 1.0) / gamma_12)
        T2 = T1 + (T2s - T1) / self.eta_c
        P2 = P1 * PR

        # Compressor work
        cp_12 = self.mean_cp_air(T1, T2)
        W_comp = m_dot * cp_12 * (T2 - T1)

        # State 3: Combustor outlet (TIT)
        T3 = TIT
        P3 = P2 * (1.0 - self.dp_comb_frac)

        # Fuel mass flow
        cp_23 = self.mean_cp_gas(T2, T3)
        Q_comb = m_dot * cp_23 * (T3 - T2)
        m_fuel = Q_comb / (self.LHV * self.eta_comb)

        # State 4: Turbine outlet
        m_dot_gas = m_dot + m_fuel
        PR_turb = P3 / P1  # Expand to ambient
        gamma_34 = self.gamma_gas(T3)
        T4s = T3 / (PR_turb ** ((gamma_34 - 1.0) / gamma_34))
        T4 = T3 - self.eta_gt * (T3 - T4s)

        # Turbine work
        cp_34 = self.mean_cp_gas(T3, T4)
        W_turb_gas = m_dot_gas * cp_34 * (T3 - T4)

        # Net gas turbine power
        W_net_gt = (W_turb_gas - W_comp) * self.eta_mech_gt
        eta_gt_cycle = W_net_gt / (m_fuel * self.LHV) if m_fuel > 0 else 0

        return {
            "T1": T1, "T2": T2, "T3": T3, "T4": T4,
            "P1": P1, "P2": P2, "P3": P3,
            "W_comp": W_comp, "W_turb_gas": W_turb_gas,
            "W_net_gt": W_net_gt,
            "m_dot_air": m_dot, "m_dot_fuel": m_fuel,
            "m_dot_gas": m_dot_gas,
            "Q_combustor": Q_comb,
            "eta_gt": eta_gt_cycle,
        }

    # ------------------------------------------------------------------
    # Rankine bottoming cycle
    # ------------------------------------------------------------------
    def rankine_cycle(self, T_exhaust, m_dot_gas):
        """
        Compute Rankine (steam) bottoming cycle.

        Parameters
        ----------
        T_exhaust : float
            Gas turbine exhaust temperature [K] (state 4).
        m_dot_gas : float
            Exhaust gas mass flow [kg/s].

        Returns dict with performance.
        """
        # Available exhaust heat
        cp_exh = self.mean_cp_gas(T_exhaust, self.T_exhaust_min)
        Q_available = m_dot_gas * cp_exh * (T_exhaust - self.T_exhaust_min)
        Q_HRSG = Q_available * self.HRSG_eff

        # Steam conditions
        T_sat_cond = self.water_sat_temp(self.P_cond)
        T_sat_HP = self.water_sat_temp(self.P_steam)

        # Limit steam temperature to exhaust - pinch
        T_steam_actual = min(self.T_steam, T_exhaust - 20.0)

        # Steam enthalpy at turbine inlet
        h_steam_in = self.steam_enthalpy(T_steam_actual, self.P_steam)

        # Feedwater enthalpy (condensate at condenser pressure)
        h_cond = self.water_enthalpy_f(T_sat_cond)

        # Feed pump work
        v_f = 0.001  # m3/kg (water specific volume)
        w_pump = v_f * (self.P_steam - self.P_cond) / self.eta_pump
        h_pump_out = h_cond + w_pump

        # Heat input per kg of steam
        q_in_steam = h_steam_in - h_pump_out

        # Steam mass flow
        if q_in_steam > 0:
            m_dot_steam = Q_HRSG / q_in_steam
        else:
            m_dot_steam = 0.0

        # Steam turbine: isentropic expansion
        # Use approximate enthalpy drop
        s_in = self.steam_entropy_approx(T_steam_actual, self.P_steam)
        # Isentropic outlet: find T such that s(T, P_cond) = s_in
        # Simplified: use enthalpy drop with efficiency
        h_fg_cond = self.water_enthalpy_fg(T_sat_cond)
        h_g_cond = h_cond + h_fg_cond

        # Approximate isentropic enthalpy at condenser
        h_out_s = h_g_cond * 0.85  # rough approximation for wet expansion
        # Actual with efficiency
        h_out = h_steam_in - self.eta_st * (h_steam_in - h_out_s)

        # Steam turbine work
        w_turb_steam = h_steam_in - h_out
        W_turb_steam = m_dot_steam * w_turb_steam * self.eta_mech_st

        # Pump power
        W_pump = m_dot_steam * w_pump

        # Net Rankine power
        W_net_rankine = W_turb_steam - W_pump

        # Condenser heat rejection
        Q_condenser = m_dot_steam * (h_out - h_cond)

        # Rankine efficiency
        eta_rankine = W_net_rankine / Q_HRSG if Q_HRSG > 0 else 0

        return {
            "Q_available": Q_available,
            "Q_HRSG": Q_HRSG,
            "m_dot_steam": m_dot_steam,
            "W_turb_steam": W_turb_steam,
            "W_pump": W_pump,
            "W_net_rankine": W_net_rankine,
            "Q_condenser": Q_condenser,
            "eta_rankine": eta_rankine,
            "T_steam_actual": T_steam_actual,
        }

    # ------------------------------------------------------------------
    # Full combined cycle
    # ------------------------------------------------------------------
    def combined_cycle(self, TIT=None, PR=None, m_dot_air=None, load_fraction=1.0):
        """
        Full CCGT steady-state analysis.

        Parameters
        ----------
        TIT : float, optional
            Turbine inlet temperature [K]. Default from params.
        PR : float, optional
            Pressure ratio. Default from params.
        m_dot_air : float, optional
            Air mass flow [kg/s]. Default from params.
        load_fraction : float
            Part-load via TIT modulation (0.3 to 1.0).

        Returns
        -------
        dict with all cycle performance data.
        """
        TIT_eff = TIT or self.TIT
        if load_fraction < 1.0:
            # Part-load: reduce TIT (simplified approach)
            TIT_min = self.T_amb + 0.3 * (TIT_eff - self.T_amb)
            TIT_eff = TIT_min + load_fraction * (TIT_eff - TIT_min)

        brayton = self.brayton_cycle(TIT_eff, PR, m_dot_air)
        rankine = self.rankine_cycle(brayton["T4"], brayton["m_dot_gas"])

        # Total power
        W_gt_elec = brayton["W_net_gt"] * self.eta_gen
        W_st_elec = rankine["W_net_rankine"] * self.eta_gen
        W_total = W_gt_elec + W_st_elec

        # Fuel input
        Q_fuel = brayton["m_dot_fuel"] * self.LHV

        # Combined efficiency
        eta_combined = W_total / Q_fuel if Q_fuel > 0 else 0

        # Heat rate [kJ/kWh]
        heat_rate = 3600.0 / eta_combined if eta_combined > 0 else float('inf')

        return {
            "brayton": brayton,
            "rankine": rankine,
            "W_gt_elec_MW": W_gt_elec / 1e6,
            "W_st_elec_MW": W_st_elec / 1e6,
            "W_total_MW": W_total / 1e6,
            "Q_fuel_MW": Q_fuel / 1e6,
            "eta_combined": eta_combined,
            "heat_rate_kJ_kWh": heat_rate,
            "eta_gt_cycle": brayton["eta_gt"],
            "eta_rankine": rankine["eta_rankine"],
            "T_exhaust_K": brayton["T4"],
            "m_dot_fuel_kgs": brayton["m_dot_fuel"],
            "load_fraction": load_fraction,
            "TIT_actual": TIT_eff,
        }

    # ------------------------------------------------------------------
    # Parametric sweeps
    # ------------------------------------------------------------------
    def sweep_pressure_ratio(self, PR_range=None, TIT=None):
        """Sweep compressor pressure ratio."""
        if PR_range is None:
            PR_range = np.arange(8, 35, 1)
        results = []
        for pr in PR_range:
            r = self.combined_cycle(TIT=TIT, PR=pr)
            results.append(r)
        return PR_range, results

    def sweep_part_load(self, load_range=None):
        """Sweep part-load fraction."""
        if load_range is None:
            load_range = np.linspace(0.3, 1.0, 15)
        results = []
        for lf in load_range:
            r = self.combined_cycle(load_fraction=lf)
            results.append(r)
        return load_range, results

    def sweep_TIT(self, TIT_range=None):
        """Sweep turbine inlet temperature."""
        if TIT_range is None:
            TIT_range = np.linspace(1273.15, 1873.15, 20)
        results = []
        for tit in TIT_range:
            r = self.combined_cycle(TIT=tit)
            results.append(r)
        return TIT_range, results
