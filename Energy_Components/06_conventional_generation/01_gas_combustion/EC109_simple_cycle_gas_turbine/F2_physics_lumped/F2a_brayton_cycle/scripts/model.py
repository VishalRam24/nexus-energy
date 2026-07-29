"""
EC109 -- Simple Cycle Gas Turbine -- F2a Brayton Cycle

Simple Brayton cycle with compressor + combustor + turbine.

Temperature-dependent specific heat:
    cp_air(T) = 1005 + 0.1*(T - 300) [J/(kg.K)]

Compressor:
    T2 = T1 * (P2/P1)^((gamma-1)/(gamma*eta_c))

Combustor:
    m_dot * cp * (T3 - T2) = m_fuel * LHV * eta_comb

Turbine:
    T4 = T3 * (1 - eta_t * (1 - (P1/P2)^((gamma-1)/gamma)))

Includes: pressure ratio sweep, part-load via TIT modulation,
ambient temperature correction.

Reference:
    Saravanamuttoo et al. (2017), Gas Turbine Theory
    Walsh & Fletcher (2004), Gas Turbine Performance
"""

import numpy as np


class SimpleGasTurbine_F2a:
    """Simple cycle gas turbine -- Brayton cycle with T-dependent properties."""

    R_air = 287.0  # J/(kg.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.m_dot_air = u["m_dot_air"]["value"]
        self.P_amb = u["P_amb"]["value"]
        self.T_amb = u["T_amb"]["value"]
        self.PR = u["PR"]["value"]
        self.eta_c = u["eta_comp"]["value"]
        self.eta_t = u["eta_turb"]["value"]
        self.TIT = u["TIT"]["value"]
        self.LHV = u["LHV"]["value"]
        self.eta_comb = u["eta_combustor"]["value"]
        self.dp_comb_frac = u["dp_combustor_frac"]["value"]
        self.eta_mech = u["eta_mech"]["value"]
        self.eta_gen = u["eta_gen"]["value"]
        self.dp_inlet_frac = u["dp_inlet_frac"]["value"]
        self.dp_exhaust_frac = u["dp_exhaust_frac"]["value"]

    # ------------------------------------------------------------------
    # Temperature-dependent air properties
    # ------------------------------------------------------------------
    @staticmethod
    def cp_air(T):
        """Specific heat of air [J/(kg.K)] -- simple T-dependent."""
        return 1005.0 + 0.1 * (T - 300.0)

    @staticmethod
    def cp_gas(T):
        """Specific heat of combustion products [J/(kg.K)]."""
        return 1050.0 + 0.12 * (T - 300.0)

    def gamma_air(self, T):
        """Specific heat ratio for air at temperature T."""
        cp = self.cp_air(T)
        return cp / (cp - self.R_air)

    def gamma_gas(self, T):
        """Specific heat ratio for combustion gas at temperature T."""
        cp = self.cp_gas(T)
        R_gas = 290.0
        return cp / (cp - R_gas)

    def mean_cp(self, T1, T2, gas=False):
        """Mean specific heat between T1 and T2."""
        n = 20
        Ts = np.linspace(T1, T2, n)
        func = self.cp_gas if gas else self.cp_air
        return np.mean([func(T) for T in Ts])

    # ------------------------------------------------------------------
    # Brayton cycle analysis
    # ------------------------------------------------------------------
    def brayton_cycle(self, TIT=None, PR=None, m_dot_air=None, T_amb=None):
        """
        Compute Brayton cycle state points and performance.

        Parameters
        ----------
        TIT : float, optional
            Turbine inlet temperature [K].
        PR : float, optional
            Compressor pressure ratio.
        m_dot_air : float, optional
            Air mass flow [kg/s].
        T_amb : float, optional
            Ambient temperature [K].

        Returns
        -------
        dict with state points and performance metrics.
        """
        TIT = TIT if TIT is not None else self.TIT
        PR = PR if PR is not None else self.PR
        m_dot = m_dot_air if m_dot_air is not None else self.m_dot_air
        T1 = T_amb if T_amb is not None else self.T_amb
        P1 = self.P_amb * (1.0 - self.dp_inlet_frac)

        # State 2: Compressor outlet
        gamma_1 = self.gamma_air(T1)
        # Isentropic temperature rise
        T2s = T1 * PR ** ((gamma_1 - 1.0) / gamma_1)
        # Actual with isentropic efficiency
        T2 = T1 + (T2s - T1) / self.eta_c
        P2 = P1 * PR

        # Compressor specific work
        cp_12 = self.mean_cp(T1, T2)
        w_comp = cp_12 * (T2 - T1)

        # State 3: Combustor outlet (TIT)
        T3 = TIT
        P3 = P2 * (1.0 - self.dp_comb_frac)

        # Fuel mass flow
        cp_23 = self.mean_cp(T2, T3, gas=True)
        q_comb = cp_23 * (T3 - T2)
        m_fuel = m_dot * q_comb / (self.LHV * self.eta_comb)

        # Total mass flow through turbine
        m_dot_gas = m_dot + m_fuel

        # State 4: Turbine outlet
        # Expansion pressure ratio including exhaust loss
        P4 = self.P_amb * (1.0 + self.dp_exhaust_frac)
        PR_turb = P3 / P4
        gamma_3 = self.gamma_gas(T3)
        # T4 using the specified formula
        T4 = T3 * (1.0 - self.eta_t * (1.0 - (1.0 / PR_turb) ** ((gamma_3 - 1.0) / gamma_3)))

        # Turbine specific work
        cp_34 = self.mean_cp(T3, T4, gas=True)
        w_turb = cp_34 * (T3 - T4)

        # Net specific work
        w_net = (m_dot_gas * w_turb - m_dot * w_comp) / m_dot

        # Powers
        W_comp = m_dot * w_comp
        W_turb = m_dot_gas * w_turb
        W_net_shaft = (W_turb - W_comp) * self.eta_mech
        W_elec = W_net_shaft * self.eta_gen

        # Fuel input
        Q_fuel = m_fuel * self.LHV

        # Efficiencies
        eta_thermal = W_net_shaft / Q_fuel if Q_fuel > 0 else 0
        eta_elec = W_elec / Q_fuel if Q_fuel > 0 else 0

        # Heat rate [kJ/kWh]
        heat_rate = 3600.0 / eta_elec if eta_elec > 0 else float('inf')

        # Specific fuel consumption [kg/kWh]
        SFC = m_fuel * 3600.0 / (W_elec / 1000.0) if W_elec > 0 else float('inf')

        # Exhaust heat available
        cp_ex = self.mean_cp(T4, T1, gas=True)
        Q_exhaust = m_dot_gas * cp_ex * (T4 - T1)

        return {
            "T1": T1, "T2": T2, "T3": T3, "T4": T4,
            "P1": P1, "P2": P2, "P3": P3, "P4": P4,
            "W_comp_MW": W_comp / 1e6,
            "W_turb_MW": W_turb / 1e6,
            "W_net_shaft_MW": W_net_shaft / 1e6,
            "W_elec_MW": W_elec / 1e6,
            "Q_fuel_MW": Q_fuel / 1e6,
            "Q_exhaust_MW": Q_exhaust / 1e6,
            "m_dot_air": m_dot,
            "m_dot_fuel": m_fuel,
            "m_dot_gas": m_dot_gas,
            "eta_thermal": eta_thermal,
            "eta_electrical": eta_elec,
            "heat_rate_kJ_kWh": heat_rate,
            "SFC_kg_kWh": SFC,
            "PR": PR,
            "TIT": TIT,
            "T_exhaust_K": T4,
        }

    # ------------------------------------------------------------------
    # Part-load operation
    # ------------------------------------------------------------------
    def part_load(self, load_fraction, T_amb=None):
        """
        Part-load via TIT modulation.

        Parameters
        ----------
        load_fraction : float
            Load fraction (0.3 to 1.0).
        T_amb : float, optional
            Ambient temperature.

        Returns dict.
        """
        # Reduce TIT for part-load
        TIT_min = (self.T_amb if T_amb is None else T_amb) + 0.3 * (self.TIT - self.T_amb)
        TIT_actual = TIT_min + load_fraction * (self.TIT - TIT_min)
        return self.brayton_cycle(TIT=TIT_actual, T_amb=T_amb)

    # ------------------------------------------------------------------
    # Parametric sweeps
    # ------------------------------------------------------------------
    def sweep_pressure_ratio(self, PR_range=None, TIT=None):
        """Sweep compressor pressure ratio."""
        if PR_range is None:
            PR_range = np.arange(5, 40, 1)
        results = [self.brayton_cycle(TIT=TIT, PR=pr) for pr in PR_range]
        return PR_range, results

    def sweep_TIT(self, TIT_range=None, PR=None):
        """Sweep turbine inlet temperature."""
        if TIT_range is None:
            TIT_range = np.linspace(1073.15, 1873.15, 25)
        results = [self.brayton_cycle(TIT=tit, PR=PR) for tit in TIT_range]
        return TIT_range, results

    def sweep_part_load(self, load_range=None, T_amb=None):
        """Sweep part-load fraction."""
        if load_range is None:
            load_range = np.linspace(0.3, 1.0, 20)
        results = [self.part_load(lf, T_amb) for lf in load_range]
        return load_range, results

    def sweep_ambient_temp(self, T_amb_range=None):
        """Sweep ambient temperature."""
        if T_amb_range is None:
            T_amb_range = np.linspace(253.15, 323.15, 20)
        results = [self.brayton_cycle(T_amb=ta) for ta in T_amb_range]
        return T_amb_range, results
