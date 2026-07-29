"""
EC111 -- Diesel Generator -- F2a Diesel Cycle Thermodynamic Model

Air-standard diesel cycle with ODE-based dynamic governor and generator model.

Diesel cycle processes:
    1 -> 2  Isentropic compression
    2 -> 3  Constant-pressure heat addition
    3 -> 4  Isentropic expansion
    4 -> 1  Constant-volume heat rejection

Thermal efficiency:
    eta_diesel = 1 - (1 / r_c^(gamma-1)) * ((r_co^gamma - 1) / (gamma * (r_co - 1)))

Dynamic model (ODE):
    d(omega)/dt = (T_engine - T_gen - b*omega) / J
    d(x_fuel)/dt = (x_fuel_cmd - x_fuel) / tau_act   (fuel actuator)
    d(int_err)/dt = omega_ref - omega                  (governor integrator)

where:
    T_engine = eta_diesel * m_fuel_dot * LHV / omega
    T_gen = P_elec / (eta_gen * omega)

Generator efficiency curve:
    eta_gen(load) = eta_rated - a*(1 - load)^2 - b*(1 - load)

BSFC (brake specific fuel consumption):
    BSFC(load) = BSFC_rated * (1 + 0.2*(1 - load) + 0.15*(1 - load)^2)

Reference:
    Heywood, J.B. (2018). Internal Combustion Engine Fundamentals, 2nd ed.
    McGraw-Hill. Chapters 5-6.
"""

import numpy as np
from scipy.integrate import solve_ivp


class DieselGeneratorF2a:
    """Diesel generator -- air-standard diesel cycle with dynamic governor."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated"]["value"]
        self.r_c = u["r_c"]["value"]
        self.gamma = u["gamma"]["value"]
        self.T1 = u["T1"]["value"]
        self.P1 = u["P1"]["value"]
        self.V_d = u["V_displaced"]["value"]
        self.n_cyl = u["n_cylinders"]["value"]
        self.rpm_nom = u["rpm_nominal"]["value"]
        self.J = u["J_engine"]["value"]
        self.b_fric = u["b_friction"]["value"]
        self.eta_gen_rated = u["eta_gen_rated"]["value"]
        self.eta_gen_a = u["eta_gen_a"]["value"]
        self.eta_gen_b = u["eta_gen_b"]["value"]
        self.LHV = u["LHV_diesel"]["value"]
        self.rho_diesel = u["rho_diesel"]["value"]
        self.BSFC_rated = u["BSFC_rated"]["value"]
        self.R_air = u["R_air"]["value"]
        # Use gamma-consistent cp and cv for air-standard analysis
        # cv = R / (gamma - 1), cp = gamma * R / (gamma - 1)
        self.cv = self.R_air / (self.gamma - 1.0)
        self.cp = self.gamma * self.R_air / (self.gamma - 1.0)
        self.Kp = u["governor_Kp"]["value"]
        self.Ki = u["governor_Ki"]["value"]
        self.tau_act = u["governor_tau"]["value"]

        self.omega_nom = self.rpm_nom * 2.0 * np.pi / 60.0

    # ---- Thermodynamic cycle calculations ----

    def diesel_efficiency(self, r_c=None, r_co=None):
        """
        Air-standard diesel cycle thermal efficiency.

        Args:
            r_c:  compression ratio (default: self.r_c)
            r_co: cutoff ratio (default: calculated from rated conditions)

        Returns:
            eta_thermal (float)
        """
        if r_c is None:
            r_c = self.r_c
        if r_co is None:
            r_co = self._design_cutoff_ratio()
        g = self.gamma
        eta = 1.0 - (1.0 / r_c ** (g - 1.0)) * (
            (r_co ** g - 1.0) / (g * (r_co - 1.0))
        )
        return eta

    def _design_cutoff_ratio(self):
        """Calculate cutoff ratio for rated load conditions (~2.0-2.5 typical)."""
        return 2.0

    def cycle_state_points(self, r_c=None, r_co=None, T1=None, P1=None):
        """
        Calculate all four state points of the air-standard diesel cycle.

        Returns:
            dict with T1..T4, P1..P4, v1..v4
        """
        if r_c is None:
            r_c = self.r_c
        if r_co is None:
            r_co = self._design_cutoff_ratio()
        if T1 is None:
            T1 = self.T1
        if P1 is None:
            P1 = self.P1

        g = self.gamma

        # State 1: intake
        v1 = self.R_air * T1 / P1  # specific volume m^3/kg

        # State 2: after isentropic compression
        v2 = v1 / r_c
        T2 = T1 * r_c ** (g - 1.0)
        P2 = P1 * r_c ** g

        # State 3: after constant-pressure heat addition
        v3 = v2 * r_co
        T3 = T2 * r_co
        P3 = P2  # constant pressure

        # State 4: after isentropic expansion
        v4 = v1  # back to v1
        expansion_ratio = v4 / v3
        T4 = T3 / expansion_ratio ** (g - 1.0)
        P4 = P3 / expansion_ratio ** g

        return {
            "T1": T1, "T2": T2, "T3": T3, "T4": T4,
            "P1": P1, "P2": P2, "P3": P3, "P4": P4,
            "v1": v1, "v2": v2, "v3": v3, "v4": v4,
        }

    def heat_added(self, r_c=None, r_co=None, T1=None):
        """Heat added per kg of air (constant pressure process 2->3)."""
        sp = self.cycle_state_points(r_c, r_co, T1)
        return self.cp * (sp["T3"] - sp["T2"])

    def heat_rejected(self, r_c=None, r_co=None, T1=None):
        """Heat rejected per kg of air (constant volume process 4->1)."""
        sp = self.cycle_state_points(r_c, r_co, T1)
        return self.cv * (sp["T4"] - sp["T1"])

    def net_work(self, r_c=None, r_co=None, T1=None):
        """Net work per kg of air."""
        return self.heat_added(r_c, r_co, T1) - self.heat_rejected(r_c, r_co, T1)

    # ---- Generator efficiency curve ----

    def generator_efficiency(self, load_frac):
        """
        Generator efficiency as a function of load fraction.

        eta_gen = eta_rated - a*(1-load)^2 - b*(1-load)
        """
        load_frac = np.clip(load_frac, 0.01, 1.1)
        return self.eta_gen_rated - self.eta_gen_a * (1.0 - load_frac) ** 2 - self.eta_gen_b * (1.0 - load_frac)

    # ---- BSFC curve ----

    def bsfc(self, load_frac):
        """
        Brake specific fuel consumption curve.

        BSFC(load) = BSFC_rated * (1 + 0.2*(1-load) + 0.15*(1-load)^2)

        Returns BSFC in kg/(W*s), multiply by 3.6e9 to get g/kWh.
        """
        load_frac = np.clip(load_frac, 0.05, 1.1)
        return self.BSFC_rated * (1.0 + 0.2 * (1.0 - load_frac) + 0.15 * (1.0 - load_frac) ** 2)

    def fuel_rate(self, P_elec, load_frac=None):
        """Fuel consumption rate in kg/s for given electrical output."""
        if load_frac is None:
            load_frac = P_elec / self.P_rated
        return self.bsfc(load_frac) * P_elec

    # ---- Steady-state analysis ----

    def steady_state(self, load_frac):
        """
        Steady-state performance at given load fraction.

        Returns dict with all key outputs.
        """
        load_frac = np.clip(load_frac, 0.01, 1.1)
        P_elec = load_frac * self.P_rated
        eta_gen = self.generator_efficiency(load_frac)
        P_mech = P_elec / eta_gen

        eta_thermal = self.diesel_efficiency()
        BSFC_val = self.bsfc(load_frac)
        m_fuel = self.fuel_rate(P_elec, load_frac)
        Q_fuel = m_fuel * self.LHV
        eta_overall = P_elec / Q_fuel if Q_fuel > 0 else 0.0

        fuel_rate_Lph = m_fuel / self.rho_diesel * 3600.0 * 1000.0  # L/h (m^3/s -> L/h)

        return {
            "P_elec_W": P_elec,
            "P_mech_W": P_mech,
            "eta_thermal": eta_thermal,
            "eta_gen": eta_gen,
            "eta_overall": eta_overall,
            "BSFC_g_per_kWh": BSFC_val * 3.6e9,
            "fuel_rate_kg_s": m_fuel,
            "fuel_rate_L_h": fuel_rate_Lph,
            "Q_fuel_W": Q_fuel,
            "omega_rpm": self.rpm_nom,
        }

    # ---- Dynamic ODE simulation ----

    def derivatives(self, t, state, P_load_func, omega_ref):
        """
        Dynamic model derivatives.

        States: [omega, x_fuel, int_err]
            omega:   angular velocity [rad/s]
            x_fuel:  fuel injection fraction [0-1]
            int_err: governor integrator state [rad]

        Args:
            P_load_func: callable(t) -> electrical load demand [W]
            omega_ref:   reference angular velocity [rad/s]
        """
        omega, x_fuel, int_err = state

        omega = max(omega, 10.0)  # prevent division by zero
        x_fuel = np.clip(x_fuel, 0.0, 1.1)

        # Maximum engine torque at rated conditions
        eta_th = self.diesel_efficiency()
        # T_engine_max = P_rated / (eta_gen * omega_nom)  plus margin for losses
        T_engine_max = self.P_rated / (self.eta_gen_rated * self.omega_nom) + self.b_fric * self.omega_nom

        # Engine torque proportional to fuel injection
        T_engine = x_fuel * T_engine_max

        # Generator load torque
        P_load = P_load_func(t)
        P_load = np.clip(P_load, 0.0, 1.1 * self.P_rated)
        load_frac = P_load / self.P_rated if self.P_rated > 0 else 0.0
        eta_gen = self.generator_efficiency(load_frac)
        T_gen = P_load / (eta_gen * omega) if omega > 0 else 0.0

        # Friction torque
        T_fric = self.b_fric * omega

        # Angular velocity dynamics
        d_omega = (T_engine - T_gen - T_fric) / self.J

        # Governor: PI control on speed error
        # Normalized error: (omega_ref - omega) / omega_ref
        error = (omega_ref - omega) / omega_ref
        # x_fuel_cmd = Kp * error + Ki * integral(error)
        x_fuel_cmd = 0.5 + 5.0 * error + 2.0 * int_err
        x_fuel_cmd = np.clip(x_fuel_cmd, 0.0, 1.1)

        d_x_fuel = (x_fuel_cmd - x_fuel) / self.tau_act
        d_int_err = error

        return [d_omega, d_x_fuel, d_int_err]

    def simulate(self, P_load, dt, duration_s, x0=None, omega_ref_rpm=None):
        """
        Simulate diesel generator dynamics.

        Args:
            P_load:       electrical load [W] (scalar or callable(t))
            dt:           output time step [s]
            duration_s:   simulation duration [s]
            x0:           initial state [omega, x_fuel, int_err]
            omega_ref_rpm: speed reference [rpm] (default: nominal)

        Returns:
            dict with time-series: t, omega_rpm, P_elec, P_engine, fuel_rate,
                                    eta_overall, frequency_Hz
        """
        if omega_ref_rpm is None:
            omega_ref_rpm = self.rpm_nom
        omega_ref = omega_ref_rpm * 2.0 * np.pi / 60.0

        if x0 is None:
            # Start at nominal speed, roughly correct fuel injection
            x0 = [self.omega_nom, 0.5, 0.0]

        _P_load = P_load if callable(P_load) else lambda t: P_load

        t_span = (0.0, duration_s)
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, state):
            return self.derivatives(t, state, _P_load, omega_ref)

        sol = solve_ivp(
            rhs, t_span, x0, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt * 10,
        )

        omega = sol.y[0]
        x_fuel = sol.y[1]
        t = sol.t

        omega_rpm = omega * 60.0 / (2.0 * np.pi)
        frequency = omega_rpm / 120.0 * 2  # For 2-pole generator: f = rpm * poles / 120

        # Recompute outputs at each time step
        eta_th = self.diesel_efficiency()
        T_engine_max = self.P_rated / (self.eta_gen_rated * self.omega_nom) + self.b_fric * self.omega_nom

        P_elec_arr = np.array([_P_load(ti) for ti in t])
        load_frac_arr = P_elec_arr / self.P_rated

        # Engine mechanical power
        P_engine_arr = x_fuel * T_engine_max * omega

        # Fuel consumption: P_engine = eta_th * m_fuel_dot * LHV
        fuel_rate_arr = np.where(
            eta_th * self.LHV > 0,
            P_engine_arr / (eta_th * self.LHV),
            0.0,
        )

        eta_gen_arr = np.array([self.generator_efficiency(lf) for lf in load_frac_arr])
        eta_overall_arr = np.where(
            fuel_rate_arr * self.LHV > 0,
            P_elec_arr / (fuel_rate_arr * self.LHV),
            0.0,
        )

        return {
            "t": t,
            "omega_rpm": omega_rpm,
            "frequency_Hz": frequency,
            "P_elec_W": P_elec_arr,
            "P_engine_W": P_engine_arr,
            "x_fuel": x_fuel,
            "fuel_rate_kg_s": fuel_rate_arr,
            "eta_overall": eta_overall_arr,
            "eta_gen": eta_gen_arr,
            "load_frac": load_frac_arr,
        }
