"""
EC085 -- Natural Gas Boiler -- F2a Dynamic Thermal Mass

Physics-lumped boiler body thermal ODE with burner modulation.

Governing equation:
    (M_w*cp_w + M_body*cp_body) * dT_w/dt = Q_burner*eta_comb*(1-flue_loss)
                                              - m_dot*cp*(T_w - T_in)
                                              - UA_loss*(T_w - T_amb)

    Q_burner = fuel_rate * LHV * modulation (0 to 1)

Features:
    - Burner on/off cycling with deadband
    - Part-load modulation (min 20%)
    - Flue gas losses
    - Standby heat loss
    - Startup transient from cold

Reference:
    Rasmussen (2012), Dynamic Modelling of Boilers
    EN 15502 Condensing Boiler Standard
"""

import numpy as np
from scipy.integrate import solve_ivp


class NatGasBoiler_F2a:
    """Natural gas boiler -- dynamic thermal mass model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.M_w = u["M_w"]["value"]                       # kg water
        self.cp_w = u["cp_w"]["value"]                     # J/(kg.K)
        self.M_body = u["M_body"]["value"]                 # kg metal
        self.cp_body = u["cp_body"]["value"]               # J/(kg.K)
        self.C_total = self.M_w * self.cp_w + self.M_body * self.cp_body  # J/K
        self.Q_max = u["Q_burner_max"]["value"]            # W
        self.eta_comb = u["eta_comb"]["value"]
        self.eta_comb_min = u["eta_comb_min"]["value"]
        self.mod_min = u["modulation_min"]["value"]
        self.m_dot_design = u["m_dot_design"]["value"]     # kg/s
        self.cp_water = u["cp_water"]["value"]             # J/(kg.K)
        self.UA_loss = u["UA_loss"]["value"]               # W/K
        self.T_amb = u["T_amb"]["value"]                   # K
        self.T_set = u["T_set"]["value"]                   # K
        self.flue_loss_frac = u["flue_loss_frac"]["value"]
        self.deadband = u["burner_cycle_deadband"]["value"]  # K

    # ------------------------------------------------------------------
    # Combustion efficiency vs part load
    # ------------------------------------------------------------------
    def combustion_efficiency(self, modulation):
        """Combustion efficiency varies with part-load modulation."""
        # Linear interpolation between min and max efficiency
        mod_clamped = np.clip(modulation, self.mod_min, 1.0)
        frac = (mod_clamped - self.mod_min) / (1.0 - self.mod_min)
        return self.eta_comb_min + frac * (self.eta_comb - self.eta_comb_min)

    # ------------------------------------------------------------------
    # Burner controller (on/off + modulation)
    # ------------------------------------------------------------------
    def burner_control(self, T_w, T_set):
        """
        Simple on/off + proportional modulation.
        Returns modulation factor (0 = off, mod_min to 1.0 = on).
        """
        error = T_set - T_w
        if error < -self.deadband / 2:
            return 0.0  # Above setpoint + deadband: off
        elif error > self.deadband / 2:
            # Proportional modulation
            mod = np.clip(error / 10.0, self.mod_min, 1.0)
            return mod
        else:
            # In deadband -- keep at minimum if below setpoint
            if T_w < T_set:
                return self.mod_min
            else:
                return 0.0

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def dTdt(self, T_w, m_dot, T_in, modulation):
        """Temperature rate of change [K/s]."""
        eta = self.combustion_efficiency(modulation)
        Q_burner = self.Q_max * modulation * eta * (1.0 - self.flue_loss_frac)
        Q_water = m_dot * self.cp_water * (T_w - T_in)
        Q_loss = self.UA_loss * (T_w - self.T_amb)
        return (Q_burner - Q_water - Q_loss) / self.C_total

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, T_init, T_in, m_dot, T_set, dt, duration_s,
                 modulation_override=None):
        """
        Simulate boiler dynamics.

        Parameters
        ----------
        T_init : float
            Initial boiler water temperature [K].
        T_in : float or callable(t)
            Return water temperature [K].
        m_dot : float or callable(t)
            Water flow rate [kg/s].
        T_set : float or callable(t)
            Setpoint temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Simulation duration [s].
        modulation_override : float or callable(t) or None
            If provided, overrides the internal controller.

        Returns
        -------
        dict with time-series.
        """
        _T_in = T_in if callable(T_in) else lambda t: T_in
        _m_dot = m_dot if callable(m_dot) else lambda t: m_dot
        _T_set = T_set if callable(T_set) else lambda t: T_set

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        # We need to track modulation state; use a wrapper with state
        mod_history = []

        def rhs(t, y):
            T_w = y[0]
            T_in_t = _T_in(t)
            m_dot_t = _m_dot(t)
            T_set_t = _T_set(t)

            if modulation_override is not None:
                mod = modulation_override(t) if callable(modulation_override) else modulation_override
            else:
                mod = self.burner_control(T_w, T_set_t)

            return [self.dTdt(T_w, m_dot_t, T_in_t, mod)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_init],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt
        )

        t_out = sol.t
        T_w_out = sol.y[0]
        Nt = len(t_out)

        # Recompute auxiliary quantities
        modulation = np.zeros(Nt)
        Q_burner = np.zeros(Nt)
        Q_output = np.zeros(Nt)
        Q_loss = np.zeros(Nt)
        eta_arr = np.zeros(Nt)
        fuel_rate = np.zeros(Nt)
        T_out_water = np.zeros(Nt)

        for k in range(Nt):
            T_w = T_w_out[k]
            T_in_t = _T_in(t_out[k])
            m_dot_t = _m_dot(t_out[k])
            T_set_t = _T_set(t_out[k])

            if modulation_override is not None:
                mod = modulation_override(t_out[k]) if callable(modulation_override) else modulation_override
            else:
                mod = self.burner_control(T_w, T_set_t)

            modulation[k] = mod
            eta = self.combustion_efficiency(mod)
            eta_arr[k] = eta
            Q_burner[k] = self.Q_max * mod * eta * (1.0 - self.flue_loss_frac)
            Q_output[k] = m_dot_t * self.cp_water * (T_w - T_in_t)
            Q_loss[k] = self.UA_loss * (T_w - self.T_amb)
            fuel_rate[k] = self.Q_max * mod  # Total fuel input [W]
            T_out_water[k] = T_w  # Boiler is well-mixed, outlet = boiler temp

        # Instantaneous thermal efficiency: Q_output / Q_fuel_gross
        # Note: can momentarily exceed eta_comb during transients (stored energy release)
        fuel_nonzero = np.where(fuel_rate > 1.0, fuel_rate, 1.0)
        thermal_eff = np.where(fuel_rate > 1.0, Q_output / fuel_nonzero, 0.0)

        return {
            "t": t_out,
            "T_boiler": T_w_out,
            "T_out_water": T_out_water,
            "modulation": modulation,
            "Q_burner_W": Q_burner,
            "Q_output_W": Q_output,
            "Q_loss_W": Q_loss,
            "eta_combustion": eta_arr,
            "thermal_efficiency": thermal_eff,
            "fuel_input_W": fuel_rate,
        }
