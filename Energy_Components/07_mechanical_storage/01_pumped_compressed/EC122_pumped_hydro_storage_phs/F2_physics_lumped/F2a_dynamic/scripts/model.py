"""
EC122 -- Pumped Hydro Storage (PHS) -- F2a Dynamic ODE Model

Simplified physics-lumped model with:
  - Penstock flow dynamics (inertia + friction)
  - Reservoir level tracking (mass balance)
  - Turbine/pump efficiency maps
  - SOC tracking

State variables:
    Q    - penstock volumetric flow [m3/s] (+turbine, -pump)
    H_up - upper reservoir level [m]
    H_lo - lower reservoir level [m]

Penstock momentum (simplified):
    T_w * dQ/dt = (H_net - h_f) * Q_rated/H_rated - Q
    T_w = L * Q_rated / (g * H_rated * A_p)  -- water starting time

Reservoir mass balance:
    dH_up/dt = -Q / A_upper
    dH_lo/dt = +Q / A_lower

Power:
    Turbine: P_elec = eta_t * rho * g * H_net * Q
    Pump:    P_elec = rho * g * H_net * |Q| / eta_p

Reference:
    Kundur (1994), Power System Stability and Control, McGraw-Hill
    Nicolet (2007), Hydroacoustic Modelling, EPFL
"""

import numpy as np
from scipy.integrate import solve_ivp


class PumpedHydroStorage_F2a:
    """Pumped Hydro Storage -- dynamic ODE model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rho = u["rho_water"]["value"]
        self.g = u["g"]["value"]
        self.H_rated = u["H_rated"]["value"]
        self.Q_rated = u["Q_rated"]["value"]
        self.L = u["L_penstock"]["value"]
        self.A_p = u["A_penstock"]["value"]
        self.f_darcy = u["f_darcy"]["value"]
        self.D = u["D_penstock"]["value"]
        self.P_rated = u["P_rated"]["value"]
        self.eta_t_max = u["eta_turbine_max"]["value"]
        self.eta_p_max = u["eta_pump_max"]["value"]
        self.A_upper = u["A_upper"]["value"]
        self.A_lower = u["A_lower"]["value"]
        self.H_upper_init = u["H_upper_init"]["value"]
        self.H_lower_init = u["H_lower_init"]["value"]

        # Water starting time constant [s]
        self.T_w = self.L * self.Q_rated / (self.g * self.H_rated * self.A_p)

    # ------------------------------------------------------------------
    # Friction head loss (Darcy-Weisbach)
    # ------------------------------------------------------------------
    def friction_loss(self, Q):
        """Friction head loss [m]."""
        V = Q / self.A_p
        return self.f_darcy * (self.L / self.D) * V * abs(V) / (2.0 * self.g)

    # ------------------------------------------------------------------
    # Turbine/pump efficiency (smooth hill chart approximation)
    # ------------------------------------------------------------------
    def turbine_efficiency(self, Q):
        """Turbine efficiency as function of flow ratio Q/Q_rated."""
        q = abs(Q) / self.Q_rated
        # Parabolic: peaks at q=1, drops off-design
        eta = self.eta_t_max * (1.0 - 0.8 * (q - 1.0)**2)
        return max(min(eta, self.eta_t_max), 0.05)

    def pump_efficiency(self, Q):
        """Pump efficiency as function of flow ratio."""
        q = abs(Q) / self.Q_rated
        eta = self.eta_p_max * (1.0 - 0.8 * (q - 1.0)**2)
        return max(min(eta, self.eta_p_max), 0.05)

    # ------------------------------------------------------------------
    # Target flow rate from power command
    # ------------------------------------------------------------------
    def target_flow(self, P_command, H_net, mode):
        """Compute target flow rate from electrical power command.

        Turbine: P = eta_t * rho * g * H * Q  => Q = P / (eta_t * rho * g * H)
        Pump:    P = rho * g * H * Q / eta_p   => Q = P * eta_p / (rho * g * H)
        """
        H = max(H_net, 1.0)  # avoid division by zero

        if mode == 'turbine' and P_command > 0:
            Q_target = P_command / (self.eta_t_max * self.rho * self.g * H)
            return min(Q_target, self.Q_rated * 1.1)
        elif mode == 'pump' and P_command < 0:
            Q_target = -abs(P_command) * self.eta_p_max / (self.rho * self.g * H)
            return max(Q_target, -self.Q_rated * 1.1)
        else:
            return 0.0

    # ------------------------------------------------------------------
    # Stored energy and SOC
    # ------------------------------------------------------------------
    def stored_energy(self, H_up, H_lo):
        """Potential energy stored [J]."""
        H_net = max(H_up - H_lo, 0.0)
        V_water = self.A_upper * H_up
        return self.rho * self.g * H_net * V_water

    def soc(self, H_up):
        """State of charge based on upper reservoir level."""
        return max(min(H_up / self.H_upper_init, 1.0), 0.0)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, P_func, mode_func):
        Q, H_up, H_lo = y
        H_net = H_up - H_lo

        mode = mode_func(t)
        P_cmd = P_func(t)

        # Target flow
        Q_target = self.target_flow(P_cmd, H_net, mode)

        # Penstock dynamics: first-order lag toward target
        # T_w * dQ/dt = Q_target - Q  (water starting time response)
        T_w = max(self.T_w, 0.5)  # minimum 0.5s time constant
        dQdt = (Q_target - Q) / T_w

        # Add friction damping
        h_f = self.friction_loss(Q)
        # Friction slows the flow: equivalent to additional damping
        Q_friction_equiv = np.sign(Q) * self.rho * self.g * abs(h_f) * self.A_p / (
            self.rho * self.g * max(self.H_rated, 1.0))
        dQdt -= Q_friction_equiv / T_w * 0.5

        # Reservoir levels
        dH_up = -Q / self.A_upper
        dH_lo = Q / self.A_lower

        # Prevent reservoirs from going negative
        if H_up <= 0.1 and Q > 0:
            dQdt = min(dQdt, 0.0)
            dH_up = max(dH_up, 0.0)
        if H_lo <= 0.1 and Q < 0:
            dQdt = max(dQdt, 0.0)
            dH_lo = min(dH_lo, 0.0)

        return [dQdt, dH_up, dH_lo]

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, P_electrical, mode, Q0=0.0,
                 H_up0=None, H_lo0=None, dt=1.0, duration_s=3600.0):
        """
        Simulate PHS dynamics.

        Parameters
        ----------
        P_electrical : float or callable(t)
            Electrical power [W]. Positive=generate, negative=pump.
        mode : str or callable(t)
            'turbine', 'pump', or 'idle'
        Q0 : float
            Initial flow rate [m3/s]
        H_up0, H_lo0 : float
            Initial reservoir levels [m]
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation time [s]
        """
        if H_up0 is None:
            H_up0 = self.H_upper_init
        if H_lo0 is None:
            H_lo0 = self.H_lower_init

        _P = P_electrical if callable(P_electrical) else lambda t: P_electrical
        _mode = mode if callable(mode) else lambda t: mode

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [Q0, H_up0, H_lo0]

        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, _P, _mode),
            (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45",
            rtol=1e-6, atol=1e-8,
        )

        t_out = sol.t
        Q_out = sol.y[0]
        H_up_out = sol.y[1]
        H_lo_out = sol.y[2]
        N = len(t_out)

        H_net = H_up_out - H_lo_out
        P_hydraulic = np.zeros(N)
        P_electrical_out = np.zeros(N)
        efficiency = np.zeros(N)
        soc_out = np.zeros(N)
        E_stored = np.zeros(N)
        omega_out = np.zeros(N)

        for i in range(N):
            Qi = Q_out[i]
            Hi = max(H_net[i], 0.0)
            m = _mode(t_out[i])

            P_hyd = self.rho * self.g * Hi * Qi
            P_hydraulic[i] = P_hyd
            P_electrical_out[i] = _P(t_out[i])

            if m == 'turbine' and Qi > 0:
                eta = self.turbine_efficiency(Qi)
                efficiency[i] = eta
            elif m == 'pump' and Qi < 0:
                eta = self.pump_efficiency(Qi)
                efficiency[i] = eta

            # Approximate omega from synchronous speed
            omega_out[i] = 18.85  # synchronous speed (fixed)

            soc_out[i] = self.soc(H_up_out[i])
            E_stored[i] = self.stored_energy(H_up_out[i], H_lo_out[i])

        return {
            "t": t_out,
            "Q": Q_out,
            "omega": omega_out,
            "H_upper": H_up_out,
            "H_lower": H_lo_out,
            "H_net": H_net,
            "P_hydraulic": P_hydraulic,
            "P_electrical": P_electrical_out,
            "efficiency": efficiency,
            "SOC": soc_out,
            "E_stored": E_stored,
        }
