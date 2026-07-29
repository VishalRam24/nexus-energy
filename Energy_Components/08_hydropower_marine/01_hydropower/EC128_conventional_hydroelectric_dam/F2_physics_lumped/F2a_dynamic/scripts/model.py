"""
EC128 -- Conventional Hydroelectric Dam -- F2a Dynamic ODE Model

Physics-lumped model with penstock momentum, turbine-governor, and reservoir dynamics.

State variables:
    H   - reservoir water level/head [m]
    Q   - penstock flow rate [m3/s]
    G   - gate opening fraction [0-1]

Governing equations:
    dH/dt = (Q_inflow - Q_turbine) / A_reservoir
    dQ/dt = (H_net - f_loss*Q^2) * g*A / L    (penstock momentum)
    dG/dt = (G_ref - G) / tau_gov              (governor response)
    Q_turbine = G * Q_rated * sqrt(H/H_rated)
    P = eta * rho * g * H * Q_turbine

Francis turbine characteristics with efficiency curve.
Governor with rate limiting. Load rejection transient modeling.

Reference:
    Kundur (1994), Power System Stability and Control, McGraw-Hill
    IEEE Std 1207 -- Glossary of Terms for Hydroelectric Generating Stations
    USBR Engineering Monograph No.20 -- Selecting Hydraulic Reaction Turbines
"""

import numpy as np
from scipy.integrate import solve_ivp


class ConventionalHydroDam_F2a:
    """Conventional Hydroelectric Dam -- penstock + turbine-governor ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.rho = u["rho_water"]["value"]
        self.g = u["g"]["value"]
        self.H_rated = u["H_rated"]["value"]
        self.Q_rated = u["Q_rated"]["value"]
        self.P_rated = u["P_rated"]["value"]
        self.L = u["L_penstock"]["value"]
        self.A_p = u["A_penstock"]["value"]
        self.D = u["D_penstock"]["value"]
        self.f_darcy = u["f_darcy"]["value"]
        self.A_res = u["A_reservoir"]["value"]
        self.H_init = u["H_reservoir_init"]["value"]
        self.H_tail = u["H_tailwater"]["value"]
        self.tau_gov = u["tau_gov"]["value"]
        self.G_init = u["G_init"]["value"]
        self.G_min = u["G_min"]["value"]
        self.G_max = u["G_max"]["value"]
        self.G_rate_max = u["G_rate_max"]["value"]
        self.eta_rated = u["eta_rated"]["value"]
        self.Q_inflow_base = u["Q_inflow_base"]["value"]
        self.J_gen = u["J_gen"]["value"]
        self.omega_sync = u["omega_sync"]["value"]
        self.D_damp = u["D_damping"]["value"]

    # ------------------------------------------------------------------
    # Francis turbine efficiency
    # ------------------------------------------------------------------
    def turbine_efficiency(self, Q, H):
        """Francis turbine efficiency as function of flow and head."""
        if Q <= 0 or H <= 0:
            return 0.0
        # Normalized operating point
        q_n = Q / self.Q_rated
        h_n = H / self.H_rated
        # Peak efficiency at rated, drops at part-load
        eta = self.eta_rated * (1.0 - 0.3 * (q_n - 1.0)**2 - 0.2 * (h_n - 1.0)**2)
        return np.clip(eta, 0.05, self.eta_rated)

    # ------------------------------------------------------------------
    # Friction head loss
    # ------------------------------------------------------------------
    def friction_loss(self, Q):
        """Darcy-Weisbach friction loss [m]."""
        V = Q / self.A_p
        h_f = self.f_darcy * (self.L / self.D) * (V * abs(V)) / (2.0 * self.g)
        return h_f

    # ------------------------------------------------------------------
    # Turbine flow
    # ------------------------------------------------------------------
    def turbine_flow(self, G, H):
        """Turbine flow [m3/s] from gate opening and head."""
        H_safe = max(H, 0.0)
        return G * self.Q_rated * np.sqrt(H_safe / self.H_rated)

    # ------------------------------------------------------------------
    # Power output
    # ------------------------------------------------------------------
    def power_output(self, Q, H):
        """Electrical power output [W]."""
        if Q <= 0 or H <= 0:
            return 0.0
        eta = self.turbine_efficiency(Q, H)
        P = eta * self.rho * self.g * H * Q
        return P

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, G_ref_func, Q_inflow_func, P_load_func):
        """
        ODE system.
        y = [H, Q, G]
        """
        H, Q, G = y
        H = max(H, 0.1)
        Q = max(Q, 0.0)
        G = np.clip(G, self.G_min, self.G_max)

        # Inflow
        Q_inflow = Q_inflow_func(t)

        # Turbine flow from gate and head
        Q_turb = self.turbine_flow(G, H)

        # Reservoir level
        dHdt = (Q_inflow - Q_turb) / self.A_res

        # Penstock momentum: dQ/dt = (g*A/L) * (H_net - f_loss)
        H_net = H - self.H_tail
        h_f = self.friction_loss(Q)
        driving_head = H_net - h_f
        dQdt = (self.g * self.A_p / self.L) * (driving_head)
        # Blend toward turbine demanded flow (numerical stability)
        dQdt += (Q_turb - Q) / 2.0

        # Governor: dG/dt = (G_ref - G) / tau_gov with rate limiting
        G_ref = G_ref_func(t)
        G_ref = np.clip(G_ref, self.G_min, self.G_max)
        dGdt_raw = (G_ref - G) / self.tau_gov
        dGdt = np.clip(dGdt_raw, -self.G_rate_max, self.G_rate_max)

        return [dHdt, dQdt, dGdt]

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    def simulate(self, G_ref=0.5, Q_inflow=None, P_load=None,
                 H0=None, Q0=None, G0=None,
                 dt=1.0, duration_s=3600.0):
        """
        Simulate hydroelectric dam dynamics.

        Parameters
        ----------
        G_ref : float or callable(t)
            Gate opening reference (0-1)
        Q_inflow : float or callable(t)
            Natural inflow [m3/s]
        P_load : float or callable(t)
            Electrical load demand [W] (for governor droop, not used in simple mode)
        H0 : float
            Initial reservoir head [m]
        Q0 : float
            Initial penstock flow [m3/s]
        G0 : float
            Initial gate opening
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation time [s]
        """
        if H0 is None:
            H0 = self.H_init
        if Q0 is None:
            Q0 = self.G_init * self.Q_rated * np.sqrt(H0 / self.H_rated)
        if G0 is None:
            G0 = self.G_init
        if Q_inflow is None:
            Q_inflow = self.Q_inflow_base

        _G_ref = G_ref if callable(G_ref) else lambda t: G_ref
        _Q_in = Q_inflow if callable(Q_inflow) else lambda t: Q_inflow
        _P_load = P_load if callable(P_load) else lambda t: (P_load if P_load else 0.0)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [H0, Q0, G0]

        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, _G_ref, _Q_in, _P_load),
            (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45",
            rtol=1e-6, atol=1e-8, max_step=dt
        )

        t_out = sol.t
        H_out = sol.y[0]
        Q_out = sol.y[1]
        G_out = sol.y[2]
        N = len(t_out)

        Q_turbine = np.zeros(N)
        P_out = np.zeros(N)
        eta_out = np.zeros(N)
        Q_inflow_out = np.zeros(N)

        for i in range(N):
            H = max(H_out[i], 0.1)
            G = np.clip(G_out[i], self.G_min, self.G_max)
            Q_turbine[i] = self.turbine_flow(G, H)
            P_out[i] = self.power_output(Q_out[i], H)
            eta_out[i] = self.turbine_efficiency(Q_out[i], H)
            Q_inflow_out[i] = _Q_in(t_out[i])

        return {
            "t": t_out,
            "H_reservoir": H_out,
            "Q_penstock": Q_out,
            "Q_turbine": Q_turbine,
            "G_gate": G_out,
            "P_output": P_out,
            "efficiency": eta_out,
            "Q_inflow": Q_inflow_out,
        }
