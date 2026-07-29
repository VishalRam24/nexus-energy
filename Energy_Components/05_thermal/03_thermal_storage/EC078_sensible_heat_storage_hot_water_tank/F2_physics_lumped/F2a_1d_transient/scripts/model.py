"""
EC078 -- Hot Water Tank TES -- F2a 1D Transient Stratified Model

20-node 1D vertical discretization with:
- Axial conduction (effective conductivity including mixing)
- Charge/discharge with inlet mixing
- Heat loss to ambient through tank walls
- Buoyancy-driven mixing (destratification limiter)

Governing ODE for each node i:
    M_i*cp * dT_i/dt = Q_conduction_i + Q_charge_i + Q_discharge_i - Q_loss_i + Q_mixing_i

References:
    Kleinbach et al. (1993), Solar Energy, 50(2), 155-166
    Duffie & Beckman (2013), Solar Engineering of Thermal Processes
    Newton (1995), ASHRAE Trans., 101(1), 702-712
"""

import numpy as np
from scipy.integrate import solve_ivp


class HotWaterTank_F2a:
    """Hot Water Tank TES -- 20-node 1D stratified transient model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_nodes = u["n_nodes"]["value"]
        self.V_tank = u["V_tank"]["value"]
        self.H_tank = u["H_tank"]["value"]
        self.D_tank = u["D_tank"]["value"]
        self.rho = u["rho_water"]["value"]
        self.cp = u["cp_water"]["value"]
        self.k_water = u["k_water"]["value"]
        self.k_eff_mult = u["k_eff_multiplier"]["value"]
        self.U_loss = u["U_loss"]["value"]
        self.T_ambient = u["T_ambient"]["value"]
        self.T_init_top = u["T_init_top"]["value"]
        self.T_init_bottom = u["T_init_bottom"]["value"]

        # Derived geometry
        self.dz = self.H_tank / self.n_nodes  # node height [m]
        self.A_cross = np.pi * (self.D_tank / 2.0) ** 2  # cross-section area [m2]
        self.V_node = self.V_tank / self.n_nodes  # volume per node [m3]
        self.M_node = self.rho * self.V_node  # mass per node [kg]

        # Surface area per node (cylinder side + top/bottom for end nodes)
        self.A_side_node = np.pi * self.D_tank * self.dz
        self.A_top = self.A_cross  # top cap
        self.A_bottom = self.A_cross  # bottom cap

        # Effective conductivity
        self.k_eff = self.k_water * self.k_eff_mult

    # ------------------------------------------------------------------
    # Initial temperature profile (linear stratification)
    # ------------------------------------------------------------------
    def initial_temperature_profile(self):
        """Linear stratification from top (hot) to bottom (cold)."""
        return np.linspace(self.T_init_top, self.T_init_bottom, self.n_nodes)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def ode_rhs(self, t, T, m_dot_charge_func, T_charge_in_func,
                m_dot_discharge_func, T_discharge_in_func):
        """
        ODE system: dT_i/dt for each of the n_nodes.

        Node 0 = top of tank (hot), Node n-1 = bottom (cold).
        Charge: hot water enters at top (node 0), cold exits at bottom (node n-1).
        Discharge: hot water exits at top (node 0), cold water enters at bottom (node n-1).
        """
        n = self.n_nodes
        dTdt = np.zeros(n)

        m_dot_ch = m_dot_charge_func(t) if callable(m_dot_charge_func) else m_dot_charge_func
        T_ch_in = T_charge_in_func(t) if callable(T_charge_in_func) else T_charge_in_func
        m_dot_dis = m_dot_discharge_func(t) if callable(m_dot_discharge_func) else m_dot_discharge_func
        T_dis_in = T_discharge_in_func(t) if callable(T_discharge_in_func) else T_discharge_in_func

        for i in range(n):
            # --- Axial conduction ---
            Q_cond = 0.0
            if i > 0:
                Q_cond += self.k_eff * self.A_cross / self.dz * (T[i - 1] - T[i])
            if i < n - 1:
                Q_cond += self.k_eff * self.A_cross / self.dz * (T[i + 1] - T[i])

            # --- Heat loss to ambient ---
            A_loss = self.A_side_node
            if i == 0:
                A_loss += self.A_top
            if i == n - 1:
                A_loss += self.A_bottom
            Q_loss = self.U_loss * A_loss * (T[i] - self.T_ambient)

            # --- Charge flow (downward: hot in at top, displaces cold at bottom) ---
            Q_charge = 0.0
            if m_dot_ch > 0:
                if i == 0:
                    # Top node receives hot charge water
                    Q_charge = m_dot_ch * self.cp * (T_ch_in - T[i])
                else:
                    # Internal nodes: advection from node above
                    Q_charge = m_dot_ch * self.cp * (T[i - 1] - T[i])

            # --- Discharge flow (upward: cold in at bottom, hot extracted at top) ---
            Q_discharge = 0.0
            if m_dot_dis > 0:
                if i == n - 1:
                    # Bottom node receives cold discharge water
                    Q_discharge = m_dot_dis * self.cp * (T_dis_in - T[i])
                else:
                    # Internal nodes: advection from node below
                    Q_discharge = m_dot_dis * self.cp * (T[i + 1] - T[i])

            dTdt[i] = (Q_cond + Q_charge + Q_discharge - Q_loss) / (self.M_node * self.cp)

        # --- Buoyancy mixing: if a lower node is hotter, mix to prevent inversion ---
        for i in range(n - 1):
            if T[i] < T[i + 1]:
                # Mix: average the two inverted nodes (instantaneous mixing)
                mix_rate = 10.0 * (T[i + 1] - T[i])  # fast mixing rate [K/s]
                dTdt[i] += mix_rate
                dTdt[i + 1] -= mix_rate

        return dTdt

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, m_dot_charge, T_charge_in, m_dot_discharge, T_discharge_in,
                 T_init, dt, duration_s):
        """
        Simulate stratified tank dynamics.

        Parameters
        ----------
        m_dot_charge : float or callable(t) -- charge flow rate [kg/s]
        T_charge_in : float or callable(t) -- charge inlet temperature [K]
        m_dot_discharge : float or callable(t) -- discharge flow rate [kg/s]
        T_discharge_in : float or callable(t) -- discharge inlet temperature (cold return) [K]
        T_init : array-like or None -- initial temperature profile [K]
        dt : float -- output time step [s]
        duration_s : float -- total duration [s]

        Returns
        -------
        dict with time-series
        """
        _mch = m_dot_charge if callable(m_dot_charge) else lambda t: m_dot_charge
        _Tch = T_charge_in if callable(T_charge_in) else lambda t: T_charge_in
        _mdis = m_dot_discharge if callable(m_dot_discharge) else lambda t: m_dot_discharge
        _Tdis = T_discharge_in if callable(T_discharge_in) else lambda t: T_discharge_in

        if T_init is None:
            T_init = self.initial_temperature_profile()
        else:
            T_init = np.array(T_init, dtype=float)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return self.ode_rhs(t, y, _mch, _Tch, _mdis, _Tdis)

        sol = solve_ivp(
            rhs, (0.0, duration_s), T_init,
            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8,
            max_step=min(dt, 10.0)
        )

        t_out = sol.t
        T_profiles = sol.y  # shape: (n_nodes, n_times)
        N = len(t_out)

        # Post-process
        T_top = T_profiles[0, :]
        T_bottom = T_profiles[-1, :]
        T_mean = np.mean(T_profiles, axis=0)

        # Stored energy relative to ambient
        E_stored = np.zeros(N)
        for i in range(N):
            E_stored[i] = np.sum(self.M_node * self.cp * (T_profiles[:, i] - self.T_ambient))

        # Stratification number (Richardson-like)
        stratification = T_top - T_bottom

        return {
            "t": t_out,
            "T_profiles": T_profiles,
            "T_top": T_top,
            "T_bottom": T_bottom,
            "T_mean": T_mean,
            "E_stored_J": E_stored,
            "E_stored_kWh": E_stored / 3.6e6,
            "stratification_K": stratification,
        }
