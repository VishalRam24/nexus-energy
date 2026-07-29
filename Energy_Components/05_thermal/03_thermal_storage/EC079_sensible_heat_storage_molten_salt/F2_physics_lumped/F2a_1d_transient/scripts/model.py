"""
EC079 -- Molten Salt TES -- F2a 1D Transient Model with T-dependent Properties

20-node 1D vertical model for Solar Salt (60% NaNO3 + 40% KNO3).
Temperature-dependent properties: density, specific heat, thermal conductivity, viscosity.

Operating range: 290-565 C (563-838 K).

Property correlations (Solar Salt, Zavoico 2001):
    rho(T) = 2090 - 0.636*T_C  [kg/m3] (T_C in Celsius)
    cp(T)  = 1443 + 0.172*T_C  [J/(kg.K)]
    k(T)   = 0.443 + 1.9e-4*T_C [W/(m.K)]
    mu(T)  = 0.02281 - 1.2e-4*T_C + 2.28e-7*T_C^2 - 1.47e-10*T_C^3 [Pa.s]

References:
    Zavoico (2001), Solar Power Tower Design Basis Document, SAND2001-2100
    Pacheco (2002), Final Test and Eval Results, SAND2002-0120
    Ferri et al. (2008), ASME J. Solar Energy Engineering
"""

import numpy as np
from scipy.integrate import solve_ivp


class MoltenSaltTES_F2a:
    """Molten Salt TES -- 20-node 1D transient model with T-dependent properties."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_nodes = u["n_nodes"]["value"]
        self.V_tank = u["V_tank"]["value"]
        self.H_tank = u["H_tank"]["value"]
        self.D_tank = u["D_tank"]["value"]
        self.T_cold = u["T_cold"]["value"]
        self.T_hot = u["T_hot"]["value"]
        self.U_loss = u["U_loss"]["value"]
        self.T_ambient = u["T_ambient"]["value"]
        self.k_eff_mult = u["k_eff_multiplier"]["value"]

        # Derived geometry
        self.dz = self.H_tank / self.n_nodes
        self.A_cross = np.pi * (self.D_tank / 2.0) ** 2
        self.V_node = self.V_tank / self.n_nodes
        self.A_side_node = np.pi * self.D_tank * self.dz
        self.A_top = self.A_cross
        self.A_bottom = self.A_cross

    # ------------------------------------------------------------------
    # Temperature-dependent Solar Salt properties
    # ------------------------------------------------------------------
    @staticmethod
    def salt_density(T_K):
        """Solar Salt density [kg/m3]. Valid 260-600 C."""
        T_C = T_K - 273.15
        return 2090.0 - 0.636 * T_C

    @staticmethod
    def salt_cp(T_K):
        """Solar Salt specific heat [J/(kg.K)]. Valid 260-600 C."""
        T_C = T_K - 273.15
        return 1443.0 + 0.172 * T_C

    @staticmethod
    def salt_conductivity(T_K):
        """Solar Salt thermal conductivity [W/(m.K)]. Valid 260-600 C."""
        T_C = T_K - 273.15
        return 0.443 + 1.9e-4 * T_C

    @staticmethod
    def salt_viscosity(T_K):
        """Solar Salt dynamic viscosity [Pa.s]. Valid 260-600 C."""
        T_C = T_K - 273.15
        return max(0.02281 - 1.2e-4 * T_C + 2.28e-7 * T_C**2 - 1.47e-10 * T_C**3, 1e-4)

    def node_mass(self, T_K):
        """Mass of salt in a node at temperature T [kg]."""
        return self.salt_density(T_K) * self.V_node

    # ------------------------------------------------------------------
    # Initial temperature profile
    # ------------------------------------------------------------------
    def initial_temperature_profile(self, mode="cold"):
        """
        Initial profile.
        mode='cold': uniform cold tank temp
        mode='hot': uniform hot tank temp
        mode='linear': linear from hot(top) to cold(bottom)
        """
        if mode == "cold":
            return np.full(self.n_nodes, self.T_cold)
        elif mode == "hot":
            return np.full(self.n_nodes, self.T_hot)
        else:
            return np.linspace(self.T_hot, self.T_cold, self.n_nodes)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def ode_rhs(self, t, T, m_dot_charge_func, T_charge_in_func,
                m_dot_discharge_func, T_discharge_in_func):
        """ODE system for 20-node molten salt tank."""
        n = self.n_nodes
        dTdt = np.zeros(n)

        m_dot_ch = m_dot_charge_func(t) if callable(m_dot_charge_func) else m_dot_charge_func
        T_ch_in = T_charge_in_func(t) if callable(T_charge_in_func) else T_charge_in_func
        m_dot_dis = m_dot_discharge_func(t) if callable(m_dot_discharge_func) else m_dot_discharge_func
        T_dis_in = T_discharge_in_func(t) if callable(T_discharge_in_func) else T_discharge_in_func

        for i in range(n):
            T_i = T[i]
            rho_i = self.salt_density(T_i)
            cp_i = self.salt_cp(T_i)
            k_i = self.salt_conductivity(T_i)
            M_i = rho_i * self.V_node

            # Effective conductivity
            k_eff = k_i * self.k_eff_mult

            # --- Axial conduction ---
            Q_cond = 0.0
            if i > 0:
                k_avg = 0.5 * (k_eff + self.salt_conductivity(T[i-1]) * self.k_eff_mult)
                Q_cond += k_avg * self.A_cross / self.dz * (T[i-1] - T_i)
            if i < n - 1:
                k_avg = 0.5 * (k_eff + self.salt_conductivity(T[i+1]) * self.k_eff_mult)
                Q_cond += k_avg * self.A_cross / self.dz * (T[i+1] - T_i)

            # --- Heat loss ---
            A_loss = self.A_side_node
            if i == 0:
                A_loss += self.A_top
            if i == n - 1:
                A_loss += self.A_bottom
            Q_loss = self.U_loss * A_loss * (T_i - self.T_ambient)

            # --- Charge flow (hot salt enters at top, cold exits at bottom) ---
            Q_charge = 0.0
            if m_dot_ch > 0:
                if i == 0:
                    Q_charge = m_dot_ch * cp_i * (T_ch_in - T_i)
                else:
                    cp_up = self.salt_cp(T[i-1])
                    Q_charge = m_dot_ch * cp_up * (T[i-1] - T_i)

            # --- Discharge flow (cold salt enters at bottom, hot exits at top) ---
            Q_discharge = 0.0
            if m_dot_dis > 0:
                if i == n - 1:
                    Q_discharge = m_dot_dis * cp_i * (T_dis_in - T_i)
                else:
                    cp_down = self.salt_cp(T[i+1])
                    Q_discharge = m_dot_dis * cp_down * (T[i+1] - T_i)

            dTdt[i] = (Q_cond + Q_charge + Q_discharge - Q_loss) / (M_i * cp_i)

        # Buoyancy mixing
        for i in range(n - 1):
            if T[i] < T[i + 1]:
                mix_rate = 5.0 * (T[i + 1] - T[i])
                dTdt[i] += mix_rate
                dTdt[i + 1] -= mix_rate

        return dTdt

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, m_dot_charge, T_charge_in, m_dot_discharge, T_discharge_in,
                 T_init, dt, duration_s):
        """
        Simulate molten salt TES dynamics.

        Parameters
        ----------
        m_dot_charge : float or callable(t) [kg/s]
        T_charge_in : float or callable(t) [K]
        m_dot_discharge : float or callable(t) [kg/s]
        T_discharge_in : float or callable(t) [K]
        T_init : array or None
        dt : float [s]
        duration_s : float [s]

        Returns
        -------
        dict with time-series
        """
        _mch = m_dot_charge if callable(m_dot_charge) else lambda t: m_dot_charge
        _Tch = T_charge_in if callable(T_charge_in) else lambda t: T_charge_in
        _mdis = m_dot_discharge if callable(m_dot_discharge) else lambda t: m_dot_discharge
        _Tdis = T_discharge_in if callable(T_discharge_in) else lambda t: T_discharge_in

        if T_init is None:
            T_init = self.initial_temperature_profile("cold")
        else:
            T_init = np.array(T_init, dtype=float)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return self.ode_rhs(t, y, _mch, _Tch, _mdis, _Tdis)

        sol = solve_ivp(
            rhs, (0.0, duration_s), T_init,
            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8,
            max_step=min(dt, 30.0)
        )

        t_out = sol.t
        T_profiles = sol.y
        N = len(t_out)

        T_top = T_profiles[0, :]
        T_bottom = T_profiles[-1, :]
        T_mean = np.mean(T_profiles, axis=0)

        # Stored energy (with T-dependent cp and rho)
        E_stored = np.zeros(N)
        for i in range(N):
            for j in range(self.n_nodes):
                T_j = T_profiles[j, i]
                rho_j = self.salt_density(T_j)
                cp_j = self.salt_cp(T_j)
                M_j = rho_j * self.V_node
                E_stored[i] += M_j * cp_j * (T_j - self.T_ambient)

        stratification = T_top - T_bottom

        # Salt properties at mean temperature
        rho_mean = np.array([self.salt_density(T) for T in T_mean])
        cp_mean = np.array([self.salt_cp(T) for T in T_mean])
        k_mean = np.array([self.salt_conductivity(T) for T in T_mean])

        return {
            "t": t_out,
            "T_profiles": T_profiles,
            "T_top": T_top,
            "T_bottom": T_bottom,
            "T_mean": T_mean,
            "E_stored_J": E_stored,
            "E_stored_MWh": E_stored / 3.6e9,
            "stratification_K": stratification,
            "rho_mean": rho_mean,
            "cp_mean": cp_mean,
            "k_mean": k_mean,
        }
