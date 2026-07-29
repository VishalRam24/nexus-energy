"""
EC080 -- Phase Change Material (PCM) Storage -- F2a Enthalpy Method

Physics-lumped multi-node enthalpy model with phase change.

State variables: enthalpy H_i per node (i = 1..N_nodes).

Governing ODE per node:
    m_i * dH_i/dt = Q_htf_i + Q_cond_{i-1,i} + Q_cond_{i,i+1} - Q_loss_i

Temperature recovery (piecewise):
    Solid:   T_i = T_ref + H_i / cp_s          for H_i < 0 (below melt enthalpy)
    Melting: T_i = T_melt                       for 0 <= H_i <= L_f
    Liquid:  T_i = T_melt + (H_i - L_f) / cp_l for H_i > L_f

Here H_i is specific enthalpy measured relative to the onset of melting:
    H = 0 at T = T_melt (solid side)
    H = L_f at T = T_melt (liquid side)

Reference:
    Voller & Cross (1990), Int. J. Heat Mass Transfer
    Zalba et al. (2003), Applied Thermal Energy, 23(3), 251-283
"""

import numpy as np
from scipy.integrate import solve_ivp


class PCMStorage_F2a:
    """PCM thermal storage -- multi-node enthalpy method with HTF coupling."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = u["N_nodes"]["value"]
        self.m_total = u["m_total"]["value"]                # kg
        self.m_node = self.m_total / self.N                  # kg per node
        self.T_melt = u["T_melt"]["value"]                  # K
        self.L_f = u["L_f"]["value"]                        # J/kg
        self.cp_s = u["cp_solid"]["value"]                  # J/(kg.K)
        self.cp_l = u["cp_liquid"]["value"]                 # J/(kg.K)
        self.UA_htf_total = u["UA_htf"]["value"]            # W/K total
        self.UA_htf_node = self.UA_htf_total / self.N       # W/K per node
        self.UA_loss = u["UA_loss"]["value"]                # W/K per node
        self.UA_node = u["UA_node"]["value"]                # W/K inter-node
        self.T_amb = u["T_amb"]["value"]                    # K
        self.T_ref = u["T_ref"]["value"]                    # K
        self.cp_htf = u["cp_htf"]["value"]                  # J/(kg.K)

    # ------------------------------------------------------------------
    # Temperature from specific enthalpy (relative to melt onset)
    # ------------------------------------------------------------------
    def enthalpy_to_temperature(self, h):
        """
        Convert specific enthalpy h [J/kg] to temperature [K].
        h is measured relative to the onset of melting:
          h < 0         -> solid
          0 <= h <= L_f -> melting (mushy zone)
          h > L_f       -> liquid
        """
        if h < 0:
            return self.T_melt + h / self.cp_s
        elif h <= self.L_f:
            return self.T_melt
        else:
            return self.T_melt + (h - self.L_f) / self.cp_l

    def temperature_to_enthalpy(self, T):
        """Convert temperature to specific enthalpy (assuming no mushy zone)."""
        if T < self.T_melt:
            return self.cp_s * (T - self.T_melt)
        elif T == self.T_melt:
            return 0.0
        else:
            return self.L_f + self.cp_l * (T - self.T_melt)

    def liquid_fraction(self, h):
        """Liquid fraction from specific enthalpy."""
        if h <= 0:
            return 0.0
        elif h >= self.L_f:
            return 1.0
        else:
            return h / self.L_f

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, H_vec, T_htf_func, m_dot_htf_func):
        """
        Right-hand side for enthalpy ODE.
        H_vec: array of specific enthalpies [J/kg] for each node.
        Returns dH/dt [J/(kg.s)] for each node.
        """
        N = self.N
        dHdt = np.zeros(N)
        T_htf = T_htf_func(t) if callable(T_htf_func) else T_htf_func
        m_dot = m_dot_htf_func(t) if callable(m_dot_htf_func) else m_dot_htf_func

        # Get temperatures of all nodes
        T = np.array([self.enthalpy_to_temperature(H_vec[i]) for i in range(N)])

        # HTF temperature along the flow path (counter-flow or co-flow)
        # Model as co-flow: HTF enters at node 0
        T_htf_local = np.zeros(N)
        T_htf_in = T_htf
        for i in range(N):
            # Effectiveness-NTU for each node segment
            if m_dot > 0:
                C_htf = m_dot * self.cp_htf
                NTU = self.UA_htf_node / C_htf
                eps = 1.0 - np.exp(-NTU)
                Q_htf = eps * C_htf * (T_htf_in - T[i])
                T_htf_out = T_htf_in - Q_htf / C_htf
                T_htf_local[i] = 0.5 * (T_htf_in + T_htf_out)
                T_htf_in = T_htf_out
            else:
                Q_htf = 0.0
                T_htf_local[i] = T_htf

            # Inter-node conduction
            Q_cond = 0.0
            if i > 0:
                Q_cond += self.UA_node * (T[i - 1] - T[i])
            if i < N - 1:
                Q_cond += self.UA_node * (T[i + 1] - T[i])

            # Heat loss to ambient
            Q_loss = self.UA_loss * (T[i] - self.T_amb)

            # HTF heat transfer to this node
            if m_dot > 0:
                C_htf = m_dot * self.cp_htf
                NTU = self.UA_htf_node / C_htf
                eps = 1.0 - np.exp(-NTU)
                # Recalculate with actual local HTF temp
                Q_htf_i = self.UA_htf_node * (T_htf_local[i] - T[i])
            else:
                Q_htf_i = 0.0

            dHdt[i] = (Q_htf_i + Q_cond - Q_loss) / self.m_node

        return dHdt

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, T_htf, m_dot_htf, T_init, dt, duration_s, mode="charge"):
        """
        Simulate PCM storage charge/discharge.

        Parameters
        ----------
        T_htf : float or callable(t)
            HTF inlet temperature [K].
        m_dot_htf : float or callable(t)
            HTF mass flow rate [kg/s].
        T_init : float
            Initial uniform PCM temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Simulation duration [s].
        mode : str
            'charge', 'discharge', or 'cycle' (informational label).

        Returns
        -------
        dict with time-series.
        """
        _T_htf = T_htf if callable(T_htf) else lambda t: T_htf
        _m_dot = m_dot_htf if callable(m_dot_htf) else lambda t: m_dot_htf

        # Initial enthalpy for each node
        H0 = np.array([self.temperature_to_enthalpy(T_init)] * self.N)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, _T_htf, _m_dot),
            (0.0, duration_s), H0,
            t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8,
            max_step=dt * 2
        )

        t_out = sol.t
        H_out = sol.y  # shape (N_nodes, N_time)
        Nt = len(t_out)

        # Post-process
        T_out = np.zeros_like(H_out)
        lf_out = np.zeros_like(H_out)
        for i in range(self.N):
            for k in range(Nt):
                T_out[i, k] = self.enthalpy_to_temperature(H_out[i, k])
                lf_out[i, k] = self.liquid_fraction(H_out[i, k])

        T_mean = np.mean(T_out, axis=0)
        lf_mean = np.mean(lf_out, axis=0)

        # Energy stored (relative to ambient)
        E_stored = np.zeros(Nt)
        for k in range(Nt):
            for i in range(self.N):
                h_ref = self.temperature_to_enthalpy(self.T_amb)
                E_stored[k] += self.m_node * (H_out[i, k] - h_ref)

        # HTF outlet temperature
        T_htf_out = np.zeros(Nt)
        for k in range(Nt):
            T_htf_in = _T_htf(t_out[k])
            mdot = _m_dot(t_out[k])
            T_in = T_htf_in
            for i in range(self.N):
                if mdot > 0:
                    C_htf = mdot * self.cp_htf
                    NTU = self.UA_htf_node / C_htf
                    eps = 1.0 - np.exp(-NTU)
                    Q = eps * C_htf * (T_in - T_out[i, k])
                    T_in = T_in - Q / C_htf
                else:
                    pass
            T_htf_out[k] = T_in

        # Power (rate of energy storage)
        Q_rate = np.gradient(E_stored, t_out)

        return {
            "t": t_out,
            "T_nodes": T_out,           # (N_nodes, Nt)
            "T_mean": T_mean,           # (Nt,)
            "H_nodes": H_out,           # (N_nodes, Nt)
            "liquid_fraction": lf_out,  # (N_nodes, Nt)
            "lf_mean": lf_mean,         # (Nt,)
            "E_stored_J": E_stored,     # (Nt,)
            "Q_rate_W": Q_rate,         # (Nt,)
            "T_htf_out": T_htf_out,     # (Nt,)
            "mode": mode,
        }
