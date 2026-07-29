"""
EC078 — Hot Water Tank TES — F1b Multi-Node Stratification Model

N-node (default 10) vertical stratification model.

Each node i has energy balance:
  m_node * cp * dT_i/dt = Q_in_i - Q_out_i
                         - UA_node * (T_i - T_amb)
                         + k_mix * (T_{i-1} + T_{i+1} - 2*T_i)

Charging: hot water enters at top (node 1), exits at bottom (node N).
Discharging: cold water enters at bottom (node N), hot drawn from top (node 1).

Inter-node mixing term k_mix models thermal conduction through water
and tank wall, plus turbulent mixing from flow. This degrades stratification.

Stratification efficiency = actual energy recoverable at T > T_threshold
                          / energy stored if perfectly stratified.

References:
    Duffie, J.A., Beckman, W.A. (2013). Solar Engineering of Thermal Processes, 4th ed., ch.8.
    De Cesaro Oliveski et al. (2003). Applied Thermal Engineering, 23, 1293-1302.
    Newton, B.J. (1995). TRNSYS Type 60 — Multinode stratified storage tank.
"""

import numpy as np


class HotWaterTankF1b:
    """Multi-node stratified hot water tank."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_L = u["volume_L"]["value"]               # L total
        self.N = int(u["N_nodes"]["value"])              # number of nodes
        self.UA_total = u["UA_loss"]["value"]            # W/K total
        self.k_mix = u["k_mix"]["value"]                 # W/K inter-node
        self.T_min = u["T_min"]["value"]                 # degC
        self.T_max = u["T_max"]["value"]                 # degC
        self.cp = u["cp_water"]["value"]                 # J/(kg.K)
        self.rho = u["rho_water"]["value"]               # kg/L
        self.T_init = u["T_initial"]["value"]            # degC

        # Derived
        self.m_node = self.V_L * self.rho / self.N       # kg per node
        self.UA_node = self.UA_total / self.N             # W/K per node

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def simulate(self, T_inlet_hot, T_inlet_cold, flow_charge, flow_discharge,
                 T_ambient, duration_s, T_initial=None, dt=10.0):
        """
        Simulate stratified tank over a time period.

        Parameters
        ----------
        T_inlet_hot : float — hot water inlet temperature (degC)
        T_inlet_cold : float — cold makeup water temperature (degC)
        flow_charge : float — charging mass flow rate (kg/s)
        flow_discharge : float — discharging mass flow rate (kg/s)
        T_ambient : float — ambient temperature (degC)
        duration_s : float — simulation duration (s)
        T_initial : array-like or float — initial node temperatures (degC)
        dt : float — time step (s)

        Returns
        -------
        dict with T_nodes, T_outlet_hot, T_outlet_cold, stored_energy_kwh,
             stratification_efficiency, T_history
        """
        T_inlet_hot = float(T_inlet_hot)
        T_inlet_cold = float(T_inlet_cold)
        flow_charge = float(flow_charge)
        flow_discharge = float(flow_discharge)
        T_ambient = float(T_ambient)
        duration_s = float(duration_s)

        N = self.N
        n_steps = max(1, int(duration_s / dt))

        # Initialize node temperatures
        if T_initial is None:
            T = np.full(N, self.T_init)
        elif np.isscalar(T_initial):
            T = np.full(N, float(T_initial))
        else:
            T = np.array(T_initial, dtype=float)
            if len(T) != N:
                raise ValueError(f"T_initial must have {N} elements, got {len(T)}")

        T_history = [T.copy()]

        for _ in range(n_steps):
            dT = np.zeros(N)

            # Heat loss to ambient
            for i in range(N):
                dT[i] -= self.UA_node * (T[i] - T_ambient) / (self.m_node * self.cp)

            # Inter-node mixing (conduction + turbulent mixing)
            for i in range(N):
                T_above = T[i - 1] if i > 0 else T[i]
                T_below = T[i + 1] if i < N - 1 else T[i]
                dT[i] += self.k_mix * (T_above + T_below - 2.0 * T[i]) / (self.m_node * self.cp)

            # Charging: hot water in at top (node 0), displaces down, cold out at bottom
            if flow_charge > 0:
                # Flow from top to bottom
                # Node 0 receives hot water
                dT[0] += flow_charge * self.cp * (T_inlet_hot - T[0]) / (self.m_node * self.cp)
                # Each subsequent node receives flow from above
                for i in range(1, N):
                    dT[i] += flow_charge * self.cp * (T[i - 1] - T[i]) / (self.m_node * self.cp)

            # Discharging: cold water in at bottom (node N-1), hot drawn from top (node 0)
            if flow_discharge > 0:
                # Flow from bottom to top
                # Node N-1 receives cold water
                dT[N - 1] += flow_discharge * self.cp * (T_inlet_cold - T[N - 1]) / (self.m_node * self.cp)
                # Each node above receives flow from below
                for i in range(N - 2, -1, -1):
                    dT[i] += flow_discharge * self.cp * (T[i + 1] - T[i]) / (self.m_node * self.cp)

            # Euler step
            T = T + dT * dt
            T = np.clip(T, self.T_min, self.T_max)

            # Buoyancy correction: if a lower node is hotter than the one above, swap
            for i in range(N - 1):
                if T[i + 1] > T[i]:
                    T[i], T[i + 1] = T[i + 1], T[i]

            T_history.append(T.copy())

        # Final outputs
        T_outlet_hot = T[0]         # drawn from top
        T_outlet_cold = T[N - 1]    # exits from bottom

        # Stored energy relative to T_min
        stored_energy_J = np.sum(self.m_node * self.cp * (T - self.T_min))
        stored_energy_kwh = stored_energy_J / 3.6e6

        # Stratification efficiency
        strat_eff = self._stratification_efficiency(T, T_inlet_cold)

        return {
            "T_nodes": T,
            "T_outlet_hot": T_outlet_hot,
            "T_outlet_cold": T_outlet_cold,
            "stored_energy_kwh": stored_energy_kwh,
            "stratification_efficiency": strat_eff,
            "T_history": np.array(T_history),
        }

    # ------------------------------------------------------------------
    # Stratification efficiency
    # ------------------------------------------------------------------

    def _stratification_efficiency(self, T_nodes, T_cold_ref):
        """
        Stratification efficiency: fraction of stored energy that is
        recoverable above a useful threshold (T_cold_ref + 5K).

        eta_strat = sum(max(T_i - T_threshold, 0)) / sum(max(T_i - T_cold_ref, 0))
        """
        T_threshold = T_cold_ref + 5.0
        energy_total = np.sum(np.maximum(T_nodes - T_cold_ref, 0.0))
        energy_useful = np.sum(np.maximum(T_nodes - T_threshold, 0.0))

        if energy_total < 1e-10:
            return 0.0
        return float(energy_useful / energy_total)
