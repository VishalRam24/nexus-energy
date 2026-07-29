"""
EC079 -- Molten Salt Thermal Energy Storage -- F1b Stratified 10-Node Model

Builds on F1a (fully mixed) by adding:
  - 10 vertical nodes with inter-node heat exchange
  - Temperature-dependent solar salt properties:
      rho(T) = 2090 - 0.636*T  [kg/m3, T in degC]
      cp(T)  = 1443 + 0.172*T  [J/(kg*K), T in degC]
  - Charge: hot fluid enters top (node 0), exits bottom (node 9)
  - Discharge: cold fluid enters bottom (node 9), exits top (node 0)
  - Inter-node destratification via effective axial conductivity
  - Wall heat loss per node (U_wall * A_node * (T_node - T_amb))
  - Freezing constraint at 220 degC

Energy balance per node i:
    m_i * cp_i * dT_i/dt = Q_flow_i + Q_destrat_i - Q_loss_i

where Q_flow_i depends on mode (charge/discharge/idle).

References:
    Herrmann, U., Kelly, B., Price, H. (2004). Energy, 29(5-6), 883-893.
    Pacheco, J.E., Showalter, S.K., Kolb, W.J. (2002). ASME J. Solar Energy Eng. 124(2), 153-159.
    Zaversky, F. et al. (2013). Applied Energy, 109, 190-200.
"""

import numpy as np


class MoltenSaltTESF1b:
    """10-node stratified molten salt TES with temperature-dependent properties."""

    def __init__(self, params: dict):
        t = params["tank"]
        sp = params["salt_properties"]

        self.volume = t["volume_m3"]["value"]           # m3
        self.height = t["tank_height_m"]["value"]       # m
        self.n_nodes = int(t["n_nodes"]["value"])       # 10
        self.U_wall = t["U_wall_W_m2K"]["value"]        # W/(m2*K)
        self.k_destrat = t["k_destratification_W_mK"]["value"]  # W/(m*K)
        self.T_hot_design = t["T_hot_design_C"]["value"]    # degC
        self.T_cold_design = t["T_cold_design_C"]["value"]  # degC
        self.T_freeze = t["T_freeze_C"]["value"]            # degC
        self.T_amb_ref = t["T_ambient_ref_C"]["value"]      # degC

        # Salt property coefficients
        self.rho_a = sp["rho_coeff_a"]["value"]  # 2090
        self.rho_b = sp["rho_coeff_b"]["value"]  # -0.636
        self.cp_a = sp["cp_coeff_a"]["value"]     # 1443
        self.cp_b = sp["cp_coeff_b"]["value"]     # 0.172

        # Geometry per node
        self.dz = self.height / self.n_nodes       # node height [m]
        self.A_cross = self.volume / self.height    # cross-sectional area [m2]
        self.V_node = self.volume / self.n_nodes    # volume per node [m3]

        # Wall area per node (cylindrical: pi*D*dz, D = sqrt(4*A_cross/pi))
        D = np.sqrt(4.0 * self.A_cross / np.pi)
        self.A_wall_node = np.pi * D * self.dz      # m2

    def rho(self, T_c):
        """Solar salt density [kg/m3] as function of temperature [degC]."""
        return self.rho_a + self.rho_b * np.asarray(T_c, dtype=float)

    def cp(self, T_c):
        """Solar salt specific heat [J/(kg*K)] as function of temperature [degC]."""
        return self.cp_a + self.cp_b * np.asarray(T_c, dtype=float)

    def node_mass(self, T_nodes):
        """Mass of salt in each node [kg]."""
        return self.rho(T_nodes) * self.V_node

    def predict(self, T_charge_degC, T_discharge_degC, flow_rate_kg_s,
                mode, T_ambient_degC=None, duration_s=3600.0,
                T_nodes_init=None):
        """
        Run the stratified model for a single operating step.

        Args:
            T_charge_degC:   Inlet temperature during charging [degC]
            T_discharge_degC: Inlet temperature during discharging [degC]
            flow_rate_kg_s:  Mass flow rate [kg/s]
            mode:            'charge', 'discharge', or 'idle'
            T_ambient_degC:  Ambient temperature [degC]
            duration_s:      Duration of the step [s]
            T_nodes_init:    Initial node temperatures (array of n_nodes) [degC],
                             defaults to uniform at midpoint

        Returns:
            dict with: T_nodes, T_outlet_degC, stored_energy_kwh,
                       thermal_efficiency, freeze_warning
        """
        if T_ambient_degC is None:
            T_ambient_degC = self.T_amb_ref

        n = self.n_nodes

        # Initialize node temperatures
        if T_nodes_init is not None:
            T = np.array(T_nodes_init, dtype=float).copy()
        else:
            T_mid = (self.T_hot_design + self.T_cold_design) / 2.0
            T = np.full(n, T_mid)

        # Time integration (Euler, dt=10s substeps)
        dt = 10.0
        n_steps = max(1, int(duration_s / dt))
        dt = duration_s / n_steps

        # Track energy in/out for efficiency
        E_in_total = 0.0
        E_out_total = 0.0

        for _ in range(n_steps):
            T_new = T.copy()
            cp_arr = self.cp(T)
            m_arr = self.node_mass(T)

            for i in range(n):
                # --- Flow heat exchange ---
                Q_flow = 0.0
                if mode == "charge" and flow_rate_kg_s > 0:
                    # Hot fluid enters node 0 (top), flows down
                    if i == 0:
                        T_in = T_charge_degC
                    else:
                        T_in = T_new[i - 1]
                    cp_flow = self.cp(0.5 * (T_in + T[i]))
                    Q_flow = flow_rate_kg_s * cp_flow * (T_in - T[i])

                elif mode == "discharge" and flow_rate_kg_s > 0:
                    # Cold fluid enters node n-1 (bottom), flows up
                    if i == n - 1:
                        T_in = T_discharge_degC
                    else:
                        T_in = T_new[i + 1]
                    cp_flow = self.cp(0.5 * (T_in + T[i]))
                    Q_flow = flow_rate_kg_s * cp_flow * (T_in - T[i])

                # --- Destratification (inter-node conduction) ---
                Q_destrat = 0.0
                if i > 0:
                    Q_destrat += self.k_destrat * self.A_cross * (T[i - 1] - T[i]) / self.dz
                if i < n - 1:
                    Q_destrat += self.k_destrat * self.A_cross * (T[i + 1] - T[i]) / self.dz

                # --- Wall heat loss ---
                Q_loss = self.U_wall * self.A_wall_node * (T[i] - T_ambient_degC)

                # --- Update ---
                dT = (Q_flow + Q_destrat - Q_loss) * dt / (m_arr[i] * cp_arr[i])
                T_new[i] = T[i] + dT

                # Track energy flows
                if Q_flow > 0:
                    E_in_total += Q_flow * dt
                else:
                    E_out_total += abs(Q_flow) * dt

            # Enforce freeze constraint
            T_new = np.maximum(T_new, self.T_freeze)
            # Cap at hot design + small margin
            T_new = np.minimum(T_new, self.T_hot_design + 20.0)

            T = T_new

        # --- Compute outputs ---
        # Outlet temperature
        if mode == "charge":
            T_outlet = float(T[-1])   # exits bottom
        elif mode == "discharge":
            T_outlet = float(T[0])    # exits top
        else:
            T_outlet = float(np.mean(T))

        # Stored energy relative to cold design [kWh]
        stored_energy_kwh = 0.0
        for i in range(n):
            m_i = self.rho(T[i]) * self.V_node
            cp_i = self.cp(T[i])
            stored_energy_kwh += m_i * cp_i * max(0.0, T[i] - self.T_cold_design)
        stored_energy_kwh /= 3.6e6  # J -> kWh

        # Thermal efficiency
        if E_in_total > 0:
            thermal_efficiency = E_out_total / E_in_total
        elif mode == "idle":
            # For idle, compute fraction of energy retained
            E_initial = sum(
                self.rho(T_nodes_init[i] if T_nodes_init is not None else (self.T_hot_design + self.T_cold_design) / 2.0)
                * self.V_node
                * self.cp(T_nodes_init[i] if T_nodes_init is not None else (self.T_hot_design + self.T_cold_design) / 2.0)
                * max(0.0, (T_nodes_init[i] if T_nodes_init is not None else (self.T_hot_design + self.T_cold_design) / 2.0) - self.T_cold_design)
                for i in range(n)
            )
            E_final = stored_energy_kwh * 3.6e6
            thermal_efficiency = E_final / E_initial if E_initial > 0 else 1.0
        else:
            thermal_efficiency = 0.0

        thermal_efficiency = float(np.clip(thermal_efficiency, 0.0, 1.0))

        # Freeze warning
        freeze_warning = bool(np.any(T < self.T_freeze + 10.0))

        return {
            "T_nodes": T.tolist(),
            "T_outlet_degC": T_outlet,
            "stored_energy_kwh": float(stored_energy_kwh),
            "thermal_efficiency": thermal_efficiency,
            "freeze_warning": freeze_warning,
        }

    def simulate(self, T_nodes_init, mode_schedule, flow_schedule,
                 T_charge_schedule, T_discharge_schedule,
                 T_ambient_schedule=None, dt_step_s=3600.0):
        """
        Multi-step simulation over a schedule.

        Args:
            T_nodes_init:        Initial node temps [degC], array of n_nodes
            mode_schedule:       List of mode strings per time step
            flow_schedule:       Array of flow rates [kg/s] per step
            T_charge_schedule:   Array of charge inlet temps [degC]
            T_discharge_schedule: Array of discharge inlet temps [degC]
            T_ambient_schedule:  Array of ambient temps [degC] (or scalar)
            dt_step_s:           Duration per step [s]

        Returns:
            dict with arrays: T_nodes_history, T_outlet, stored_energy_kwh,
                              thermal_efficiency, freeze_warning
        """
        N = len(mode_schedule)
        if T_ambient_schedule is None:
            T_amb_arr = np.full(N, self.T_amb_ref)
        else:
            T_amb_arr = np.asarray(T_ambient_schedule, dtype=float)
            if T_amb_arr.ndim == 0:
                T_amb_arr = np.full(N, float(T_amb_arr))

        T_current = np.array(T_nodes_init, dtype=float).copy()
        history = {
            "T_nodes_history": [],
            "T_outlet": [],
            "stored_energy_kwh": [],
            "thermal_efficiency": [],
            "freeze_warning": [],
        }

        for i in range(N):
            result = self.predict(
                T_charge_degC=float(T_charge_schedule[i]),
                T_discharge_degC=float(T_discharge_schedule[i]),
                flow_rate_kg_s=float(flow_schedule[i]),
                mode=mode_schedule[i],
                T_ambient_degC=float(T_amb_arr[i]),
                duration_s=dt_step_s,
                T_nodes_init=T_current,
            )
            T_current = np.array(result["T_nodes"])
            history["T_nodes_history"].append(result["T_nodes"])
            history["T_outlet"].append(result["T_outlet_degC"])
            history["stored_energy_kwh"].append(result["stored_energy_kwh"])
            history["thermal_efficiency"].append(result["thermal_efficiency"])
            history["freeze_warning"].append(result["freeze_warning"])

        for k in history:
            if k != "T_nodes_history":
                history[k] = np.array(history[k])

        return history
