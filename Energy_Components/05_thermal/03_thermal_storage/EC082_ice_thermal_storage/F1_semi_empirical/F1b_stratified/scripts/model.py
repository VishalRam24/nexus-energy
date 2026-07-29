"""
EC082 — Ice Thermal Storage — F1b Stratified Model

Multi-node (default 8) stratified ice-on-coil TES.

Physical principles:
  - Ice forms from BOTTOM upward: glycol refrigerant coils are at the base.
    Cold enters bottom → ice nucleates at base first.
    Node numbering: node 0 = bottom (coldest), node N-1 = top.
  - Ice MELTS from TOP downward: warm return water contacts upper surface.
    During discharge, warm fluid enters at top; melt front moves down.
  - Each node tracks its ice fraction f_i ∈ [0, 1] (0 = fully melted, 1 = fully frozen).
  - The effective temperature of a node during phase change is T_phase_change (0 °C).
    Before/after phase change, nodes track sensible temperature.

Simplified approach (F1b semi-empirical):
  - During charging: Q_charge is distributed bottom-up proportional to
    unfilled ice fraction (preferential freezing from bottom).
  - During discharge: Q_discharge distributed top-down proportional to
    existing ice fraction (melt from top).
  - Heat loss UA_loss is split uniformly across nodes.
  - Axial conduction k_axial between adjacent nodes.

Stratification index:
    SI = 1 - (f_mean - f_profile_ideal) / f_mean
    where f_profile_ideal is a perfectly stratified profile
    (all ice at bottom, none at top).

References:
    ASHRAE (2020). HVAC Systems and Equipment, ch.51.
    MacPhee, D., Dincer, I. (2009). Int. J. Heat Mass Transfer 52, 1753-1762.
    Jekel, T.B., Mitchell, J.W., Klein, S.A. (1993). ASHRAE Trans. 99, 1016-1024.
"""

import numpy as np


class IceTESF1b:
    """Stratified ice TES — multi-node ice fraction tracking."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_kwh  = u["capacity_kwh"]["value"]
        self.T_pc          = u["T_phase_change"]["value"]          # degC
        self.h_fusion      = u["h_fusion"]["value"] * 1000.0       # J/kg
        self.mass_total    = u["mass_water_kg"]["value"]            # kg
        self.Q_ch_max      = u["Q_charge_max"]["value"] * 1000.0   # W
        self.Q_dis_max     = u["Q_discharge_max"]["value"] * 1000.0 # W
        self.UA_loss       = u["UA_loss"]["value"]                  # W/K
        self.T_amb_default = u["T_amb_default"]["value"]
        self.N             = int(u["N_nodes"]["value"])
        self.k_axial       = u["k_axial"]["value"]                  # W/K
        self.eta_ch        = u["charge_effectiveness"]["value"]
        self.eta_dis       = u["discharge_effectiveness"]["value"]

        # Energy per node at full ice
        self.E_node_J = self.mass_total * self.h_fusion / self.N
        self.m_node   = self.mass_total / self.N

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, q_charge_W, q_discharge_W, T_ambient,
                 duration_s, f_initial=None, dt=60.0):
        """
        Simulate stratified ice TES over a time period.

        Parameters
        ----------
        q_charge_W    : float — charging power [W, positive = freezing]
        q_discharge_W : float — discharge power [W, positive = melting]
        T_ambient     : float — ambient temperature [degC]
        duration_s    : float — simulation duration [s]
        f_initial     : array (N,) or float — initial ice fractions per node
                        (0=liquid, 1=fully frozen); default all zeros
        dt            : float — time step [s]

        Returns
        -------
        dict with f_nodes, soc, Q_actual_charge_kw, Q_actual_discharge_kw,
                  Q_loss_kw, stratification_index, f_history
        """
        q_charge_W    = float(q_charge_W)
        q_discharge_W = float(q_discharge_W)
        T_ambient     = float(T_ambient)
        duration_s    = float(duration_s)

        N = self.N
        n_steps = max(1, int(duration_s / dt))

        # Initialize ice fractions
        if f_initial is None:
            f = np.zeros(N)
        elif np.isscalar(f_initial):
            f = np.full(N, float(f_initial))
        else:
            f = np.array(f_initial, dtype=float)
            if len(f) != N:
                raise ValueError(f"f_initial must have {N} elements")

        f = np.clip(f, 0.0, 1.0)
        f_history = [f.copy()]

        Q_ch_total  = 0.0
        Q_dis_total = 0.0
        Q_loss_total = 0.0

        # Heat loss per node per second (constant UA, uniform split)
        UA_node = self.UA_loss / N
        # Axial conduction coefficient (W/K between adjacent nodes)
        k_ax = self.k_axial

        for _ in range(n_steps):
            df = np.zeros(N)

            # --- Heat loss from each node to ambient ---
            # Nodes at T_pc during phase change: heat loss tries to melt ice
            q_loss = UA_node * (T_ambient - self.T_pc)  # W per node (positive = heat in from ambient)
            Q_loss_total += q_loss * N * dt              # total heat infiltration

            # Heat infiltration melts ice in each node from top down
            for i in range(N - 1, -1, -1):
                if q_loss > 0 and f[i] > 0:
                    df_loss = q_loss * dt / self.E_node_J
                    df[i] -= min(df_loss, f[i])

            # --- Charging: freeze from BOTTOM (node 0) upward ---
            if q_charge_W > 0:
                q_ch_avail = min(q_charge_W, self.Q_ch_max) * self.eta_ch
                q_ch_remain = q_ch_avail * dt
                for i in range(N):         # bottom → top
                    if q_ch_remain <= 0:
                        break
                    f_unfilled = 1.0 - f[i]
                    if f_unfilled > 1e-9:
                        dE = min(q_ch_remain, f_unfilled * self.E_node_J)
                        df[i] += dE / self.E_node_J
                        q_ch_remain -= dE
                Q_ch_total += (q_ch_avail * dt - max(q_ch_remain, 0.0))

            # --- Discharge: melt from TOP (node N-1) downward ---
            if q_discharge_W > 0:
                q_dis_avail = min(q_discharge_W, self.Q_dis_max) * self.eta_dis
                q_dis_remain = q_dis_avail * dt
                for i in range(N - 1, -1, -1):   # top → bottom
                    if q_dis_remain <= 0:
                        break
                    if f[i] > 1e-9:
                        dE = min(q_dis_remain, f[i] * self.E_node_J)
                        df[i] -= dE / self.E_node_J
                        q_dis_remain -= dE
                Q_dis_total += (q_dis_avail * dt - max(q_dis_remain, 0.0))

            # --- Axial conduction (smears stratification slightly) ---
            df_axial = np.zeros(N)
            for i in range(N):
                f_above = f[i - 1] if i > 0     else f[i]
                f_below = f[i + 1] if i < N - 1 else f[i]
                # Conduction in terms of ice-fraction gradient
                df_axial[i] += k_ax * (f_above + f_below - 2.0 * f[i]) * dt / self.E_node_J

            f = np.clip(f + df + df_axial, 0.0, 1.0)
            f_history.append(f.copy())

        soc = float(np.mean(f))
        SI  = self._stratification_index(f)

        return {
            "f_nodes":               f,
            "soc":                   soc,
            "Q_actual_charge_kw":    Q_ch_total  / dt / n_steps / 1000.0,
            "Q_actual_discharge_kw": Q_dis_total / dt / n_steps / 1000.0,
            "Q_loss_kw":             Q_loss_total / duration_s / 1000.0,
            "stratification_index":  SI,
            "f_history":             np.array(f_history),
        }

    # ------------------------------------------------------------------
    # Stratification index
    # ------------------------------------------------------------------

    def _stratification_index(self, f_nodes):
        """
        Stratification index SI ∈ [0, 1].

        SI = 1 means fully stratified: all ice at bottom, liquid at top.
        SI = 0 means uniform (fully mixed).

        Computed as 1 - mean absolute deviation from ideal profile
        (0...0, 1...1 from top, filling bottom-up).
        """
        N = len(f_nodes)
        soc = np.mean(f_nodes)
        if soc < 1e-6 or soc > 1.0 - 1e-6:
            return 1.0   # trivial cases: all melted or all frozen

        # Ideal bottom-up profile with same total ice
        n_full  = int(soc * N)
        partial = soc * N - n_full
        f_ideal = np.zeros(N)
        for i in range(n_full):
            f_ideal[i] = 1.0
        if n_full < N:
            f_ideal[n_full] = partial

        mad_actual = np.mean(np.abs(f_nodes - f_ideal))
        mad_max    = 2.0 * soc * (1.0 - soc)   # max possible MAD for given SOC
        if mad_max < 1e-10:
            return 1.0
        return float(np.clip(1.0 - mad_actual / mad_max, 0.0, 1.0))
