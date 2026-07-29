"""
EC081 — Thermochemical Energy Storage — F1b Stratified Multi-Node Model

Thermochemical TES stores energy as chemical reaction enthalpy:
    Charging (dehydration):    AB + Q_in  → A + B  (endothermic)
    Discharging (hydration):   A + B       → AB + Q_out  (exothermic)

Physics additions over F1a (fully mixed):
  1. Multi-node reaction front tracking (N nodes along sorbent bed)
  2. Thermal losses: Q_loss_i = UA_i * (T_node_i - T_amb)
  3. Charge-discharge asymmetry: round-trip eta applied on CHARGE SIDE only
     (consistent with EC082 fix — η_rt not double-applied on discharge)
  4. Axial heat conduction between nodes (destratification)

Reaction extent per node:
    x_i ∈ [0, 1]:  0 = fully hydrated (stored), 1 = fully dehydrated (discharged)

Energy per node at full charge:
    E_node = mass_node * dH_rxn  [J]

Charging: fills reaction extent from bottom node upward (inlet at base).
Discharging: draws from top node downward (hot fluid contacts top).

Round-trip efficiency η_rt:
    Applied as effective charging power = P_charge * η_rt
    Discharge side is NOT penalised (η_rt on charge side only — avoids double-counting).

Reference:
    Kerskes, H. et al. (2012). Solar Energy 86, 2533-2542.
    N'Tsoukpoe, K.E. et al. (2009). Renew. Sustain. Energy Rev. 13, 2639-2652.
    Tatsidjodoung, P. et al. (2013). Renew. Sustain. Energy Rev. 18, 327-349.
"""

import numpy as np


class ThermochemicalTESF1b:
    """Multi-node thermochemical TES with reaction front, thermal losses, and η_rt on charge."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_kwh  = u["capacity_kwh"]["value"]
        self.dH_rxn        = u["dH_rxn_kj_kg"]["value"] * 1000.0   # J/kg
        self.mass_total    = u["mass_sorbent_kg"]["value"]           # kg
        self.Q_ch_max      = u["Q_charge_max_kw"]["value"] * 1000.0  # W
        self.Q_dis_max     = u["Q_discharge_max_kw"]["value"] * 1000.0 # W
        self.UA_loss       = u["UA_loss_w_per_k"]["value"]           # W/K
        self.T_amb_default = u["T_ambient_default"]["value"]         # degC
        self.N             = int(u["N_nodes"]["value"])
        self.k_axial       = u["k_axial_w_per_k"]["value"]           # W/K
        self.eta_rt        = u["eta_rt"]["value"]                    # round-trip η [0,1]
        # Note: η_rt is applied to charge side only — not discharge (avoids double-count).

        self.E_node_J = self.mass_total * self.dH_rxn / self.N
        self.m_node   = self.mass_total / self.N

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, q_charge_W, q_discharge_W, T_ambient,
                 duration_s, x_initial=None, dt=60.0):
        """
        Simulate thermochemical TES over a time period.

        Parameters
        ----------
        q_charge_W    : float — charging power [W, dehydration]
        q_discharge_W : float — discharge power [W, hydration demand]
        T_ambient     : float — ambient temperature [degC]
        duration_s    : float — simulation duration [s]
        x_initial     : array (N,) or float — initial reaction extent per node
                        (0 = fully hydrated/discharged, 1 = dehydrated/charged)
        dt            : float — time step [s]

        Returns
        -------
        dict with x_nodes, soc, Q_actual_charge_kw, Q_actual_discharge_kw,
                  Q_loss_kw, stratification_index, x_history
        """
        q_charge_W    = float(q_charge_W)
        q_discharge_W = float(q_discharge_W)
        T_ambient     = float(T_ambient)
        duration_s    = float(duration_s)

        N = self.N
        n_steps = max(1, int(duration_s / dt))

        # Initialize reaction extent
        if x_initial is None:
            x = np.zeros(N)
        elif np.isscalar(x_initial):
            x = np.full(N, float(x_initial))
        else:
            x = np.array(x_initial, dtype=float)
            if len(x) != N:
                raise ValueError(f"x_initial must have {N} elements")
        x = np.clip(x, 0.0, 1.0)
        x_history = [x.copy()]

        Q_ch_total   = 0.0
        Q_dis_total  = 0.0
        Q_loss_total = 0.0

        # Average node temperature approximation for loss calculation
        # Thermochemical reaction temperature ~ T_rxn (constant during phase)
        T_rxn = self.T_amb_default + 80.0  # typical reaction temperature offset [degC]
        UA_node = self.UA_loss / N
        k_ax = self.k_axial

        for _ in range(n_steps):
            dx = np.zeros(N)

            # --- Heat loss to ambient ---
            # Each node loses heat proportional to (T_rxn - T_amb)
            q_loss_node = UA_node * (T_rxn - T_ambient)  # W per node (positive = out)
            Q_loss_total += q_loss_node * N * dt

            # Heat loss drains charge (converts dehydrated sorbent back)
            for i in range(N - 1, -1, -1):
                if q_loss_node > 0 and x[i] > 0:
                    dx_loss = q_loss_node * dt / self.E_node_J
                    dx[i] -= min(dx_loss, x[i])

            # --- Charging: dehydrate from BOTTOM (node 0) upward ---
            # RATIONALE: η_rt applied charge-side only (EC082 fix pattern)
            if q_charge_W > 0:
                q_ch_avail = min(q_charge_W, self.Q_ch_max) * self.eta_rt
                q_ch_remain = q_ch_avail * dt
                for i in range(N):
                    if q_ch_remain <= 0:
                        break
                    x_unfilled = 1.0 - x[i]
                    if x_unfilled > 1e-9:
                        dE = min(q_ch_remain, x_unfilled * self.E_node_J)
                        dx[i] += dE / self.E_node_J
                        q_ch_remain -= dE
                Q_ch_total += (q_ch_avail * dt - max(q_ch_remain, 0.0))

            # --- Discharge: hydrate from TOP (node N-1) downward ---
            # No η_rt here — round-trip efficiency is fully accounted on charge side.
            if q_discharge_W > 0:
                q_dis_avail = min(q_discharge_W, self.Q_dis_max)
                q_dis_remain = q_dis_avail * dt
                for i in range(N - 1, -1, -1):
                    if q_dis_remain <= 0:
                        break
                    if x[i] > 1e-9:
                        dE = min(q_dis_remain, x[i] * self.E_node_J)
                        dx[i] -= dE / self.E_node_J
                        q_dis_remain -= dE
                Q_dis_total += (q_dis_avail * dt - max(q_dis_remain, 0.0))

            # --- Axial conduction (smears stratification slightly) ---
            dx_axial = np.zeros(N)
            for i in range(N):
                x_above = x[i - 1] if i > 0     else x[i]
                x_below = x[i + 1] if i < N - 1 else x[i]
                dx_axial[i] += k_ax * (x_above + x_below - 2.0 * x[i]) * dt / self.E_node_J

            x = np.clip(x + dx + dx_axial, 0.0, 1.0)
            x_history.append(x.copy())

        soc = float(np.mean(x))
        SI  = self._stratification_index(x)

        return {
            "x_nodes":               x,
            "soc":                   soc,
            "Q_actual_charge_kw":    Q_ch_total  / dt / n_steps / 1000.0,
            "Q_actual_discharge_kw": Q_dis_total / dt / n_steps / 1000.0,
            "Q_loss_kw":             Q_loss_total / duration_s / 1000.0,
            "stratification_index":  SI,
            "x_history":             np.array(x_history),
        }

    # ------------------------------------------------------------------
    # Stratification index
    # ------------------------------------------------------------------

    def _stratification_index(self, x_nodes):
        """
        Stratification index SI ∈ [0, 1].
        SI = 1 means fully stratified (all charged material at bottom, ideal for TCES).
        """
        N = len(x_nodes)
        soc = np.mean(x_nodes)
        if soc < 1e-6 or soc > 1.0 - 1e-6:
            return 1.0

        n_full  = int(soc * N)
        partial = soc * N - n_full
        x_ideal = np.zeros(N)
        for i in range(n_full):
            x_ideal[i] = 1.0
        if n_full < N:
            x_ideal[n_full] = partial

        mad_actual = np.mean(np.abs(x_nodes - x_ideal))
        mad_max    = 2.0 * soc * (1.0 - soc)
        if mad_max < 1e-10:
            return 1.0
        return float(np.clip(1.0 - mad_actual / mad_max, 0.0, 1.0))
