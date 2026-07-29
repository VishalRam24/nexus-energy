"""
EC083 — Borehole Thermal Energy Storage (BTES) — F1b Stratified Multi-Node Model

BTES stores thermal energy in the ground via vertical boreholes.
Heat is injected/extracted by circulating fluid through U-tube heat exchangers.

Physics additions over F1a (fully mixed):
  1. Multi-node vertical stratification (N nodes, top hot, bottom cold in heat mode)
  2. Thermal losses to surrounding undisturbed ground: Q_loss_i = UA_i * (T_i - T_ground)
  3. Charge-discharge asymmetry: η_rt applied on CHARGE SIDE only
  4. Axial thermal conduction between nodes (ground conduction)
  5. Effective borehole thermal resistance (R_b) linking fluid to ground

Charge: hot fluid injected at top (node 0), flows downward → heats upper nodes first.
Discharge: cool fluid injected at bottom (node N-1), flows upward → draws from upper nodes.

Round-trip efficiency η_rt:
    Applied as effective_charging = Q_charge * η_rt (charge side only — not on discharge).

Reference:
    Hellstrom, G. (1991). Ground Heat Storage — Thermal Analysis of Duct Storage Systems.
    Chapuis, S. & Bernier, M. (2009). ASHRAE Trans. 115, 649-662.
    Pahud, D. & Matthey, B. (2001). Geothermics 30, 651-673.
"""

import numpy as np


class BoreholeTESF1b:
    """Multi-node BTES with stratification, ground losses, and η_rt on charge."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_kwh     = u["capacity_kwh"]["value"]
        self.volume_m3        = u["volume_m3"]["value"]
        self.cp_ground        = u["cp_ground_j_kgk"]["value"]        # J/(kg·K)
        self.rho_ground       = u["rho_ground_kg_m3"]["value"]       # kg/m3
        self.T_ground_far     = u["T_ground_undisturbed"]["value"]   # degC
        self.UA_loss          = u["UA_loss_w_per_k"]["value"]        # W/K
        self.Q_ch_max         = u["Q_charge_max_kw"]["value"] * 1000.0  # W
        self.Q_dis_max        = u["Q_discharge_max_kw"]["value"] * 1000.0 # W
        self.N                = int(u["N_nodes"]["value"])
        self.k_axial          = u["k_axial_w_per_k"]["value"]        # W/K
        self.eta_rt           = u["eta_rt"]["value"]
        # Note: η_rt applied charge side only — consistent with EC082 fix.

        # Mass per node
        self.m_node = self.rho_ground * self.volume_m3 / self.N
        # Energy capacity per node for 1K temperature change [J/K]
        self.C_node = self.m_node * self.cp_ground

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, q_charge_W, q_discharge_W, T_ambient,
                 duration_s, T_initial=None, dt=300.0):
        """
        Simulate BTES over a time period.

        Parameters
        ----------
        q_charge_W    : float — charging power [W, heat injection]
        q_discharge_W : float — discharge power [W, heat extraction]
        T_ambient     : float — ambient/surface temperature [degC] (affects top node)
        duration_s    : float — simulation duration [s]
        T_initial     : array (N,) or float — initial ground temperatures per node [degC]
        dt            : float — time step [s]

        Returns
        -------
        dict with T_nodes, soc, Q_actual_charge_kw, Q_actual_discharge_kw,
                  Q_loss_kw, stratification_index, T_history
        """
        q_charge_W    = float(q_charge_W)
        q_discharge_W = float(q_discharge_W)
        T_ambient     = float(T_ambient)
        duration_s    = float(duration_s)

        N = self.N
        n_steps = max(1, int(duration_s / dt))

        # Initialize temperatures
        T_max_design = self.T_ground_far + 30.0   # design max storage temp
        if T_initial is None:
            T = np.full(N, self.T_ground_far)
        elif np.isscalar(T_initial):
            T = np.full(N, float(T_initial))
        else:
            T = np.array(T_initial, dtype=float)
            if len(T) != N:
                raise ValueError(f"T_initial must have {N} elements")

        T_history = [T.copy()]
        Q_ch_total   = 0.0
        Q_dis_total  = 0.0
        Q_loss_total = 0.0

        UA_node = self.UA_loss / N
        k_ax    = self.k_axial

        for _ in range(n_steps):
            dT = np.zeros(N)

            # --- Thermal loss to undisturbed ground ---
            for i in range(N):
                q_loss_i = UA_node * (T[i] - self.T_ground_far)
                dT[i] -= q_loss_i * dt / self.C_node
                Q_loss_total += q_loss_i * dt

            # --- Charging: inject heat top-down ---
            # η_rt applied on charge side only (avoids double-counting)
            if q_charge_W > 0:
                q_ch_avail  = min(q_charge_W, self.Q_ch_max) * self.eta_rt
                q_ch_remain = q_ch_avail * dt
                # Distribute to nodes that are below T_max (top → bottom)
                for i in range(N):
                    if q_ch_remain <= 0:
                        break
                    headroom = (T_max_design - T[i]) * self.C_node
                    if headroom > 0:
                        dE = min(q_ch_remain, headroom)
                        dT[i] += dE / self.C_node
                        q_ch_remain -= dE
                Q_ch_total += (q_ch_avail * dt - max(q_ch_remain, 0.0))

            # --- Discharge: extract heat from top downward ---
            if q_discharge_W > 0:
                q_dis_avail  = min(q_discharge_W, self.Q_dis_max)
                q_dis_remain = q_dis_avail * dt
                for i in range(N):
                    if q_dis_remain <= 0:
                        break
                    available = (T[i] - self.T_ground_far) * self.C_node
                    if available > 0:
                        dE = min(q_dis_remain, available)
                        dT[i] -= dE / self.C_node
                        q_dis_remain -= dE
                Q_dis_total += (q_dis_avail * dt - max(q_dis_remain, 0.0))

            # --- Axial conduction ---
            dT_ax = np.zeros(N)
            for i in range(N):
                T_above = T[i - 1] if i > 0     else T_ambient
                T_below = T[i + 1] if i < N - 1 else self.T_ground_far
                dT_ax[i] += k_ax * (T_above + T_below - 2.0 * T[i]) * dt / self.C_node

            T = T + dT + dT_ax
            T_history.append(T.copy())

        # SOC based on stored thermal energy vs design capacity
        E_stored_J = np.sum((T - self.T_ground_far) * self.C_node)
        E_stored_J = max(0.0, E_stored_J)
        E_cap_J    = self.capacity_kwh * 3.6e6
        soc = float(np.clip(E_stored_J / E_cap_J, 0.0, 1.0))
        SI  = self._stratification_index(T)

        return {
            "T_nodes":               T,
            "soc":                   soc,
            "Q_actual_charge_kw":    Q_ch_total  / dt / n_steps / 1000.0,
            "Q_actual_discharge_kw": Q_dis_total / dt / n_steps / 1000.0,
            "Q_loss_kw":             Q_loss_total / duration_s / 1000.0,
            "stratification_index":  SI,
            "T_history":             np.array(T_history),
        }

    # ------------------------------------------------------------------
    # Stratification index
    # ------------------------------------------------------------------

    def _stratification_index(self, T_nodes):
        """SI = 1: fully stratified (hot top, cold bottom). SI = 0: uniform."""
        N = len(T_nodes)
        T_top = np.mean(T_nodes[:N // 2])
        T_bot = np.mean(T_nodes[N // 2:])
        T_span = np.max(T_nodes) - np.min(T_nodes)
        if T_span < 0.01:
            return 0.0
        SI = (T_top - T_bot) / T_span
        return float(np.clip(SI, 0.0, 1.0))
