"""
EC084 — Aquifer Thermal Energy Storage (ATES) — F1b Stratified Multi-Node Model

ATES stores thermal energy in a groundwater aquifer using warm/cold well pairs.
During summer: warm water pumped to warm well (charging), cold from cold well.
During winter: warm water extracted from warm well (discharging), injected cold.

Physics additions over F1a (fully mixed):
  1. Multi-node thermal plume tracking (N nodes representing radial/vertical zones)
  2. Thermal losses: Q_loss_i = UA_i * (T_i - T_aquifer_undisturbed)
     (heat loss to surrounding aquifer by conduction/dispersion)
  3. Charge-discharge asymmetry: η_rt applied on CHARGE SIDE only
  4. Axial thermal dispersion between nodes

Node numbering: node 0 = warm well (highest T), node N-1 = cold well (lowest T).

Round-trip efficiency η_rt:
    Accounts for thermal dispersion and mixing during injection/extraction.
    Applied charge side only — EC082 fix pattern.

Reference:
    Sanner, B. et al. (2003). Geothermics 32, 579-588.
    Bloemendal, M. & Hartog, N. (2018). Renewable Energy 120, 39-50.
    Dincer, I. & Rosen, M.A. (2011). Thermal Energy Storage, Wiley.
"""

import numpy as np


class AquiferTESF1b:
    """Multi-node ATES with plume stratification, aquifer losses, and η_rt on charge."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_kwh       = u["capacity_kwh"]["value"]
        self.aquifer_volume_m3  = u["aquifer_volume_m3"]["value"]
        self.cp_water           = u["cp_water_j_kgk"]["value"]
        self.rho_water          = u["rho_water_kg_m3"]["value"]
        self.T_aquifer_natural  = u["T_aquifer_natural"]["value"]   # degC
        self.UA_loss            = u["UA_loss_w_per_k"]["value"]     # W/K
        self.Q_ch_max           = u["Q_charge_max_kw"]["value"] * 1000.0  # W
        self.Q_dis_max          = u["Q_discharge_max_kw"]["value"] * 1000.0 # W
        self.N                  = int(u["N_nodes"]["value"])
        self.k_dispersion       = u["k_dispersion_w_per_k"]["value"]  # W/K thermal dispersion
        self.eta_rt             = u["eta_rt"]["value"]
        # Note: η_rt applied charge side only — consistent with EC082 fix.

        self.V_node = self.aquifer_volume_m3 / self.N
        self.m_node = self.rho_water * self.V_node
        self.C_node = self.m_node * self.cp_water   # J/K

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, q_charge_W, q_discharge_W, T_ambient,
                 duration_s, T_initial=None, dt=600.0):
        """
        Simulate ATES over a time period.

        Parameters
        ----------
        q_charge_W    : float — charging power [W, heat injection to warm well]
        q_discharge_W : float — discharge power [W, heat extraction from warm well]
        T_ambient     : float — surface/ambient temperature [degC]
        duration_s    : float — simulation duration [s]
        T_initial     : array (N,) or float — initial aquifer zone temperatures [degC]
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

        T_max_warm = self.T_aquifer_natural + 30.0  # max warm-well temp
        T_min_cold = self.T_aquifer_natural - 10.0  # min cold-well temp

        # Initialize temperatures
        if T_initial is None:
            T = np.full(N, self.T_aquifer_natural)
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

        UA_node  = self.UA_loss / N
        k_disp   = self.k_dispersion

        for _ in range(n_steps):
            dT = np.zeros(N)

            # --- Thermal loss/gain to undisturbed aquifer ---
            for i in range(N):
                q_loss_i = UA_node * (T[i] - self.T_aquifer_natural)
                dT[i] -= q_loss_i * dt / self.C_node
                Q_loss_total += q_loss_i * dt

            # --- Charging: inject heat at warm-well end (node 0) ---
            # η_rt applied charge side only
            if q_charge_W > 0:
                q_ch_avail  = min(q_charge_W, self.Q_ch_max) * self.eta_rt
                q_ch_remain = q_ch_avail * dt
                for i in range(N):
                    if q_ch_remain <= 0:
                        break
                    headroom = (T_max_warm - T[i]) * self.C_node
                    if headroom > 0:
                        dE = min(q_ch_remain, headroom)
                        dT[i] += dE / self.C_node
                        q_ch_remain -= dE
                Q_ch_total += (q_ch_avail * dt - max(q_ch_remain, 0.0))

            # --- Discharge: extract heat from warm-well end (node 0 first) ---
            if q_discharge_W > 0:
                q_dis_avail  = min(q_discharge_W, self.Q_dis_max)
                q_dis_remain = q_dis_avail * dt
                for i in range(N):
                    if q_dis_remain <= 0:
                        break
                    available = (T[i] - self.T_aquifer_natural) * self.C_node
                    if available > 0:
                        dE = min(q_dis_remain, available)
                        dT[i] -= dE / self.C_node
                        q_dis_remain -= dE
                Q_dis_total += (q_dis_avail * dt - max(q_dis_remain, 0.0))

            # --- Thermal dispersion between nodes ---
            dT_disp = np.zeros(N)
            for i in range(N):
                T_left  = T[i - 1] if i > 0     else self.T_aquifer_natural
                T_right = T[i + 1] if i < N - 1 else self.T_aquifer_natural
                dT_disp[i] += k_disp * (T_left + T_right - 2.0 * T[i]) * dt / self.C_node

            T = T + dT + dT_disp
            T = np.maximum(T, T_min_cold)
            T_history.append(T.copy())

        # SOC: fraction of design capacity stored
        E_stored_J = np.sum(np.maximum(T - self.T_aquifer_natural, 0.0) * self.C_node)
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
        """SI = 1: warm well end hottest. SI = 0: uniform temperature."""
        T_span = float(np.max(T_nodes) - np.min(T_nodes))
        if T_span < 0.01:
            return 0.0
        T_warm_end = float(T_nodes[0])
        T_cold_end = float(T_nodes[-1])
        SI = (T_warm_end - T_cold_end) / T_span
        return float(np.clip(SI, 0.0, 1.0))
