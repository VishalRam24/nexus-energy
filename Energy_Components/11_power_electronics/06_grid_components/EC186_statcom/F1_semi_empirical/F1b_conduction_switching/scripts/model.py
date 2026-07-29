"""
EC186 -- STATCOM -- F1b Conduction + Switching Loss Model

Extends F1a with an IGBT-level loss breakdown for a VSC-based STATCOM:

**IGBT Conduction Losses (per device):**
    P_cond_igbt = V_ce0 * I_avg + r_ce(T_j) * I_rms^2
    where:
        r_ce(T_j) = r_ce_ref * (1 + alpha * (T_j - T_j_ref))
    For a purely reactive converter (power factor ≈ 0 from the AC grid perspective,
    but IGBT current is non-zero):
        I_rms = Q * 1e6 / (sqrt(3) * V_LL)   [A]
        I_avg = I_rms * sqrt(2) / pi            [A] (sinusoidal)

**IGBT Switching Losses (per device):**
    P_sw_igbt = E_sw * f_sw * (V_dc / V_ref) * (I_sw / I_ref)
    where I_sw = I_rms * sqrt(2) / pi (avg current during switching events)

**Diode (freewheeling) Losses:**
    Approximately 30% of IGBT losses for modern IGBTs (IEC 62927 guidance).
    P_diode = 0.30 * (P_cond_igbt + P_sw_igbt)

**Transformer copper losses:**
    P_transformer = R_pu * Q^2 / Q_rated^2 * Q_rated   [approx, per unit]
    Exact: P_tr = R_pu * S^2 / S_rated  where S = Q (reactive only)

**Total:**
    P_loss_MW = n_devices * (P_cond + P_sw + P_diode) / 1e6 + P_transformer + P_standby

Note: For a purely reactive STATCOM, P_active_delivered = 0, and all converter
current produces only reactive power. The losses are real power drawn from the grid.

References:
    Hingorani & Gyugyi (2000). Understanding FACTS. IEEE Press. Chapter 6.
    Cigre TB 401 (2009). VSC Transmission Systems.
    IEC 62927:2018. Voltage sourced converter (VSC) valves for STATCOM.
    Semikron (2021). IGBT Module Application Manual.
    Bahrman & Johnson (2007). The ABCs of HVDC transmission technologies.
    IEEE Power & Energy Magazine, 5(2), 32-44.
"""

import numpy as np


class STATCOMF1b:
    """VSC-based STATCOM -- IGBT conduction + switching loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_max = u["Q_max_MVAR"]["value"]
        self.Q_min = u["Q_min_MVAR"]["value"]
        self.V_rated_kV = u["V_rated_kV"]["value"]
        self.S_rated = u["S_rated_MVA"]["value"]
        self.V_dc = u["V_dc_kV"]["value"] * 1000.0    # V
        self.V_ce0 = u["IGBT_V_ce0"]["value"]
        self.r_ce_ref = u["IGBT_r_ce"]["value"]
        self.r_ce_alpha = u["IGBT_r_ce_alpha"]["value"]
        self.T_j_ref = u["IGBT_T_j_ref"]["value"]
        self.E_sw = u["IGBT_E_sw"]["value"]
        self.V_ref = u["IGBT_V_ref"]["value"]
        self.I_ref = u["IGBT_I_ref"]["value"]
        self.f_sw = u["f_sw_Hz"]["value"]
        self.n_igbt_per_phase = u["n_IGBT_per_phase"]["value"]
        self.n_phases = u["n_phases"]["value"]
        self.R_tr_pu = u["transformer_R_pu"]["value"]
        self.P_standby = u["P_standby_MW"]["value"]
        self.T_j_default = u["T_j"]["value"]

        self.n_devices = self.n_phases * self.n_igbt_per_phase  # total IGBT count

    def r_ce(self, T_j: float) -> float:
        """Temperature-dependent IGBT on-resistance."""
        T = np.asarray(T_j, dtype=float)
        return self.r_ce_ref * (1.0 + self.r_ce_alpha * (T - self.T_j_ref))

    def igbt_losses_per_device(self, I_rms: float, T_j: float = 125.0) -> tuple:
        """
        IGBT conduction + switching loss per device [W].
        Returns (P_cond_W, P_sw_W)
        """
        I_rms = np.asarray(I_rms, dtype=float)
        I_avg = I_rms * np.sqrt(2.0) / np.pi
        r = self.r_ce(T_j)
        P_cond = self.V_ce0 * I_avg + r * I_rms ** 2
        I_sw = I_avg  # switching at avg current (conservative approximation)
        P_sw = self.E_sw * self.f_sw * (self.V_dc / self.V_ref) * (I_sw / self.I_ref)
        return P_cond, P_sw

    def compute(self, Q_demand_MVAR: float, V_pu: float = 1.0,
                T_j: float = None) -> dict:
        """
        Parameters
        ----------
        Q_demand_MVAR : Requested reactive output [MVAR]; + = capacitive
        V_pu          : Terminal voltage [pu] (STATCOM Q independent of V -- VSC property)
        T_j           : IGBT junction temperature [degC] (default from params)

        Returns
        -------
        dict with:
            Q_out_MVAR, Q_limited,
            P_cond_MW, P_sw_MW, P_diode_MW, P_transformer_MW, P_standby_MW, P_total_loss_MW,
            I_rms_A, r_ce_Ohm, operating_mode, utilization
        """
        if T_j is None:
            T_j = self.T_j_default

        Q_dem = np.asarray(Q_demand_MVAR, dtype=float)
        V_pu = np.asarray(V_pu, dtype=float)

        Q_out = np.clip(Q_dem, self.Q_min, self.Q_max)
        Q_limited = Q_dem != Q_out

        # VSC property: Q_out is INDEPENDENT of V_pu (unlike SVC)
        # (The converter controls current, not admittance)
        # I_rms from converter current (all reactive)
        V_V = self.V_rated_kV * 1000.0
        I_rms = np.abs(Q_out) * 1e6 / (np.sqrt(3.0) * V_V)

        # IGBT losses
        P_cond_per, P_sw_per = self.igbt_losses_per_device(I_rms, T_j)
        P_diode_per = 0.30 * (P_cond_per + P_sw_per)  # diode ~ 30% of IGBT (IEC 62927)

        P_cond_MW = self.n_devices * P_cond_per / 1e6
        P_sw_MW = self.n_devices * P_sw_per / 1e6
        P_diode_MW = self.n_devices * P_diode_per / 1e6

        # Transformer copper loss: P_tr = R_pu * (|Q|/Q_rated)^2 * S_rated
        P_tr_MW = self.R_tr_pu * (np.abs(Q_out) / (self.S_rated + 1e-12)) ** 2 * self.S_rated

        P_total_loss = P_cond_MW + P_sw_MW + P_diode_MW + P_tr_MW + self.P_standby

        # Mode
        if np.ndim(Q_out) == 0:
            if float(Q_out) > 0.01:
                mode = "capacitive"
            elif float(Q_out) < -0.01:
                mode = "inductive"
            else:
                mode = "standby"
        else:
            mode = "vectorized"

        Q_range = self.Q_max
        utilization = np.abs(Q_out) / (Q_range + 1e-12)

        return {
            "Q_out_MVAR": Q_out,
            "Q_limited": Q_limited,
            "P_cond_MW": P_cond_MW,
            "P_sw_MW": P_sw_MW,
            "P_diode_MW": P_diode_MW,
            "P_transformer_MW": P_tr_MW,
            "P_standby_MW": np.full_like(P_cond_MW, self.P_standby) if np.ndim(P_cond_MW) > 0 else self.P_standby,
            "P_total_loss_MW": P_total_loss,
            "I_rms_A": I_rms,
            "r_ce_Ohm": self.r_ce(T_j),
            "operating_mode": mode,
            "utilization": utilization,
        }
