"""
EC186 — STATCOM — F1a Reactive Compensation Model

VSC-based STATCOM differences from SVC:
1. Symmetric Q range: Q ∈ [-Q_max, +Q_max] (TCR not needed for inductive)
2. Q_out independent of terminal voltage (VSC maintains current, not admittance)
3. Faster response (~5 ms vs ~30 ms for SVC)
4. Higher losses (~1.5% vs ~1% for SVC) due to VSC switching

P_loss = P_standby + loss_factor * |Q_out|   [MW]

Reference:
    Hingorani, N.G. & Gyugyi, L. (2000). Understanding FACTS. IEEE Press. Chapter 6.
    IEEE Std 2800-2022: Interconnection Requirements for Inverter-Based Resources.
"""

import numpy as np


class STATCOMModel:
    """VSC-based STATCOM F1a model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_max = u["Q_max_MVAR"]["value"]       # MVAR
        self.Q_min = u["Q_min_MVAR"]["value"]        # MVAR (negative)
        self.loss_factor = u["loss_factor"]["value"]
        self.P_standby = u["P_standby_MW"]["value"]  # MW

    def compute(self, Q_demand_MVAR: float, V_pu: float = 1.0) -> dict:
        """
        Parameters
        ----------
        Q_demand_MVAR : Requested reactive output [MVAR]; + = capacitive
        V_pu          : Terminal voltage [pu] (does NOT affect Q range — VSC property)

        Returns
        -------
        dict with Q_out_MVAR, Q_limited, P_loss_MW, P_standby_MW,
        P_total_loss_MW, operating_mode, utilization
        """
        Q_dem = np.asarray(Q_demand_MVAR, dtype=float)
        V_pu = np.asarray(V_pu, dtype=float)

        Q_out = np.clip(Q_dem, self.Q_min, self.Q_max)
        Q_limited = Q_dem != Q_out

        P_loss_variable = self.loss_factor * np.abs(Q_out)
        P_total_loss = self.P_standby + P_loss_variable

        if np.ndim(Q_out) == 0:
            if float(Q_out) > 0.01:
                mode = "capacitive"
            elif float(Q_out) < -0.01:
                mode = "inductive"
            else:
                mode = "standby"
        else:
            mode = "vectorized"

        Q_range = self.Q_max  # symmetric
        utilization = np.abs(Q_out) / (Q_range + 1e-12)

        return {
            "Q_out_MVAR": Q_out,
            "Q_limited": Q_limited,
            "P_loss_MW": P_loss_variable,
            "P_standby_MW": np.full_like(P_loss_variable, self.P_standby) if np.ndim(P_loss_variable) > 0 else self.P_standby,
            "P_total_loss_MW": P_total_loss,
            "operating_mode": mode,
            "utilization": utilization,
        }
