"""
EC185 — Static VAR Compensator (SVC) — F1a Reactive Compensation Model

SVC (TCR+TSC topology):
    Q_out = clamp(Q_demand, Q_min, Q_max)
    Sign convention: Q_out > 0 → capacitive (injecting Q into grid)
                     Q_out < 0 → inductive (absorbing Q from grid, TCR mode)

Note: SVC output is V-dependent (Q ∝ V^2 for capacitor bank).
At F1a fidelity we treat Q_out as a direct demand-limited model (V-droop neglected).

Losses:
    P_loss = loss_factor * |Q_out|   [MVAR → MW for consistency]

Reference:
    Hingorani, N.G. & Gyugyi, L. (2000). Understanding FACTS. IEEE Press.
"""

import numpy as np


class SVCModel:
    """SVC (TCR+TSC) reactive compensation model — F1a."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_max = u["Q_max_MVAR"]["value"]   # MVAR (capacitive, +)
        self.Q_min = u["Q_min_MVAR"]["value"]   # MVAR (inductive, -)
        self.loss_factor = u["loss_factor"]["value"]

    def compute(self, Q_demand_MVAR: float, V_pu: float = 1.0) -> dict:
        """
        Parameters
        ----------
        Q_demand_MVAR : Requested reactive output [MVAR]; + = capacitive
        V_pu          : Terminal voltage [pu] (used for saturation check only at F1a)

        Returns
        -------
        dict with Q_out_MVAR, Q_limited, P_loss_MW, operating_mode, utilization
        """
        Q_dem = np.asarray(Q_demand_MVAR, dtype=float)
        V_pu = np.asarray(V_pu, dtype=float)

        Q_out = np.clip(Q_dem, self.Q_min, self.Q_max)
        Q_limited = Q_dem != Q_out  # True if clamped

        P_loss = self.loss_factor * np.abs(Q_out)  # MW

        # Operating mode string (not vectorized for simplicity)
        if np.ndim(Q_out) == 0:
            if float(Q_out) > 0.01:
                mode = "capacitive"
            elif float(Q_out) < -0.01:
                mode = "inductive"
            else:
                mode = "standby"
        else:
            mode = "vectorized"

        utilization = np.where(Q_out >= 0, Q_out / (self.Q_max + 1e-12),
                               Q_out / (self.Q_min - 1e-12)) if np.ndim(Q_out) > 0 else (
            float(Q_out) / self.Q_max if float(Q_out) >= 0 else float(Q_out) / self.Q_min)

        return {
            "Q_out_MVAR": Q_out,
            "Q_limited": Q_limited,
            "P_loss_MW": P_loss,
            "operating_mode": mode,
            "utilization": utilization,
        }
