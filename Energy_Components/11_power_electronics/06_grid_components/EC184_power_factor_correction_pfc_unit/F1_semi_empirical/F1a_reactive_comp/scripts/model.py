"""
EC184 — Power Factor Correction Unit — F1a Reactive Compensation Model

Capacitor bank PFC:
    Q_required = P * (tan(phi1) - tan(phi2))
    where phi1 = arccos(pf_initial), phi2 = arccos(pf_target)

Actual Q_compensated = min(Q_required, Q_rated)

Achieved power factor:
    Q_residual = Q_load - Q_compensated
    pf_achieved = P / sqrt(P^2 + Q_residual^2)

Capacitor bank losses (dielectric + switching):
    P_loss = loss_factor * Q_compensated    [kW]

Reference:
    Acha, E. et al. (2004). FACTS: Modelling and Simulation in Power Networks. Wiley.
    IEEE Std 1036-2010: Guide for Application of Shunt Capacitors.
"""

import numpy as np


class PFCUnitModel:
    """Capacitor bank power factor correction model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated_kVAR = u["Q_rated_kVAR"]["value"]
        self.loss_factor = u["loss_factor"]["value"]
        self.pf_target_default = u["pf_target"]["value"]

    def compute(self, P_kW: float, pf_initial: float,
                pf_target: float = None, Q_comp_override_kVAR: float = None) -> dict:
        """
        Parameters
        ----------
        P_kW               : Active load [kW]
        pf_initial         : Initial load power factor (lagging, 0–1)
        pf_target          : Desired power factor after compensation
        Q_comp_override_kVAR: If set, use this fixed Q_comp instead of computing from pf_target

        Returns
        -------
        dict with Q_required_kVAR, Q_compensated_kVAR, pf_achieved,
        P_loss_kW, Q_load_kVAR, Q_residual_kVAR, bank_utilization
        """
        if pf_target is None:
            pf_target = self.pf_target_default

        P = np.asarray(P_kW, dtype=float)
        pf1 = np.asarray(pf_initial, dtype=float)
        pf2 = float(pf_target)

        # Clamp pf to avoid numerical issues
        pf1_safe = np.clip(pf1, 1e-6, 1.0 - 1e-9)
        pf2_safe = np.clip(pf2, 1e-6, 1.0 - 1e-9)

        phi1 = np.arccos(pf1_safe)
        phi2 = np.arccos(pf2_safe)

        Q_load = P * np.tan(phi1)
        Q_required = P * (np.tan(phi1) - np.tan(phi2))

        if Q_comp_override_kVAR is not None:
            Q_comp = np.clip(np.asarray(Q_comp_override_kVAR, dtype=float),
                             0.0, self.Q_rated_kVAR)
        else:
            Q_comp = np.clip(Q_required, 0.0, self.Q_rated_kVAR)

        Q_residual = Q_load - Q_comp

        # Achieved PF
        S_after = np.sqrt(P ** 2 + Q_residual ** 2)
        safe_S = np.where(S_after > 0, S_after, 1e-12) if np.ndim(S_after) > 0 else (S_after if S_after > 0 else 1e-12)
        pf_achieved = np.where(S_after > 0, P / safe_S, 1.0) if np.ndim(S_after) > 0 else (P / safe_S if S_after > 0 else 1.0)

        # Losses
        P_loss = self.loss_factor * Q_comp

        # Bank utilization
        bank_utilization = Q_comp / self.Q_rated_kVAR

        return {
            "Q_load_kVAR": Q_load,
            "Q_required_kVAR": Q_required,
            "Q_compensated_kVAR": Q_comp,
            "Q_residual_kVAR": Q_residual,
            "pf_achieved": pf_achieved,
            "P_loss_kW": P_loss,
            "bank_utilization": bank_utilization,
        }
