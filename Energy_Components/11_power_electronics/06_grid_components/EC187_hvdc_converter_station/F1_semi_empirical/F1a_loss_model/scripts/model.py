"""
EC187 — HVDC Converter Station — F1a Loss Model

Single converter station (rectifier or inverter):
    P_loss = P_no_load + loss_factor * P_transfer    [MW]

For a point-to-point HVDC link (two stations):
    P_out = P_in - P_loss_rect - P_loss_cable - P_loss_inv

At F1a, we model one station only (cable losses are handled separately in line model).

Efficiency per station:
    eta = P_out / P_in
    Rectifier: P_out = P_DC = P_in - P_loss
    Inverter:  P_out = P_AC = P_DC - P_loss

Reference:
    Cigre TB 492 (2012). VSC Transmission. Working Group B4-37.
    Alstom (2010). HVDC: Connecting to the Future.
"""

import numpy as np


class HVDCConverterModel:
    """HVDC converter station F1a loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_MW"]["value"]
        self.loss_factor = u["loss_factor_station"]["value"]
        self.P_no_load = u["P_no_load_MW"]["value"]
        self.V_dc_kV = u["V_dc_kV"]["value"]
        self.Q_cap = u["Q_capability_MVAR"]["value"]
        self.converter_type = u["type"]["value"]

    def compute(self, P_transfer_MW: float, direction: str = "rectifier",
                Q_request_MVAR: float = 0.0) -> dict:
        """
        Parameters
        ----------
        P_transfer_MW   : Active power transfer [MW] (0 to P_rated)
        direction       : "rectifier" (AC→DC) or "inverter" (DC→AC)
        Q_request_MVAR  : Reactive power support request [MVAR] (VSC only)

        Returns
        -------
        dict with P_in_MW, P_out_MW, P_loss_MW, efficiency,
        I_dc_kA, utilization, Q_delivered_MVAR
        """
        P = np.asarray(P_transfer_MW, dtype=float)
        P = np.clip(P, 0.0, self.P_rated)

        P_loss = self.P_no_load + self.loss_factor * P

        if direction == "rectifier":
            P_in = P + P_loss   # AC input
            P_out = P           # DC output
        else:  # inverter
            P_in = P            # DC input
            P_out = P - P_loss  # AC output
            P_out = np.maximum(P_out, 0.0)

        safe_Pin = np.where(P_in > 0, P_in, 1e-12) if np.ndim(P_in) > 0 else (P_in if P_in > 0 else 1e-12)
        eta = np.where(P_in > 0, P_out / safe_Pin, 0.0) if np.ndim(P_in) > 0 else (P_out / safe_Pin if P_in > 0 else 0.0)

        # DC current
        V_dc = self.V_dc_kV * 1000.0  # V
        safe_V = V_dc if V_dc > 0 else 1e-12
        I_dc_kA = P * 1e6 / (safe_V * 1000.0)  # kA

        utilization = P / (self.P_rated + 1e-12)

        # VSC reactive capability (capped by Q_capability)
        Q_delivered = np.clip(np.asarray(Q_request_MVAR, dtype=float),
                              -self.Q_cap, self.Q_cap)

        return {
            "P_in_MW": P_in,
            "P_out_MW": P_out,
            "P_loss_MW": P_loss,
            "efficiency": eta,
            "I_dc_kA": I_dc_kA,
            "utilization": utilization,
            "Q_delivered_MVAR": Q_delivered,
            "direction": direction,
        }
