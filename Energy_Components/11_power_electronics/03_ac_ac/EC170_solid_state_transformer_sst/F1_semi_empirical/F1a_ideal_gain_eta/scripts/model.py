"""
EC170 -- Solid State Transformer (SST) -- F1a Ideal Gain + Fixed Efficiency

SST ideal voltage conversion:
    V_out = N * V_in

where N = N2/N1 (turns ratio of the internal HF transformer).

SST chain: MV AC -> Rectifier -> HF DC-DC (isolated) -> Inverter -> LV AC

Bidirectional operation:
    Forward (MV -> LV): P_out = eta * P_in
    Reverse (LV -> MV): P_in = P_out / eta   (power flows from secondary to primary)

The sign convention: P_in > 0 is forward (MV to LV); P_in < 0 is reverse.

References:
    Huang, A.Q., Crow, M.L., Heydt, G.T., Zheng, J.P., & Dale, S.J. (2011).
    The Future Renewable Electric Energy Delivery and Management (FREEDM) System.
    IEEE Trans. Ind. Electron., 58(7), 2799-2809.
"""

import numpy as np


class SSTF1a:
    """Solid State Transformer -- ideal voltage ratio + fixed efficiency, bidirectional."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = u["n_turns"]["value"]      # N2/N1
        self.eta = u["eta"]["value"]
        self.P_rated = u["p_rated"]["value"]

    def output_voltage(self, v_in):
        """V_out = N * V_in  [V]."""
        return self.N * np.asarray(v_in, dtype=float)

    def output_power(self, p_in):
        """
        P_out [W] accounting for direction:
          Forward (p_in > 0): P_out = eta * p_in
          Reverse (p_in < 0): P_out = p_in / eta  (more input needed)
        """
        p = np.asarray(p_in, dtype=float)
        return np.where(p >= 0, self.eta * p, p / self.eta)

    def losses(self, p_in):
        """Absolute power loss [W] >= 0."""
        p = np.asarray(p_in, dtype=float)
        # Forward: P_loss = (1-eta)*P_in; Reverse: P_loss = (1/eta-1)*|P_in|
        return np.where(p >= 0,
                        (1.0 - self.eta) * p,
                        (1.0 / self.eta - 1.0) * np.abs(p))
