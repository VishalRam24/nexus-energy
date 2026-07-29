"""
EC162 -- Resonant LLC Converter -- F1a Ideal Gain + Fixed Efficiency

LLC resonant converter ideal voltage gain:
    V_out = N * V_in * M(fn)

where M(fn) is the voltage gain as a function of normalized switching frequency fn = f_sw/f_res.
At resonance (fn = 1): M = 1 (unity gain, ignoring quality factor).

Simplified gain approximation (first-harmonic analysis, Q → 0 assumption):
    M(fn) ≈ 1  (near resonance: fn in [0.8, 1.2])
    M(fn) < 1  for fn > 1 (above resonance, step-down behavior)
    M(fn) > 1  for fn < 1 (below resonance, step-up behavior, limited by Lr/Lm ratio)

For this F1a model:
    M(fn) = 1 / sqrt(1 + (fn^2 - 1)^2 * k^2)   (simplified first-harmonic with quality factor Q≈0)
    where k = (Lr/(Lr+Lm)) = 0.1 (typical for LLC design)

Power with fixed efficiency:
    P_out = eta * P_in

References:
    Yang, B., Lee, F.C., Zhang, A.J., & Huang, G. (2002). LLC Resonant Converter for Front End DC/DC.
    APEC 2002.
"""

import numpy as np


class LLCF1a:
    """LLC Resonant Converter -- ideal gain + fixed efficiency."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = u["n_turns"]["value"]
        self.eta = u["eta"]["value"]
        self.P_rated = u["p_rated"]["value"]
        self.f_res = u["f_res"]["value"]
        self.M_nom = u["m_gain_nominal"]["value"]
        # LLC gain curve shape parameter (k = Lr/(Lr+Lm), typical ~0.1)
        self.k = 0.1

    def gain_M(self, fn):
        """
        Simplified LLC voltage gain as function of normalized frequency fn = f_sw / f_res.

        For fn >= 1 (above resonance, ZVS region):
            M = 1 / sqrt(1 + (fn^2 - 1)^2 / k^2)   -> M < 1 (step-down)

        For fn < 1 (below resonance):
            M = 1 + (1-fn) * m_boost  -> M > 1 (step-up, limited by Lm/Lr ratio)

        At fn=1: M=1 (unity gain). This approximation captures the asymmetric
        nature of the LLC gain curve (above resonance step-down, below step-up).
        """
        fn = np.asarray(fn, dtype=float)
        # Above resonance: symmetric first-harmonic drop
        m_above = 1.0 / np.sqrt(1.0 + (fn**2 - 1.0)**2 * (1.0 / self.k)**2 * 0.01)
        # Below resonance: linear boost (simplified; real LLC uses Lm/Lr for limit)
        m_below = 1.0 + (1.0 - fn) * 0.5  # gain increases below resonance
        return np.where(fn >= 1.0, m_above, m_below)

    def output_voltage(self, v_in, fn):
        """V_out = N * V_in * M(fn)  [V]."""
        v_in = np.asarray(v_in, dtype=float)
        M = self.gain_M(fn)
        return self.N * v_in * M

    def output_power(self, p_in):
        """P_out = eta * P_in  [W]."""
        return self.eta * np.asarray(p_in, dtype=float)

    def input_power(self, p_out):
        """P_in = P_out / eta  [W]."""
        return np.asarray(p_out, dtype=float) / self.eta

    def losses(self, p_in):
        """P_loss = (1 - eta) * P_in  [W]."""
        return (1.0 - self.eta) * np.asarray(p_in, dtype=float)
