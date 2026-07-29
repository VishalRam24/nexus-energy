"""
EC173 -- Distribution Transformer -- F1a Efficiency Map

Transformer efficiency via standard two-loss model (IEC 60076-1):
    eta = P_out / (P_out + P_core + P_cu * load^2)

where:
    P_core = p_core_frac * S_rated     [W] -- constant core (iron/no-load) losses
    P_cu   = p_cu_frac * S_rated       [W] -- full-load copper losses
    load   = S_actual / S_rated        [-] -- per-unit load fraction (0 to 1.5)

Actual output power:
    P_out = load * S_rated * power_factor

Peak efficiency at load*:
    load* = sqrt(P_core / P_cu)   (when core losses = copper losses)
    For P_core=0.002, P_cu=0.010: load* = sqrt(0.002/0.010) = 0.447

Output voltage (turns ratio):
    V_out = (N2/N1) * V_in   (ideal, ignoring impedance drop)

References:
    IEC 60076-1:2011. Power transformers -- Part 1: General.
    IEEE C57.12.00-2015. IEEE Standard for General Requirements for Liquid-Immersed Distribution,
    Power, and Regulating Transformers.
"""

import numpy as np


class DistributionTransformerF1a:
    """Distribution Transformer -- two-loss efficiency map."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_rated = u["s_rated"]["value"]        # VA
        self.P_core = u["p_core_frac"]["value"] * self.S_rated  # W
        self.P_cu = u["p_cu_frac"]["value"] * self.S_rated     # W
        self.N = u["n_turns"]["value"]              # N2/N1
        self.V_primary = u["v_primary"]["value"]
        self.V_secondary = u["v_secondary"]["value"]

    def efficiency(self, load_fraction, power_factor=1.0):
        """
        eta = P_out / (P_out + P_core + P_cu * load^2)
        """
        plr = np.asarray(load_fraction, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        p_out = plr * self.S_rated * pf
        p_loss = self.P_core + self.P_cu * plr**2
        p_in = p_out + p_loss
        safe = p_in > 0
        return np.where(safe, p_out / np.where(safe, p_in, 1.0), 0.0)

    def output_power(self, load_fraction, power_factor=1.0):
        """P_out [W]."""
        plr = np.asarray(load_fraction, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        return plr * self.S_rated * pf

    def input_power(self, load_fraction, power_factor=1.0):
        """P_in = P_out + P_core + P_cu*load^2  [W]."""
        plr = np.asarray(load_fraction, dtype=float)
        p_out = self.output_power(plr, power_factor)
        p_loss = self.losses(plr)
        return p_out + p_loss

    def losses(self, load_fraction):
        """Total losses [W] = P_core + P_cu * load^2."""
        plr = np.asarray(load_fraction, dtype=float)
        return self.P_core + self.P_cu * plr**2

    def core_losses(self):
        """No-load (core/iron) losses [W] -- constant."""
        return self.P_core

    def copper_losses(self, load_fraction):
        """Load-dependent copper losses [W] = P_cu * load^2."""
        plr = np.asarray(load_fraction, dtype=float)
        return self.P_cu * plr**2

    def output_voltage(self, v_in):
        """V_out = N * V_in  [V]."""
        return self.N * np.asarray(v_in, dtype=float)

    def peak_efficiency_load(self):
        """Load fraction at peak efficiency = sqrt(P_core / P_cu)."""
        return np.sqrt(self.P_core / self.P_cu)
