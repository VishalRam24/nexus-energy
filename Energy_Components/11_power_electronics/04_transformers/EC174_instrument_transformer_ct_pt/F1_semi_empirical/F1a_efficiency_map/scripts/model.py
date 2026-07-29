"""
EC174 -- Instrument Transformer (CT/PT) -- F1a Ideal Ratio + Accuracy Class Model

Current Transformer (CT):
    I_secondary = I_primary / N_ct      (ideal ratio)
    Ratio error  = epsilon_r [%]        (per IEC 61869-2)
    Phase error  = delta [arcmin]       (per IEC 61869-2)

Potential Transformer (PT):
    V_secondary = V_primary * N_pt      (ideal ratio, N_pt = 1/ratio)
    Same accuracy class limits apply

For IEC Accuracy Class 0.2:
    |ratio_error| <= 0.2%   (at 1-120% rated current / 80-120% rated voltage)
    |phase_error| <= 10 arcmin

This F1a model computes:
    - Ideal transformation ratio
    - Accuracy limits per IEC Class 0.2
    - Burden power (VA loading of secondary winding)

The ratio error and phase displacement are specified as limits, not simulated explicitly
(F1b would model magnetic core nonlinearity to compute actual errors).

References:
    IEC 61869-2:2012. Instrument Transformers -- Part 2: CT additional requirements.
    IEC 61869-3:2011. Instrument Transformers -- Part 3: PT additional requirements.
"""

import numpy as np


class InstrumentTransformerF1a:
    """Instrument Transformer (CT/PT) -- ideal ratio + IEC accuracy class limits."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_ct = u["ct_ratio"]["value"]         # I_pri / I_sec
        self.N_pt = 1.0 / u["pt_ratio"]["value"]   # V_pri / V_sec  (ratio = V_sec/V_pri -> N_pt = 1/ratio)
        # Actually pt_ratio = V_sec/V_pri = 110/11000 = 0.01, so N_pt_transformation = V_sec = V_pri * N_pt
        # Re-read: pt_ratio=0.001 = 1/1000 weird; let's use actual values
        self.I_rated_pri = u["ct_primary_current"]["value"]
        self.I_rated_sec = u["ct_secondary_current"]["value"]
        self.V_rated_pri = u["pt_primary_voltage"]["value"]
        self.V_rated_sec = u["pt_secondary_voltage"]["value"]
        self.ratio_error_pct = u["ratio_error_pct"]["value"]
        self.phase_error_min = u["phase_error_min"]["value"]
        self.ct_burden = u["ct_burden"]["value"]
        self.pt_burden = u["pt_burden"]["value"]
        # Derived turns ratios
        self.ct_N = self.I_rated_pri / self.I_rated_sec   # = 100
        self.pt_N = self.V_rated_sec / self.V_rated_pri   # = 0.01

    def ct_secondary_current(self, i_primary):
        """I_sec = I_pri / N_ct  [A]."""
        return np.asarray(i_primary, dtype=float) / self.ct_N

    def pt_secondary_voltage(self, v_primary):
        """V_sec = V_pri * N_pt  [V]."""
        return np.asarray(v_primary, dtype=float) * self.pt_N

    def ct_burden_power(self, i_primary):
        """
        Secondary burden power [VA].
        At rated current: burden = ct_burden [VA].
        Scales with (I_sec)^2 = (I_pri/N_ct)^2.
        """
        i_sec = self.ct_secondary_current(i_primary)
        i_sec_rated = self.I_rated_sec
        return self.ct_burden * (i_sec / i_sec_rated)**2

    def pt_burden_power(self, v_primary):
        """Secondary burden power [VA] = fixed (voltage source)."""
        return np.full_like(np.asarray(v_primary, dtype=float), self.pt_burden)

    def accuracy_within_class(self, current_fraction):
        """
        Returns True/False: whether operating point is within Class 0.2 accuracy.
        IEC 61869-2 requires Class 0.2 performance at 1-120% rated current.
        """
        cf = np.asarray(current_fraction, dtype=float)
        return (cf >= 0.01) & (cf <= 1.2)
