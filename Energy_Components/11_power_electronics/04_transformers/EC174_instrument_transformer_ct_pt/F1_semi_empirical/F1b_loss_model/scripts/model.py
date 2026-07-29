"""
EC174 -- Instrument Transformer (CT / PT) -- F1b Accuracy + Loss Model

Instrument transformers (CTs and PTs/VTs) are precision devices used for
measurement and protection. Unlike power transformers, their primary metric
is *accuracy* — how closely the secondary current/voltage tracks the primary
quantity — in addition to losses.

The model covers both types in a unified framework:

════════════════════════════════════════════════════════════════════
CURRENT TRANSFORMER (CT):
════════════════════════════════════════════════════════════════════
CT equivalent circuit (referred to secondary):
    Primary current source: I1 / n  (n = N1/N2 turns ratio)
    Magnetizing branch: Rc (core loss resistance) || Xm (magnetizing reactance)
    Secondary series impedance: R2 + jX2  (winding resistance + leakage)
    Burden: Z_burden = R_b + jX_b  (load impedance on secondary)

Ratio error (current error, %):
    epsilon_I = (I2 / I1_ref - 1) * 100  [%]
    where I1_ref = I1 / n (ideal secondary)
    Simplified: epsilon_I ≈ (Im * R_b - Ie * X_b) / (I2 * Z_b) * 100
    Or using the magnetizing current split:
    Ie (in-phase component) = I2 * cos(delta_c)  -- causes ratio error
    Im (quadrature) = I2 * sin(delta_c)           -- causes phase error

For a simplified model using measured accuracy class parameters:
    At accuracy limit factor (ALF), excitation current rises → error increases.

Core loss in CT:
    P_core = I2^2 * R_b * (1 - eta_ct^2) / eta_ct^2
    Simplified: P_core_ct = V_sec^2 / Rc
    where V_sec = I2 * Z_burden_total

Copper loss in CT:
    P_cu_ct = I2^2 * R2

════════════════════════════════════════════════════════════════════
POTENTIAL TRANSFORMER (PT / VT):
════════════════════════════════════════════════════════════════════
PT equivalent circuit (referred to secondary):
    Primary voltage source: V1 / n
    Primary impedance: R1/n^2 + jX1/n^2
    Magnetizing branch: Rc || Xm (as seen from secondary)
    Secondary impedance: R2 + jX2
    Burden: Z_burden

Voltage ratio error (%):
    epsilon_V = (V2_actual / V2_ideal - 1) * 100
    V2_actual = V1/n * Z_burden / (Z_total)
    Z_total = (R1/n^2 + R2 + R_b) + j(X1/n^2 + X2 + X_b)
    Simplified small-error expression:
    epsilon_V ≈ -(I2 * (R1/n^2 + R2) * cos(phi) + I2 * (X1/n^2 + X2) * sin(phi)) / V2_ideal * 100

Core loss in PT:
    P_core_pt = V2^2 / Rc  (magnetizing branch dissipates energy)

Copper loss in PT:
    P_cu_pt = I2^2 * (R1/n^2 + R2)

Burden loss (power dissipated in connected metering/protection devices):
    P_burden = |I2|^2 * R_b   (CT)
    P_burden = |V2|^2 / R_b   (PT)

Total losses:
    CT: P_total = P_cu_ct + P_core_ct
    PT: P_total = P_cu_pt + P_core_pt

Temperature rise:
    T_winding = T_a + P_total * R_th  (simple lumped thermal)

Accuracy classes:
    CT: IEC 60044-1 classes 0.1, 0.2, 0.5, 1, 3, 5 (% current error at rated I)
    PT: IEC 60044-2 classes 0.1, 0.2, 0.5, 1, 3 (% voltage error at rated V)

References:
    IEC 60044-1:1996. Instrument transformers — Part 1: Current transformers.
    IEC 60044-2:1997. Instrument transformers — Part 2: Inductive voltage transformers.
    Slemon, G.R. & Straughen, A. (1980). Electric Machines. Addison-Wesley.
    Hoblit, F.M. (1980). Current Transformers. IEEE Trans. Ind. Appl.
"""

import numpy as np


class InstrumentTransformerF1b:
    """Current and potential transformer -- accuracy + loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.type = u["type"]["value"]          # "CT" or "PT"
        self.n = u["turns_ratio"]["value"]       # N1/N2 for CT; N_primary/N_secondary for PT
        self.I_rated = u["I_rated"]["value"]     # A  (primary rated current for CT)
        self.V_rated = u["V_rated"]["value"]     # V  (primary rated voltage for PT)
        self.S_burden = u["S_burden_VA"]["value"]# VA  (rated burden)
        self.R2 = u["R2"]["value"]               # Ohm  (secondary winding resistance)
        self.X2 = u["X2"]["value"]               # Ohm  (secondary leakage reactance)
        self.Rc = u["Rc"]["value"]               # Ohm  (core loss resistance, referred to secondary)
        self.Xm = u["Xm"]["value"]               # Ohm  (magnetizing reactance, referred to secondary)
        self.pf_burden = u["pf_burden"]["value"] # power factor of burden
        self.R_th = u["R_th"]["value"]           # K/W  (thermal resistance)
        self.T_a = u["T_a"]["value"]             # degC
        self.accuracy_class = u["accuracy_class"]["value"]  # e.g. 0.5

    # ------------------------------------------------------------------
    # CT model
    # ------------------------------------------------------------------

    def _ct_secondary_current(self, i_primary):
        """Ideal secondary current [A]: I2_ideal = I1 / n."""
        return np.asarray(i_primary, dtype=float) / self.n

    def _ct_burden_impedance(self):
        """Burden impedance [Ohm]: Z_b = S_burden / I2_rated^2 at given PF."""
        i2_rated = self.I_rated / self.n
        Z_b = self.S_burden / i2_rated ** 2
        R_b = Z_b * self.pf_burden
        X_b = Z_b * np.sqrt(1.0 - self.pf_burden ** 2)
        return R_b, X_b, Z_b

    def ct_secondary_voltage(self, i_primary):
        """Secondary terminal voltage [V] = I2 * (R2 + R_b) (simplified, ignore reactance)."""
        i2 = self._ct_secondary_current(i_primary)
        R_b, _, _ = self._ct_burden_impedance()
        return i2 * (self.R2 + R_b)

    def ct_ratio_error_pct(self, i_primary):
        """
        Current ratio error [%].
        Simplified: epsilon_I ≈ I_e / I1_nominal * 100
        I_e = V_sec / Rc  (in-phase excitation current causing ratio error)
        V_sec = I2 * (R2 + R_b)
        """
        i_primary = np.asarray(i_primary, dtype=float)
        V_sec = self.ct_secondary_voltage(i_primary)
        I_e = V_sec / self.Rc          # in-phase magnetizing current
        I2_ideal = i_primary / self.n
        safe = np.where(I2_ideal > 1e-9, I2_ideal, 1.0)
        return np.where(I2_ideal > 1e-9, (I_e / safe) * 100.0, 0.0)

    def ct_losses(self, i_primary):
        """CT copper + core losses [W]."""
        i2 = self._ct_secondary_current(i_primary)
        V_sec = self.ct_secondary_voltage(i_primary)
        P_cu = i2 ** 2 * self.R2
        P_core = V_sec ** 2 / self.Rc
        return P_cu, P_core

    # ------------------------------------------------------------------
    # PT model
    # ------------------------------------------------------------------

    def _pt_secondary_voltage(self, v_primary):
        """Ideal secondary voltage [V]: V2 = V1 / n."""
        return np.asarray(v_primary, dtype=float) / self.n

    def _pt_burden_impedance(self):
        """PT burden impedance [Ohm] = V2_rated^2 / S_burden."""
        v2_rated = self.V_rated / self.n
        Z_b = v2_rated ** 2 / self.S_burden
        R_b = Z_b * self.pf_burden
        return R_b, Z_b

    def pt_voltage_error_pct(self, v_primary):
        """
        Voltage ratio error [%].
        Simplified: epsilon_V ≈ -I2 * (R1/n^2 + R2) / V2_ideal * 100 (resistive drop at unity PF)
        I2 = V2 / Z_b
        """
        v_primary = np.asarray(v_primary, dtype=float)
        V2 = self._pt_secondary_voltage(v_primary)
        R_b, Z_b = self._pt_burden_impedance()
        I2 = V2 / np.where(Z_b > 0, Z_b, 1.0)

        # Total series resistance (primary referred to secondary: R1/n^2 ≈ R2 for symmetric winding)
        R_series = 2.0 * self.R2  # approximate: R1/n^2 ≈ R2
        volt_drop = I2 * R_series * self.pf_burden

        safe_V2 = np.where(V2 > 0, V2, 1.0)
        return np.where(V2 > 0, -volt_drop / safe_V2 * 100.0, 0.0)

    def pt_losses(self, v_primary):
        """PT copper + core losses [W]."""
        V2 = self._pt_secondary_voltage(v_primary)
        R_b, Z_b = self._pt_burden_impedance()
        I2 = V2 / np.where(Z_b > 0, Z_b, 1.0)
        P_cu = I2 ** 2 * 2.0 * self.R2  # winding copper
        P_core = V2 ** 2 / self.Rc      # core
        return P_cu, P_core

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def loss_breakdown(self, input_value):
        """
        Loss breakdown [W].
        input_value: primary current [A] for CT, primary voltage [V] for PT.
        """
        if self.type == "CT":
            P_cu, P_core = self.ct_losses(input_value)
        else:
            P_cu, P_core = self.pt_losses(input_value)
        return {
            "p_copper_w": P_cu,
            "p_core_w": P_core,
        }

    def total_losses(self, input_value):
        bd = self.loss_breakdown(input_value)
        return bd["p_copper_w"] + bd["p_core_w"]

    def accuracy_error_pct(self, input_value):
        """
        Ratio error [%].
        For CT: current ratio error.
        For PT: voltage ratio error.
        """
        if self.type == "CT":
            return self.ct_ratio_error_pct(input_value)
        return self.pt_voltage_error_pct(input_value)

    def within_accuracy_class(self, input_value):
        """
        Boolean: True if |ratio_error| < accuracy_class (%).
        (simplified: only checks at this operating point)
        """
        err = np.abs(self.accuracy_error_pct(input_value))
        return err <= float(self.accuracy_class)

    def junction_temperature(self, input_value):
        """Winding temperature [degC] = T_a + P_total * R_th."""
        P_total = self.total_losses(input_value)
        return self.T_a + P_total * self.R_th
