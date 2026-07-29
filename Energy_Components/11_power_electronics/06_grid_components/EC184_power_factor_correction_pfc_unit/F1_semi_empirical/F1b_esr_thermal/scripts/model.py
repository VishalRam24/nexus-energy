"""
EC184 -- Power Factor Correction Unit (Capacitor Bank) -- F1b ESR Thermal Model

Extends F1a with:

1. ESR-based conduction losses:
       I_cap = Q_comp * 1000 / (sqrt(3) * V_rated * 1000)   [A rms, 3-phase bank]
       P_ESR = ESR(T) * I_cap^2                              [W = kW/1000]

2. Temperature-dependent ESR (polypropylene film):
       ESR(T) = ESR_ref * (1 + ESR_alpha * (T - T_ref))
   Note: PP film capacitors have a slight NEGATIVE temp coefficient (ESR decreases
   with temperature up to ~70 C, then rises sharply). ESR_alpha ~ -0.003 /K.
   Reference: Murata (2021) Film Capacitor Application Guide.

3. Dielectric (tan_delta) losses:
       P_dielectric = tan_delta(T) * Q_comp    [kW]
       tan_delta(T) = tan_delta_ref + tan_delta_alpha * (T - T_ref)
   This represents the resistive component of the capacitor admittance:
       P_diel = V^2 * omega * C * tan_delta = Q_comp * tan_delta

4. Total losses:
       P_loss_kW = P_ESR_kW + P_dielectric_kW

5. Thermal derating (IEC 60831-1):
       Q_rated_available(T) = Q_rated * max(0, 1 - derating_slope*(T - T_max))
   When T > T_max = 70 C, rated Q is derated by ~1.5%/K to protect dielectric.

References:
    IEEE Std 1036-2010. Guide for Application of Shunt Capacitors.
    IEC 60831-1:2014. Shunt power capacitors of the self-healing type.
    Murata (2021). Film Capacitor Technical Notes.
    Mohan, Undeland & Robbins (2003). Power Electronics. Wiley.
"""

import numpy as np


class PFCUnitF1b:
    """Capacitor bank PFC -- ESR-thermal loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_rated_kVAR = u["Q_rated_kVAR"]["value"]
        self.V_rated_kV = u["V_rated_kV"]["value"]
        self.C_uF = u["C_uF"]["value"]
        self.ESR_ref = u["ESR_ref_ohm"]["value"]         # Ohm
        self.T_ref = u["T_ref"]["value"]                  # degC
        self.ESR_alpha = u["ESR_alpha"]["value"]           # 1/K
        self.tan_d_ref = u["tan_delta_ref"]["value"]       # dimensionless
        self.tan_d_alpha = u["tan_delta_alpha"]["value"]   # 1/K
        self.T_max = u["T_max_C"]["value"]                 # degC
        self.derating_slope = u["derating_slope"]["value"] # 1/K
        self.f = u["f_system_Hz"]["value"]
        self.pf_target_default = u["pf_target"]["value"]
        self.omega = 2.0 * np.pi * self.f

    def esr(self, T_cap: float) -> float:
        """ESR(T) = ESR_ref * (1 + alpha * (T - T_ref))  [Ohm]."""
        T = np.asarray(T_cap, dtype=float)
        return self.ESR_ref * (1.0 + self.ESR_alpha * (T - self.T_ref))

    def tan_delta(self, T_cap: float) -> float:
        """tan_delta(T) = tan_d_ref + tan_d_alpha*(T-T_ref)."""
        T = np.asarray(T_cap, dtype=float)
        td = self.tan_d_ref + self.tan_d_alpha * (T - self.T_ref)
        return np.maximum(td, 0.0)

    def q_rated_available(self, T_cap: float) -> float:
        """
        Derated Q rating [kVAR] when capacitor temperature exceeds T_max.
        IEC 60831-1 mandates derating to avoid premature aging/failure.
        """
        T = np.asarray(T_cap, dtype=float)
        factor = 1.0 - self.derating_slope * np.maximum(T - self.T_max, 0.0)
        factor = np.clip(factor, 0.0, 1.0)
        return self.Q_rated_kVAR * factor

    def cap_current_A(self, Q_comp_kVAR: float) -> float:
        """
        Capacitor rms current [A].
        For a 3-phase bank: I = Q / (sqrt(3) * V_LL)
        """
        Q = np.asarray(Q_comp_kVAR, dtype=float)
        # Q in VAR = Q_kVAR * 1000
        V_V = self.V_rated_kV * 1000.0
        return Q * 1000.0 / (np.sqrt(3.0) * V_V)

    def compute(self, P_kW: float, pf_initial: float,
                T_cap: float = 25.0, pf_target: float = None,
                Q_comp_override_kVAR: float = None) -> dict:
        """
        Parameters
        ----------
        P_kW               : Active load [kW]
        pf_initial         : Load power factor before compensation (lagging, 0-1)
        T_cap              : Capacitor temperature [degC] (default 25)
        pf_target          : Desired power factor (default from params)
        Q_comp_override_kVAR : Override Q output [kVAR] (optional)

        Returns
        -------
        dict with:
            Q_load_kVAR, Q_required_kVAR, Q_compensated_kVAR, Q_residual_kVAR,
            pf_achieved, P_ESR_kW, P_dielectric_kW, P_loss_kW, P_loss_F1a_kW,
            I_cap_A, ESR_Ohm, tan_delta, bank_utilization, Q_rated_available_kVAR
        """
        if pf_target is None:
            pf_target = self.pf_target_default

        P = np.asarray(P_kW, dtype=float)
        pf1 = np.asarray(pf_initial, dtype=float)
        pf2 = float(pf_target)

        pf1_s = np.clip(pf1, 1e-6, 1.0 - 1e-9)
        pf2_s = np.clip(pf2, 1e-6, 1.0 - 1e-9)

        phi1 = np.arccos(pf1_s)
        phi2 = np.arccos(pf2_s)
        Q_load = P * np.tan(phi1)
        Q_required = P * (np.tan(phi1) - np.tan(phi2))

        Q_max_avail = self.q_rated_available(T_cap)

        if Q_comp_override_kVAR is not None:
            Q_comp = np.clip(np.asarray(Q_comp_override_kVAR, dtype=float), 0.0, Q_max_avail)
        else:
            Q_comp = np.clip(Q_required, 0.0, Q_max_avail)

        Q_residual = Q_load - Q_comp

        S_after = np.sqrt(P ** 2 + Q_residual ** 2)
        safe_S = np.where(S_after > 0, S_after, 1e-12) if np.ndim(S_after) > 0 else (S_after if S_after > 0 else 1e-12)
        pf_achieved = np.where(S_after > 0, P / safe_S, 1.0) if np.ndim(S_after) > 0 else (P / safe_S if S_after > 0 else 1.0)

        # F1b losses
        R_esr = self.esr(T_cap)
        td = self.tan_delta(T_cap)
        I_cap = self.cap_current_A(Q_comp)

        P_ESR_kW = R_esr * I_cap ** 2 / 1000.0       # W -> kW
        P_diel_kW = td * Q_comp                        # kW (tan_delta * kVAR = kW)
        P_loss_kW = P_ESR_kW + P_diel_kW

        # F1a loss for comparison (loss_factor from F1a params: 0.3% of Q_comp)
        # We derive an equivalent loss_factor for reference
        P_loss_F1a_kW = 0.003 * Q_comp  # approx F1a value

        bank_utilization = Q_comp / (self.Q_rated_kVAR + 1e-12)

        return {
            "Q_load_kVAR": Q_load,
            "Q_required_kVAR": Q_required,
            "Q_compensated_kVAR": Q_comp,
            "Q_residual_kVAR": Q_residual,
            "pf_achieved": pf_achieved,
            "P_ESR_kW": P_ESR_kW,
            "P_dielectric_kW": P_diel_kW,
            "P_loss_kW": P_loss_kW,
            "P_loss_F1a_kW": P_loss_F1a_kW,
            "I_cap_A": I_cap,
            "ESR_Ohm": R_esr,
            "tan_delta": td,
            "bank_utilization": bank_utilization,
            "Q_rated_available_kVAR": Q_max_avail,
        }
