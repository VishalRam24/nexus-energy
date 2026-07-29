"""
EC182 — Distribution Line — F1a R+jX Model (SI units)

At distribution voltage levels (4–35 kV, feeder lengths typically < 20 km),
shunt capacitance is negligible. Model uses series R+jX only.

Approximate voltage drop (Mordey/Bapat formula for small angles):
    dV ≈ (P*R + Q*X) / V_s    [V, single-phase equivalent]

Exact (iterative for three-phase balanced):
    I = conj(S / (sqrt(3) * V_s))   — line current from 3-ph load
    V_r = V_s - Z * I               — phasor voltage at receiving end

Active loss:
    P_loss = 3 * |I|^2 * R_total    [W] — three-phase
    (= |I_line|^2 * R_total for single-phase equivalent)

Reference:
    Kersting (2012). Distribution System Modeling and Analysis, 3rd ed.
"""

import numpy as np


class DistributionLinePiModel:
    """R+jX model for distribution feeder (SI units, 3-phase balanced)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.R_ohm_per_km = u["R_ohm_per_km"]["value"]
        self.X_ohm_per_km = u["X_ohm_per_km"]["value"]
        self.V_base_kV = u["V_base_kV"]["value"]
        self.length_km_default = u["length_km"]["value"]

    def compute(self, V_s_kV: float, P_load_kW: float,
                Q_load_kVAR: float, length_km: float = None) -> dict:
        """
        Parameters
        ----------
        V_s_kV      : Sending-end line-to-line voltage [kV]
        P_load_kW   : Active load at receiving end [kW]
        Q_load_kVAR : Reactive load [kVAR] (positive = inductive)
        length_km   : Feeder length [km]

        Returns
        -------
        dict with V_r_kV, I_line_A, P_loss_kW, Q_loss_kVAR,
        efficiency, voltage_drop_kV, voltage_drop_pct, power_factor_load
        """
        if length_km is None:
            length_km = self.length_km_default

        V_s_kV = np.asarray(V_s_kV, dtype=float)
        P_kW = np.asarray(P_load_kW, dtype=float)
        Q_kVAR = np.asarray(Q_load_kVAR, dtype=float)
        L = float(length_km)

        R_total = self.R_ohm_per_km * L  # Ohm
        X_total = self.X_ohm_per_km * L  # Ohm
        Z = R_total + 1j * X_total

        # Phase voltage at sending end
        V_s_phase = V_s_kV * 1000.0 / np.sqrt(3.0)  # V

        # 3-phase complex power
        S_VA = (P_kW + 1j * Q_kVAR) * 1000.0  # VA

        # Iterative solution
        V_r_phase = V_s_phase.copy() if np.ndim(V_s_phase) > 0 else complex(V_s_phase)
        for _ in range(5):
            safe = np.abs(V_r_phase) > 1.0
            V_r_safe = np.where(safe, V_r_phase, 1.0 + 0j) if np.ndim(V_r_phase) > 0 else (V_r_phase if abs(V_r_phase) > 1.0 else 1.0 + 0j)
            I_line = np.conj(S_VA / (3.0 * V_r_safe))
            V_r_phase = V_s_phase - Z * I_line

        I_mag = np.abs(I_line)
        V_r_phase_mag = np.abs(V_r_phase)
        V_r_kV = V_r_phase_mag * np.sqrt(3.0) / 1000.0

        P_loss_kW = 3.0 * I_mag ** 2 * R_total / 1000.0
        Q_loss_kVAR = 3.0 * I_mag ** 2 * X_total / 1000.0

        voltage_drop_kV = V_s_kV - V_r_kV
        safe_V = np.where(V_s_kV > 0, V_s_kV, 1e-12) if np.ndim(V_s_kV) > 0 else (V_s_kV if V_s_kV > 0 else 1e-12)
        voltage_drop_pct = voltage_drop_kV / safe_V * 100.0

        # Efficiency (active power)
        P_s_kW = P_kW + P_loss_kW
        safe_Ps = np.where(P_s_kW > 0, P_s_kW, 1e-12) if np.ndim(P_s_kW) > 0 else (P_s_kW if P_s_kW > 0 else 1e-12)
        eta = np.where(P_s_kW > 0, P_kW / safe_Ps, 0.0) if np.ndim(P_s_kW) > 0 else (P_kW / safe_Ps if P_s_kW > 0 else 0.0)

        # Load power factor
        S_load = np.sqrt(P_kW ** 2 + Q_kVAR ** 2)
        safe_S = np.where(S_load > 0, S_load, 1e-12) if np.ndim(S_load) > 0 else (S_load if S_load > 0 else 1e-12)
        pf_load = np.where(S_load > 0, P_kW / safe_S, 1.0) if np.ndim(S_load) > 0 else (P_kW / safe_S if S_load > 0 else 1.0)

        return {
            "V_r_kV": V_r_kV,
            "I_line_A": I_mag,
            "P_loss_kW": P_loss_kW,
            "Q_loss_kVAR": Q_loss_kVAR,
            "P_s_kW": P_s_kW,
            "efficiency": eta,
            "voltage_drop_kV": voltage_drop_kV,
            "voltage_drop_pct": voltage_drop_pct,
            "power_factor_load": pf_load,
        }
