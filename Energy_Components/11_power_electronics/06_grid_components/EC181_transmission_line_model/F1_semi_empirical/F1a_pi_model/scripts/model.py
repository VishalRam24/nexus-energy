"""
EC181 — Transmission Line — F1a Lumped Pi-Model

Lumped pi-model:
    Z = R + jX    (total series impedance, pu)
    Y = jB        (total shunt admittance = jB, split equally B/2 at each end)

Given sending-end voltage V_s (complex pu) and load (P_load, Q_load in pu):
    I_load = conj((P_load + jQ_load) / V_r)  — solved iteratively
    V_r = V_s - Z * I_s,  where I_s = I_load + V_s * jB/2

Simplified (direct load-flow):
    Compute I_series from P/Q load at receiving end (assume V_r ~ V_s first,
    then correct once).  For F1a we use a single Newton step:
        V_r = V_s - Z * I_s      (I_s includes shunt charging current at sending end)

Physics:
    P_loss = I_series^2 * R        [pu]
    Q_loss = I_series^2 * X        [pu]
    |V_r| <= |V_s|  (for P>0 lagging load)

Reference:
    Glover, Sarma, Overbye (2012). Power Systems Analysis and Design, 5th ed.
"""

import numpy as np


class TransmissionLinePiModel:
    """Lumped pi-model for a transmission line (all quantities in per-unit)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.R_pu_per_km = u["R_pu_per_km"]["value"]
        self.X_pu_per_km = u["X_pu_per_km"]["value"]
        self.B_pu_per_km = u["B_pu_per_km"]["value"]
        self.V_base_kV = u["V_base_kV"]["value"]
        self.S_base_MVA = u["S_base_MVA"]["value"]
        self.length_km_default = u["length_km"]["value"]

    def _line_params(self, length_km):
        """Return total R, X, B for given length."""
        R = self.R_pu_per_km * length_km
        X = self.X_pu_per_km * length_km
        B = self.B_pu_per_km * length_km
        return R, X, B

    def compute(self, V_s_pu: float, delta_s_rad: float,
                P_load_pu: float, Q_load_pu: float,
                length_km: float = None) -> dict:
        """
        Pi-model power flow.

        Parameters
        ----------
        V_s_pu      : Sending-end voltage magnitude [pu]
        delta_s_rad : Sending-end voltage angle [rad]
        P_load_pu   : Receiving-end active load [pu]
        Q_load_pu   : Receiving-end reactive load [pu] (positive = inductive)
        length_km   : Line length [km]; uses default if None

        Returns
        -------
        dict with V_r_pu, delta_r_rad, I_series_pu, P_loss_pu, Q_loss_pu,
        P_s_pu, Q_s_pu, efficiency, voltage_drop_pu
        """
        if length_km is None:
            length_km = self.length_km_default

        V_s_pu = np.asarray(V_s_pu, dtype=float)
        delta_s_rad = np.asarray(delta_s_rad, dtype=float)
        P_load_pu = np.asarray(P_load_pu, dtype=float)
        Q_load_pu = np.asarray(Q_load_pu, dtype=float)

        R, X, B = self._line_params(float(length_km))
        Z = complex(R, X)
        Y_half = complex(0, B / 2.0)

        # Sending-end complex voltage
        V_s = V_s_pu * np.exp(1j * delta_s_rad)

        # Iterative solve: start with V_r = V_s, iterate 3 times (converges fast)
        V_r = V_s.copy() if np.ndim(V_s) > 0 else complex(V_s)
        for _ in range(5):
            # Load current at receiving end
            S_load = P_load_pu + 1j * Q_load_pu
            # Avoid division by zero
            V_r_safe = np.where(np.abs(V_r) > 1e-12, V_r, 1e-12 + 0j) if np.ndim(V_r) > 0 else (V_r if abs(V_r) > 1e-12 else 1e-12 + 0j)
            I_r = np.conj(S_load / V_r_safe)
            # Shunt current at receiving end
            I_shunt_r = V_r * Y_half
            # Series current
            I_series = I_r + I_shunt_r
            # Shunt current at sending end
            I_shunt_s = V_s * Y_half
            # Sending-end current
            I_s = I_series + I_shunt_s
            # Update V_r
            V_r = V_s - Z * I_series

        # Extract results
        V_r_mag = np.abs(V_r)
        V_r_ang = np.angle(V_r)
        I_series_mag = np.abs(I_series)

        P_loss = I_series_mag ** 2 * R
        Q_loss = I_series_mag ** 2 * X

        S_s = V_s * np.conj(I_s)
        P_s = np.real(S_s)
        Q_s = np.imag(S_s)

        voltage_drop = V_s_pu - V_r_mag

        # Efficiency: P_r / P_s (active power transfer efficiency)
        safe_Ps = np.where(np.abs(P_s) > 1e-12, P_s, 1e-12) if np.ndim(P_s) > 0 else (P_s if abs(P_s) > 1e-12 else 1e-12)
        eta = np.where(P_s > 0, P_load_pu / safe_Ps, 0.0) if np.ndim(P_s) > 0 else (P_load_pu / safe_Ps if P_s > 0 else 0.0)

        return {
            "V_r_pu": V_r_mag,
            "delta_r_rad": V_r_ang,
            "I_series_pu": I_series_mag,
            "P_loss_pu": P_loss,
            "Q_loss_pu": Q_loss,
            "P_s_pu": P_s,
            "Q_s_pu": Q_s,
            "efficiency": eta,
            "voltage_drop_pu": voltage_drop,
        }
