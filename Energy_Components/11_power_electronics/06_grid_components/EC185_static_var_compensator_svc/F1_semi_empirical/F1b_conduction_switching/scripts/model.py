"""
EC185 -- Static VAR Compensator (SVC) -- F1b Conduction + Switching Loss Model

Extends F1a with a physics-based loss breakdown:

**TCR (Thyristor Controlled Reactor) — inductive mode (Q < 0):**
    Current through the TCR thyristors and reactor is:
        I_TCR_rms = |Q_inductive| * 1e6 / (sqrt(3) * V_kV * 1000)   [A]
    Per-thyristor conduction loss (n_series thyristors per valve, 2 valves):
        P_thyristor = 2 * n_T * (V_T0 * I_TCR_avg + r_T * I_TCR_rms^2)
    Reactor copper loss:
        P_reactor = R_reactor * I_TCR_rms^2
    For sinusoidal current:
        I_avg  = I_rms * sqrt(2) / pi

**TSC (Thyristor Switched Capacitor) — capacitive mode (Q > 0):**
    Current through the capacitor bank and its ESR:
        I_TSC_rms = Q_capacitive * 1e6 / (sqrt(3) * V_kV * 1000)    [A]
    TSC capacitor ESR loss:
        P_TSC = ESR_TSC * I_TSC_rms^2
    TSC thyristor loss (switching only at zero-crossing, minor):
        P_thy_TSC = 2 * n_T * V_T0 * I_TSC_avg  (conduction during TSC switching)

**Standby (Q ≈ 0):**
    Losses ~ cooling power only (thyristors not conducting)

**Voltage dependence (Q ∝ V^2 for TSC, Q ∝ 1/V for TCR control):**
    At F1b we include a first-order voltage correction:
        Q_effective = Q_demand * (V_pu / V_ref)^2  (capacitive)
        Q_effective = Q_demand * (V_pu / V_ref)     (inductive, TCR controls firing angle)

**Total losses:**
    P_loss_MW = P_thyristor_MW + P_reactor_MW + P_ESR_MW + P_cooling_MW

References:
    Hingorani, N.G. & Gyugyi, L. (2000). Understanding FACTS. IEEE Press. Ch. 5.
    Cigre TB 25 (1986). Static VAR Compensators.
    ABB SVC Application Guide (2018).
    IEEE Std 1031-2011: Guide for Functional Specifications for Transmission SVCs.
"""

import numpy as np


class SVCF1b:
    """SVC (TCR+TSC) -- detailed conduction + switching loss model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_max = u["Q_max_MVAR"]["value"]           # MVAR, capacitive
        self.Q_min = u["Q_min_MVAR"]["value"]           # MVAR, inductive (negative)
        self.V_rated_kV = u["V_rated_kV"]["value"]
        self.V_ref_pu = u["V_ref_pu"]["value"]
        self.V_T0 = u["thyristor_V_T0"]["value"]        # V
        self.r_T = u["thyristor_r_T"]["value"]           # Ohm per thyristor
        self.n_T = u["n_thyristors_series"]["value"]     # series thyristors per valve
        self.R_reactor = u["reactor_R_ohm"]["value"]     # Ohm
        self.ESR_TSC = u["tsc_ESR_ohm"]["value"]         # Ohm
        self.P_cooling = u["cooling_loss_MW"]["value"]   # MW
        self.Q_rated = u["Q_rated_MVAR"]["value"]
        self.f = u["f_system_Hz"]["value"]

    def _rms_current(self, Q_MVAR: float) -> float:
        """RMS current [A] for given reactive power output [MVAR]."""
        Q = np.asarray(np.abs(Q_MVAR), dtype=float)
        V_V = self.V_rated_kV * 1000.0
        return Q * 1e6 / (np.sqrt(3.0) * V_V)  # A

    def _thyristor_loss_MW(self, I_rms: float, n_valves: int = 2) -> float:
        """
        Thyristor conduction loss [MW].
        n_valves: 2 for TCR (two back-to-back thyristors); 2 for TSC switching
        I_avg = I_rms * sqrt(2) / pi  (sinusoidal approximation)
        """
        I_avg = I_rms * np.sqrt(2.0) / np.pi
        P = n_valves * self.n_T * (self.V_T0 * I_avg + self.r_T * I_rms ** 2)
        return P / 1e6  # W -> MW

    def compute(self, Q_demand_MVAR: float, V_pu: float = 1.0) -> dict:
        """
        Parameters
        ----------
        Q_demand_MVAR : Requested reactive output [MVAR]; + = capacitive
        V_pu          : Terminal voltage per-unit

        Returns
        -------
        dict with:
            Q_out_MVAR, Q_limited, Q_effective_MVAR (after V-correction),
            P_thyristor_MW, P_reactor_MW, P_ESR_TSC_MW, P_cooling_MW, P_loss_MW,
            I_rms_A, operating_mode, utilization
        """
        Q_dem = np.asarray(Q_demand_MVAR, dtype=float)
        V = np.asarray(V_pu, dtype=float)

        # Clamp to rated range
        Q_out = np.clip(Q_dem, self.Q_min, self.Q_max)
        Q_limited = Q_dem != Q_out

        # Voltage correction:
        #   Capacitive (TSC): Q_cap ∝ V^2  (capacitor admittance × V^2)
        #   Inductive (TCR):  Q_ind ∝ V^2  (similar, firing-angle adjusts)
        # Both scale with V^2 / V_ref^2 at F1b
        V_ratio_sq = (V / self.V_ref_pu) ** 2
        Q_effective = Q_out * V_ratio_sq

        # Operating mode
        is_cap = Q_effective > 0.01
        is_ind = Q_effective < -0.01
        is_stby = ~is_cap & ~is_ind

        # RMS currents
        I_cap = self._rms_current(np.where(is_cap, Q_effective, 0.0))
        I_ind = self._rms_current(np.where(is_ind, Q_effective, 0.0))

        # Losses per mode
        P_thy_ind = self._thyristor_loss_MW(I_ind, n_valves=2)  # TCR (2 thyristors in series each half cycle, 2 per phase -> n_T each)
        P_reactor = self.R_reactor * I_ind ** 2 / 1e6             # MW
        P_ESR_TSC = self.ESR_TSC * I_cap ** 2 / 1e6               # MW
        P_thy_cap = self._thyristor_loss_MW(I_cap, n_valves=2)    # TSC switching losses (minor)

        # Combine (only active path contributes)
        P_thyristor_MW = np.where(is_ind, P_thy_ind, np.where(is_cap, P_thy_cap, 0.0)) if np.ndim(Q_out) > 0 else (P_thy_ind if float(Q_effective) < 0 else (P_thy_cap if float(Q_effective) > 0 else 0.0))
        P_reactor_MW = np.where(is_ind, P_reactor, 0.0) if np.ndim(Q_out) > 0 else (P_reactor if float(Q_effective) < 0 else 0.0)
        P_ESR_MW = np.where(is_cap, P_ESR_TSC, 0.0) if np.ndim(Q_out) > 0 else (P_ESR_TSC if float(Q_effective) > 0 else 0.0)

        P_loss_MW = P_thyristor_MW + P_reactor_MW + P_ESR_MW + self.P_cooling

        # Mode string
        if np.ndim(Q_out) == 0:
            if float(Q_effective) > 0.01:
                mode = "capacitive"
            elif float(Q_effective) < -0.01:
                mode = "inductive"
            else:
                mode = "standby"
        else:
            mode = "vectorized"

        utilization = np.abs(Q_out) / (max(abs(self.Q_max), abs(self.Q_min)) + 1e-12)

        return {
            "Q_out_MVAR": Q_out,
            "Q_effective_MVAR": Q_effective,
            "Q_limited": Q_limited,
            "P_thyristor_MW": P_thyristor_MW,
            "P_reactor_MW": P_reactor_MW,
            "P_ESR_TSC_MW": P_ESR_MW,
            "P_cooling_MW": np.full_like(P_loss_MW, self.P_cooling) if np.ndim(P_loss_MW) > 0 else self.P_cooling,
            "P_loss_MW": P_loss_MW,
            "I_rms_A": np.maximum(I_cap, I_ind),
            "operating_mode": mode,
            "utilization": utilization,
        }
