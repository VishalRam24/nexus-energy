"""
EC188 — SMES — F1b Self-Discharge + Cryogenic Cooling Load Model

Extends F1a energy model with:
  1. Superconductor self-discharge / flux creep:
     In practice, HTS coils have near-zero DC resistance, but AC ripple from
     the converter causes periodic field changes → AC losses.
     Self-discharge rate modeled as:
         dE/dt|_self = -P_ac_loss(I) - P_cryo
     AC loss in HTS tape (Norris model for elliptical/rectangular tape):
         Q_ac = 4 * mu_0 * Ic^2 * f * (1 - (I_ac/Ic)) * ...
     Simplified to: P_ac_loss = k_ac * I_coil^n [W], with k_ac and n fitted
     to HTS tape data (typical k_ac from Norris formula).
  2. Cryogenic cooling load model (Stirling / GM cooler):
     P_cryo_actual = P_cryo_rated * (1 + k_cryo_T * (T_op - T_rated)) / COP_cryo
     Total cooling load reflects both static heat leak (conduction, radiation)
     and AC loss dissipation.
  3. Round-trip efficiency accounting for self-discharge over cycle:
     eta_RT = (E_out_MJ) / (E_in_MJ) = 1 - (P_self * t) / E_stored
  4. Idle self-discharge time constant: tau = E_max / (P_cryo + P_ac_idle)

References:
    Buckles, W. & Hassenzahl, W.V. (2000). Superconducting magnetic energy storage.
        IEEE Power Eng. Rev. 20(5):16-20.
    Kalsi, S.S. (2011). Applications of HTS Superconductors. Wiley. Ch. 6.
    Norris, W.T. (1970). Calculation of hysteresis losses in hard superconductors
        carrying AC: isolated conductors and edges of thin sheets.
        J. Phys. D: Appl. Phys. 3:489-507.
    Xue, X.D. et al. (2006). Study of art of automotive active suspensions.
        (COP_cryo reference: commercial GM cooler ~0.5-2% Carnot).
"""

import numpy as np


class SMESF1b:
    """
    SMES F1b model: energy storage + converter + cryogenic cooling load.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["cryo"]

        # Electrical
        self.L           = u["L_H"]["value"]
        self.I_max       = u["I_max_A"]["value"]
        self.I_min       = u["I_min_A"]["value"]
        self.P_rated     = u["P_rated_MW"]["value"]
        self.eta_conv    = u["eta_converter"]["value"]

        # AC losses (Norris-type simplified)
        self.k_ac        = u["k_ac_loss"]["value"]        # W/(A^n)
        self.n_ac        = u["n_ac_loss"]["value"]        # exponent (typically 2-3 for HTS)
        self.f_ripple    = u["f_ripple_Hz"]["value"]      # converter ripple frequency [Hz]

        # Cryogenic
        self.P_cryo_rated   = c["P_cryo_rated_MW"]["value"]  # MW at rated operating T
        self.T_op_rated     = c["T_op_rated_K"]["value"]     # K nominal (e.g., 4.2 K LTS or 20-40 K HTS)
        self.T_cryo_ref     = c["T_cryo_ref_K"]["value"]     # K where P_cryo_rated was measured
        self.COP_cryo       = c["COP_cryo"]["value"]         # actual COP (fraction of Carnot)
        self.T_ambient_K    = c["T_ambient_K"]["value"]      # K
        self.k_cryo_T       = c["k_cryo_T"]["value"]         # sensitivity [1/K]
        self.heat_leak_MW   = c["heat_leak_MW"]["value"]     # static conduction/radiation

        self.E_max     = 0.5 * self.L * self.I_max ** 2   # J
        self.E_max_MJ  = self.E_max / 1e6
        self.E_min     = 0.5 * self.L * self.I_min ** 2
        self.E_min_MJ  = self.E_min / 1e6

    # ------------------------------------------------------------------
    # AC losses (self-discharge mechanism)
    # ------------------------------------------------------------------
    def ac_loss_power_W(self, I_coil_A):
        """
        AC hysteresis/eddy loss [W] in HTS tape due to converter current ripple.
        P_ac = k_ac * |I_coil|^n_ac

        Physical basis (Norris model simplified):
            Q_ac ∝ I^3/Ic for I << Ic, so n~2-3 depending on regime.
        Note: very small at design (typical 0.01-0.1% of stored energy per cycle).
        """
        I = np.asarray(I_coil_A, dtype=float)
        return self.k_ac * np.abs(I) ** self.n_ac

    # ------------------------------------------------------------------
    # Cryogenic cooling load
    # ------------------------------------------------------------------
    def cryo_power_MW(self, T_op_K=None, P_ac_loss_W=0.0):
        """
        Total cryogenic cooling load [MW] drawn from the grid.

        P_cryo_total = (heat_leak + P_ac_loss) / COP_cryo

        where COP_cryo includes the Carnot-limit fraction:
            COP_Carnot = T_cold / (T_warm - T_cold)
            COP_actual = COP_efficiency * COP_Carnot
        Temperature correction: P_cryo scales if T_op deviates from design.
        """
        T_op = self.T_op_rated if T_op_K is None else np.asarray(T_op_K, dtype=float)
        P_ac = np.asarray(P_ac_loss_W, dtype=float)

        # Heat load to be removed from cold end [W]
        heat_load_W = self.heat_leak_MW * 1e6 + P_ac   # W

        # COP of actual refrigerator
        T_warm = self.T_ambient_K
        T_cold = T_op
        COP_Carnot = T_cold / np.maximum(T_warm - T_cold, 1.0)
        COP_actual = self.COP_cryo * COP_Carnot

        # Grid power to run the cryocooler
        P_grid_cryo_W = heat_load_W / np.maximum(COP_actual, 1e-6)
        return P_grid_cryo_W / 1e6  # MW

    # ------------------------------------------------------------------
    # Self-discharge time constant
    # ------------------------------------------------------------------
    def self_discharge_tau_h(self, SOC=1.0, T_op_K=None):
        """
        Time constant for self-discharge to E_min [hours].
        tau = (E - E_min) / (P_cryo + P_ac_loss)

        At full SOC: I = I_max → highest AC losses.
        """
        SOC_c = np.asarray(np.clip(SOC, 0.0, 1.0), dtype=float)
        E_MJ  = SOC_c * self.E_max_MJ
        dE    = np.maximum(E_MJ - self.E_min_MJ, 0.0)

        I_coil = np.sqrt(2.0 * E_MJ * 1e6 / (self.L + 1e-30))  # A
        P_ac_W = self.ac_loss_power_W(I_coil)
        P_cryo = self.cryo_power_MW(T_op_K, P_ac_W) * 1e6       # W

        P_self = P_cryo + P_ac_W
        tau_s  = dE * 1e6 / np.maximum(P_self, 1.0)             # seconds
        return tau_s / 3600.0                                    # hours

    # ------------------------------------------------------------------
    # Full charge/discharge cycle
    # ------------------------------------------------------------------
    def compute(self, SOC: float, P_request_MW: float,
                mode: str = "discharge", dt_s: float = 1.0,
                T_op_K: float = None) -> dict:
        """
        Full SMES computation with self-discharge and cryo loading.

        Parameters
        ----------
        SOC           : State of charge [0-1]
        P_request_MW  : Requested power [MW]
        mode          : "charge" or "discharge"
        dt_s          : Time step [s]
        T_op_K        : Operating temperature [K]

        Returns
        -------
        dict with P_delivered_MW, P_grid_MW, P_cryo_load_MW, P_ac_loss_MW,
        SOC_new, E_stored_MJ, eta_instantaneous, eta_rt_estimate,
        self_discharge_tau_h, I_coil_A
        """
        SOC = np.asarray(np.clip(SOC, 0.0, 1.0), dtype=float)
        P_req = np.asarray(np.clip(P_request_MW, 0.0, self.P_rated), dtype=float)

        # Coil current from SOC
        E_MJ = SOC * self.E_max_MJ
        I_coil = np.sqrt(2.0 * E_MJ * 1e6 / (self.L + 1e-30))
        I_coil = np.clip(I_coil, 0.0, self.I_max)

        # AC losses
        P_ac_W = self.ac_loss_power_W(I_coil)
        P_ac_MW = P_ac_W / 1e6

        # Cryogenic load
        P_cryo_MW = self.cryo_power_MW(T_op_K, P_ac_W)

        if mode == "discharge":
            P_dc = P_req
            P_ac = P_dc * self.eta_conv
            P_grid = P_ac - P_cryo_MW
            P_grid = np.maximum(P_grid, 0.0)
            dE = -(P_dc + P_ac_MW) * dt_s / 1e6   # net energy lost (power + self-losses)
        else:  # charge
            P_ac = P_req
            P_dc = P_ac * self.eta_conv
            P_grid = -(P_ac + P_cryo_MW)  # total drawn from grid
            dE = (P_dc - P_ac_MW) * dt_s / 1e6   # net energy stored

        SOC_new = np.clip(SOC + dE / (self.E_max_MJ + 1e-30), 0.0, 1.0)
        E_stored_MJ = SOC_new * self.E_max_MJ

        # Instantaneous efficiency
        if mode == "discharge":
            denom = np.where(P_req > 0, P_req, 1e-12) if np.ndim(P_req) > 0 else (P_req if P_req > 0 else 1e-12)
            eta_inst = np.where(P_req > 0, (P_ac - P_cryo_MW) / denom, 0.0) if np.ndim(P_req) > 0 else ((P_ac - P_cryo_MW) / denom if P_req > 0 else 0.0)
            eta_inst = np.maximum(eta_inst, 0.0)
        else:
            total_in = P_ac + P_cryo_MW
            safe_in = np.where(total_in > 0, total_in, 1e-12) if np.ndim(total_in) > 0 else (total_in if total_in > 0 else 1e-12)
            eta_inst = np.where(total_in > 0, P_dc / safe_in, 0.0) if np.ndim(total_in) > 0 else (P_dc / safe_in if total_in > 0 else 0.0)

        # Round-trip efficiency estimate: eta_conv^2 * (1 - P_cryo/P_charge_avg)
        eta_rt = self.eta_conv ** 2 * np.maximum(1.0 - P_cryo_MW / (P_req + 1e-12), 0.5)

        tau_h = self.self_discharge_tau_h(SOC, T_op_K)

        return {
            "P_delivered_MW":     P_ac if mode == "discharge" else P_dc,
            "P_grid_MW":          P_grid,
            "P_cryo_load_MW":     P_cryo_MW,
            "P_ac_loss_MW":       P_ac_MW,
            "SOC_new":            SOC_new,
            "E_stored_MJ":        E_stored_MJ,
            "I_coil_A":           I_coil,
            "eta_instantaneous":  eta_inst,
            "eta_rt_estimate":    eta_rt,
            "self_discharge_tau_h": tau_h,
            "mode":               mode,
        }
