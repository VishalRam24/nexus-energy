"""
EC187 — HVDC Converter Station — F1b Full Link: Station + DC Line + Reactive Model

Extends F1a (single station loss model) with:
  1. Full point-to-point link: rectifier + DC cable/OHL losses + inverter.
  2. DC line resistance temperature correction:
         R_line(T) = R_line_ref * [1 + alpha * (T_line - T_ref)]
  3. LCC reactive power demand model:
         Q_LCC = Q_factor * P_transfer   (typically 0.4–0.6 at rated power)
     Reactive demand varies with firing angle; at part-load it increases.
         Q_LCC(PLR) = P_rated * (q_a + q_b / (PLR + eps))   [MVAR]
  4. Harmonic filter / shunt capacitor offset: Q_net = Q_LCC - Q_cap
  5. DC line current and I^2*R losses.
  6. Overall link efficiency from AC in to AC out.

References:
    Cigre TB 492 (2012). VSC Transmission. WG B4-37.
    Cigre TB 388 (2009). VSC HVDC for Power Transmission. WG B4-37.
    Kundur, P. (1994). Power System Stability and Control. McGraw-Hill. Ch. 8.
    ABB (2019). HVDC Reference Projects Technical Overview.
    EPRI (2012). HVDC Transmission Design Guide.
"""

import numpy as np


class HVDCLinkF1b:
    """
    Full HVDC link model: rectifier station + DC line + inverter station.
    Supports LCC and VSC converter types.
    """

    def __init__(self, params: dict):
        u = params["unit"]

        self.P_rated        = u["P_rated_MW"]["value"]             # MW
        self.V_dc_kV        = u["V_dc_kV"]["value"]               # kV (one pole)
        self.converter_type = u["converter_type"]["value"]         # "LCC" or "VSC"
        self.loss_factor    = u["loss_factor_station"]["value"]    # fraction per station
        self.P_no_load      = u["P_no_load_MW"]["value"]          # MW per station
        self.line_length    = u["line_length_km"]["value"]        # km
        self.R_line_per_km  = u["line_R_ohm_per_km"]["value"]    # Ohm/km
        self.Q_dem_factor   = u["Q_demand_factor"]["value"]       # for LCC
        self.Q_cap          = u["Q_cap_MVAR"]["value"]            # shunt capacitor bank per station
        self.alpha_line     = u["alpha_line_R"]["value"]          # 1/K
        self.T_ref_line     = u["T_ref_line"]["value"]            # degC

    # ------------------------------------------------------------------
    # Station losses
    # ------------------------------------------------------------------
    def station_losses_MW(self, P_transfer_MW: np.ndarray) -> np.ndarray:
        """Per-station losses [MW]: no-load + proportional."""
        P = np.clip(np.asarray(P_transfer_MW, dtype=float), 0.0, self.P_rated)
        return self.P_no_load + self.loss_factor * P

    # ------------------------------------------------------------------
    # DC line losses
    # ------------------------------------------------------------------
    def dc_line_resistance(self, T_line_C: float = 20.0) -> float:
        """Total DC line resistance [Ohm] with temperature correction (bipolar: 2 poles)."""
        R_per_km = self.R_line_per_km * (1.0 + self.alpha_line * (T_line_C - self.T_ref_line))
        # Bipolar link: total resistance = R_per_km * length (each pole carries I_dc)
        return R_per_km * self.line_length   # Ohm per pole

    def dc_current_kA(self, P_transfer_MW: np.ndarray) -> np.ndarray:
        """DC current [kA] = P / (V_dc * 2) for bipolar link."""
        P = np.asarray(P_transfer_MW, dtype=float)
        V_pole = self.V_dc_kV * 1000.0   # V
        I_kA = P * 1e6 / (2.0 * V_pole * 1000.0)  # kA
        return I_kA

    def line_losses_MW(self, P_transfer_MW: np.ndarray, T_line_C: float = 20.0) -> np.ndarray:
        """DC line ohmic losses [MW] = I_dc^2 * R_line (both poles)."""
        I_kA = self.dc_current_kA(P_transfer_MW)
        I_A  = I_kA * 1000.0
        R_ohm = self.dc_line_resistance(T_line_C)  # one pole
        P_loss_W = 2.0 * I_A ** 2 * R_ohm         # both poles
        return P_loss_W / 1e6  # MW

    # ------------------------------------------------------------------
    # LCC reactive power demand
    # ------------------------------------------------------------------
    def lcc_reactive_demand_MVAR(self, P_transfer_MW: np.ndarray) -> np.ndarray:
        """
        LCC reactive demand [MVAR].
        Q_LCC = Q_factor * P  at rated (firing angle ~15 deg).
        At part-load, firing angle increases → Q_factor increases.
        Simplified Kundur model: Q_LCC = q_factor * P_rated * (0.1 + 0.5/(PLR+0.01))
        Net Q after harmonic filters: Q_net = Q_LCC - Q_cap (per station).
        """
        P = np.asarray(P_transfer_MW, dtype=float)
        PLR = P / (self.P_rated + 1e-12)
        if self.converter_type == "LCC":
            # Cigre TB 388: Q ~0.5*P at rated, rises at part-load
            Q_LCC = self.Q_dem_factor * self.P_rated * (0.1 + 0.9 * PLR)
            Q_net = np.maximum(Q_LCC - 2.0 * self.Q_cap, 0.0)   # two stations
        else:
            # VSC: fully controllable Q, no reactive demand from converter
            Q_net = np.zeros_like(P)
        return Q_net

    # ------------------------------------------------------------------
    # Full link efficiency
    # ------------------------------------------------------------------
    def compute(self, P_transfer_MW: float, T_line_C: float = 20.0) -> dict:
        """
        Full HVDC link analysis from AC in to AC out.

        Parameters
        ----------
        P_transfer_MW : Active power at rectifier DC output [MW]
        T_line_C      : DC line conductor temperature [degC]

        Returns
        -------
        dict with P_AC_in_MW, P_AC_out_MW, P_loss_total_MW,
        P_loss_rect_MW, P_loss_line_MW, P_loss_inv_MW,
        I_dc_kA, R_line_ohm, link_efficiency,
        Q_reactive_demand_MVAR (LCC only)
        """
        P = np.asarray(P_transfer_MW, dtype=float)
        P = np.clip(P, 0.0, self.P_rated)

        P_loss_rect = self.station_losses_MW(P)
        P_AC_in     = P + P_loss_rect             # AC input power

        P_loss_line = self.line_losses_MW(P, T_line_C)
        P_inv_in    = P - P_loss_line             # power arriving at inverter

        P_loss_inv  = self.station_losses_MW(P_inv_in)
        P_AC_out    = np.maximum(P_inv_in - P_loss_inv, 0.0)

        P_loss_total = P_AC_in - P_AC_out

        safe_in = np.where(P_AC_in > 0, P_AC_in, 1e-12) if np.ndim(P_AC_in) > 0 else (P_AC_in if P_AC_in > 0 else 1e-12)
        eta = np.where(P_AC_in > 0, P_AC_out / safe_in, 0.0) if np.ndim(P_AC_in) > 0 else (P_AC_out / safe_in if P_AC_in > 0 else 0.0)

        I_dc_kA = self.dc_current_kA(P)
        R_line  = self.dc_line_resistance(T_line_C)
        Q_dem   = self.lcc_reactive_demand_MVAR(P)

        return {
            "P_AC_in_MW":             P_AC_in,
            "P_AC_out_MW":            P_AC_out,
            "P_loss_total_MW":        P_loss_total,
            "P_loss_rect_MW":         P_loss_rect,
            "P_loss_line_MW":         P_loss_line,
            "P_loss_inv_MW":          P_loss_inv,
            "I_dc_kA":                I_dc_kA,
            "R_line_ohm":             R_line,
            "link_efficiency":        eta,
            "Q_reactive_demand_MVAR": Q_dem,
        }
