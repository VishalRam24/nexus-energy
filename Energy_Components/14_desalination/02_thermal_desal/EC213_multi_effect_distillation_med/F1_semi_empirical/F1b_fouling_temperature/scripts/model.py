"""
EC213 — Multi-Effect Distillation (MED) — F1b GOR + Top Brine Temperature + Scaling Model

Extends F1a with:
  1. GOR as function of N_effects and TBT:
     GOR ≈ N_effects * 0.8  (ideal, each effect reuses latent heat)
     More precisely: GOR = N_eff * eta_eff where eta_eff accounts for boiling
     point elevation (BPE), non-equilibrium allowance (NEA), and demister losses.
  2. TBT effect: higher TBT allows more effects (larger temperature range) but
     increases scaling risk (CaCO3 onset ~70 degC, harder to control than MSF).
  3. Scaling model: BPE and scaling are stronger in MED than MSF at same temperature.
     BPE correction: T_eff_i = T_steam - i * dT/N - BPE_cumulative
  4. Thermal compression option: TVC (thermo-vapor compressor) improves GOR.

References:
    El-Dessouky, H.T. et al. (2000). Chem. Eng. J., 79(2-3), 165-183.
    Alasfour, F.N. et al. (2005). Desalination, 174(3), 209-228.
    Ettouney, H. (2006). Desalination, 196(1-3), 132-155.
"""

import numpy as np

CP_BRINE  = 3.90    # kJ/(kg*K)


def _latent_heat(T_degC):
    """Latent heat of vaporization [kJ/kg] at temperature T."""
    T = np.asarray(T_degC, dtype=float)
    return np.clip(2501.0 - 2.37 * T, 1500.0, 2600.0)


class MEDF1b:
    """MED — GOR + TBT + scaling model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_effects      = u["N_effects"]["value"]
        self.TBT_ref        = u["TBT_ref"]["value"]         # degC
        self.T_last_effect  = u["T_last_effect"]["value"]   # degC condenser
        self.T_scale_lim    = u["T_scale_limit"]["value"]   # degC MED scaling onset
        self.k_scale        = u["k_scale"]["value"]
        self.eta_eff        = u["eta_eff"]["value"]          # per-effect efficiency
        self.BPE_per_effect = u["BPE_per_effect"]["value"]  # degC boiling point elevation
        self.pump_SEC_ref   = u["pump_SEC_ref"]["value"]    # kWh_e/m3

    # ------------------------------------------------------------------ #
    #  GOR
    # ------------------------------------------------------------------ #

    def gor(self, TBT_degC, N_effects=None):
        """GOR as function of TBT and N_effects.
        Temperature range available: TBT - T_last
        More range → more effects can be driven.
        Effective N_effects limited by dT/effect ≥ BPE_per_effect + NEA (≥ 2 degC).
        GOR = eta_eff * N_eff_actual
        """
        TBT  = np.asarray(TBT_degC, dtype=float)
        N_e  = N_effects if N_effects is not None else self.N_effects
        dT_total = np.clip(TBT - self.T_last_effect, 1.0, None)
        # Maximum effects limited by temperature range
        min_dT_per_effect = self.BPE_per_effect + 1.5  # BPE + NEA
        N_max = dT_total / min_dT_per_effect
        N_actual = np.minimum(np.full_like(TBT, float(N_e)), N_max)
        GOR = self.eta_eff * N_actual
        return np.clip(GOR, 2.0, 18.0)

    def thermal_sec_kwh_m3(self, TBT_degC, steam_temperature_degC=None):
        """Thermal SEC [kWh_th/m3 distillate].
        SEC_th = L_v(T_steam) / GOR / 3.6
        Scale penalty above T_scale_limit (MED: ~70-75 degC for CaCO3 with acid dosing).
        """
        TBT     = np.asarray(TBT_degC, dtype=float)
        T_stm   = steam_temperature_degC if steam_temperature_degC is not None else TBT + 10.0
        GOR_val = self.gor(TBT)
        L_v     = _latent_heat(T_stm)
        sec     = L_v / GOR_val / 3.6

        # Scale penalty
        excess       = np.clip(TBT - self.T_scale_lim, 0.0, None)
        scale_factor = 1.0 + self.k_scale * excess
        return np.clip(sec * scale_factor, 15.0, 300.0)

    def pump_sec_kwh_m3(self, plr):
        """Pumping SEC [kWh_e/m3]. MED pumping lower than MSF."""
        plr = np.asarray(plr, dtype=float)
        sec = self.pump_SEC_ref * (0.85 + 0.15 / np.clip(plr, 0.2, 1.0))
        return np.clip(sec, 0.3, 5.0)

    def scaling_risk(self, TBT_degC):
        """CaCO3 scaling risk index [0-1]. MED onset lower than MSF (acid dosing needed above 70C).
        Risk increases linearly from T_scale_lim to T_scale_lim + 15C.
        """
        TBT  = np.asarray(TBT_degC, dtype=float)
        risk = np.clip((TBT - self.T_scale_lim) / 15.0, 0.0, 1.0)
        return risk

    def bpr_correction_degC(self, N_effects=None):
        """Total boiling point rise correction [degC] — reduces effective temperature range."""
        N_e = N_effects if N_effects is not None else self.N_effects
        return float(N_e) * self.BPE_per_effect

    # ------------------------------------------------------------------ #
    #  Main compute
    # ------------------------------------------------------------------ #

    def compute(self, TBT_degC, plr, steam_temperature_degC=None):
        """Full computation.

        Parameters
        ----------
        TBT_degC                : degC  — top brine temperature (55-75 degC for LT-MED, up to 70C)
        plr                     : 0-1   — plant load ratio
        steam_temperature_degC  : degC  — steam supply temperature

        Returns
        -------
        dict with gor, thermal_sec_kwh_m3, pump_sec_kwh_m3, total_sec_kwh_m3,
                  scaling_risk_index, bpr_total_degC
        """
        TBT = np.asarray(TBT_degC, dtype=float)
        plr = np.asarray(plr, dtype=float)

        GOR_val   = self.gor(TBT)
        th_sec    = self.thermal_sec_kwh_m3(TBT, steam_temperature_degC)
        pump_sec  = self.pump_sec_kwh_m3(plr)
        total_sec = th_sec + pump_sec
        scale_r   = self.scaling_risk(TBT)
        bpr       = self.bpr_correction_degC()

        return {
            "gor":                  GOR_val,
            "thermal_sec_kwh_m3":   th_sec,
            "pump_sec_kwh_m3":      pump_sec,
            "total_sec_kwh_m3":     total_sec,
            "scaling_risk_index":   scale_r,
            "bpr_total_degC":       np.full_like(TBT, bpr),
        }
