"""
EC089 — Hydrogen Boiler — F1b Flue Loss (H2O-rich exhaust) + Condensing Mode + Standby

Extends F1a (constant efficiency) with:
  1. Quadratic part-load curve: eta(PLR) = a0 + a1*PLR + a2*PLR^2
  2. H2-specific flue gas model:
       - H2 combustion produces ONLY water vapour (+ nitrogen from air)
         2 H2 + O2 -> 2 H2O
       - Flue gas is H2O-rich: cp_flue ~ 1.9 kJ/kgK (vs 1.05 for NG)
       - Dew point of H2 flue (~55-58 degC) higher than NG (~50-55 degC)
         because H2O mole fraction is much higher
  3. Condensing vs non-condensing modes:
       - Non-condensing: flue exits above dew point; only sensible heat is lost
       - Condensing: flue cooled below dew point; latent heat partially recovered
         eta_condensing_boost = (HHV - LHV) / LHV * recovery_fraction
  4. Standby loss from thermal mass

Stoichiometry:
    2 H2 + O2 -> 2 H2O
    Mass balance: 1 kg H2 + 8 kg O2 -> 9 kg H2O
    Stoich AFR = 8 (from air mass perspective: 8 kg O2 / 0.232 = 34.5 kg air per kg H2)
    Wait — standard value: AFR_stoich(H2) = 34.3 (mass of air / mass of fuel)

Flue gas composition:
    m_air   = m_fuel * lambda * AFR_stoich
    m_H2O   = m_fuel * 9.0   (mass ratio, from stoichiometry)
    m_flue  = m_air + m_H2O  (approximately, nitrogen dominates by mass)
    But mass-weighted cp is dominated by water fraction => higher cp

References:
    Cellek, M.S. & Pinarbasi, A. (2018) Int. J. Hydrogen Energy 43, 1194-1207.
    Woolley, E. et al. (2022) Appl. Energy 323, 119577.
    Hy4Heat WP6 Technical Report (2021). BEIS, UK.
    Gas Quality Harmonisation (ACER/CEER 2021).
"""

import numpy as np

# Stoichiometric air-fuel mass ratio for H2
_AFR_STOICH_H2 = 34.3  # kg air / kg H2


class HydrogenBoilerF1b:
    """Hydrogen boiler with H2O-rich flue loss model and condensing mode."""

    def __init__(self, params: dict):
        self.Q_rated       = float(params["Q_rated"])
        self.a0            = float(params["a0"])
        self.a1            = float(params["a1"])
        self.a2            = float(params["a2"])
        self.PLR_min       = float(params["PLR_min"])
        self.LHV_H2        = float(params["LHV_H2_MJ_kg"]) * 1000.0   # kJ/kg
        self.HHV_H2        = float(params["HHV_H2_MJ_kg"]) * 1000.0   # kJ/kg
        self.excess_air    = float(params["excess_air_ratio"])
        self.T_flue_noncond = float(params["T_flue_noncond"])
        self.T_flue_cond   = float(params["T_flue_cond"])
        self.T_ambient     = float(params["T_ambient"])
        self.cp_flue       = float(params["flue_gas_cp"])
        self.condensing    = bool(params["condensing"])
        self.standby_frac  = float(params["standby_loss_fraction"])

    # ------------------------------------------------------------------

    def _T_flue_design(self):
        return self.T_flue_cond if self.condensing else self.T_flue_noncond

    # ------------------------------------------------------------------
    # Part-load efficiency (LHV basis)
    # ------------------------------------------------------------------

    def efficiency(self, PLR):
        """
        eta_LHV(PLR) = a0 + a1*PLR + a2*PLR^2

        For condensing mode, add latent recovery benefit.
        The condensing boost is embedded in the polynomial coefficients,
        which are calibrated to reflect condensing vs non-condensing operation.
        """
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        eta = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        return np.clip(eta, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Fuel and heat flows
    # ------------------------------------------------------------------

    def heat_output_kw(self, PLR):
        PLR = np.asarray(PLR, dtype=float)
        return np.maximum(PLR, self.PLR_min) * self.Q_rated

    def fuel_input_kw(self, PLR):
        """Fuel thermal input on LHV basis [kW]."""
        Q_out = self.heat_output_kw(PLR)
        eta = self.efficiency(PLR)
        safe = np.where(eta > 0.01, eta, 0.01)
        return Q_out / safe

    def h2_mass_flow_kg_s(self, PLR):
        """Hydrogen mass flow [kg/s]."""
        Q_fuel = self.fuel_input_kw(PLR)   # kJ/s
        return Q_fuel / self.LHV_H2

    # ------------------------------------------------------------------
    # H2O-rich flue gas loss
    # ------------------------------------------------------------------

    def flue_gas_temp(self, PLR):
        """
        Flue gas exit temperature.
        Scales with PLR: lower load => lower flame => cooler flue.
        T_flue(PLR) = T_amb + (T_flue_design - T_amb) * (0.4 + 0.6*PLR)
        """
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        T_design = self._T_flue_design()
        return self.T_ambient + (T_design - self.T_ambient) * (0.4 + 0.6 * PLR_eff)

    def flue_loss_kw(self, PLR, T_flue_override=None):
        """
        Flue gas sensible heat loss [kW].

        H2 combustion: 1 kg H2 -> 9 kg H2O + (lambda-1)*AFR_stoich kg excess air
        m_flue = m_H2 * (lambda * AFR_stoich + 9)
        Q_flue = m_flue * cp_flue * (T_flue - T_air)

        The factor (9) comes from 1 kg H2 + 8 kg O2 -> 9 kg H2O.
        Excess air adds additional nitrogen/O2 to flue stream.
        """
        PLR = np.asarray(PLR, dtype=float)
        m_H2 = self.h2_mass_flow_kg_s(PLR)

        # Flue = combustion products (H2O) + air (lambda * AFR stoich)
        # Per kg H2: 9 kg H2O + lambda * AFR_stoich kg air
        m_flue = m_H2 * (self.excess_air * _AFR_STOICH_H2 + 9.0)

        if T_flue_override is not None:
            T_flue = np.asarray(T_flue_override, dtype=float)
        else:
            T_flue = self.flue_gas_temp(PLR)

        # cp_flue accounts for H2O-rich mixture (~1.9 kJ/kgK)
        Q_flue = m_flue * self.cp_flue * (T_flue - self.T_ambient)
        return np.maximum(Q_flue, 0.0)

    def latent_recovery_kw(self, PLR):
        """
        Latent heat recovered in condensing mode [kW].
        = m_H2O_condensed * h_fg  (approximate, partial condensation)
        Only non-zero in condensing mode.
        For condensing: assume ~50% of H2O condensed.
        """
        if not self.condensing:
            PLR = np.asarray(PLR, dtype=float)
            return np.zeros_like(PLR)

        PLR = np.asarray(PLR, dtype=float)
        m_H2 = self.h2_mass_flow_kg_s(PLR)
        m_H2O_total = m_H2 * 9.0      # kg/s water vapour produced
        condensation_fraction = 0.50   # ~50% condensed in typical condensing boiler
        h_fg = 2442.0                  # kJ/kg
        return m_H2O_total * condensation_fraction * h_fg  # kW

    # ------------------------------------------------------------------
    # Standby loss
    # ------------------------------------------------------------------

    def standby_loss_kw(self):
        return self.standby_frac * self.Q_rated

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, T_flue_override=None):
        PLR = np.asarray(PLR, dtype=float)
        return {
            "efficiency":          self.efficiency(PLR),
            "heat_output_kw":      self.heat_output_kw(PLR),
            "fuel_input_kw":       self.fuel_input_kw(PLR),
            "flue_loss_kw":        self.flue_loss_kw(PLR, T_flue_override),
            "latent_recovery_kw":  self.latent_recovery_kw(PLR),
            "standby_loss_kw":     np.full_like(PLR, self.standby_loss_kw()),
            "h2_flow_kg_s":        self.h2_mass_flow_kg_s(PLR),
            "flue_gas_temp_c":     self.flue_gas_temp(PLR),
            "condensing":          np.full_like(PLR, float(self.condensing)),
        }
