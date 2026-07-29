"""
EC087 — Biomass Boiler — F1b Flue Gas Loss + Moisture Correction + Cycling Standby

Extends F1a (constant efficiency) with:
  1. Quadratic part-load curve: eta(PLR) = a0 + a1*PLR + a2*PLR^2
  2. Flue gas sensible-heat loss with:
       a. Excess air ratio (lambda) for biomass (typically 1.2-1.5)
       b. Moisture correction: evaporating fuel moisture increases flue volume and
          reduces effective calorific value (Obernberger & Thek 2008)
       c. Flue temperature scaling with PLR
  3. Cycling standby loss: constant casing + cycling losses that penalise
     frequent start/stop at low PLR

Effective LHV with moisture:
    LHV_eff = LHV_dry * (1 - w) - h_fg * w
    where w = moisture_content [mass fraction, wet basis]
          h_fg = 2442 kJ/kg (latent heat of water at ~25 degC)

Fuel input based on corrected LHV:
    Q_fuel = Q_out / eta(PLR)
    m_fuel = Q_fuel / LHV_eff   [kg/s]

Flue gas mass flow:
    m_flue = m_fuel * [(1 - w) * (1 + lambda * AFR_stoich) + w]
    (dry combustion products + excess air + moisture from fuel)

Flue gas heat loss:
    Q_flue = m_flue * cp_flue * (T_flue - T_air)

References:
    EN 303-5:2012 — Heating boilers — Solid fuel heating boilers.
    Obernberger, I. & Thek, G. (2008) The Pellet Handbook. Earthscan.
    Jenkins, B.M. et al. (1998) Prog. Energy Comb. Sci. 24, 47-81.
    BRE Report BR 443 (2006) Conventions for U-value calculations.
"""

import numpy as np

_H_FG = 2442.0   # kJ/kg — latent heat of water at ~25 degC


class BiomassBOilerF1b:
    """Biomass boiler with part-load curve, moisture-corrected flue loss, and cycling standby."""

    def __init__(self, params: dict):
        self.Q_rated      = float(params["Q_rated"])
        self.a0           = float(params["a0"])
        self.a1           = float(params["a1"])
        self.a2           = float(params["a2"])
        self.PLR_min      = float(params["PLR_min"])
        self.LHV_dry      = float(params["LHV_dry_MJ_kg"]) * 1000.0  # kJ/kg
        self.moisture     = float(params["moisture_content"])          # [0, 1]
        self.excess_air   = float(params["excess_air_ratio"])
        self.stoich_afr   = float(params["stoich_afr"])
        self.cp_flue      = float(params["flue_gas_cp"])               # kJ/kgK
        self.T_flue_full  = float(params["T_flue_full"])               # degC
        self.T_ambient    = float(params["T_ambient"])                 # degC
        self.standby_frac = float(params["standby_loss_fraction"])
        self.cycling_frac = float(params["cycling_loss_fraction"])

        # Pre-compute moisture-corrected LHV
        self._LHV_eff = self.LHV_dry * (1.0 - self.moisture) - _H_FG * self.moisture

    # ------------------------------------------------------------------

    @property
    def LHV_eff_kj_kg(self):
        """Effective LHV on wet fuel basis [kJ/kg]."""
        return self._LHV_eff

    # ------------------------------------------------------------------
    # Part-load efficiency
    # ------------------------------------------------------------------

    def efficiency(self, PLR):
        """
        eta(PLR) = (a0 + a1*PLR + a2*PLR^2) * moisture_factor

        moisture_factor = LHV_eff / LHV_dry
        Higher moisture content reduces LHV of wet fuel, which means less
        available energy per unit mass. The polynomial coefficients are
        calibrated for dry fuel; moisture penalises the fraction of calorific
        value actually available after evaporating the fuel moisture.

        Peaks near PLR ~ -a1/(2*a2).
        """
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        eta_base = self.a0 + self.a1 * PLR_eff + self.a2 * PLR_eff ** 2
        # Moisture correction: penalise by ratio of effective to dry LHV
        moisture_factor = max(self._LHV_eff / self.LHV_dry, 0.0)
        return np.clip(eta_base * moisture_factor, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Thermal output / fuel
    # ------------------------------------------------------------------

    def heat_output_kw(self, PLR):
        PLR = np.asarray(PLR, dtype=float)
        return np.maximum(PLR, self.PLR_min) * self.Q_rated

    def fuel_input_kw(self, PLR):
        """Q_fuel = Q_out / eta [kW, LHV basis]."""
        Q_out = self.heat_output_kw(PLR)
        eta = self.efficiency(PLR)
        safe = np.where(eta > 0.01, eta, 0.01)
        return Q_out / safe

    def fuel_mass_flow_kg_s(self, PLR):
        """Wet-basis fuel mass flow [kg/s]."""
        Q_fuel = self.fuel_input_kw(PLR)   # kW = kJ/s
        LHV = max(self._LHV_eff, 1.0)     # guard
        return Q_fuel / LHV

    # ------------------------------------------------------------------
    # Flue gas loss
    # ------------------------------------------------------------------

    def flue_gas_temp(self, PLR):
        """
        Flue temperature scales with PLR.
        T_flue(PLR) = T_air + (T_full - T_air) * (0.35 + 0.65*PLR)
        Biomass has higher flue temp at full load (char/ash effects).
        """
        PLR = np.asarray(PLR, dtype=float)
        PLR_eff = np.maximum(PLR, self.PLR_min)
        return self.T_ambient + (self.T_flue_full - self.T_ambient) * (0.35 + 0.65 * PLR_eff)

    def flue_loss_kw(self, PLR, T_flue_override=None):
        """
        Flue gas sensible heat loss [kW].

        m_flue = m_fuel * [(1-w)*(1 + lambda*AFR) + w]
        Q_flue = m_flue * cp * (T_flue - T_air)

        Moisture correction: wet fuel mass flow already includes water;
        water evaporation joins the flue stream.
        """
        PLR = np.asarray(PLR, dtype=float)
        m_fuel = self.fuel_mass_flow_kg_s(PLR)

        # dry-fuel fraction of m_fuel contributes combustion products
        # water fraction enters as steam in flue
        m_flue = m_fuel * (
            (1.0 - self.moisture) * (1.0 + self.excess_air * self.stoich_afr)
            + self.moisture
        )

        T_flue = T_flue_override if T_flue_override is not None else self.flue_gas_temp(PLR)
        T_flue = np.asarray(T_flue, dtype=float)

        return m_flue * self.cp_flue * (T_flue - self.T_ambient)

    # ------------------------------------------------------------------
    # Standby + cycling loss
    # ------------------------------------------------------------------

    def standby_loss_kw(self):
        """Constant standby heat loss [kW]: casing + induced-draft fan idle."""
        return self.standby_frac * self.Q_rated

    def cycling_loss_kw(self, PLR):
        """
        Cycling loss [kW]: increases at low PLR as start/stop frequency rises.
        Approximated as: Q_cycle = cycling_frac * Q_rated * (1 - PLR)
        At PLR=1 there is no cycling. At PLR=0.15 cycling is maximum.
        """
        PLR = np.asarray(PLR, dtype=float)
        return self.cycling_frac * self.Q_rated * (1.0 - np.maximum(PLR, self.PLR_min))

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, PLR, T_flue_override=None):
        PLR = np.asarray(PLR, dtype=float)
        return {
            "efficiency":       self.efficiency(PLR),
            "heat_output_kw":   self.heat_output_kw(PLR),
            "fuel_input_kw":    self.fuel_input_kw(PLR),
            "flue_loss_kw":     self.flue_loss_kw(PLR, T_flue_override),
            "standby_loss_kw":  np.full_like(PLR, self.standby_loss_kw()),
            "cycling_loss_kw":  self.cycling_loss_kw(PLR),
            "flue_gas_temp_c":  self.flue_gas_temp(PLR),
        }
