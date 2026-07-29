"""
EC004 -- Phosphoric Acid Fuel Cell (PAFC) -- F1b Polarization-Thermal Model
Temperature-dependent polarization with H3PO4 electrolyte conductivity.

Extends F1a by making all loss mechanisms explicitly temperature-dependent:
  - Nernst potential: E_rev(T) = E0_ref - k_T*(T - T_ref)
                                  + RT/(2F)*ln(pH2 * sqrt(pO2))
    Typical PAFC: operates at 150–210 C (423–483 K).
    Temperature coefficient k_T ~ 0.000846 V/K for H2/O2.

  - H3PO4 electrolyte conductivity (Razaq 1989 / Appleby 1993):
        sigma_H3PO4(T) = sigma_ref * exp(E_act_sigma/R * (1/T_ref - 1/T))
    (Arrhenius-type; sigma strongly increases with T for concentrated H3PO4)

  - Ohmic loss: V_ohm = j * t_acid / sigma_H3PO4(T)

  - Exchange current density (Arrhenius):
        i0_cat(T) = i0_ref * exp(-E_act/R * (1/T - 1/T_ref))
    Cathode (O2 reduction) is rate-determining in PAFC.
    PAFC benefits greatly from elevated T: i0_cat at 200C >> i0_cat at 100C.

  - Activation loss: V_act = RT/(alpha*n*F) * arcsinh(j/(2*i0_cat(T)))

  - CO poisoning tolerance: PAFC can tolerate ~1-2% CO in reformate;
    not modelled here but noted in limitations.

  - Concentration loss: V_conc = -B*ln(1 - j/j_L)

  - Heat generation: Q = j*(E_tn - V_cell)  [E_tn = 1.254 V at 200C]
    Note: E_tn for PAFC is slightly lower than PEMFC due to higher T.

References:
    Razaq M. et al. (1989). J. Electrochem. Soc., 136(2), 385-390.
        H3PO4 conductivity model.
    Appleby A.J. & Foulkes F.R. (1989). Fuel Cell Handbook. Van Nostrand Reinhold.
    Patel K.K. et al. (2012). Int. J. Hydrogen Energy, 37(3), 2346-2359.
        PAFC polarization at elevated temperature.
    Li Q. et al. (2003). Chem. Mater. 15(26), 4896-4915.
        Phosphoric acid electrolyte conductivity.
"""

import numpy as np


class PAFCThermalModel:
    """
    Phosphoric acid fuel cell with explicit temperature dependence.
    Operating range: 423-483 K (150-210 C).
    """

    # Physical constants
    R = 8.314       # J/(mol K)
    F = 96485.0     # C/mol
    n = 2           # electrons per H2 molecule

    def __init__(self, params: dict):
        self.T_ref        = float(params["T_ref"])
        self.N_cells      = int(params["N_cells"])
        self.A_cell       = float(params["A_cell"])
        self.pH2          = float(params["pH2"])
        self.pO2          = float(params["pO2"])
        self.j_L          = float(params["j_L"])
        self.i0_ref       = float(params["i0_ref"])
        self.E_act        = float(params["E_act"])
        self.alpha        = float(params.get("alpha", 0.5))
        self.B_conc       = float(params.get("B_conc", 0.012))
        self.sigma_ref    = float(params["sigma_ref_H3PO4"])   # S/cm at T_ref
        self.E_act_sigma  = float(params["E_act_sigma"])        # J/mol, Arrhenius for sigma
        self.t_acid       = float(params["t_acid"])             # cm, acid layer thickness
        # Thermoneutral voltage at PAFC temperature (~200C) — HHV basis:
        # E_tn drops slightly with T: 1.481 - 0.000126*(T-298)
        # At 450K: ~1.481 - 0.000126*152 ≈ 1.462 V
        self.E_tn_ref     = float(params.get("E_tn_ref", 1.481))  # V at 298K
        self.k_tn         = float(params.get("k_tn", 0.000126))   # V/K, thermoneutral dT

    # ------------------------------------------------------------------
    # Thermoneutral voltage (T-dependent)
    # ------------------------------------------------------------------

    def thermoneutral_voltage(self, T):
        """
        Thermoneutral voltage [V] as function of temperature.
        At elevated T, water product is vapor: E_tn decreases with T.
        HHV-based: E_tn ~ 1.481 - 0.000126*(T - 298.15) [V]
        Valid approximate range: 298–600 K.
        """
        T = np.asarray(T, dtype=float)
        return self.E_tn_ref - self.k_tn * (T - 298.15)

    # ------------------------------------------------------------------
    # Nernst (reversible) voltage
    # ------------------------------------------------------------------

    def nernst_voltage(self, T):
        """
        Open-circuit (Nernst) voltage [V].
        Standard potential: E0(T) ~ 1.229 - 0.000846*(T - 298.15)
        Nernst correction: +RT/(2F)*ln(pH2 * sqrt(pO2))
        At 200C (473K): E0 ~ 1.229 - 0.000846*175 ~ 1.081 V
        """
        T = np.asarray(T, dtype=float)
        return (
            1.229
            - 0.000846 * (T - 298.15)
            + (self.R * T) / (2.0 * self.F) * np.log(self.pH2 * np.sqrt(self.pO2))
        )

    # ------------------------------------------------------------------
    # H3PO4 conductivity (Arrhenius-type, Razaq 1989)
    # ------------------------------------------------------------------

    def acid_conductivity(self, T):
        """
        Concentrated H3PO4 electrolyte conductivity [S/cm].

        Arrhenius-type fit based on Razaq et al. (1989) and Li et al. (2003):
            sigma(T) = sigma_ref * exp(E_act_sigma/R * (1/T_ref - 1/T))

        For 95-100% H3PO4:
            sigma_ref ~ 0.15 S/cm at T_ref = 450 K (177 C)
            E_act_sigma ~ 20000 J/mol (activation energy for ionic conduction)
        """
        T = np.asarray(T, dtype=float)
        return self.sigma_ref * np.exp(
            self.E_act_sigma / self.R * (1.0 / self.T_ref - 1.0 / T)
        )

    def acid_resistance(self, T):
        """Area-specific H3PO4 electrolyte resistance [ohm cm2]."""
        return self.t_acid / self.acid_conductivity(T)

    # ------------------------------------------------------------------
    # Exchange current density (Arrhenius — cathode limited)
    # ------------------------------------------------------------------

    def exchange_current_density(self, T):
        """
        Temperature-dependent cathode exchange current density [A/cm2].
        O2 reduction in H3PO4 has high activation energy (~60-80 kJ/mol).
        """
        T = np.asarray(T, dtype=float)
        return self.i0_ref * np.exp(
            -self.E_act / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    # ------------------------------------------------------------------
    # Activation loss (Butler-Volmer arcsinh form)
    # ------------------------------------------------------------------

    def activation_loss(self, j, T):
        """Activation overpotential [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        i0_T = self.exchange_current_density(T)
        j_safe = np.maximum(j, 1e-10)
        return (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(
            j_safe / (2.0 * i0_T)
        )

    # ------------------------------------------------------------------
    # Ohmic loss via H3PO4 conductivity
    # ------------------------------------------------------------------

    def ohmic_loss(self, j, T):
        """Ohmic voltage loss [V] using H3PO4 conductivity model."""
        j = np.asarray(j, dtype=float)
        return j * self.acid_resistance(T)

    # ------------------------------------------------------------------
    # Concentration loss
    # ------------------------------------------------------------------

    def concentration_loss(self, j):
        """Concentration (mass transport) loss [V]."""
        j = np.asarray(j, dtype=float)
        ratio = np.minimum(j / self.j_L, 0.9999)
        return np.where(j > 0, -self.B_conc * np.log(1.0 - ratio), 0.0)

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------

    def cell_voltage(self, j, T):
        """Net cell voltage [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E = self.nernst_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        return np.clip(E - V_act - V_ohm - V_conc, 0.0, None)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def power_density(self, j, T):
        """Power density [W/cm2]."""
        return np.asarray(j, dtype=float) * self.cell_voltage(j, T)

    def efficiency(self, j, T):
        """Voltage efficiency relative to HHV thermoneutral voltage [-]."""
        V = self.cell_voltage(j, T)
        E_tn = self.thermoneutral_voltage(T)
        return np.where(E_tn > 0, V / E_tn, 0.0)

    def heat_generation(self, j, T):
        """
        Heat generation per unit area [W/cm2].
        Q = j*(E_tn(T) - V_cell).  Always >= 0 for fuel cells.
        """
        j = np.asarray(j, dtype=float)
        E_tn = self.thermoneutral_voltage(T)
        V = self.cell_voltage(j, T)
        return j * np.maximum(E_tn - V, 0.0)

    def evaluate(self, j, T):
        """
        Full operating-point evaluation.

        Parameters
        ----------
        j : float or array -- current density [A/cm2]
        T : float or array -- temperature [K]

        Returns
        -------
        dict with all outputs
        """
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)

        E = self.nernst_voltage(T)
        E_tn = self.thermoneutral_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        V_cell = np.clip(E - V_act - V_ohm - V_conc, 0.0, None)
        P_density = j * V_cell
        eta = np.where(E_tn > 0, V_cell / E_tn, 0.0)
        Q = j * np.maximum(E_tn - V_cell, 0.0)
        R_acid = self.acid_resistance(T)
        sigma = self.acid_conductivity(T)

        return {
            "cell_voltage":          V_cell,
            "power_density":         P_density,
            "efficiency":            eta,
            "heat_generation":       Q,
            "acid_resistance":       R_acid,
            "acid_conductivity":     sigma,
            "thermoneutral_voltage": E_tn,
            "E_nernst":              E,
            "V_act":                 V_act,
            "V_ohm":                 V_ohm,
            "V_conc":                V_conc,
        }
