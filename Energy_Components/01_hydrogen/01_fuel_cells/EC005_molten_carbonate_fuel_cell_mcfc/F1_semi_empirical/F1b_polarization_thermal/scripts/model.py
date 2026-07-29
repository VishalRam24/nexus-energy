"""
EC005 -- Molten Carbonate Fuel Cell (MCFC) -- F1b Polarization-Thermal Model
Temperature-dependent polarization with molten carbonate conductivity.

Extends F1a by making all loss mechanisms explicitly temperature-dependent:

MCFC specific:
  - Operates at 600-700 C (873-973 K) with molten Li₂CO₃/K₂CO₃ electrolyte
  - CO2 must be supplied to the cathode (carbonate cycling):
        Cathode: O2 + 2CO2 + 4e- → 2CO3(2-)
        Anode:   H2 + CO3(2-) → H2O + CO2 + 2e-
  - Nernst potential includes CO2 partial pressures:
        E_rev(T) = E0(T) + RT/(2F)*ln(pH2 * pCO2_cat * sqrt(pO2) / (pH2O * pCO2_an))
    where the +RT/(2F)*ln(...) comes from the carbonate half-reaction.

  - Molten carbonate conductivity (Uchida et al. 1983 / Selman 1988):
        sigma_mc(T) = A_mc * exp(-E_act_mc / (R*T))
    Li₂CO₃/K₂CO₃ eutectic (62/38 mol%):
        A_mc ~ 2.0e5 S/cm, E_act_mc ~ 28 kJ/mol

  - Ohmic loss: V_ohm = j * t_mc / sigma_mc(T)

  - Exchange current densities (Arrhenius):
        i0_a(T) = i0_a_ref * exp(-E_act_a/R*(1/T - 1/T_ref))
        i0_c(T) = i0_c_ref * exp(-E_act_c/R*(1/T - 1/T_ref))

  - Activation loss: combined anode + cathode
        V_act = RT/(alpha*n*F) * [arcsinh(j/(2*i0_a)) + arcsinh(j/(2*i0_c))]

  - Thermoneutral voltage: E_tn(T) ~ 1.21 V at 650C (lower than PEMFC due to
    higher T and CO2 cycle; based on HHV of H2 - CO2 latent heat)

  - Heat generation: Q = j*(E_tn - V_cell)

References:
    Uchida I. et al. (1983). Electrochim. Acta, 28(10), 1423-1431.
        Molten carbonate conductivity.
    Selman J.R. (1988). In: Fuel Cells: Trends in R&D, Plenum Press.
    Lu S.T. & Selman J.R. (1984). J. Electrochem. Soc., 131(12), 2827-2833.
    Yuh C. & Selman J.R. (1991). J. Electrochem. Soc., 138(12), 3649-3655.
        MCFC polarization curve analysis.
"""

import numpy as np


class MCFCThermalModel:
    """
    Molten carbonate fuel cell with explicit temperature dependence.
    Operating range: 873-973 K (600-700 C).
    """

    # Physical constants
    R = 8.314       # J/(mol K)
    F = 96485.0     # C/mol
    n = 2           # electrons per H2 molecule

    def __init__(self, params: dict):
        self.T_ref       = float(params["T_ref"])
        self.N_cells     = int(params["N_cells"])
        self.A_cell      = float(params["A_cell"])
        self.pH2         = float(params["pH2"])
        self.pO2         = float(params["pO2"])
        self.pH2O        = float(params["pH2O"])
        self.pCO2_cat    = float(params["pCO2_cathode"])   # CO2 at cathode (atm)
        self.pCO2_an     = float(params["pCO2_anode"])     # CO2 at anode (atm)
        self.j_L         = float(params["j_L"])
        self.i0_a_ref    = float(params["i0_anode_ref"])
        self.E_act_a     = float(params["E_act_anode"])
        self.i0_c_ref    = float(params["i0_cathode_ref"])
        self.E_act_c     = float(params["E_act_cathode"])
        self.alpha       = float(params.get("alpha", 0.5))
        self.B_conc      = float(params.get("B_conc", 0.008))
        self.A_mc        = float(params["A_mc"])            # S/cm, Arrhenius pre-factor
        self.E_act_mc    = float(params["E_act_mc"])        # J/mol, carbonate Eact
        self.t_mc        = float(params["t_mc"])            # cm, electrolyte thickness
        self.E_tn        = float(params.get("E_tn", 1.21)) # V thermoneutral at ~650C

    # ------------------------------------------------------------------
    # Molten carbonate conductivity (Uchida 1983)
    # ------------------------------------------------------------------

    def carbonate_conductivity(self, T):
        """
        Molten Li₂CO₃/K₂CO₃ eutectic conductivity [S/cm].

        Arrhenius model from Uchida et al. (1983):
            sigma(T) = A_mc * exp(-E_act_mc / (R*T))

        Typical values at 923 K (650C): ~2.5-3.5 S/cm.
        """
        T = np.asarray(T, dtype=float)
        return self.A_mc * np.exp(-self.E_act_mc / (self.R * T))

    def carbonate_resistance(self, T):
        """Area-specific molten carbonate resistance [ohm cm2]."""
        sigma = self.carbonate_conductivity(T)
        return self.t_mc / sigma

    # ------------------------------------------------------------------
    # Nernst (reversible) voltage with CO2 correction
    # ------------------------------------------------------------------

    def nernst_voltage(self, T):
        """
        Open-circuit (Nernst) voltage [V] for MCFC.

        Includes CO2 partial pressures at anode and cathode.
        Reaction: H2 + 1/2*O2 + CO2(cat) -> H2O + CO2(an)
        E = E0(T) + RT/(2F)*ln(pH2 * pCO2_cat * sqrt(pO2) / (pH2O * pCO2_an))

        Standard potential E0(T) ~ 1.05 - 0.000288*(T - 873) V (Lu & Selman 1984)
        """
        T = np.asarray(T, dtype=float)
        E0_T = 1.05 - 0.000288 * (T - 873.15)
        nernst_log = np.log(
            self.pH2 * self.pCO2_cat * np.sqrt(self.pO2)
            / (self.pH2O * self.pCO2_an)
        )
        return E0_T + (self.R * T) / (2.0 * self.F) * nernst_log

    # ------------------------------------------------------------------
    # Exchange current densities (Arrhenius)
    # ------------------------------------------------------------------

    def i0_anode(self, T):
        """Anode exchange current density [A/cm2]."""
        T = np.asarray(T, dtype=float)
        return self.i0_a_ref * np.exp(
            -self.E_act_a / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    def i0_cathode(self, T):
        """Cathode exchange current density [A/cm2]."""
        T = np.asarray(T, dtype=float)
        return self.i0_c_ref * np.exp(
            -self.E_act_c / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def activation_loss(self, j, T):
        """Combined anode + cathode activation overpotential [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_safe = np.maximum(j, 1e-10)
        i0_a = self.i0_anode(T)
        i0_c = self.i0_cathode(T)
        V_a = (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(j_safe / (2.0 * i0_a))
        V_c = (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(j_safe / (2.0 * i0_c))
        return V_a + V_c

    def ohmic_loss(self, j, T):
        """Ohmic voltage loss [V] via molten carbonate resistance."""
        j = np.asarray(j, dtype=float)
        return j * self.carbonate_resistance(T)

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
        """Voltage efficiency relative to E_tn [-]."""
        V = self.cell_voltage(j, T)
        return np.where(self.E_tn > 0, V / self.E_tn, 0.0)

    def heat_generation(self, j, T):
        """Heat generation per unit area [W/cm2]. Q = j*(E_tn - V_cell)."""
        j = np.asarray(j, dtype=float)
        V = self.cell_voltage(j, T)
        return j * np.maximum(self.E_tn - V, 0.0)

    def evaluate(self, j, T):
        """
        Full operating-point evaluation.

        Parameters
        ----------
        j : float or array -- current density [A/cm2]
        T : float or array -- temperature [K]

        Returns
        -------
        dict
        """
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)

        E = self.nernst_voltage(T)
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        V_cell = np.clip(E - V_act - V_ohm - V_conc, 0.0, None)
        P_density = j * V_cell
        eta = np.where(self.E_tn > 0, V_cell / self.E_tn, 0.0)
        Q = j * np.maximum(self.E_tn - V_cell, 0.0)
        R_mc = self.carbonate_resistance(T)
        sigma_mc = self.carbonate_conductivity(T)

        return {
            "cell_voltage":          V_cell,
            "power_density":         P_density,
            "efficiency":            eta,
            "heat_generation":       Q,
            "carbonate_resistance":  R_mc,
            "carbonate_conductivity": sigma_mc,
            "E_nernst":              E,
            "V_act":                 V_act,
            "V_ohm":                 V_ohm,
            "V_conc":                V_conc,
            "i0_anode":              self.i0_anode(T),
            "i0_cathode":            self.i0_cathode(T),
        }
