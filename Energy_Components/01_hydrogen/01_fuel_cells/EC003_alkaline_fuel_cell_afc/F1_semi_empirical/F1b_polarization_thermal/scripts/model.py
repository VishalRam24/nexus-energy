"""
EC003 -- Alkaline Fuel Cell (AFC) -- F1b Polarization-Thermal Model
Temperature-dependent polarization curve with KOH electrolyte conductivity.

Extends F1a by making all loss mechanisms explicitly temperature-dependent:
  - Nernst potential: E_rev(T) = 1.229 - 0.000846*(T - 298.15)
                                  + RT/(2F)*ln(pH2 * sqrt(pO2))
  - KOH electrolyte conductivity (liquid): sigma_KOH(T, c) from Gilliam (2007)
      sigma = a1*c^2 + a2*T + a3*c*T + a4*c + a5*T^2 + a6
  - Ohmic loss: V_ohm = j * L_elec / sigma_KOH(T)
  - Exchange current density (Arrhenius):
      i0(T) = i0_ref * exp(-E_act/R * (1/T - 1/T_ref))
  - Activation loss: V_act = RT/(alpha*n*F) * arcsinh(j / (2*i0(T)))
  - Concentration loss: V_conc = -B*ln(1 - j/j_L)
  - Heat generation: Q = j*(E_tn - V_cell)  [E_tn = 1.481 V, HHV]

AFC-specific physics:
  - KOH electrolyte conductivity is much higher than Nafion and strongly T-dependent
  - AFC operates at 60-90 C (333-363 K) under alkaline conditions
  - CO2 poisoning not modelled here (F1b concentrates on thermal effects)

References:
    Gilliam et al. (2007) Int. J. Hydrogen Energy, 32(3), 359-364.
        KOH conductivity regression: sigma = f(T, c_KOH)
    Appleby & Foulkes (1989) Fuel Cell Handbook, Van Nostrand Reinhold.
    Larminie & Dicks (2003) Fuel Cell Systems Explained, 2nd Ed., Wiley.
"""

import numpy as np


class AFCThermalModel:
    """
    Alkaline fuel cell with explicit temperature dependence.
    Operating range: 333-363 K (60-90 C), KOH electrolyte.
    """

    # Physical constants
    R = 8.314       # J/(mol K)
    F = 96485.0     # C/mol
    n = 2           # electrons per H2 molecule
    E_tn = 1.481    # thermoneutral voltage (HHV basis) [V]

    def __init__(self, params: dict):
        self.T_ref       = float(params["T_ref"])
        self.N_cells     = int(params["N_cells"])
        self.A_cell      = float(params["A_cell"])
        self.pH2         = float(params["pH2"])
        self.pO2         = float(params["pO2"])
        self.j_L         = float(params["j_L"])
        self.i0_ref      = float(params["i0_ref"])
        self.E_act       = float(params["E_act"])
        self.alpha       = float(params.get("alpha", 0.5))
        self.B_conc      = float(params.get("B_conc", 0.010))
        self.c_KOH       = float(params.get("c_KOH", 6.0))     # mol/L KOH concentration
        self.L_elec      = float(params["L_electrolyte"])       # cm  electrode gap

    # ------------------------------------------------------------------
    # KOH electrolyte conductivity (Gilliam 2007 regression)
    # ------------------------------------------------------------------

    def koh_conductivity(self, T):
        """
        KOH electrolyte conductivity [S/cm].

        Gilliam et al. (2007) empirical fit for aqueous KOH:
            sigma [S/m] = -2.041*c^2 - 0.0028*T*c^2 + 0.005332*T*c
                          + 207.2*c/T + 0.001043*T^2 - 0.0000003*T^3
                          - 0.04755*c^3 - 0.1316
        c in mol/L, T in K.  Result here converted to S/cm (divide by 100).

        Valid range: 2–18 mol/L KOH, 273–373 K.
        """
        T = np.asarray(T, dtype=float)
        c = self.c_KOH
        sigma_Sm = (
            -2.041 * c**2
            - 0.0028 * T * c**2
            + 0.005332 * T * c
            + 207.2 * c / T
            + 0.001043 * T**2
            - 0.0000003 * T**3
            - 0.04755 * c**3
            - 0.1316
        )
        return np.maximum(sigma_Sm / 100.0, 0.01)   # S/cm; floor at 0.01

    def electrolyte_resistance(self, T):
        """Area-specific electrolyte resistance [ohm cm2]."""
        sigma = self.koh_conductivity(T)
        return self.L_elec / sigma

    # ------------------------------------------------------------------
    # Nernst (reversible) voltage
    # ------------------------------------------------------------------

    def nernst_voltage(self, T):
        """Open-circuit (Nernst) voltage [V]."""
        T = np.asarray(T, dtype=float)
        return (
            1.229
            - 0.000846 * (T - 298.15)
            + (self.R * T) / (2.0 * self.F) * np.log(self.pH2 * np.sqrt(self.pO2))
        )

    # ------------------------------------------------------------------
    # Exchange current density (Arrhenius)
    # ------------------------------------------------------------------

    def exchange_current_density(self, T):
        """Temperature-dependent exchange current density [A/cm2]."""
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
    # Ohmic loss via KOH conductivity
    # ------------------------------------------------------------------

    def ohmic_loss(self, j, T):
        """Ohmic voltage loss [V] using KOH conductivity model."""
        j = np.asarray(j, dtype=float)
        return j * self.electrolyte_resistance(T)

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
        """Voltage efficiency relative to HHV (1.481 V) [-]."""
        return self.cell_voltage(j, T) / self.E_tn

    def heat_generation(self, j, T):
        """
        Heat generation per unit area [W/cm2].
        Q = j*(E_tn - V_cell).  Always >= 0 for fuel cells.
        """
        j = np.asarray(j, dtype=float)
        return j * (self.E_tn - self.cell_voltage(j, T))

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
        V_act = self.activation_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        V_cell = np.clip(E - V_act - V_ohm - V_conc, 0.0, None)
        P_density = j * V_cell
        eta = V_cell / self.E_tn
        Q = j * (self.E_tn - V_cell)
        R_elec = self.electrolyte_resistance(T)
        sigma = self.koh_conductivity(T)

        return {
            "cell_voltage":           V_cell,
            "power_density":          P_density,
            "efficiency":             eta,
            "heat_generation":        Q,
            "electrolyte_resistance": R_elec,
            "koh_conductivity":       sigma,
            "E_nernst":               E,
            "V_act":                  V_act,
            "V_ohm":                  V_ohm,
            "V_conc":                 V_conc,
        }
