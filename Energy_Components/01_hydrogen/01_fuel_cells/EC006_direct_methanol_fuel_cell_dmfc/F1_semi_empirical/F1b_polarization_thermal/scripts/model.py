"""
EC006 -- Direct Methanol Fuel Cell (DMFC) -- F1b Polarization-Thermal Model
Temperature-dependent polarization with methanol crossover current correction.

Extends F1a by making all loss mechanisms explicitly temperature-dependent:

DMFC-specific physics:
  - Anode: CH3OH + H2O → CO2 + 6H+ + 6e-   (n=6 electrons per methanol)
  - Cathode: 3/2 O2 + 6H+ + 6e- → 3H2O

  - Nernst potential: E_rev(T) = E_ref + (dE/dT)*(T - T_ref)
                                  + RT/(6F)*ln(c_MeOH * pO2^(3/2))
    where E_ref ~ 1.21 V (standard) and dE/dT ~ -1.4e-4 V/K for methanol

  - Methanol crossover current density (Nafion, T-dependent):
        j_cross(T) = j_cross_ref * exp(E_act_cross/R * (1/T_ref - 1/T))
    Crossover increases strongly with T (diffusion + electro-osmosis).
    Effective current: j_eff = j + j_cross (for activation loss only)
    Mixed potential loss: V_mix ~ RT/(alpha*6*F)*ln(j_cross/(j + j_cross))
                          (approx.; Kulikovsky 2002 mixed potential model)

  - Nafion membrane conductivity (Springer 1991):
        sigma(T, lambda) = (0.005139*lambda - 0.00326)*exp(1268*(1/303 - 1/T))
    (Same as PEMFC but DMFC runs cooler; methanol-laden membrane)

  - Activation loss (anode-limited in DMFC):
        V_act = RT/(alpha*n*F)*arcsinh(j_eff / (2*i0(T)))
    where n=6, alpha typically 0.5

  - Heat generation: Q = j*(E_tn_MeOH - V_cell)
    E_tn_MeOH = 1.21 V (thermoneutral for methanol HHV, per 6e-)

References:
    Scott K. et al. (1999). J. Power Sources, 79(1), 43-59.
    Kulikovsky A.A. (2002). Fuel Cells, 2(2), 94-106. (Methanol crossover model)
    Nordlund J. & Lindbergh G. (2002). J. Electrochem. Soc., 149(9), A1107-A1113.
    Larminie J. & Dicks A. (2003). Fuel Cell Systems Explained, 2nd Ed., Wiley.
"""

import numpy as np


class DMFCThermalModel:
    """
    Direct methanol fuel cell with explicit temperature dependence.
    Operating range: 333-383 K (60-110 C).
    """

    # Physical constants
    R = 8.314       # J/(mol K)
    F = 96485.0     # C/mol
    n = 6           # electrons per CH3OH molecule
    E_tn = 1.21     # thermoneutral voltage (methanol HHV basis) [V]
    # MeOH std: HHV = 726 kJ/mol => E_tn = 726000/(6*96485) ~ 1.253 V
    # At ~70C practical E_tn ~ 1.21 V (per Larminie)

    def __init__(self, params: dict):
        self.T_ref          = float(params["T_ref"])
        self.N_cells        = int(params["N_cells"])
        self.A_cell         = float(params["A_cell"])
        self.c_MeOH         = float(params["c_MeOH"])          # mol/L methanol conc
        self.pO2            = float(params["pO2"])
        self.j_L            = float(params["j_L"])
        self.i0_ref         = float(params["i0_ref"])           # anode exchange current
        self.E_act          = float(params["E_act"])
        self.alpha          = float(params.get("alpha", 0.5))
        self.B_conc         = float(params.get("B_conc", 0.020))
        self.j_cross_ref    = float(params["j_cross_ref"])      # A/cm2 crossover at T_ref
        self.E_act_cross    = float(params["E_act_cross"])      # J/mol crossover activation
        self.lambda_mem     = float(params.get("lambda_mem", 12.0))
        self.t_mem          = float(params["membrane_thickness"])  # cm
        # Standard potential temperature coefficient
        self.dEdT           = float(params.get("dEdT_MeOH", -1.4e-4))  # V/K
        self.E_ref          = float(params.get("E_ref_std", 1.214))     # V at 298K

    # ------------------------------------------------------------------
    # Nernst (reversible) voltage
    # ------------------------------------------------------------------

    def nernst_voltage(self, T):
        """
        Open-circuit voltage [V] for methanol oxidation.

        E(T) = E_ref + dE/dT*(T - 298) + RT/(6F)*ln(c_MeOH * pO2^1.5)

        Note: dE/dT < 0 for methanol oxidation (thermodynamic).
        Standard potential at 298K: 1.214 V (Gibbs / 6F).
        """
        T = np.asarray(T, dtype=float)
        E0_T = self.E_ref + self.dEdT * (T - 298.15)
        # Nernst: activities: c_MeOH for methanol (assume activity ~ c/c_std), pO2 atm
        # simplified: ln(c_MeOH * pO2^1.5)
        log_term = np.log(np.maximum(self.c_MeOH, 1e-6) * self.pO2 ** 1.5)
        return E0_T + (self.R * T) / (self.n * self.F) * log_term

    # ------------------------------------------------------------------
    # Methanol crossover current (T-dependent, Arrhenius diffusion)
    # ------------------------------------------------------------------

    def crossover_current(self, T):
        """
        Methanol crossover current density equivalent [A/cm2].

        Crossover flux increases with T (diffusion coefficient Arrhenius).
        j_cross(T) = j_cross_ref * exp(E_act_cross/R * (1/T_ref - 1/T))

        Physical range: ~0.02-0.15 A/cm2 for Nafion DMFC at 60-100C.
        """
        T = np.asarray(T, dtype=float)
        return self.j_cross_ref * np.exp(
            self.E_act_cross / self.R * (1.0 / self.T_ref - 1.0 / T)
        )

    # ------------------------------------------------------------------
    # Exchange current density (anode limited, Arrhenius)
    # ------------------------------------------------------------------

    def exchange_current_density(self, T):
        """Anode exchange current density [A/cm2] (MeOH oxidation rate-limiting)."""
        T = np.asarray(T, dtype=float)
        return self.i0_ref * np.exp(
            -self.E_act / self.R * (1.0 / T - 1.0 / self.T_ref)
        )

    # ------------------------------------------------------------------
    # Membrane conductivity (Springer model)
    # ------------------------------------------------------------------

    def membrane_conductivity(self, T):
        """Nafion conductivity [S/cm] — same model as PEMFC."""
        T = np.asarray(T, dtype=float)
        sigma_303 = 0.005139 * self.lambda_mem - 0.00326
        return np.maximum(sigma_303 * np.exp(1268.0 * (1.0 / 303.0 - 1.0 / T)), 1e-3)

    def membrane_resistance(self, T):
        """Area-specific membrane resistance [ohm cm2]."""
        return self.t_mem / self.membrane_conductivity(T)

    # ------------------------------------------------------------------
    # Mixed potential / crossover loss
    # ------------------------------------------------------------------

    def mixed_potential_loss(self, j, T):
        """
        Approximate mixed-potential voltage penalty due to methanol crossover [V].

        At open circuit, crossover oxidation raises cathode surface potential,
        depressing cell voltage below E_nernst.  Under load, the mixed potential
        effect is partially quenched by j.

        Kulikovsky (2002) approximate form:
            V_mix = RT/(alpha*n*F) * ln((j + j_cross) / j_cross) ... (penalty)
        This is the REDUCTION in cathode potential due to local MeOH oxidation.
        Sign convention: positive loss (subtracted from cell voltage).
        """
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        j_x = self.crossover_current(T)
        j_safe = np.maximum(j, 1e-10)
        # Mixed potential reduces cell voltage; effect decreases as j >> j_cross
        return (self.R * T) / (self.alpha * self.n * self.F) * np.log(
            (j_safe + j_x) / j_x
        )

    # ------------------------------------------------------------------
    # Activation loss (anode-limited)
    # ------------------------------------------------------------------

    def activation_loss(self, j, T):
        """Activation overpotential [V] — anode dominated."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        i0_T = self.exchange_current_density(T)
        j_x = self.crossover_current(T)
        # Effective current for activation includes crossover
        j_eff = np.maximum(j + j_x, 1e-10)
        return (self.R * T) / (self.alpha * self.n * self.F) * np.arcsinh(
            j_eff / (2.0 * i0_T)
        )

    # ------------------------------------------------------------------
    # Ohmic loss
    # ------------------------------------------------------------------

    def ohmic_loss(self, j, T):
        """Ohmic voltage loss [V]."""
        j = np.asarray(j, dtype=float)
        return j * self.membrane_resistance(T)

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
        V_mix = self.mixed_potential_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        return np.clip(E - V_act - V_mix - V_ohm - V_conc, 0.0, None)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def power_density(self, j, T):
        """Power density [W/cm2]."""
        return np.asarray(j, dtype=float) * self.cell_voltage(j, T)

    def efficiency(self, j, T):
        """Voltage efficiency relative to E_tn [-]."""
        V = self.cell_voltage(j, T)
        return V / self.E_tn

    def heat_generation(self, j, T):
        """Heat generation [W/cm2]. Q = j*(E_tn - V_cell)."""
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
        V_mix = self.mixed_potential_loss(j, T)
        V_ohm = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        V_cell = np.clip(E - V_act - V_mix - V_ohm - V_conc, 0.0, None)
        P_density = j * V_cell
        eta = V_cell / self.E_tn
        Q = j * np.maximum(self.E_tn - V_cell, 0.0)
        R_mem = self.membrane_resistance(T)
        j_x = self.crossover_current(T)

        return {
            "cell_voltage":       V_cell,
            "power_density":      P_density,
            "efficiency":         eta,
            "heat_generation":    Q,
            "membrane_resistance": R_mem,
            "crossover_current":  j_x,
            "E_nernst":           E,
            "V_act":              V_act,
            "V_mix":              V_mix,
            "V_ohm":              V_ohm,
            "V_conc":             V_conc,
        }
