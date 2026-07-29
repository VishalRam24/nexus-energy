"""
EC001 — PEM Fuel Cell (PEMFC) — F1a Polarization Curve
Physics equations class.

Model: Amphlett et al. (1995) polarization curve
Source: Amphlett et al. (1995), J. Electrochem. Soc., 142(1), 1-8
        Also: OPEM library (MIT License), https://github.com/ECSIM/opem
"""

import numpy as np


class PEMFuelCellModel:
    """
    Semi-empirical single-cell polarization curve for a PEM fuel cell.

    Governing equations (Amphlett model)
    -------------------------------------
    E_Nernst = 1.229 - 0.00085*(T - 298.15)
               + R*T/(2*F) * ln(pH2 * sqrt(pO2))        [V]

    cO2      = pO2 / (5.08e6 * exp(-498/T))             [mol/cm³]
    V_act    = xi1 + xi2*T + xi3*T*ln(cO2) + xi4*T*ln(j) [V]
    sigma_mem = (0.005139*lambda_mem - 0.00326)
                * exp(1268*(1/303 - 1/T))               [S/cm]
    V_ohm    = j * t_mem / sigma_mem                    [V]
    V_conc   = -B * ln(1 - j/j_L)                       [V]

    V_cell   = E_Nernst - V_act - V_ohm - V_conc        [V]
    V_stack  = N_cells * V_cell                          [V]
    """

    # Physical constants
    R = 8.314       # J/(mol·K)
    F = 96485.0     # C/mol
    n = 2           # electrons per H2O molecule

    # Amphlett semi-empirical coefficients (fitted to Ballard Mark IV data)
    xi1 = -0.948
    xi2 =  0.00286
    xi3 = 7.6e-5
    xi4 = -1.93e-4

    # Concentration loss coefficient
    B = 0.016       # V

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict
            T           : temperature [K]
            N_cells     : number of cells in stack
            electrode_area : active electrode area [cm²]
            pH2         : hydrogen partial pressure [atm]
            pO2         : oxygen partial pressure [atm]
            j_L         : limiting current density [A/cm²]
            t_mem       : membrane thickness [cm]
            lambda_mem  : membrane water content [-] (default 14)
            P_rated     : rated power for stack [W]
        """
        self.T            = float(params["T"])
        self.N_cells      = int(params["N_cells"])
        self.electrode_area = float(params["electrode_area"])
        self.pH2          = float(params["pH2"])
        self.pO2          = float(params["pO2"])
        self.j_L          = float(params["j_L"])
        self.t_mem        = float(params["t_mem"])
        self.lambda_mem   = float(params.get("lambda_mem", 14.0))
        self.P_rated      = float(params.get("P_rated", 1000.0))

    # ------------------------------------------------------------------
    # Nernst voltage
    # ------------------------------------------------------------------

    def nernst_voltage(self, T: float = None) -> float:
        """Open-circuit (Nernst) voltage [V]."""
        T = T if T is not None else self.T
        return (
            1.229
            - 0.00085 * (T - 298.15)
            + (self.R * T) / (2.0 * self.F) * np.log(self.pH2 * np.sqrt(self.pO2))
        )

    # ------------------------------------------------------------------
    # Activation loss
    # ------------------------------------------------------------------

    def oxygen_concentration(self, T: float = None) -> float:
        """O2 dissolved concentration at cathode [mol/cm³]."""
        T = T if T is not None else self.T
        return self.pO2 / (5.08e6 * np.exp(-498.0 / T))

    def activation_loss(self, j: float, T: float = None) -> float:
        """Activation overpotential [V] — Amphlett empirical fit."""
        T = T if T is not None else self.T
        if j <= 0:
            # At zero current activation loss is taken as the minimum value
            # evaluated at a tiny current to avoid log singularity
            j = 1e-6
        cO2 = self.oxygen_concentration(T)
        return -(self.xi1 + self.xi2 * T + self.xi3 * T * np.log(cO2) + self.xi4 * T * np.log(j))

    # ------------------------------------------------------------------
    # Ohmic loss
    # ------------------------------------------------------------------

    def membrane_conductivity(self, T: float = None) -> float:
        """Nafion membrane ionic conductivity [S/cm] — Springer model."""
        T = T if T is not None else self.T
        lam = self.lambda_mem
        return (0.005139 * lam - 0.00326) * np.exp(1268.0 * (1.0 / 303.15 - 1.0 / T))

    def ohmic_loss(self, j: float, T: float = None) -> float:
        """Ohmic voltage loss [V]."""
        T = T if T is not None else self.T
        sigma = self.membrane_conductivity(T)
        return j * self.t_mem / sigma

    # ------------------------------------------------------------------
    # Concentration loss
    # ------------------------------------------------------------------

    def concentration_loss(self, j: float) -> float:
        """Mass-transport (concentration) voltage loss [V]."""
        if j <= 0:
            return 0.0
        ratio = j / self.j_L
        if ratio >= 1.0:
            return float("inf")
        return -self.B * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell and stack voltage
    # ------------------------------------------------------------------

    def cell_voltage(self, j: float, T: float = None) -> float:
        """Net cell voltage [V]."""
        T = T if T is not None else self.T
        E = self.nernst_voltage(T)
        V_act  = self.activation_loss(j, T)
        V_ohm  = self.ohmic_loss(j, T)
        V_conc = self.concentration_loss(j)
        return E - V_act - V_ohm - V_conc

    def stack_voltage(self, j: float, T: float = None) -> float:
        """Stack voltage [V]."""
        return self.N_cells * self.cell_voltage(j, T)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def power_density(self, j: float, T: float = None) -> float:
        """Power density [W/cm²]."""
        return j * self.cell_voltage(j, T)

    def stack_power(self, j: float, T: float = None) -> float:
        """Stack electrical output power [W]."""
        return self.N_cells * self.power_density(j, T) * self.electrode_area

    def efficiency(self, j: float, T: float = None) -> float:
        """
        Voltage efficiency relative to HHV (1.481 V per cell) [-].
        eta = V_cell / 1.481
        """
        V = self.cell_voltage(j, T)
        return max(0.0, V / 1.481)

    # ------------------------------------------------------------------
    # Full operating-point evaluation
    # ------------------------------------------------------------------

    def evaluate(self, j: float, T_celsius: float = None) -> dict:
        """
        Return all outputs for a given current density and temperature.

        Parameters
        ----------
        j         : current density [A/cm²]
        T_celsius : temperature [°C]; uses default T if None
        """
        if j < 0:
            raise ValueError(f"Current density must be >= 0, got {j}")
        if j >= self.j_L:
            raise ValueError(
                f"Current density {j} >= limiting current density {self.j_L}"
            )

        T_K = (T_celsius + 273.15) if T_celsius is not None else self.T

        E    = self.nernst_voltage(T_K)
        V_act  = self.activation_loss(j, T_K)
        V_ohm  = self.ohmic_loss(j, T_K)
        V_conc = self.concentration_loss(j)
        V_cell = E - V_act - V_ohm - V_conc
        V_stack = self.N_cells * V_cell
        P_density = j * V_cell
        P_stack = self.N_cells * P_density * self.electrode_area
        eta = self.efficiency(j, T_K)

        return {
            "j_A_cm2":          j,
            "T_K":              T_K,
            "E_Nernst_V":       E,
            "V_act_V":          V_act,
            "V_ohm_V":          V_ohm,
            "V_conc_V":         V_conc,
            "cell_voltage_V":   V_cell,
            "stack_voltage_V":  V_stack,
            "power_density_W_cm2": P_density,
            "stack_power_W":    P_stack,
            "efficiency":       eta,
        }
