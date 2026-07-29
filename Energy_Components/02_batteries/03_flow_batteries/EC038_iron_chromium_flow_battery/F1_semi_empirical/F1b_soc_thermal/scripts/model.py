"""
EC038 -- Iron-Chromium Flow Battery (ICFB) -- F1b SOC-Thermal Model

Extends F1a (Nernst + ohmic) by adding temperature dependence:
    E_Nernst(SOC, T) = E0(T) + 2*(R_gas*T)/(n*F) * ln(SOC/(1-SOC))
    E0(T) = E0_ref + dOCV_dT * (T - T_ref)    [temperature correction of standard potential]
    R_cell(T) = R_cell_ref * exp(E_a/R_gas * (1/T - 1/T_ref))   [Arrhenius]
    V_cell = E_Nernst - I * R_cell(T)
    V_stack = N_cells * V_cell
    Q_gen = I^2 * R_cell(T) * N_cells + I * N_cells * T * dOCV_dT   [W, per stack]
    P_pump = k_pump * I^2   [W, parasitic pump loss proportional to flow rate ~ current]

ICFB chemistry:
    Positive: Fe3+ + e- <-> Fe2+   E0 = +0.77 V vs SHE
    Negative: Cr3+ + e- <-> Cr2+   E0 = -0.41 V vs SHE
    Cell: E0 = 1.18 V at standard conditions

Temperature range: 15-65 degC. ICFB tolerates higher temperature than Zn-Br2;
Cr kinetics sluggish below 15 degC.

References:
    Hruska, L. W., Savinell, R. F. (1981). J. Electrochem. Soc. 128, 18-25.
    Shibata, S., Sumino, M. P. (1988). J. Power Sources 25, 177-184.
    Zeng, Y. K. et al. (2015). J. Power Sources 278, 294-302.
"""

import numpy as np

R_GAS = 8.314
F_CONST = 96485.0


class IronChromiumFlowF1b:
    """Iron-chromium flow battery stack -- voltage as f(SOC, current, temperature)."""

    SOC_MIN = 0.01
    SOC_MAX = 0.99

    def __init__(self, params: dict):
        u = params["unit"]
        therm = params["thermal"]

        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2 = u["electrode_area_cm2"]["value"]
        self.E0_ref = u["E0"]["value"]
        self.R_ohm_cm2_ref = u["R_cell_ohm_cm2_ref"]["value"]
        self.n = int(u["n"]["value"])
        self.R_cell_ref = self.R_ohm_cm2_ref / self.A_cm2   # Ohm per cell at T_ref
        self.k_pump = u["pump_loss_coefficient"]["value"]    # W/A^2

        self.T_ref = therm["T_ref"]["value"]
        self.E_a = therm["E_a"]["value"]
        self.dOCV_dT = therm["dOCV_dT"]["value"]            # V/K per cell

    def e0_thermal(self, temperature):
        """Temperature-corrected standard potential (per cell)."""
        temperature = np.asarray(temperature, dtype=float)
        return self.E0_ref + self.dOCV_dT * (temperature - self.T_ref)

    def r_cell(self, temperature):
        """Temperature-dependent cell resistance via Arrhenius."""
        temperature = np.asarray(temperature, dtype=float)
        return self.R_cell_ref * np.exp(
            self.E_a / R_GAS * (1.0 / temperature - 1.0 / self.T_ref)
        )

    def e_nernst(self, soc, temperature):
        """Nernst cell potential at given SOC and temperature."""
        soc = np.asarray(soc, dtype=float)
        if np.any(soc < 0.0) or np.any(soc > 1.0):
            raise ValueError("SOC must be in [0, 1].")
        soc = np.clip(soc, self.SOC_MIN, self.SOC_MAX)
        temperature = np.asarray(temperature, dtype=float)
        thermal_factor = R_GAS * temperature / (self.n * F_CONST)
        return self.e0_thermal(temperature) + 2.0 * thermal_factor * np.log(soc / (1.0 - soc))

    def cell_voltage(self, soc, current, temperature):
        """Single cell terminal voltage."""
        current = np.asarray(current, dtype=float)
        return self.e_nernst(soc, temperature) - current * self.r_cell(temperature)

    def stack_voltage(self, soc, current, temperature):
        """Stack terminal voltage = N_cells * V_cell."""
        return self.N_cells * self.cell_voltage(soc, current, temperature)

    def pump_loss(self, current):
        """
        Parasitic pump power [W].
        Flow rate is proportional to current density; pump power ~ flow^2.
        P_pump = k_pump * I^2
        """
        current = np.asarray(current, dtype=float)
        return self.k_pump * current**2

    def heat_generation(self, soc, current, temperature):
        """
        Stack heat generation rate [W].
        Q = I^2 * R_stack(T) + I * N_cells * T * dOCV_dT
        where R_stack = N_cells * R_cell.
        Reversible entropic heat included; pump losses dissipated externally.
        """
        current = np.asarray(current, dtype=float)
        temperature = np.asarray(temperature, dtype=float)
        R_stack = self.N_cells * self.r_cell(temperature)
        q_joule = current**2 * R_stack
        q_reversible = current * self.N_cells * temperature * self.dOCV_dT
        return q_joule + q_reversible

    def power_w(self, soc, current, temperature):
        """Net stack power [W] = electrical - pump losses. Positive = discharge."""
        v = self.stack_voltage(soc, current, temperature)
        current = np.asarray(current, dtype=float)
        p_elec = v * current
        return p_elec - self.pump_loss(current)

    def efficiency(self, soc, current, temperature):
        """Voltage efficiency: V_discharge / V_charge, clipped to [0, 1]."""
        current = np.asarray(current, dtype=float)
        V_dis = self.cell_voltage(soc, np.abs(current), temperature)
        V_chg = self.cell_voltage(soc, -np.abs(current), temperature)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(V_chg > 0, V_dis / V_chg, 0.0)
        return np.clip(eta, 0.0, 1.0)
