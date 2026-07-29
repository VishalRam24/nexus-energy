"""
EC036 — Vanadium Redox Flow Battery (VRFB) — F1b SOC + Crossover Model

Extends F1a Nernst+Ohmic with vanadium crossover through the membrane.

Crossover flux (Fick's law through membrane):
    J_cross = D_v * (c_pos - c_neg) / d_membrane   [mol/(m2*s)]

At a given SOC, positive-side V(V) concentration = SOC * c_total,
negative-side V(II) concentration = SOC * c_total.
Net crossover drives self-discharge and capacity fade.

Capacity fade per cycle:
    dQ = n * F * J_cross * A_membrane * t_cycle   [C/cycle]
    capacity_fade_pct = cycle_number * dQ / Q_nominal * 100

Self-discharge current from crossover:
    I_cross = n * F * J_cross * A_membrane   [A]

Coulombic efficiency:
    eta_c = 1 - I_cross / |I_discharge|

References:
    Skyllas-Kazacos, M. et al. (2011). Progress in Flow Battery Research and Development.
    J. Electrochem. Soc., 158(8), R55-R79.
    Blanc, C., Rufer, A. (2010). Paths to Sustainable Energy, InTech.
"""

import numpy as np

R_GAS = 8.314
F_CONST = 96485.0


class VRFBF1b:
    """VRFB stack voltage with vanadium crossover model."""

    SOC_MIN = 0.01
    SOC_MAX = 0.99

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2 = u["electrode_area_cm2"]["value"]
        self.E0 = u["E0"]["value"]
        self.R_ohm_cm2 = u["R_cell_ohm_cm2"]["value"]
        self.T = u["T_K"]["value"]
        self.n = int(u["n"]["value"])
        self.R_cell = self.R_ohm_cm2 / self.A_cm2

        # Crossover parameters
        self.D_v = u["D_v"]["value"]              # m2/s
        self.d_mem = u["d_membrane"]["value"]      # m
        self.A_mem = u["A_membrane_cm2"]["value"] * 1e-4  # m2
        self.c_total = u["electrolyte_conc_M"]["value"] * 1000.0  # mol/m3
        self.V_elec = u["V_electrolyte_L"]["value"] * 1e-3  # m3
        self.Q_nominal = u["Q_nominal_Ah"]["value"] * 3600.0  # C

    def _thermal_factor(self):
        return R_GAS * self.T / (self.n * F_CONST)

    def e_nernst(self, soc):
        soc = np.asarray(soc, dtype=float)
        soc = np.clip(soc, self.SOC_MIN, self.SOC_MAX)
        return self.E0 + 2.0 * self._thermal_factor() * np.log(soc / (1.0 - soc))

    def crossover_flux(self, soc):
        """Vanadium crossover flux [mol/(m2*s)] due to concentration gradient."""
        soc = np.asarray(soc, dtype=float)
        soc = np.clip(soc, self.SOC_MIN, self.SOC_MAX)
        # Concentration difference: at SOC, pos has SOC*c_total of V(V),
        # neg has SOC*c_total of V(II). Net driving force proportional to SOC imbalance.
        # Simplified: crossover proportional to total concentration gradient across membrane.
        # Higher SOC -> more V(V) on positive side -> net flux to negative side
        delta_c = self.c_total * (2.0 * soc - 1.0)  # net concentration gradient
        return self.D_v * np.abs(delta_c) / self.d_mem

    def crossover_current(self, soc):
        """Self-discharge current from crossover [A]."""
        J = self.crossover_flux(soc)
        return self.n * F_CONST * J * self.A_mem

    def capacity_fade_pct(self, soc, cycle_number, t_cycle_s=14400.0):
        """
        Cumulative capacity fade [%] from crossover.
        t_cycle_s: average cycle time in seconds (default 4h = 14400s).
        """
        soc = np.asarray(soc, dtype=float)
        cycle = np.asarray(cycle_number, dtype=float)
        J = self.crossover_flux(soc)
        dQ_per_cycle = self.n * F_CONST * J * self.A_mem * t_cycle_s  # C/cycle
        return cycle * dQ_per_cycle / self.Q_nominal * 100.0

    def cell_voltage(self, soc, current):
        """Terminal cell voltage [V]. current > 0 = discharge."""
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        return self.e_nernst(soc) - current * self.R_cell

    def stack_voltage(self, soc, current):
        return self.N_cells * self.cell_voltage(soc, current)

    def power_w(self, soc, current):
        return self.stack_voltage(soc, current) * np.asarray(current, dtype=float)

    def coulombic_efficiency(self, soc, current):
        """Coulombic efficiency accounting for crossover loss."""
        I_cross = self.crossover_current(soc)
        I = np.abs(np.asarray(current, dtype=float))
        return np.where(I > 0.01, np.clip(1.0 - I_cross / I, 0.0, 1.0), 0.0)

    def voltage_efficiency(self, soc, current):
        current = np.asarray(current, dtype=float)
        V_dis = self.cell_voltage(soc, current)
        V_chg = self.cell_voltage(soc, -np.abs(current))
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(V_chg > 0, V_dis / V_chg, 0.0)
        return np.clip(eta, 0.0, 1.0)
