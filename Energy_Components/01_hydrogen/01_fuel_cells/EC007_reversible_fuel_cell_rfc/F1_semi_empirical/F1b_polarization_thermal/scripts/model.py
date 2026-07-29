"""
EC007 -- Reversible Fuel Cell (RFC) -- F1b Polarization-Thermal Model

Extends F1a by making all loss terms explicitly temperature-dependent.
RFC operates in two modes:

FC mode (discharging):
    V_cell = E_nernst(T) - V_act(j,T) - V_ohm(j,T) - V_conc(j)
    Q_gen  = j * A_cell * N_cells * (E_tn - V_cell)   [W; always >= 0]

Electrolyser mode (charging):
    V_cell = E_nernst(T) + V_act_el(j,T) + V_ohm_el(j,T) + V_conc_el(j)
    Q_gen  = j * A_cell * N_cells * (V_cell - E_tn_el) [W; always >= 0]

Temperature-dependent terms (both modes):
    Nernst:    E_rev(T) = 1.229 - 0.000846*(T - 298.15) + RT/(2F)*ln(pH2*sqrt(pO2))
    i0(T)    = i0_ref * exp(-E_act/R * (1/T - 1/T_ref))   [Arrhenius]
    sigma(T) = sigma_303 * exp(1268*(1/303 - 1/T))         [Nafion; Springer 1991]

Lumped thermal balance:
    m_stack * cp * dT/dt = Q_gen - UA_cool * (T - T_cool)

References:
    Amphlett et al. (1995). J. Electrochem. Soc. 142(1), 1-8.
    Springer et al. (1991). J. Electrochem. Soc. 138(8), 2334-2342.
    Grigoriev et al. (2020). Int. J. Hydrogen Energy 45(53), 26651-26657.
    Ito et al. (2012). Int. J. Hydrogen Energy 37(10), 8639-8644.
"""

import numpy as np

R_GAS = 8.314      # J/(mol K)
F_FAR = 96485.0    # C/mol
N_EL  = 2          # electrons per H2 molecule


class RFCThermalModel:
    """
    Reversible Fuel Cell with explicit temperature dependence.
    Supports both FC mode (discharge) and electrolyser mode (charge).
    """

    def __init__(self, params: dict):
        cell = params["cell"]
        fc   = params["fuel_cell_mode"]
        el   = params["electrolyser_mode"]
        th   = params["thermal"]

        self.N_cells  = int(cell["N_cells"]["value"])
        self.A_cell   = float(cell["A_cell"]["value"])        # cm2
        self.T_ref    = float(cell["T_ref"]["value"])         # K
        self.pH2      = float(cell["pH2"]["value"])           # bar
        self.pO2      = float(cell["pO2"]["value"])           # bar
        self.j_L_fc   = float(cell["j_L_fc"]["value"])        # A/cm2
        self.j_L_el   = float(cell["j_L_el"]["value"])        # A/cm2

        # FC mode parameters
        self.i0_fc_ref  = float(fc["i0_ref"]["value"])
        self.E_act_fc   = float(fc["E_act_fc"]["value"])
        self.sigma_ref  = float(fc["sigma_ref"]["value"])
        self.t_mem      = float(fc["t_mem"]["value"])
        self.lambda_mem = float(fc["lambda_mem"]["value"])
        self.alpha_fc   = float(fc["alpha_fc"]["value"])
        self.B_conc_fc  = float(fc["B_conc_fc"]["value"])

        # Electrolyser mode parameters
        self.i0_el_ref  = float(el["i0_el_ref"]["value"])
        self.E_act_el   = float(el["E_act_el"]["value"])
        self.alpha_el   = float(el["alpha_el"]["value"])
        self.B_conc_el  = float(el["B_conc_el"]["value"])
        self.R_ohm_el   = float(el["R_ohm_el"]["value"])

        # Thermal
        self.E_tn       = float(th["E_tn"]["value"])
        self.E_tn_el    = float(th["E_tn_el"]["value"])
        self.m_stack    = float(th["m_stack_kg"]["value"])
        self.cp_stack   = float(th["cp_stack"]["value"])
        self.UA_cool    = float(th["UA_cool"]["value"])
        self.T_cool     = float(th["T_cool"]["value"])

    # ------------------------------------------------------------------
    # Shared: Nernst voltage
    # ------------------------------------------------------------------

    def nernst_voltage(self, T):
        """Reversible open-circuit voltage [V]."""
        T = np.asarray(T, dtype=float)
        return (
            1.229
            - 0.000846 * (T - 298.15)
            + (R_GAS * T) / (N_EL * F_FAR) * np.log(self.pH2 * np.sqrt(self.pO2))
        )

    # ------------------------------------------------------------------
    # FC mode methods
    # ------------------------------------------------------------------

    def exchange_current_fc(self, T):
        """Arrhenius exchange current density for ORR [A/cm2]."""
        T = np.asarray(T, dtype=float)
        return self.i0_fc_ref * np.exp(
            -self.E_act_fc / R_GAS * (1.0 / T - 1.0 / self.T_ref)
        )

    def membrane_conductivity(self, T):
        """Nafion membrane conductivity [S/cm] -- Springer 1991."""
        T = np.asarray(T, dtype=float)
        lam = self.lambda_mem
        sigma_303 = 0.005139 * lam - 0.00326
        return sigma_303 * np.exp(1268.0 * (1.0 / 303.0 - 1.0 / T))

    def membrane_resistance(self, T):
        """Membrane area-specific resistance [ohm cm2]."""
        return self.t_mem / self.membrane_conductivity(T)

    def activation_loss_fc(self, j, T):
        """FC activation overpotential [V] (Butler-Volmer arcsinh form)."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        i0 = self.exchange_current_fc(T)
        j_safe = np.maximum(j, 1e-10)
        return (R_GAS * T) / (self.alpha_fc * N_EL * F_FAR) * np.arcsinh(
            j_safe / (2.0 * i0)
        )

    def ohmic_loss_fc(self, j, T):
        """FC ohmic voltage loss [V]."""
        return np.asarray(j, dtype=float) * self.membrane_resistance(T)

    def concentration_loss_fc(self, j):
        """FC concentration overpotential [V]."""
        j = np.asarray(j, dtype=float)
        r = np.minimum(j / self.j_L_fc, 0.9999)
        return np.where(j > 0, -self.B_conc_fc * np.log(1.0 - r), 0.0)

    def cell_voltage_fc(self, j, T):
        """FC mode cell voltage [V] (clipped to [0, E_nernst])."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E = self.nernst_voltage(T)
        V = (E
             - self.activation_loss_fc(j, T)
             - self.ohmic_loss_fc(j, T)
             - self.concentration_loss_fc(j))
        return np.clip(V, 0.0, E)

    def heat_generation_fc(self, j, T):
        """FC heat generation per unit area [W/cm2]; Q = j*(E_tn - V_cell) >= 0."""
        V = self.cell_voltage_fc(j, T)
        return np.asarray(j, dtype=float) * (self.E_tn - V)

    def efficiency_fc(self, j, T):
        """Voltage efficiency vs. HHV (1.481 V)."""
        return self.cell_voltage_fc(j, T) / self.E_tn

    # ------------------------------------------------------------------
    # Electrolyser mode methods
    # ------------------------------------------------------------------

    def exchange_current_el(self, T):
        """Arrhenius exchange current density for OER [A/cm2]."""
        T = np.asarray(T, dtype=float)
        return self.i0_el_ref * np.exp(
            -self.E_act_el / R_GAS * (1.0 / T - 1.0 / self.T_ref)
        )

    def activation_loss_el(self, j, T):
        """Electrolyser activation overpotential [V]."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        i0 = self.exchange_current_el(T)
        j_safe = np.maximum(j, 1e-10)
        return (R_GAS * T) / (self.alpha_el * N_EL * F_FAR) * np.arcsinh(
            j_safe / (2.0 * i0)
        )

    def ohmic_loss_el(self, j, T):
        """Electrolyser ohmic loss [V] -- includes membrane + ionomer contacts."""
        j = np.asarray(j, dtype=float)
        # Membrane resistance is same Nafion, contact resistance added via R_ohm_el
        R_mem = self.membrane_resistance(T)
        return j * (R_mem + self.R_ohm_el)

    def concentration_loss_el(self, j):
        """Electrolyser bubble-induced concentration overpotential [V]."""
        j = np.asarray(j, dtype=float)
        r = np.minimum(j / self.j_L_el, 0.9999)
        return np.where(j > 0, self.B_conc_el * np.log(1.0 / (1.0 - r)), 0.0)

    def cell_voltage_el(self, j, T):
        """Electrolyser mode cell voltage [V] (>= E_nernst)."""
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)
        E = self.nernst_voltage(T)
        V = (E
             + self.activation_loss_el(j, T)
             + self.ohmic_loss_el(j, T)
             + self.concentration_loss_el(j))
        return np.maximum(V, E)

    def heat_generation_el(self, j, T):
        """Electrolyser heat per area [W/cm2]; Q = j*(V_cell - E_tn_el) >= 0."""
        V = self.cell_voltage_el(j, T)
        Q = np.asarray(j, dtype=float) * (V - self.E_tn_el)
        return np.maximum(Q, 0.0)

    def efficiency_el(self, j, T):
        """Electrolyser efficiency = E_tn / V_cell (HHV basis)."""
        V = self.cell_voltage_el(j, T)
        return np.where(V > 0, self.E_tn_el / V, 0.0)

    # ------------------------------------------------------------------
    # Thermal balance (lumped)
    # ------------------------------------------------------------------

    def dTdt(self, T, j, mode="fc"):
        """
        Stack temperature derivative [K/s] from lumped thermal balance.

        dT/dt = (Q_gen_stack - Q_cool) / (m_stack * cp_stack)
        Q_gen_stack [W] = q [W/cm2] * A_cell * N_cells
        Q_cool [W]      = UA_cool * (T - T_cool)
        """
        T = np.asarray(T, dtype=float)
        j = np.asarray(j, dtype=float)
        if mode == "fc":
            q_area = self.heat_generation_fc(j, T)
        else:
            q_area = self.heat_generation_el(j, T)
        Q_gen  = q_area * self.A_cell * self.N_cells
        Q_cool = self.UA_cool * (T - self.T_cool)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Unified evaluate
    # ------------------------------------------------------------------

    def evaluate(self, j, T, mode="fc"):
        """
        Full operating-point evaluation.

        Parameters
        ----------
        j    : float or array -- current density [A/cm2]
        T    : float or array -- temperature [K]
        mode : 'fc' or 'electrolyser'

        Returns
        -------
        dict with all outputs
        """
        j = np.asarray(j, dtype=float)
        T = np.asarray(T, dtype=float)

        E = self.nernst_voltage(T)

        if mode == "fc":
            V_act  = self.activation_loss_fc(j, T)
            V_ohm  = self.ohmic_loss_fc(j, T)
            V_conc = self.concentration_loss_fc(j)
            V_cell = np.clip(E - V_act - V_ohm - V_conc, 0.0, E)
            Q_area = j * (self.E_tn - V_cell)
            eta    = V_cell / self.E_tn
            R_mem  = self.membrane_resistance(T)
        else:
            V_act  = self.activation_loss_el(j, T)
            V_ohm  = self.ohmic_loss_el(j, T)
            V_conc = self.concentration_loss_el(j)
            V_cell = np.maximum(E + V_act + V_ohm + V_conc, E)
            Q_area = np.maximum(j * (V_cell - self.E_tn_el), 0.0)
            eta    = np.where(V_cell > 0, self.E_tn_el / V_cell, 0.0)
            R_mem  = self.membrane_resistance(T)

        P_stack_kW  = j * V_cell * self.A_cell * self.N_cells / 1000.0
        Q_stack_W   = Q_area * self.A_cell * self.N_cells
        dT_dt       = self.dTdt(T, j, mode=mode)

        return {
            "cell_voltage":      V_cell,
            "power_stack_kW":    P_stack_kW,
            "efficiency":        eta,
            "heat_area":         Q_area,
            "heat_stack_W":      Q_stack_W,
            "membrane_resistance": R_mem,
            "E_nernst":          E,
            "V_act":             V_act,
            "V_ohm":             V_ohm,
            "V_conc":            V_conc,
            "dTdt_K_s":          dT_dt,
        }
