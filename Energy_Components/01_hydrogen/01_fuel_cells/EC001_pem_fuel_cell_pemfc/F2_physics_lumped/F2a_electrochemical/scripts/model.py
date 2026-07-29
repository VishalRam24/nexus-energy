"""
EC001 -- PEM Fuel Cell (PEMFC) -- F2a Full Electrochemical Model

Physics-lumped model with coupled electrochemical + thermal ODE.

Voltage model (first-principles at each time step):
    V_cell = E_nernst(T, P_h2, P_o2)
             - eta_act(j, T)           Tafel kinetics
             - eta_ohm(j, T, lambda)   Springer membrane conductivity
             - eta_conc(j, j_L)        mass-transport limiting

Thermal ODE:
    m*cp * dT/dt = Q_gen - Q_cool
    Q_gen = N_cells * A_cell * j * (E_thermo - V_cell)
    Q_cool = hA * (T - T_coolant)

Reference:
    Springer et al. (1991), J. Electrochem. Soc., 138(8), 2334-2342
    Amphlett et al. (1995), J. Electrochem. Soc., 142(1), 1-8
    Barbir (2005), PEM Fuel Cells: Theory and Practice, Elsevier
"""

import numpy as np
from scipy.integrate import solve_ivp


class PEMFC_F2a:
    """PEM Fuel Cell -- full electrochemical model with thermal dynamics."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 2              # electrons per H2

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]               # cm2
        self.t_mem = u["t_mem"]["value"]                  # cm
        self.lambda_mem = u["lambda_mem"]["value"]
        self.j_L = u["j_L"]["value"]                     # A/cm2
        self.alpha_c = u["alpha_cathode"]["value"]
        self.j0_ref = u["j0_cathode_ref"]["value"]       # A/cm2
        self.E_act = u["E_act_cathode"]["value"]          # J/mol
        self.T_ref = u["T_ref"]["value"]                  # K
        self.m_stack = u["m_stack"]["value"]               # kg
        self.cp_stack = u["cp_stack"]["value"]             # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]               # W/K
        self.T_coolant = u["T_coolant"]["value"]           # K

    # ------------------------------------------------------------------
    # Nernst (thermodynamic) voltage
    # ------------------------------------------------------------------
    def nernst_voltage(self, T, P_h2, P_o2):
        """Nernst open-circuit voltage [V]."""
        # Standard potential with temperature correction
        E0 = 1.229 - 0.000846 * (T - 298.15)
        # Pressure correction via Nernst equation
        E = E0 + (self.R * T) / (2.0 * self.F) * np.log(P_h2 * np.sqrt(P_o2))
        return E

    # ------------------------------------------------------------------
    # Thermoneutral voltage (for heat generation)
    # ------------------------------------------------------------------
    def thermoneutral_voltage(self, T):
        """Thermoneutral voltage [V] -- enthalpy voltage (HHV basis)."""
        return 1.481 - 0.000226 * (T - 298.15)

    # ------------------------------------------------------------------
    # Activation overpotential -- Tafel equation
    # ------------------------------------------------------------------
    def activation_overpotential(self, j, T):
        """Cathode activation overpotential [V] from Tafel kinetics."""
        if j <= 0:
            return 0.0
        # Exchange current density with Arrhenius temperature dependence
        j0 = self.j0_ref * np.exp(
            (-self.E_act / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        j0 = max(j0, 1e-12)
        # Tafel equation
        eta_act = (self.R * T) / (self.alpha_c * self.n * self.F) * np.log(j / j0)
        return max(eta_act, 0.0)

    # ------------------------------------------------------------------
    # Ohmic overpotential -- Springer membrane model
    # ------------------------------------------------------------------
    def membrane_conductivity(self, T, lam=None):
        """Nafion ionic conductivity [S/cm] -- Springer (1991) model."""
        if lam is None:
            lam = self.lambda_mem
        sigma = (0.005139 * lam - 0.00326) * np.exp(
            1268.0 * (1.0 / 303.15 - 1.0 / T)
        )
        return max(sigma, 1e-6)

    def ohmic_overpotential(self, j, T, lam=None):
        """Ohmic loss [V] through membrane."""
        sigma = self.membrane_conductivity(T, lam)
        return j * self.t_mem / sigma

    # ------------------------------------------------------------------
    # Concentration overpotential -- mass transport
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, j_L=None):
        """Concentration (mass-transport) overpotential [V]."""
        if j_L is None:
            j_L = self.j_L
        if j <= 0:
            return 0.0
        ratio = j / j_L
        if ratio >= 1.0:
            return 10.0  # effectively infinite -- cell flooded
        return -(self.R * 353.15) / (self.n * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, P_h2, P_o2):
        """Net single-cell voltage [V]."""
        E = self.nernst_voltage(T, P_h2, P_o2)
        eta_act = self.activation_overpotential(j, T)
        eta_ohm = self.ohmic_overpotential(j, T)
        eta_conc = self.concentration_overpotential(j)
        V = E - eta_act - eta_ohm - eta_conc
        return max(V, 0.0)

    # ------------------------------------------------------------------
    # Thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j, P_h2, P_o2):
        """Temperature rate of change [K/s]."""
        V_cell = self.cell_voltage(j, T, P_h2, P_o2)
        E_th = self.thermoneutral_voltage(T)
        # Heat generated by irreversibilities
        Q_gen = self.N_cells * self.A_cell * j * (E_th - V_cell)
        # Heat removed by coolant
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, P_h2_atm, P_o2_atm,
                 dt, duration_s):
        """
        Simulate PEMFC dynamics with coupled thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2]
        T_cell_K : float
            Initial cell temperature [K]
        P_h2_atm : float
            Hydrogen partial pressure [atm]
        P_o2_atm : float
            Oxygen partial pressure [atm]
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation duration [s]

        Returns
        -------
        dict with time-series: t, voltage, power_density, efficiency,
             temperature, overpotentials (dict of arrays)
        """
        _j = current_density_A_cm2 if callable(current_density_A_cm2) else lambda t: current_density_A_cm2

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            return [self.dTdt(T, j, P_h2_atm, P_o2_atm)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_cell_K],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        voltage = np.zeros(N)
        power_density = np.zeros(N)
        efficiency = np.zeros(N)
        eta_act = np.zeros(N)
        eta_ohm = np.zeros(N)
        eta_conc = np.zeros(N)
        E_nernst = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            E_nernst[i] = self.nernst_voltage(T, P_h2_atm, P_o2_atm)
            eta_act[i] = self.activation_overpotential(j, T)
            eta_ohm[i] = self.ohmic_overpotential(j, T)
            eta_conc[i] = self.concentration_overpotential(j)
            voltage[i] = self.cell_voltage(j, T, P_h2_atm, P_o2_atm)
            power_density[i] = j * voltage[i]
            E_th = self.thermoneutral_voltage(T)
            efficiency[i] = voltage[i] / E_th if E_th > 0 else 0.0

        return {
            "t": t_out,
            "voltage": voltage,
            "power_density": power_density,
            "efficiency": efficiency,
            "temperature": T_out,
            "overpotentials": {
                "E_nernst": E_nernst,
                "activation": eta_act,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }
