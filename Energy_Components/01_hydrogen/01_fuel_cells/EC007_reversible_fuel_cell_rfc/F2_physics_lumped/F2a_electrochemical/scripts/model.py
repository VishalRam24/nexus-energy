"""
EC007 -- Reversible Fuel Cell (RFC / URFC) -- F2a Bidirectional Electrochemical Model

Physics-lumped, first-principles model with coupled electrochemical + thermal ODE
that handles BOTH operating modes of a unitised regenerative fuel cell:

    Discharge  (Fuel-Cell mode,     j > 0):  V_cell = E_nernst - eta_act - eta_ohm - eta_conc
    Charge     (Electrolysis mode,  j < 0):  V_cell = E_nernst + eta_act + eta_ohm + eta_conc
    Open circuit                  (j = 0):    V_cell = E_nernst

The sign of the current density selects the mode. The same membrane-electrode
assembly carries both reactions, so the membrane (ohmic, Springer 1991) physics
is shared, while the activation kinetics (Tafel/Butler-Volmer) and the
mass-transport limiting currents use mode-specific parameters because the
bifunctional electrocatalyst is asymmetric (ORR/HOR for FC, OER/HER for EL).

Voltage model (first-principles, evaluated each time step):
    E_nernst(T, P_h2, P_o2)            Nernst thermodynamic potential
    eta_act(|j|, T, mode)              Tafel kinetics (Arrhenius j0)
    eta_ohm(|j|, T)                    Springer (1991) membrane conductivity
    eta_conc(|j|, j_L_mode)            mass-transport limiting current

Thermal ODE (lumped 0D energy balance):
    m*cp * dT/dt = Q_gen - Q_cool
    FC mode:  Q_gen = N*A*j*(E_thermoneutral - V_cell)        (exothermic)
    EL mode:  Q_gen = N*A*|j|*(V_cell - E_thermoneutral)      (Joule + entropic)
    Q_cool   = hA*(T - T_coolant)

Round-trip efficiency for a charge/discharge pair at current density j:
    eta_rt = V_FC(j) / V_EL(j)   (voltaic; < 1 by the 2nd law)

References:
    Springer, Zawodzinski & Gottesfeld (1991), J. Electrochem. Soc., 138(8), 2334-2342
    Amphlett et al. (1995), J. Electrochem. Soc., 142(1), 1-8
    Marangio, Santarelli & Calì (2009), Int. J. Hydrogen Energy, 34, 1143-1158
    Doddathimmaiah & Andrews (2009), Int. J. Hydrogen Energy, 34, 8157-8170 (URFC)
    Barbir (2005), PEM Fuel Cells: Theory and Practice, Elsevier
"""

import numpy as np
from scipy.integrate import solve_ivp


class RFC_F2a:
    """Reversible (unitised regenerative) fuel cell -- bidirectional electrochemical
    model with lumped thermal dynamics."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 2              # electrons per H2 / per H2O

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]                # cm2
        self.t_mem = u["t_mem"]["value"]                  # cm
        self.lambda_mem = u["lambda_mem"]["value"]
        self.j_L_fc = u["j_L_fc"]["value"]                # A/cm2
        self.j_L_el = u["j_L_el"]["value"]                # A/cm2
        self.alpha_fc = u["alpha_fc"]["value"]
        self.alpha_el = u["alpha_el"]["value"]
        self.j0_fc_ref = u["j0_fc_ref"]["value"]          # A/cm2
        self.j0_el_ref = u["j0_el_ref"]["value"]          # A/cm2
        self.E_act_fc = u["E_act_fc"]["value"]            # J/mol
        self.E_act_el = u["E_act_el"]["value"]            # J/mol
        self.T_ref = u["T_ref"]["value"]                  # K
        self.m_stack = u["m_stack"]["value"]              # kg
        self.cp_stack = u["cp_stack"]["value"]            # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]              # W/K
        self.T_coolant = u["T_coolant"]["value"]          # K

    # ------------------------------------------------------------------
    # Nernst (thermodynamic) reversible voltage -- shared by both modes
    # ------------------------------------------------------------------
    def nernst_voltage(self, T, P_h2, P_o2):
        """Nernst reversible open-circuit voltage [V] (Amphlett 1995)."""
        E0 = 1.229 - 0.000846 * (T - 298.15)
        E = E0 + (self.R * T) / (2.0 * self.F) * np.log(P_h2 * np.sqrt(P_o2))
        return E

    # ------------------------------------------------------------------
    # Thermoneutral voltage (enthalpy / HHV basis) -- for heat generation
    # ------------------------------------------------------------------
    def thermoneutral_voltage(self, T):
        """Thermoneutral voltage [V] (HHV basis), shared by both modes."""
        return 1.481 - 0.000226 * (T - 298.15)

    # ------------------------------------------------------------------
    # Activation overpotential (magnitude) -- Tafel kinetics, mode-specific
    # ------------------------------------------------------------------
    def activation_overpotential(self, abs_j, T, mode):
        """Activation overpotential magnitude [V] for the given mode."""
        if abs_j <= 0:
            return 0.0
        if mode == "FC":
            j0_ref, E_act, alpha = self.j0_fc_ref, self.E_act_fc, self.alpha_fc
        else:  # EL
            j0_ref, E_act, alpha = self.j0_el_ref, self.E_act_el, self.alpha_el
        # Arrhenius exchange-current density
        j0 = j0_ref * np.exp((-E_act / self.R) * (1.0 / T - 1.0 / self.T_ref))
        j0 = max(j0, 1e-14)
        eta = (self.R * T) / (alpha * self.n * self.F) * np.log(abs_j / j0)
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Ohmic overpotential -- Springer (1991) membrane, shared by both modes
    # ------------------------------------------------------------------
    def membrane_conductivity(self, T, lam=None):
        """Nafion ionic conductivity [S/cm] -- Springer (1991)."""
        if lam is None:
            lam = self.lambda_mem
        sigma = (0.005139 * lam - 0.00326) * np.exp(1268.0 * (1.0 / 303.15 - 1.0 / T))
        return max(sigma, 1e-6)

    def ohmic_overpotential(self, abs_j, T, lam=None):
        """Ohmic loss magnitude [V] through the membrane."""
        sigma = self.membrane_conductivity(T, lam)
        return abs_j * self.t_mem / sigma

    # ------------------------------------------------------------------
    # Concentration overpotential (magnitude) -- mass transport, mode-specific j_L
    # ------------------------------------------------------------------
    def concentration_overpotential(self, abs_j, T, mode):
        """Concentration (mass-transport) overpotential magnitude [V]."""
        if abs_j <= 0:
            return 0.0
        j_L = self.j_L_fc if mode == "FC" else self.j_L_el
        ratio = abs_j / j_L
        if ratio >= 1.0:
            return 10.0  # limiting current reached -- effectively infinite
        return -(self.R * T) / (self.n * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell voltage -- mode selected by the SIGN of j
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, P_h2, P_o2):
        """Net single-cell voltage [V]. j>0 FC (discharge), j<0 EL (charge)."""
        E = self.nernst_voltage(T, P_h2, P_o2)
        if j == 0:
            return E
        mode = "FC" if j > 0 else "EL"
        abs_j = abs(j)
        eta_act = self.activation_overpotential(abs_j, T, mode)
        eta_ohm = self.ohmic_overpotential(abs_j, T)
        eta_conc = self.concentration_overpotential(abs_j, T, mode)
        eta = eta_act + eta_ohm + eta_conc
        if mode == "FC":
            return max(E - eta, 0.0)     # discharge: losses subtract
        else:
            return E + eta               # charge: losses add (V above E_rev)

    def mode_of(self, j):
        if j > 0:
            return "FC"
        if j < 0:
            return "EL"
        return "OCV"

    # ------------------------------------------------------------------
    # Round-trip (voltaic) efficiency for a symmetric charge/discharge pair
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, j_mag, T, P_h2, P_o2):
        """Voltaic round-trip efficiency = V_FC(+j) / V_EL(-j) for |j|=j_mag."""
        V_fc = self.cell_voltage(abs(j_mag), T, P_h2, P_o2)
        V_el = self.cell_voltage(-abs(j_mag), T, P_h2, P_o2)
        return V_fc / V_el if V_el > 0 else 0.0

    # ------------------------------------------------------------------
    # Thermal ODE derivative -- both modes are dissipative (Q_gen >= 0)
    # ------------------------------------------------------------------
    def dTdt(self, T, j, P_h2, P_o2):
        """Temperature rate of change [K/s]."""
        V_cell = self.cell_voltage(j, T, P_h2, P_o2)
        E_th = self.thermoneutral_voltage(T)
        abs_j = abs(j)
        if j >= 0:
            # Fuel cell: heat = (E_th - V) * j   (V < E_th -> exothermic)
            Q_gen = self.N_cells * self.A_cell * abs_j * (E_th - V_cell)
        else:
            # Electrolysis: heat = (V - E_th) * |j|  (V > E_th above thermoneutral)
            Q_gen = self.N_cells * self.A_cell * abs_j * (V_cell - E_th)
        Q_gen = max(Q_gen, 0.0)
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation (coupled thermal ODE via solve_ivp)
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, P_h2_atm, P_o2_atm,
                 dt, duration_s):
        """
        Simulate RFC dynamics with the coupled thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2]. >0 discharge (FC), <0 charge (EL).
        T_cell_K : float       Initial cell temperature [K]
        P_h2_atm : float       Hydrogen partial pressure [atm]
        P_o2_atm : float       Oxygen partial pressure [atm]
        dt : float             Output time step [s]
        duration_s : float     Total simulation duration [s]

        Returns
        -------
        dict with time-series: t, voltage, power_density (signed: +out / -in),
            efficiency, temperature, mode (list), overpotentials (dict of arrays).
        """
        _j = current_density_A_cm2 if callable(current_density_A_cm2) \
            else (lambda t: current_density_A_cm2)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            return [self.dTdt(T, j, P_h2_atm, P_o2_atm)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_cell_K],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10, max_step=dt,
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
        modes = []

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            mode = self.mode_of(j)
            modes.append(mode)
            abs_j = abs(j)
            E_nernst[i] = self.nernst_voltage(T, P_h2_atm, P_o2_atm)
            if mode == "OCV":
                eta_act[i] = eta_ohm[i] = eta_conc[i] = 0.0
            else:
                eta_act[i] = self.activation_overpotential(abs_j, T, mode)
                eta_ohm[i] = self.ohmic_overpotential(abs_j, T)
                eta_conc[i] = self.concentration_overpotential(abs_j, T, mode)
            V = self.cell_voltage(j, T, P_h2_atm, P_o2_atm)
            voltage[i] = V
            # Signed electrical power density: positive = delivered (FC),
            # negative = consumed (EL). P = V*j carries the sign of j.
            power_density[i] = V * j
            E_th = self.thermoneutral_voltage(T)
            if mode == "FC":
                # FC efficiency vs thermoneutral (HHV) potential
                efficiency[i] = V / E_th if E_th > 0 else 0.0
            elif mode == "EL":
                # EL efficiency = thermoneutral / cell voltage (energy in vs HHV stored)
                efficiency[i] = E_th / V if V > 0 else 0.0
            else:
                efficiency[i] = 1.0

        return {
            "t": t_out,
            "voltage": voltage,
            "power_density": power_density,
            "efficiency": efficiency,
            "temperature": T_out,
            "mode": modes,
            "overpotentials": {
                "E_nernst": E_nernst,
                "activation": eta_act,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }
