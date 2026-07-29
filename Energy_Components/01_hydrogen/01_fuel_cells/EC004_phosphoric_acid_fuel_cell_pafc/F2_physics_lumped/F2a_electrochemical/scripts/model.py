"""
EC004 -- Phosphoric Acid Fuel Cell (PAFC) -- F2a Full Electrochemical Model

Physics-lumped (0D) first-principles model: coupled electrochemical voltage
balance + a single thermal-mass ODE integrated with scipy.solve_ivp.

PAFC distinctives vs PEMFC (EC001):
  * Operates at 150-210 C (423-483 K) with a concentrated (95-100%) H3PO4
    electrolyte held in a SiC matrix -- not a Nafion membrane.
  * Ohmic loss is governed by H3PO4 ionic conductivity (Arrhenius in T),
    which rises strongly with temperature.
  * Slow O2 reduction (ORR) kinetics in H3PO4 -> cathode-limited; large
    activation energy (~70 kJ/mol).
  * CO TOLERANCE: at PAFC temperatures the Pt anode tolerates ~1-2 % CO in
    reformate; CO coverage / poisoning penalty falls as temperature rises
    (modelled here as a small, T-decaying anode overpotential).

Voltage model (first-principles, per cell, each time step):
    V_cell = E_nernst(T, P_h2, P_o2)              Nernst / thermodynamic
             - eta_act(j, T)                       Butler-Volmer (arcsinh) ORR
             - eta_ohm(j, T)                       H3PO4 conductivity (Razaq 1991)
             - eta_conc(j, j_L)                    mass-transport limiting
             - eta_CO(T, x_CO)                      CO poisoning (T-decaying)

Thermal lumped ODE (single stack node):
    m*cp * dT/dt = Q_gen - Q_cool
    Q_gen  = N_cells * A_cell * j * (E_tn(T) - V_cell)   (irreversibilities, >= 0)
    Q_cool = hA * (T - T_coolant)

Energy conservation: every Joule of (E_tn - V) * I that is not electrical work
appears as Q_gen; at steady state Q_gen == Q_cool.

References:
    Razaq M., Razaq A., Yeager E., DesMarteau D.D., Singh S. (1989).
        J. Electrochem. Soc. 136(2), 385-390.  [H3PO4 conductivity]
    Appleby A.J. & Foulkes F.R. (1989). Fuel Cell Handbook. Van Nostrand Reinhold.
    Li Q., He R., Jensen J.O., Bjerrum N.J. (2003). Chem. Mater. 15(26),
        4896-4915.  [phosphoric-acid electrolyte conduction]
    Patel K.K. et al. (2012). Int. J. Hydrogen Energy 37(3), 2346-2359.
        [PAFC polarization at elevated temperature]
    EG&G Technical Services / U.S. DOE (2004). Fuel Cell Handbook, 7th ed.
        [PAFC operating envelope, CO tolerance, performance]
    O'Hayre R., Cha S.-W., Colella W., Prinz F.B. (2016). Fuel Cell
        Fundamentals, 3rd ed., Wiley.  [Butler-Volmer / Nernst forms]
"""

import numpy as np
from scipy.integrate import solve_ivp


class PAFC_F2a:
    """Phosphoric Acid Fuel Cell -- full electrochemical model + thermal ODE."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 2              # electrons per H2

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]                 # cm2
        self.t_acid = u["t_acid"]["value"]                 # cm
        self.sigma_ref = u["sigma_ref_H3PO4"]["value"]     # S/cm at T_ref
        self.E_act_sigma = u["E_act_sigma"]["value"]       # J/mol
        self.j_L = u["j_L"]["value"]                       # A/cm2
        self.alpha_c = u["alpha_cathode"]["value"]
        self.j0_ref = u["j0_cathode_ref"]["value"]         # A/cm2
        self.E_act = u["E_act_cathode"]["value"]           # J/mol
        self.B_conc = u["B_conc"]["value"]                 # V
        self.T_ref = u["T_ref"]["value"]                   # K
        self.E_tn_ref = u["E_tn_ref"]["value"]             # V at 298K
        self.k_tn = u["k_tn"]["value"]                     # V/K
        self.m_stack = u["m_stack"]["value"]               # kg
        self.cp_stack = u["cp_stack"]["value"]             # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]               # W/K
        self.T_coolant = u["T_coolant"]["value"]           # K
        self.x_CO = u["CO_anode"]["value"]                 # mole fraction
        self.k_CO_ref = u["k_CO_ref"]["value"]             # V
        self.E_act_CO = u["E_act_CO"]["value"]             # J/mol

    # ------------------------------------------------------------------
    # Nernst (thermodynamic) voltage
    # ------------------------------------------------------------------
    def nernst_voltage(self, T, P_h2, P_o2):
        """Reversible open-circuit voltage [V] (O'Hayre 2016, Nernst eq.)."""
        # Standard potential with temperature correction (entropy term)
        E0 = 1.229 - 0.000846 * (T - 298.15)
        # Nernst pressure correction
        E = E0 + (self.R * T) / (2.0 * self.F) * np.log(P_h2 * np.sqrt(P_o2))
        return E

    # ------------------------------------------------------------------
    # Thermoneutral voltage (for heat generation, HHV basis)
    # ------------------------------------------------------------------
    def thermoneutral_voltage(self, T):
        """Thermoneutral (enthalpy) voltage [V], HHV basis."""
        return self.E_tn_ref - self.k_tn * (T - 298.15)

    # ------------------------------------------------------------------
    # Cathode exchange current density (Arrhenius in T)
    # ------------------------------------------------------------------
    def exchange_current_density(self, T):
        """ORR exchange current density [A/cm2], Arrhenius temperature law."""
        j0 = self.j0_ref * np.exp(
            -self.E_act / self.R * (1.0 / T - 1.0 / self.T_ref)
        )
        return max(j0, 1e-15)

    # ------------------------------------------------------------------
    # Activation overpotential -- Butler-Volmer (arcsinh form)
    # ------------------------------------------------------------------
    def activation_overpotential(self, j, T):
        """Cathode activation overpotential [V], full Butler-Volmer arcsinh."""
        if j <= 0:
            return 0.0
        j0 = self.exchange_current_density(T)
        eta = (self.R * T) / (self.alpha_c * self.n * self.F) * np.arcsinh(
            j / (2.0 * j0)
        )
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # H3PO4 ionic conductivity (Arrhenius -- Razaq 1989 / Li 2003)
    # ------------------------------------------------------------------
    def acid_conductivity(self, T):
        """Concentrated H3PO4 ionic conductivity [S/cm]."""
        sigma = self.sigma_ref * np.exp(
            self.E_act_sigma / self.R * (1.0 / self.T_ref - 1.0 / T)
        )
        return max(sigma, 1e-6)

    def ohmic_overpotential(self, j, T):
        """Ohmic loss [V] through the H3PO4/SiC matrix."""
        return j * self.t_acid / self.acid_conductivity(T)

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
            return 10.0  # effectively infinite -- mass-transport collapse
        return -self.B_conc * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # CO poisoning penalty -- PAFC-specific, decays with temperature
    # ------------------------------------------------------------------
    def co_overpotential(self, T, x_CO=None):
        """
        Anode CO-poisoning overpotential [V].

        At PAFC temperatures the Pt anode tolerates CO; the penalty falls as
        T rises because CO desorption is thermally activated. Modelled as a
        log-coverage penalty whose prefactor decays with an Arrhenius factor:
            eta_CO = k_CO(T) * ln(1 + x_CO / x_ref),  x_ref = 0.001
            k_CO(T) = k_CO_ref * exp(E_act_CO/R * (1/T - 1/T_ref))
        (penalty smaller at high T => CO tolerance). See DOE Fuel Cell
        Handbook (2004), PAFC CO-tolerance discussion.
        """
        if x_CO is None:
            x_CO = self.x_CO
        if x_CO <= 0:
            return 0.0
        k_CO = self.k_CO_ref * np.exp(
            self.E_act_CO / self.R * (1.0 / T - 1.0 / self.T_ref)
        )
        return k_CO * np.log(1.0 + x_CO / 0.001)

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, P_h2, P_o2, x_CO=None):
        """Net single-cell voltage [V]."""
        E = self.nernst_voltage(T, P_h2, P_o2)
        eta_act = self.activation_overpotential(j, T)
        eta_ohm = self.ohmic_overpotential(j, T)
        eta_conc = self.concentration_overpotential(j)
        eta_co = self.co_overpotential(T, x_CO)
        V = E - eta_act - eta_ohm - eta_conc - eta_co
        return max(V, 0.0)

    # ------------------------------------------------------------------
    # Thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j, P_h2, P_o2, x_CO=None):
        """Temperature rate of change [K/s] (lumped stack energy balance)."""
        V_cell = self.cell_voltage(j, T, P_h2, P_o2, x_CO)
        E_th = self.thermoneutral_voltage(T)
        # Heat generated by irreversibilities (>= 0)
        Q_gen = self.N_cells * self.A_cell * j * max(E_th - V_cell, 0.0)
        # Heat removed by coolant
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, P_h2_atm, P_o2_atm,
                 dt, duration_s, x_CO=None):
        """
        Simulate PAFC dynamics with the coupled thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2].
        T_cell_K : float
            Initial stack temperature [K].
        P_h2_atm, P_o2_atm : float
            Reactant partial pressures [atm].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation duration [s].
        x_CO : float, optional
            Anode CO mole fraction (defaults to parameter value).

        Returns
        -------
        dict of time-series arrays: t, voltage, power_density, efficiency,
            temperature, overpotentials (dict of arrays).
        """
        _j = current_density_A_cm2 if callable(current_density_A_cm2) \
            else (lambda t: current_density_A_cm2)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            return [self.dTdt(T, j, P_h2_atm, P_o2_atm, x_CO)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_cell_K],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt,
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
        eta_co = np.zeros(N)
        E_nernst = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            E_nernst[i] = self.nernst_voltage(T, P_h2_atm, P_o2_atm)
            eta_act[i] = self.activation_overpotential(j, T)
            eta_ohm[i] = self.ohmic_overpotential(j, T)
            eta_conc[i] = self.concentration_overpotential(j)
            eta_co[i] = self.co_overpotential(T, x_CO)
            voltage[i] = self.cell_voltage(j, T, P_h2_atm, P_o2_atm, x_CO)
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
                "co_poisoning": eta_co,
            },
        }
