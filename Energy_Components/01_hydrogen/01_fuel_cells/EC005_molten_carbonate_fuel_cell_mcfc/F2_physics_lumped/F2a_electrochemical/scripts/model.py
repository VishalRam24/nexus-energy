"""
EC005 -- Molten Carbonate Fuel Cell (MCFC) -- F2a Full Electrochemical Model

Physics-lumped (0D) first-principles model with a coupled electrochemical +
thermal ODE, integrated with scipy.integrate.solve_ivp.

MCFC specifics
--------------
Operates at ~923 K (650 degC) with a molten Li2CO3/K2CO3 (62/38 mol%) eutectic
electrolyte held in a porous SiC matrix. The charge carrier is the carbonate
ion CO3(2-) (n = 2 electrons per carbonate / per H2). CO2 must be cycled from
the anode exhaust to the cathode feed:

    Cathode (ORR):  1/2 O2 + CO2(cat) + 2 e-  ->  CO3(2-)
    Anode  (HOR):   H2 + CO3(2-)  ->  H2O + CO2(an) + 2 e-
    Overall:        H2 + 1/2 O2 + CO2(cat) -> H2O + CO2(an)

Because CO2 appears on both electrodes, the Nernst equation carries CO2 partial
pressures on both sides (this is the defining feature of MCFC thermodynamics).

Voltage model (evaluated at each integration step)
--------------------------------------------------
    V_cell = E_nernst(T, pH2, pO2, pH2O, pCO2_cat, pCO2_an)
             - eta_act(j, T)      Butler-Volmer (arcsinh) anode + cathode kinetics
             - eta_ohm(j, T)      molten-carbonate ohmic loss (Uchida Arrhenius sigma)
             - eta_conc(j, j_L)   mass-transport limiting loss

    E_nernst(T) = E0(T) + RT/(2F) * ln( pH2 * sqrt(pO2) * pCO2_cat
                                        / (pH2O * pCO2_an) )
    E0(T)       = 1.05 - 2.88e-4 * (T - 873.15)         [V]  (Lu & Selman 1984)

Molten carbonate conductivity (Uchida et al. 1983):
    sigma_mc(T) = A_mc * exp(-E_act_mc / (R T))         [S/cm]
    eta_ohm     = j * t_mc / sigma_mc(T)

Activation (combined anode + cathode, Butler-Volmer symmetric form):
    eta_act = RT/(alpha n F) * [ asinh(j / 2 i0_a) + asinh(j / 2 i0_c) ]
    i0_x(T) = i0_x_ref * exp(-E_act_x / R * (1/T - 1/T_ref))   (Arrhenius)

Thermal ODE (lumped, 0D energy balance)
---------------------------------------
    m_stack cp_stack dT/dt = Q_gen - Q_cool
    Q_gen  = N_cells * A_cell * j * (E_tn - V_cell)     irreversible + entropic heat
    Q_cool = hA_cool * (T - T_coolant)                  cathode-air sweep cooling

E_tn is the thermoneutral voltage (~1.21 V at 650 degC). Energy conservation:
all electrical work that is NOT delivered to the load (E_tn - V_cell) becomes
heat, so Q_gen >= 0 for j >= 0 and the first law is satisfied per unit charge.

References
----------
    Uchida I. et al. (1983). Electrochim. Acta, 28(10), 1423-1431.
        Molten carbonate ionic conductivity (Arrhenius).
    Lu S.T. & Selman J.R. (1984). J. Electrochem. Soc., 131(12), 2827-2833.
        MCFC reversible potential / temperature dependence.
    Yuh C. & Selman J.R. (1991). J. Electrochem. Soc., 138(12), 3649-3655.
        MCFC polarization, electrode kinetics.
    Bischoff M. & Huppmann G. (2002). J. Power Sources, 105(2), 216-223.
        Hot-Module MCFC stack thermal behaviour.
    Larminie J. & Dicks A. (2003). Fuel Cell Systems Explained, 2nd ed., Wiley.
        Thermoneutral voltage, lumped thermal balance.
"""

import numpy as np
from scipy.integrate import solve_ivp


class MCFC_F2a:
    """Molten Carbonate Fuel Cell -- full electrochemical model with thermal ODE."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 2              # electrons per H2 / per CO3(2-)

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells   = u["N_cells"]["value"]
        self.A_cell    = u["A_cell"]["value"]            # cm2
        self.t_mc      = u["t_mc"]["value"]              # cm
        self.A_mc      = u["A_mc"]["value"]              # S/cm
        self.E_act_mc  = u["E_act_mc"]["value"]          # J/mol
        self.j_L       = u["j_L"]["value"]               # A/cm2
        self.alpha     = u["alpha"]["value"]
        self.i0_a_ref  = u["i0_anode_ref"]["value"]      # A/cm2
        self.E_act_a   = u["E_act_anode"]["value"]       # J/mol
        self.i0_c_ref  = u["i0_cathode_ref"]["value"]    # A/cm2
        self.E_act_c   = u["E_act_cathode"]["value"]     # J/mol
        self.B_conc    = u["B_conc"]["value"]            # V
        self.E_tn      = u["E_tn"]["value"]              # V
        self.T_ref     = u["T_ref"]["value"]             # K
        self.m_stack   = u["m_stack"]["value"]           # kg
        self.cp_stack  = u["cp_stack"]["value"]          # J/(kg.K)
        self.hA_cool   = u["hA_cool"]["value"]           # W/K
        self.T_coolant = u["T_coolant"]["value"]         # K

    # ------------------------------------------------------------------
    # Molten carbonate conductivity (Uchida 1983)
    # ------------------------------------------------------------------
    def carbonate_conductivity(self, T):
        """Molten Li2CO3/K2CO3 conductivity [S/cm] -- Arrhenius (Uchida 1983)."""
        return self.A_mc * np.exp(-self.E_act_mc / (self.R * T))

    def carbonate_resistance(self, T):
        """Area-specific carbonate resistance [ohm.cm2]."""
        return self.t_mc / self.carbonate_conductivity(T)

    # ------------------------------------------------------------------
    # Nernst (thermodynamic) voltage -- CO2 on both electrodes
    # ------------------------------------------------------------------
    def nernst_voltage(self, T, pH2, pO2, pH2O, pCO2_cat, pCO2_an):
        """MCFC open-circuit Nernst voltage [V] including CO2 partial pressures."""
        E0_T = 1.05 - 0.000288 * (T - 873.15)            # Lu & Selman (1984)
        arg = (pH2 * np.sqrt(pO2) * pCO2_cat) / (pH2O * pCO2_an)
        return E0_T + (self.R * T) / (self.n * self.F) * np.log(arg)

    # ------------------------------------------------------------------
    # Thermoneutral voltage (for heat generation)
    # ------------------------------------------------------------------
    def thermoneutral_voltage(self, T):
        """Thermoneutral voltage [V] -- weak T dependence (Larminie & Dicks 2003)."""
        return self.E_tn - 0.00010 * (T - 923.15)

    # ------------------------------------------------------------------
    # Exchange current densities (Arrhenius)
    # ------------------------------------------------------------------
    def i0_anode(self, T):
        return self.i0_a_ref * np.exp(-self.E_act_a / self.R * (1.0 / T - 1.0 / self.T_ref))

    def i0_cathode(self, T):
        return self.i0_c_ref * np.exp(-self.E_act_c / self.R * (1.0 / T - 1.0 / self.T_ref))

    # ------------------------------------------------------------------
    # Activation overpotential -- Butler-Volmer (arcsinh) anode + cathode
    # ------------------------------------------------------------------
    def activation_overpotential(self, j, T):
        """Combined anode + cathode activation overpotential [V]."""
        if j <= 0:
            return 0.0
        i0_a = max(self.i0_anode(T), 1e-12)
        i0_c = max(self.i0_cathode(T), 1e-12)
        coeff = (self.R * T) / (self.alpha * self.n * self.F)
        V_a = coeff * np.arcsinh(j / (2.0 * i0_a))
        V_c = coeff * np.arcsinh(j / (2.0 * i0_c))
        return max(V_a + V_c, 0.0)

    # ------------------------------------------------------------------
    # Ohmic overpotential -- molten carbonate resistance
    # ------------------------------------------------------------------
    def ohmic_overpotential(self, j, T):
        """Ohmic loss [V] through molten carbonate electrolyte."""
        return j * self.carbonate_resistance(T)

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
            return 10.0  # effectively infinite -- transport limited
        return -self.B_conc * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, pH2, pO2, pH2O, pCO2_cat, pCO2_an):
        """Net single-cell voltage [V]."""
        E = self.nernst_voltage(T, pH2, pO2, pH2O, pCO2_cat, pCO2_an)
        eta_act = self.activation_overpotential(j, T)
        eta_ohm = self.ohmic_overpotential(j, T)
        eta_conc = self.concentration_overpotential(j)
        V = E - eta_act - eta_ohm - eta_conc
        return max(V, 0.0)

    # ------------------------------------------------------------------
    # Thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j, pH2, pO2, pH2O, pCO2_cat, pCO2_an):
        """Temperature rate of change [K/s] -- lumped energy balance."""
        V_cell = self.cell_voltage(j, T, pH2, pO2, pH2O, pCO2_cat, pCO2_an)
        E_th = self.thermoneutral_voltage(T)
        # Heat generated by irreversibilities + entropy (per unit charge x current)
        Q_gen = self.N_cells * self.A_cell * j * max(E_th - V_cell, 0.0)
        # Heat removed by cathode-air sweep
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K,
                 pH2, pO2, pH2O, pCO2_cat, pCO2_an, dt, duration_s):
        """
        Simulate MCFC dynamics with the coupled thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2]
        T_cell_K : float
            Initial cell temperature [K]
        pH2, pO2, pH2O, pCO2_cat, pCO2_an : float
            Partial pressures [atm] (H2 anode, O2 cathode, H2O anode,
            CO2 cathode, CO2 anode)
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation duration [s]

        Returns
        -------
        dict with time-series: t, voltage, power_density, efficiency,
             temperature, overpotentials (dict of arrays)
        """
        _j = current_density_A_cm2 if callable(current_density_A_cm2) \
            else (lambda t: current_density_A_cm2)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            return [self.dTdt(T, j, pH2, pO2, pH2O, pCO2_cat, pCO2_an)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_cell_K],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
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
        E_nernst = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            E_nernst[i] = self.nernst_voltage(T, pH2, pO2, pH2O, pCO2_cat, pCO2_an)
            eta_act[i] = self.activation_overpotential(j, T)
            eta_ohm[i] = self.ohmic_overpotential(j, T)
            eta_conc[i] = self.concentration_overpotential(j)
            voltage[i] = self.cell_voltage(j, T, pH2, pO2, pH2O, pCO2_cat, pCO2_an)
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
