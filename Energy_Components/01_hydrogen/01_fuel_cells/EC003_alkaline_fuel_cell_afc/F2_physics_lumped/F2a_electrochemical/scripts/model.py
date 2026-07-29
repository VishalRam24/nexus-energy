"""
EC003 -- Alkaline Fuel Cell (AFC) -- F2a Full Electrochemical Model

Physics-lumped 0D model with coupled electrochemical voltage + lumped thermal ODE.
First-principles overpotentials with a *KOH-electrolyte* ohmic model -- this is the
defining difference from the PEMFC F2a (EC001), which uses a Springer/Nafion membrane.

Reaction (alkaline):
    Anode:    H2  + 2 OH-  ->  2 H2O + 2 e-
    Cathode:  1/2 O2 + H2O + 2 e-  ->  2 OH-
    Overall:  H2 + 1/2 O2 -> H2O

Voltage model (first-principles at each instant):
    V_cell = E_nernst(T, P_h2, P_o2)
             - eta_act(j, T)        Butler-Volmer / Tafel kinetics (ORR, alkaline)
             - eta_ohm(j, T, c_KOH) KOH electrolyte ionic conduction (Gilliam 2007)
             - eta_conc(j, j_L)     mass-transport limiting (liquid electrolyte)

Lumped thermal ODE (first-order, single stack node):
    m*cp * dT/dt = Q_gen - Q_cool
    Q_gen  = N_cells * A_cell * j * (E_tn(T) - V_cell)   (irreversible + entropic heat)
    Q_cool = hA * (T - T_coolant)

KOH ionic conductivity (Gilliam et al. 2007, Int. J. Hydrogen Energy 32(3):359-364),
an empirical fit valid over 0-12 mol/L and 273-373 K:

    kappa(c,T) [S/cm] = -2.041 c - 0.0028 c^2 + 0.005332 c T
                        + 207.2 c/T + 0.001043 c^3 - 0.0000003 c^2 T^2

The electrolyte ohmic loss for an inter-electrode gap d is then
    eta_ohm = j * d / kappa.

References:
    Larminie & Dicks (2003), Fuel Cell Systems Explained, 2nd Ed., Wiley.
    Appleby & Foulkes (1989), Fuel Cell Handbook, Van Nostrand Reinhold.
    Gilliam, Graydon, Kirk, Thorpe (2007), Int. J. Hydrogen Energy 32(3), 359-364.
    Kordesch & Simader (1996), Fuel Cells and Their Applications, VCH.
"""

import numpy as np
from scipy.integrate import solve_ivp


class AFC_F2a:
    """Alkaline Fuel Cell -- full electrochemical model with KOH ohmics + thermal ODE."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 2              # electrons per H2

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]                # cm2
        self.d_elec = u["d_electrolyte"]["value"]          # cm  (inter-electrode gap)
        self.c_KOH = u["c_KOH"]["value"]                   # mol/L
        self.j_L = u["j_L"]["value"]                       # A/cm2
        self.alpha_c = u["alpha_cathode"]["value"]
        self.j0_ref = u["j0_cathode_ref"]["value"]         # A/cm2
        self.E_act = u["E_act_cathode"]["value"]           # J/mol
        self.T_ref = u["T_ref"]["value"]                   # K
        self.m_stack = u["m_stack"]["value"]               # kg
        self.cp_stack = u["cp_stack"]["value"]             # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]               # W/K
        self.T_coolant = u["T_coolant"]["value"]           # K

    # ------------------------------------------------------------------
    # Nernst (thermodynamic) reversible voltage
    # ------------------------------------------------------------------
    def nernst_voltage(self, T, P_h2, P_o2):
        """Reversible open-circuit voltage [V] (Nernst, liquid-water product)."""
        # Standard reversible potential with temperature correction (dE/dT = -0.846 mV/K)
        E0 = 1.229 - 0.000846 * (T - 298.15)
        # Pressure correction: activity of liquid water taken as 1
        E = E0 + (self.R * T) / (2.0 * self.F) * np.log(P_h2 * np.sqrt(P_o2))
        return E

    # ------------------------------------------------------------------
    # Thermoneutral (enthalpy) voltage -- for heat generation
    # ------------------------------------------------------------------
    def thermoneutral_voltage(self, T):
        """Thermoneutral voltage [V] (HHV basis, liquid water)."""
        return 1.481 - 0.000226 * (T - 298.15)

    # ------------------------------------------------------------------
    # Activation overpotential -- Tafel kinetics (cathode ORR dominates)
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
        if j <= j0:
            return 0.0
        eta_act = (self.R * T) / (self.alpha_c * self.n * self.F) * np.log(j / j0)
        return max(eta_act, 0.0)

    # ------------------------------------------------------------------
    # KOH electrolyte conductivity -- Gilliam et al. (2007)
    # ------------------------------------------------------------------
    def koh_conductivity(self, T, c=None):
        """Specific ionic conductivity of KOH solution [S/cm].

        Empirical correlation of Gilliam et al. (2007), valid 0-12 mol/L,
        273-373 K. Conductivity peaks near ~6-7 mol/L at AFC temperatures.
        """
        if c is None:
            c = self.c_KOH
        kappa = (
            -2.041 * c
            - 0.0028 * c ** 2
            + 0.005332 * c * T
            + 207.2 * c / T
            + 0.001043 * c ** 3
            - 0.0000003 * c ** 2 * T ** 2
        )
        return max(kappa, 1e-4)

    def ohmic_overpotential(self, j, T, c=None):
        """Ohmic loss [V] across the KOH electrolyte gap."""
        kappa = self.koh_conductivity(T, c)          # S/cm
        R_area = self.d_elec / kappa                  # ohm.cm2
        return j * R_area

    # ------------------------------------------------------------------
    # Concentration overpotential -- mass transport in liquid electrolyte
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, j_L=None, T=343.15):
        """Concentration (mass-transport) overpotential [V]."""
        if j_L is None:
            j_L = self.j_L
        if j <= 0:
            return 0.0
        ratio = j / j_L
        if ratio >= 1.0:
            return 10.0  # effectively infinite -- electrode starved/flooded
        return -(self.R * T) / (self.n * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, P_h2, P_o2, c=None):
        """Net single-cell voltage [V]."""
        E = self.nernst_voltage(T, P_h2, P_o2)
        eta_act = self.activation_overpotential(j, T)
        eta_ohm = self.ohmic_overpotential(j, T, c)
        eta_conc = self.concentration_overpotential(j, T=T)
        V = E - eta_act - eta_ohm - eta_conc
        return max(V, 0.0)

    # ------------------------------------------------------------------
    # Lumped thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j, P_h2, P_o2, c=None):
        """Temperature rate of change [K/s]."""
        V_cell = self.cell_voltage(j, T, P_h2, P_o2, c)
        E_th = self.thermoneutral_voltage(T)
        # Heat generated by irreversibilities + reaction entropy (E_tn - V_cell >= 0)
        Q_gen = self.N_cells * self.A_cell * j * (E_th - V_cell)
        # Heat removed by coolant / circulating electrolyte loop
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, P_h2_atm, P_o2_atm,
                 dt, duration_s, c_KOH=None):
        """
        Simulate AFC dynamics with coupled lumped thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2]
        T_cell_K : float
            Initial cell temperature [K]
        P_h2_atm, P_o2_atm : float
            Reactant partial pressures [atm]
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation duration [s]
        c_KOH : float, optional
            KOH concentration [mol/L]; defaults to parameter value.

        Returns
        -------
        dict with time-series: t, voltage, power_density, efficiency,
             temperature, koh_conductivity, overpotentials (dict of arrays)
        """
        _j = current_density_A_cm2 if callable(current_density_A_cm2) \
            else (lambda t: current_density_A_cm2)
        c = c_KOH if c_KOH is not None else self.c_KOH

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            return [self.dTdt(T, j, P_h2_atm, P_o2_atm, c)]

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
        E_nernst = np.zeros(N)
        kappa = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            E_nernst[i] = self.nernst_voltage(T, P_h2_atm, P_o2_atm)
            eta_act[i] = self.activation_overpotential(j, T)
            eta_ohm[i] = self.ohmic_overpotential(j, T, c)
            eta_conc[i] = self.concentration_overpotential(j, T=T)
            voltage[i] = self.cell_voltage(j, T, P_h2_atm, P_o2_atm, c)
            power_density[i] = j * voltage[i]
            kappa[i] = self.koh_conductivity(T, c)
            E_th = self.thermoneutral_voltage(T)
            efficiency[i] = voltage[i] / E_th if E_th > 0 else 0.0

        return {
            "t": t_out,
            "voltage": voltage,
            "power_density": power_density,
            "efficiency": efficiency,
            "temperature": T_out,
            "koh_conductivity": kappa,
            "overpotentials": {
                "E_nernst": E_nernst,
                "activation": eta_act,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }
