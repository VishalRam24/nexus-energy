"""
EC011 -- Anion Exchange Membrane (AEM) Electrolyser -- F2a Full Electrochemical Model

Physics-lumped first-principles model with coupled electrochemical + thermal ODE.
Unlike a fuel cell, this is an ELECTROLYSIS device: electrical work is supplied to
split water, so the operating cell voltage is ABOVE the reversible potential:

    V_cell = E_rev(T)
             + eta_act_a(j, T)         anode OER Tafel kinetics (alkaline)
             + eta_act_c(j, T)         cathode HER Tafel kinetics (alkaline)
             + eta_ohm(j, T)           AEM membrane (OH- transport) + KOH electrolyte
             + eta_conc(j, j_L)        mass-transport / bubble limitation

AEM-specific physics
--------------------
* Charge carrier is the hydroxide ion OH- (not H+ as in PEM).  The anode reaction
  is  4 OH-  ->  O2 + 2 H2O + 4 e-  (OER) and the cathode reaction is
  2 H2O + 2 e-  ->  H2 + 2 OH-  (HER), so water is consumed at the cathode and OH-
  shuttles through the membrane.  Net:  2 H2O -> 2 H2 + O2.
* Alkaline (high-pH) environment enables NON-NOBLE Ni / NiFe catalysts, captured
  through the (lower) OER exchange current density and OER activation energy.
* AEM ionic (OH-) conductivity is lower than Nafion's proton conductivity, giving a
  larger membrane ohmic term; conductivity is Arrhenius in temperature.

Faraday's law (H2 production, with current-dependent Faradaic efficiency):

    n_H2 = eta_F(j,T) * N_cells * I / (2 F)         [mol/s]

The Faradaic efficiency penalty captures gas crossover / parasitic currents in the
alkaline cell (Ulleberg 2003 empirical form).

Lumped thermal ODE (energy balance on the stack thermal mass):

    m*cp dT/dt = Q_gen - Q_cool
    Q_gen  = N_cells * A_cell * j * (V_cell - E_tn(T))   (>0: irreversibilities heat the stack)
    Q_cool = hA * (T - T_coolant)

E_tn is the thermoneutral voltage (enthalpy basis): operating above E_tn releases
heat, below E_tn the cell would absorb heat.  Energy conservation:  the electrical
power supplied equals the chemical (HHV) energy stored in H2 plus the heat losses.

References
----------
Vincent & Bessarabov (2018), Renew. Sustain. Energy Rev. 81, 1690-1704.
Henkensmeier et al. (2021), J. Electrochem. Energy Conv. Storage 18, 024001.
Liu et al. (2022), J. Power Sources 524, 231087 (AEM cell modelling).
Ulleberg (2003), Int. J. Hydrogen Energy 28, 21-33 (lumped thermal + Faraday eff.).
"""

import numpy as np
from scipy.integrate import solve_ivp


class AEM_F2a:
    """AEM water electrolyser -- full electrochemical model with thermal dynamics."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 2              # electrons per H2

    # Reference conductivity temperature (Arrhenius reference for membrane)
    T_sigma_ref = 333.15  # K (consistent with sigma_mem_ref at 60 C)

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]                 # cm2
        self.t_mem = u["t_mem"]["value"]                   # cm
        self.sigma_ref = u["sigma_mem_ref"]["value"]       # S/cm at T_ref
        self.Ea_sigma = u["Ea_sigma"]["value"]             # J/mol
        self.r_elec = u["r_electrolyte"]["value"]          # Ohm.cm2
        self.j0_a_ref = u["j0_anode_ref"]["value"]         # A/cm2
        self.j0_c_ref = u["j0_cathode_ref"]["value"]       # A/cm2
        self.Ea_a = u["Ea_anode"]["value"]                 # J/mol
        self.Ea_c = u["Ea_cathode"]["value"]               # J/mol
        self.alpha_a = u["alpha_anode"]["value"]
        self.alpha_c = u["alpha_cathode"]["value"]
        self.j_L = u["j_L"]["value"]                       # A/cm2
        self.T_ref = u["T_ref"]["value"]                   # K
        self.eta_F_max = u["eta_F_max"]["value"]
        self.f1 = u["f1_cross"]["value"]                   # (mA/cm2)^2
        self.f2 = u["f2_cross"]["value"]
        self.m_stack = u["m_stack"]["value"]               # kg
        self.cp_stack = u["cp_stack"]["value"]             # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]               # W/K
        self.T_coolant = u["T_coolant"]["value"]           # K

    # ------------------------------------------------------------------
    # Reversible (thermodynamic) voltage with Nernst pressure correction
    # ------------------------------------------------------------------
    def reversible_voltage(self, T, P_h2=1.0, P_o2=1.0):
        """Reversible cell voltage E_rev [V] for water splitting.

        E_rev(T) decreases with temperature (entropy of reaction); the Nernst
        term RAISES the required voltage when product gases are pressurised
        (P in bar, liquid water activity = 1).
        """
        # Temperature dependence of standard reversible potential
        # (Le Roy / NIST fit, ~ -0.9 mV/K near 25 C)
        E0 = 1.229 - 0.000846 * (T - 298.15)
        # Nernst: 2 H2O(l) -> 2 H2 + O2 ; per-cell form for products only
        nernst = (self.R * T) / (self.n * self.F) * np.log(P_h2 * np.sqrt(P_o2))
        return E0 + nernst

    # ------------------------------------------------------------------
    # Thermoneutral voltage (enthalpy basis) -- for heat generation
    # ------------------------------------------------------------------
    def thermoneutral_voltage(self, T):
        """Thermoneutral voltage E_tn [V] (HHV enthalpy basis ~1.481 V at 25 C)."""
        return 1.481 - 0.000226 * (T - 298.15)

    # ------------------------------------------------------------------
    # Activation overpotentials -- Tafel kinetics (Arrhenius j0)
    # ------------------------------------------------------------------
    def _j0(self, j0_ref, Ea, T):
        j0 = j0_ref * np.exp((-Ea / self.R) * (1.0 / T - 1.0 / self.T_ref))
        return max(j0, 1e-15)

    def activation_anode(self, j, T):
        """Anode (OER, non-noble Ni catalyst) activation overpotential [V]."""
        if j <= 0:
            return 0.0
        j0 = self._j0(self.j0_a_ref, self.Ea_a, T)
        eta = (self.R * T) / (self.alpha_a * self.n * self.F) * np.log(j / j0)
        return max(eta, 0.0)

    def activation_cathode(self, j, T):
        """Cathode (HER) activation overpotential [V]."""
        if j <= 0:
            return 0.0
        j0 = self._j0(self.j0_c_ref, self.Ea_c, T)
        eta = (self.R * T) / (self.alpha_c * self.n * self.F) * np.log(j / j0)
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Ohmic overpotential -- AEM (OH-) membrane + KOH electrolyte
    # ------------------------------------------------------------------
    def membrane_conductivity(self, T):
        """AEM hydroxide (OH-) conductivity [S/cm], Arrhenius in temperature."""
        sigma = self.sigma_ref * np.exp(
            (-self.Ea_sigma / self.R) * (1.0 / T - 1.0 / self.T_sigma_ref)
        )
        return max(sigma, 1e-6)

    def ohmic_overpotential(self, j, T):
        """Ohmic loss [V]: membrane OH- transport + electrolyte/bubble ASR."""
        sigma = self.membrane_conductivity(T)
        asr = self.t_mem / sigma + self.r_elec   # Ohm.cm2
        return j * asr

    # ------------------------------------------------------------------
    # Concentration overpotential -- mass transport / bubble limit
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, T=333.15):
        """Concentration (mass-transport) overpotential [V]."""
        if j <= 0:
            return 0.0
        ratio = j / self.j_L
        if ratio >= 1.0:
            return 10.0  # bubble-blocked / OH- depletion -- effectively infinite
        return (self.R * T) / (self.n * self.F) * np.log(1.0 / (1.0 - ratio))

    # ------------------------------------------------------------------
    # Faradaic efficiency (Ulleberg 2003 empirical crossover penalty)
    # ------------------------------------------------------------------
    def faradaic_efficiency(self, j):
        """Faradaic efficiency in (0, 1]; drops at low current density."""
        j_mA = max(j, 0.0) * 1000.0  # mA/cm2
        eff = (j_mA ** 2) / (self.f1 + j_mA ** 2) * self.f2
        return float(np.clip(min(eff, self.eta_F_max), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Cell voltage  (V_cell = E_rev + sum of overpotentials, V_cell > E_rev)
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, P_h2=1.0, P_o2=1.0):
        """Net single-cell operating voltage [V] (electrolysis: above E_rev)."""
        E = self.reversible_voltage(T, P_h2, P_o2)
        eta = (self.activation_anode(j, T)
               + self.activation_cathode(j, T)
               + self.ohmic_overpotential(j, T)
               + self.concentration_overpotential(j, T))
        return E + eta

    # ------------------------------------------------------------------
    # Hydrogen production (Faraday's law)
    # ------------------------------------------------------------------
    def hydrogen_rate(self, j, T=333.15):
        """H2 production rate [mol/s] for the whole stack."""
        I = max(j, 0.0) * self.A_cell        # A per cell
        return self.faradaic_efficiency(j) * self.N_cells * I / (self.n * self.F)

    # ------------------------------------------------------------------
    # Thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j, P_h2=1.0, P_o2=1.0):
        """Temperature rate of change [K/s]."""
        V_cell = self.cell_voltage(j, T, P_h2, P_o2)
        E_tn = self.thermoneutral_voltage(T)
        # Heat from operating above thermoneutral voltage
        Q_gen = self.N_cells * self.A_cell * j * (V_cell - E_tn)   # W
        Q_cool = self.hA_cool * (T - self.T_coolant)               # W
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation (coupled electrochemistry + thermal ODE)
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, P_h2_bar, P_o2_bar,
                 dt, duration_s):
        """
        Simulate AEM electrolyser dynamics with coupled thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)   operating current density
        T_cell_K  : float    initial stack temperature [K]
        P_h2_bar  : float    H2 product pressure [bar]
        P_o2_bar  : float    O2 product pressure [bar]
        dt        : float    output time step [s]
        duration_s: float    total simulation duration [s]

        Returns
        -------
        dict of time series: t, voltage (cell), stack_voltage, power_kW,
            h2_rate_mol_s, efficiency (HHV), faradaic_eff, temperature,
            overpotentials (dict of arrays incl. E_rev, anode, cathode,
            ohmic, concentration).
        """
        _j = current_density_A_cm2 if callable(current_density_A_cm2) \
            else (lambda t: current_density_A_cm2)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], _j(t), P_h2_bar, P_o2_bar)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_cell_K],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        HHV = 285800.0  # J/mol H2 (higher heating value)

        voltage = np.zeros(N)
        stack_voltage = np.zeros(N)
        power_kW = np.zeros(N)
        h2_rate = np.zeros(N)
        efficiency = np.zeros(N)
        far_eff = np.zeros(N)
        E_rev = np.zeros(N)
        eta_a = np.zeros(N)
        eta_c = np.zeros(N)
        eta_ohm = np.zeros(N)
        eta_conc = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            E_rev[i] = self.reversible_voltage(T, P_h2_bar, P_o2_bar)
            eta_a[i] = self.activation_anode(j, T)
            eta_c[i] = self.activation_cathode(j, T)
            eta_ohm[i] = self.ohmic_overpotential(j, T)
            eta_conc[i] = self.concentration_overpotential(j, T)
            voltage[i] = self.cell_voltage(j, T, P_h2_bar, P_o2_bar)
            stack_voltage[i] = self.N_cells * voltage[i]
            I = j * self.A_cell                                  # A
            power_kW[i] = stack_voltage[i] * I / 1000.0
            far_eff[i] = self.faradaic_efficiency(j)
            h2_rate[i] = self.hydrogen_rate(j, T)
            p_el = power_kW[i] * 1000.0
            efficiency[i] = (h2_rate[i] * HHV / p_el) if p_el > 0 else 0.0

        return {
            "t": t_out,
            "voltage": voltage,
            "stack_voltage": stack_voltage,
            "power_kW": power_kW,
            "h2_rate_mol_s": h2_rate,
            "efficiency": efficiency,
            "faradaic_eff": far_eff,
            "temperature": T_out,
            "overpotentials": {
                "E_rev": E_rev,
                "anode": eta_a,
                "cathode": eta_c,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }
