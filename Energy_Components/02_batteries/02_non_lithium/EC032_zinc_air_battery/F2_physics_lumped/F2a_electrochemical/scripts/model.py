"""
EC032 -- Zinc-Air Battery -- F2a Air-Cathode Electrochemical Model

Physics-lumped (0D) metal-air model with coupled electrochemical + thermal ODE.
Captures the flat discharge plateau and the hard air-electrode limiting current
imposed by O2 diffusion through the gas-diffusion layer (GDL).

Discharge cell voltage at each instant (first-principles overpotential sum):

    V_cell = E_eq(T, P_o2)                  equilibrium (Nernst) potential
             - eta_act_orr(j, T, P_o2)      cathode ORR Tafel activation
             - eta_act_zn(j, T)             anode Zn-dissolution activation
             - eta_ohm(j, T)                ohmic (KOH electrolyte + separator)
             - eta_conc(j, j_L)             O2 mass-transport at air electrode

Half reactions (alkaline):
    Cathode (air): O2 + 2 H2O + 4 e-  -> 4 OH-     (ORR, +0.40 V)
    Anode  (Zn):   Zn + 4 OH-  -> Zn(OH)4^2- + 2 e-  -> ZnO + H2O   (-1.25 V)
    Overall:       2 Zn + O2 -> 2 ZnO            E_eq ~ 1.65 V

Thermal lumped ODE (single-node energy balance):
    m*cp * dT/dt = Q_gen - Q_loss
    Q_gen  = A*j * (E_th - V_cell)            irreversibility heat (incl. entropic)
    E_th   = E_eq - T*dE/dT                   enthalpy (thermoneutral) potential
    Q_loss = hA * (T - T_amb)

State-of-charge ODE (Coulomb counting):
    dSOC/dt = -A*j / (Q_cap * 3600)          (discharge, j > 0)

Time integration via scipy.integrate.solve_ivp (RK45) on the coupled [T, SOC].

References:
    Deiss et al. (2002), Electrochim. Acta 47, 3995-4010  -- air-electrode ORR + limiting current
    Mao & White (1992), J. Electrochem. Soc. 139, 1105     -- Zn electrode kinetics in KOH
    Schroder & Krewer (2014), Electrochim. Acta 117, 541-553 -- Zn-air cell model, GDL transport
    Fu et al. (2010), J. Electrochem. Soc. 157(1), A50-A56  -- Zn-air OCV / plateau
    Parker et al. (2017), Science 356, 415-418             -- entropic coefficient, 3D Zn sponge
    Newman & Thomas-Alyea (2004), Electrochemical Systems   -- Tafel / Butler-Volmer, conc. overpotential
"""

import numpy as np
from scipy.integrate import solve_ivp


class ZincAirF2a:
    """Zinc-air cell -- air-cathode electrochemical model with thermal + SOC dynamics."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_electrode"]["value"]          # cm2
        self.E_eq0 = u["E_eq"]["value"]             # V at T_ref, P_o2_ref
        self.dE_dT = u["dE_dT"]["value"]            # V/K (negative)

        self.j0_orr = u["j0_orr"]["value"]          # A/cm2
        self.alpha_orr = u["alpha_orr"]["value"]
        self.n_orr = u["n_orr"]["value"]
        self.j0_zn = u["j0_zn"]["value"]            # A/cm2
        self.alpha_zn = u["alpha_zn"]["value"]
        self.n_zn = u["n_zn"]["value"]
        self.E_act = u["E_act"]["value"]            # J/mol

        self.j_L0 = u["j_L"]["value"]               # A/cm2 (at P_o2_ref)
        self.P_o2_ref = u["P_o2_ref"]["value"]      # atm
        self.R_ohm_area = u["R_ohm_area"]["value"]  # Ohm.cm2
        self.T_ref = u["T_ref"]["value"]            # K

        self.Q_cap = u["capacity_Ah"]["value"]      # Ah
        self.m_cell = u["m_cell"]["value"]          # kg
        self.cp_cell = u["cp_cell"]["value"]        # J/(kg.K)
        self.hA_loss = u["hA_loss"]["value"]        # W/K
        self.T_amb = u["T_amb"]["value"]            # K

    # ------------------------------------------------------------------
    # Equilibrium (Nernst) potential
    # ------------------------------------------------------------------
    def equilibrium_voltage(self, T, P_o2):
        """Equilibrium cell potential [V] with temperature + O2 Nernst correction.

        Overall 4-electron O2 reaction sets the pO2 dependence:
            E = E_eq0 + (R T)/(4 F) * ln(P_o2 / P_o2_ref) + dE/dT*(T - T_ref)
        """
        E = (self.E_eq0
             + self.dE_dT * (T - self.T_ref)
             + (self.R * T) / (self.n_orr * self.F) * np.log(P_o2 / self.P_o2_ref))
        return E

    def thermoneutral_voltage(self, T):
        """Enthalpy (thermoneutral) potential [V]: E_th = E_eq - T*dE/dT.

        Heat = A*j*(E_th - V) captures both irreversible and entropic (T*dS) heat.
        """
        return self.E_eq0 - T * self.dE_dT

    # ------------------------------------------------------------------
    # Cathode ORR activation -- Tafel
    # ------------------------------------------------------------------
    def activation_orr(self, j, T, P_o2):
        """Air-cathode oxygen-reduction activation overpotential [V] (Tafel).

        Arrhenius temperature dependence on j0; mild pO2 scaling of j0.
        """
        if j <= 0:
            return 0.0
        j0 = self.j0_orr * np.exp(
            (-self.E_act / self.R) * (1.0 / T - 1.0 / self.T_ref)
        ) * (P_o2 / self.P_o2_ref)
        j0 = max(j0, 1e-15)
        eta = (self.R * T) / (self.alpha_orr * self.n_orr * self.F) * np.log(j / j0)
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Anode Zn activation -- Tafel (fast, small)
    # ------------------------------------------------------------------
    def activation_zn(self, j, T):
        """Zn-anode dissolution activation overpotential [V] (Tafel)."""
        if j <= 0:
            return 0.0
        eta = (self.R * T) / (self.alpha_zn * self.n_zn * self.F) * np.log(j / self.j0_zn)
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Ohmic -- KOH electrolyte + separator
    # ------------------------------------------------------------------
    def ohmic_overpotential(self, j, T):
        """Ohmic loss [V]. Area-specific resistance with mild KOH conductivity T-dependence.

        KOH conductivity rises ~ exp(-Ea/R (1/T-1/Tref)); resistance is the inverse.
        """
        R_T = self.R_ohm_area * np.exp(
            (self.E_act * 0.4 / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        return j * R_T

    # ------------------------------------------------------------------
    # Concentration -- O2 mass transport through GDL (the metal-air signature)
    # ------------------------------------------------------------------
    def limiting_current(self, T, P_o2):
        """Air-electrode O2 diffusion limiting current density [A/cm2].

        j_L scales with P_o2 (Fick's law driving force) and weakly with T
        (D_O2 in gas/electrolyte increases with T).
        """
        return self.j_L0 * (P_o2 / self.P_o2_ref) * (T / self.T_ref) ** 1.5

    def concentration_overpotential(self, j, T, P_o2):
        """O2 mass-transport (concentration) overpotential [V] at the air electrode."""
        if j <= 0:
            return 0.0
        j_L = self.limiting_current(T, P_o2)
        ratio = j / j_L
        if ratio >= 1.0:
            return 50.0  # diffusion-limited: voltage collapses (cell cannot sustain j)
        return -(self.R * T) / (self.n_orr * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, P_o2):
        """Net discharge cell voltage [V] at current density j [A/cm2]."""
        E = self.equilibrium_voltage(T, P_o2)
        eta_orr = self.activation_orr(j, T, P_o2)
        eta_zn = self.activation_zn(j, T)
        eta_ohm = self.ohmic_overpotential(j, T)
        eta_conc = self.concentration_overpotential(j, T, P_o2)
        V = E - eta_orr - eta_zn - eta_ohm - eta_conc
        return max(V, 0.0)

    # ------------------------------------------------------------------
    # Coupled ODE derivatives
    # ------------------------------------------------------------------
    def dTdt(self, T, j, P_o2):
        """Temperature rate of change [K/s]."""
        V = self.cell_voltage(j, T, P_o2)
        E_th = self.thermoneutral_voltage(T)
        Q_gen = self.A * j * (E_th - V)        # W (irreversible + entropic)
        Q_loss = self.hA_loss * (T - self.T_amb)
        return (Q_gen - Q_loss) / (self.m_cell * self.cp_cell)

    def dSOCdt(self, j):
        """State-of-charge rate [1/s] from Coulomb counting (j>0 = discharge)."""
        I = self.A * j                          # A
        return -I / (self.Q_cap * 3600.0)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, P_o2_atm,
                 dt, duration_s, soc0=1.0):
        """
        Simulate Zn-air discharge with coupled thermal + SOC ODEs.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)  -- discharge current density
        T_cell_K  : float  -- initial cell temperature [K]
        P_o2_atm  : float  -- O2 partial pressure [atm] (0.21 = ambient air)
        dt        : float  -- output time step [s]
        duration_s: float  -- total duration [s]
        soc0      : float  -- initial state of charge (0-1)

        Returns
        -------
        dict of time-series arrays: t, voltage, power_density, efficiency,
            temperature, soc, overpotentials{...}
        """
        _j = (current_density_A_cm2 if callable(current_density_A_cm2)
              else (lambda t: current_density_A_cm2))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T, soc = y[0], y[1]
            j = _j(t)
            if soc <= 0.0:
                j = 0.0
            return [self.dTdt(T, j, P_o2_atm), self.dSOCdt(j)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_cell_K, soc0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        soc_out = np.clip(sol.y[1], 0.0, 1.0)
        N = len(t_out)

        voltage = np.zeros(N)
        power_density = np.zeros(N)
        efficiency = np.zeros(N)
        eta_orr = np.zeros(N)
        eta_zn = np.zeros(N)
        eta_ohm = np.zeros(N)
        eta_conc = np.zeros(N)
        E_eq = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            if soc_out[i] <= 0.0:
                j = 0.0
            T = T_out[i]
            E_eq[i] = self.equilibrium_voltage(T, P_o2_atm)
            eta_orr[i] = self.activation_orr(j, T, P_o2_atm)
            eta_zn[i] = self.activation_zn(j, T)
            eta_ohm[i] = self.ohmic_overpotential(j, T)
            eta_conc[i] = self.concentration_overpotential(j, T, P_o2_atm)
            voltage[i] = self.cell_voltage(j, T, P_o2_atm)
            power_density[i] = j * voltage[i]
            E_th = self.thermoneutral_voltage(T)
            efficiency[i] = voltage[i] / E_th if E_th > 0 else 0.0

        return {
            "t": t_out,
            "voltage": voltage,
            "power_density": power_density,
            "efficiency": efficiency,
            "temperature": T_out,
            "soc": soc_out,
            "overpotentials": {
                "E_eq": E_eq,
                "activation_orr": eta_orr,
                "activation_zn": eta_zn,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }

    # ------------------------------------------------------------------
    # Steady polarization curve helper
    # ------------------------------------------------------------------
    def polarization_curve(self, j_array, T, P_o2):
        """Return V(j) [V] over an array of current densities (steady, fixed T)."""
        return np.array([self.cell_voltage(j, T, P_o2) for j in j_array])
