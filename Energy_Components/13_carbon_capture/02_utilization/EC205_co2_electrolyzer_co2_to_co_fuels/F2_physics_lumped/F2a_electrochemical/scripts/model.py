"""
EC205 -- CO2 Electrolyzer (CO2 -> CO / fuels) -- F2a Full Electrochemical Model

Physics-lumped model of a CO2 reduction (CO2RR) electrolysis cell with a coupled
electrochemical + thermal ODE.  A gas-diffusion-electrode (GDE) / membrane-electrode
assembly (MEA) cell drives the cathodic CO2 reduction reaction against the anodic
oxygen evolution reaction (OER):

    Cathode (CO2RR):  CO2 + 2 H+ + 2 e-  -> CO + H2O      E0 = -0.11 V vs SHE
    Cathode (HER):    2 H+ + 2 e-        -> H2            E0 =  0.00 V vs SHE  (parasitic)
    Anode  (OER):     2 H2O              -> O2 + 4 H+ + 4 e-   E0 = +1.23 V vs SHE
    --------------------------------------------------------------------------
    Net:   CO2 -> CO + 1/2 O2     E_rev = E_anode - E_cathode ~= 1.34 V

Cell voltage (driven cell, V > E_rev; all overpotentials ADD to the applied voltage):

    V_cell = E_rev
             + eta_act_cathode(j, T)     Tafel kinetics (CO2RR, sluggish)
             + eta_act_anode(j, T)       Tafel kinetics (OER)
             + eta_ohm(j, T)             membrane + electrolyte ohmic drop
             + eta_conc(j, j_L)          CO2 mass-transport limit at the GDE

Faradaic efficiency toward CO (vs. parasitic H2 evolution).  At low/moderate j the
CO2RR dominates; as j approaches the CO2 mass-transport limit (j_L) the local CO2
depletes and the H2 evolution reaction (HER) takes an increasing share of the current,
so FE_CO falls and FE_H2 = 1 - FE_CO rises.  Modelled with a smooth logistic roll-off
in (j / j_L) (Endrodi 2017; Burdyny & Sinton 2019):

    FE_CO(j) = FE_max / (1 + exp(k * (j/j_L - x0)))     0 < FE_CO < 1

Product formation rate via Faraday's law (mol/s, per cell area A, N_cells in series):

    n_dot_CO = FE_CO * I / (n_e * F),     I = j * A * N_cells

Energy per mole of CO:

    E_per_mol_CO = (n_e * F * V_cell) / FE_CO        [J/mol]

Lumped thermal ODE (single thermal node, stack mass m, heat capacity cp):

    m*cp dT/dt = Q_gen - Q_cool
    Q_gen = I_total * (V_cell - E_tn)        ohmic + kinetic dissipation (E_tn thermoneutral)
    Q_cool = hA * (T - T_coolant)

Integrated with scipy.integrate.solve_ivp (RK45).

References:
    Endrodi, B. et al. (2017). Continuous-flow electroreduction of CO2.
        Prog. Energy Combust. Sci. 62, 133-154.
    Burdyny, T. & Smith, W. A. (2019). CO2 reduction on gas-diffusion electrodes
        and why catalytic performance must be assessed at commercially-relevant
        conditions. Energy Environ. Sci. 12, 1442-1453.
    Weng, L.-C., Bell, A. T. & Weber, A. Z. (2018). Modeling gas-diffusion
        electrodes for CO2 reduction. Phys. Chem. Chem. Phys. 20, 16973-16984.
    Jouny, M., Luc, W. & Jiao, F. (2018). General techno-economic analysis of CO2
        electrolysis systems. Ind. Eng. Chem. Res. 57(6), 2165-2177.
"""

import numpy as np
from scipy.integrate import solve_ivp


class CO2Electrolyzer_F2a:
    """CO2 electrolyzer -- full electrochemical model with thermal dynamics."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]                 # cm2
        self.n_e = u["n_electrons"]["value"]               # electrons per CO
        self.E_rev = u["E_rev"]["value"]                   # V (reversible cell voltage)
        self.E_tn = u["E_tn"]["value"]                     # V (thermoneutral voltage)

        # Cathode (CO2RR) Tafel kinetics
        self.alpha_c = u["alpha_cathode"]["value"]
        self.j0_c_ref = u["j0_cathode_ref"]["value"]       # A/cm2
        self.E_act_c = u["E_act_cathode"]["value"]         # J/mol
        # Anode (OER) Tafel kinetics
        self.alpha_a = u["alpha_anode"]["value"]
        self.j0_a_ref = u["j0_anode_ref"]["value"]         # A/cm2
        self.E_act_a = u["E_act_anode"]["value"]           # J/mol
        self.T_ref = u["T_ref"]["value"]                   # K

        # Ohmic
        self.R_area = u["R_area_ohm_cm2"]["value"]         # ohm.cm2 (area-specific resistance)

        # Mass transport / Faradaic efficiency
        self.j_L = u["j_L"]["value"]                       # A/cm2 (CO2 transport limit)
        self.FE_max = u["FE_max"]["value"]                 # peak Faradaic efficiency to CO
        self.fe_k = u["FE_rolloff_k"]["value"]             # logistic steepness
        self.fe_x0 = u["FE_rolloff_x0"]["value"]           # logistic midpoint in (j/j_L)

        # Thermal
        self.m_stack = u["m_stack"]["value"]               # kg
        self.cp_stack = u["cp_stack"]["value"]             # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]               # W/K
        self.T_coolant = u["T_coolant"]["value"]           # K

        # Molar masses (g/mol) for mass-rate reporting
        self.M_CO = 28.01
        self.M_CO2 = 44.01

    # ------------------------------------------------------------------
    # Tafel activation overpotentials (positive losses, add to V_cell)
    # ------------------------------------------------------------------
    def _tafel(self, j, T, j0, alpha):
        """Generic Tafel overpotential [V] (>=0) for a half-cell."""
        if j <= 0:
            return 0.0
        j0 = max(j0, 1e-14)
        eta = (self.R * T) / (alpha * self.F) * np.log(j / j0)
        return max(eta, 0.0)

    def activation_cathode(self, j, T):
        """Cathodic CO2RR activation overpotential [V]."""
        j0 = self.j0_c_ref * np.exp(
            (-self.E_act_c / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        return self._tafel(j, T, j0, self.alpha_c)

    def activation_anode(self, j, T):
        """Anodic OER activation overpotential [V]."""
        j0 = self.j0_a_ref * np.exp(
            (-self.E_act_a / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        return self._tafel(j, T, j0, self.alpha_a)

    # ------------------------------------------------------------------
    # Ohmic overpotential (membrane + electrolyte, area-specific R)
    # ------------------------------------------------------------------
    def ohmic_overpotential(self, j, T):
        """Ohmic loss [V].  ASR decreases mildly with T (ionic conduction)."""
        # Arrhenius-like conductivity increase => resistance falls ~2%/10K
        R_area = self.R_area * np.exp(1500.0 * (1.0 / T - 1.0 / self.T_ref))
        return j * R_area

    # ------------------------------------------------------------------
    # Concentration / mass-transport overpotential at the GDE
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, j_L=None, T=298.15):
        """CO2 mass-transport overpotential [V] (diverges as j -> j_L)."""
        if j_L is None:
            j_L = self.j_L
        if j <= 0:
            return 0.0
        ratio = j / j_L
        if ratio >= 1.0:
            return 10.0  # CO2-starved GDE -- effectively unbounded
        return -(self.R * T) / (self.n_e * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell voltage (driven: losses ADD to the reversible voltage)
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T):
        """Net single-cell applied voltage [V] (V > E_rev for j > 0)."""
        eta_c = self.activation_cathode(j, T)
        eta_a = self.activation_anode(j, T)
        eta_ohm = self.ohmic_overpotential(j, T)
        eta_conc = self.concentration_overpotential(j, T=T)
        return self.E_rev + eta_c + eta_a + eta_ohm + eta_conc

    # ------------------------------------------------------------------
    # Faradaic efficiency toward CO (vs. parasitic H2)
    # ------------------------------------------------------------------
    def faradaic_efficiency(self, j, j_L=None):
        """FE toward CO in (0, FE_max).  Falls as j -> j_L (HER takes over)."""
        if j_L is None:
            j_L = self.j_L
        if j <= 0:
            return 0.0
        ratio = j / j_L
        fe = self.FE_max / (1.0 + np.exp(self.fe_k * (ratio - self.fe_x0)))
        return float(np.clip(fe, 1e-6, self.FE_max))

    def fe_h2(self, j, j_L=None):
        """Parasitic FE toward H2 evolution (= 1 - FE_CO)."""
        return 1.0 - self.faradaic_efficiency(j, j_L)

    # ------------------------------------------------------------------
    # Product formation rate via Faraday's law
    # ------------------------------------------------------------------
    def co_production_rate(self, j, T=None):
        """CO production rate [mol/s] for the whole stack at current density j."""
        I_total = j * self.A_cell * self.N_cells       # A
        fe = self.faradaic_efficiency(j)
        return fe * I_total / (self.n_e * self.F)

    def co_mass_rate(self, j):
        """CO production rate [kg/h] for the whole stack."""
        return self.co_production_rate(j) * self.M_CO / 1000.0 * 3600.0

    def energy_per_mol_CO(self, j, T):
        """Electrical energy per mole CO produced [J/mol] = n_e F V / FE."""
        V = self.cell_voltage(j, T)
        fe = self.faradaic_efficiency(j)
        return self.n_e * self.F * V / fe

    def energy_per_kg_CO_kWh(self, j, T):
        """Specific energy consumption [kWh/kg CO]."""
        J_per_mol = self.energy_per_mol_CO(j, T)
        J_per_kg = J_per_mol / (self.M_CO / 1000.0)
        return J_per_kg / 3.6e6

    # ------------------------------------------------------------------
    # Thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j):
        """Temperature rate of change [K/s] (lumped single-node)."""
        V_cell = self.cell_voltage(j, T)
        I_total = j * self.A_cell * self.N_cells           # A
        # Heat dissipated = I * (V_applied - V_thermoneutral) per cell, summed.
        # V_cell is per-cell; multiply by N_cells * (j*A) current.
        Q_gen = self.N_cells * (j * self.A_cell) * (V_cell - self.E_tn)
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_cool) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation (coupled thermal ODE)
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, dt, duration_s):
        """
        Simulate CO2 electrolyzer dynamics with the coupled thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2].
        T_cell_K : float
            Initial cell temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation duration [s].

        Returns
        -------
        dict of time-series arrays: t, voltage, power_density, temperature,
            faradaic_efficiency, co_rate_mol_s, energy_per_mol_CO, sec_kWh_kg,
            overpotentials (dict).
        """
        _j = (current_density_A_cm2 if callable(current_density_A_cm2)
              else (lambda t: current_density_A_cm2))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            return [self.dTdt(T, j)]

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
        fe = np.zeros(N)
        co_rate = np.zeros(N)
        e_per_mol = np.zeros(N)
        sec = np.zeros(N)
        eta_c = np.zeros(N)
        eta_a = np.zeros(N)
        eta_ohm = np.zeros(N)
        eta_conc = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            eta_c[i] = self.activation_cathode(j, T)
            eta_a[i] = self.activation_anode(j, T)
            eta_ohm[i] = self.ohmic_overpotential(j, T)
            eta_conc[i] = self.concentration_overpotential(j, T=T)
            voltage[i] = self.cell_voltage(j, T)
            power_density[i] = j * voltage[i]
            fe[i] = self.faradaic_efficiency(j)
            co_rate[i] = self.co_production_rate(j)
            e_per_mol[i] = self.energy_per_mol_CO(j, T)
            sec[i] = self.energy_per_kg_CO_kWh(j, T)

        return {
            "t": t_out,
            "voltage": voltage,
            "power_density": power_density,
            "temperature": T_out,
            "faradaic_efficiency": fe,
            "co_rate_mol_s": co_rate,
            "energy_per_mol_CO": e_per_mol,
            "sec_kWh_kg": sec,
            "overpotentials": {
                "E_rev": np.full(N, self.E_rev),
                "activation_cathode": eta_c,
                "activation_anode": eta_a,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }
