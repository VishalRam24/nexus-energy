"""
EC006 -- Direct Methanol Fuel Cell (DMFC) -- F2a Full Electrochemical Model

Physics-lumped (0D) first-principles model with coupled electrochemical losses,
methanol crossover (mixed-potential), and a lumped thermal ODE.

Overall reaction:
    Anode:   CH3OH + H2O  ->  CO2 + 6 H+ + 6 e-      (n = 6)
    Cathode: 3/2 O2 + 6 H+ + 6 e-  ->  3 H2O
    Net:     CH3OH + 3/2 O2  ->  CO2 + 2 H2O,   E_rev ~ 1.21 V

Key DMFC physics captured (distinct from a PEMFC):
  1. Methanol crossover. Unreacted methanol permeates the Nafion membrane and is
     parasitically oxidised at the cathode. This (a) wastes fuel and (b) sets up a
     MIXED POTENTIAL at the cathode that depresses the open-circuit voltage from the
     thermodynamic ~1.21 V down to the measured ~0.5-0.7 V. We model the crossover as
     an equivalent leakage current density j_cross(T, c) that adds to the cathodic
     kinetic load, and reduces the fuel/current efficiency.
        Kulikovsky (2002), Nordlund & Lindbergh (2002).
  2. Sluggish anode methanol-oxidation kinetics on Pt-Ru (large eta_act, Tafel/BV).
        Scott et al. (1999).
  3. Cathode oxygen-reduction kinetics (Tafel).
  4. Ohmic loss through the proton-conducting membrane (Springer 1991 conductivity).
  5. Methanol mass-transport (concentration) limiting current j_L.

Cell voltage (first-principles, evaluated each time step):
    V_cell = E_rev(T)
             - eta_act_anode(j + j_cross, T, c)   anode Butler-Volmer / Tafel
             - eta_mix_cathode(j + j_cross, T)     cathode ORR incl. crossover load
             - eta_ohm(j, T)                        membrane ohmic
             - eta_conc(j, j_L)                     methanol transport limit
The cathode kinetic load carries (j + j_cross): the catalyst must drive both the
useful proton current j and the parasitic crossover oxidation j_cross. At j -> 0 the
cathode still sustains j_cross, which is exactly the mixed-potential OCV depression.

Thermal ODE (lumped, solved with scipy.integrate.solve_ivp):
    m*cp * dT/dt = Q_gen - Q_loss
    Q_gen  = N_cells * A_cell * [ j*(E_th - V_cell) + j_cross*E_th ]
    Q_loss = hA * (T - T_coolant)
The crossover term j_cross*E_th adds the full enthalpy of the methanol burned
parasitically at the cathode (it produces heat, not electrical work).

Conventions / enforced bounds:
    V_cell < E_rev (second law);  0 < efficiency < 1;  overpotentials >= 0;
    fuel (current) efficiency = j / (j + j_cross) in (0,1).

References:
    Scott, Taama, Argyropoulos (1999), J. Power Sources, 79, 43-59.
    Kulikovsky (2002), Electrochem. Communications, 4, 939-946.
    Nordlund & Lindbergh (2002), J. Electrochem. Soc., 149(9), A1107.
    Springer, Zawodzinski, Gottesfeld (1991), J. Electrochem. Soc., 138(8), 2334.
    Larminie & Dicks (2003), Fuel Cell Systems Explained, 2nd ed., Wiley.
"""

import numpy as np
from scipy.integrate import solve_ivp


class DMFC_F2a:
    """Direct Methanol Fuel Cell -- full electrochemical model with thermal dynamics."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol
    n = 6              # electrons per methanol molecule

    def __init__(self, params: dict):
        u = params["unit"]
        self.N_cells = u["N_cells"]["value"]
        self.A_cell = u["A_cell"]["value"]                # cm2
        self.t_mem = u["t_mem"]["value"]                  # cm
        self.lambda_mem = u["lambda_mem"]["value"]
        self.j_L = u["j_L"]["value"]                      # A/cm2
        self.alpha_a = u["alpha_anode"]["value"]
        self.alpha_c = u["alpha_cathode"]["value"]
        # Electrons transferred in the rate-determining step (sets the Tafel slope
        # b = RT/(alpha*n_rds*F)). For both methanol oxidation and ORR the RDS is a
        # single-electron step (n_rds=1), NOT the overall stoichiometric n=6 -- using
        # n=6 would collapse the Tafel slopes ~6x and wrongly leave OCV ~ E_rev.
        # Scott et al. (1999); Larminie & Dicks (2003).
        self.n_rds_a = u.get("n_rds_anode", {"value": 1.0})["value"]
        self.n_rds_c = u.get("n_rds_cathode", {"value": 1.0})["value"]
        self.j0_a_ref = u["j0_anode_ref"]["value"]        # A/cm2
        self.j0_c_ref = u["j0_cathode_ref"]["value"]      # A/cm2
        self.E_act_a = u["E_act_anode"]["value"]          # J/mol
        self.E_act_c = u["E_act_cathode"]["value"]        # J/mol
        self.j_cross_ref = u["j_cross_ref"]["value"]      # A/cm2
        self.E_act_cross = u["E_act_cross"]["value"]      # J/mol
        self.c_MeOH_ref = u["c_MeOH_ref"]["value"]        # mol/L
        self.T_ref = u["T_ref"]["value"]                  # K
        self.E_ref_std = u["E_ref_std"]["value"]          # V (at 298.15 K)
        self.dEdT = u["dEdT"]["value"]                    # V/K
        self.m_stack = u["m_stack"]["value"]              # kg
        self.cp_stack = u["cp_stack"]["value"]            # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]              # W/K
        self.T_coolant = u["T_coolant"]["value"]          # K

    # ------------------------------------------------------------------
    # Reversible (thermodynamic) cell voltage
    # ------------------------------------------------------------------
    def reversible_voltage(self, T):
        """Reversible cell potential E_rev(T) [V] for CH3OH/O2.

        Linear temperature correction about 298.15 K (Larminie & Dicks 2003).
        This is the thermodynamic ceiling; the *measured* OCV is far lower because
        of methanol crossover (handled in cell_voltage via the cathode mixed term).
        """
        return self.E_ref_std + self.dEdT * (T - 298.15)

    def thermoneutral_voltage(self, T):
        """Thermoneutral (enthalpy) voltage [V] for MeOH full oxidation.

        DeltaH = -726.6 kJ/mol (liquid MeOH, HHV) over n*F gives ~1.255 V.
        Used as the heat-generation reference (E_th - V drives Q_gen).
        """
        # 726600 J/mol / (6 * 96485 C/mol) = 1.2549 V ; mild T-dependence neglected
        return 1.2549

    # ------------------------------------------------------------------
    # Methanol crossover (equivalent leakage current density)
    # ------------------------------------------------------------------
    def crossover_current(self, T, c_MeOH=None):
        """Equivalent methanol crossover current density [A/cm2].

        Methanol permeates the membrane by diffusion (driven by concentration) with
        Arrhenius-activated permeability. Proportional to feed concentration.
        Nordlund & Lindbergh (2002); Kulikovsky (2002).
        """
        if c_MeOH is None:
            c_MeOH = self.c_MeOH_ref
        arr = np.exp((-self.E_act_cross / self.R) * (1.0 / T - 1.0 / self.T_ref))
        j_x = self.j_cross_ref * (c_MeOH / self.c_MeOH_ref) * arr
        return max(j_x, 0.0)

    # ------------------------------------------------------------------
    # Anode activation overpotential -- Tafel (slow MeOH oxidation)
    # ------------------------------------------------------------------
    def anode_overpotential(self, j_total, T):
        """Anode activation overpotential [V] (Pt-Ru methanol oxidation, Tafel)."""
        if j_total <= 0:
            return 0.0
        j0 = self.j0_a_ref * np.exp(
            (-self.E_act_a / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        j0 = max(j0, 1e-14)
        eta = (self.R * T) / (self.alpha_a * self.n_rds_a * self.F) * np.log(j_total / j0)
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Cathode overpotential incl. crossover (mixed-potential) load
    # ------------------------------------------------------------------
    def cathode_overpotential(self, j, T, j_cross):
        """Cathode ORR activation overpotential [V].

        The cathode catalyst must reduce O2 for BOTH the useful proton current j
        AND oxidise the crossed-over methanol (j_cross). The kinetic current is
        therefore (j + j_cross). At j=0 the cathode still carries j_cross -> finite
        overpotential -> mixed potential -> depressed OCV (the DMFC signature).
        """
        j_kin = j + j_cross
        if j_kin <= 0:
            return 0.0
        j0 = self.j0_c_ref * np.exp(
            (-self.E_act_c / self.R) * (1.0 / T - 1.0 / self.T_ref)
        )
        j0 = max(j0, 1e-14)
        eta = (self.R * T) / (self.alpha_c * self.n_rds_c * self.F) * np.log(j_kin / j0)
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Ohmic overpotential -- Springer (1991) membrane conductivity
    # ------------------------------------------------------------------
    def membrane_conductivity(self, T, lam=None):
        """Nafion ionic conductivity [S/cm] -- Springer (1991)."""
        if lam is None:
            lam = self.lambda_mem
        sigma = (0.005139 * lam - 0.00326) * np.exp(
            1268.0 * (1.0 / 303.15 - 1.0 / T)
        )
        return max(sigma, 1e-6)

    def ohmic_overpotential(self, j, T, lam=None):
        """Ohmic loss [V] across the membrane (only the useful current j ohmic-drops)."""
        sigma = self.membrane_conductivity(T, lam)
        return j * self.t_mem / sigma

    # ------------------------------------------------------------------
    # Concentration overpotential -- methanol mass-transport limit
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, T, j_L=None):
        """Concentration (methanol-transport) overpotential [V]."""
        if j_L is None:
            j_L = self.j_L
        if j <= 0:
            return 0.0
        ratio = j / j_L
        if ratio >= 1.0:
            return 10.0  # methanol-starved -- voltage collapses
        return -(self.R * T) / (self.n * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j, T, c_MeOH=None):
        """Net single-cell voltage [V]."""
        E = self.reversible_voltage(T)
        j_cross = self.crossover_current(T, c_MeOH)
        eta_a = self.anode_overpotential(j, T)
        eta_c = self.cathode_overpotential(j, T, j_cross)
        eta_ohm = self.ohmic_overpotential(j, T)
        eta_conc = self.concentration_overpotential(j, T)
        V = E - eta_a - eta_c - eta_ohm - eta_conc
        return max(V, 0.0)

    def fuel_efficiency(self, j, T, c_MeOH=None):
        """Faradaic / current efficiency = j / (j + j_cross), in (0,1)."""
        j_cross = self.crossover_current(T, c_MeOH)
        if j <= 0:
            return 0.0
        return j / (j + j_cross)

    # ------------------------------------------------------------------
    # Thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j, c_MeOH=None):
        """Temperature rate of change [K/s]."""
        V_cell = self.cell_voltage(j, T, c_MeOH)
        E_th = self.thermoneutral_voltage(T)
        j_cross = self.crossover_current(T, c_MeOH)
        # Useful current: heat = irreversibility (E_th - V_cell).
        # Crossover current: fully combusted to heat at the cathode -> E_th * j_cross.
        Q_gen = self.N_cells * self.A_cell * (
            j * (E_th - V_cell) + j_cross * E_th
        )
        Q_loss = self.hA_cool * (T - self.T_coolant)
        return (Q_gen - Q_loss) / (self.m_stack * self.cp_stack)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_cell_K, c_MeOH_molar,
                 dt, duration_s):
        """
        Simulate DMFC dynamics with coupled thermal ODE.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2]
        T_cell_K : float
            Initial cell temperature [K]
        c_MeOH_molar : float
            Methanol feed concentration [mol/L]
        dt : float
            Output time step [s]
        duration_s : float
            Total simulation duration [s]

        Returns
        -------
        dict with time-series arrays: t, voltage, power_density, efficiency,
            fuel_efficiency, temperature, crossover_current,
            overpotentials (dict of arrays).
        """
        _j = (current_density_A_cm2 if callable(current_density_A_cm2)
              else (lambda t: current_density_A_cm2))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            return [self.dTdt(T, j, c_MeOH_molar)]

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
        fuel_eff = np.zeros(N)
        j_cross_arr = np.zeros(N)
        eta_a = np.zeros(N)
        eta_c = np.zeros(N)
        eta_ohm = np.zeros(N)
        eta_conc = np.zeros(N)
        E_rev = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            jx = self.crossover_current(T, c_MeOH_molar)
            E_rev[i] = self.reversible_voltage(T)
            j_cross_arr[i] = jx
            eta_a[i] = self.anode_overpotential(j, T)
            eta_c[i] = self.cathode_overpotential(j, T, jx)
            eta_ohm[i] = self.ohmic_overpotential(j, T)
            eta_conc[i] = self.concentration_overpotential(j, T)
            voltage[i] = self.cell_voltage(j, T, c_MeOH_molar)
            power_density[i] = j * voltage[i]
            E_th = self.thermoneutral_voltage(T)
            # System (voltage) efficiency referenced to enthalpy voltage,
            # multiplied by fuel utilisation -> overall energy efficiency.
            volt_eff = voltage[i] / E_th if E_th > 0 else 0.0
            fuel_eff[i] = self.fuel_efficiency(j, T, c_MeOH_molar)
            efficiency[i] = volt_eff * fuel_eff[i]

        return {
            "t": t_out,
            "voltage": voltage,
            "power_density": power_density,
            "efficiency": efficiency,
            "fuel_efficiency": fuel_eff,
            "temperature": T_out,
            "crossover_current": j_cross_arr,
            "overpotentials": {
                "E_rev": E_rev,
                "anode": eta_a,
                "cathode": eta_c,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }
