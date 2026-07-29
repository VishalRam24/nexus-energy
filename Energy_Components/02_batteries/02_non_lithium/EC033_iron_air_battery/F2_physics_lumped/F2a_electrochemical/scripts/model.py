"""
EC033 -- Iron-Air Battery (Fe-Air) -- F2a Physics-Lumped Electrochemical Model

0D first-principles electrochemical model of a secondary iron-air cell with a
metallic-iron anode and a bifunctional air cathode, coupled to a thermal ODE.

Electrochemistry (alkaline KOH):
    Anode (discharge):   Fe + 2 OH-  ->  Fe(OH)2 + 2 e-          E0 ~ -0.877 V vs SHE
    Cathode ORR (disch): O2 + 2 H2O + 4 e-  ->  4 OH-           E0 ~ +0.401 V vs SHE
    Cathode OER (charge):4 OH-  ->  O2 + 2 H2O + 4 e-           (reverse, bifunctional)
    Parasitic HER (chg): 2 H2O + 2 e-  ->  H2 + 2 OH-           E0 ~ -0.828 V vs SHE
    Full-cell OCV ~ 1.28 V (E0_cathode - E0_anode), nominal discharge ~1.0 V.

Cell voltage (terminal) is built from a thermodynamic OCV minus / plus the
kinetic + transport + ohmic overpotentials, sign-switched between discharge and
charge so that
    V_discharge < OCV < V_charge,
which produces the large charge-discharge voltage gap (hysteresis) characteristic
of iron-air -> intrinsically LOW round-trip (voltaic) efficiency.

    V = OCV(T)
        - sign(I) * [ eta_anode + eta_cathode + eta_conc ]      kinetic + O2 transport
        - I_area * R_ohmic                                       ohmic (sign-consistent)

Kinetics use the Tafel/Butler-Volmer form
    eta = (R T)/(alpha n F) * ln(|j| / j0)        (Newman & Thomas-Alyea 2004)
with Arrhenius temperature dependence of j0 and R_ohmic.

Air-cathode O2 mass transport (gas-diffusion electrode) is a concentration
overpotential that diverges as the discharge ORR current approaches the O2
limiting current density j_L_O2:
    eta_conc = -(R T)/(n F) * ln(1 - j/j_L_O2)    (Barbir 2005; McKerracher 2015)

Coulombic efficiency on CHARGE is reduced by parasitic hydrogen evolution on the
iron electrode. The HER current is computed from the charge overpotential via a
Butler-Volmer branch; the coulombic efficiency is the fraction of charge current
that goes into iron reduction rather than H2:
    CE_charge = j_Fe / (j_Fe + j_HER)             (Manohar 2012; Form Energy 2022)
Only j_Fe charges the SOC, so Coulomb is conserved exactly per the bookkeeping
(charge in - H2 loss = stored).

Thermal ODE (lumped, scipy.solve_ivp):
    m cp dT/dt = Q_gen - Q_loss
    Q_gen  = A_cell * [ |I_area| * |V - OCV|   (irreversible, always >= 0)
                        + I_area * T * dOCV/dT  (reversible entropic) ]
    Q_loss = hA * (T - T_amb)

References:
    Manohar, A. K. et al. (2012). J. Electrochem. Soc. 159(8), A1209-A1214.
    Trocino, S. et al. (2022). J. Power Sources 523, 230999.
    McKerracher, R. D. et al. (2015). Electrochim. Acta 184, 264-275.
    Barbir, F. (2005). PEM Fuel Cells: Theory and Practice, Elsevier (transport).
    Newman, J. & Thomas-Alyea, K. (2004). Electrochemical Systems, 3rd ed., Wiley.
    Form Energy (2022). Iron-air long-duration storage technology report.
"""

import numpy as np
from scipy.integrate import solve_ivp


class IronAirF2a:
    """Iron-air cell -- physics-lumped electrochemical model with thermal ODE."""

    # Physical constants
    R = 8.314          # J/(mol.K)
    F = 96485.0        # C/mol

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_cell = u["A_cell"]["value"]                # cm2
        self.capacity_ref = u["capacity_ref"]["value"]    # Ah

        # Thermodynamics
        self.E0_anode = u["E0_anode"]["value"]            # V vs SHE
        self.E0_cathode = u["E0_cathode"]["value"]        # V vs SHE
        self.E0_HER = u["E0_HER"]["value"]                # V vs SHE
        self.OCV0 = self.E0_cathode - self.E0_anode       # ~1.28 V

        # Kinetics: exchange current densities (A/cm2)
        self.j0_anode = u["j0_anode"]["value"]
        self.j0_ORR = u["j0_ORR"]["value"]
        self.j0_OER = u["j0_OER"]["value"]
        self.j0_HER = u["j0_HER"]["value"]
        # electrons per reaction
        self.n_anode = 2
        self.n_ORR = 4
        self.n_OER = 4
        self.n_HER = 2
        self.alpha_anode = u["alpha_anode"]["value"]
        self.alpha_ORR = u["alpha_ORR"]["value"]
        self.alpha_OER = u["alpha_OER"]["value"]
        self.alpha_HER = u["alpha_HER"]["value"]

        # Air-cathode O2 mass transport
        self.j_L_O2 = u["j_L_O2"]["value"]                # A/cm2

        # Ohmic
        self.R_ohmic = u["R_ohmic"]["value"]              # Ohm.cm2

        # Temperature dependence
        self.T_ref = u["T_ref"]["value"]                  # K
        self.E_act = u["E_act"]["value"]                  # J/mol
        self.dOCV_dT = u["dOCV_dT"]["value"]              # V/K

        # Thermal
        self.m_cell = u["m_cell"]["value"]                # kg
        self.cp_cell = u["cp_cell"]["value"]              # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]              # W/K
        self.T_amb = u["T_amb"]["value"]                  # K

    # ------------------------------------------------------------------
    # Thermodynamic open-circuit voltage
    # ------------------------------------------------------------------
    def ocv(self, T):
        """Full-cell open-circuit voltage [V] with entropic temperature term."""
        return self.OCV0 + self.dOCV_dT * (T - self.T_ref)

    def _arrhenius(self, j0, T):
        """Arrhenius scaling of an exchange current density / kinetic rate."""
        return j0 * np.exp((-self.E_act / self.R) * (1.0 / T - 1.0 / self.T_ref))

    # ------------------------------------------------------------------
    # Tafel / Butler-Volmer activation overpotentials
    # ------------------------------------------------------------------
    def _tafel(self, j_mag, j0, alpha, n, T):
        """Magnitude of activation overpotential [V] (>=0) from Tafel kinetics."""
        if j_mag <= 0.0:
            return 0.0
        j0T = max(self._arrhenius(j0, T), 1e-15)
        eta = (self.R * T) / (alpha * n * self.F) * np.log(max(j_mag / j0T, 1.0))
        return max(eta, 0.0)

    def anode_overpotential(self, j_mag, T):
        """Fe/Fe(OH)2 anode activation overpotential [V] (same form charge/discharge)."""
        return self._tafel(j_mag, self.j0_anode, self.alpha_anode, self.n_anode, T)

    def cathode_overpotential(self, j_mag, T, charging):
        """Air-cathode activation overpotential [V]: ORR on discharge, OER on charge.

        OER (charge) is far slower than ORR -> much larger overpotential, the main
        driver of the charge-discharge voltage gap.
        """
        if charging:
            return self._tafel(j_mag, self.j0_OER, self.alpha_OER, self.n_OER, T)
        return self._tafel(j_mag, self.j0_ORR, self.alpha_ORR, self.n_ORR, T)

    # ------------------------------------------------------------------
    # Air-cathode O2 mass-transport (concentration) overpotential
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j_mag, T, charging):
        """O2 transport-limited concentration overpotential [V] (discharge only)."""
        # On charge the cathode evolves O2 (OER) -> not O2-supply limited.
        if charging or j_mag <= 0.0:
            return 0.0
        ratio = j_mag / self.j_L_O2
        if ratio >= 1.0:
            return 5.0  # effectively starved of oxygen
        return -(self.R * T) / (self.n_ORR * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Parasitic hydrogen evolution on charge -> coulombic efficiency
    # ------------------------------------------------------------------
    def her_current(self, eta_charge_neg, T):
        """Parasitic HER current density [A/cm2] given the (negative) cathodic
        polarization of the iron electrode during charging.

        Butler-Volmer cathodic branch: j_HER = j0_HER * exp(alpha n F |eta| / R T).
        |eta| is how far the iron electrode is driven below the HER equilibrium.
        """
        if eta_charge_neg <= 0.0:
            return 0.0
        j0T = self._arrhenius(self.j0_HER, T)
        arg = (self.alpha_HER * self.n_HER * self.F * eta_charge_neg) / (self.R * T)
        arg = min(arg, 50.0)  # numerical guard
        return j0T * np.expm1(arg) if arg < 1e-6 else j0T * (np.exp(arg) - 1.0)

    def coulombic_efficiency_charge(self, j_charge_mag, T):
        """Coulombic efficiency on charge: fraction of charge current storing Fe.

        The applied charge current splits between iron reduction (j_Fe) and
        parasitic H2 evolution (j_HER). The iron-electrode overpotential grows
        with the demanded Fe current; we size HER off the anode (Fe) overpotential.
        Returns (CE in (0,1], j_Fe, j_HER).
        """
        if j_charge_mag <= 0.0:
            return 1.0, 0.0, 0.0
        eta_fe = self.anode_overpotential(j_charge_mag, T)
        j_her = self.her_current(eta_fe, T)
        ce = j_charge_mag / (j_charge_mag + j_her)
        return ce, j_charge_mag, j_her

    # ------------------------------------------------------------------
    # Terminal cell voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, j_area, T):
        """Terminal voltage [V]. j_area: A/cm2, positive=discharge, negative=charge.

        Enforces V_discharge < OCV < V_charge by sign-switching the overpotentials.
        """
        E = self.ocv(T)
        if j_area == 0.0:
            return E
        charging = j_area < 0.0
        j_mag = abs(j_area)
        eta = (self.anode_overpotential(j_mag, T)
               + self.cathode_overpotential(j_mag, T, charging)
               + self.concentration_overpotential(j_mag, T, charging))
        eta_ohm = j_mag * self.R_ohmic
        if charging:
            # voltage you must apply -> above OCV
            return E + eta + eta_ohm
        # discharge -> below OCV
        return max(E - eta - eta_ohm, 0.0)

    # ------------------------------------------------------------------
    # Thermal ODE derivative
    # ------------------------------------------------------------------
    def dTdt(self, T, j_area):
        """Temperature rate of change [K/s] from irreversible + entropic heat."""
        V = self.cell_voltage(j_area, T)
        E = self.ocv(T)
        # Irreversible heat (always heating): |I| * |V - OCV|
        I = j_area * self.A_cell                       # A
        Q_irr = abs(I) * abs(V - E)                    # W
        # Reversible entropic heat: I * T * dOCV/dT (sign of I matters)
        Q_rev = I * T * self.dOCV_dT                   # W
        Q_gen = Q_irr + Q_rev
        Q_loss = self.hA_cool * (T - self.T_amb)
        return (Q_gen - Q_loss) / (self.m_cell * self.cp_cell)

    # ------------------------------------------------------------------
    # Round-trip / sub-efficiencies at an operating current
    # ------------------------------------------------------------------
    def voltaic_efficiency(self, j_area_mag, T):
        """Voltaic efficiency = V_discharge / V_charge at current magnitude j (<1)."""
        V_d = self.cell_voltage(+j_area_mag, T)
        V_c = self.cell_voltage(-j_area_mag, T)
        return V_d / V_c if V_c > 0 else 0.0

    def round_trip_efficiency(self, j_area_mag, T):
        """Round-trip efficiency = voltaic * coulombic (<1)."""
        eta_v = self.voltaic_efficiency(j_area_mag, T)
        ce, _, _ = self.coulombic_efficiency_charge(j_area_mag, T)
        return eta_v * ce

    # ------------------------------------------------------------------
    # Time-domain simulation (thermal ODE via solve_ivp)
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_cm2, T_init_K, dt, duration_s,
                 soc_init=0.5):
        """Simulate iron-air cell dynamics with coupled thermal ODE + SOC bookkeeping.

        Parameters
        ----------
        current_density_A_cm2 : float or callable(t)
            Operating current density [A/cm2]; positive=discharge, negative=charge.
        T_init_K : float
            Initial cell temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation duration [s].
        soc_init : float
            Initial state of charge (0-1).

        Returns
        -------
        dict of time-series arrays: t, voltage, ocv, power_density, soc,
            temperature, coulombic_eff, her_current, overpotentials{...}
        """
        _j = (current_density_A_cm2 if callable(current_density_A_cm2)
              else (lambda t: current_density_A_cm2))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        # Capacity in Coulombs of charge for SOC integration
        Q_cap_C = self.capacity_ref * 3600.0           # As

        def rhs(t, y):
            T = y[0]
            j = _j(t)
            dT = self.dTdt(T, j)
            # SOC dynamics: discharge (j>0) lowers SOC; charge (j<0) raises SOC,
            # but only the coulombic-efficient fraction is stored.
            I = j * self.A_cell                        # A
            if I < 0.0:  # charging
                ce, _, _ = self.coulombic_efficiency_charge(abs(j), T)
                dQ = -I * ce                           # >0, stored charge rate [A]
            else:        # discharging (CE ~ 1 on discharge)
                dQ = -I                                # <0
            dsoc = dQ / Q_cap_C
            return [dT, dsoc]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_init_K, soc_init],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        soc_out = np.clip(sol.y[1], 0.0, 1.0)
        N = len(t_out)

        voltage = np.zeros(N)
        ocv = np.zeros(N)
        power_density = np.zeros(N)
        ce_arr = np.zeros(N)
        her_arr = np.zeros(N)
        eta_anode = np.zeros(N)
        eta_cathode = np.zeros(N)
        eta_conc = np.zeros(N)

        for i in range(N):
            j = _j(t_out[i])
            T = T_out[i]
            charging = j < 0.0
            j_mag = abs(j)
            ocv[i] = self.ocv(T)
            voltage[i] = self.cell_voltage(j, T)
            power_density[i] = j * voltage[i]          # W/cm2, +=discharge
            eta_anode[i] = self.anode_overpotential(j_mag, T)
            eta_cathode[i] = self.cathode_overpotential(j_mag, T, charging)
            eta_conc[i] = self.concentration_overpotential(j_mag, T, charging)
            if charging:
                ce, _, j_her = self.coulombic_efficiency_charge(j_mag, T)
                ce_arr[i] = ce
                her_arr[i] = j_her
            else:
                ce_arr[i] = 1.0
                her_arr[i] = 0.0

        return {
            "t": t_out,
            "voltage": voltage,
            "ocv": ocv,
            "power_density": power_density,
            "soc": soc_out,
            "temperature": T_out,
            "coulombic_eff": ce_arr,
            "her_current": her_arr,
            "overpotentials": {
                "anode": eta_anode,
                "cathode": eta_cathode,
                "concentration": eta_conc,
            },
        }
