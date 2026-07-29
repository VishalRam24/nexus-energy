"""
EC039 -- Organic Flow Battery (OFB) -- F2a Physics-Lumped Stack Model

0D/1D physics-lumped model of an aqueous organic redox flow battery (AORFB)
stack with coupled electrochemical, state-of-charge, capacity-fade/crossover,
and thermal ODEs. Pure Python + NumPy + SciPy.

Representative chemistry: AQDS / ferricyanide (anthraquinone-2,7-disulfonate
negolyte, ferri/ferrocyanide posolyte) -- the metal-free organic-inorganic
couple introduced by the Aziz group (Huskinson 2014).

    Negative (negolyte): AQDS + 2 H+ + 2 e- <-> H2AQDS      E0 ~ -0.21 V (vs SHE, acid)
    Positive (posolyte): Fe(CN)6^3- + e-   <-> Fe(CN)6^4-   E0 ~ +0.49 V (alkaline)
    Cell (alkaline quinone form, Lin 2015):                 E0_cell ~ 0.70-1.20 V
    n = 2-electron quinone reduction.

----------------------------------------------------------------------------
Voltage model (first-principles, per cell, at each time step)
----------------------------------------------------------------------------
    E_nernst(SOC, T) = E0(T)
        + (R T)/(n F) * ln( SOC_p /(1-SOC_p) )      posolyte (1 e-)
        + (R T)/(n F) * ln( SOC_n /(1-SOC_n) )      negolyte
    (with symmetric SOC both sides -> the familiar
        E0(T) + 2 (R T)/(n F) ln(SOC/(1-SOC)) form, Nernst 1889 / Bard-Faulkner 2001)
    E0(T) = E0_ref + dOCV_dT * (T - T_ref)           entropic temperature correction

    Charge (I<0) :  V = E_nernst + |eta_act| + |eta_ohm| + |eta_conc|
    Discharge(I>0):  V = E_nernst - |eta_act| - |eta_ohm| - |eta_conc|

    eta_act  : Butler-Volmer, solved by the symmetric-alpha closed form
               eta = (2 R T)/(n F) * asinh( j / (2 j0) )       (Bard-Faulkner 2001, Eq 3.4.14)
               j0 = j0_ref * exp(-Ea/R (1/T - 1/Tref))         (Arrhenius)
    eta_ohm  : j * ASR(T) ;  ASR(T) = ASR_ref * exp(Ea/R (1/T - 1/Tref))
    eta_conc : -(R T)/(n F) * ln(1 - j/j_L(SOC, Q))            (Newman-Thomas-Alyea 2004)
               j_L scales with reactant availability (SOC) and flow factor.

----------------------------------------------------------------------------
State-of-charge ODE (Coulomb counting with flow-limited charge transfer)
----------------------------------------------------------------------------
    dSOC/dt = -(I * CE_eff) / (n F C_act V_tank)        [discharge depletes SOC]
    where CE_eff = CE on discharge, 1/CE consumed on charge (coulombic inefficiency
    parks charge into side reactions / crossover). C_act = active-species molarity,
    V_tank = electrolyte tank volume. Conservation: total charge passed minus
    crossover/decomposition loss equals delta-SOC * capacity.

----------------------------------------------------------------------------
Capacity fade / crossover ODE (organic molecule decomposition)
----------------------------------------------------------------------------
    dCcap/dt = -k_fade * Ccap                            first-order temporal fade
    Coulombic efficiency CE < 1 because organic active species decompose
    (e.g. anthrone formation, Michael addition, dimerisation) and crossover
    through the membrane. Fade rate is Arrhenius-accelerated with T.
        k_fade(T) = k_fade_ref * exp(-Ea_fade/R (1/T - 1/Tref))
    Aziz-group lifetime studies report ~0.1 %/day (AQDS) down to <0.01 %/day
    (DHAQ, ((BTMAP-Vi)) engineered molecules).

----------------------------------------------------------------------------
Lumped thermal ODE (solved with scipy.integrate.solve_ivp)
----------------------------------------------------------------------------
    (m cp) dT/dt = Q_gen - Q_cool
    Q_gen  = N_cells * A * |j| * |V_cell - E_nernst|      irreversible overpotential heat
             + I * N_cells * T * dOCV_dT                  reversible entropic heat
             + P_pump                                     parasitic pump dissipation
    Q_cool = hA * (T - T_amb)                             Newton cooling to ambient

References
----------
    Huskinson, B.; Marshak, M. P.; ... Aziz, M. J.; Gordon, R. G. (2014).
        A metal-free organic-inorganic aqueous flow battery. Nature 505, 195-198.
    Lin, K.; ... Aziz, M. J. (2015). Alkaline quinone flow battery.
        Science 349, 1529-1532.
    Kwabi, D. G.; Ji, Y.; Aziz, M. J. (2020). Electrolyte lifetime in aqueous
        organic redox flow batteries: a critical review. Chem. Rev. 120, 6467-6489.
    Bard, A. J.; Faulkner, L. R. (2001). Electrochemical Methods, 2nd ed., Wiley
        (Butler-Volmer, Nernst).
    Newman, J.; Thomas-Alyea, K. (2004). Electrochemical Systems, 3rd ed., Wiley
        (mass-transport limiting current).
    Tang, A.; Bao, J.; Skyllas-Kazacos, M. (2014). Studies on pressure losses and
        flow rate optimization in vanadium redox flow battery. J. Power Sources 248
        (flow-dependent stack behaviour, pump power scaling).
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314      # J/(mol.K)
F_CONST = 96485.0  # C/mol


class OrganicFlowF2a:
    """Physics-lumped organic redox flow battery stack.

    Sign convention: current I > 0 = discharge (SOC falls),
    I < 0 = charge (SOC rises). Voltages are per-cell unless _stack.
    """

    SOC_MIN = 0.01
    SOC_MAX = 0.99

    def __init__(self, params: dict):
        u = params["unit"]
        therm = params["thermal"]
        soc_p = params["soc"]

        # --- electrochemical / geometric ---
        self.N_cells = int(u["N_cells"]["value"])
        self.A_cm2 = float(u["electrode_area_cm2"]["value"])
        self.E0_ref = float(u["E0"]["value"])
        self.ASR_ref = float(u["R_cell_ohm_cm2_ref"]["value"])   # Ohm.cm2 at T_ref
        self.n = int(u["n"]["value"])
        self.j0_ref = float(u["j0_A_cm2"]["value"])              # exchange c.d. at T_ref
        self.jL_ref = float(u["j_L_A_cm2"]["value"])             # limiting c.d. (full SOC, ref flow)
        self.k_pump = float(u["pump_loss_coefficient"]["value"])  # W/A^2
        self.C_act = float(u["electrolyte_conc_M"]["value"])      # mol/L
        self.V_tank_L = float(u["tank_volume_L"]["value"])        # L per side

        # --- SOC / capacity ---
        self.CE = float(soc_p["coulombic_efficiency"]["value"])   # 0<CE<1
        self.flow_factor = float(soc_p["flow_factor"]["value"])   # mass-transport multiplier (>=~1)

        # --- thermal ---
        self.T_ref = float(therm["T_ref"]["value"])
        self.E_a = float(therm["E_a"]["value"])                   # J/mol, kinetics+ASR
        self.dOCV_dT = float(therm["dOCV_dT"]["value"])           # V/K per cell
        self.m_cp = float(therm["m_cp"]["value"])                 # J/K lumped heat capacity
        self.hA = float(therm["hA"]["value"])                    # W/K to ambient
        self.T_amb = float(therm["T_amb"]["value"])              # K

        # --- fade / crossover ---
        self.k_fade_ref = float(therm["k_fade_ref"]["value"])     # 1/s temporal fade at T_ref
        self.Ea_fade = float(therm["Ea_fade"]["value"])           # J/mol

        # capacity charge content [C] for one side: n F * (C_act mol/L * V_tank L)
        self.Q_cap_C = self.n * F_CONST * self.C_act * self.V_tank_L

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------
    def e0_thermal(self, T):
        """Temperature-corrected standard cell potential [V] (per cell)."""
        return self.E0_ref + self.dOCV_dT * (T - self.T_ref)

    def e_nernst(self, soc, T):
        """Nernst open-circuit cell voltage [V] at given SOC and T.

        Symmetric SOC both electrodes -> 2*(RT/nF) ln(SOC/(1-SOC)).
        """
        soc = float(np.clip(soc, self.SOC_MIN, self.SOC_MAX))
        thermal_factor = R_GAS * T / (self.n * F_CONST)
        return self.e0_thermal(T) + 2.0 * thermal_factor * np.log(soc / (1.0 - soc))

    # ------------------------------------------------------------------
    # Kinetics / transport (all magnitudes; sign applied at cell_voltage)
    # ------------------------------------------------------------------
    def _arrhenius_up(self, T):
        """Thermally-activated INCREASE (kinetics): exp(-Ea/R (1/T-1/Tref))."""
        return np.exp(-self.E_a / R_GAS * (1.0 / T - 1.0 / self.T_ref))

    def _arrhenius_down(self, T):
        """Resistance DECREASES with T: exp(+Ea/R (1/T-1/Tref))."""
        return np.exp(self.E_a / R_GAS * (1.0 / T - 1.0 / self.T_ref))

    def activation_overpotential(self, current, T):
        """|eta_act| [V] from Butler-Volmer (symmetric alpha, asinh form)."""
        j = abs(current) / self.A_cm2          # A/cm2
        j0 = max(self.j0_ref * self._arrhenius_up(T), 1e-12)
        return (2.0 * R_GAS * T) / (self.n * F_CONST) * np.arcsinh(j / (2.0 * j0))

    def asr(self, T):
        """Area-specific resistance [Ohm.cm2], Arrhenius (falls with T)."""
        return self.ASR_ref * self._arrhenius_down(T)

    def ohmic_overpotential(self, current, T):
        """|eta_ohm| [V] = j * ASR(T)."""
        j = abs(current) / self.A_cm2
        return j * self.asr(T)

    def limiting_current(self, soc):
        """Flow/SOC-limited current density [A/cm2].

        On discharge reactant is the charged species (~SOC); near full
        discharge (SOC->0) transport collapses. flow_factor>=~1 lifts j_L
        with higher electrolyte flow rate (Tang 2014).
        """
        soc = float(np.clip(soc, self.SOC_MIN, self.SOC_MAX))
        return self.jL_ref * self.flow_factor * soc

    def concentration_overpotential(self, current, soc):
        """|eta_conc| [V], diverges as |j| -> j_L (Newman-Thomas-Alyea 2004)."""
        j = abs(current) / self.A_cm2
        jL = self.limiting_current(soc)
        ratio = j / jL
        if ratio >= 0.999:
            return 5.0  # effectively transport-limited (capped, finite)
        return -(R_GAS * 298.15) / (self.n * F_CONST) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Cell / stack voltage
    # ------------------------------------------------------------------
    def cell_voltage(self, soc, current, T):
        """Per-cell terminal voltage [V]. I>0 discharge, I<0 charge."""
        E = self.e_nernst(soc, T)
        eta = (self.activation_overpotential(current, T)
               + self.ohmic_overpotential(current, T)
               + self.concentration_overpotential(current, soc))
        if current >= 0.0:        # discharge: losses subtract
            # floor at 0 V: a real stack hits its BMS cutoff at the transport
            # wall rather than driving the terminal voltage negative.
            return max(E - eta, 0.0)
        else:                     # charge: losses add (cell pushed above OCV)
            return E + eta

    def stack_voltage(self, soc, current, T):
        """Stack terminal voltage [V] = N_cells * V_cell."""
        return self.N_cells * self.cell_voltage(soc, current, T)

    def pump_loss(self, current):
        """Parasitic pump power [W]. P = k_pump * I^2 (Tang 2014 scaling)."""
        return self.k_pump * current**2

    # ------------------------------------------------------------------
    # Heat generation
    # ------------------------------------------------------------------
    def heat_generation(self, soc, current, T):
        """Stack heat generation rate [W].

        Q = irreversible overpotential heat + reversible entropic + pump.

        Irreversible heat is taken as N*|I|*|E - V_cell| with V_cell the
        BMS-clipped terminal voltage, so the heat stays consistent with the
        actual (bounded) terminal voltage past the discharge cutoff.
        """
        E = self.e_nernst(soc, T)
        V = self.cell_voltage(soc, current, T)
        q_irrev = self.N_cells * abs(current) * abs(E - V)
        # reversible entropic: sign such that discharge of -dOCV_dT<0 releases heat
        q_rev = current * self.N_cells * T * self.dOCV_dT
        q_pump = self.pump_loss(current)
        return q_irrev + q_rev + q_pump

    # ------------------------------------------------------------------
    # Rates: SOC and capacity fade
    # ------------------------------------------------------------------
    def dSOC_dt(self, soc, current):
        """SOC rate [1/s] from Coulomb counting with coulombic efficiency.

        Discharge (I>0): deliver less charge usefully than parked
        (CE<1 means useful delivery costs more SOC drop). Charge (I<0):
        only CE fraction of input charge raises SOC (rest lost to side rxns).
        """
        if current >= 0.0:        # discharge: SOC falls; inefficiency steepens fall
            return -(current / self.CE) / self.Q_cap_C
        else:                     # charge: SOC rises; only CE of input is stored
            return -(current * self.CE) / self.Q_cap_C

    def dCap_dt(self, cap, T):
        """Capacity fraction fade rate [1/s] -- first-order, Arrhenius in T."""
        k = self.k_fade_ref * self._arrhenius_up_fade(T)
        return -k * cap

    def _arrhenius_up_fade(self, T):
        return np.exp(-self.Ea_fade / R_GAS * (1.0 / T - 1.0 / self.T_ref))

    # ------------------------------------------------------------------
    # Efficiencies
    # ------------------------------------------------------------------
    def voltage_efficiency(self, soc, current, T):
        """Voltage efficiency V_disch/V_charge at |I|, clipped to (0,1)."""
        I = abs(current) if current != 0 else 1.0
        V_dis = self.cell_voltage(soc, I, T)
        V_chg = self.cell_voltage(soc, -I, T)
        if V_chg <= 0:
            return 0.0
        return float(np.clip(V_dis / V_chg, 0.0, 1.0))

    def coulombic_efficiency(self):
        """Coulombic efficiency (constant material property, 0<CE<1)."""
        return self.CE

    def energy_efficiency(self, soc, current, T):
        """Round-trip energy efficiency = CE * voltage_efficiency."""
        return self.coulombic_efficiency() * self.voltage_efficiency(soc, current, T)

    # ------------------------------------------------------------------
    # Time-domain simulation (coupled SOC + capacity + thermal ODEs)
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0, T0, dt, duration_s, cap0=1.0):
        """Integrate coupled ODE system with scipy.integrate.solve_ivp.

        Parameters
        ----------
        current_A : float or callable(t)
            Stack current [A]. I>0 discharge, I<0 charge.
        soc0 : float       initial state of charge in (0,1)
        T0   : float       initial stack temperature [K]
        dt   : float       output sample step [s]
        duration_s : float total time [s]
        cap0 : float       initial capacity fraction (default 1.0)

        Returns
        -------
        dict of time-series arrays: t, soc, capacity, voltage (stack),
            cell_voltage, power, temperature, efficiency, overpotentials(dict).
        """
        _I = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            soc, cap, T = y
            soc_c = float(np.clip(soc, self.SOC_MIN, self.SOC_MAX))
            I = _I(t)
            return [
                self.dSOC_dt(soc_c, I),
                self.dCap_dt(cap, T),
                self.heat_generation(soc_c, I, T) - self.hA * (T - self.T_amb),
            ]

        # thermal eqn divides by m_cp -> fold into rhs cleanly
        def rhs_scaled(t, y):
            d = rhs(t, y)
            d[2] = d[2] / self.m_cp
            return d

        sol = solve_ivp(
            rhs_scaled, (0.0, duration_s), [soc0, cap0, T0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t_out = sol.t
        soc_out = np.clip(sol.y[0], self.SOC_MIN, self.SOC_MAX)
        cap_out = sol.y[1]
        T_out = sol.y[2]
        N = len(t_out)

        v_stack = np.zeros(N)
        v_cell = np.zeros(N)
        power = np.zeros(N)
        eff = np.zeros(N)
        E_n = np.zeros(N)
        eta_a = np.zeros(N)
        eta_o = np.zeros(N)
        eta_c = np.zeros(N)

        for i in range(N):
            I = _I(t_out[i])
            s = soc_out[i]
            T = T_out[i]
            v_cell[i] = self.cell_voltage(s, I, T)
            v_stack[i] = self.N_cells * v_cell[i]
            power[i] = v_stack[i] * I - self.pump_loss(I)
            eff[i] = self.energy_efficiency(s, I, T)
            E_n[i] = self.e_nernst(s, T)
            eta_a[i] = self.activation_overpotential(I, T)
            eta_o[i] = self.ohmic_overpotential(I, T)
            eta_c[i] = self.concentration_overpotential(I, s)

        return {
            "t": t_out,
            "soc": soc_out,
            "capacity": cap_out,
            "voltage": v_stack,
            "cell_voltage": v_cell,
            "power": power,
            "temperature": T_out,
            "efficiency": eff,
            "overpotentials": {
                "E_nernst": E_n,
                "activation": eta_a,
                "ohmic": eta_o,
                "concentration": eta_c,
            },
        }
