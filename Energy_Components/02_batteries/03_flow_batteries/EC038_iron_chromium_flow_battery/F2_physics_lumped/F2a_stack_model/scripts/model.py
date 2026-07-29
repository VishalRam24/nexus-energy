"""
EC038 -- Iron-Chromium Flow Battery (ICFB / Fe-Cr RFB) -- F2a Physics-Lumped Stack Model

0D physics-lumped model of a NASA-class Fe-Cr redox flow battery stack with coupled
SOC and thermal ODEs, integrated with scipy.integrate.solve_ivp.

Chemistry (NASA Redox system):
    Positive (posolyte):  Fe3+ + e-  <-> Fe2+      E0_Fe = +0.77 V vs SHE
    Negative (negolyte):  Cr3+ + e-  <-> Cr2+      E0_Cr = -0.41 V vs SHE
    Cell:  Fe3+ + Cr2+ <-> Fe2+ + Cr3+             E0_cell = 1.18 V

Single-cell open-circuit (Nernst) voltage, written from BOTH couples so the two
half-cell concentration ratios appear explicitly (SOC defined as the charged fraction):
    On charge we make Fe2+->Fe3+ (positive) and Cr3+->Cr2+ (negative), so
        positive ratio  [Fe3+]/[Fe2+] = SOC/(1-SOC)
        negative ratio  [Cr2+]/[Cr3+] = SOC/(1-SOC)
    E_pos = E0_Fe + (RT/nF) ln([Fe3+]/[Fe2+]) = E0_Fe + (RT/nF) ln(SOC/(1-SOC))
    E_neg = E0_Cr + (RT/nF) ln([Cr3+]/[Cr2+]) = E0_Cr - (RT/nF) ln(SOC/(1-SOC))
    E_ocv = E_pos - E_neg = (E0_Fe-E0_Cr) + 2*(RT/nF) ln(SOC/(1-SOC))

Polarisation (loss) terms, evaluated per cell, j = |I|/A_cell:
    eta_act  : Butler-Volmer activation, solved for both electrodes. The Cr electrode
               has a much smaller exchange current density (sluggish kinetics) so it
               dominates the activation loss -- the defining feature of Fe-Cr cells.
    eta_ohm  : j * R_area(T), Arrhenius temperature-dependent area-specific resistance.
    eta_conc : -(RT/nF) ln(1 - j/j_L), Nernstian mass-transport limit.

Terminal voltage:
    discharge (I>0):  V = N*(E_ocv - eta_act - eta_ohm - eta_conc)   (V < V_ocv)
    charge   (I<0):  V = N*(E_ocv + eta_act + eta_ohm + eta_conc)   (V > V_ocv)
  => V_charge > V_ocv > V_discharge, guaranteeing round-trip eff < 1.

Coulombic loss / parasitics (charge only):
    Hydrogen evolution on the Cr (negative) side competes with Cr3+ reduction:
        2 H+ + 2 e- -> H2     (parasitic, NASA's main coulombic-efficiency loss).
    A fraction of the charge current goes to HER, so the current that actually
    charges the battery is I_eff = I_charge - I_H2, giving coulombic eff < 1.
    On discharge HER does not run (cathodic-only here); the stored charge is delivered.
    Self-discharge crossover of Fe3+/Cr2+ through the membrane is a further
    Coulombic loss modelled as an SOC decay term.

SOC ODE (charged-species balance over tank inventory):
    Q_cap = n*F*c_active*V_tank      [Coulomb] total tank capacity
    dSOC/dt = ( -I_faradaic_into_charge ) / Q_cap  - k_self*SOC
      where I_faradaic_into_charge = -(I + I_H2*sign) accounts for parasitics, and
      k_self captures membrane crossover self-discharge.
    Flow rate sets the limiting current j_L (mass transport) and the well-mixed
    assumption -- higher Q_flow -> higher j_L -> lower concentration loss.

Thermal ODE (lumped, stack + electrolyte tanks):
    m*cp dT/dt = Q_gen - Q_loss
    Q_gen = |I| * |V_terminal/N_cells - E_ocv| * N_cells  (overpotential heating)
            + |I_H2-equivalent| * E_ocv * N_cells          (parasitic heat)
    Q_loss = hA*(T - T_amb)

References:
    Thaller, L.H. NASA / US Patent 3,996,064 (1974) -- redox flow cell concept.
    Hagedorn, N.H. (1984) NASA TM-83677 "NASA Redox Storage System Development Project".
    Gahn, R.F. et al. (1985) NASA TM-87034 -- Fe/Cr single-cell performance.
    Hruska, L.W. & Savinell, R.F. (1981) J. Electrochem. Soc. 128, 18 -- Fe/Cr kinetics.
    Zeng, Y.K. et al. (2015) J. Power Sources 278, 294 -- Fe/Cr RFB modelling.
    Wang, W. et al. (2013) Adv. Funct. Mater. 23, 970 -- RFB review.
    Newman & Thomas-Alyea (2004) "Electrochemical Systems" -- Butler-Volmer/Nernst.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class FeCrFlowBatteryF2a:
    """Iron-Chromium flow battery -- physics-lumped stack model (F2a)."""

    R = 8.314        # J/(mol.K) gas constant
    F = 96485.0      # C/mol Faraday constant

    def __init__(self, params: dict):
        u = params["unit"]
        g = lambda k: u[k]["value"]

        self.N_cells = g("N_cells")
        self.A_cell = g("A_cell")              # cm2
        self.E0_Fe = g("E0_Fe")
        self.E0_Cr = g("E0_Cr")
        self.n = g("n")
        self.c_Fe = g("c_Fe_total")            # mol/m3
        self.c_Cr = g("c_Cr_total")            # mol/m3
        self.j0_Fe = g("j0_Fe")                # A/cm2
        self.j0_Cr = g("j0_Cr")                # A/cm2
        self.alpha = g("alpha")
        self.E_act_Fe = g("E_act_Fe")
        self.E_act_Cr = g("E_act_Cr")
        self.R_area_ref = g("R_area_ref")      # ohm.cm2
        self.E_act_R = g("E_act_R")
        self.j_L_ref = g("j_L")                # A/cm2 (at nominal flow)
        self.i_H2_ref = g("i_H2_ref")          # A/cm2
        self.E_act_H2 = g("E_act_H2")
        self.k_cross = g("k_cross")            # m2/s (membrane diffusion coeff, Fick)
        self.t_mem = g("t_mem")                # m
        self.V_tank = g("V_tank")              # m3
        self.Q_flow_ref = g("Q_flow")          # m3/s
        self.m_thermal = g("m_thermal")
        self.cp_thermal = g("cp_thermal")
        self.hA_loss = g("hA_loss")
        self.T_amb = g("T_amb")
        self.T_ref = g("T_ref")

        # Total per-side tank capacity in Coulombs (active-species inventory)
        c_active = min(self.c_Fe, self.c_Cr)   # limiting reactant
        self.Q_cap = self.n * self.F * c_active * self.V_tank   # Coulomb

        self.E0_cell = self.E0_Fe - self.E0_Cr  # 1.18 V nominal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_soc(soc):
        return min(0.999, max(0.001, float(soc)))

    def _arrhenius(self, k_ref, E_act, T):
        """Arrhenius scaling of a rate/exchange-current relative to T_ref."""
        return k_ref * np.exp(-(E_act / self.R) * (1.0 / T - 1.0 / self.T_ref))

    # ------------------------------------------------------------------
    # Nernst open-circuit voltage (per cell) -- both couples
    # ------------------------------------------------------------------
    def nernst_voltage(self, soc, T):
        """Single-cell OCV [V] from Fe3+/Fe2+ and Cr3+/Cr2+ Nernst terms."""
        soc = self._clip_soc(soc)
        ratio = soc / (1.0 - soc)
        rt_nf = self.R * T / (self.n * self.F)
        E_pos = self.E0_Fe + rt_nf * np.log(ratio)
        E_neg = self.E0_Cr - rt_nf * np.log(ratio)
        return E_pos - E_neg                       # = E0_cell + 2*rt_nf*ln(ratio)

    # ------------------------------------------------------------------
    # Limiting current density -- depends on flow rate
    # ------------------------------------------------------------------
    def limiting_current(self, Q_flow):
        """Mass-transport limiting current density [A/cm2] ~ flow^0.5 (Sherwood)."""
        Q_flow = max(Q_flow, 1e-9)
        return self.j_L_ref * np.sqrt(Q_flow / self.Q_flow_ref)

    # ------------------------------------------------------------------
    # Activation overpotential -- Butler-Volmer, both electrodes
    # ------------------------------------------------------------------
    def _bv_overpotential(self, j, j0, T):
        """Solve Butler-Volmer |j| = j0[exp(a f eta) - exp(-(1-a) f eta)] for eta>=0.

        Returns the magnitude of activation overpotential [V] for a net current
        density j (A/cm2) drawn through an electrode with exchange current j0.
        """
        if j <= 0.0:
            return 0.0
        j0 = max(j0, 1e-12)
        f = self.F / (self.R * T)
        a = self.alpha
        # f(eta) = j0[exp(a f eta) - exp(-(1-a) f eta)] - j = 0
        g = lambda eta: j0 * (np.exp(a * f * eta) - np.exp(-(1.0 - a) * f * eta)) - j
        # bracket: eta=0 -> -j < 0 ; grow upper bound until positive
        hi = 0.5
        for _ in range(60):
            if g(hi) > 0:
                break
            hi *= 1.6
        else:
            return hi  # extreme; fall back
        return brentq(g, 0.0, hi, xtol=1e-9, rtol=1e-9, maxiter=200)

    def activation_overpotential(self, j, T):
        """Total activation loss [V] = Fe electrode + Cr electrode (Cr dominates)."""
        j0_Fe = self._arrhenius(self.j0_Fe, self.E_act_Fe, T)
        j0_Cr = self._arrhenius(self.j0_Cr, self.E_act_Cr, T)
        eta_Fe = self._bv_overpotential(j, j0_Fe, T)
        eta_Cr = self._bv_overpotential(j, j0_Cr, T)
        return eta_Fe + eta_Cr

    # ------------------------------------------------------------------
    # Ohmic overpotential
    # ------------------------------------------------------------------
    def ohmic_overpotential(self, j, T):
        """Ohmic loss [V] = j * area-specific resistance(T)."""
        # Resistance falls with temperature (ionic conduction improves) -> use
        # +E_act in Arrhenius so R decreases as T rises.
        R_area = self.R_area_ref * np.exp((self.E_act_R / self.R) * (1.0 / T - 1.0 / self.T_ref))
        return j * R_area

    # ------------------------------------------------------------------
    # Concentration overpotential
    # ------------------------------------------------------------------
    def concentration_overpotential(self, j, T, j_L):
        """Mass-transport (concentration) loss [V]."""
        if j <= 0.0:
            return 0.0
        ratio = j / j_L
        if ratio >= 0.999:
            return 5.0  # effectively starved
        return -(self.R * T) / (self.n * self.F) * np.log(1.0 - ratio)

    # ------------------------------------------------------------------
    # Parasitic hydrogen-evolution current (Cr side, charge only)
    # ------------------------------------------------------------------
    def h2_parasitic_current(self, I, soc, T):
        """Parasitic H2-evolution current magnitude [A] on the Cr electrode.

        Active only while charging (I<0). Grows with SOC (more negative potential
        at the Cr electrode as Cr2+ accumulates) and with temperature (Arrhenius).
        Always returns a value strictly less than |I| so coulombic eff in (0,1).
        """
        if I >= 0.0:
            return 0.0
        soc = self._clip_soc(soc)
        i_H2 = self._arrhenius(self.i_H2_ref, self.E_act_H2, T)   # A/cm2 at SOC=1
        I_H2 = i_H2 * self.A_cell * self.N_cells * soc           # A, scales with SOC
        return min(I_H2, 0.95 * abs(I))                          # cap < |I|

    # ------------------------------------------------------------------
    # Cell / stack terminal voltage
    # ------------------------------------------------------------------
    def terminal_voltage(self, I, soc, T, Q_flow=None):
        """Stack terminal voltage [V]. I>0 discharge, I<0 charge."""
        if Q_flow is None:
            Q_flow = self.Q_flow_ref
        j = abs(I) / (self.A_cell)                     # A/cm2 (current per cell area)
        E = self.nernst_voltage(soc, T)
        j_L = self.limiting_current(Q_flow)
        eta = (self.activation_overpotential(j, T)
               + self.ohmic_overpotential(j, T)
               + self.concentration_overpotential(j, T, j_L))
        if I >= 0.0:                                    # discharge -> lose voltage
            V_cell = E - eta
        else:                                          # charge -> add voltage
            V_cell = E + eta
        return V_cell * self.N_cells

    # ------------------------------------------------------------------
    # ODE right-hand side: state y = [SOC, T]
    # ------------------------------------------------------------------
    def _rhs(self, t, y, I_func, Q_flow):
        soc = self._clip_soc(y[0])
        T = y[1]
        I = I_func(t)

        # --- Faradaic current that actually changes SOC ---
        I_H2 = self.h2_parasitic_current(I, soc, T)
        # On charge (I<0): current into the cell is |I|, but I_H2 is "wasted" to H2.
        #   I_charge_effective = |I| - I_H2.   dSOC/dt = +I_charge_eff / Q_cap
        # On discharge (I>0): all faradaic current discharges. dSOC/dt = -I/Q_cap
        if I < 0.0:
            I_faradaic = (abs(I) - I_H2)               # >=0, charging
            dsoc_current = I_faradaic / self.Q_cap
        else:
            dsoc_current = -I / self.Q_cap

        # --- Self-discharge via membrane crossover (Fick) ---
        # flux of active species ~ k_cross * c_active / t_mem; normalise to SOC.
        c_active = min(self.c_Fe, self.c_Cr)
        A_mem_m2 = self.A_cell * 1e-4 * self.N_cells   # cm2 -> m2, all cells
        N_cross = self.k_cross * (c_active / self.t_mem)        # mol/(m2.s)
        I_cross = self.n * self.F * N_cross * A_mem_m2 * soc    # A, scales with SOC
        dsoc_self = -I_cross / self.Q_cap

        dSOC_dt = dsoc_current + dsoc_self

        # --- Thermal ODE ---
        V_term = self.terminal_voltage(I, soc, T, Q_flow)
        E_ocv_stack = self.nernst_voltage(soc, T) * self.N_cells
        # overpotential (irreversible) heating = |I| * |V - E_ocv|
        Q_over = abs(I) * abs(V_term - E_ocv_stack)
        # parasitic HER heat ~ I_H2 * E_ocv_cell (charge inefficiency dumped as heat)
        Q_par = I_H2 * self.nernst_voltage(soc, T)
        Q_gen = Q_over + Q_par
        Q_loss = self.hA_loss * (T - self.T_amb)
        dT_dt = (Q_gen - Q_loss) / (self.m_thermal * self.cp_thermal)

        return [dSOC_dt, dT_dt]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0, T0, dt, duration_s, Q_flow=None):
        """
        Simulate Fe-Cr stack dynamics (SOC + thermal ODEs).

        Parameters
        ----------
        current_A : float or callable(t)  -- stack current [A], >0 discharge, <0 charge
        soc0      : float                 -- initial state of charge [0,1]
        T0        : float                 -- initial temperature [K]
        dt        : float                 -- output time step [s]
        duration_s: float                 -- total duration [s]
        Q_flow    : float or None         -- electrolyte flow per side [m3/s]

        Returns
        -------
        dict of time-series arrays: t, soc, temperature, voltage, power,
            current, ocv, efficiency, coulombic_eff, I_H2, overpotentials{...}
        """
        if Q_flow is None:
            Q_flow = self.Q_flow_ref
        I_func = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [self._clip_soc(soc0), T0],
            t_eval=t_eval, args=(I_func, Q_flow),
            method="LSODA", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        soc = np.clip(sol.y[0], 0.001, 0.999)
        T = sol.y[1]
        N = len(t)

        voltage = np.zeros(N); power = np.zeros(N); current = np.zeros(N)
        ocv = np.zeros(N); eff = np.zeros(N); ceff = np.zeros(N); I_H2 = np.zeros(N)
        eta_act = np.zeros(N); eta_ohm = np.zeros(N); eta_conc = np.zeros(N)
        j_L = self.limiting_current(Q_flow)

        for i in range(N):
            I = I_func(t[i])
            current[i] = I
            ocv[i] = self.nernst_voltage(soc[i], T[i]) * self.N_cells
            voltage[i] = self.terminal_voltage(I, soc[i], T[i], Q_flow)
            power[i] = voltage[i] * I
            j = abs(I) / self.A_cell
            eta_act[i] = self.activation_overpotential(j, T[i]) * self.N_cells
            eta_ohm[i] = self.ohmic_overpotential(j, T[i]) * self.N_cells
            eta_conc[i] = self.concentration_overpotential(j, T[i], j_L) * self.N_cells
            I_H2[i] = self.h2_parasitic_current(I, soc[i], T[i])
            # voltage efficiency (per direction)
            if I > 0 and ocv[i] > 0:
                eff[i] = voltage[i] / ocv[i]
                ceff[i] = 1.0
            elif I < 0 and voltage[i] > 0:
                eff[i] = ocv[i] / voltage[i]
                ceff[i] = (abs(I) - I_H2[i]) / abs(I)
            else:
                eff[i] = 1.0
                ceff[i] = 1.0

        return {
            "t": t,
            "soc": soc,
            "temperature": T,
            "voltage": voltage,
            "power": power,
            "current": current,
            "ocv": ocv,
            "efficiency": eff,
            "coulombic_eff": ceff,
            "I_H2": I_H2,
            "overpotentials": {
                "activation": eta_act,
                "ohmic": eta_ohm,
                "concentration": eta_conc,
            },
        }
