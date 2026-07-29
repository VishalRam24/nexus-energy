"""
EC026 -- Lithium-Air Battery (Li-O2 / Li-Air) -- F2a Physics-Lumped
Pore-Passivation Electrochemical / Thevenin Hybrid with Thermal ODE

0D first-principles model of an aprotic Li-O2 cell. The discharge reaction

        2 Li + O2  ->  Li2O2            (E_eq ~ 2.96 V vs Li/Li+)

deposits *electronically insulating* lithium peroxide inside the porous
carbon air-cathode. As discharge proceeds the Li2O2 progressively fills and
finally clogs the cathode pores; once the pore-fill fraction reaches a
critical value, oxygen / Li+ transport collapses and the cell suffers
"sudden death" -- the practical capacity is set by pore saturation, NOT by
the lithium inventory (Sandhu, Fellner & Brooks 2007; Bao et al. 2015).

Because both the oxygen-reduction reaction (ORR, discharge) and the
oxygen-evolution reaction (OER, charge) are kinetically sluggish, and OER is
markedly worse than ORR, the round-trip voltage gap is very large
(~1 V hysteresis): V_discharge is depressed ~0.3 V below E_eq while
V_charge sits ~0.7-1.3 V above E_eq (Lu et al. 2013; McCloskey et al. 2011).

State vector (integrated with scipy.integrate.solve_ivp):
    y[0] = SOC          state of charge in [0, 1]
    y[1] = theta        Li2O2 pore-fill fraction in [0, 1] (discharge fills, charge clears)
    y[2] = V_rc         voltage across the Thevenin R_ct||C branch [V]
    y[3] = T            cell temperature [K]

Voltage model at each instant (I>0 discharge, I<0 charge):
    E_eq(SOC,T)  = E_eq_ref + dE/dSOC term + dOCV_dT*(T-T_ref)     (thermodynamic)
    eta_kin(I,T) = ORR/OER Arrhenius+Tafel kinetic overpotential   (sign by direction)
    eta_pass     = passivation overpotential, grows with theta (discharge only)
    V_ohm        = I * R0(T)                                       (film/ohmic)
    V_rc         = relaxation branch (RC ODE)
    V_terminal   = E_eq - sign(I)*eta_kin - eta_pass - I*R0 - V_rc

Thermal ODE:
    m*cp dT/dt = Q_gen - Q_loss
    Q_gen  = I*(E_eq - V_terminal)  +  I*T*dOCV_dT      (irrev. + reversible/entropic)
    Q_loss = hA*(T - T_amb)

Enforced physics:
    * Coulomb conservation: dSOC/dt = -I/(C*3600); pore fill tied to same charge.
    * V_discharge < E_eq < V_charge by construction (kinetic + passivation gaps).
    * Capacity cutoff at pore saturation (theta -> pore_cutoff_frac on discharge).
    * 0 < round-trip / voltaic efficiency < 1.

References:
    Laoire, Mukerjee, Abraham, Plichta & Hendrickson (2010),
        J. Electrochem. Soc. 157(7), A821-A826.
    Abraham & Jiang (1996), J. Electrochem. Soc. 143(1), 1-5.
    Viswanathan, Thygesen, Norskov et al. (2011), J. Chem. Phys. 135, 214704.
    Lu, Lee, Liang, Shao-Horn et al. (2013), Nat. Chem. 5, 527-538 (review).
    McCloskey et al. (2011), J. Phys. Chem. Lett. 2(10), 1161-1166.
    Sandhu, Fellner & Brooks (2007), J. Power Sources 164(1), 365-371
        (discharge model with cathode pore clogging by Li2O2).
    Bao, Xu, Zhang et al. (2015), J. Electrochem. Soc. 162(7), A1602.
    Read (2002), J. Electrochem. Soc. 149(9), A1190 (O2 solubility vs T).
"""

import numpy as np
from scipy.integrate import solve_ivp


class LiAirF2a:
    """Aprotic Li-O2 cell -- physics-lumped pore-passivation + thermal ODE."""

    # Finite ceiling on the passivation overpotential [V]. Far larger than the
    # available voltage window, so the terminal voltage still clips to v_min,
    # but bounded enough to keep the ODE non-stiff near pore saturation.
    ETA_PASS_MAX = 50.0

    def __init__(self, params: dict):
        u = params["unit"]
        self.E_eq_ref = u["E_eq_ref"]["value"]
        self.v_max = u["voltage_max"]["value"]
        self.v_min = u["voltage_min"]["value"]

        self.capacity_ref = u["capacity_ref"]["value"]      # Ah
        self.Q_pore_max = u["Q_pore_max"]["value"]          # Ah
        self.pore_cutoff = u["pore_cutoff_frac"]["value"]

        self.R_ref = u["R_ref"]["value"]                    # Ohm
        self.tau_ct = u["tau_ct"]["value"]                  # s
        self.R_ct_ref = u["R_ct_ref"]["value"]              # Ohm

        self.eta_orr = u["eta_orr"]["value"]                # V
        self.eta_oer = u["eta_oer"]["value"]                # V
        self.i_ref = u["i_ref"]["value"]                    # A
        self.tafel = u["tafel_slope"]["value"]              # V/decade
        self.k_passiv = u["k_passiv"]["value"]              # V

        self.T_ref = u["T_ref"]["value"]                    # K
        self.E_a = u["E_a"]["value"]                        # J/mol
        self.dOCV_dT = u["dOCV_dT"]["value"]                # V/K
        self.R_gas = u["R_gas"]["value"]
        self.n_e = u["n_e"]["value"]
        self.F = u["F_const"]["value"]

        self.m_cell = u["m_cell"]["value"]
        self.cp_cell = u["cp_cell"]["value"]
        self.hA_amb = u["hA_amb"]["value"]
        self.T_amb = u["T_amb"]["value"]

    # ------------------------------------------------------------------
    # Thermodynamic equilibrium potential E_eq(SOC, T)
    # ------------------------------------------------------------------
    def equilibrium_voltage(self, soc, T):
        """Equilibrium (open-circuit) cell potential [V].

        Mild SOC dependence around the Li2O2 plateau plus the (large, negative)
        entropic temperature correction. This is the reversible thermodynamic
        potential -- discharge stays below it, charge above it.
        """
        soc = np.clip(soc, 0.0, 1.0)
        # gentle plateau: small slope so E_eq stays near the 2.96 V anchor
        e_soc = self.E_eq_ref + 0.12 * (soc - 0.5)
        return e_soc + self.dOCV_dT * (T - self.T_ref)

    # ------------------------------------------------------------------
    # Arrhenius temperature factor for kinetics / resistance
    # ------------------------------------------------------------------
    def _arrhenius(self, T):
        return np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def ohmic_resistance(self, T):
        """Series ohmic resistance R0(T) [Ohm] (Li2O2 film, Arrhenius)."""
        return self.R_ref * self._arrhenius(T)

    def charge_transfer_resistance(self, T):
        """RC-branch charge-transfer resistance R_ct(T) [Ohm]."""
        return self.R_ct_ref * self._arrhenius(T)

    # ------------------------------------------------------------------
    # Kinetic overpotential -- ORR (discharge) vs OER (charge) asymmetry
    # ------------------------------------------------------------------
    def kinetic_overpotential(self, current, T):
        """Magnitude of ORR/OER kinetic overpotential [V] (always >= 0).

        OER (charge) overpotential >> ORR (discharge) overpotential, which is
        the origin of the ~1 V Li-air round-trip hysteresis. Tafel-type
        logarithmic current dependence with Arrhenius temperature activation.
        """
        I = float(current)
        if I == 0.0:
            return 0.0
        base = self.eta_oer if I < 0.0 else self.eta_orr   # charge=OER, discharge=ORR
        # Tafel current dependence (relative to reference current), >=0
        tafel_term = self.tafel * np.log10(max(abs(I), 1e-6) / self.i_ref)
        # thermal activation: colder -> larger overpotential
        therm = self._arrhenius(T)
        eta = (base + tafel_term) * therm
        return max(eta, 0.0)

    # ------------------------------------------------------------------
    # Passivation overpotential from Li2O2 pore filling (discharge only)
    # ------------------------------------------------------------------
    def passivation_overpotential(self, theta, current):
        """Extra discharge overpotential [V] from insulating Li2O2 film.

        Grows sharply as the pore-fill fraction theta approaches the cutoff,
        diverging at sudden death. Zero on charge (film is being stripped).
        """
        if current <= 0.0:
            return 0.0
        x = np.clip(theta / self.pore_cutoff, 0.0, 0.999)
        # film resistance ~ x/(1-x): negligible early, climbs steeply near saturation.
        # Capped at a finite ceiling (ETA_PASS_MAX) so the term stays bounded inside
        # the ODE -- physically the terminal voltage is clipped to v_min anyway, so an
        # unbounded value would only make the system numerically stiff with no effect
        # on the (clipped) output.
        return min(self.k_passiv * x / (1.0 - x), self.ETA_PASS_MAX)

    # ------------------------------------------------------------------
    # Terminal voltage given full instantaneous state
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, theta, V_rc, T, current):
        """Terminal voltage [V]. I>0 discharge (V<E_eq), I<0 charge (V>E_eq)."""
        E_eq = self.equilibrium_voltage(soc, T)
        I = float(current)
        eta_kin = self.kinetic_overpotential(I, T)
        eta_pass = self.passivation_overpotential(theta, I)
        R0 = self.ohmic_resistance(T)
        # sign(I): discharge subtracts overpotentials, charge adds them
        s = np.sign(I) if I != 0.0 else 0.0
        V = E_eq - s * eta_kin - eta_pass - I * R0 - V_rc
        return float(np.clip(V, self.v_min, self.v_max))

    # ------------------------------------------------------------------
    # Coupled state derivatives
    # ------------------------------------------------------------------
    def derivatives(self, t, y, current_fn):
        soc, theta, V_rc, T = y
        I = float(current_fn(t))

        # --- gate current at limits (Coulomb conservation + pore cutoff) ---
        # stop discharge if empty or pores saturated; stop charge if full or clean
        if I > 0.0 and (soc <= 0.0 or theta >= self.pore_cutoff):
            I = 0.0
        if I < 0.0 and (soc >= 1.0 or theta <= 0.0):
            I = 0.0

        # SOC: Coulomb counting against nominal capacity
        dsoc = -I / (self.capacity_ref * 3600.0)

        # Pore fill: discharge deposits Li2O2 (theta up), charge removes (theta down).
        # Fill rate tied to the SAME charge throughput -> conservation.
        dtheta = I / (self.Q_pore_max * 3600.0)

        # RC relaxation branch: dV_rc/dt = I*R_ct/tau - V_rc/tau
        R_ct = self.charge_transfer_resistance(T)
        dvrc = (I * R_ct - V_rc) / self.tau_ct

        # Thermal ODE
        V = self.terminal_voltage(soc, theta, V_rc, T, I)
        E_eq = self.equilibrium_voltage(soc, T)
        q_irrev = I * (E_eq - V)              # >=0: dissipated overpotential power
        q_rev = I * T * self.dOCV_dT          # entropic (sign depends on direction)
        Q_gen = q_irrev + q_rev
        Q_loss = self.hA_amb * (T - self.T_amb)
        dT = (Q_gen - Q_loss) / (self.m_cell * self.cp_cell)

        return [dsoc, dtheta, dvrc, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current, soc0=1.0, theta0=0.0, T0=None,
                 dt=10.0, duration_s=3600.0):
        """Simulate Li-O2 dynamics with the coupled pore/thermal ODE.

        Parameters
        ----------
        current : float or callable(t)
            Cell current [A]. Positive = discharge, negative = charge.
        soc0 : float        initial state of charge [0,1]
        theta0 : float      initial Li2O2 pore-fill fraction [0,1]
        T0 : float          initial temperature [K] (default T_amb)
        dt : float          output time step [s]
        duration_s : float  total duration [s]

        Returns
        -------
        dict of time series: t, voltage, current, soc, theta (pore fill),
            equilibrium_voltage, power, efficiency, temperature, overpotentials.
        """
        if T0 is None:
            T0 = self.T_amb
        cur_fn = current if callable(current) else (lambda t: current)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [float(np.clip(soc0, 0.0, 1.0)),
              float(np.clip(theta0, 0.0, 1.0)),
              0.0, float(T0)]

        sol = solve_ivp(
            self.derivatives, (0.0, duration_s), y0,
            t_eval=t_eval, args=(cur_fn,),
            method="LSODA", rtol=1e-6, atol=1e-8, max_step=dt,
        )

        t_out = sol.t
        soc = np.clip(sol.y[0], 0.0, 1.0)
        theta = np.clip(sol.y[1], 0.0, 1.0)
        V_rc = sol.y[2]
        T_out = sol.y[3]
        N = len(t_out)

        voltage = np.zeros(N)
        E_eq = np.zeros(N)
        power = np.zeros(N)
        eff = np.zeros(N)
        I_arr = np.zeros(N)
        eta_kin = np.zeros(N)
        eta_pass = np.zeros(N)
        v_ohm = np.zeros(N)

        for i in range(N):
            I = float(cur_fn(t_out[i]))
            # apply the same gating used in the ODE for reporting consistency
            if I > 0.0 and (soc[i] <= 1e-6 or theta[i] >= self.pore_cutoff):
                I = 0.0
            if I < 0.0 and (soc[i] >= 1.0 - 1e-6 or theta[i] <= 0.0):
                I = 0.0
            I_arr[i] = I
            E_eq[i] = self.equilibrium_voltage(soc[i], T_out[i])
            voltage[i] = self.terminal_voltage(soc[i], theta[i], V_rc[i], T_out[i], I)
            power[i] = voltage[i] * I
            eta_kin[i] = self.kinetic_overpotential(I, T_out[i])
            eta_pass[i] = self.passivation_overpotential(theta[i], I)
            v_ohm[i] = abs(I) * self.ohmic_resistance(T_out[i])
            # voltaic efficiency relative to equilibrium (in (0,1) under load)
            if I > 0.0:                      # discharge: V/E_eq
                eff[i] = voltage[i] / E_eq[i] if E_eq[i] > 0 else 0.0
            elif I < 0.0:                    # charge: E_eq/V
                eff[i] = E_eq[i] / voltage[i] if voltage[i] > 0 else 0.0
            else:
                eff[i] = 0.0

        return {
            "t": t_out,
            "voltage": voltage,
            "current": I_arr,
            "soc": soc,
            "theta": theta,
            "equilibrium_voltage": E_eq,
            "power": power,
            "efficiency": eff,
            "temperature": T_out,
            "overpotentials": {
                "E_eq": E_eq,
                "kinetic": eta_kin,
                "passivation": eta_pass,
                "ohmic": v_ohm,
                "rc_branch": V_rc,
            },
        }

    # ------------------------------------------------------------------
    # Round-trip voltaic efficiency from a discharge/charge V pair
    # ------------------------------------------------------------------
    def round_trip_voltage_gap(self, current_mag=1.0, soc=0.5, theta=0.1, T=None):
        """Return (V_discharge, V_charge, gap) at matched |I| -- the hysteresis."""
        if T is None:
            T = self.T_ref
        Vd = self.terminal_voltage(soc, theta, 0.0, T, +abs(current_mag))
        Vc = self.terminal_voltage(soc, theta, 0.0, T, -abs(current_mag))
        return Vd, Vc, (Vc - Vd)
