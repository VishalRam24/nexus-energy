"""
EC041 -- EDLC Supercapacitor -- F2a Zubieta 3-Branch Equivalent Circuit

Physics-lumped first-principles dynamic model of an electric double-layer
capacitor (EDLC / supercapacitor) using the classic three-branch equivalent
circuit of Zubieta & Bonert (2000), coupled to a lumped thermal ODE.

Equivalent circuit (3 parallel branches + leakage), state = [v1, v2, v3, T]:

    Immediate branch : Ri --- C_imm(v1) = C0 + kv*v1   (voltage-dependent EDL)
    Delayed branch   : Rd --- Cd                        (charge redistribution, ~seconds)
    Long-term branch : Rl --- Cl                        (redistribution, ~minutes-hours)
    Leakage          : R_leak in parallel (self-discharge)

The immediate branch is the only one wired directly to the terminals; the
delayed and long-term branches exchange charge with v1 through Rd, Rl. This
captures the characteristic EDLC charge redistribution (terminal voltage
"recovers"/"sags" after a current step is removed) -- the key physics the
single-RC F1 model cannot reproduce.

Voltage-dependent differential capacitance (Zubieta & Bonert 2000):
    C_imm(v1) = C0 + kv * v1            [F]
Charge on the immediate branch (nonlinear capacitor):
    q1(v1) = C0*v1 + 0.5*kv*v1^2        [C]   (so dq1 = (C0+kv*v1) dv1)

State equations (KCL at each capacitor node):
    Immediate node (carries the external/terminal current I):
        I = i_imm + (v1/R_leak) + i_d + i_l
        i_imm = C_imm(v1) * dv1/dt
      => dv1/dt = ( I - v1/R_leak - i_d - i_l ) / (C0 + kv*v1)
      where  i_d = (v1 - v2)/Rd ,  i_l = (v1 - v3)/Rl
    Delayed node:    Cd * dv2/dt =  i_d = (v1 - v2)/Rd
    Long-term node:  Cl * dv3/dt =  i_l = (v1 - v3)/Rl

Sign convention: I > 0 charges the device (current into immediate node).

ESR temperature dependence (Arrhenius), Berrueta et al. (2019):
    Ri(T) = Ri_ref * exp( E_a/R_gas * (1/T - 1/T_ref) )

Terminal voltage (immediate branch only conducts external current):
    V_term = v1 + I * Ri(T)              (I>0 charging raises terminal V)

Lumped thermal ODE (Joule heating in ESR + redistribution resistors):
    m*cp * dT/dt = Q_gen - Q_amb
    Q_gen = I^2*Ri(T) + i_d^2*Rd + i_l^2*Rl + v1^2/R_leak   [W]
    Q_amb = hA*(T - T_amb)
EDLC stores energy electrostatically (no faradaic reaction) -> heat is purely
ohmic/Joule, no reversible entropic term.

Stored energy (sum over the three nonlinear/linear branch capacitors):
    E = (0.5*C0*v1^2 + (1/3)*kv*v1^3) + 0.5*Cd*v2^2 + 0.5*Cl*v3^2   [J]
(the immediate-branch term is the integral of q1(v1) dv1).

References:
    Zubieta, L. & Bonert, R. (2000). "Characterization of Double-Layer
        Capacitors for Power Electronics Applications." IEEE Trans. Ind.
        Appl., 36(1), 199-205.
    Conway, B. E. (1999). Electrochemical Supercapacitors. Kluwer/Plenum.
    Rafik, F. et al. (2007). "Frequency, thermal and voltage supercapacitor
        characterization and modeling." J. Power Sources, 165, 928-934.
    Berrueta, A. et al. (2019). "Supercapacitors: Electrical Characteristics,
        Modeling, Applications, and Future Trends." IEEE Trans. Ind.
        Electron., 66(6), 4750-4759.
"""

import numpy as np
from scipy.integrate import solve_ivp


class EDLC_F2a:
    """EDLC supercapacitor -- Zubieta 3-branch ECM with C(V) and thermal ODE."""

    R_gas = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.C0 = u["C0"]["value"]            # F
        self.kv = u["kv"]["value"]            # F/V
        self.Ri_ref = u["Ri"]["value"]        # Ohm (ESR at T_ref)
        self.Rd = u["Rd"]["value"]            # Ohm
        self.Cd = u["Cd"]["value"]            # F
        self.Rl = u["Rl"]["value"]            # Ohm
        self.Cl = u["Cl"]["value"]            # F
        self.R_leak = u["R_leak"]["value"]    # Ohm
        self.v_max = u["v_max"]["value"]      # V
        self.v_min = u["v_min"]["value"]      # V

        self.T_ref = u["T_ref"]["value"]      # K
        self.E_a = u["E_a_esr"]["value"]      # J/mol
        self.m_cell = u["m_cell"]["value"]    # kg
        self.cp_cell = u["cp_cell"]["value"]  # J/(kg.K)
        self.hA = u["hA_amb"]["value"]        # W/K
        self.T_amb = u["T_amb"]["value"]      # K

    # ------------------------------------------------------------------
    # Constitutive relations
    # ------------------------------------------------------------------
    def C_imm(self, v1):
        """Voltage-dependent immediate-branch differential capacitance [F]."""
        v1 = np.asarray(v1, dtype=float)
        return self.C0 + self.kv * v1

    def Ri(self, T):
        """ESR with Arrhenius temperature dependence [Ohm]."""
        T = np.asarray(T, dtype=float)
        return self.Ri_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def stored_energy(self, v1, v2, v3):
        """Total electrostatic energy across the three branches [J].

        Immediate branch is a nonlinear capacitor q1 = C0*v1 + 0.5*kv*v1^2,
        so its energy is integral(v1 dq1) = 0.5*C0*v1^2 + (1/3)*kv*v1^3.
        """
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        v3 = np.asarray(v3, dtype=float)
        E_imm = 0.5 * self.C0 * v1**2 + (1.0 / 3.0) * self.kv * v1**3
        E_d = 0.5 * self.Cd * v2**2
        E_l = 0.5 * self.Cl * v3**2
        return E_imm + E_d + E_l

    def terminal_voltage(self, v1, current, T):
        """Terminal voltage [V]. current>0 = charging (raises terminal V)."""
        v1 = np.asarray(v1, dtype=float)
        current = np.asarray(current, dtype=float)
        return v1 + current * self.Ri(T)

    def heat_generation(self, v1, v2, v3, current, T):
        """Total Joule heat dissipation [W] (ESR + redistribution + leakage)."""
        i_d = (v1 - v2) / self.Rd
        i_l = (v1 - v3) / self.Rl
        q = (current**2 * self.Ri(T)
             + i_d**2 * self.Rd
             + i_l**2 * self.Rl
             + v1**2 / self.R_leak)
        return q

    def power(self, v1, current, T):
        """Terminal electrical power [W]. Positive = delivered on discharge."""
        # discharge current (I_load = -current) times terminal voltage
        return self.terminal_voltage(v1, current, T) * (-np.asarray(current, dtype=float))

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def derivatives(self, v1, v2, v3, T, current):
        """State derivatives [dv1, dv2, dv3, dT]."""
        i_d = (v1 - v2) / self.Rd
        i_l = (v1 - v3) / self.Rl
        i_leak = v1 / self.R_leak

        C1 = self.C0 + self.kv * v1
        # KCL at immediate node: external current splits to imm/leak/delayed/long
        dv1 = (current - i_leak - i_d - i_l) / C1
        dv2 = i_d / self.Cd
        dv3 = i_l / self.Cl

        Q_gen = (current**2 * self.Ri(T)
                 + i_d**2 * self.Rd
                 + i_l**2 * self.Rl
                 + v1**2 / self.R_leak)
        Q_amb = self.hA * (T - self.T_amb)
        dT = (Q_gen - Q_amb) / (self.m_cell * self.cp_cell)
        return dv1, dv2, dv3, dT

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, v0_V=0.0, T0_K=None, dt=0.1, duration_s=60.0):
        """
        Integrate the 3-branch + thermal ODE with scipy.solve_ivp.

        Parameters
        ----------
        current_A : float or callable(t)
            Terminal current [A]; >0 charges, <0 discharges.
        v0_V : float
            Initial voltage on all branches [V] (uniform start).
        T0_K : float or None
            Initial temperature [K]; defaults to ambient.
        dt : float
            Output time step [s].
        duration_s : float
            Total duration [s].

        Returns
        -------
        dict of time series: t, v_terminal, v1, v2, v3, current,
            energy_J, power_W, temperature, esr_Ohm, heat_W.
        """
        if T0_K is None:
            T0_K = self.T_amb
        _I = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            v1, v2, v3, T = y
            dv1, dv2, dv3, dT = self.derivatives(v1, v2, v3, T, _I(t))
            return [dv1, dv2, dv3, dT]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [v0_V, v0_V, v0_V, T0_K],
            t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9,
            max_step=max(dt, 1e-3),
        )

        t_out = sol.t
        v1 = sol.y[0]
        v2 = sol.y[1]
        v3 = sol.y[2]
        T_out = sol.y[3]
        N = len(t_out)

        I_arr = np.array([_I(t) for t in t_out], dtype=float)
        esr = self.Ri(T_out)
        v_term = v1 + I_arr * esr
        energy = self.stored_energy(v1, v2, v3)
        power = v_term * (-I_arr)
        heat = self.heat_generation(v1, v2, v3, I_arr, T_out)

        return {
            "t": t_out,
            "v_terminal": v_term,
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "current": I_arr,
            "energy_J": energy,
            "power_W": power,
            "temperature": T_out,
            "esr_Ohm": esr,
            "heat_W": heat,
        }

    # ------------------------------------------------------------------
    # Round-trip efficiency helper (charge then discharge same |I|)
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, I_mag, v_top=2.5, dt=0.05):
        """
        Galvanostatic charge from 0 to v_top then discharge back, with the
        immediate branch only (fast cycle). Returns (eff, E_in, E_out).
        eff = energy delivered on discharge / energy supplied on charge.
        """
        # Charge phase: integrate until v1 reaches v_top
        v1 = 0.0
        E_in = 0.0
        T = self.T_amb
        max_steps = int(2e6)
        steps = 0
        while v1 < v_top and steps < max_steps:
            esr = self.Ri(T)
            v_term = v1 + I_mag * esr
            E_in += v_term * I_mag * dt
            C1 = self.C0 + self.kv * v1
            v1 += (I_mag - v1 / self.R_leak) / C1 * dt
            steps += 1
        # Discharge phase
        E_out = 0.0
        steps = 0
        while v1 > 0.0 and steps < max_steps:
            esr = self.Ri(T)
            v_term = v1 - I_mag * esr
            if v_term <= 0:
                break
            E_out += v_term * I_mag * dt
            C1 = self.C0 + self.kv * v1
            v1 += (-I_mag - v1 / self.R_leak) / C1 * dt
            steps += 1
        eff = E_out / E_in if E_in > 0 else 0.0
        return eff, E_in, E_out
