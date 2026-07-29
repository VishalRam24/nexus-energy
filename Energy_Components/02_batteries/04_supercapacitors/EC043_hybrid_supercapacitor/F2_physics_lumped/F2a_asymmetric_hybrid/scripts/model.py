"""
EC043 -- Hybrid Supercapacitor (Lithium-Ion Capacitor) -- F2a Asymmetric Hybrid Model

Physics-lumped 0D model of an ASYMMETRIC hybrid supercapacitor cell:
two electrodes in series share the same charge q,

    (1) a battery-type FARADAIC electrode (e.g. LTO or pre-lithiated graphite)
        whose potential follows an OCV-vs-state-of-charge curve (intercalation),
    (2) a capacitive EDLC electrode (activated carbon) whose potential is a
        linear function of charge through its double-layer capacitance C_dl.

The terminal open-circuit voltage is the sum of the two electrode contributions,
which produces the characteristic SLOPING 2.2--3.8 V profile of a lithium-ion
capacitor -- higher energy density than a symmetric EDLC but with a non-flat
voltage, unlike a pure battery.

    V_oc(s)  =  V_min + (V_max - V_min) * [ f_far * g_far(s) + (1 - f_far) * s ]

      s          = q / Q_max                     (normalised charge, 0..1)
      g_far(s)   = faradaic OCV shape (Nernst-like intercalation branch)
      (1-f_far)*s= linear EDLC double-layer branch (V = q / C_dl, normalised)

Terminal voltage under load (positive current = DISCHARGE):

    V_term = V_oc(s) - I * R_esr(T)              (ESR / IR drop)

Charge balance ODE (leakage / self-discharge through R_leak):

    dq/dt = -I - V_oc(s) / R_leak

Lumped thermal ODE (Conway 1999; Omar 2015):

    m*cp dT/dt = Q_joule + Q_leak - hA*(T - T_amb)
    Q_joule    = I^2 * R_esr(T)                  (irreversible Joule heating)
    Q_leak     = V_oc^2 / R_leak                 (self-discharge dissipation)

Both ODEs are integrated together with scipy.integrate.solve_ivp.

References
----------
    Conway, B.E. (1999). Electrochemical Supercapacitors: Scientific
        Fundamentals and Technological Applications. Kluwer/Plenum.
    Omar, N. et al. (2015). "Lithium-ion capacitor -- advanced technology
        for rechargeable energy storage systems." IEEE Trans. Ind.
        Electron. 62(10), 6738-6745.
    Soltani, M. & Beheshti, S.H. (2021). "A comprehensive review of
        lithium-ion capacitor technology." J. Energy Storage 34, 102019.
    Firouz, Y. & Van Mierlo, J. (2017). "Lithium-ion capacitor --
        characterization and development of new electrical model."
        Electrochim. Acta 256.
"""

import numpy as np
from scipy.integrate import solve_ivp


class HybridSupercapacitorF2a:
    """Asymmetric lithium-ion capacitor -- faradaic + EDLC branches + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_max = u["Q_max"]["value"]        # C
        self.C_dl = u["C_dl"]["value"]          # F
        self.f_far = u["f_far"]["value"]        # -
        self.V_max = u["V_max"]["value"]        # V
        self.V_min = u["V_min"]["value"]        # V
        self.R_esr0 = u["R_esr"]["value"]       # Ohm (at T_amb)
        self.R_leak = u["R_leak"]["value"]      # Ohm
        self.k_far = u["k_far"]["value"]        # V (curvature)
        self.m_cell = u["m_cell"]["value"]      # kg
        self.cp_cell = u["cp_cell"]["value"]    # J/(kg.K)
        self.hA_amb = u["hA_amb"]["value"]      # W/K
        self.T_amb = u["T_amb"]["value"]        # K
        self.alpha_R = u["alpha_R"]["value"]    # 1/K

    # ------------------------------------------------------------------
    # Faradaic (battery-type) OCV shape -- intercalation electrode
    # ------------------------------------------------------------------
    def _g_far(self, s):
        """
        Dimensionless faradaic OCV shape in [0,1], Nernst-like intercalation.

        g_far(s) = s + k * [ ln(s/(1-s)) ] normalised so g_far(0)=0, g_far(1)=1.
        Produces an S-shaped sloping plateau characteristic of LTO/graphite
        intercalation, monotone increasing in s.
        """
        s = np.clip(s, 1e-6, 1.0 - 1e-6)
        # logit gives the characteristic flatten-in-the-middle, steep-at-edges shape
        raw = s + self.k_far * (np.log(s / (1.0 - s)) + 0.0)
        # normalise endpoints using fixed reference at the clipped extremes
        s0, s1 = 1e-6, 1.0 - 1e-6
        raw0 = s0 + self.k_far * np.log(s0 / (1.0 - s0))
        raw1 = s1 + self.k_far * np.log(s1 / (1.0 - s1))
        g = (raw - raw0) / (raw1 - raw0)
        return np.clip(g, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Open-circuit (equilibrium) terminal voltage
    # ------------------------------------------------------------------
    def ocv(self, q):
        """Open-circuit terminal voltage [V] vs stored charge q [C]."""
        s = np.clip(q / self.Q_max, 0.0, 1.0)
        far_branch = self.f_far * self._g_far(s)        # battery-type electrode
        edlc_branch = (1.0 - self.f_far) * s            # double-layer electrode
        return self.V_min + (self.V_max - self.V_min) * (far_branch + edlc_branch)

    # ------------------------------------------------------------------
    # Temperature-dependent ESR
    # ------------------------------------------------------------------
    def esr(self, T):
        """ESR [Ohm] with linear temperature coefficient (ESR drops as T rises)."""
        R = self.R_esr0 * (1.0 + self.alpha_R * (T - self.T_amb))
        return max(R, 1e-5)

    # ------------------------------------------------------------------
    # Terminal voltage under load (positive I = discharge)
    # ------------------------------------------------------------------
    def terminal_voltage(self, q, I, T):
        """Terminal voltage [V] = OCV - I*ESR (positive I discharges)."""
        return self.ocv(q) - I * self.esr(T)

    # ------------------------------------------------------------------
    # Stored electrostatic + faradaic energy (state function, by integration)
    # ------------------------------------------------------------------
    def stored_energy(self, q):
        """Energy [J] stored = integral_0^q OCV(q') dq' (reversible energy)."""
        q = float(np.clip(q, 0.0, self.Q_max))
        qs = np.linspace(0.0, q, 200)
        return float(np.trapz(self.ocv(qs), qs))

    # ------------------------------------------------------------------
    # Coupled charge + thermal ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, I_func):
        q, T = y
        I = I_func(t)
        q = np.clip(q, 0.0, self.Q_max)
        V_oc = self.ocv(q)
        R = self.esr(T)

        # charge balance: applied discharge current + leakage self-discharge
        dqdt = -I - V_oc / self.R_leak

        # thermal balance
        Q_joule = I * I * R
        Q_leak = V_oc * V_oc / self.R_leak
        dTdt = (Q_joule + Q_leak - self.hA_amb * (T - self.T_amb)) / (self.m_cell * self.cp_cell)

        return [dqdt, dTdt]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, q0_C, T0_K, dt, duration_s):
        """
        Simulate the coupled charge/thermal dynamics.

        Parameters
        ----------
        current_A : float or callable(t)
            Load current [A]; positive = discharge, negative = charge.
        q0_C : float
            Initial stored charge [C].
        T0_K : float
            Initial cell temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation time [s].

        Returns
        -------
        dict with time-series arrays:
            t, charge, soc, v_oc, v_terminal, power, efficiency,
            temperature, energy_J
        """
        I_func = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [q0_C, T0_K],
            t_eval=t_eval, args=(I_func,),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t_out = sol.t
        q_out = np.clip(sol.y[0], 0.0, self.Q_max)
        T_out = sol.y[1]
        N = len(t_out)

        v_oc = np.zeros(N)
        v_term = np.zeros(N)
        power = np.zeros(N)
        efficiency = np.zeros(N)
        energy = np.zeros(N)

        for i in range(N):
            I = I_func(t_out[i])
            v_oc[i] = self.ocv(q_out[i])
            v_term[i] = self.terminal_voltage(q_out[i], I, T_out[i])
            power[i] = v_term[i] * I
            # round-trip-ish voltaic efficiency at this instant: terminal/OCV on
            # discharge, OCV/terminal on charge -- always in (0,1) for I != 0
            if I > 0:      # discharge: deliver less than OCV
                efficiency[i] = v_term[i] / v_oc[i] if v_oc[i] > 0 else 0.0
            elif I < 0:    # charge: must push above OCV
                efficiency[i] = v_oc[i] / v_term[i] if v_term[i] > 0 else 0.0
            else:
                efficiency[i] = 1.0
            efficiency[i] = min(max(efficiency[i], 0.0), 1.0)
            energy[i] = self.stored_energy(q_out[i])

        soc = q_out / self.Q_max

        return {
            "t": t_out,
            "charge": q_out,
            "soc": soc,
            "v_oc": v_oc,
            "v_terminal": v_term,
            "power": power,
            "efficiency": efficiency,
            "temperature": T_out,
            "energy_J": energy,
        }
