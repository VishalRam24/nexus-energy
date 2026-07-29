"""
EC023 -- LMO Battery (Lithium Manganese Oxide spinel) -- F2a Thevenin ECM

Physics-lumped equivalent-circuit model (1-RC or 2-RC, selectable) with
Coulomb-counted SOC, LMO-specific OCV(SOC), Arrhenius temperature-dependent
resistances, and a lumped Bernardi thermal energy balance. All dynamic states
are advanced together by scipy.integrate.solve_ivp.

State vector  y = [SOC, V_rc1, V_rc2, T]  (V_rc2 = 0 when n_rc = 1)

Governing ODEs
--------------
1. Coulomb counting (state of charge):
       dSOC/dt = -(eta(I) * I) / (Q_cap * 3600)
   where I > 0 is discharge, I < 0 is charge, and the Coulombic efficiency
   eta is applied only to charging current (0 < eta_c < 1) so that one full
   charge/discharge round-trip loses (1 - eta_c) of the charge -> Coulomb
   conservation with a single, well-defined loss channel.

2. RC (polarisation) branches -- first-order relaxation:
       dV_rc_k/dt = -V_rc_k / (R_k(T) * C_k) + I / C_k
   The steady-state of branch k is I*R_k(T); the transient time constant is
   tau_k = R_k(T) * C_k. (Thevenin / dual-polarisation model.)

3. Bernardi lumped thermal energy balance:
       m*cp dT/dt = Q_gen - hA*(T - T_amb)
       Q_gen = I*(OCV - V_term)            (irreversible / Joule + polarisation)
             + I * T * dOCV/dT             (reversible / entropic)
   This is the reduced single-cell form of Bernardi et al. (1985).

Terminal voltage (Thevenin):
       V_term = OCV(SOC) - I*R0(T) - sum_k V_rc_k

OCV(SOC):
   5th-order polynomial fitted to spinel LiMn2O4 OCV (Liaw et al. 2003).
   Spinel shows the characteristic two-stage (dual-plateau) Li (de)intercalation
   near ~4.05 V and ~3.95 V; the polynomial reproduces the overall monotone
   3.0 -> 4.2 V rise and the gentle plateau region.

Resistances R0, R1, R2 share an Arrhenius temperature law:
       R(T) = R_ref * exp( (E_a/R_gas) * (1/T - 1/T_ref) )
   so resistance rises as the cell cools (E_a > 0).

Aging note (not integrated in F2a -- see F2c):
   LMO capacity fade is driven by Mn(III) disproportionation
   (2 Mn3+ -> Mn2+ + Mn4+) and Mn2+ dissolution into the electrolyte at high
   temperature and high SOC, which poisons the anode SEI. The conservative
   upper temperature bound (60 C) reflects this. Cf. Vetter et al. (2005).

References
----------
    Liaw, B.Y. et al. (2003). J. Power Sources 119-121, 874-882.
    Hu, X., Li, S., Peng, H. (2012). J. Power Sources 198, 359-367.
        (comparison of 12 equivalent-circuit models; 1-RC/2-RC Thevenin).
    Bernardi, D. et al. (1985). J. Electrochem. Soc. 132(1), 5-12.
        (general energy balance for battery systems).
    Forgez, C. et al. (2010). J. Power Sources 195, 2961-2968.
        (lumped thermal model + entropic coefficient of a Li-ion cell).
    Thomas, K.E., Newman, J. (2003). J. Electrochem. Soc. 150, A176.
    Vetter, J. et al. (2005). J. Power Sources 147, 269-281. (aging mechanisms).
"""

import numpy as np
from scipy.integrate import solve_ivp


class LMO_Thevenin_F2a:
    """LMO spinel Thevenin equivalent-circuit model with thermal coupling."""

    R_gas = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_cap = u["capacity_ref_Ah"]["value"]      # Ah
        self.v_max = u["v_max"]["value"]
        self.v_min = u["v_min"]["value"]

        self.R0_ref = u["R0_ref"]["value"]              # Ohm
        self.R1_ref = u["R1_ref"]["value"]
        self.C1 = u["C1"]["value"]                      # F
        self.R2_ref = u["R2_ref"]["value"]
        self.C2 = u["C2"]["value"]
        self.n_rc = int(u["n_rc"]["value"])

        self.eta_c = u["coulombic_efficiency"]["value"]  # (0,1)
        self.T_ref = u["T_ref"]["value"]                # K
        self.E_a = u["E_a"]["value"]                    # J/mol
        self.dOCV_dT = u["dOCV_dT"]["value"]            # V/K

        self.m_cell = u["m_cell"]["value"]              # kg
        self.cp_cell = u["cp_cell"]["value"]            # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]            # W/K
        self.T_amb = u["T_ambient"]["value"]            # K

        self.ocv_coeff = np.array([u[f"ocv_a{i}"]["value"] for i in range(6)])

    # ------------------------------------------------------------------
    # Open-circuit voltage (spinel dual-plateau)
    # ------------------------------------------------------------------
    def ocv(self, soc):
        """Open-circuit voltage [V] as a function of SOC in [0, 1]."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([s ** i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def docv_dsoc(self, soc):
        """dOCV/dSOC [V] (analytic derivative of the OCV polynomial)."""
        s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([i * s ** (i - 1) if i >= 1 else np.zeros_like(s)
                           for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    # ------------------------------------------------------------------
    # Arrhenius resistances
    # ------------------------------------------------------------------
    def _arrhenius(self, R_ref, T):
        return R_ref * np.exp((self.E_a / self.R_gas) * (1.0 / T - 1.0 / self.T_ref))

    def R0(self, T):
        return self._arrhenius(self.R0_ref, T)

    def R1(self, T):
        return self._arrhenius(self.R1_ref, T)

    def R2(self, T):
        return self._arrhenius(self.R2_ref, T)

    # ------------------------------------------------------------------
    # Terminal voltage
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, current, T, v_rc1=0.0, v_rc2=0.0):
        """
        Thevenin terminal voltage [V].

        current > 0 -> discharge (V_term < OCV); current < 0 -> charge.
        v_rc1, v_rc2 are the instantaneous RC polarisation voltages.
        """
        v = self.ocv(soc) - current * self.R0(T) - v_rc1
        if self.n_rc >= 2:
            v = v - v_rc2
        return v

    # ------------------------------------------------------------------
    # Heat generation -- Bernardi (1985)
    # ------------------------------------------------------------------
    def heat_generation(self, soc, current, T, v_rc1=0.0, v_rc2=0.0):
        """
        Lumped Bernardi heat generation [W].

        Q = I*(OCV - V_term)        irreversible (ohmic + polarisation, >= 0)
          + I * T * dOCV/dT         reversible (entropic, sign depends on I)
        """
        v_term = self.terminal_voltage(soc, current, T, v_rc1, v_rc2)
        q_irrev = current * (self.ocv(soc) - v_term)          # = I^2 R0 + I*sum Vrc
        q_rev = current * T * self.dOCV_dT
        return q_irrev + q_rev

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, current_fn):
        soc, v_rc1, v_rc2, T = y
        I = current_fn(t)

        # 1) Coulomb counting with Coulombic efficiency on charge
        #    discharge (I>0): full Coulombs leave; charge (I<0): only eta_c stored
        eff_I = I if I >= 0 else I * self.eta_c
        dsoc = -eff_I / (self.Q_cap * 3600.0)

        # 2) RC branches
        tau1 = self.R1(T) * self.C1
        dv_rc1 = -v_rc1 / tau1 + I / self.C1
        if self.n_rc >= 2:
            tau2 = self.R2(T) * self.C2
            dv_rc2 = -v_rc2 / tau2 + I / self.C2
        else:
            dv_rc2 = 0.0

        # 3) Thermal balance (Bernardi)
        Q_gen = self.heat_generation(soc, I, T, v_rc1, v_rc2)
        Q_cool = self.hA_cool * (T - self.T_amb)
        dT = (Q_gen - Q_cool) / (self.m_cell * self.cp_cell)

        return [dsoc, dv_rc1, dv_rc2, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.8, T0=298.15, dt=1.0, duration_s=600.0):
        """
        Simulate the coupled ECM + thermal dynamics.

        Parameters
        ----------
        current_A : float or callable(t)
            Cell current [A]; positive = discharge, negative = charge.
        soc0 : float            initial state of charge in [0, 1]
        T0 : float              initial cell temperature [K]
        dt : float              output time step [s]
        duration_s : float      total simulation time [s]

        Returns
        -------
        dict of time-series arrays:
            t, soc, voltage, current, power, efficiency, temperature,
            heat_generation, v_rc1, v_rc2
        """
        current_fn = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [float(np.clip(soc0, 0.0, 1.0)), 0.0, 0.0, float(T0)]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt, args=(current_fn,),
        )

        t_out = sol.t
        soc = np.clip(sol.y[0], 0.0, 1.0)
        v_rc1 = sol.y[1]
        v_rc2 = sol.y[2]
        T = sol.y[3]
        N = len(t_out)

        voltage = np.zeros(N)
        current = np.zeros(N)
        power = np.zeros(N)
        efficiency = np.zeros(N)
        q_gen = np.zeros(N)

        for i in range(N):
            I = current_fn(t_out[i])
            current[i] = I
            voltage[i] = self.terminal_voltage(soc[i], I, T[i], v_rc1[i], v_rc2[i])
            power[i] = voltage[i] * I
            ocv_i = self.ocv(soc[i])
            # round-trip-style instantaneous efficiency in (0,1):
            #   discharge -> V_term/OCV < 1 ; charge -> OCV/V_term < 1
            if I > 0:
                efficiency[i] = voltage[i] / ocv_i
            elif I < 0:
                efficiency[i] = ocv_i / voltage[i]
            else:
                efficiency[i] = 1.0
            q_gen[i] = self.heat_generation(soc[i], I, T[i], v_rc1[i], v_rc2[i])

        return {
            "t": t_out,
            "soc": soc,
            "voltage": voltage,
            "current": current,
            "power": power,
            "efficiency": efficiency,
            "temperature": T,
            "heat_generation": q_gen,
            "v_rc1": v_rc1,
            "v_rc2": v_rc2,
        }
