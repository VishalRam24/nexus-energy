"""
EC034 -- Aluminum-Ion Battery -- F2a Thevenin Equivalent-Circuit Model (ECM)

Physics-lumped 0D dynamic model. The cell state is integrated as a coupled
ODE system with scipy.integrate.solve_ivp:

    state y = [ SOC, V_rc1, (V_rc2), T_cell ]

1) Coulomb-counted SOC (state of charge):
       dSOC/dt = -eta_I * I / (C_eff(T) * 3600)
   where I>0 is discharge, I<0 is charge, eta_I is the coulombic efficiency
   applied on charge only (Plett 2015, Battery Management Systems Vol.1).

2) Thevenin RC branches (1-RC default, optional 2-RC):
       dV_rc_k/dt = I / C_k - V_rc_k / (R_k(T) * C_k)
   The terminal voltage is
       V_term = OCV(SOC) - I*R0(T) - sum_k V_rc_k
   (Plett 2015; Hu et al. 2012, J. Power Sources 198, 359-367 -- ECM comparison.)

3) Al-ion OCV(SOC): multi-step graphite / chloroaluminate (AlCl4-) intercalation
   into the graphite cathode with Al anode dissolution. Staging produces
   ~1.8-2.2 V plateaus; modeled by a monotone 5th-order polynomial fit.
   (Lin et al. 2015, Nature 520, 324-328; Pang et al. 2019, Joule 3(1), 136-148.)

4) Arrhenius temperature dependence of all resistances:
       R(T) = R_ref * exp( E_a/R_gas * (1/T - 1/T_ref) )
   (Guo et al. 2020, Energy Storage Mater. 28, 240-248.)

5) Lumped thermal ODE (single-node energy balance):
       m*cp dT/dt = Q_gen - hA*(T - T_amb)
       Q_gen = I^2*R0(T) + sum_k V_rc_k^2 / R_k(T)   (irreversible, Joule)
               + I * T * dOCV/dT                       (reversible entropic)
   (Bernardi et al. 1985, J. Electrochem. Soc. 132(1), 5-12.)

Al-ion device traits reflected:
   - very high rate capability (low R0, short RC time constants ~ tens of s),
   - ultra-long cycle life / high coulombic efficiency (Lin 2015: 7500+ cycles).

References:
    Lin, M.-C. et al. (2015). An ultrafast rechargeable aluminium-ion battery.
        Nature 520, 324-328.
    Pang, Q. et al. (2019). Joule 3(1), 136-148.
    Guo, Y. et al. (2020). Energy Storage Materials 28, 240-248.
    Bernardi, D. et al. (1985). J. Electrochem. Soc. 132(1), 5-12.
    Plett, G. (2015). Battery Management Systems, Vol. 1. Artech House.
    Hu, X. et al. (2012). J. Power Sources 198, 359-367.
"""

import numpy as np
from scipy.integrate import solve_ivp


class AluminumIonECM_F2a:
    """Aluminum-ion battery -- Thevenin ECM (1-RC/2-RC) with thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.capacity_ref = float(u["capacity_ref"]["value"])      # Ah
        self.v_max = float(u["voltage_max"]["value"])              # V
        self.v_min = float(u["voltage_min"]["value"])              # V

        self.R0_ref = float(u["R0_ref"]["value"])                  # Ohm
        self.R1_ref = float(u["R1_ref"]["value"])                  # Ohm
        self.C1 = float(u["C1"]["value"])                          # F
        self.R2_ref = float(u["R2_ref"]["value"])                  # Ohm
        self.C2 = float(u["C2"]["value"])                          # F
        self.n_rc = int(u["n_rc"]["value"])                        # 1 or 2

        self.T_ref = float(u["T_ref"]["value"])                    # K
        self.E_a = float(u["E_a"]["value"])                        # J/mol
        self.alpha_c = float(u["alpha_c"]["value"])                # 1/K
        self.dOCV_dT = float(u["dOCV_dT"]["value"])                # V/K
        self.R_gas = float(u["R_gas"]["value"])                    # J/(mol.K)

        self.m_cell = float(u["m_cell"]["value"])                  # kg
        self.cp_cell = float(u["cp_cell"]["value"])                # J/(kg.K)
        self.hA = float(u["hA"]["value"])                          # W/K
        self.T_amb = float(u["T_amb"]["value"])                    # K

        self.coulomb_eff = float(u["coulomb_eff"]["value"])        # -

        oc = params["ocv_coefficients"]
        self.ocv_coeff = np.array([oc[f"a{i}"] for i in range(6)], dtype=float)

    # ------------------------------------------------------------------
    # Al-ion open-circuit voltage OCV(SOC)
    # ------------------------------------------------------------------
    def ocv(self, soc):
        """Open-circuit voltage [V] vs SOC (0-1). Monotone increasing."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc ** i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def docv_dsoc(self, soc):
        """Analytic dOCV/dSOC [V] (>0 for monotone-increasing OCV)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([(i * soc ** (i - 1) if i >= 1 else np.zeros_like(soc))
                           for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    # ------------------------------------------------------------------
    # Arrhenius resistances + temperature-dependent capacity
    # ------------------------------------------------------------------
    def _arrhenius(self, R_ref, T):
        return R_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def R0(self, T):
        return self._arrhenius(self.R0_ref, T)

    def R1(self, T):
        return self._arrhenius(self.R1_ref, T)

    def R2(self, T):
        return self._arrhenius(self.R2_ref, T)

    def effective_capacity(self, T):
        """Temperature-corrected usable capacity [Ah] (clamped > 0)."""
        C = self.capacity_ref * (1.0 + self.alpha_c * (T - self.T_ref))
        return max(float(C), 1e-3)

    # ------------------------------------------------------------------
    # Terminal voltage given full state
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, I, T, v_rc1=0.0, v_rc2=0.0):
        """Terminal voltage [V]. I>0 discharge, I<0 charge."""
        v = self.ocv(soc) - I * self.R0(T) - v_rc1
        if self.n_rc >= 2:
            v = v - v_rc2
        return float(np.clip(v, 0.0, self.v_max + 0.5))

    def coulombic_current(self, I):
        """Effective charge-transfer current for SOC integration.

        Discharge (I>0) removes charge at face value; charge (I<0) stores
        only eta_I of it (Coulombic loss). 0 < eta_I < 1 enforced upstream.
        """
        return I if I >= 0 else I * self.coulomb_eff

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, I_func):
        soc = y[0]
        T = y[-1]
        I = float(I_func(t))

        # Coulomb counting (state of charge)
        I_eff = self.coulombic_current(I)
        C_eff = self.effective_capacity(T)
        dsoc = -I_eff / (C_eff * 3600.0)

        derivs = [dsoc]

        # RC branch dynamics: dV/dt = I/C - V/(R*C)
        v_rc1 = y[1]
        dvrc1 = I / self.C1 - v_rc1 / (self.R1(T) * self.C1)
        derivs.append(dvrc1)

        q_rc = v_rc1 ** 2 / self.R1(T)
        if self.n_rc >= 2:
            v_rc2 = y[2]
            dvrc2 = I / self.C2 - v_rc2 / (self.R2(T) * self.C2)
            derivs.append(dvrc2)
            q_rc += v_rc2 ** 2 / self.R2(T)

        # Lumped thermal ODE (Bernardi heat generation)
        q_ohmic = I ** 2 * self.R0(T) + q_rc       # irreversible (>= 0)
        q_rev = I * T * self.dOCV_dT               # reversible (entropic)
        Q_gen = q_ohmic + q_rev
        Q_cool = self.hA * (T - self.T_amb)
        dT = (Q_gen - Q_cool) / (self.m_cell * self.cp_cell)
        derivs.append(dT)

        return derivs

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.9, T0=None, dt=1.0, duration_s=600.0):
        """
        Integrate the coupled ECM + thermal ODE.

        Parameters
        ----------
        current_A : float or callable(t)
            Load current [A]; >0 discharge, <0 charge.
        soc0 : float
            Initial state of charge (0-1).
        T0 : float or None
            Initial cell temperature [K] (default = ambient).
        dt : float
            Output sampling step [s].
        duration_s : float
            Total simulated time [s].

        Returns dict of time-series arrays:
            t, soc, voltage, ocv, current, power, temperature, efficiency,
            v_rc (dict), heat_gen
        """
        if T0 is None:
            T0 = self.T_amb
        I_func = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        n_state = 2 + (1 if self.n_rc >= 2 else 0)  # soc, v_rc1, [v_rc2], T
        y0 = [float(np.clip(soc0, 0.0, 1.0)), 0.0]
        if self.n_rc >= 2:
            y0.append(0.0)
        y0.append(float(T0))

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0, args=(I_func,),
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=max(dt, 1.0),
        )

        t_out = sol.t
        soc_out = np.clip(sol.y[0], 0.0, 1.0)
        v_rc1_out = sol.y[1]
        if self.n_rc >= 2:
            v_rc2_out = sol.y[2]
        else:
            v_rc2_out = np.zeros_like(t_out)
        T_out = sol.y[-1]

        N = len(t_out)
        voltage = np.zeros(N)
        ocv_arr = np.zeros(N)
        current = np.zeros(N)
        power = np.zeros(N)
        efficiency = np.zeros(N)
        heat_gen = np.zeros(N)

        for i in range(N):
            I = float(I_func(t_out[i]))
            T = T_out[i]
            ocv_arr[i] = self.ocv(soc_out[i])
            voltage[i] = self.terminal_voltage(
                soc_out[i], I, T, v_rc1_out[i], v_rc2_out[i])
            current[i] = I
            power[i] = voltage[i] * I
            # Round-trip-style efficiency relative to OCV (0<eff<1):
            # discharge: V/OCV (<1, ohmic loss); charge: OCV/V (<1).
            if abs(I) < 1e-9 or ocv_arr[i] <= 0:
                efficiency[i] = 1.0
            elif I > 0:
                efficiency[i] = voltage[i] / ocv_arr[i]
            else:
                efficiency[i] = ocv_arr[i] / voltage[i] if voltage[i] > 0 else 0.0
            q = (I ** 2 * self.R0(T)
                 + v_rc1_out[i] ** 2 / self.R1(T)
                 + (v_rc2_out[i] ** 2 / self.R2(T) if self.n_rc >= 2 else 0.0)
                 + I * T * self.dOCV_dT)
            heat_gen[i] = q

        return {
            "t": t_out,
            "soc": soc_out,
            "voltage": voltage,
            "ocv": ocv_arr,
            "current": current,
            "power": power,
            "temperature": T_out,
            "efficiency": efficiency,
            "heat_gen": heat_gen,
            "v_rc": {"rc1": v_rc1_out, "rc2": v_rc2_out},
        }
