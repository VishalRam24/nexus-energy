"""
EC027 -- Solid-State Lithium Battery -- F2a Thevenin Equivalent-Circuit Model

Physics-lumped 0D dynamic model of a Li-metal / solid-electrolyte / cathode cell.
A Thevenin equivalent circuit (1-RC or 2-RC) is coupled to a Coulomb-counting SOC
state and a lumped thermal ODE, integrated together with scipy.integrate.solve_ivp.

State vector (n_rc = 2):
    y = [SOC, V_rc1, V_rc2, T]

Governing equations
-------------------
1. Coulomb counting (charge conservation):
       dSOC/dt = -I / (3600 * C_cap)          [C_cap in Ah, I in A, load sign]
   This is an exact integral of current -> total Coulombs in/out is conserved.

2. Open-circuit voltage (monotonic in SOC), 5th-order polynomial fit:
       OCV(SOC) = sum_{i=0..5} a_i * SOC^i

3. Series solid-electrolyte ionic resistance (DOMINANT, strong Arrhenius):
       R0(T) = R0_ref * exp( E_a_R0 / R_gas * (1/T - 1/T_ref) )
   The activation energy E_a_R0 (~38 kJ/mol) is much larger than for liquid
   electrolytes (~15-20 kJ/mol), so the cell performs very poorly when cold.

4. Interfacial charge-transfer RC branch(es) (Thevenin dynamics):
       dV_rc_k/dt = -V_rc_k / (R_k(T) C_k) + I / C_k
       R_k(T)     = R_k_ref * exp( E_a_Rk / R_gas * (1/T - 1/T_ref) )

5. Terminal voltage (load sign: I>0 discharge lowers terminal voltage):
       V = OCV(SOC) - I*R0(T) - sum_k V_rc_k

6. Lumped thermal ODE (Bernardi heat-generation balance):
       m cp dT/dt = Q_gen - Q_conv
       Q_gen  = I*(OCV - V) + I*T*dOCV/dT        (irreversible I^2*R-type + reversible entropic)
       Q_conv = hA*(T - T_amb)

References
----------
    Janek & Zeier (2016). A solid future for battery development. Nat. Energy 1, 16141.
    Takada (2013). Progress and prospective of solid-state Li batteries. Acta Mat. 61, 759-770.
    Murugan, Thangadurai & Weppner (2007). Fast Li conduction in garnet-type Li7La3Zr2O12.
        Angew. Chem. Int. Ed. 46, 7778-7781. (LLZO E_a ~ 0.30 eV)
    Kato et al. (2016). High-power all-solid-state batteries using sulfide superionic
        conductors (Li10GeP2S12 / Li9.54Si1.74P1.44S11.7Cl0.3). Nat. Energy 1, 16030.
    Bernardi, Pawlikowski & Newman (1985). A general energy balance for battery systems.
        J. Electrochem. Soc. 132(1), 5-12.
    Plett (2015). Battery Management Systems, Vol. I: Battery Modeling. Artech House.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SolidStateLiECM_F2a:
    """Thevenin ECM for a solid-state Li cell with Arrhenius R(T) and thermal ODE."""

    def __init__(self, params: dict):
        cell = params["cell"]
        ocv = params["ocv_coefficients"]
        ecm = params["ecm"]
        therm = params["thermal"]

        # Cell / OCV
        self.C_cap = cell["capacity_ref"]["value"]          # Ah
        self.v_max = cell["voltage_max"]["value"]           # V
        self.v_min = cell["voltage_min"]["value"]           # V
        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

        # ECM resistances / capacitances
        self.R0_ref = ecm["R0_ref"]["value"]                # Ohm  (series SE ionic)
        self.E_a_R0 = ecm["E_a_R0"]["value"]                # J/mol
        self.n_rc = int(ecm["n_rc"]["value"])
        self.R1_ref = ecm["R1_ref"]["value"]
        self.E_a_R1 = ecm["E_a_R1"]["value"]
        self.C1 = ecm["C1"]["value"]
        self.R2_ref = ecm["R2_ref"]["value"]
        self.E_a_R2 = ecm["E_a_R2"]["value"]
        self.C2 = ecm["C2"]["value"]

        # Thermal
        self.T_ref = therm["T_ref"]["value"]                # K
        self.R_gas = therm["R_gas"]["value"]
        self.dOCV_dT = therm["dOCV_dT"]["value"]            # V/K
        self.m_cell = therm["m_cell"]["value"]              # kg
        self.cp_cell = therm["cp_cell"]["value"]            # J/(kg K)
        self.hA = therm["hA"]["value"]                      # W/K
        self.T_amb_default = therm["T_amb"]["value"]        # K

    # ------------------------------------------------------------------
    # Open-circuit voltage  OCV(SOC)
    # ------------------------------------------------------------------
    def ocv(self, soc):
        """Open-circuit voltage [V] as a function of SOC (clipped to [0,1])."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc ** i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def docv_dsoc(self, soc):
        """dOCV/dSOC [V] (analytic derivative of the polynomial)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        d = np.zeros_like(soc, dtype=float)
        for i in range(1, 6):
            d = d + i * self.ocv_coeff[i] * soc ** (i - 1)
        return d

    # ------------------------------------------------------------------
    # Arrhenius resistances  R(T)
    # ------------------------------------------------------------------
    def _arrhenius(self, R_ref, E_a, T):
        T = np.asarray(T, dtype=float)
        return R_ref * np.exp(E_a / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def R0(self, T):
        """Series solid-electrolyte ionic resistance [Ohm] -- dominant, strong T-dependence."""
        return self._arrhenius(self.R0_ref, self.E_a_R0, T)

    def R1(self, T):
        """First interfacial RC resistance [Ohm]."""
        return self._arrhenius(self.R1_ref, self.E_a_R1, T)

    def R2(self, T):
        """Second interfacial RC resistance [Ohm]."""
        return self._arrhenius(self.R2_ref, self.E_a_R2, T)

    # ------------------------------------------------------------------
    # Terminal voltage
    # ------------------------------------------------------------------
    def terminal_voltage(self, soc, current, T, v_rc):
        """
        Terminal voltage [V].  Load sign: current>0 = discharge (lowers V).
        v_rc : sequence of RC-branch voltages [V].
        """
        v = self.ocv(soc) - current * self.R0(T)
        for vk in v_rc:
            v = v - vk
        return float(np.clip(v, self.v_min, self.v_max))

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, current_fn, T_amb):
        soc = y[0]
        T = y[-1]
        I = current_fn(t)

        # 1) Coulomb counting
        dsoc = -I / (3600.0 * self.C_cap)

        # 2) RC branch dynamics
        derivs = [dsoc]
        v_rc = []
        if self.n_rc >= 1:
            v1 = y[1]
            v_rc.append(v1)
            tau1 = max(self.R1(T) * self.C1, 1e-9)
            dv1 = -v1 / tau1 + I / self.C1
            derivs.append(dv1)
        if self.n_rc >= 2:
            v2 = y[2]
            v_rc.append(v2)
            tau2 = max(self.R2(T) * self.C2, 1e-9)
            dv2 = -v2 / tau2 + I / self.C2
            derivs.append(dv2)

        # 3) Thermal ODE (Bernardi energy balance)
        V = self.terminal_voltage(soc, I, T, v_rc)
        ocv = float(self.ocv(soc))
        Q_irr = I * (ocv - V)                       # always >= 0 (dissipative)
        Q_rev = I * T * self.dOCV_dT                # reversible entropic heat
        Q_gen = Q_irr + Q_rev
        Q_conv = self.hA * (T - T_amb)
        dT = (Q_gen - Q_conv) / (self.m_cell * self.cp_cell)
        derivs.append(dT)

        return derivs

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0=0.8, T0=298.15, T_amb=None,
                 dt=1.0, duration_s=600.0):
        """
        Simulate the coupled ECM + thermal dynamics with scipy.solve_ivp.

        Parameters
        ----------
        current_A : float or callable(t)
            Load current [A]; >0 discharge, <0 charge.
        soc0 : float       initial state of charge (0-1)
        T0 : float         initial cell temperature [K]
        T_amb : float      ambient temperature [K] (default from params)
        dt : float         output time step [s]
        duration_s : float total duration [s]

        Returns
        -------
        dict with arrays: t, soc, voltage, current, power, temperature,
             R0, ocv, v_rc (list of arrays), and scalar 'coulombic_efficiency'.
        """
        if T_amb is None:
            T_amb = self.T_amb_default
        cur_fn = current_A if callable(current_A) else (lambda t: float(current_A))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [float(soc0)]
        if self.n_rc >= 1:
            y0.append(0.0)
        if self.n_rc >= 2:
            y0.append(0.0)
        y0.append(float(T0))

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(cur_fn, T_amb),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t_out = sol.t
        N = len(t_out)
        soc_out = np.clip(sol.y[0], 0.0, 1.0)
        T_out = sol.y[-1]

        v_rc_arrays = []
        idx = 1
        if self.n_rc >= 1:
            v_rc_arrays.append(sol.y[idx]); idx += 1
        if self.n_rc >= 2:
            v_rc_arrays.append(sol.y[idx]); idx += 1

        voltage = np.zeros(N)
        current = np.zeros(N)
        power = np.zeros(N)
        R0_arr = np.zeros(N)
        ocv_arr = np.zeros(N)
        for i in range(N):
            I = cur_fn(t_out[i])
            v_rc_i = [arr[i] for arr in v_rc_arrays]
            voltage[i] = self.terminal_voltage(soc_out[i], I, T_out[i], v_rc_i)
            current[i] = I
            power[i] = voltage[i] * I
            R0_arr[i] = float(self.R0(T_out[i]))
            ocv_arr[i] = float(self.ocv(soc_out[i]))

        # Coulombic efficiency proxy: useful energy out / OCV energy throughput.
        # 0 < eff < 1 because of the dissipative R-drops.
        dt_arr = np.diff(t_out, prepend=t_out[0])
        e_terminal = np.sum(np.abs(voltage * current) * dt_arr)
        e_ocv = np.sum(np.abs(ocv_arr * current) * dt_arr)
        coul_eff = e_terminal / e_ocv if e_ocv > 1e-12 else 1.0

        return {
            "t": t_out,
            "soc": soc_out,
            "voltage": voltage,
            "current": current,
            "power": power,
            "temperature": T_out,
            "R0": R0_arr,
            "ocv": ocv_arr,
            "v_rc": v_rc_arrays,
            "coulombic_efficiency": float(coul_eff),
            "success": bool(sol.success),
        }
