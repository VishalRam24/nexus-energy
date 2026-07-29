"""
EC035 -- Sodium-Sulfur (NaS) Battery -- F2a Thevenin 1-RC ECM (physics-lumped)

High-temperature chemistry: a NaS cell uses MOLTEN sodium (anode) and molten
sulfur/sodium-polysulfide (cathode) separated by a beta-alumina SOLID
electrolyte. It only works while held at ~300-350 degC, where Na and S are
liquid and the beta-alumina is ionically conductive. Outside this window the
electrodes solidify and the cell is non-functional -- so a thermal-management
HEATER is part of the device, fighting heat loss and assisted by I^2*R
self-heating during operation.

State vector y = [SOC, V_rc, T]:

  1) Coulomb-counted SOC (charge conservation):
        dSOC/dt = -I / (Q(T) * 3600)            [I>0 discharge, I<0 charge]
        Q(T)    = Q_ref * (1 + alpha_c*(T - T_op_ref))

  2) Thevenin 1-RC overpotential branch (Hu et al. 2012; Plett 2015):
        dV_rc/dt = -V_rc / (R1(T) C1) + I / C1
        V_term   = OCV(SOC) + dOCV/dT*(T - T_op_ref) - I*R0(T) - V_rc
     (sign convention: I>0 discharge lowers terminal voltage below OCV)

  3) Beta-alumina Arrhenius series + polarisation resistance (strong T-dep):
        R0(T) = R0_ref * exp( Ea/Rg * (1/T - 1/T_op_ref) )
        R1(T) = R1_ref * exp( Ea/Rg * (1/T - 1/T_op_ref) )
     Ionic conductivity sigma ~ exp(-Ea/RgT), so resistance ~ exp(+Ea/RgT):
     resistance falls as temperature rises within the window.

  4) Lumped thermal ODE (energy balance, must hold T in band):
        m*cp dT/dt = Q_ohmic + Q_rev + Q_heater - Q_loss
        Q_ohmic  = I^2*R0(T) + V_rc^2 / R1(T)          (Joule, irreversible >= 0)
        Q_rev    = -I * T * dOCV/dT                     (entropic, signed)
        Q_heater = P_htr * clamp((T_set - T)/band, 0, 1) (proportional heater)
        Q_loss   = hA_loss * (T - T_ambient)            (through insulation)

Integrated with scipy.integrate.solve_ivp (LSODA, stiff-capable because the
RC and thermal time constants differ by orders of magnitude).

References:
    Sudworth, J. L. & Tilley, A. R. (1985). The Sodium Sulfur Battery.
        Chapman & Hall. (OCV, entropy, beta-alumina conduction.)
    Wen, Z. et al. (2008). Mater. Sci. Eng. B 154-155, 73-78. (Cell resistance.)
    Hueso, K. B. et al. (2013). Energy Environ. Sci. 6, 734-749. (NaS review.)
    Hu, X., Li, S. & Peng, H. (2012). J. Power Sources 198, 359-367. (ECM/RC.)
    Plett, G. L. (2015). Battery Management Systems, Vol. I. Artech House. (ECM.)
"""

import numpy as np
from scipy.integrate import solve_ivp


class NaSBatteryF2a:
    """NaS high-temperature battery -- Thevenin 1-RC ECM with thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        c = params["constants"]
        ocv = params["ocv_coefficients"]

        # Electrical
        self.Q_ref = u["capacity_ref"]["value"]      # Ah
        self.v_max = u["voltage_max"]["value"]        # V
        self.v_min = u["voltage_min"]["value"]        # V
        self.R0_ref = u["R0_ref"]["value"]            # Ohm
        self.R1_ref = u["R1_ref"]["value"]            # Ohm
        self.C1 = u["C1"]["value"]                    # F

        # Thermal / Arrhenius
        self.E_a = u["E_a_R"]["value"]                # J/mol
        self.alpha_c = u["alpha_c"]["value"]          # 1/K
        self.dOCV_dT = u["dOCV_dT"]["value"]          # V/K
        self.T_op_ref = u["T_op_ref"]["value"]        # K
        self.T_op_min = u["T_op_min"]["value"]        # K
        self.T_op_max = u["T_op_max"]["value"]        # K
        self.m_cell = u["m_cell"]["value"]            # kg
        self.cp_cell = u["cp_cell"]["value"]          # J/(kg.K)
        self.hA_loss = u["hA_loss"]["value"]          # W/K
        self.T_ambient = u["T_ambient"]["value"]      # K
        self.P_htr_max = u["heater_power_max"]["value"]   # W
        self.T_set = u["T_heater_setpoint"]["value"]  # K
        self.heater_band = u["heater_band"]["value"]  # K

        self.R_gas = c["R_gas"]["value"]              # J/(mol.K)
        self.ocv_coeff = np.array([ocv[f"a{i}"] for i in range(6)])

    # ------------------------------------------------------------------
    # Static relations
    # ------------------------------------------------------------------
    def is_functional(self, T):
        """True while T within the 300-350 degC molten/conductive window."""
        T = np.asarray(T, dtype=float)
        return (T >= self.T_op_min) & (T <= self.T_op_max)

    def ocv(self, soc):
        """Open-circuit voltage [V] vs SOC -- NaS two-plateau curve (~1.78-2.08 V)."""
        soc = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
        powers = np.stack([soc ** i for i in range(6)], axis=-1)
        return np.dot(powers, self.ocv_coeff)

    def _arrhenius(self, R_ref, T):
        """beta-alumina Arrhenius scaling: R(T) = R_ref*exp(Ea/Rg*(1/T - 1/T_ref))."""
        T = np.asarray(T, dtype=float)
        return R_ref * np.exp(self.E_a / self.R_gas * (1.0 / T - 1.0 / self.T_op_ref))

    def R0(self, T):
        """Series ohmic resistance [Ohm] (beta-alumina + leads), Arrhenius in T."""
        return self._arrhenius(self.R0_ref, T)

    def R1(self, T):
        """Polarisation (RC-branch) resistance [Ohm], Arrhenius in T."""
        return self._arrhenius(self.R1_ref, T)

    def capacity(self, T):
        """Temperature-corrected capacity [Ah] within window."""
        T = np.asarray(T, dtype=float)
        return self.Q_ref * (1.0 + self.alpha_c * (T - self.T_op_ref))

    def heater_power(self, T):
        """
        Proportional thermostat heater [W]: full power well below setpoint,
        ~0 above it. Implemented as a SMOOTH (C-infinity) saturating logistic
        rather than a hard clip -- the smoothness removes the derivative kink
        at the setpoint that would otherwise stall the stiff ODE solver, while
        reproducing the same proportional-band behaviour.
        """
        T = np.asarray(T, dtype=float)
        # logistic centred half a band below setpoint, width ~ heater_band
        z = (self.T_set - self.heater_band - T) / (0.5 * self.heater_band)
        frac = 1.0 / (1.0 + np.exp(-z))
        return self.P_htr_max * frac

    def terminal_voltage(self, soc, current, T, V_rc=0.0):
        """
        Terminal voltage [V]. I>0 discharge => V below OCV.
        Returns 0 V when the cell is outside the operating window.
        """
        soc = np.asarray(soc, dtype=float)
        current = np.asarray(current, dtype=float)
        T = np.asarray(T, dtype=float)
        V_rc = np.asarray(V_rc, dtype=float)
        ocv = self.ocv(soc) + self.dOCV_dT * (T - self.T_op_ref)
        v = ocv - current * self.R0(T) - V_rc
        v = np.clip(v, self.v_min, self.v_max)
        return np.where(self.is_functional(T), v, 0.0)

    def heat_generation(self, current, T, V_rc=0.0):
        """
        Net internal heat source [W] = Joule(R0)+Joule(R1) + reversible entropic.
        Joule terms are non-negative; entropic term is signed.
        """
        current = np.asarray(current, dtype=float)
        T = np.asarray(T, dtype=float)
        V_rc = np.asarray(V_rc, dtype=float)
        q_ohmic = current ** 2 * self.R0(T) + V_rc ** 2 / self.R1(T)
        q_rev = -current * T * self.dOCV_dT
        return q_ohmic + q_rev

    # ------------------------------------------------------------------
    # ODE right-hand side: y = [SOC, V_rc, T]
    # ------------------------------------------------------------------
    def _window_factor(self, T):
        """
        Smooth reactivity gate w(T) in [0,1]: ~1 inside the 300-350 C molten
        window, smoothly -> 0 outside it. Implemented as a product of two
        logistics so the ODE right-hand side stays C-infinity continuous
        (a hard on/off flag injects a derivative discontinuity that stalls
        the stiff solver at the window edges). Physically: outside the window
        Na/S freeze and beta-alumina stops conducting, so the cell stops
        reacting -- no coulomb throughput, no reaction heat.
        """
        edge = 2.0  # K, smoothing half-width at each window edge
        lo = 1.0 / (1.0 + np.exp(-(T - self.T_op_min) / edge))
        hi = 1.0 / (1.0 + np.exp((T - self.T_op_max) / edge))
        return lo * hi

    def _soc_gate(self, soc, I):
        """
        Smooth end-stop: when discharging (I>0) at empty SOC, or charging
        (I<0) at full SOC, the coulomb throughput tapers to zero so SOC does
        not run past [0,1]. C-infinity so the solver stays happy.
        """
        edge = 0.01
        if I > 0.0:   # discharge -> blocked near soc=0
            return 1.0 / (1.0 + np.exp(-(soc - 0.0) / edge))
        elif I < 0.0:  # charge -> blocked near soc=1
            return 1.0 / (1.0 + np.exp((soc - 1.0) / edge))
        return 1.0

    def _rhs(self, t, y, I_func):
        soc, V_rc, T = y
        I = float(I_func(t))

        # smooth reactivity gate (molten window) and SOC end-stop
        w = self._window_factor(T)
        g = self._soc_gate(soc, I)
        I_eff = I * w * g

        Q_eff = max(self.capacity(T), 1e-6)
        dsoc = -I_eff / (Q_eff * 3600.0)

        tau1 = max(self.R1(T) * self.C1, 1e-6)
        dV_rc = -V_rc / tau1 + I_eff / self.C1

        # reaction heat scaled by the same window factor (no reaction => no
        # reaction heat); the RC-dissipation term uses the actual V_rc state.
        q_int = w * self.heat_generation(I_eff, T, V_rc)

        q_heater = self.heater_power(T)
        q_loss = self.hA_loss * (T - self.T_ambient)
        dT = (q_int + q_heater - q_loss) / (self.m_cell * self.cp_cell)
        return [dsoc, dV_rc, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, current_A, soc0, T0_K, dt, duration_s, V_rc0=0.0):
        """
        Integrate the coupled SOC / RC / thermal ODEs.

        Parameters
        ----------
        current_A : float or callable(t)   [A] (>0 discharge, <0 charge)
        soc0      : float                   initial SOC (0-1)
        T0_K      : float                   initial cell temperature [K]
        dt        : float                   output time step [s]
        duration_s: float                   total duration [s]
        V_rc0     : float                   initial RC overpotential [V]

        Returns
        -------
        dict of time-series arrays: t, soc, voltage, current, power,
            temperature, v_rc, R0, R1, heat_gen, heater_power, functional.
        """
        I_func = current_A if callable(current_A) else (lambda t: current_A)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            lambda t, y: self._rhs(t, y, I_func),
            (0.0, duration_s), [float(soc0), float(V_rc0), float(T0_K)],
            t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9,
            max_step=max(dt, 1.0),
        )

        t = sol.t
        soc = np.clip(sol.y[0], 0.0, 1.0)
        V_rc = sol.y[1]
        T = sol.y[2]
        N = len(t)

        I = np.array([float(I_func(ti)) for ti in t])
        voltage = np.array([self.terminal_voltage(soc[i], I[i], T[i], V_rc[i]) for i in range(N)])
        power = voltage * I
        R0 = self.R0(T)
        R1 = self.R1(T)
        heat = np.array([self.heat_generation(I[i], T[i], V_rc[i]) for i in range(N)])
        q_htr = self.heater_power(T)
        functional = self.is_functional(T)

        return {
            "t": t,
            "soc": soc,
            "voltage": voltage,
            "current": I,
            "power": power,
            "temperature": T,
            "v_rc": V_rc,
            "R0": R0,
            "R1": R1,
            "heat_gen": heat,
            "heater_power": q_htr,
            "functional": functional,
        }
