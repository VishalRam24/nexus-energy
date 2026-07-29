"""
EC086 -- Electric Boiler / Resistance Heater -- F2a Dynamic Thermal Mass

Physics-lumped (0D) model of an electric resistance / electrode boiler.
A single lumped-capacitance energy balance ODE tracks the bulk water +
vessel temperature; the resistance element delivers near-100% of its
electrical input as heat, the jacket loses heat to ambient, and a draw of
cold feed water removes enthalpy.

Energy balance (lumped capacitance, Incropera & DeWitt 2007, Ch. 5):

    C_th * dT/dt = Q_elec(T,t) - Q_loss(T) - Q_load(T,t)

where
    C_th    = m_water*cp_water + m_vessel*cp_vessel      [J/K]
    Q_elec  = u(t) * eta_elec * P_rated                  electrical heat in [W]
    Q_loss  = UA_loss * (T - T_ambient)                  standby jacket loss [W]
    Q_load  = mdot * cp_water * (T - T_inlet)            load enthalpy out [W]
    u(t)    in [0,1] is the firing fraction set by the controller.

Control:
    - 'onoff'      : thermostat with hysteresis deadband (bang-bang).
                     u = 1 below (T_set - db/2), u = 0 above (T_set + db/2).
    - 'modulating' : proportional control clamped to [0,1] that holds T near
                     setpoint (fast-response electric element).

Resistance heaters have no combustion, no flue and no fuel, so the
electrical-to-thermal efficiency is essentially unity (~0.99 accounting for
small cabinet/control losses); they respond near-instantly to the control
signal (no thermal lag in the element itself), the dynamics being dominated
by the water/vessel thermal mass.

Integrated with scipy.integrate.solve_ivp (stiff-safe 'RK45'/'LSODA').

References:
    ASHRAE Handbook -- HVAC Systems & Equipment (2020), Ch. 32 'Boilers'.
    Incropera & DeWitt (2007), Fundamentals of Heat and Mass Transfer,
        6th ed., Wiley, Ch. 5 (Lumped Capacitance Method).
    Wagner & Pruss (2002), J. Phys. Chem. Ref. Data 31(2) 387-535
        (IAPWS-95) -- water cp = 4185 J/(kg.K) near 60 C.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ElectricBoilerF2a:
    """Electric resistance boiler -- lumped thermal-mass dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_kw"]["value"] * 1000.0      # W
        self.eta_elec = u["eta_elec"]["value"]                # -
        self.V_water = u["V_water_L"]["value"] / 1000.0       # m3
        self.m_vessel = u["m_vessel_kg"]["value"]             # kg
        self.cp_vessel = u["cp_vessel"]["value"]              # J/(kg.K)
        self.rho_water = u["rho_water"]["value"]              # kg/m3
        self.cp_water = u["cp_water"]["value"]                # J/(kg.K)
        self.UA_loss = u["UA_loss"]["value"]                  # W/K
        self.T_ambient = u["T_ambient_K"]["value"]            # K
        self.T_set = u["T_set_K"]["value"]                    # K
        self.T_db = u["T_deadband_K"]["value"]                # K
        self.T_inlet = u["T_inlet_K"]["value"]                # K

        if not (0.0 < self.eta_elec <= 1.0):
            raise ValueError(f"eta_elec must be in (0,1], got {self.eta_elec}")
        if self.P_rated <= 0:
            raise ValueError(f"P_rated must be > 0, got {self.P_rated}")

        self.m_water = self.rho_water * self.V_water          # kg
        # Lumped thermal capacitance [J/K]
        self.C_th = self.m_water * self.cp_water + self.m_vessel * self.cp_vessel

    # ------------------------------------------------------------------
    # Controller firing fraction u(t) in [0, 1]
    # ------------------------------------------------------------------
    def firing_fraction(self, T, state, T_set, control):
        """
        Return (u, new_state) for the heating element.

        state : bool, latched on/off memory for hysteresis (on=True).
        """
        if control == "modulating":
            # Proportional band of 2*db around setpoint, clamped [0,1].
            band = max(self.T_db, 1e-6)
            u = (T_set - T) / band + 0.0
            return float(np.clip(u, 0.0, 1.0)), state
        # default: on/off thermostat with hysteresis
        lo = T_set - 0.5 * self.T_db
        hi = T_set + 0.5 * self.T_db
        if T <= lo:
            state = True
        elif T >= hi:
            state = False
        return (1.0 if state else 0.0), state

    # ------------------------------------------------------------------
    # Instantaneous power terms [W]
    # ------------------------------------------------------------------
    def q_elec(self, u):
        """Thermal power delivered by the resistance element [W]."""
        return u * self.eta_elec * self.P_rated

    def q_loss(self, T):
        """Standby jacket loss to ambient [W]."""
        return self.UA_loss * (T - self.T_ambient)

    def q_load(self, T, mdot):
        """Enthalpy removed by drawn load water [W] (only when T>T_inlet)."""
        return mdot * self.cp_water * (T - self.T_inlet)

    # ------------------------------------------------------------------
    # Dynamic simulation
    # ------------------------------------------------------------------
    def simulate(self, T_init, mdot, dt=1.0, duration_s=3600.0,
                 T_set=None, control="onoff", P_input=None):
        """
        Integrate the lumped energy balance.

        Parameters
        ----------
        T_init : float            initial water temperature [K]
        mdot   : float or callable  load mass flow [kg/s]; callable(t)->kg/s
        dt     : float            output sampling step [s]
        duration_s : float        total simulated time [s]
        T_set  : float or None    thermostat setpoint [K] (default param)
        control: str              'onoff' or 'modulating'
        P_input: float or None    override electrical input [W] (uncontrolled);
                                  if given, u=1 forced and element runs at
                                  P_input (used for energy-balance checks).
        """
        if T_set is None:
            T_set = self.T_set
        mdot_fn = mdot if callable(mdot) else (lambda t: mdot)

        # For smooth (non-hysteretic) control the RHS is continuous, so we
        # integrate with scipy.solve_ivp and cross-check the energy balance.
        if control == "modulating" or P_input is not None:
            return self._simulate_ivp(T_init, mdot_fn, dt, duration_s,
                                      T_set, control, P_input)

        t_arr = np.arange(0.0, duration_s + 0.5 * dt, dt)
        n = len(t_arr)

        T_arr = np.empty(n)
        u_arr = np.empty(n)
        md_arr = np.empty(n)
        # Energy accumulators integrated step-by-step so the balance is
        # EXACT (each step uses the same constant power held over [t,t+dt]
        # that drives the analytic temperature update -- no post-hoc
        # reconstruction mismatch).
        E_elec = E_in_th = E_loss = E_load = 0.0

        T = float(T_init)
        state = T_init < T_set         # latched on/off memory

        for i in range(n):
            t = t_arr[i]
            md = mdot_fn(t)
            # Controller decision at the start of the step, held constant.
            if P_input is not None:
                u = 1.0
                q_elec_th = self.eta_elec * P_input
                P_elec = P_input
            else:
                u, state = self.firing_fraction(T, state, T_set, control)
                q_elec_th = self.q_elec(u)
                P_elec = u * self.P_rated

            T_arr[i] = T
            u_arr[i] = u
            md_arr[i] = md

            if i == n - 1:
                break

            h = t_arr[i + 1] - t_arr[i]
            # Linear ODE over the step with constant coefficients:
            #   C dT/dt = q_elec_th - UA(T-Ta) - md*cp*(T-Tin)
            #           = S - k*T,   k = UA + md*cp
            k = self.UA_loss + md * self.cp_water
            S = q_elec_th + self.UA_loss * self.T_ambient \
                + md * self.cp_water * self.T_inlet
            tau = self.C_th / k                       # k > 0 always
            T_eq = S / k
            T_next = T_eq + (T - T_eq) * np.exp(-h / tau)

            # Exact step-averaged energies [J] (integrals of the analytic T(t))
            # mean over the step of (T - T_eq) is tau/h * (T - T_next)
            mean_dev = (T - T_next) * tau / h
            T_mean = T_eq + mean_dev
            E_elec += P_elec * h
            E_in_th += q_elec_th * h
            E_loss += self.UA_loss * (T_mean - self.T_ambient) * h
            E_load += md * self.cp_water * (T_mean - self.T_inlet) * h
            T = T_next

        # Instantaneous signals at the sample points.
        if P_input is not None:
            P_elec_arr = np.full(n, float(P_input))
        else:
            P_elec_arr = u_arr * self.P_rated
        q_elec_arr = self.eta_elec * P_elec_arr
        q_loss_arr = self.q_loss(T_arr)
        q_load_arr = md_arr * self.cp_water * (T_arr - self.T_inlet)

        # Efficiency = useful thermal delivered to load / electrical input
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(P_elec_arr > 1e-9,
                           np.clip(q_load_arr / P_elec_arr, 0.0, 1.0),
                           0.0)

        E_stored = self.C_th * (T_arr[-1] - T_arr[0])

        return {
            "t": t_arr,
            "temperature": T_arr,
            "firing_fraction": u_arr,
            "P_elec_W": P_elec_arr,
            "Q_elec_W": q_elec_arr,
            "Q_loss_W": q_loss_arr,
            "Q_load_W": q_load_arr,
            "mdot_kg_s": md_arr,
            "efficiency": eff,
            "energy": {
                "E_elec_J": float(E_elec),
                "E_thermal_in_J": float(E_in_th),
                "E_loss_J": float(E_loss),
                "E_load_J": float(E_load),
                "E_stored_J": float(E_stored),
                # residual should be ~0 by conservation
                "E_residual_J": float(E_in_th - E_loss - E_load - E_stored),
            },
        }

    # ------------------------------------------------------------------
    def _simulate_ivp(self, T_init, mdot_fn, dt, duration_s,
                      T_set, control, P_input):
        """
        Continuous-control integration via scipy.integrate.solve_ivp.
        Augmented state y = [T, E_elec, E_th_in, E_loss, E_load] so the
        energy integrals are produced by the same integrator that advances
        T -> energy conservation is exact to solver tolerance.
        """
        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)

        def control_power(T):
            if P_input is not None:
                return 1.0, P_input
            u, _ = self.firing_fraction(T, True, T_set, control)
            return u, u * self.P_rated

        def rhs(t, y):
            T = y[0]
            md = mdot_fn(t)
            u, P_elec = control_power(T)
            q_in = self.eta_elec * P_elec
            q_loss = self.q_loss(T)
            q_load = self.q_load(T, md)
            dT = (q_in - q_loss - q_load) / self.C_th
            return [dT, P_elec, q_in, q_loss, q_load]

        sol = solve_ivp(rhs, (0.0, duration_s),
                        [T_init, 0.0, 0.0, 0.0, 0.0],
                        t_eval=t_eval, method="LSODA", max_step=dt,
                        rtol=1e-8, atol=1e-8)

        t_arr = sol.t
        T_arr = sol.y[0]
        E_elec, E_in_th, E_loss, E_load = (sol.y[1][-1], sol.y[2][-1],
                                           sol.y[3][-1], sol.y[4][-1])
        n = len(t_arr)
        u_arr = np.empty(n)
        P_elec_arr = np.empty(n)
        for i, T in enumerate(T_arr):
            u_arr[i], P_elec_arr[i] = control_power(T)
        md_arr = np.array([mdot_fn(t) for t in t_arr])
        q_elec_arr = self.eta_elec * P_elec_arr
        q_loss_arr = self.q_loss(T_arr)
        q_load_arr = md_arr * self.cp_water * (T_arr - self.T_inlet)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(P_elec_arr > 1e-9,
                           np.clip(q_load_arr / P_elec_arr, 0.0, 1.0), 0.0)
        E_stored = self.C_th * (T_arr[-1] - T_arr[0])
        return {
            "t": t_arr, "temperature": T_arr, "firing_fraction": u_arr,
            "P_elec_W": P_elec_arr, "Q_elec_W": q_elec_arr,
            "Q_loss_W": q_loss_arr, "Q_load_W": q_load_arr,
            "mdot_kg_s": md_arr, "efficiency": eff,
            "energy": {
                "E_elec_J": float(E_elec),
                "E_thermal_in_J": float(E_in_th),
                "E_loss_J": float(E_loss),
                "E_load_J": float(E_load),
                "E_stored_J": float(E_stored),
                "E_residual_J": float(E_in_th - E_loss - E_load - E_stored),
            },
        }

    # ------------------------------------------------------------------
    def steady_temperature(self, mdot):
        """
        Analytic steady-state water temperature with element full on
        (u=1), used as a sanity bound.  Setting dT/dt=0:
            eta*P = UA(T-Ta) + mdot*cp*(T-Tin)
        """
        a = self.UA_loss + mdot * self.cp_water
        b = self.eta_elec * self.P_rated + self.UA_loss * self.T_ambient \
            + mdot * self.cp_water * self.T_inlet
        return b / a
