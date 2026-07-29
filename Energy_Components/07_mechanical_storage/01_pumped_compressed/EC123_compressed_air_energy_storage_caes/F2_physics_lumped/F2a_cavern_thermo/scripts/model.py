"""
EC123 — Compressed Air Energy Storage (CAES), Diabatic — F2a Cavern Thermodynamics

Physics-lumped (0D) first-principles model of a diabatic CAES plant of the
Huntorf / McIntosh class:

  CHARGE   : ambient air drawn through a multistage intercooled compressor train
             (per-stage polytropic compression, intercooling back to ~T_intercool)
             and injected into a fixed-volume (constant-V) underground salt cavern.

  STORE    : the cavern is treated as an open thermodynamic control volume of
             fixed volume V.  Air mass m(t) and temperature T(t) evolve by a
             coupled mass + energy balance ODE; pressure follows from the ideal-gas
             law P = m R T / V.  Heat leaks to the surrounding rock wall.

  DISCHARGE: cavern air is throttled to the turbine, heated by combustion of
             natural gas to T_turb_in, then expanded through a turbine to ambient.
             The fuel energy is booked so a round-trip efficiency that *includes
             fuel input* can be reported (diabatic CAES electric RTE alone exceeds
             one because the turbine work is partly fuel-derived).

Governing equations
-------------------
Cavern control volume (open system, fixed V), ideal gas, lumped (Raju & Khaitan 2012):

    dm/dt = m_dot_in - m_dot_out                                    (mass balance)

    d(m u)/dt = m_dot_in h_in - m_dot_out h_out - Q_loss            (energy balance, 1st law)

with u = cv*T,  h = cp*T,  expanding and using cv,cp constant:

    m cv dT/dt = m_dot_in cp T_in - m_dot_out cp T - cv T (m_dot_in - m_dot_out) - Q_loss

    Q_loss = UA (T - T_rock)                                        (Newton wall loss)

    P = m R T / V                                                   (ideal-gas EOS)

Multistage intercooled compressor (per stage polytropic), Succar & Williams (2008):

    pressure ratio per stage  r = (P_cav / P_intake)^(1/n_stages)
    w_stage = (gamma/(gamma-1)) * R * T_in * (r^((gamma-1)/(gamma*eta_poly)) - 1)
    w_comp  = n_stages * w_stage      (intercooled back to T_in before each stage)
    P_elec_in = m_dot * w_comp / eta_motor

Fuel-fired turbine (heat air to T_turb_in, expand to ambient), Crotogino (2001):

    q_fuel = cp * (T_turb_in - T_cav)                     (combustor specific heat add)
    T_turb_out_s = T_turb_in * (P_out/P_cav)^((gamma-1)/gamma)   (isentropic)
    w_turb = eta_turb * cp * (T_turb_in - T_turb_out_s)
    P_elec_out = m_dot * w_turb * eta_gen

References
----------
    Crotogino, F., Mohmeyer, K.-U., Scharf, R. (2001). Huntorf CAES: More than 20
        years of successful operation. SMRI Spring Meeting, Orlando.
    Succar, S. & Williams, R.H. (2008). Compressed Air Energy Storage: Theory,
        Resources, and Applications for Wind Power. Princeton Environmental Institute.
    Raju, M. & Khaitan, S.K. (2012). Modeling and simulation of compressed air
        storage in caverns: A case study of the Huntorf plant. Applied Energy, 89, 474-481.
    Budt, M., Wolf, D., Span, R., Yan, J. (2016). A review on compressed air energy
        storage. Applied Energy, 170, 250-268.
    Cengel, Y. & Boles, M. (2015). Thermodynamics: An Engineering Approach, 8th ed.
"""

import numpy as np
from scipy.integrate import solve_ivp, trapezoid


class CAESF2a:
    """Diabatic CAES — lumped cavern mass/energy balance with fuel-fired expansion."""

    def __init__(self, params: dict):
        u = params["unit"]
        # Cavern
        self.V = u["cavern_volume"]["value"]            # m3
        self.p_max = u["p_max"]["value"]                # Pa
        self.p_min = u["p_min"]["value"]                # Pa
        self.T_rock = u["T_rock"]["value"]              # K
        self.UA = u["UA_cav_rock"]["value"]             # W/K

        # Compressor train
        self.n_stages = int(u["n_stages_comp"]["value"])
        self.eta_poly = u["eta_poly_comp"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.T_intercool = u["T_intercool"]["value"]    # K (cavern inlet T)
        self.T_intake = u["T_intake"]["value"]          # K
        self.p_intake = u["p_intake"]["value"]          # Pa

        # Turbine / combustor
        self.eta_turb = u["eta_turb"]["value"]
        self.eta_gen = u["eta_gen"]["value"]
        self.T_turb_in = u["T_turb_in"]["value"]        # K
        self.p_turb_out = u["p_turb_out"]["value"]      # Pa
        self.fuel_lhv = u["fuel_lhv"]["value"]          # J/kg

        # Air properties (hardcoded, cited — Cengel & Boles 2015)
        self.cp = u["cp_air"]["value"]                  # J/(kg.K)
        self.cv = u["cv_air"]["value"]                  # J/(kg.K)
        self.gamma = u["gamma_air"]["value"]
        self.R = u["R_air"]["value"]                    # J/(kg.K)

        # Derived limits at rock temperature
        self.m_max = self.p_max * self.V / (self.R * self.T_rock)
        self.m_min = self.p_min * self.V / (self.R * self.T_rock)

    # ------------------------------------------------------------------
    # Equation of state
    # ------------------------------------------------------------------
    def pressure(self, m, T):
        """Cavern pressure [Pa] from ideal-gas law."""
        return m * self.R * T / self.V

    def mass_from_pressure(self, P, T):
        """Air mass [kg] in cavern at pressure P and temperature T."""
        return P * self.V / (self.R * T)

    # ------------------------------------------------------------------
    # Multistage intercooled compressor
    # ------------------------------------------------------------------
    def compressor_specific_work(self, P_cavern):
        """
        Specific compression work [J/kg] for an n-stage intercooled train that
        raises air from intake (p_intake, T_intake) to cavern pressure P_cavern.

        Per stage uses equal pressure ratio and full intercooling back to T_intake.
        Polytropic-efficiency form (Succar & Williams 2008; Cengel & Boles 2015).
        """
        P_cavern = max(float(P_cavern), self.p_intake * 1.001)
        r_total = P_cavern / self.p_intake
        r_stage = r_total ** (1.0 / self.n_stages)
        exponent = (self.gamma - 1.0) / (self.gamma * self.eta_poly)
        w_stage = (self.gamma / (self.gamma - 1.0)) * self.R * self.T_intake * (
            r_stage ** exponent - 1.0
        )
        return self.n_stages * w_stage

    def charge_power(self, m_dot, P_cavern):
        """Electrical input power [W] to compress m_dot [kg/s] into cavern."""
        if m_dot <= 0:
            return 0.0
        w = self.compressor_specific_work(P_cavern)
        return m_dot * w / self.eta_motor

    # ------------------------------------------------------------------
    # Fuel-fired turbine (combustor + expander)
    # ------------------------------------------------------------------
    def fuel_specific_heat(self, T_cav):
        """Specific combustor heat add [J/kg air] to reach turbine inlet T."""
        return max(self.cp * (self.T_turb_in - T_cav), 0.0)

    def turbine_specific_work(self, P_cavern):
        """
        Specific turbine work [J/kg] expanding heated air (T_turb_in, P_cavern)
        down to exhaust pressure p_turb_out (isentropic * eta_turb).
        """
        P_cavern = max(float(P_cavern), self.p_turb_out * 1.001)
        T_out_s = self.T_turb_in * (self.p_turb_out / P_cavern) ** (
            (self.gamma - 1.0) / self.gamma
        )
        return self.eta_turb * self.cp * (self.T_turb_in - T_out_s)

    def discharge_power(self, m_dot, P_cavern):
        """Electrical output power [W] from expanding m_dot [kg/s] of cavern air."""
        if m_dot <= 0:
            return 0.0
        w = self.turbine_specific_work(P_cavern)
        return m_dot * w * self.eta_gen

    def fuel_power(self, m_dot, T_cav):
        """Thermal fuel power input [W] to the combustor for m_dot [kg/s]."""
        if m_dot <= 0:
            return 0.0
        return m_dot * self.fuel_specific_heat(T_cav)

    # ------------------------------------------------------------------
    # Cavern ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, m_dot_in_fn, m_dot_out_fn):
        """
        State y = [m, T] (cavern air mass [kg], temperature [K]).

        Mass:   dm/dt = m_in - m_out
        Energy: m cv dT/dt = m_in cp T_in - m_out cp T
                              - cv T (m_in - m_out) - UA (T - T_rock)
        """
        m, T = y
        m = max(m, 1e-6)
        m_in = max(m_dot_in_fn(t), 0.0)
        m_out = max(m_dot_out_fn(t), 0.0)
        T_in = self.T_intercool  # compressed air enters cavern after intercooling

        dm = m_in - m_out
        Q_loss = self.UA * (T - self.T_rock)
        # d(m u)/dt = m_in h_in - m_out h_out - Q_loss, with u=cv T, h=cp T
        d_mU = m_in * self.cp * T_in - m_out * self.cp * T - Q_loss
        # m cv dT/dt = d(mU)/dt - u * dm/dt
        dT = (d_mU - self.cv * T * dm) / (m * self.cv)
        return [dm, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, mode, m_dot, T0_K, P0_Pa, dt, duration_s):
        """
        Simulate cavern dynamics under a constant charge/discharge/idle command.

        Parameters
        ----------
        mode : str       "charge", "discharge", or "idle"
        m_dot : float    air mass flow [kg/s] (in for charge, out for discharge)
        T0_K : float     initial cavern temperature [K]
        P0_Pa : float    initial cavern pressure [Pa]
        dt : float       output time step [s]
        duration_s : float total duration [s]

        Returns
        -------
        dict of time series + cumulative energy accounting.
        """
        m0 = self.mass_from_pressure(P0_Pa, T0_K)

        if mode == "charge":
            m_in_fn = lambda t: m_dot
            m_out_fn = lambda t: 0.0
        elif mode == "discharge":
            m_in_fn = lambda t: 0.0
            m_out_fn = lambda t: m_dot
        else:  # idle
            m_in_fn = lambda t: 0.0
            m_out_fn = lambda t: 0.0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [m0, T0_K],
            t_eval=t_eval, args=(m_in_fn, m_out_fn),
            method="RK45", rtol=1e-7, atol=1e-6, max_step=dt,
        )

        t = sol.t
        m = sol.y[0]
        T = sol.y[1]
        N = len(t)
        P = self.pressure(m, T)

        P_elec = np.zeros(N)     # electrical power (+in charge, +out discharge sign by mode)
        P_fuel = np.zeros(N)     # fuel thermal power [W]
        for i in range(N):
            if mode == "charge":
                P_elec[i] = self.charge_power(m_dot, P[i])
            elif mode == "discharge":
                P_elec[i] = self.discharge_power(m_dot, P[i])
                P_fuel[i] = self.fuel_power(m_dot, T[i])

        # Cumulative energy [J] via trapezoid
        E_elec = trapezoid(P_elec, t) if N > 1 else 0.0
        E_fuel = trapezoid(P_fuel, t) if N > 1 else 0.0
        m_fuel = E_fuel / self.fuel_lhv  # kg natural gas

        return {
            "t": t,
            "mass": m,
            "temperature": T,
            "pressure": P,
            "soc": np.clip((m - self.m_min) / (self.m_max - self.m_min), 0.0, 1.0),
            "P_elec": P_elec,
            "P_fuel": P_fuel,
            "E_elec_J": float(E_elec),
            "E_fuel_J": float(E_fuel),
            "m_fuel_kg": float(m_fuel),
            "mode": mode,
        }

    # ------------------------------------------------------------------
    # Round-trip efficiency (steady specific-energy accounting)
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, P_charge=None, P_discharge=None):
        """
        Diabatic round-trip efficiency including fuel input:

            eta_RT = w_out_elec / (w_in_elec + q_fuel)

        evaluated on a per-kg-of-air basis at representative cavern pressures.
        Charge work is referenced to P_charge (default p_max), discharge work and
        fuel to P_discharge (default p_min, the turbine throttle pressure).
        """
        if P_charge is None:
            P_charge = self.p_max
        if P_discharge is None:
            P_discharge = self.p_min
        w_in = self.compressor_specific_work(P_charge) / self.eta_motor
        w_out = self.turbine_specific_work(P_discharge) * self.eta_gen
        q_fuel = self.fuel_specific_heat(self.T_rock)
        return w_out / (w_in + q_fuel)

    def electric_rte(self, P_charge=None, P_discharge=None):
        """Electricity-only RTE (excludes fuel; >1 for diabatic CAES)."""
        if P_charge is None:
            P_charge = self.p_max
        if P_discharge is None:
            P_discharge = self.p_min
        w_in = self.compressor_specific_work(P_charge) / self.eta_motor
        w_out = self.turbine_specific_work(P_discharge) * self.eta_gen
        return w_out / w_in

    def heat_rate(self, P_discharge=None):
        """Plant heat rate [kJ/kWh_e] = fuel heat per unit electric output."""
        if P_discharge is None:
            P_discharge = self.p_min
        q_fuel = self.fuel_specific_heat(self.T_rock)          # J/kg
        w_out = self.turbine_specific_work(P_discharge) * self.eta_gen  # J/kg
        return (q_fuel / w_out) * 3600.0  # J/J -> kJ/kWh
