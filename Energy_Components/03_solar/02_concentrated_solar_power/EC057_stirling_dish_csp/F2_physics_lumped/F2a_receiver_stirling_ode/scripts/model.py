"""
EC057 -- Stirling Dish CSP -- F2a Physics-Lumped (Receiver Thermal ODE + Stirling Engine)

A 0-D first-principles transient model of a parabolic-dish / cavity-receiver /
Stirling-engine unit.  The parabolic dish concentrates DNI (~3000x) onto a cavity
receiver; the receiver wall has a lumped thermal capacitance whose temperature is
governed by an energy-balance ODE.  The net thermal power extracted from the
receiver gas drives a Stirling engine whose conversion efficiency is Carnot-limited
and de-rated by regenerator/mechanical (2nd-law) losses, giving net electrical power.

------------------------------------------------------------------------------
Receiver lumped energy balance (transient state variable: receiver wall T_rec):

    m_rec * cp_rec * dT_rec/dt = Q_absorbed - Q_conv - Q_rad - Q_cond - Q_engine

    Q_absorbed = alpha_rec * eta_optical * IAM(theta) * DNI * A_dish        [W]
    Q_conv     = h_cav  * A_rec * (T_rec - T_amb)                           [W]
    Q_cond     = U_cond * A_rec * (T_rec - T_amb)                           [W]
    Q_rad      = eps_rec * sigma * A_rec * (T_rec^4 - T_sky^4)              [W]  (Stefan-Boltzmann, T^4)
    Q_engine   = thermal power drawn by the Stirling engine                 [W]

The radiative term is the dominant high-temperature loss for point-focus cavity
receivers and is rigorously proportional to T^4 (Stine & Diver 1994; Nepveu 2009).

------------------------------------------------------------------------------
Stirling engine thermodynamics (Carnot-limited with internal losses):

    eta_carnot   = 1 - T_sink / T_hot                  (must bound the engine)
    eta_stirling = eta_internal * eta_carnot * eta_alt (relative-Carnot / Beale form)

    T_hot  = T_rec - dT_receiver        (working-gas hot-space temperature)   [K]
    T_sink = T_amb + T_approach         (cooler/radiator cold-space temp)     [K]

    eta_internal lumps regenerator effectiveness, mechanical friction, pumping
    and hysteresis losses into a single relative-Carnot factor (~0.5-0.6 for a
    well-developed kinematic Stirling, Stine & Diver 1994, EuroDish/SES data).

    Net electric:  P_elec = max(0, eta_stirling * Q_engine - P_parasitic)

The engine only draws power when the receiver gas can deliver more than a minimum
threshold (de-stroke control); below that, Q_engine = 0 and the receiver simply
heats/cools passively.  This makes the model a genuine coupled ODE (receiver
temperature feeds engine efficiency AND engine draw feeds the receiver balance).

------------------------------------------------------------------------------
Physical guarantees enforced by construction:
  * Radiative loss strictly proportional to T_rec^4.
  * eta_stirling < eta_carnot  (internal+alternator factors are < 1).
  * P_elec = 0 when DNI = 0 at steady state (no absorbed power -> receiver cools,
    Q_engine -> 0).
  * Energy conservation: Q_absorbed = sum(losses) + Q_engine + storage term, to
    machine precision (verified in test_model.py).

References:
    Stine, W. & Diver, R. (1994). 'A Compendium of Solar Dish/Stirling
        Technology', SAND93-7026, Sandia National Laboratories.
    Mancini, T. et al. (2003). 'Dish-Stirling Systems: An Overview of
        Development and Status', J. Sol. Energy Eng. 125(2), 135-151.
    Nepveu, F., Ferriere, A. & Bataille, F. (2009). 'Thermal model of a
        dish/Stirling system', Solar Energy 83(1), 81-89.
    Stine, W. & Geyer, M. (2001). 'Power From The Sun' (online textbook),
        Ch. 12 Concentrator Optics & Receivers.
"""

import numpy as np
from scipy.integrate import solve_ivp


class StirlingDishF2a:
    """Dish-Stirling unit -- lumped receiver thermal ODE + Carnot-limited engine."""

    SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/m2K4]

    def __init__(self, params: dict):
        u = params["unit"]
        # Optics
        self.A_dish = u["A_dish"]["value"]
        self.eta_optical = u["eta_optical"]["value"]
        # Receiver geometry / radiative-convective
        self.A_rec = u["A_rec"]["value"]
        self.eps_rec = u["eps_rec"]["value"]
        self.alpha_rec = u["alpha_rec"]["value"]
        self.h_cav = u["h_cav"]["value"]
        self.U_cond = u["U_cond"]["value"]
        self.T_sky_offset = u["T_sky_offset"]["value"]
        # Receiver thermal mass
        self.m_rec = u["m_rec"]["value"]
        self.cp_rec = u["cp_rec"]["value"]
        self.C_rec = self.m_rec * self.cp_rec  # J/K lumped heat capacity
        # Engine coupling temperatures
        self.dT_receiver = u["dT_receiver"]["value"]
        self.T_approach = u["T_approach"]["value"]
        # Engine conversion
        self.eta_internal = u["eta_internal"]["value"]
        self.eta_alt = u["eta_alt"]["value"]
        self.P_parasitic = u["P_parasitic_kw"]["value"] * 1000.0   # W
        self.Q_engine_min = u["Q_engine_min_kw"]["value"] * 1000.0  # W
        self.T_engine_on = u["T_engine_on_c"]["value"]             # degC heater-head startup
        self.T_rec_design = u["T_rec_design"]["value"]             # degC design head temp
        self.P_rated = u["P_rated_kw"]["value"]

    # ------------------------------------------------------------------
    # Optics: Incidence-Angle Modifier (two-axis tracking, residual error)
    # ------------------------------------------------------------------
    def iam(self, theta_deg):
        """IAM = cos(theta) for residual tracking error, clipped to [0,1]."""
        theta = np.asarray(theta_deg, dtype=float)
        return np.clip(np.cos(np.radians(np.minimum(theta, 89.9))), 0.0, 1.0)

    def Q_absorbed(self, dni, theta_deg):
        """Concentrated solar power absorbed by the cavity wall [W]."""
        G = np.asarray(dni, dtype=float)
        return self.alpha_rec * self.eta_optical * self.iam(theta_deg) * G * self.A_dish

    # ------------------------------------------------------------------
    # Receiver loss terms [W] (T_rec, T_amb in degC)
    # ------------------------------------------------------------------
    def Q_conv(self, T_rec_c, T_amb_c):
        T_rec = T_rec_c + 273.15
        T_amb = T_amb_c + 273.15
        return self.h_cav * self.A_rec * (T_rec - T_amb)

    def Q_cond(self, T_rec_c, T_amb_c):
        T_rec = T_rec_c + 273.15
        T_amb = T_amb_c + 273.15
        return self.U_cond * self.A_rec * (T_rec - T_amb)

    def Q_rad(self, T_rec_c, T_amb_c):
        """Radiative re-emission, strictly proportional to T_rec^4."""
        T_rec = T_rec_c + 273.15
        T_sky = (T_amb_c + 273.15) - self.T_sky_offset
        return self.eps_rec * self.SIGMA * self.A_rec * (T_rec**4 - T_sky**4)

    def Q_loss(self, T_rec_c, T_amb_c):
        """Total passive receiver loss [W] (convection + conduction + radiation)."""
        return (self.Q_conv(T_rec_c, T_amb_c)
                + self.Q_cond(T_rec_c, T_amb_c)
                + self.Q_rad(T_rec_c, T_amb_c))

    # ------------------------------------------------------------------
    # Stirling engine efficiency (Carnot-limited)
    # ------------------------------------------------------------------
    def eta_carnot(self, T_rec_c, T_amb_c):
        """Ideal Carnot efficiency between hot gas and cold sink [-]."""
        T_hot = (T_rec_c - self.dT_receiver) + 273.15
        T_sink = (T_amb_c + self.T_approach) + 273.15
        return np.where(T_hot > T_sink, 1.0 - T_sink / np.maximum(T_hot, 1e-6), 0.0)

    def eta_stirling(self, T_rec_c, T_amb_c):
        """
        Net thermal-to-electric efficiency, relative-Carnot (Beale-style) form.
        eta = eta_internal * eta_carnot * eta_alt  -- strictly below Carnot.
        """
        eta_c = self.eta_carnot(T_rec_c, T_amb_c)
        eta = self.eta_internal * eta_c * self.eta_alt
        # Hard guarantee: never exceed (or equal) the Carnot limit.
        return np.clip(eta, 0.0, np.maximum(eta_c - 1e-12, 0.0))

    # ------------------------------------------------------------------
    # Engine thermal draw + net electric power
    # ------------------------------------------------------------------
    def engine_load_fraction(self, T_rec_c):
        """
        Heater-head de-stroke / load fraction in [0,1].
        The Stirling engine cannot run until the heater head reaches its startup
        temperature (T_engine_on); it then ramps to full stroke at the design
        temperature.  This staged engagement (a) reflects real dish-Stirling
        control (Stine & Diver 1994) and (b) makes the receiver a true storage
        node that warms up before the engine begins extracting heat.
        """
        T = np.asarray(T_rec_c, dtype=float)
        frac = (T - self.T_engine_on) / max(self.T_rec_design - self.T_engine_on, 1e-6)
        return np.clip(frac, 0.0, 1.0)

    def Q_engine(self, T_rec_c, T_amb_c, dni, theta_deg):
        """
        Thermal power [W] the engine extracts from the receiver gas.
        Draw = load_fraction(T_rec) * available net flux, engaged only above the
        startup temperature and above the minimum de-stroke threshold.  Because
        the load fraction is 0 below T_engine_on, a cold receiver keeps all the
        absorbed flux as stored enthalpy (m*cp*dT/dt) and warms up.
        """
        Q_avail = self.Q_absorbed(dni, theta_deg) - self.Q_loss(T_rec_c, T_amb_c)
        Q_avail = np.asarray(Q_avail, dtype=float)
        Q_draw = self.engine_load_fraction(T_rec_c) * np.maximum(Q_avail, 0.0)
        return np.where(Q_draw > self.Q_engine_min, Q_draw, 0.0)

    def power_output_w(self, T_rec_c, T_amb_c, dni, theta_deg):
        """Net electrical output [W] after engine efficiency and parasitics."""
        Q_eng = self.Q_engine(T_rec_c, T_amb_c, dni, theta_deg)
        eta = self.eta_stirling(T_rec_c, T_amb_c)
        P_gross = eta * Q_eng
        running = Q_eng > 0.0
        P_net = np.where(running, P_gross - self.P_parasitic, 0.0)
        return np.maximum(P_net, 0.0)

    # ------------------------------------------------------------------
    # Receiver thermal ODE
    # ------------------------------------------------------------------
    def dTdt(self, T_rec_c, T_amb_c, dni, theta_deg):
        """Receiver wall temperature rate of change [degC/s] = [K/s]."""
        Q_abs = self.Q_absorbed(dni, theta_deg)
        Q_loss = self.Q_loss(T_rec_c, T_amb_c)
        Q_eng = self.Q_engine(T_rec_c, T_amb_c, dni, theta_deg)
        return (Q_abs - Q_loss - Q_eng) / self.C_rec

    # ------------------------------------------------------------------
    # Time-domain simulation via scipy.solve_ivp
    # ------------------------------------------------------------------
    def simulate(self, dni, theta_deg, T_rec_init_c, T_amb_c, dt, duration_s):
        """
        Integrate the receiver thermal ODE and reconstruct engine output.

        Parameters
        ----------
        dni          : float or callable(t)  -- Direct Normal Irradiance [W/m2]
        theta_deg    : float                 -- residual tracking incidence angle [deg]
        T_rec_init_c : float                 -- initial receiver wall temperature [degC]
        T_amb_c      : float                 -- ambient temperature [degC]
        dt           : float                 -- output time step [s]
        duration_s   : float                 -- total duration [s]

        Returns
        -------
        dict of time-series arrays.
        """
        _dni = dni if callable(dni) else (lambda t: dni)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T_rec = y[0]
            return [self.dTdt(T_rec, T_amb_c, _dni(t), theta_deg)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_rec_init_c],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-9,
            max_step=max(dt, 1.0),
        )

        t_out = sol.t
        T_rec = sol.y[0]
        N = len(t_out)

        P_elec = np.zeros(N)
        eta_eng = np.zeros(N)
        eta_car = np.zeros(N)
        Q_abs = np.zeros(N)
        Q_loss = np.zeros(N)
        Q_rad = np.zeros(N)
        Q_eng = np.zeros(N)
        eta_sys = np.zeros(N)

        for i in range(N):
            g = _dni(t_out[i])
            Tr = T_rec[i]
            Q_abs[i] = self.Q_absorbed(g, theta_deg)
            Q_loss[i] = self.Q_loss(Tr, T_amb_c)
            Q_rad[i] = self.Q_rad(Tr, T_amb_c)
            Q_eng[i] = self.Q_engine(Tr, T_amb_c, g, theta_deg)
            eta_car[i] = self.eta_carnot(Tr, T_amb_c)
            eta_eng[i] = self.eta_stirling(Tr, T_amb_c)
            P_elec[i] = self.power_output_w(Tr, T_amb_c, g, theta_deg)
            P_incident = g * self.A_dish
            eta_sys[i] = P_elec[i] / P_incident if P_incident > 1.0 else 0.0

        return {
            "t": t_out,
            "T_rec_c": T_rec,
            "P_elec_kw": P_elec / 1000.0,
            "Q_absorbed_kw": Q_abs / 1000.0,
            "Q_loss_kw": Q_loss / 1000.0,
            "Q_rad_kw": Q_rad / 1000.0,
            "Q_engine_kw": Q_eng / 1000.0,
            "eta_carnot": eta_car,
            "eta_stirling": eta_eng,
            "eta_system": eta_sys,
        }

    # ------------------------------------------------------------------
    # Steady-state receiver temperature (engine OFF, for diagnostics)
    # ------------------------------------------------------------------
    def steady_state_temperature(self, dni, theta_deg, T_amb_c,
                                 T_guess_c=700.0, duration_s=3600.0):
        """Long-run integration to the steady receiver temperature [degC]."""
        r = self.simulate(dni, theta_deg, T_guess_c, T_amb_c, 10.0, duration_s)
        return r["T_rec_c"][-1]
