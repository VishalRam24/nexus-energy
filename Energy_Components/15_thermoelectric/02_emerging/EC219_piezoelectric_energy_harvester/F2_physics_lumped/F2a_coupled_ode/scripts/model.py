"""
EC219 -- Piezoelectric Energy Harvester -- F2a Coupled Electromechanical ODE
============================================================================

Physics-lumped (0-D, 1-DOF) time-domain model of a piezoelectric bimorph
cantilever harvester under harmonic base excitation, integrated with
scipy.integrate.solve_ivp.

The device is reduced to a single mechanical mode (tip displacement x relative
to the base) electromechanically coupled to the piezoelectric electrical branch
(clamped capacitance C_p in parallel with a resistive load R_L). The governing
lumped equations (Erturk & Inman 2011, Ch. 3-4; duToit et al. 2005) are:

    Mechanical:   m*x_ddot + c*x_dot + k*x - theta*V = -m*a_base(t)
    Electrical:   C_p*V_dot + V/R_L + theta*x_dot = 0

where
    x          relative tip displacement                       [m]
    V          voltage across the load                          [V]
    m          effective modal mass                             [kg]
    c          mechanical damping coefficient                   [N.s/m]
    k          effective modal stiffness                        [N/m]
    theta      electromechanical coupling                       [N/V] = [C/m]
    C_p        piezo clamped (parallel) capacitance             [F]
    R_L        electrical load resistance                       [ohm]
    a_base(t)  base acceleration (here A*sin(omega*t))          [m/s^2]

State vector y = [x, x_dot, V]. Recast as first-order system:

    x_dot   = v
    v_dot   = (-c*v - k*x + theta*V - m*a)/m
    V_dot   = (-V/R_L - theta*v)/C_p

Effective lumped parameters are derived from the same PZT-5A bimorph geometry
used by the F1 model (modal mass fraction 0.236 for a uniform cantilever,
Erturk & Inman 2011 Eq. 3.32; bimorph series capacitance and bending-coupling
theta from the neutral-axis to piezo-centroid distance).

Physics guarantees enforced by the test suite:
  * Steady-state average harvested power peaks when excitation frequency
    matches resonance (Erturk & Inman 2011).
  * An optimal load resistance exists (impedance matching, R_opt ~ 1/(w_n*C_p);
    duToit et al. 2005, Roundy 2005).
  * Energy conservation: input work by base force = mechanical damping
    dissipation + electrical (resistor) dissipation + stored (mech+elec) energy.
  * Average power scales with acceleration^2 (linear system).

References:
    Erturk, A. & Inman, D.J. (2011). Piezoelectric Energy Harvesting. Wiley.
    duToit, N.E., Wardle, B.L. & Kim, S.-G. (2005). Integr. Ferroelectr. 71, 121-160.
    Roundy, S. (2005). J. Intell. Mater. Syst. Struct. 16(10), 809-823.
"""

import numpy as np
from scipy.integrate import solve_ivp


class PiezoF2a:
    """Coupled electromechanical ODE model of a piezoelectric bimorph harvester."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.d31 = abs(u["d31"]["value"])                  # m/V
        self.Y_p = u["Y_piezo"]["value"]                   # Pa
        self.Y_s = u["Y_substrate"]["value"]               # Pa
        self.eps_33T = u["epsilon_33_T"]["value"]          # F/m
        self.L = u["beam_length"]["value"]                 # m
        self.b = u["beam_width"]["value"]                  # m
        self.t_p = u["beam_thickness_piezo"]["value"]      # m
        self.t_s = u["beam_thickness_substrate"]["value"]  # m
        self.rho_p = u["rho_piezo"]["value"]               # kg/m^3
        self.rho_s = u["rho_substrate"]["value"]           # kg/m^3
        self.m_tip = u["mass_tip"]["value"]                # kg
        self.zeta_mech = u["zeta_mech"]["value"]           # -
        self.f_n = u["resonance_freq_hz"]["value"]         # Hz
        self.k31 = u["k31"]["value"]                       # -

        self.omega_n = 2.0 * np.pi * self.f_n              # rad/s (short-circuit)

        # --- Effective modal mass: 0.236 of distributed beam mass + tip mass ---
        m_beam_p = self.rho_p * self.L * self.b * self.t_p * 2.0  # two PZT layers
        m_beam_s = self.rho_s * self.L * self.b * self.t_s
        self.m_eff = 0.236 * (m_beam_p + m_beam_s) + self.m_tip   # kg

        # --- Effective modal stiffness from short-circuit resonance ---
        self.k_eff = self.m_eff * self.omega_n ** 2              # N/m

        # --- Mechanical damping coefficient c = 2*zeta*sqrt(k*m) ---
        self.c_mech = 2.0 * self.zeta_mech * np.sqrt(self.k_eff * self.m_eff)  # N.s/m

        # --- Piezo clamped capacitance (series bimorph) ---
        self.C_p = 0.5 * self.eps_33T * self.b * self.L / self.t_p  # F

        # --- Electromechanical coupling theta [N/V] ---
        t_pc = (self.t_s / 2.0 + self.t_p / 2.0)  # neutral axis -> piezo centroid
        self.theta = self.d31 * self.Y_p * self.b * t_pc           # N/V

    # ------------------------------------------------------------------ #
    #  Right-hand side of the coupled first-order ODE system
    # ------------------------------------------------------------------ #
    def _rhs(self, t, y, accel_func, R_L):
        x, v, V = y
        a = accel_func(t)
        dxdt = v
        dvdt = (-self.c_mech * v - self.k_eff * x + self.theta * V
                - self.m_eff * a) / self.m_eff
        dVdt = (-V / R_L - self.theta * v) / self.C_p
        return [dxdt, dvdt, dVdt]

    # ------------------------------------------------------------------ #
    #  Time-domain simulation
    # ------------------------------------------------------------------ #
    def simulate(self, acceleration_ms2, frequency_hz, R_load_ohm,
                 duration_s=None, n_periods=40, n_per_period=64, y0=None):
        """
        Integrate the coupled ODEs under harmonic base excitation
        a_base(t) = A * sin(omega * t).

        Parameters
        ----------
        acceleration_ms2 : float -- amplitude A of base acceleration [m/s^2]
        frequency_hz     : float -- excitation frequency f [Hz]
        R_load_ohm       : float -- load resistance [ohm]
        duration_s       : float -- total integration time; default n_periods/f
        n_periods        : int   -- number of forcing periods (if duration auto)
        n_per_period     : int   -- output samples per period
        y0               : list  -- initial [x, v, V]; default zeros

        Returns
        -------
        dict with time-series arrays and steady-state scalar metrics.
        """
        A = float(acceleration_ms2)
        f = float(frequency_hz)
        R_L = float(R_load_ohm)
        omega = 2.0 * np.pi * f

        if duration_s is None:
            duration_s = n_periods / f
        n_pts = max(int(round(duration_s * f * n_per_period)), 200)
        t_eval = np.linspace(0.0, duration_s, n_pts)

        accel_func = lambda t: A * np.sin(omega * t)
        if y0 is None:
            y0 = [0.0, 0.0, 0.0]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(accel_func, R_L),
            method="RK45", rtol=1e-8, atol=1e-12, max_step=1.0 / (f * 20.0),
        )

        t = sol.t
        x = sol.y[0]
        v = sol.y[1]
        V = sol.y[2]

        # Instantaneous electrical power into the load
        P_inst = V ** 2 / R_L

        # Steady-state window = last 50% of the record (transient decayed)
        ss = t >= (0.5 * duration_s)
        P_avg = float(np.mean(P_inst[ss]))            # W, average harvested power
        V_rms = float(np.sqrt(np.mean(V[ss] ** 2)))   # V
        V_amp = float(np.max(np.abs(V[ss])))          # V
        x_amp = float(np.max(np.abs(x[ss])))          # m

        return {
            "t": t,
            "x": x,
            "v": v,
            "voltage": V,
            "power_inst": P_inst,
            "P_avg": P_avg,
            "V_rms": V_rms,
            "V_amp": V_amp,
            "x_amp": x_amp,
            "frequency_hz": f,
            "acceleration_ms2": A,
            "R_load_ohm": R_L,
            "frequency_ratio": f / self.f_n,
        }

    # ------------------------------------------------------------------ #
    #  Analytical helpers (steady-state, for cross-checks)
    # ------------------------------------------------------------------ #
    def optimal_load(self, frequency_hz=None):
        """Impedance-matched load resistance R_opt = 1/(omega*C_p).
        duToit et al. (2005); Roundy (2005)."""
        f = self.f_n if frequency_hz is None else float(frequency_hz)
        omega = 2.0 * np.pi * f
        return 1.0 / (omega * self.C_p)

    def steady_state_power(self, acceleration_ms2, frequency_hz, R_load_ohm):
        """Closed-form steady-state average power from the coupled
        frequency-domain transfer function (Erturk & Inman 2011, Eq. 6.x).
        Average resistor power for sinusoidal V is |V|^2/(2 R_L)."""
        A = float(acceleration_ms2)
        omega = 2.0 * np.pi * float(frequency_hz)
        R_L = float(R_load_ohm)
        F = self.m_eff * A
        Z_mech = self.k_eff - self.m_eff * omega ** 2 + 1j * self.c_mech * omega
        D = Z_mech * (1.0 + 1j * omega * R_L * self.C_p) + 1j * omega * self.theta ** 2 * R_L
        V_load = -1j * omega * self.theta * R_L * F / D
        P = 0.5 * np.abs(V_load) ** 2 / R_L
        return float(P), float(np.abs(V_load))

    def energy_balance(self, sim_result):
        """
        Verify global energy conservation over the simulation record:

            W_in = E_diss_mech + E_diss_elec + dE_stored

        where
            W_in        = integral of base force power  ( -m*a * v ) dt
            E_diss_mech = integral of c*v^2 dt
            E_diss_elec = integral of V^2/R_L dt
            dE_stored   = change in (0.5 m v^2 + 0.5 k x^2 + 0.5 C_p V^2)

        Returns a dict with each term and the relative closure error.
        """
        t = sim_result["t"]
        x = sim_result["x"]
        v = sim_result["v"]
        V = sim_result["voltage"]
        R_L = sim_result["R_load_ohm"]
        A = sim_result["acceleration_ms2"]
        omega = 2.0 * np.pi * sim_result["frequency_hz"]
        a = A * np.sin(omega * t)

        # Power injected by base excitation onto the modal mass:
        # equation of motion forcing term is (-m*a); power = force * velocity.
        P_in = -self.m_eff * a * v
        W_in = np.trapz(P_in, t)

        E_diss_mech = np.trapz(self.c_mech * v ** 2, t)
        E_diss_elec = np.trapz(V ** 2 / R_L, t)

        E_stored = (0.5 * self.m_eff * v ** 2
                    + 0.5 * self.k_eff * x ** 2
                    + 0.5 * self.C_p * V ** 2)
        dE_stored = float(E_stored[-1] - E_stored[0])

        residual = W_in - (E_diss_mech + E_diss_elec + dE_stored)
        scale = abs(W_in) + abs(E_diss_mech) + abs(E_diss_elec) + 1e-30
        rel_err = abs(residual) / scale

        return {
            "W_in": float(W_in),
            "E_diss_mech": float(E_diss_mech),
            "E_diss_elec": float(E_diss_elec),
            "dE_stored": dE_stored,
            "residual": float(residual),
            "rel_error": float(rel_err),
        }
