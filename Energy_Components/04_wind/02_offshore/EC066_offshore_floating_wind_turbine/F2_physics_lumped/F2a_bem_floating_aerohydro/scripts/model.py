"""
EC066 -- Offshore Floating Wind Turbine -- F2a BEM Rotor + Floating-Platform
        Aero-Hydro Coupled Lumped Model

First-principles 0D lumped dynamic model of a large multi-MW floating offshore
wind turbine. Three coupled ODEs are integrated with scipy.integrate.solve_ivp:

  State vector  y = [Omega, x, x_dot, theta, theta_dot]

  (1) Rotor drivetrain (Newton's 2nd law for rotation):
        I_rotor * dOmega/dt = Q_aero(V_rel, Omega, beta) - Q_gen(Omega)
      Aerodynamic torque from BEM-based power coefficient:
        P_aero = 0.5 * rho * A * Cp(lambda, beta) * V_rel^3
        Q_aero = P_aero / Omega                       (Hansen 2015, Burton 2011)
      Cp(lambda,beta) uses the Heier/Slootweg parametric BEM fit, rescaled to
      the IEA-15MW peak Cp = 0.489 (< Betz 16/27 = 0.593).

  (2) Platform surge (2nd-order mass-spring-damper, floating foundation):
        (m + a_surge) * x_ddot = F_thrust - c_surge*x_dot - k_surge*x + F_wave
      F_thrust = 0.5*rho*A*Ct(lambda,beta)*V_rel^2  (rotor aero thrust)
      k_surge  = mooring restoring stiffness;  F_wave = linear-wave excitation.

  (3) Platform pitch (2nd-order, hydrostatic + mooring restoring):
        I_pitch * theta_ddot = F_thrust*L_hub - c_pitch*theta_dot
                               - k_pitch*theta + M_wave
      k_pitch = hydrostatic + mooring pitch restoring stiffness.

  AERO-HYDRO COUPLING (the physics this model demonstrates):
      The platform surge/pitch velocities change the *relative* wind seen by the
      rotor at hub height:
        V_rel = V_wind - x_dot - L_hub * theta_dot
      A nodding/surging floater therefore modulates V_rel, hence Cp, torque,
      power AND thrust -- a genuine two-way aero-hydro feedback loop. Fore-aft
      platform motion can drive negative aerodynamic damping and floating-
      specific power fluctuations (Jonkman 2010; Larsen & Hanson 2007).

Conservation / bounds enforced:
  * Cp(lambda,beta) <= Betz limit 16/27 by construction (clipped).
  * Extracted aero power <= kinetic flux 0.5*rho*A*V_rel^3 (Cp<=1 guarantees).
  * Platform motion bounded by linear restoring stiffness (stable spring-damper).
  * Generator/electrical efficiency in (0,1).

References:
  Hansen, M.O.L. (2015). Aerodynamics of Wind Turbines, 3rd ed., Routledge (BEM).
  Burton, T. et al. (2011). Wind Energy Handbook, 2nd ed., Wiley.
  Heier, S. (1998); Slootweg et al. (2003) IEEE Trans. PWRS 18(1) -- Cp(lambda,beta).
  Gaertner, E. et al. (2020). NREL/TP-5000-75698, IEA 15 MW RWT.
  Allen, C. et al. (2020). NREL/TP-5000-76773, UMaine VolturnUS-S platform.
  Jonkman, J. (2010). NREL/TP-500-47535, OC3-Hywind spar dynamics.
  Larsen, T.J. & Hanson, T.D. (2007). J. Phys. Conf. Ser. 75, 012073
      (negative aerodynamic damping of floating wind turbines).
"""

import numpy as np
from scipy.integrate import solve_ivp

BETZ = 16.0 / 27.0  # 0.5926 -- theoretical max power coefficient


class FloatingWindF2a:
    """Floating offshore wind turbine: BEM rotor coupled to surge/pitch floater."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power"]["value"] * 1e3        # W
        self.D = u["rotor_diameter"]["value"]                 # m
        self.R = self.D / 2.0                                 # m
        self.A = np.pi * self.R ** 2                          # m2
        self.H = u["hub_height"]["value"]                     # m
        self.rho = u["rho_air"]["value"]                      # kg/m3
        self.cut_in = u["cut_in_speed"]["value"]
        self.cut_out = u["cut_out_speed"]["value"]
        self.B = u["n_blades"]["value"]

        self.Omega_rated = u["rated_rotor_speed"]["value"]    # rad/s
        self.tsr_opt = u["tsr_opt"]["value"]
        self.cp_max = u["cp_max"]["value"]
        self.I_rotor = u["I_rotor"]["value"]                  # kg.m2
        self.eta_gen = u["generator_efficiency"]["value"]

        # Cp(lambda,beta) parametric coefficients (Heier/Slootweg)
        self.cp_a = u["cp_a"]["value"]
        self.cp_c2 = u["cp_c2"]["value"]
        self.cp_c3 = u["cp_c3"]["value"]
        self.cp_c4 = u["cp_c4"]["value"]
        self.cp_c5 = u["cp_c5"]["value"]
        self.cp_c6 = u["cp_c6"]["value"]
        self.cp_scale = u["cp_scale"]["value"]

        # Floating platform -- surge
        self.m_plat = u["platform_mass"]["value"]
        self.a_surge = u["platform_added_mass_surge"]["value"]
        self.k_surge = u["mooring_surge_stiffness"]["value"]
        self.c_surge = u["mooring_surge_damping"]["value"]

        # Floating platform -- pitch
        self.I_pitch = u["platform_pitch_inertia"]["value"]
        self.k_pitch = u["pitch_restoring_stiffness"]["value"]
        self.c_pitch = u["pitch_damping"]["value"]
        self.L_hub = u["thrust_lever_arm"]["value"]

        # Thrust coefficient parametric peak
        self.ct_a = u["ct_a"]["value"]
        self.ct_rated = u["ct_rated"]["value"]

        # Rated torque target for the simple torque controller
        self.Q_gen_rated = self.P_rated / self.Omega_rated / self.eta_gen

    # ------------------------------------------------------------------
    # Aerodynamics: Cp(lambda, beta) parametric BEM fit
    # ------------------------------------------------------------------
    def power_coefficient(self, lam, beta_deg=0.0):
        """
        Rotor power coefficient Cp(tip-speed-ratio, blade pitch).
        Heier (1998) / Slootweg et al. (2003) parametric form derived from BEM,
        rescaled so the peak matches the IEA-15MW BEM value (0.489).
        Hard-clipped to the Betz limit (16/27) to enforce momentum theory.
        """
        lam = np.asarray(lam, dtype=float)
        beta = np.asarray(beta_deg, dtype=float)
        # 1/lambda_i with pitch dependence (Slootweg form)
        inv_li = 1.0 / (lam + 0.08 * beta) - 0.035 / (beta ** 3 + 1.0)
        # guard: only valid for lam>0
        inv_li = np.where(lam > 0, inv_li, 0.0)
        cp = self.cp_a * (
            self.cp_c2 * inv_li - self.cp_c3 * beta - self.cp_c4
        ) * np.exp(-self.cp_c5 * inv_li) + self.cp_c6 * lam
        cp = self.cp_scale * cp
        cp = np.clip(cp, 0.0, BETZ)
        # zero Cp outside a sensible TSR window
        cp = np.where(lam > 0.5, cp, 0.0)
        return cp

    def thrust_coefficient(self, lam, beta_deg=0.0):
        """
        Rotor thrust coefficient Ct(lambda, beta), bounded by 1D momentum limit
        Ct <= 1 (here capped at 0.95). Ct is high at low V_rel (high lambda) and
        falls toward rated/feathered operation. Simple physically-shaped curve.
        """
        lam = np.asarray(lam, dtype=float)
        beta = np.asarray(beta_deg, dtype=float)
        # Ct grows with tip-speed ratio (more induction), drops with pitch (feather)
        ct = self.ct_rated * (lam / self.tsr_opt) * np.exp(-0.05 * beta)
        ct = np.clip(ct, 0.0, 0.95)
        return ct

    def tip_speed_ratio(self, Omega, V_rel):
        """lambda = Omega * R / V_rel."""
        V_rel = np.maximum(V_rel, 1e-3)
        return Omega * self.R / V_rel

    # ------------------------------------------------------------------
    # Aerodynamic loads at given relative wind
    # ------------------------------------------------------------------
    def aero_power(self, V_rel, Omega, beta_deg=0.0):
        """Aerodynamic (rotor) power [W] = 0.5 rho A Cp V_rel^3."""
        lam = self.tip_speed_ratio(Omega, V_rel)
        cp = self.power_coefficient(lam, beta_deg)
        return 0.5 * self.rho * self.A * cp * np.maximum(V_rel, 0.0) ** 3

    def aero_torque(self, V_rel, Omega, beta_deg=0.0):
        """Aerodynamic rotor torque [N.m] = P_aero / Omega."""
        Omega = np.maximum(Omega, 1e-3)
        return self.aero_power(V_rel, Omega, beta_deg) / Omega

    def aero_thrust(self, V_rel, Omega, beta_deg=0.0):
        """Rotor aero thrust [N] = 0.5 rho A Ct V_rel^2."""
        lam = self.tip_speed_ratio(Omega, V_rel)
        ct = self.thrust_coefficient(lam, beta_deg)
        return 0.5 * self.rho * self.A * ct * np.maximum(V_rel, 0.0) ** 2

    def generator_torque(self, Omega):
        """
        Simple Region-2 / Region-3 torque controller (Kistner/NREL ROSCO-like).
        Region 2 (below rated): Q_gen = K_opt * Omega^2 tracks Cp_max.
        Region 3 (above rated): hold rated torque to cap power.
        """
        # Optimal-mode-gain: K = 0.5 rho A R^3 Cp_max / lambda_opt^3
        K_opt = 0.5 * self.rho * self.A * self.R ** 3 * self.cp_max / self.tsr_opt ** 3
        Q = K_opt * Omega ** 2
        return np.minimum(Q, self.Q_gen_rated)

    # ------------------------------------------------------------------
    # Wave forcing (linear Airy-type harmonic excitation)
    # ------------------------------------------------------------------
    def wave_forcing(self, t, H_wave, T_wave):
        """
        Linear regular-wave excitation: surge force and pitch moment.
        Amplitude scaled by wave height; sinusoidal at the wave frequency.
        (Simplified Morison/diffraction surrogate -- Faltinsen 1990.)
        """
        if H_wave <= 0.0 or T_wave <= 0.0:
            return 0.0, 0.0
        omega_w = 2.0 * np.pi / T_wave
        # excitation amplitudes scale ~ with wave amplitude (H/2)
        F_amp = 5.0e5 * (H_wave / 2.0)        # N per m of amplitude (order-of-mag)
        M_amp = 1.2e8 * (H_wave / 2.0)        # N.m per m of amplitude
        F = F_amp * np.cos(omega_w * t)
        M = M_amp * np.cos(omega_w * t)
        return F, M

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, V_wind_fn, beta_deg, H_wave, T_wave):
        Omega, x, x_dot, theta, theta_dot = y

        V_wind = V_wind_fn(t)
        # AERO-HYDRO COUPLING: platform motion modifies relative wind at hub
        V_rel = V_wind - x_dot - self.L_hub * theta_dot
        V_rel = max(V_rel, 0.0)

        # Below cut-in / above cut-out: rotor parked (no aero loads)
        active = (V_wind >= self.cut_in) and (V_wind <= self.cut_out)

        if active:
            Q_aero = self.aero_torque(V_rel, Omega, beta_deg)
            F_thrust = self.aero_thrust(V_rel, Omega, beta_deg)
        else:
            Q_aero = 0.0
            F_thrust = 0.0

        Q_gen = self.generator_torque(Omega)

        # (1) Rotor angular acceleration
        Omega_dot = (Q_aero - Q_gen) / self.I_rotor

        # Wave excitation
        F_wave, M_wave = self.wave_forcing(t, H_wave, T_wave)

        # (2) Surge: (m + a) x_ddot = F_thrust - c x_dot - k x + F_wave
        x_ddot = (F_thrust - self.c_surge * x_dot - self.k_surge * x + F_wave) \
            / (self.m_plat + self.a_surge)

        # (3) Pitch: I theta_ddot = F_thrust*L - c theta_dot - k theta + M_wave
        theta_ddot = (F_thrust * self.L_hub - self.c_pitch * theta_dot
                      - self.k_pitch * theta + M_wave) / self.I_pitch

        return [Omega_dot, x_dot, x_ddot, theta_dot, theta_ddot]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, wind_speed, beta_deg=0.0, dt=0.05, duration_s=120.0,
                 H_wave=0.0, T_wave=10.0, Omega0=None, x0=0.0, theta0=0.0):
        """
        Integrate the coupled aero-hydro ODE system with solve_ivp (RK45).

        Args:
            wind_speed : float or callable(t)->m/s   hub-height free wind.
            beta_deg   : blade pitch angle [deg].
            dt         : output sample spacing [s].
            duration_s : simulation horizon [s].
            H_wave     : regular wave height [m] (0 = calm).
            T_wave     : wave period [s].
            Omega0     : initial rotor speed [rad/s] (default: steady estimate).
            x0, theta0 : initial surge [m] / pitch [rad].

        Returns dict of time-series arrays.
        """
        if callable(wind_speed):
            V_fn = wind_speed
            V_mean = float(np.mean([wind_speed(tt)
                                    for tt in np.linspace(0, duration_s, 11)]))
        else:
            V_const = float(wind_speed)
            V_fn = lambda t: V_const
            V_mean = V_const

        if Omega0 is None:
            # Steady estimate: track optimal TSR below rated, cap at rated.
            Omega0 = min(self.tsr_opt * max(V_mean, self.cut_in) / self.R,
                         self.Omega_rated)

        y0 = [Omega0, x0, 0.0, theta0, 0.0]
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45",
            args=(V_fn, beta_deg, H_wave, T_wave),
            rtol=1e-6, atol=1e-8, max_step=dt,
        )

        t = sol.t
        Omega = sol.y[0]
        x = sol.y[1]
        x_dot = sol.y[2]
        theta = sol.y[3]
        theta_dot = sol.y[4]

        # Post-process derived quantities along the trajectory
        V_wind = np.array([V_fn(tt) for tt in t])
        V_rel = np.maximum(V_wind - x_dot - self.L_hub * theta_dot, 0.0)
        active = (V_wind >= self.cut_in) & (V_wind <= self.cut_out)

        lam = self.tip_speed_ratio(Omega, V_rel)
        cp = np.where(active, self.power_coefficient(lam, beta_deg), 0.0)
        P_aero = np.where(active, self.aero_power(V_rel, Omega, beta_deg), 0.0)
        F_thrust = np.where(active, self.aero_thrust(V_rel, Omega, beta_deg), 0.0)
        Q_gen = self.generator_torque(Omega)
        # Electrical power = generator torque * speed * conversion efficiency, capped
        P_elec = np.minimum(Q_gen * Omega * self.eta_gen, self.P_rated)
        P_elec = np.where(active, P_elec, 0.0)

        return {
            "t": t,
            "rotor_speed": Omega,             # rad/s
            "tip_speed_ratio": lam,
            "cp": cp,
            "V_rel": V_rel,                   # m/s relative wind at hub
            "V_wind": V_wind,                 # m/s free wind
            "power_aero": P_aero,             # W
            "power_elec": P_elec,             # W
            "thrust": F_thrust,               # N
            "surge": x,                       # m
            "surge_vel": x_dot,               # m/s
            "pitch": theta,                   # rad
            "pitch_deg": np.degrees(theta),   # deg
            "pitch_vel": theta_dot,           # rad/s
        }
