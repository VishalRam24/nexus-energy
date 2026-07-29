"""
EC135 -- Point Absorber WEC -- F2a Heaving-Buoy Linear Hydrodynamic Model

Physics-lumped (0-D) first-principles model of a single heaving point absorber.
The float is treated as a rigid body with one degree of freedom (vertical heave x).
Linear (Cummins / frequency-domain-consistent) hydrodynamics give a
mass-spring-damper equation of motion:

    (m + m_a) x_ddot + (B_rad + B_pto) x_dot + (C_hyd + C_pto) x = F_exc(t)

where
    m       structural + ballast mass                       [kg]
    m_a     heave added mass                                [kg]
    B_rad   radiation damping coefficient                   [N.s/m]
    B_pto   power-take-off (generator) damping              [N.s/m]
    C_hyd   hydrostatic restoring stiffness = rho*g*A_wp    [N/m]
    C_pto   PTO reactive (spring) stiffness                 [N/m]
    F_exc   wave excitation force (regular wave)            [N]

Regular (monochromatic) wave excitation:
    F_exc(t) = F0 * cos(omega t),   F0 = F_exc_coeff * a,   a = H/2
with omega = 2*pi / T the wave angular frequency.

The instantaneous PTO power extracted is  P_pto(t) = B_pto * x_dot(t)^2  (>= 0
for a resistive PTO), and the time-mean absorbed power over an integer number of
periods gives the absorbed power. Electrical power = eta_pto * P_pto.

Capture width  CW = P_absorbed / J   where  J = rho g^2 H^2 T / (32 pi) is the
incident power per metre of wave crest for a regular wave (Falnes eq. 6.x).

Theory landmarks reproduced by this model:
  * Resonance: undamped natural period T_n = 2*pi*sqrt((m+m_a)/C_hyd); absorbed
    power peaks when the excitation period equals T_n (Falnes ch. 6).
  * Optimal PTO damping (resistive control, no reactive term): the absorbed power
    is maximised at  B_pto* = sqrt(B_rad^2 + (omega(m+m_a) - C/omega)^2 / omega^2)
    i.e. |Z_i|, the magnitude of the intrinsic mechanical impedance
    (Falnes 2002, eq. 6.46; Babarit 2015).
  * Theoretical capture-width limit for an axisymmetric heaving body:
    CW_max = wavelength / (2 pi) = g T^2 / (4 pi^2)   (Budal-Falnes / point-absorber
    bound, Falnes 2002 sec. 6.3).

References
----------
    Falnes, J. (2002). Ocean Waves and Oscillating Systems. Cambridge Univ. Press.
        (Eq. 5.x added mass/radiation; Ch. 6 absorbed power & optimal damping;
         Sec. 6.3 maximum capture width.)
    Babarit, A. et al. (2012). Renew. Energy, 41, 44-63.
    Babarit, A. (2015). A database of capture width ratio of WECs.
        Renew. Energy, 80, 610-628.
    Pecher, A. & Kofoed, J.P. (2017). Handbook of Ocean Wave Energy. Springer.
"""

import numpy as np
from scipy.integrate import solve_ivp

_G = 9.81


class PointAbsorberF2a:
    """One-DOF heaving point-absorber WEC (linear hydrodynamic mass-spring-damper)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.m        = u["mass_kg"]["value"]
        self.m_a      = u["added_mass_kg"]["value"]
        self.B_rad    = u["B_rad_Ns_per_m"]["value"]
        self.rho      = u["rho_water"]["value"]
        self.A_wp     = u["A_wp_m2"]["value"]
        self.F_coeff  = u["F_exc_coeff_N_per_m"]["value"]
        self.B_pto0   = u["B_pto_Ns_per_m"]["value"]
        self.C_pto0   = u["C_pto_N_per_m"]["value"]
        self.eta_pto  = u["eta_pto"]["value"]
        self.x_max    = u["x_max_m"]["value"]

        # Hydrostatic restoring stiffness (buoyancy): C = rho * g * A_wp
        self.C_hyd = self.rho * _G * self.A_wp

    # ------------------------------------------------------------------ helpers
    def natural_period(self):
        """Undamped heave natural period T_n = 2 pi sqrt((m+m_a)/C_hyd) [s]."""
        return 2.0 * np.pi * np.sqrt((self.m + self.m_a) / self.C_hyd)

    def natural_frequency(self):
        """Undamped heave natural angular frequency [rad/s]."""
        return np.sqrt(self.C_hyd / (self.m + self.m_a))

    def intrinsic_impedance_mag(self, omega):
        """
        Magnitude of intrinsic mechanical impedance |Z_i| [N.s/m].
            Z_i(omega) = B_rad + i*( omega*(m+m_a) - C/omega )
        The optimal resistive PTO damping equals |Z_i| (Falnes eq. 6.46).
        """
        reactance = omega * (self.m + self.m_a) - self.C_hyd / omega
        return np.sqrt(self.B_rad**2 + reactance**2)

    def optimal_B_pto(self, T):
        """Optimal resistive PTO damping for energy period T [N.s/m]."""
        omega = 2.0 * np.pi / T
        return self.intrinsic_impedance_mag(omega)

    def incident_power_per_metre(self, H, T):
        """
        Incident wave power per metre of crest for a *regular* wave [W/m]:
            J = rho g^2 H^2 T / (32 pi)
        (For a regular wave H is the wave height; cf. Falnes eq. 6.4.)
        """
        return self.rho * _G**2 * H**2 * T / (32.0 * np.pi)

    def max_capture_width(self, T):
        """
        Point-absorber theoretical capture-width upper bound for heave [m]:
            CW_max = lambda / (2 pi) = g T^2 / (4 pi^2)
        (Budal-Falnes bound, Falnes 2002 sec. 6.3.)
        """
        return _G * T**2 / (4.0 * np.pi**2)

    def excitation_force(self, t, H, T):
        """Regular-wave excitation force F_exc(t) = F0 cos(omega t) [N]."""
        a = 0.5 * H                     # wave amplitude
        F0 = self.F_coeff * a
        omega = 2.0 * np.pi / T
        return F0 * np.cos(omega * t)

    # ------------------------------------------------------------- ODE / solver
    def _rhs(self, t, y, H, T, B_pto, C_pto):
        """State y = [x, x_dot]; returns [x_dot, x_ddot]."""
        x, xdot = y
        M = self.m + self.m_a
        B = self.B_rad + B_pto
        C = self.C_hyd + C_pto
        F = self.excitation_force(t, H, T)
        xddot = (F - B * xdot - C * x) / M
        return [xdot, xddot]

    def simulate(self, H=1.0, T=10.0, B_pto=None, C_pto=None,
                 dt=0.05, duration_s=None, x0=0.0, v0=0.0):
        """
        Time-domain simulation of the heaving point absorber under a regular wave.

        Parameters
        ----------
        H : regular wave height [m]
        T : wave period [s]
        B_pto : PTO damping [N.s/m] (default from parameters)
        C_pto : PTO reactive stiffness [N/m] (default from parameters)
        dt : output time step [s]
        duration_s : total time [s]; default = 60 wave periods (long enough to
                     reach steady state and average over an integer # of periods)

        Returns
        -------
        dict with time series and scalar performance metrics. Mean powers are
        averaged over the last integer number of wave periods (steady state).
        """
        B_pto = self.B_pto0 if B_pto is None else B_pto
        C_pto = self.C_pto0 if C_pto is None else C_pto
        if duration_s is None:
            duration_s = 60.0 * T

        n_steps = int(round(duration_s / dt))
        t_eval = np.linspace(0.0, duration_s, n_steps + 1)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [x0, v0],
            t_eval=t_eval, args=(H, T, B_pto, C_pto),
            method="RK45", rtol=1e-8, atol=1e-9, max_step=dt,
        )
        t = sol.t
        x = sol.y[0]
        xdot = sol.y[1]

        F_exc = self.excitation_force(t, H, T)

        # Instantaneous powers
        P_pto_inst = B_pto * xdot**2                    # mechanical absorbed by PTO [W]
        P_exc_inst = F_exc * xdot                       # power input by wave to body [W]
        P_rad_inst = self.B_rad * xdot**2               # radiated (re-emitted) [W]

        # Average over the last integer number of periods (steady state window)
        n_avg_periods = max(1, int((duration_s / T) // 2))   # last ~half the run
        t_start = duration_s - n_avg_periods * T
        mask = t >= t_start - 1e-9
        # ensure window covers integer periods of samples
        Pm_pto = np.trapz(P_pto_inst[mask], t[mask]) / (t[mask][-1] - t[mask][0])
        Pm_exc = np.trapz(P_exc_inst[mask], t[mask]) / (t[mask][-1] - t[mask][0])
        Pm_rad = np.trapz(P_rad_inst[mask], t[mask]) / (t[mask][-1] - t[mask][0])

        P_elec_mean = self.eta_pto * Pm_pto

        J = self.incident_power_per_metre(H, T)         # W/m
        capture_width = Pm_pto / J if J > 0 else 0.0    # m
        cw_max = self.max_capture_width(T)
        cwr = capture_width / (2.0 * self.radius()) if self.radius() > 0 else 0.0

        return {
            "t": t,
            "x": x,
            "x_dot": xdot,
            "F_exc": F_exc,
            "P_pto_inst": P_pto_inst,
            "P_exc_inst": P_exc_inst,
            "P_pto_mean": Pm_pto,
            "P_exc_mean": Pm_exc,
            "P_rad_mean": Pm_rad,
            "P_elec_mean": P_elec_mean,
            "amplitude": float(np.max(np.abs(x[mask]))),
            "incident_power_per_m": J,
            "capture_width": capture_width,
            "capture_width_max": cw_max,
            "capture_width_ratio": cwr,
            "B_pto": B_pto,
            "C_pto": C_pto,
            "T_natural": self.natural_period(),
        }

    def radius(self):
        """Float radius from water-plane area (A_wp = pi r^2)."""
        return np.sqrt(self.A_wp / np.pi)

    # ----------------------------------------------------- frequency-domain check
    def steady_amplitude(self, H, T, B_pto=None, C_pto=None):
        """
        Closed-form steady-state heave amplitude for the regular-wave forced
        oscillator (frequency-domain solution), used to verify the ODE result:
            |X| = F0 / sqrt( (C - M w^2)^2 + (B w)^2 )
        """
        B_pto = self.B_pto0 if B_pto is None else B_pto
        C_pto = self.C_pto0 if C_pto is None else C_pto
        omega = 2.0 * np.pi / T
        M = self.m + self.m_a
        B = self.B_rad + B_pto
        C = self.C_hyd + C_pto
        F0 = self.F_coeff * 0.5 * H
        denom = np.sqrt((C - M * omega**2) ** 2 + (B * omega) ** 2)
        return F0 / denom

    def mean_power_analytic(self, H, T, B_pto=None, C_pto=None):
        """
        Closed-form mean PTO power for the forced oscillator (Falnes ch. 6):
            P_mean = 0.5 * B_pto * omega^2 * |X|^2
        """
        B_pto = self.B_pto0 if B_pto is None else B_pto
        omega = 2.0 * np.pi / T
        X = self.steady_amplitude(H, T, B_pto, C_pto)
        return 0.5 * B_pto * omega**2 * X**2
