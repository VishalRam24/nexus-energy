"""
EC134 -- Oscillating Water Column (OWC) -- F2a Physics-Lumped Dynamics

Physics-lumped (0D) model of a fixed/shoreline OWC wave energy converter.
The internal water column is treated as a rigid piston (mass-spring-damper)
driven by the wave excitation force; the trapped air chamber acts as a
pneumatic spring whose pressure drives a self-rectifying Wells air turbine
that extracts power. Two coupled ODEs are integrated with scipy.solve_ivp.

State variables (lumped piston + chamber):
    x      = water-column surface displacement [m] (positive up)
    xdot   = water-column surface velocity [m/s]
    p      = chamber gauge pressure [Pa]

(1) Water-column momentum (Newton's 2nd law for the oscillating piston):
        (m + m_a) * xddot = F_exc(t)                 wave excitation force
                             - rho*g*S * x            hydrostatic restoring (spring)
                             - B_rad * xdot           radiation damping (Evans 1978)
                             - p * S                  pneumatic back-force from chamber
    where m   = rho * S * L      (entrained water mass, L = effective column length)
          m_a = added (radiation) mass ~ rho * S * (8/3pi) sqrt(S/pi)  (Evans 1978)
          S   = water-column plan area.

(2) Air-chamber pneumatics (linearised isentropic / compressibility, Sarmento
    & Falcao 1985): the chamber pressure responds to the difference between the
    volume displaced by the moving water column (S*xdot) and the volume flow
    swallowed by the turbine (q_turb):
        dp/dt = (gamma * p_atm / V0) * (S * xdot - q_turb / rho_air)
    For a Wells turbine the volume flow is proportional to chamber pressure
    (linear pressure-flow characteristic, Falcao & Henriques 2016):
        m_dot_turb = C_turb * p      ->  q_turb = m_dot_turb (mass flow)

Wave excitation force (regular-wave Haskind/Evans relation):
        F_exc(t) = rho * g * b * H/2 * G(omega) * cos(omega t)
    G(omega) is an excitation-force transfer gain; the excitation damping
    equals the radiation damping by the Haskind relation (Falnes 2002).

Power flows (energy conservation):
        P_exc(t)   = F_exc(t) * xdot                 power input by waves
        P_pneu(t)  = p * S * xdot                    pneumatic power into air
        P_turb(t)  = eta_wells(phi) * P_pneu_avail   shaft power
        P_elec(t)  = eta_generator * P_turb

Wells turbine aerodynamic efficiency (Raghunathan 1995; Falcao & Henriques
2016): efficiency rises with flow coefficient phi then collapses past the
stall point phi_stall:
        eta_wells(phi) = eta_peak * (phi/phi_stall) * exp(1 - phi/phi_stall)
    (a single-peaked curve, zero at phi=0, peak at phi=phi_stall).

Capture-width ratio (CWR) is computed a posteriori from the mean absorbed
pneumatic power and the incident wave power per metre (always < 1).

References:
    Evans, D.V. (1978). "The oscillating water column wave-energy device."
        J. Inst. Maths Applics / J. Fluid Mech. 77, 1-25.
    Sarmento, A.J.N.A. & Falcao, A.F. de O. (1985). "Wave generation by an
        oscillating surface-pressure ...", J. Fluid Mech. 150, 467-485.
    Falcao, A.F. de O. & Henriques, J.C.C. (2016). "Oscillating-water-column
        wave energy converters and air turbines: A review."
        Renewable Energy 85, 1391-1424.
    Falnes, J. (2002). Ocean Waves and Oscillating Systems. Cambridge UP.
"""

import numpy as np
from scipy.integrate import solve_ivp

_G = 9.81  # m/s^2


class OWC_F2a:
    """Oscillating Water Column -- physics-lumped piston + pneumatic ODE model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S = u["S_wc"]["value"]                 # m2 water-column plan area
        self.b = u["b_owc"]["value"]                # m frontage width
        self.L = u["depth_lip"]["value"]            # m effective column length
        self.rho_w = u["rho_water"]["value"]        # kg/m3
        self.rho_a = u["rho_air"]["value"]          # kg/m3
        self.gamma = u["gamma_air"]["value"]
        self.p_atm = u["p_atm"]["value"]            # Pa
        self.V0 = u["V_chamber"]["value"]           # m3
        self.B_rad = u["B_rad"]["value"]            # N.s/m
        self.C_turb = u["C_turb"]["value"]          # kg/(s.Pa)
        self.K_wells = u["K_wells"]["value"]
        self.phi_stall = u["phi_stall"]["value"]
        self.eta_peak = u["eta_turb_peak"]["value"]
        self.eta_gen = u["eta_generator"]["value"]

        # Lumped masses (Evans 1978)
        self.m_water = self.rho_w * self.S * self.L
        # added (radiation) mass for a piston of plan area S over deep water
        self.m_added = self.rho_w * self.S * (8.0 / (3.0 * np.pi)) * np.sqrt(self.S / np.pi)
        self.M = self.m_water + self.m_added

        # hydrostatic stiffness (restoring spring): k = rho * g * S
        self.k = self.rho_w * _G * self.S

        # reference velocity scale for the Wells flow coefficient
        self._u_ref = max(np.sqrt(_G * self.L), 1.0)

    # ------------------------------------------------------------------
    # Wave kinematics / excitation
    # ------------------------------------------------------------------
    def omega_from_Te(self, T_e):
        """Angular frequency from energy period (omega = 2 pi / T)."""
        return 2.0 * np.pi / T_e

    def natural_frequency(self):
        """Undamped natural angular frequency of the water column [rad/s]."""
        return np.sqrt(self.k / self.M)

    def excitation_gain(self, omega):
        """
        Excitation-force transfer gain G(omega) [-].

        By the Haskind relation the excitation damping equals the radiation
        damping; for a piston-mode OWC a smooth roll-off with frequency is a
        standard lumped approximation (Falnes 2002, Evans 1978). Bounded in
        (0, 1], near unity for long waves, decaying for short waves.
        """
        omega = np.asarray(omega, dtype=float)
        omega_c = np.sqrt(_G / self.L)  # characteristic cut-off
        return 1.0 / (1.0 + (omega / omega_c) ** 2)

    def excitation_force_amp(self, H, omega):
        """
        Regular-wave excitation force amplitude [N] for wave height H.

        F0 = rho * g * b * (H/2) * G(omega)
        """
        return self.rho_w * _G * self.b * (H / 2.0) * self.excitation_gain(omega)

    # ------------------------------------------------------------------
    # Wells turbine
    # ------------------------------------------------------------------
    def flow_coefficient(self, xdot):
        """Non-dimensional flow coefficient phi = |U_water| / u_ref."""
        return np.abs(xdot) / self._u_ref

    def wells_efficiency(self, phi):
        """
        Wells turbine aerodynamic efficiency vs flow coefficient.

        Single-peaked curve (Raghunathan 1995; Falcao & Henriques 2016):
            eta = eta_peak * (phi/phi_stall) * exp(1 - phi/phi_stall)
        Zero at phi=0, peaks at phi=phi_stall, decays past stall.
        """
        phi = np.asarray(phi, dtype=float)
        r = phi / self.phi_stall
        eta = self.eta_peak * r * np.exp(1.0 - r)
        return np.clip(eta, 0.0, self.eta_peak)

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------
    def _rhs(self, t, y, F0, omega):
        x, xdot, p = y
        F_exc = F0 * np.cos(omega * t)
        # piston momentum balance
        xddot = (F_exc - self.k * x - self.B_rad * xdot - p * self.S) / self.M
        # chamber pneumatics: linearised isentropic compressibility
        m_dot_turb = self.C_turb * p            # mass flow through turbine (linear)
        q_disp = self.S * xdot                  # volume displacement rate by column
        q_turb = m_dot_turb / self.rho_a        # volume flow swallowed by turbine
        dp = (self.gamma * self.p_atm / self.V0) * (q_disp - q_turb)
        return [xdot, xddot, dp]

    def simulate(self, H_s, T_e, dt=0.05, duration_s=120.0):
        """
        Integrate the coupled OWC dynamics for a regular wave (H_s, T_e).

        Returns a dict of time-series and scalar mean-power metrics.
        """
        H = float(H_s)
        T = float(T_e)
        omega = self.omega_from_Te(T)
        F0 = self.excitation_force_amp(H, omega)

        n_steps = int(round(duration_s / dt)) + 1
        t_eval = np.linspace(0.0, duration_s, n_steps)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [0.0, 0.0, 0.0],
            t_eval=t_eval, args=(F0, omega),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )
        t = sol.t
        x = sol.y[0]
        xdot = sol.y[1]
        p = sol.y[2]

        F_exc = F0 * np.cos(omega * t)

        # power flows
        P_exc = F_exc * xdot                      # W, power delivered by waves
        P_pneu = p * self.S * xdot                # W, pneumatic power into air (chamber)
        # turbine extracts the pneumatic power flowing through it = p * q_turb
        m_dot_turb = self.C_turb * p
        q_turb = m_dot_turb / self.rho_a
        P_avail = p * q_turb                      # W, pneumatic power dissipated at turbine (>=0)
        phi = self.flow_coefficient(xdot)
        eta_w = self.wells_efficiency(phi)
        P_turb = eta_w * P_avail                  # W, shaft power
        P_elec = self.eta_gen * P_turb            # W, electrical power

        # discard first half as start-up transient for mean metrics; a lightly
        # damped resonant water column needs many cycles to reach a stationary
        # oscillation, so the second half gives a well-converged cycle-mean.
        i0 = len(t) // 2
        mean_P_exc = float(np.mean(P_exc[i0:]))
        mean_P_avail = float(np.mean(P_avail[i0:]))   # mean absorbed pneumatic power
        mean_P_turb = float(np.mean(P_turb[i0:]))
        mean_P_elec = float(np.mean(P_elec[i0:]))
        # radiation-damping dissipation (for energy-balance accounting)
        mean_P_rad = float(self.B_rad * np.mean(xdot[i0:] ** 2))

        # incident wave power per metre and CWR
        J = (self.rho_w * _G ** 2 * H ** 2 * T) / (64.0 * np.pi)  # W/m (Falnes 2002)
        P_incident = J * self.b                                   # W on frontage
        cwr = mean_P_avail / P_incident if P_incident > 0 else 0.0
        capture_eff = mean_P_elec / P_incident if P_incident > 0 else 0.0

        return {
            "t": t,
            "x": x,
            "xdot": xdot,
            "pressure": p,
            "P_exc": P_exc,
            "P_pneu": P_pneu,
            "P_avail": P_avail,
            "P_turb": P_turb,
            "P_elec": P_elec,
            "eta_wells": eta_w,
            "phi": phi,
            "mean_P_exc_W": mean_P_exc,
            "mean_P_avail_W": mean_P_avail,
            "mean_P_rad_W": mean_P_rad,
            "mean_P_turb_W": mean_P_turb,
            "mean_P_elec_W": mean_P_elec,
            "mean_P_elec_kW": mean_P_elec / 1e3,
            "P_incident_W": P_incident,
            "wave_power_per_m_W": J,
            "capture_width_ratio": cwr,
            "capture_efficiency": capture_eff,
            "omega": omega,
            "omega_n": self.natural_frequency(),
        }

    # ------------------------------------------------------------------
    # Spectrum-averaged mean power (irregular sea)
    # ------------------------------------------------------------------
    def mean_power_spectrum(self, H_s, T_e, n_freq=20, duration_s=80.0, dt=0.1):
        """
        Mean electrical power for an irregular sea characterised by (H_s, T_e)
        using a Pierson-Moskowitz spectrum discretised into n_freq regular
        components and superposed by linear (energy-weighted) averaging of the
        per-component capture. Returns mean electrical power [W] and CWR.

        S_PM(omega) = (alpha) * omega^-5 * exp(-beta (omega_p/omega)^4)
        with peak frequency omega_p ~ 0.857 * 2pi/T_e (energy->peak conversion).
        """
        Hs = float(H_s)
        Te = float(T_e)
        omega_p = 0.857 * (2.0 * np.pi / Te)
        # frequency grid spanning the spectrum
        omegas = np.linspace(0.3 * omega_p, 3.0 * omega_p, n_freq)
        # Pierson-Moskowitz shape (un-normalised, then scaled to variance Hs^2/16)
        S = (omegas ** -5) * np.exp(-1.25 * (omega_p / omegas) ** 4)
        domega = omegas[1] - omegas[0]
        var_target = (Hs ** 2) / 16.0
        S *= var_target / (np.sum(S) * domega)

        # per-component wave amplitude a_i = sqrt(2 S domega) -> H_i = 2 a_i
        P_elec_acc = 0.0
        P_avail_acc = 0.0
        for omg, Sw in zip(omegas, S):
            a_i = np.sqrt(2.0 * Sw * domega)
            H_i = 2.0 * a_i
            T_i = 2.0 * np.pi / omg
            if H_i < 1e-4:
                continue
            r = self.simulate(H_i, T_i, dt=dt, duration_s=duration_s)
            P_elec_acc += r["mean_P_elec_W"]
            P_avail_acc += r["mean_P_avail_W"]

        J = (self.rho_w * _G ** 2 * Hs ** 2 * Te) / (64.0 * np.pi)
        P_incident = J * self.b
        cwr = P_avail_acc / P_incident if P_incident > 0 else 0.0
        return {
            "mean_P_elec_W": P_elec_acc,
            "mean_P_elec_kW": P_elec_acc / 1e3,
            "mean_P_avail_W": P_avail_acc,
            "P_incident_W": P_incident,
            "capture_width_ratio": cwr,
        }
