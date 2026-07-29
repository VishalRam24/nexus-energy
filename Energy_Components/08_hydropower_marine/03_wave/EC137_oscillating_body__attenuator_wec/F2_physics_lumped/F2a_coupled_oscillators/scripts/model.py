"""
EC137 -- Oscillating Body / Attenuator WEC (Pelamis-type) -- F2a
Coupled Hinged-Raft Oscillators with Hydraulic PTO

Physics-lumped (0D-per-joint) first-principles model.

------------------------------------------------------------------------
Concept
------------------------------------------------------------------------
A floating attenuator (Pelamis) is a chain of `n_segments` long, slender,
semi-submerged cylinders connected by `n_joints` hinges, moored so the
chain lies *parallel* to the dominant wave direction. As a wave crest
travels down the chain, successive joints flex (relative pitch motion).
Each hinge carries a hydraulic ram (PTO) that resists the relative
angular velocity; the work done against that resistance is the captured
power (Yemm et al. 2012).

We model the relative pitch angle theta_i(t) at each joint i as a damped,
driven, *coupled* oscillator. The joints are coupled because adjacent
joints share a common middle segment (the inertia/stiffness of an
internal segment appears in two joints' equations -> a tri-diagonal mass
and stiffness matrix, exactly the hinged-raft structure of Newman 1979).

------------------------------------------------------------------------
Equation of motion (matrix form)
------------------------------------------------------------------------
    J_eff * theta_ddot + (B_rad + B_pto) * theta_dot + (K_h + K_pto) * theta
        = M_exc(t)

where (all per-joint, theta is the vector of n_joints relative angles):

  J_eff = I_seg + I_added        rotational inertia about hinge incl. added mass
                                 (tri-diagonal: shared internal segments couple
                                  neighbouring joints)
  B_rad                          hydrodynamic radiation damping (energy radiated
                                  away as waves), Falnes (2002)
  B_pto                          PTO damping (diagonal) -- the useful extractor
  K_h                            hydrostatic (buoyancy) restoring stiffness
  K_pto                          optional reactive PTO stiffness (latching/tuning)
  M_exc(t)                       wave excitation moment; for a regular wave of
                                  frequency omega the moment at joint i is phase-
                                  shifted by the wave travel time between joints
                                  (k * spacing) -> this phasing is what makes an
                                  attenuator directional and is the source of
                                  joint-to-joint power variation.

------------------------------------------------------------------------
Power & energy
------------------------------------------------------------------------
Instantaneous mechanical power absorbed by the PTO at joint i:
    P_pto,i(t) = B_pto * theta_dot_i(t)^2            (>= 0 always, resistive)
Total absorbed (mechanical):  P_abs = sum_i P_pto,i
Electrical:  P_elec = eta_hyd * eta_gen * P_abs

Mean over a wave period gives the time-averaged power. The capture width
    CW = P_abs_mean / J            [m]      (J = wave power per metre crest)
and capture width ratio CWR = CW / device_width. Falnes' theorem bounds
the absorbed power so CW cannot exceed the radiation-limited maximum; we
also clip CWR to a physical ceiling.

------------------------------------------------------------------------
References
------------------------------------------------------------------------
  Newman, J.N. (1979). "Absorption of wave energy by elongated bodies."
      Applied Ocean Research, 1(4), 189-196.   (hinged-raft / attenuator theory)
  Falnes, J. (2002). "Ocean Waves and Oscillating Systems." Cambridge UP.
      (radiation damping, excitation force, max absorbed-power theorem)
  Yemm, R., Pizer, D., Retzler, C., Henderson, R. (2012). "Pelamis:
      experience from concept to connection." Phil. Trans. R. Soc. A,
      370, 365-380.
  Henderson, R. (2006). "Design, simulation, and testing of a novel
      hydraulic power take-off." Applied Ocean Research, 28, 297-307.
"""

import numpy as np
from scipy.integrate import solve_ivp

_G = 9.81


class AttenuatorWEC_F2a:
    """Pelamis-type attenuator: coupled hinged-raft oscillators + hydraulic PTO."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_seg   = int(u["n_segments"]["value"])
        self.n_joint = int(u["n_joints"]["value"])
        self.L_seg   = u["L_segment"]["value"]
        self.D_seg   = u["D_segment"]["value"]
        self.draft   = u["draft_frac"]["value"]
        self.rho_w   = u["rho_water"]["value"]
        self.rho_b   = u["rho_body"]["value"]
        self.Ca      = u["added_mass_coeff"]["value"]
        self.Cb_rad  = u["rad_damp_coeff"]["value"]
        self.Cexc    = u["excite_coeff"]["value"]
        self.B_pto0  = u["B_pto"]["value"]
        self.K_pto0  = u["K_pto"]["value"]
        self.eta_hyd = u["eta_hyd"]["value"]
        self.eta_gen = u["eta_gen"]["value"]
        self.theta_max = u["theta_max"]["value"]

        # --- derived segment properties -------------------------------------
        r = self.D_seg / 2.0
        # Submerged cross-section approximated as draft_frac of the circle area.
        A_cross = np.pi * r * r
        self.vol_seg = A_cross * self.L_seg * self.draft       # displaced volume [m3]
        self.m_seg = self.rho_b * A_cross * self.L_seg          # structural mass [kg]
        # Pitch moment of inertia of a slender cylinder about its hinge end:
        #   I = (1/3) m L^2   (rod about one end). Added mass scales it up.
        self.I_seg = (1.0 / 3.0) * self.m_seg * self.L_seg ** 2
        # Hydrodynamic added inertia (water entrained), Newman (1979) O(1) coeff.
        m_disp = self.rho_w * self.vol_seg
        self.I_added = self.Ca * (1.0 / 3.0) * m_disp * self.L_seg ** 2

        self.device_width = self.D_seg                          # crest width [m]
        self.device_length = self.n_seg * self.L_seg

        # Build constant structural matrices (independent of sea state).
        self._build_matrices()

    # ----------------------------------------------------------------------
    # Structural matrices (tri-diagonal hinged-raft coupling)
    # ----------------------------------------------------------------------
    def _build_matrices(self):
        n = self.n_joint
        I_tot = self.I_seg + self.I_added       # per half-segment about a hinge
        # Each joint sees the inertia of the two half-segments it connects.
        # Adjacent joints share an internal segment -> off-diagonal coupling.
        J = np.zeros((n, n))
        for i in range(n):
            J[i, i] = 2.0 * I_tot
            if i + 1 < n:
                J[i, i + 1] = 0.5 * I_tot       # shared internal segment
                J[i + 1, i] = 0.5 * I_tot
        self.J_eff = J

        # Hydrostatic restoring stiffness per joint (buoyancy couple resisting
        # relative pitch). K_h ~ rho g * (waterplane second moment) per segment.
        r = self.D_seg / 2.0
        b_wp = 2.0 * r                          # waterline beam ~ diameter
        I_wp = (b_wp * self.L_seg ** 3) / 12.0  # waterplane 2nd moment of area
        Kh = self.rho_w * _G * I_wp             # restoring moment / rad [N.m/rad]
        self.K_h_scalar = Kh
        self.K_h = np.eye(n) * Kh

    # ----------------------------------------------------------------------
    # Sea-state-dependent hydrodynamic coefficients
    # ----------------------------------------------------------------------
    @staticmethod
    def dispersion_k(omega, depth=None):
        """Deep-water wave number k = omega^2 / g  [rad/m]."""
        return omega * omega / _G

    def radiation_damping(self, omega):
        """Linear radiation damping per joint [N.m.s/rad], Falnes (2002).

        Scaled from the characteristic impedance rho * I_added * omega.
        """
        return self.Cb_rad * (self.rho_w * self.vol_seg) * self.L_seg ** 2 * omega

    def excitation_moment_amp(self, H_s, omega):
        """Amplitude of wave excitation moment per joint [N.m].

        Tied to the radiation damping through the Haskind relation (Falnes
        2002, ch. 5): the absorbed-power theorem gives the maximum power a
        single oscillating mode can extract,

            P_max = |M_exc|^2 / (8 * B_rad)

        and the corresponding maximum capture width is the point-absorber
        bound  CW_max = (lambda / 2*pi) per mode.  Inverting,

            |M_exc| = sqrt( 8 * B_rad * J * CW_max )

        with J the incident wave power per metre and CW_max scaled by the
        O(1) shape factor `Cexc` (<1, attenuator pitch mode couples to only a
        fraction of the theoretical point-absorber limit).  This construction
        guarantees the radiation-limited capture width stays physical while
        still scaling the moment linearly with wave amplitude (M ~ H_s).
        """
        B_rad = self.radiation_damping(omega)
        J = self.wave_power_per_metre(H_s, 2.0 * np.pi / omega)
        wavelength = 2.0 * np.pi / self.dispersion_k(omega)
        cw_max = self.Cexc * wavelength / (2.0 * np.pi)      # per-joint bound [m]
        return np.sqrt(8.0 * B_rad * J * cw_max)

    # ----------------------------------------------------------------------
    # Wave resource
    # ----------------------------------------------------------------------
    def wave_power_per_metre(self, H_s, T_e):
        """Incident wave energy flux per metre of crest [W/m] (deep water)."""
        return (self.rho_w * _G ** 2 * H_s ** 2 * T_e) / (64.0 * np.pi)

    # ----------------------------------------------------------------------
    # Equation of motion RHS  (state y = [theta(n), theta_dot(n)])
    # ----------------------------------------------------------------------
    def _rhs(self, t, y, omega, M_amp, B_tot, K_tot, k_spacing, Jinv):
        n = self.n_joint
        theta = y[:n]
        omega_dot = y[n:]
        # Excitation moment at each joint, phase-shifted by wave travel time.
        # Joint i is a distance i*L_seg further along the chain -> phase k*x.
        phase = k_spacing * np.arange(n)
        M = M_amp * np.cos(omega * t - phase)
        # Soft end-stop: extra restoring beyond theta_max (quadratic penalty)
        over = np.clip(np.abs(theta) - self.theta_max, 0.0, None)
        M_stop = -np.sign(theta) * (50.0 * self.K_h_scalar) * over
        rhs = M + M_stop - B_tot @ omega_dot - K_tot @ theta
        theta_ddot = Jinv @ rhs
        return np.concatenate([omega_dot, theta_ddot])

    # ----------------------------------------------------------------------
    # Time-domain simulation
    # ----------------------------------------------------------------------
    def simulate(self, H_s, T_e, B_pto=None, K_pto=None, dt=0.1,
                 duration_s=120.0, settle_frac=0.4):
        """
        Simulate the attenuator under a regular wave of height H_s, period T_e.

        Parameters
        ----------
        H_s : significant / regular wave height [m]
        T_e : energy / wave period [s]
        B_pto : PTO damping per joint [N.m.s/rad]  (default from params)
        K_pto : PTO reactive stiffness per joint [N.m/rad] (default from params)
        dt : output time step [s]
        duration_s : total simulated time [s]
        settle_frac : fraction of the record (start) discarded as transient
                      before averaging power.

        Returns
        -------
        dict : t, theta (n_joint x N), theta_dot, power_pto (per joint),
               power_total_mech, power_total_elec, mean_power_elec_W,
               capture_width_m, capture_width_ratio, efficiency, energy_check
        """
        if B_pto is None:
            B_pto = self.B_pto0
        if K_pto is None:
            K_pto = self.K_pto0

        n = self.n_joint
        omega = 2.0 * np.pi / T_e
        k = self.dispersion_k(omega)

        M_amp = self.excitation_moment_amp(H_s, omega)
        B_rad = self.radiation_damping(omega)

        B_tot = np.eye(n) * (B_rad + B_pto)
        K_tot = self.K_h + np.eye(n) * K_pto
        Jinv = np.linalg.inv(self.J_eff)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]
        y0 = np.zeros(2 * n)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt, args=(omega, M_amp, B_tot, K_tot, k, Jinv),
        )

        t_out = sol.t
        theta = sol.y[:n]                 # (n, N)
        theta_dot = sol.y[n:]             # (n, N)

        # Instantaneous PTO power per joint (resistive, always >= 0).
        power_pto = B_pto * theta_dot ** 2          # (n, N)
        power_total_mech = power_pto.sum(axis=0)    # (N,)
        power_total_elec = self.eta_hyd * self.eta_gen * power_total_mech

        # Average over the settled portion of the record.
        i0 = int(settle_frac * len(t_out))
        i0 = min(i0, len(t_out) - 2)
        mean_mech = float(np.mean(power_total_mech[i0:]))
        mean_elec = float(np.mean(power_total_elec[i0:]))

        # Capture width & ratio.
        J_res = self.wave_power_per_metre(H_s, T_e)          # W/m
        cw = mean_mech / J_res if J_res > 0 else 0.0          # m (mechanical)
        cwr = cw / self.device_width if self.device_width > 0 else 0.0
        # Physical ceiling: capture width is bounded by the radiation limit
        # (sum of per-joint point-absorber bounds). Cap CWR relative to the
        # narrow device width -- attenuators legitimately exceed 1 here because
        # they are long relative to their width (Babarit 2015).
        cwr_capped = min(cwr, 8.0)

        # Energy balance check: over the settled window, energy radiated +
        # absorbed should be supplied by the excitation work (within solver tol).
        rad_power = B_rad * (theta_dot ** 2).sum(axis=0)
        exc_power = self._excitation_power(t_out, theta_dot, omega, M_amp, k)
        mean_exc = float(np.mean(exc_power[i0:]))
        mean_diss = float(np.mean((power_total_mech + rad_power)[i0:]))
        energy_residual = (mean_exc - mean_diss) / (abs(mean_exc) + 1e-9)

        return {
            "t": t_out,
            "theta": theta,
            "theta_dot": theta_dot,
            "power_pto": power_pto,
            "power_total_mech": power_total_mech,
            "power_total_elec": power_total_elec,
            "mean_power_mech_W": mean_mech,
            "mean_power_elec_W": mean_elec,
            "wave_power_per_m_W": J_res,
            "capture_width_m": cw,
            "capture_width_ratio": cwr_capped,
            "efficiency": self.eta_hyd * self.eta_gen,
            "B_pto": B_pto,
            "energy_residual": energy_residual,
        }

    def _excitation_power(self, t, theta_dot, omega, M_amp, k):
        """Instantaneous power input by the excitation moment [W]."""
        n = self.n_joint
        phase = k * np.arange(n)[:, None]
        M = M_amp * np.cos(omega * t[None, :] - phase)        # (n, N)
        return (M * theta_dot).sum(axis=0)

    # ----------------------------------------------------------------------
    # Optimal PTO damping
    # ----------------------------------------------------------------------
    def optimal_B_pto(self, H_s, T_e, dt=0.1, duration_s=120.0,
                      n_scan=15, B_lo=None, B_hi=None):
        """Scan B_pto to maximise mean electrical power.

        For a single linear oscillator the optimum is B_pto = sqrt(B_rad^2 +
        ((K - J w^2)/w)^2) (Falnes 2002 impedance match). We bracket that and
        refine by simulation, returning the best B_pto and the power curve.
        """
        omega = 2.0 * np.pi / T_e
        B_rad = self.radiation_damping(omega)
        # Analytic single-DOF optimum as a starting guess.
        J_eff_scalar = self.J_eff[0, 0]
        reactance = (self.K_h_scalar - J_eff_scalar * omega ** 2) / omega
        B_opt_analytic = np.sqrt(B_rad ** 2 + reactance ** 2)
        if B_lo is None:
            B_lo = max(0.05 * B_opt_analytic, 1.0e5)
        if B_hi is None:
            B_hi = 10.0 * B_opt_analytic
        B_grid = np.geomspace(B_lo, B_hi, n_scan)
        P = np.array([
            self.simulate(H_s, T_e, B_pto=B, dt=dt, duration_s=duration_s)["mean_power_elec_W"]
            for B in B_grid
        ])
        i_best = int(np.argmax(P))
        return {
            "B_opt": float(B_grid[i_best]),
            "B_opt_analytic": float(B_opt_analytic),
            "P_max_elec_W": float(P[i_best]),
            "B_grid": B_grid,
            "P_grid": P,
        }
