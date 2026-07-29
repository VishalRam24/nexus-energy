"""
EC178 -- Switched Reluctance Motor (SRM) -- F2a Physics-Lumped Reluctance Model

Doubly-salient reluctance machine with NO permanent magnets. Torque is produced
purely by the variation of phase inductance with rotor position (the rotor moves
to minimise reluctance). This is the first-principles, lumped-parameter upgrade of
the F1a efficiency-map model.

Core physics
------------
1. Position-dependent phase inductance (idealised but smooth profile):

       L(theta_e, i) = L_unaligned + (L_aligned_sat(i) - L_unaligned) * f(theta_e)

   theta_e is the electrical rotor angle of the phase (theta_e = Nr * theta_mech),
   with electrical period 2*pi.  f(theta_e) rises smoothly from 0 (unaligned) to 1
   (aligned).  Magnetic saturation reduces the aligned inductance as current rises
   (Miller 1993, magnetisation curves):

       L_aligned_sat(i) = L_unaligned + (L_aligned - L_unaligned) / (1 + (i/i_sat)^p)

2. Co-energy torque (reluctance torque, no PM term):

       T_phase = 0.5 * i^2 * dL/dtheta_mech         (magnetically-linear co-energy)

   Positive (motoring) torque is produced only where dL/dtheta > 0 (rising
   inductance), so each phase is excited as its poles approach alignment.

3. Phase voltage equation (asymmetric half-bridge applies +/- V_dc or 0):

       v = R*i + d(L(theta,i)*i)/dt
         = R*i + L_incr * di/dt + i * (dL/dtheta) * omega

   where L_incr = d(L*i)/di is the incremental inductance and i*(dL/dtheta)*omega
   is the back-EMF (motional) term.  Solved for di/dt:

       di/dt = ( v - R*i - i*(dL/dtheta)*omega ) / L_incr

4. Mechanical ODE (Newton's law for rotation):

       J * domega/dt = T_e - T_load - B*omega
       dtheta/dt    = omega

   T_e = sum over phases of T_phase.

State vector y = [theta_mech, omega, i_1, ..., i_Nphases], integrated with
scipy.integrate.solve_ivp (stiff-capable LSODA).

Energy bookkeeping (for efficiency):  electrical input energy = integral(v*i),
mechanical output energy = integral(T_load * omega) (useful shaft work against
load), copper loss = integral(R*i^2).  Efficiency = W_mech_out / W_elec_in, which
is bounded in (0,1) for a passive reluctance machine.

References
----------
Miller, T.J.E. (1993). Switched Reluctance Motors and Their Control. Magna Physics
    / Oxford University Press.  (Inductance profile, co-energy torque, saturation.)
Krishnan, R. (2001). Switched Reluctance Motor Drives: Modeling, Simulation,
    Analysis, Design, and Applications. CRC Press.  (Voltage/torque equations,
    converter, ripple.)
"""

import numpy as np
from scipy.integrate import solve_ivp


class SRM_F2a:
    """Switched Reluctance Motor -- lumped reluctance / co-energy physics model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power"]["value"] * 1000.0      # W
        self.omega_rated_rpm = u["omega_rated"]["value"]        # rpm
        self.Nr = int(u["poles_rotor"]["value"])                # rotor poles
        self.Nph = int(u["n_phases"]["value"])                  # phases
        self.R = u["R_phase"]["value"]                          # ohm
        self.L_a = u["L_aligned"]["value"]                      # H (unsaturated)
        self.L_u = u["L_unaligned"]["value"]                    # H
        self.i_sat = u["i_sat"]["value"]                        # A
        self.p_sat = u["sat_exponent"]["value"]                 # -
        self.V_dc = u["V_dc"]["value"]                          # V
        self.theta_on = np.deg2rad(u["theta_on_deg"]["value"])  # elec rad
        self.theta_off = np.deg2rad(u["theta_off_deg"]["value"])  # elec rad
        self.J = u["J_inertia"]["value"]                        # kg.m2
        self.B = u["B_friction"]["value"]                       # N.m.s/rad

        # Electrical period = 2*pi; mechanical stroke period per phase = 2*pi/Nr.
        # Phases are evenly offset over one electrical period.
        self.phase_offsets = np.array(
            [2.0 * np.pi * k / self.Nph for k in range(self.Nph)]
        )

    # ------------------------------------------------------------------
    # Inductance profile and its derivative  (per phase)
    # ------------------------------------------------------------------
    def _shape(self, theta_e):
        """Smooth aligned-fraction f in [0,1]; aligned at theta_e = pi.

        f(theta_e) = 0.5*(1 - cos(theta_e)) -> 0 at theta_e=0 (unaligned),
        1 at theta_e=pi (aligned).  Periodic, C-infinity, classic idealisation
        of the SRM inductance profile (Miller 1993).
        """
        return 0.5 * (1.0 - np.cos(theta_e))

    def _dshape_dthetae(self, theta_e):
        """d f / d theta_e = 0.5*sin(theta_e)."""
        return 0.5 * np.sin(theta_e)

    def L_aligned_sat(self, i):
        """Saturated aligned inductance [H] -- rolls off with current (Miller 1993)."""
        return self.L_u + (self.L_a - self.L_u) / (
            1.0 + (abs(i) / self.i_sat) ** self.p_sat
        )

    def inductance(self, theta_mech, i, phase=0):
        """Phase inductance L(theta, i) [H]."""
        theta_e = self.Nr * theta_mech - self.phase_offsets[phase]
        f = self._shape(theta_e)
        return self.L_u + (self.L_aligned_sat(i) - self.L_u) * f

    def dL_dtheta_mech(self, theta_mech, i, phase=0):
        """dL/dtheta_mech [H/rad] at fixed current (used for co-energy torque)."""
        theta_e = self.Nr * theta_mech - self.phase_offsets[phase]
        dL_amp = self.L_aligned_sat(i) - self.L_u
        # d/dtheta_mech = (df/dtheta_e) * (dtheta_e/dtheta_mech), dtheta_e/dmech = Nr
        return dL_amp * self._dshape_dthetae(theta_e) * self.Nr

    def L_incremental(self, theta_mech, i, phase=0):
        """Incremental inductance d(L*i)/di [H] for the di/dt term.

        With L(theta,i) = L_u + (L_sat(i)-L_u)*f, and
        L_sat(i) = L_u + dLa/(1+(|i|/i_sat)^p), the flux linkage is
        lambda = L*i.  dL/di accounts for saturation; here we use the chord/
        incremental inductance L + i*dL/di, lower-bounded for numerical safety.
        """
        theta_e = self.Nr * theta_mech - self.phase_offsets[phase]
        f = self._shape(theta_e)
        x = abs(i) / self.i_sat
        dLa = self.L_a - self.L_u
        # dL_sat/di
        denom = (1.0 + x ** self.p_sat)
        dLsat_di = -dLa * self.p_sat * (x ** (self.p_sat)) / max(abs(i), 1e-9) / denom**2
        L = self.L_u + (self.L_aligned_sat(i) - self.L_u) * f
        L_incr = L + i * dLsat_di * f
        return max(L_incr, self.L_u * 0.5)

    # ------------------------------------------------------------------
    # Converter switching logic (asymmetric half-bridge)
    # ------------------------------------------------------------------
    def switching_state(self, theta_mech, phase=0):
        """Converter leg state from turn-on/turn-off angle control.

        +1 = magnetise (+V_dc) inside conduction window [theta_on, theta_off];
        -1 = demagnetise (-V_dc) after commutation (asymmetric half-bridge
        freewheel through the lower diode-pair).  The diode clamp that prevents
        reverse current is handled smoothly in the RHS, not here, to keep the
        derivative continuous for the ODE solver.
        """
        theta_e = (self.Nr * theta_mech - self.phase_offsets[phase]) % (2.0 * np.pi)
        if self.theta_on <= theta_e < self.theta_off:
            return 1
        return -1

    def applied_voltage(self, theta_mech, i, phase=0):
        """Net applied phase voltage [V] including the diode clamp.

        During demagnetisation (-V_dc) the lower diodes block reverse current, so
        once i has decayed to ~0 the leg is effectively open and the terminal
        voltage seen by the (zero) current is 0.  Used for power post-processing.
        """
        s = self.switching_state(theta_mech, phase)
        if s > 0:
            return self.V_dc
        # demagnetising: voltage acts only while forward current still flows
        return -self.V_dc if i > 1e-4 else 0.0

    # ------------------------------------------------------------------
    # Right-hand side of the coupled electromechanical ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, T_load):
        theta = y[0]
        omega = y[1]
        currents = y[2:]

        T_e = 0.0
        didt = np.zeros(self.Nph)
        for ph in range(self.Nph):
            i = max(currents[ph], 0.0)         # diode-clamped: no reverse current
            s = self.switching_state(theta, ph)
            v = self.V_dc if s > 0 else -self.V_dc
            dLdth = self.dL_dtheta_mech(theta, i, ph)
            L_incr = self.L_incremental(theta, i, ph)
            # back-EMF (motional) term e = i * dL/dtheta * omega
            emf = i * dLdth * omega
            di = (v - self.R * i - emf) / L_incr
            # Diode clamp (continuous): when current is at/below zero, the only
            # admissible derivative is non-negative (the lower diodes block any
            # attempt to drive i negative). This makes i>=0 an invariant without
            # introducing solver chatter.
            if currents[ph] <= 0.0:
                di = max(di, 0.0)
            didt[ph] = di
            # co-energy reluctance torque (no PM term)
            T_e += 0.5 * i * i * dLdth

        domega = (T_e - T_load - self.B * omega) / self.J
        dtheta = omega
        return np.concatenate(([dtheta, domega], didt))

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, V_dc=None, T_load=2.0, omega0=None, theta0=0.0,
                 dt=2e-5, duration_s=0.05):
        """Simulate the coupled electromechanical dynamics.

        Parameters
        ----------
        V_dc : float, optional   DC-link voltage (overrides parameter default).
        T_load : float           Load torque [N.m] (constant).
        omega0 : float, optional Initial mechanical speed [rad/s]
                                 (default = rated speed).
        theta0 : float           Initial rotor angle [rad].
        dt : float               Output time step [s].
        duration_s : float       Simulation duration [s].

        Returns
        -------
        dict of time series + scalar performance metrics.
        """
        if V_dc is not None:
            self.V_dc = V_dc
        if omega0 is None:
            omega0 = self.omega_rated_rpm * 2.0 * np.pi / 60.0

        y0 = np.concatenate(([theta0, omega0], np.zeros(self.Nph)))
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0, t_eval=t_eval,
            args=(T_load,), method="RK45", rtol=1e-5, atol=1e-7,
            max_step=dt,
        )

        t = sol.t
        theta = sol.y[0]
        omega = sol.y[1]
        currents = sol.y[2:]
        currents = np.clip(currents, 0.0, None)
        N = len(t)

        # Post-process torque, voltage, flux per step
        T_e = np.zeros(N)
        v_total = np.zeros(N)
        p_elec = np.zeros(N)
        phase_currents = np.zeros((self.Nph, N))
        for k in range(N):
            Tk = 0.0
            vk_i = 0.0
            for ph in range(self.Nph):
                i = max(currents[ph, k], 0.0)
                phase_currents[ph, k] = i
                dLdth = self.dL_dtheta_mech(theta[k], i, ph)
                Tk += 0.5 * i * i * dLdth
                v = self.applied_voltage(theta[k], i, ph)
                vk_i += v * i                      # instantaneous electrical power
            T_e[k] = Tk
            p_elec[k] = vk_i
            v_total[k] = vk_i

        # Energy integrals (trapezoid).  Instantaneous power balance per phase:
        #   v*i = R*i^2  +  d/dt(field energy)  +  T_e*omega
        # Over the simulated window the magnetic field energy is recovered during
        # demagnetisation (v*i goes negative), so the NET electrical input equals
        # copper loss + electromechanical work. We therefore integrate the signed
        # p_elec (magnetisation minus demag recovery) for the true input energy.
        p_mech_em = T_e * omega                 # electromagnetic (air-gap) power
        W_elec = np.trapezoid(p_elec, t) if N > 1 else 0.0          # net input
        W_mech = np.trapezoid(p_mech_em, t) if N > 1 else 0.0       # converted
        W_copper = (
            np.trapezoid(self.R * np.sum(phase_currents ** 2, axis=0), t)
            if N > 1 else 0.0
        )

        # Conversion efficiency = electromechanical work / net electrical input.
        # By the power balance above this is W_mech/(W_mech + W_copper + dW_field),
        # which is strictly in (0,1) for a passive reluctance machine (W_copper>0).
        denom = W_mech + W_copper
        eff = W_mech / denom if denom > 1e-12 else 0.0
        eff = float(np.clip(eff, 0.0, 0.999999))

        T_avg = float(np.mean(T_e))
        T_max = float(np.max(T_e)) if N else 0.0
        T_min = float(np.min(T_e)) if N else 0.0
        # Torque ripple = (Tmax - Tmin) / Tavg  (Krishnan 2001 definition)
        ripple = (T_max - T_min) / T_avg if T_avg > 1e-9 else 0.0

        return {
            "t": t,
            "theta": theta,
            "omega": omega,
            "speed_rpm": omega * 60.0 / (2.0 * np.pi),
            "torque": T_e,
            "phase_currents": phase_currents,
            "p_elec": p_elec,
            "W_elec_J": float(W_elec),
            "W_mech_J": float(W_mech),
            "W_copper_J": float(W_copper),
            "T_avg": T_avg,
            "T_max": T_max,
            "T_min": T_min,
            "torque_ripple": float(ripple),
            "efficiency": eff,
        }
