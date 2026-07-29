"""
EC180 -- Doubly-Fed Induction Generator (DFIG) -- F2a dq-Frame Dynamic Model

Physics-lumped doubly-fed induction machine. The STATOR is connected directly
to the grid (fixed voltage, fixed frequency); the ROTOR is fed by a back-to-back
voltage-source converter that injects controllable dq voltages (v_dr, v_qr).
This is the defining feature of a DFIG: both windings are excited ("doubly fed").

State vector (synchronous dq reference frame, aligned with the grid voltage):
    x = [i_ds, i_qs, i_dr, i_qr, omega_r]
where omega_r is the ELECTRICAL rotor angular speed [rad/s].

Voltage equations (Krause/Abad, motor sign convention, currents into machine):
    v_ds = Rs*i_ds + dpsi_ds/dt - omega_s*psi_qs
    v_qs = Rs*i_qs + dpsi_qs/dt + omega_s*psi_ds
    v_dr = Rr*i_dr + dpsi_dr/dt - (omega_s-omega_r)*psi_qr
    v_qr = Rr*i_qr + dpsi_qr/dt + (omega_s-omega_r)*psi_dr

Flux linkages:
    psi_ds = Ls*i_ds + Lm*i_dr ,  psi_qs = Ls*i_qs + Lm*i_qr
    psi_dr = Lr*i_dr + Lm*i_ds ,  psi_qr = Lr*i_qr + Lm*i_qs
with Ls = Lls + Lm, Lr = Llr + Lm.

Solving the four flux-derivative equations for the current derivatives gives a
linear system M * di/dt = rhs, where M is the constant 4x4 inductance matrix
    M = [[Ls,0,Lm,0],[0,Ls,0,Lm],[Lm,0,Lr,0],[0,Lm,0,Lr]].
We invert M once and integrate di/dt = M^{-1} * rhs with scipy.solve_ivp.

Electromagnetic torque (motor convention, P = pole pairs):
    T_e = 1.5 * P * Lm * (i_qs*i_dr - i_ds*i_qr)
Mechanical equation (generator: prime mover supplies T_mech > 0 motoring sense):
    d(omega_r)/dt = (P/J) * (T_e - T_mech_em) - (B/J)*omega_r ... see simulate.

Power (grid/generator convention, P_stator<0 = delivered to grid):
    P_s = 1.5*(v_ds*i_ds + v_qs*i_qs)      Q_s = 1.5*(v_qs*i_ds - v_ds*i_qs)
    P_r = 1.5*(v_dr*i_dr + v_qr*i_qr)

Slip / slip-power relation (Pena 1996, Abad 2011):
    s = (omega_s - omega_r) / omega_s
    P_rotor (air-gap, neglecting losses) = -s * P_stator
    -> super-synchronous (s<0): rotor delivers power to grid through the converter
    -> sub-synchronous   (s>0): rotor draws power from the grid

Stator-flux-oriented power control (Pena 1996): aligning the d-axis with the
stator flux makes stator active power proportional to i_qr and reactive power to
i_dr, decoupling P and Q. We use this to compute the rotor currents (hence rotor
voltages) needed to hit (P_stator_ref, Q_stator_ref) set-points.

References:
    Pena, R., Clare, J.C., Asher, G.M. (1996). "Doubly fed induction generator
      using back-to-back PWM converters and its application to variable-speed
      wind-energy generation." IEE Proc. Electr. Power Appl. 143(3):231-241.
    Abad, G., Lopez, J., Rodriguez, M., Marroyo, L., Iwanski, G. (2011).
      Doubly Fed Induction Machine: Modeling and Control for Wind Energy
      Generation. Wiley-IEEE Press.
    Krause, P.C., Wasynczuk, O., Sudhoff, S.D. (2013). Analysis of Electric
      Machinery and Drive Systems, 3rd ed. Wiley-IEEE Press.
"""

import numpy as np
from scipy.integrate import solve_ivp


class DFIG_F2a:
    """Doubly-fed induction generator -- dq-frame dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Rs = u["Rs"]["value"]
        self.Rr = u["Rr"]["value"]
        self.Lls = u["Lls"]["value"]
        self.Llr = u["Llr"]["value"]
        self.Lm = u["Lm"]["value"]
        self.Ls = self.Lls + self.Lm
        self.Lr = self.Llr + self.Lm
        self.P = u["P"]["value"]                 # pole pairs
        self.J = u["J"]["value"]
        self.B = u["B"]["value"]
        self.f_grid = u["f_grid_Hz"]["value"]
        self.V_stator_LL = u["V_stator_LL_V"]["value"]
        self.eta_conv = u["eta_converter"]["value"]
        self.slip_min = u["slip_min"]["value"]
        self.slip_max = u["slip_max"]["value"]

        self.omega_s = 2.0 * np.pi * self.f_grid          # electrical rad/s
        # Peak phase voltage of the grid-tied stator, d-axis aligned with it
        self.v_ds = self.V_stator_LL * np.sqrt(2.0 / 3.0)
        self.v_qs = 0.0

        # Constant inductance matrix M (di/dt couplings) and its inverse
        Ls, Lr, Lm = self.Ls, self.Lr, self.Lm
        self._M = np.array([
            [Ls, 0.0, Lm, 0.0],
            [0.0, Ls, 0.0, Lm],
            [Lm, 0.0, Lr, 0.0],
            [0.0, Lm, 0.0, Lr],
        ])
        self._Minv = np.linalg.inv(self._M)

    # ------------------------------------------------------------------ core
    def derivatives(self, t, x, v_dr, v_qr):
        """
        State derivatives for x = [i_ds, i_qs, i_dr, i_qr, omega_r].

        Stator is grid-fed: v_ds, v_qs are fixed by the grid.
        Rotor is converter-fed: v_dr, v_qr supplied by the rotor-side converter.
        Speed is treated as a slow input here (held by prime mover); the
        mechanical ODE is integrated in simulate() variants that drive speed.
        """
        i_ds, i_qs, i_dr, i_qr, omega_r = x
        ws = self.omega_s
        wsl = ws - omega_r  # slip electrical frequency

        # psi terms appearing in the back-EMF (speed-voltage) couplings
        psi_ds = self.Ls * i_ds + self.Lm * i_dr
        psi_qs = self.Ls * i_qs + self.Lm * i_qr
        psi_dr = self.Lr * i_dr + self.Lm * i_ds
        psi_qr = self.Lr * i_qr + self.Lm * i_qs

        # rhs_k = v_k - Rk*i_k + speed-voltage term  (= M * di/dt)
        r0 = self.v_ds - self.Rs * i_ds + ws * psi_qs
        r1 = self.v_qs - self.Rs * i_qs - ws * psi_ds
        r2 = v_dr - self.Rr * i_dr + wsl * psi_qr
        r3 = v_qr - self.Rr * i_qr - wsl * psi_dr

        di = self._Minv @ np.array([r0, r1, r2, r3])
        return di  # length-4 current derivatives

    def torque(self, i_ds, i_qs, i_dr, i_qr):
        """Electromagnetic torque [Nm] (motor convention)."""
        return 1.5 * self.P * self.Lm * (i_qs * i_dr - i_ds * i_qr)

    def stator_power(self, i_ds, i_qs):
        """Stator active/reactive power [W, VAr] (generator: P<0 -> to grid)."""
        P_s = 1.5 * (self.v_ds * i_ds + self.v_qs * i_qs)
        Q_s = 1.5 * (self.v_qs * i_ds - self.v_ds * i_qs)
        return P_s, Q_s

    def rotor_power(self, v_dr, v_qr, i_dr, i_qr):
        """Rotor (slip) power through the converter [W]."""
        return 1.5 * (v_dr * i_dr + v_qr * i_qr)

    def slip(self, omega_r):
        """Slip s = (omega_s - omega_r)/omega_s."""
        return (self.omega_s - omega_r) / self.omega_s

    def omega_r_from_slip(self, s):
        """Electrical rotor speed [rad/s] for a given slip."""
        return self.omega_s * (1.0 - s)

    def speed_rpm(self, omega_r):
        """Mechanical rotor speed [rpm]."""
        return omega_r / self.P * 30.0 / np.pi

    # ------------------------------------------- stator-flux-oriented control
    def reference_currents(self, P_stator_ref, Q_stator_ref):
        """
        Rotor currents (i_dr, i_qr) required to deliver the requested stator
        active/reactive power (Pena 1996 decoupled P/Q control).

        Here the d-axis is aligned with the GRID (stator) VOLTAGE: v_ds = V_peak,
        v_qs = 0. With stator resistance neglected the steady-state stator flux
        satisfies psi_ds ~= 0, psi_qs ~= -V/omega_s, which through the flux
        constraints gives:
            i_ds = -(Lm/Ls) * i_dr
            i_qs = -(V/omega_s + Lm*i_qr) / Ls
        and therefore (with P_s = 1.5*v_ds*i_ds, Q_s = -1.5*v_ds*i_qs):
            P_s = -1.5 * (Lm/Ls) * v_s * i_dr            -> i_dr controls P
            Q_s =  1.5 * (v_s/Ls) * (V/omega_s + Lm*i_qr) -> i_qr controls Q
        Invert for the reference rotor currents.
        """
        v_s = self.v_ds  # peak stator phase voltage (= V_peak)
        i_dr_ref = -P_stator_ref / (1.5 * (self.Lm / self.Ls) * v_s)
        i_qr_ref = ((Q_stator_ref * self.Ls) / (1.5 * v_s)
                    - v_s / self.omega_s) / self.Lm
        return i_dr_ref, i_qr_ref

    def rotor_voltage_for_currents(self, i_dr_ref, i_qr_ref, i_ds, i_qs, omega_r):
        """
        Steady-state rotor voltage that holds the rotor currents at their
        references (di_dr/dt = di_qr/dt = 0 in the rotor voltage equations):
            v_dr = Rr*i_dr - wsl*psi_qr
            v_qr = Rr*i_qr + wsl*psi_dr
        """
        wsl = self.omega_s - omega_r
        psi_dr = self.Lr * i_dr_ref + self.Lm * i_ds
        psi_qr = self.Lr * i_qr_ref + self.Lm * i_qs
        v_dr = self.Rr * i_dr_ref - wsl * psi_qr
        v_qr = self.Rr * i_qr_ref + wsl * psi_dr
        return v_dr, v_qr

    # -------------------------------------------------------------- simulate
    def simulate(self, v_dr, v_qr, slip=-0.2, dt=1e-4, duration_s=1.0, x0=None):
        """
        Simulate the electrical dynamics at a FIXED rotor speed (set by slip),
        with prescribed rotor-side converter voltages (open-loop / direct mode).

        Args:
            v_dr, v_qr: rotor dq voltages [V] (scalar or callable(t))
            slip:       operating slip (rotor speed held by prime mover)
            dt:         output time step [s]
            duration_s: total time [s]
            x0:         initial [i_ds, i_qs, i_dr, i_qr]
        """
        _vd = v_dr if callable(v_dr) else (lambda t: v_dr)
        _vq = v_qr if callable(v_qr) else (lambda t: v_qr)
        omega_r = self.omega_r_from_slip(slip)

        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 0.0]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, x):
            xfull = [x[0], x[1], x[2], x[3], omega_r]
            return self.derivatives(t, xfull, _vd(t), _vq(t))

        sol = solve_ivp(rhs, (0.0, duration_s), x0, t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=dt)
        return self._package(sol, lambda t: _vd(t), lambda t: _vq(t),
                             omega_r_arr=np.full_like(sol.t, omega_r))

    def simulate_power_control(self, P_stator_ref, Q_stator_ref, slip=-0.2,
                               dt=1e-4, duration_s=1.0, x0=None):
        """
        Closed-loop-equivalent mode: the rotor-side converter regulates stator
        active and reactive power to set-points via stator-flux orientation
        (Pena 1996). Rotor speed fixed by slip. Returns the achieved P_s, Q_s.

        Args:
            P_stator_ref: stator active power set-point [W] (<0 -> to grid)
            Q_stator_ref: stator reactive power set-point [VAr]
            slip:         operating slip
        """
        _Pref = P_stator_ref if callable(P_stator_ref) else (lambda t: P_stator_ref)
        _Qref = Q_stator_ref if callable(Q_stator_ref) else (lambda t: Q_stator_ref)
        omega_r = self.omega_r_from_slip(slip)

        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 0.0]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        # PI rotor-side-converter inner current loop with active cross-coupling
        # decoupling, tuned by internal-model control (Abad 2011, ch. 7):
        #   sigma = leakage factor, sigma_Lr = transient rotor inductance,
        #   bandwidth alpha = 1/tau_i  ->  Kp = sigma_Lr*alpha, Ki = Rr*alpha.
        # The slip-frequency cross terms (+/- omega_sl*sigma_Lr*i_other) and the
        # rotational-EMF term (omega_sl*Lm/Ls*psi_s) are fed forward so the two
        # rotor-current axes become decoupled first-order loops. The integral
        # term zeroes steady-state error; the loop is stable at BOTH sub- and
        # super-synchronous slip.
        sigma_Lr = self.Lr - self.Lm**2 / self.Ls
        alpha = 1.0 / 1e-3                      # 1 ms current-loop bandwidth
        Kp = sigma_Lr * alpha
        Ki = self.Rr * alpha
        psi_s = self.v_ds / self.omega_s        # stator flux magnitude (on -q)

        # state = [i_ds, i_qs, i_dr, i_qr, xi_d, xi_q] (xi = integral of error)
        x0 = list(x0) + [0.0, 0.0]

        def control(t, x):
            i_ds, i_qs, i_dr, i_qr = x[0], x[1], x[2], x[3]
            xi_d, xi_q = x[4], x[5]
            omega_sl = self.omega_s - omega_r
            idr_ref, iqr_ref = self.reference_currents(_Pref(t), _Qref(t))
            e_d = idr_ref - i_dr
            e_q = iqr_ref - i_qr
            # PI + active decoupling of the slip-frequency cross terms
            v_dr = Kp * e_d + Ki * xi_d - omega_sl * sigma_Lr * i_qr
            v_qr = (Kp * e_q + Ki * xi_q + omega_sl * sigma_Lr * i_dr
                    + omega_sl * (self.Lm / self.Ls) * psi_s)
            return v_dr, v_qr, e_d, e_q

        def rhs(t, x):
            v_dr, v_qr, e_d, e_q = control(t, x)
            di = self.derivatives(t, [x[0], x[1], x[2], x[3], omega_r],
                                  v_dr, v_qr)
            return [di[0], di[1], di[2], di[3], e_d, e_q]

        sol = solve_ivp(rhs, (0.0, duration_s), x0, t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=dt)

        # trim integrator states for packaging; reconstruct rotor voltages
        vdr_arr = np.zeros_like(sol.t)
        vqr_arr = np.zeros_like(sol.t)
        for k in range(len(sol.t)):
            vdr_arr[k], vqr_arr[k], _, _ = control(sol.t[k], sol.y[:, k])
        sol.y = sol.y[:4]  # keep only electrical states for _package
        return self._package(sol,
                             lambda t: np.interp(t, sol.t, vdr_arr),
                             lambda t: np.interp(t, sol.t, vqr_arr),
                             omega_r_arr=np.full_like(sol.t, omega_r),
                             vdr_arr=vdr_arr, vqr_arr=vqr_arr)

    def simulate_mechanical(self, v_dr, v_qr, T_mech_Nm, dt=1e-4,
                            duration_s=2.0, x0=None, omega_r0=None):
        """
        Full electromechanical run: 4 electrical + 1 mechanical ODE.
        A prime-mover mechanical torque T_mech (wind, motor convention < 0 for
        generating drive) accelerates the rotor; converter voltages excite the
        rotor. Demonstrates variable-speed operation and energy exchange.
        """
        _vd = v_dr if callable(v_dr) else (lambda t: v_dr)
        _vq = v_qr if callable(v_qr) else (lambda t: v_qr)
        _Tm = T_mech_Nm if callable(T_mech_Nm) else (lambda t: T_mech_Nm)

        if x0 is None:
            x0 = [0.0, 0.0, 0.0, 0.0]
        if omega_r0 is None:
            omega_r0 = self.omega_s  # start at synchronous
        x0 = list(x0) + [omega_r0]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, x):
            di = self.derivatives(t, x, _vd(t), _vq(t))
            i_ds, i_qs, i_dr, i_qr, omega_r = x
            T_e = self.torque(i_ds, i_qs, i_dr, i_qr)
            # mechanical: J*domega_m/dt = T_mech + T_e - B*omega_m
            # omega_m = omega_r / P ; T_mech is prime-mover torque
            omega_m = omega_r / self.P
            domega_m = (_Tm(t) + T_e - self.B * omega_m) / self.J
            domega_r = self.P * domega_m
            return [di[0], di[1], di[2], di[3], domega_r]

        sol = solve_ivp(rhs, (0.0, duration_s), x0, t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-8, max_step=dt)
        return self._package(sol, lambda t: _vd(t), lambda t: _vq(t),
                             omega_r_arr=sol.y[4])

    # --------------------------------------------------------------- packing
    def _package(self, sol, vdr_fun, vqr_fun, omega_r_arr,
                 vdr_arr=None, vqr_arr=None):
        t = sol.t
        i_ds, i_qs, i_dr, i_qr = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
        if vdr_arr is None:
            vdr_arr = np.array([vdr_fun(ti) for ti in t])
        if vqr_arr is None:
            vqr_arr = np.array([vqr_fun(ti) for ti in t])

        T_e = self.torque(i_ds, i_qs, i_dr, i_qr)
        P_s, Q_s = self.stator_power(i_ds, i_qs)
        P_r = self.rotor_power(vdr_arr, vqr_arr, i_dr, i_qr)
        s = self.slip(omega_r_arr)
        i_s = np.sqrt(i_ds**2 + i_qs**2)
        i_r = np.sqrt(i_dr**2 + i_qr**2)
        # Total power delivered to grid (generator): stator + converter (rotor)
        # account for converter efficiency on the rotor branch
        P_grid = P_s + np.where(P_r < 0, P_r * self.eta_conv, P_r / self.eta_conv)

        return {
            "t": t,
            "i_ds": i_ds, "i_qs": i_qs, "i_dr": i_dr, "i_qr": i_qr,
            "i_stator": i_s, "i_rotor": i_r,
            "v_dr": vdr_arr, "v_qr": vqr_arr,
            "torque": T_e,
            "omega_r": omega_r_arr,
            "speed_rpm": self.speed_rpm(omega_r_arr),
            "slip": s,
            "P_stator": P_s, "Q_stator": Q_s,
            "P_rotor": P_r, "P_grid": P_grid,
        }
