"""
EC177 -- Brushless DC Motor (BLDC) -- F2a Physics-Lumped dq / phase-domain Model

A trapezoidal-back-EMF permanent-magnet machine driven by six-step (120-deg)
commutation. Because only two of the three phases conduct at any instant in
six-step operation, the three-phase machine reduces to an *effective two-phase-
in-series DC equivalent circuit* (the classic BLDC lumped model), with an
electrical state equation and a mechanical state equation integrated together
with scipy.solve_ivp.

State vector  y = [i, omega, theta_e]
    i        : effective conducting-loop current            [A]
    omega    : rotor mechanical angular speed               [rad/s]
    theta_e  : electrical rotor angle (for commutation/EMF) [rad]

Electrical ODE (line-to-line loop; two phases in series):
    L_eq * di/dt = v_app - R_eq * i - e_bemf
        R_eq = 2*Rs        (two phases in series)
        L_eq = 2*Ls        (two phases in series)
        e_bemf = Ke_ll * omega * f_trap(theta_e)
        v_app  = Vdc * s(theta_e)     six-step switching sign / duty

Electromagnetic torque (trapezoidal product):
    T_e = Kt_ll * f_trap(theta_e) * i        (== Ke_ll for a balanced PM machine)
    In the flat-top region f_trap = +/-1 so T_e = Kt * i  -> torque proportional to current.

Mechanical ODE (rigid rotor):
    J * domega/dt = T_e - T_load - B*omega - Tc*sign(omega)
    dtheta_e/dt   = (poles/2) * omega

Speed-torque curve (steady state, flat-top operation):
    omega_no_load = Vdc / Ke          (i -> 0, T_e -> 0)
    T_stall       = Kt * Vdc / R_eq   (omega = 0)
    omega = (Vdc - R_eq*T/Kt) / Ke    -> the characteristic straight line.

References:
    Krishnan, R. (2010). "Permanent Magnet Synchronous and Brushless DC Motor
        Drives." CRC Press. Ch. 9 (BLDC machine modeling & six-step drive).
    Hanselman, D.C. (2006). "Brushless Permanent Magnet Motor Design." Magna Physics.
    Pillay, P. & Krishnan, R. (1989). "Modeling of permanent magnet motor drives."
        IEEE Trans. Ind. Electron., 35(4), 537-541.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BLDC_F2a:
    """BLDC motor -- six-step trapezoidal-EMF lumped electrical+mechanical model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["rated_power_W"]["value"]          # W
        self.omega_rated = u["omega_rated_rpm"]["value"] * 2.0 * np.pi / 60.0  # rad/s
        self.Vdc = u["Vdc"]["value"]                        # V
        self.Kt = u["Kt"]["value"]                          # Nm/A (per-phase flat-top)
        self.Ke = u["Ke"]["value"]                          # V/(rad/s)
        self.poles = u["poles"]["value"]
        self.Rs = u["Rs"]["value"]                          # ohm (per phase)
        self.Ls = u["Ls"]["value"]                          # H (per phase)
        self.J = u["J"]["value"]                            # kg.m2
        self.B = u["B"]["value"]                            # N.m.s/rad
        self.Tc = u["Tc"]["value"]                          # N.m

        # Six-step equivalent: two phases conduct in series.
        self.R_eq = 2.0 * self.Rs
        self.L_eq = 2.0 * self.Ls
        # In flat-top six-step the two series phase EMFs/torques add to give the
        # line constants; the per-phase Kt/Ke already represent the conducting pair
        # for this 1 kW machine (calibrated so no-load speed ~ Vdc/Ke).
        self.Ke_eq = self.Ke
        self.Kt_eq = self.Kt

    # ------------------------------------------------------------------
    # Trapezoidal back-EMF / commutation shape function
    # ------------------------------------------------------------------
    @staticmethod
    def f_trap(theta_e):
        """Normalized trapezoidal back-EMF shape f(theta_e) in [-1, 1].

        Standard ideal trapezoid: 120-deg flat top of +1, 60-deg linear ramp,
        120-deg flat top of -1, 60-deg ramp. (Krishnan 2010, Fig. 9.x.)
        """
        th = np.mod(theta_e, 2.0 * np.pi)
        # piecewise over 6 x 60-deg sectors
        d = np.pi / 3.0  # 60 deg
        if th < 2 * d:            # 0..120 : +1
            return 1.0
        elif th < 3 * d:          # 120..180 : ramp +1 -> -1
            return 1.0 - 2.0 * (th - 2 * d) / d
        elif th < 5 * d:          # 180..300 : -1
            return -1.0
        else:                     # 300..360 : ramp -1 -> +1
            return -1.0 + 2.0 * (th - 5 * d) / d

    @staticmethod
    def s_switch(theta_e):
        """Six-step applied-voltage sign s(theta_e) in {+1,-1}.

        The inverter applies +Vdc to the conducting loop when the back-EMF is
        positive and -Vdc when negative, so the switching function is aligned
        with the sign of the (flat-top) trapezoid -> drives current in phase
        with EMF for maximum torque.
        """
        f = BLDC_F2a.f_trap(theta_e)
        if f > 1e-9:
            return 1.0
        elif f < -1e-9:
            return -1.0
        return 0.0

    # ------------------------------------------------------------------
    # Back-EMF and torque (instantaneous)
    # ------------------------------------------------------------------
    def back_emf(self, omega, theta_e):
        """Instantaneous loop back-EMF [V] = Ke * omega * f_trap.  e ∝ omega."""
        return self.Ke_eq * omega * self.f_trap(theta_e)

    def torque_e(self, i, theta_e):
        """Electromagnetic torque [Nm] = Kt * f_trap * i.  In flat-top: T = Kt*i."""
        return self.Kt_eq * self.f_trap(theta_e) * i

    def friction_torque(self, omega):
        """Viscous + Coulomb friction [Nm]."""
        return self.B * omega + self.Tc * np.tanh(omega / 1.0)

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, T_load_fn, duty):
        i, omega, theta_e = y
        f = self.f_trap(theta_e)
        v_app = duty * self.Vdc * self.s_switch(theta_e)
        e = self.Ke_eq * omega * f
        di = (v_app - self.R_eq * i - e) / self.L_eq
        T_e = self.Kt_eq * f * i
        T_l = T_load_fn(t)
        domega = (T_e - T_l - self.friction_torque(omega)) / self.J
        dtheta = (self.poles / 2.0) * omega
        return [di, domega, dtheta]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, T_load=0.1, Vdc=None, duty=1.0, dt=2e-4, duration_s=0.4,
                 i0=0.0, omega0=0.0, theta0=0.0):
        """Integrate the coupled electrical + mechanical ODE.

        Parameters
        ----------
        T_load   : float or callable(t)  load torque [Nm]
        Vdc      : float or None          override bus voltage [V]
        duty     : float                  PWM duty (0..1) scaling applied voltage
        dt       : float                  output time step [s]
        duration_s : float                total sim time [s]
        i0, omega0, theta0 : initial states

        Returns
        -------
        dict of time-series arrays + scalar summary metrics.
        """
        if Vdc is not None:
            self.Vdc = float(Vdc)
        T_load_fn = T_load if callable(T_load) else (lambda t: float(T_load))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [i0, omega0, theta0],
            t_eval=t_eval, args=(T_load_fn, duty),
            method="LSODA", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        i = sol.y[0]
        omega = sol.y[1]
        theta_e = sol.y[2]
        N = len(t)

        e_bemf = np.array([self.back_emf(omega[k], theta_e[k]) for k in range(N)])
        T_e = np.array([self.torque_e(i[k], theta_e[k]) for k in range(N)])
        T_l = np.array([T_load_fn(t[k]) for k in range(N)])

        P_mech = T_e * omega                                  # W (air-gap power)
        P_shaft = T_l * omega                                 # W delivered to load
        v_app = np.array([duty * self.Vdc * self.s_switch(theta_e[k]) for k in range(N)])
        P_elec = v_app * i                                    # W instantaneous input
        P_cu = self.R_eq * i**2                               # copper loss W

        # Steady-state (last 20%) efficiency: useful shaft power / electrical input
        tail = slice(max(1, int(0.8 * N)), N)
        P_in_avg = np.mean(np.abs(P_elec[tail]))
        P_out_avg = np.mean(np.maximum(P_shaft[tail], 0.0))
        eff = (P_out_avg / P_in_avg) if P_in_avg > 1e-9 else 0.0
        eff = float(np.clip(eff, 0.0, 0.9999))

        return {
            "t": t,
            "current": i,
            "omega": omega,
            "speed_rpm": omega * 60.0 / (2.0 * np.pi),
            "theta_e": theta_e,
            "back_emf": e_bemf,
            "torque_e": T_e,
            "torque_load": T_l,
            "P_mech": P_mech,
            "P_elec": P_elec,
            "P_cu": P_cu,
            "efficiency": eff,
            "omega_final": float(omega[-1]),
            "current_final": float(i[-1]),
            "torque_e_final": float(T_e[-1]),
        }

    # ------------------------------------------------------------------
    # Analytic steady-state speed-torque characteristic (flat-top)
    # ------------------------------------------------------------------
    def no_load_speed(self, Vdc=None):
        """omega_0 = Vdc / Ke  [rad/s]."""
        V = self.Vdc if Vdc is None else Vdc
        return V / self.Ke_eq

    def stall_torque(self, Vdc=None):
        """T_stall = Kt * Vdc / R_eq  [Nm]."""
        V = self.Vdc if Vdc is None else Vdc
        return self.Kt_eq * V / self.R_eq

    def speed_at_torque(self, T, Vdc=None):
        """omega = (Vdc - R_eq*T/Kt) / Ke  [rad/s]  -- the linear T-omega line."""
        V = self.Vdc if Vdc is None else Vdc
        return (V - self.R_eq * T / self.Kt_eq) / self.Ke_eq
