"""
EC185 -- Static VAR Compensator (SVC) -- F2a Physics-Lumped Voltage Control

Physics-lumped (0D) dynamic model of a Fixed-Capacitor / Thyristor-Controlled-
Reactor (FC-TCR) + Thyristor-Switched-Capacitor (TSC) SVC providing continuously
variable shunt reactive power for bus-voltage regulation.

----------------------------------------------------------------------------
1. TCR susceptance vs firing angle  B_L(alpha)
----------------------------------------------------------------------------
A thyristor-controlled reactor conducts for a partial-conduction angle
sigma = 2*(pi - alpha), measured from the voltage peak. Fourier analysis of the
chopped reactor current gives the fundamental-frequency effective susceptance
(Hingorani & Gyugyi 2000, Eq. 5.4; Mathur & Varma 2002, Eq. 3.x):

    B_L(alpha) = (2*(pi - alpha) + sin(2*alpha)) / (pi * X_L)

with alpha the firing angle measured from the zero crossing of the applied
voltage, alpha in [pi/2, pi]:
    alpha = pi/2  -> full conduction   -> B_L = 1/X_L              (max inductive)
    alpha = pi    -> no  conduction    -> B_L = 0
B_L is monotonically DECREASING in alpha over [pi/2, pi].

----------------------------------------------------------------------------
2. Net SVC susceptance and reactive power
----------------------------------------------------------------------------
The fixed/switched capacitor contributes a constant capacitive susceptance B_C.
The net SVC susceptance (capacitive positive) is:

    B_svc(alpha) = B_C - B_L(alpha)

The reactive power injected into the bus is the classic shunt-susceptance law:

    Q_svc = B_svc * V^2            (per-unit, S_base)         [Q>0 = capacitive]

so Q ranges continuously from inductive (B_L large, B_svc<0) to capacitive
(B_L=0, B_svc=B_C>0) -- the SVC's defining capability.

----------------------------------------------------------------------------
3. V-Q droop (slope-reactance) control characteristic
----------------------------------------------------------------------------
Within the controllable range the SVC enforces a sloped V-I (equivalently V-Q)
characteristic (Mathur & Varma 2002, Sec. 4; Hingorani & Gyugyi 2000, Fig. 5.x):

    V_bus = V_ref + X_SL * I_svc  ~=  V_ref + X_SL * (Q_svc / V_bus)

The droop slope X_SL (typ. 1-5 %) shares reactive load between parallel
compensators and limits hunting. Outside the range the SVC saturates to a
fixed capacitor (V low) or fixed reactor (V high).

----------------------------------------------------------------------------
4. Lumped voltage-control loop ODE  (scipy.solve_ivp)
----------------------------------------------------------------------------
The closed-loop regulator + thyristor firing is represented by two lumped
first-order states: the measured/regulated susceptance command B_cmd and the
realised susceptance B_act (firing/transport lag). With a droop-augmented
voltage error the regulator drives B toward the value that nulls
(V_ref - V_bus + X_SL*Q):

    T_r  dB_cmd/dt = K*(V_ref - V_bus + X_SL*Q_svc) - dB_cmd_offset
    T_th dB_act/dt = B_cmd - B_act

and the algebraic network couples B_act back to V_bus through a Thevenin
source  E behind reactance X_thev:

    V_bus = E / (1 - X_thev * B_act)      (shunt-susceptance voltage divider)

This is a genuine 0D/1D ODE system integrated in time -- the physics-lumped
upgrade of the algebraic F1 models.

References:
    Hingorani, N.G. & Gyugyi, L. (2000). Understanding FACTS: Concepts and
        Technology of Flexible AC Transmission Systems. IEEE Press. Ch. 5.
    Mathur, R.M. & Varma, R.K. (2002). Thyristor-Based FACTS Controllers for
        Electrical Transmission Systems. IEEE Press / Wiley. Ch. 3-4.
    IEEE Std 1031-2011: Guide for the Functional Specification of TSC/TCR SVCs.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SVC_F2a:
    """SVC FC-TCR/TSC -- physics-lumped voltage-control model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_base = u["S_base_MVA"]["value"]            # MVA
        self.V_rated_kV = u["V_rated_kV"]["value"]
        self.B_C = u["B_cap_pu"]["value"]                 # pu, fixed/switched cap
        self.X_L = u["X_L_pu"]["value"]                   # pu, TCR reactor reactance
        self.B_L_max = u["B_ind_max_pu"]["value"]         # pu, = 1/X_L
        self.alpha_min = np.radians(u["alpha_min_deg"]["value"])  # rad (full cond.)
        self.alpha_max = np.radians(u["alpha_max_deg"]["value"])  # rad (no cond.)
        self.V_ref = u["V_ref_pu"]["value"]
        self.X_SL = u["droop_slope_pu"]["value"]          # droop slope reactance
        self.T_r = u["T_regulator_s"]["value"]            # s
        self.K = u["K_regulator"]["value"]
        self.T_th = u["T_thyristor_s"]["value"]           # s
        self.loss_factor = u["loss_factor"]["value"]
        # Net susceptance limits (capacitive positive)
        self.B_svc_max = self.B_C                         # alpha=pi  (full cap)
        self.B_svc_min = self.B_C - self.B_L_max          # alpha=pi/2 (full ind)

    # ------------------------------------------------------------------
    # 1. TCR susceptance vs firing angle  B_L(alpha)
    # ------------------------------------------------------------------
    def tcr_susceptance(self, alpha):
        """
        TCR fundamental susceptance [pu on S_base] for firing angle alpha [rad].

        B_L(alpha) = (2*(pi - alpha) + sin(2*alpha)) / (pi * X_L)

        Clamped to [pi/2, pi]; returns value in [0, 1/X_L].
        """
        a = np.clip(alpha, self.alpha_min, self.alpha_max)
        return (2.0 * (np.pi - a) + np.sin(2.0 * a)) / (np.pi * self.X_L)

    def net_susceptance(self, alpha):
        """Net SVC susceptance B_svc = B_C - B_L(alpha) [pu] (cap. positive)."""
        return self.B_C - self.tcr_susceptance(alpha)

    def alpha_from_B(self, B_svc_target):
        """
        Invert B_svc(alpha) -> firing angle alpha [rad] by monotone bisection.
        B_L_target = B_C - B_svc_target must lie in [0, 1/X_L].
        """
        B_L_target = np.clip(self.B_C - B_svc_target, 0.0, self.B_L_max)
        lo, hi = self.alpha_min, self.alpha_max  # B_L decreasing in alpha
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            B_mid = self.tcr_susceptance(mid)
            if B_mid > B_L_target:   # too much conduction -> raise alpha
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------
    # 2. Reactive power from net susceptance
    # ------------------------------------------------------------------
    def reactive_power_pu(self, B_svc, V_pu):
        """Q_svc [pu on S_base] = B_svc * V^2 (capacitive positive)."""
        return B_svc * V_pu ** 2

    def reactive_power_MVAR(self, B_svc, V_pu):
        """Q_svc in MVAR (capacitive positive)."""
        return self.reactive_power_pu(B_svc, V_pu) * self.S_base

    def losses_MW(self, Q_MVAR):
        """SVC active-power losses [MW] ~ loss_factor * |Q|."""
        return self.loss_factor * np.abs(Q_MVAR)

    # ------------------------------------------------------------------
    # 3. V-Q droop characteristic
    # ------------------------------------------------------------------
    def droop_voltage(self, Q_pu):
        """
        Steady-state bus voltage on the SVC slope characteristic [pu]:
            V_bus = V_ref + X_SL * I_svc ;  I_svc = Q_pu / V (~ Q_pu near 1 pu)
        Sign: capacitive Q (>0) raises V relative to V_ref via the droop.
        """
        return self.V_ref + self.X_SL * Q_pu

    def steady_state_susceptance(self, E_thev, X_thev):
        """
        Solve the algebraic steady state of the droop-regulated SVC against a
        Thevenin source E behind X_thev.  Network voltage divider:
            V = E / (1 - X_thev * B)
        Droop regulator target (B>0 capacitive):
            V = V_ref + X_SL * (B * V^2)
        Solve for B by fixed-point / bisection over the controllable B range.
        Returns (B_svc_pu, V_pu, saturated_flag).
        """
        def V_of_B(B):
            denom = 1.0 - X_thev * B
            denom = denom if abs(denom) > 1e-9 else 1e-9
            return E_thev / denom

        def residual(B):
            V = V_of_B(B)
            Q = B * V ** 2
            # regulator wants V == V_ref + X_SL*Q
            return V - (self.V_ref + self.X_SL * Q)

        lo, hi = self.B_svc_min, self.B_svc_max
        r_lo, r_hi = residual(lo), residual(hi)
        if r_lo * r_hi > 0:
            # No root in controllable band -> SVC saturates at the nearer limit
            B_sat = lo if abs(r_lo) < abs(r_hi) else hi
            return B_sat, V_of_B(B_sat), True
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            r_mid = residual(mid)
            if r_lo * r_mid <= 0:
                hi, r_hi = mid, r_mid
            else:
                lo, r_lo = mid, r_mid
        B = 0.5 * (lo + hi)
        return B, V_of_B(B), False

    # ------------------------------------------------------------------
    # 4. Lumped voltage-control loop ODE
    # ------------------------------------------------------------------
    def _V_bus(self, B_act, E_thev, X_thev):
        """Algebraic bus voltage from realised susceptance (cap. positive)."""
        denom = 1.0 - X_thev * B_act
        if abs(denom) < 1e-9:
            denom = 1e-9 if denom >= 0 else -1e-9
        return E_thev / denom

    def simulate(self, E_thev, X_thev, dt, duration_s,
                 V_ref=None, B0=None, E_disturbance=None):
        """
        Closed-loop voltage-regulation transient via scipy.solve_ivp.

        States:
            y[0] = B_cmd  -- regulator susceptance command [pu]
            y[1] = B_act  -- realised SVC susceptance       [pu]

        Parameters
        ----------
        E_thev : float
            Thevenin source voltage behind X_thev [pu] (the disturbance the SVC
            corrects).  May be overridden by E_disturbance(t).
        X_thev : float
            System/source short-circuit reactance [pu on S_base].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation time [s].
        V_ref : float, optional
            Regulator setpoint [pu] (default parameter V_ref).
        B0 : float, optional
            Initial susceptance [pu] (default = mid of controllable range).
        E_disturbance : callable(t) -> float, optional
            Time-varying Thevenin voltage; overrides E_thev when given.

        Returns
        -------
        dict of time-series:
            t, V_bus, B_act, B_cmd, alpha_deg, Q_MVAR, Q_pu, P_loss_MW, mode
        """
        Vref = self.V_ref if V_ref is None else V_ref
        _E = E_disturbance if callable(E_disturbance) else (lambda t: E_thev)

        if B0 is None:
            B0 = 0.5 * (self.B_svc_min + self.B_svc_max)
        B0 = float(np.clip(B0, self.B_svc_min, self.B_svc_max))

        Kd = self.K / max(self.X_SL, 1e-6)  # droop -> effective regulator gain

        def rhs(t, y):
            B_cmd, B_act = y
            E = _E(t)
            V = self._V_bus(B_act, E, X_thev)
            Q = B_act * V ** 2
            # droop-augmented voltage error: regulator nulls (Vref - V + X_SL*Q)
            err = (Vref - V) + self.X_SL * Q
            dB_cmd = (Kd * err) / self.T_r
            # anti-windup: hold command within controllable band
            if (B_cmd >= self.B_svc_max and dB_cmd > 0) or \
               (B_cmd <= self.B_svc_min and dB_cmd < 0):
                dB_cmd = 0.0
            dB_act = (B_cmd - B_act) / self.T_th
            return [dB_cmd, dB_act]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [B0, B0],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        t_out = sol.t
        B_cmd_out = np.clip(sol.y[0], self.B_svc_min, self.B_svc_max)
        B_act_out = np.clip(sol.y[1], self.B_svc_min, self.B_svc_max)
        N = len(t_out)

        V_bus = np.zeros(N)
        Q_pu = np.zeros(N)
        Q_MVAR = np.zeros(N)
        alpha_deg = np.zeros(N)
        P_loss = np.zeros(N)
        for i in range(N):
            E = _E(t_out[i])
            V_bus[i] = self._V_bus(B_act_out[i], E, X_thev)
            Q_pu[i] = B_act_out[i] * V_bus[i] ** 2
            Q_MVAR[i] = Q_pu[i] * self.S_base
            alpha_deg[i] = np.degrees(self.alpha_from_B(B_act_out[i]))
            P_loss[i] = self.losses_MW(Q_MVAR[i])

        if Q_MVAR[-1] > 0.5:
            mode = "capacitive"
        elif Q_MVAR[-1] < -0.5:
            mode = "inductive"
        else:
            mode = "floating"

        return {
            "t": t_out,
            "V_bus": V_bus,
            "B_act": B_act_out,
            "B_cmd": B_cmd_out,
            "alpha_deg": alpha_deg,
            "Q_pu": Q_pu,
            "Q_MVAR": Q_MVAR,
            "P_loss_MW": P_loss,
            "mode": mode,
        }
