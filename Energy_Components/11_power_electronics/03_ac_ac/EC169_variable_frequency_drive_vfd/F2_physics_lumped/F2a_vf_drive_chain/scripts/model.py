"""
EC169 -- Variable Frequency Drive (VFD) -- F2a Physics-Lumped V/f Drive Chain
=============================================================================

A physics-lumped (0D) averaged model of the complete VFD power chain:

    AC grid -> diode-bridge RECTIFIER -> DC-LINK capacitor -> PWM INVERTER
            -> 3-phase INDUCTION MOTOR (V/f scalar control)

Two coupled state variables are integrated with scipy.integrate.solve_ivp:

    1. DC-link voltage   V_dc(t)   -- capacitor charge balance
    2. Rotor speed       omega_m(t) -- mechanical Newton's law (rotational)

----------------------------------------------------------------------------
1.  V/f (constant volts-per-hertz) CONTROL  -- Bose (2006) ch.8; Mohan ch.14
----------------------------------------------------------------------------
The drive sets the inverter output frequency f_out to command motor speed.
Below base frequency the terminal voltage is kept proportional to frequency so
that the air-gap flux  Phi ~ V/f  stays constant (constant-torque region):

        V_out(f) = V_boost + (V_rated - V_boost) * (f / f_rated),   f <= f_rated
        V_out(f) = V_rated,                                          f  > f_rated   (field weakening)

V_boost offsets the stator IR drop at low speed (Bose, "voltage boost").

----------------------------------------------------------------------------
2.  INDUCTION-MOTOR TORQUE-SPEED  -- steady-state (Steinmetz) equivalent circuit
----------------------------------------------------------------------------
For an averaged model we use the per-phase Thevenin/Kloss steady-state torque
(Krause et al. 2013; Mohan ch.14). Synchronous and slip speeds:

        omega_s   = 2*pi*f_out * (2/poles)      [rad/s, mechanical synchronous]
        slip  s   = (omega_s - omega_m) / omega_s
        omega_e   = 2*pi*f_out                  [rad/s, electrical]

Reactances scale with frequency (X = 2*pi*f*L):

        X_s(f) = X_s_rated * f/f_rated ,  X_r(f) = X_r_rated * f/f_rated

Per-phase electromagnetic torque (3-phase, Thevenin form):

        V_th = V_ph * Xm_approx ... (we use stator voltage directly, standard
               approximation for leakage-dominated machines):

        T_em = (3 / omega_e) * V_ph^2 * (R_r/s)
                 / ( (R_s + R_r/s)^2 + (X_s + X_r)^2 )

where V_ph = V_out_ll / sqrt(3) is the per-phase RMS voltage applied by the
inverter. This reproduces the classic torque-speed curve: zero torque at
synchronous speed, a breakdown (pull-out) torque, and the linear
constant-flux region near synchronism.

----------------------------------------------------------------------------
3.  DC-LINK CAPACITOR  ODE  -- charge balance  (Mohan ch.5,8)
----------------------------------------------------------------------------
        C_dc * dV_dc/dt = I_rect - I_inv - V_dc/R_dc

    I_rect = eta_rect * P_grid_avail / V_dc           (rectifier feeds the bus)
    I_inv  = P_inv_in / V_dc = (P_mech + P_motor_loss) / (eta_inv * V_dc)

The bus is a stiff source: I_rect is whatever current holds V_dc at V_dc_nom in
steady state, so the capacitor ODE relaxes V_dc back to its nominal value while
supplying transient inverter current draw.

----------------------------------------------------------------------------
4.  MECHANICAL  ODE  -- rotor speed  (Krause 2013 eq. 4.3-6; Mohan ch.14)
----------------------------------------------------------------------------
        J * d(omega_m)/dt = T_em(omega_m, f_out, V_out) - T_load - B*omega_m

----------------------------------------------------------------------------
ENERGY / EFFICIENCY  (Mohan ch.5,8; Bose ch.1)
----------------------------------------------------------------------------
Drive-chain efficiency is the product of stage efficiencies times the motor
electrical efficiency:

        eta_drive = eta_rect * eta_inv                       (power-electronics)
        eta_motor = P_mech / P_motor_elec   (= 1 - copper-loss fraction)
        eta_chain = P_mech / P_grid    (0 < eta_chain < 1, energy-conserving)

References
---------
* N. Mohan, T. Undeland, W. Robbins (2003). *Power Electronics: Converters,
  Applications, and Design*, 3rd ed. Wiley. (ch.5 rectifiers, ch.8 PWM
  inverters, ch.14 motor drives & V/f control).
* B. K. Bose (2006). *Power Electronics and Motor Drives: Advances and Trends*.
  Academic Press. (ch.8 scalar V/f control, voltage boost).
* P. Krause, O. Wasynczuk, S. Sudhoff (2013). *Analysis of Electric Machinery
  and Drive Systems*, 3rd ed. IEEE/Wiley. (induction-machine torque, swing eq.)
"""

import numpy as np
from scipy.integrate import solve_ivp


class VFDF2a:
    """Physics-lumped V/f drive chain: rectifier + DC-link + PWM inverter + IM."""

    def __init__(self, params: dict):
        u = params["unit"]
        # --- Power electronics ---
        self.V_dc_nom = u["V_dc_nom"]["value"]
        self.C_dc = u["C_dc"]["value"]
        self.R_dc = u["R_dc_load"]["value"]
        self.eta_rect = u["eta_rect"]["value"]
        self.eta_inv = u["eta_inv"]["value"]

        # --- V/f profile ---
        self.V_rated = u["V_rated"]["value"]
        self.f_rated = u["f_rated"]["value"]
        self.f_max = u["f_out_max"]["value"]
        self.V_boost = u["V_boost"]["value"]
        self.poles = int(u["poles"]["value"])

        # --- Induction motor (Steinmetz per-phase) ---
        self.R_s = u["R_s"]["value"]
        self.R_r = u["R_r"]["value"]
        self.X_s = u["X_s"]["value"]
        self.X_r = u["X_r"]["value"]
        self.J = u["J"]["value"]
        self.B = u["B_visc"]["value"]
        self.P_rated = u["P_rated"]["value"]

    # ------------------------------------------------------------------
    # V/f control law  (Bose 2006; Mohan ch.14)
    # ------------------------------------------------------------------
    def output_voltage(self, f_out):
        """
        Inverter terminal voltage (line-to-line RMS) under constant-V/f control.
        Below base: V = V_boost + (V_rated - V_boost)*(f/f_rated).
        Above base: clamped to V_rated (field weakening).
        """
        f = np.clip(np.asarray(f_out, dtype=float), 0.0, self.f_max)
        ratio = f / self.f_rated
        v = self.V_boost + (self.V_rated - self.V_boost) * ratio
        return np.minimum(v, self.V_rated)

    def vf_ratio(self, f_out):
        """Volts-per-Hz ratio [V/Hz] (constant below base, ignoring boost)."""
        f = np.asarray(f_out, dtype=float)
        v = self.output_voltage(f)
        return np.where(f > 1e-9, v / f, 0.0)

    # ------------------------------------------------------------------
    # Speeds
    # ------------------------------------------------------------------
    def sync_speed_mech(self, f_out):
        """Mechanical synchronous speed [rad/s] = 2*pi*f * (2/poles)."""
        return 2.0 * np.pi * np.asarray(f_out, dtype=float) * (2.0 / self.poles)

    def slip(self, omega_m, f_out):
        """Per-unit slip s = (omega_s - omega_m)/omega_s."""
        omega_s = self.sync_speed_mech(f_out)
        return np.where(omega_s > 1e-6, (omega_s - omega_m) / omega_s, 0.0)

    # ------------------------------------------------------------------
    # Induction-motor electromagnetic torque  (Krause 2013; Mohan ch.14)
    # ------------------------------------------------------------------
    def motor_torque(self, omega_m, f_out):
        """
        Steady-state per-phase Steinmetz electromagnetic torque [N.m].

            T_em = (3/omega_e) * V_ph^2 * (R_r/s)
                     / ((R_s + R_r/s)^2 + (X_s+X_r)^2)

        Reactances scale linearly with frequency. Returns 0 at f_out=0.
        """
        f = np.clip(np.asarray(f_out, dtype=float), 0.0, self.f_max)
        if np.all(f < 1e-6):
            return np.zeros_like(np.asarray(omega_m, dtype=float))

        omega_e = 2.0 * np.pi * f
        omega_s = self.sync_speed_mech(f)
        s = np.where(omega_s > 1e-6, (omega_s - omega_m) / omega_s, 1.0)
        # avoid singularity at s=0
        s = np.where(np.abs(s) < 1e-6, np.sign(s) * 1e-6 + 1e-6, s)

        V_ll = self.output_voltage(f)
        V_ph = V_ll / np.sqrt(3.0)

        scale = f / self.f_rated
        Xs = self.X_s * scale
        Xr = self.X_r * scale

        Rr_s = self.R_r / s
        denom = (self.R_s + Rr_s) ** 2 + (Xs + Xr) ** 2
        T = np.where(
            (omega_e > 1e-6) & (denom > 1e-12),
            (3.0 / omega_e) * V_ph ** 2 * Rr_s / denom,
            0.0,
        )
        return T

    def breakdown_torque(self, f_out, n=400):
        """Maximum (pull-out) torque over the speed sweep at given f_out [N.m]."""
        omega_s = float(self.sync_speed_mech(f_out))
        if omega_s <= 1e-6:
            return 0.0
        omegas = np.linspace(0.0, omega_s * 0.999, n)
        return float(np.max(self.motor_torque(omegas, f_out)))

    # ------------------------------------------------------------------
    # Power / efficiency  (Mohan ch.5,8; Bose ch.1)
    # ------------------------------------------------------------------
    def air_gap_power(self, omega_m, f_out):
        """P_ag = T_em * omega_s  [W] (power crossing the air gap)."""
        T = self.motor_torque(omega_m, f_out)
        return T * self.sync_speed_mech(f_out)

    def mech_power(self, omega_m, f_out):
        """P_mech = T_em * omega_m  [W] (developed shaft power)."""
        return self.motor_torque(omega_m, f_out) * np.asarray(omega_m, dtype=float)

    def rotor_copper_loss(self, omega_m, f_out):
        """Rotor copper loss = s * P_ag  [W] (Krause 2013)."""
        s = self.slip(omega_m, f_out)
        return np.abs(s) * self.air_gap_power(omega_m, f_out)

    def motor_elec_power(self, omega_m, f_out):
        """
        Approximate motor electrical input power [W]:
          P_elec = P_ag + stator_copper_loss.
        Stator copper loss estimated from air-gap power and the R_s/(R_s+R_r/s)
        ratio (lumped). Bounded so P_elec >= P_mech.
        """
        Pag = self.air_gap_power(omega_m, f_out)
        s = self.slip(omega_m, f_out)
        s_safe = np.where(np.abs(s) < 1e-6, 1e-6, s)
        Rr_s = self.R_r / s_safe
        stator_frac = self.R_s / (self.R_s + np.abs(Rr_s) + 1e-12)
        P_stator_loss = stator_frac * np.abs(Pag)
        return np.abs(Pag) + P_stator_loss

    def chain_efficiency(self, omega_m, f_out):
        """
        End-to-end drive-chain efficiency  P_mech / P_grid  in (0,1).
            P_grid = P_motor_elec / (eta_rect * eta_inv)
        """
        P_mech = self.mech_power(omega_m, f_out)
        P_elec = self.motor_elec_power(omega_m, f_out)
        P_grid = P_elec / (self.eta_rect * self.eta_inv)
        eta = np.where(P_grid > 1e-6, P_mech / P_grid, 0.0)
        return np.clip(eta, 0.0, 0.9999)

    # ------------------------------------------------------------------
    # Coupled ODE system: [V_dc, omega_m]
    # ------------------------------------------------------------------
    def _rhs(self, t, y, f_cmd, T_load_fn):
        V_dc, omega_m = y
        f_out = float(f_cmd(t))
        T_load = float(T_load_fn(t))

        # --- Mechanical ODE (swing equation) ---
        T_em = float(self.motor_torque(np.array([omega_m]), f_out)[0])
        domega = (T_em - T_load - self.B * omega_m) / self.J

        # --- DC-link capacitor charge balance ---
        # Inverter input power = motor electrical power / eta_inv (>=0 motoring).
        P_elec = float(self.motor_elec_power(np.array([omega_m]), f_out)[0])
        P_inv_in = P_elec / self.eta_inv
        V_dc_safe = max(V_dc, 1.0)
        I_inv = P_inv_in / V_dc_safe
        # Stiff rectifier source: current that pulls V_dc back to nominal
        # plus the steady draw -> models a regulated bus with capacitor dynamics.
        I_rect = self.eta_rect * (I_inv + self.C_dc * (self.V_dc_nom - V_dc) / 0.02)
        dV_dc = (I_rect - I_inv - V_dc / self.R_dc) / self.C_dc

        return [dV_dc, domega]

    def simulate(self, f_set, T_load=None, V_dc0=None, omega_m0=0.0,
                 dt=0.005, duration_s=3.0):
        """
        Integrate the coupled (V_dc, omega_m) ODE with solve_ivp (LSODA).

        Parameters
        ----------
        f_set : float or callable(t)->Hz   inverter output frequency command
        T_load : float or callable(t)->N.m  load torque  (default: rated-ish)
        V_dc0  : float   initial DC-link voltage (default V_dc_nom)
        omega_m0 : float initial rotor speed [rad/s]
        dt, duration_s : output sampling and horizon

        Returns dict of time-series arrays.
        """
        if callable(f_set):
            f_cmd = f_set
        else:
            f_cmd = lambda t, _f=float(f_set): _f
        if T_load is None:
            T_load_fn = lambda t: 50.0
        elif callable(T_load):
            T_load_fn = T_load
        else:
            T_load_fn = lambda t, _T=float(T_load): _T

        if V_dc0 is None:
            V_dc0 = self.V_dc_nom

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [V_dc0, omega_m0],
            t_eval=t_eval, args=(f_cmd, T_load_fn),
            method="LSODA", rtol=1e-6, atol=1e-8, max_step=dt,
        )

        t = sol.t
        V_dc = sol.y[0]
        omega_m = sol.y[1]
        f_out = np.array([f_cmd(ti) for ti in t])

        V_out = self.output_voltage(f_out)
        omega_s = self.sync_speed_mech(f_out)
        slip = self.slip(omega_m, f_out)
        T_em = self.motor_torque(omega_m, f_out)
        P_mech = self.mech_power(omega_m, f_out)
        eta = self.chain_efficiency(omega_m, f_out)
        rpm = omega_m * 60.0 / (2.0 * np.pi)

        return {
            "t": t,
            "f_out": f_out,
            "V_out": V_out,
            "vf_ratio": self.vf_ratio(f_out),
            "V_dc": V_dc,
            "omega_m": omega_m,
            "speed_rpm": rpm,
            "omega_sync": omega_s,
            "slip": slip,
            "torque": T_em,
            "P_mech": P_mech,
            "efficiency": eta,
        }
