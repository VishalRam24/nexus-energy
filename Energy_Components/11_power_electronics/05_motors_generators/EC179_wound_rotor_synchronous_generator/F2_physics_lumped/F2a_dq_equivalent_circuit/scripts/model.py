"""
EC179 -- Wound Rotor Synchronous Generator -- F2a dq-frame Physics-Lumped Model

Physics-lumped synchronous-machine model in the rotor dq (Park) reference frame
with DC field excitation and rotor swing dynamics integrated by scipy.solve_ivp.

Core steady-state relations (per-unit on machine MVA base):
    Internal EMF behind synchronous reactance from field excitation:
        Ef = kf * If                              (linear / unsaturated air-gap line)
    Power-angle (active power) characteristic, round-rotor:
        P = (Ef * Vt / Xs) * sin(delta)           (Kundur 1994, eq. 3.105;
                                                    Fitzgerald 2003, Ch. 5)
    Reactive power vs excitation (over/under-excited):
        Q = (Ef * Vt / Xs) * cos(delta) - Vt^2 / Xs
        Ef > Vt  ->  over-excited  ->  Q > 0  (delivers VARs, lagging)
        Ef < Vt  ->  under-excited ->  Q < 0  (absorbs VARs, leading)
    Salient-pole reluctance-power extension (optional):
        P = (Ef*Vt/Xd) sin(delta) + (Vt^2/2)(1/Xq - 1/Xd) sin(2 delta)

Rotor swing equation (electromechanical dynamics, per-unit, Kundur eq. 3.209):
    d(delta)/dt = omega_0 * (omega_pu - 1)
    2H * d(omega_pu)/dt = Pm - Pe - D*(omega_pu - 1)
        with omega_0 = 2*pi*f  (electrical rad/s) and torque ~ power at omega~1.

This is the mechanical-input / electrical-output (generator) convention:
    J * d(omega_m)/dt = T_mech - T_elec    <=>    2H d(omega_pu)/dt = Pm - Pe - D dw.

AVR field-control loop (IEEE Type DC1A-like, lumped first order):
    Td0' * dEf/dt = Ka*(Vref - Vt) - Ef        (drives terminal voltage to Vref)

References:
    Kundur, P. (1994). Power System Stability and Control, McGraw-Hill,
        Ch. 3 (synchronous machine), Ch. 5 (excitation systems), Ch. 13 (swing eq).
    Fitzgerald, A.E., Kingsley, C., Umans, S.D. (2003). Electric Machinery,
        6th ed., McGraw-Hill, Ch. 5 (synchronous machines, capability curve).
    Boldea, I. (2015). Synchronous Generators, 2nd ed., CRC Press.
"""

import numpy as np
from scipy.integrate import solve_ivp


class WRSyncGenF2a:
    """Wound-rotor synchronous generator: dq-frame + field excitation + swing ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_base = u["S_base_VA"]["value"]            # VA
        self.V_LL = u["V_terminal_LL"]["value"]          # V (line-line, rated -> 1.0 pu)
        self.pf_rated = u["pf_rated"]["value"]
        self.f = u["frequency_Hz"]["value"]              # Hz
        self.poles = u["poles"]["value"]
        self.Xs = u["Xs_pu"]["value"]                    # pu (== Xd round-rotor)
        self.Xd = u["Xd_pu"]["value"]                    # pu
        self.Xq = u["Xq_pu"]["value"]                    # pu
        self.Ra = u["Ra_pu"]["value"]                    # pu
        self.H = u["H_s"]["value"]                       # s
        self.D = u["D_pu"]["value"]                      # pu damping
        self.Ef_rated = u["Ef_rated_pu"]["value"]        # pu
        self.If_rated = u["If_rated_A"]["value"]         # A
        self.kf = u["kf_EMF_per_A"]["value"]             # pu/A
        self.T_field = u["T_field_s"]["value"]           # s
        self.Ka = u["AVR_Ka"]["value"]
        self.Ta = u["AVR_Ta_s"]["value"]                 # s

        self.omega_0 = 2.0 * np.pi * self.f              # electrical rad/s base
        self.omega_m_sync = 4.0 * np.pi * self.f / self.poles  # mechanical rad/s

    # ------------------------------------------------------------------
    # Field excitation -> internal EMF
    # ------------------------------------------------------------------
    def emf_from_field(self, If_A):
        """Internal EMF Ef [pu] from DC field current via air-gap line Ef = kf*If."""
        return self.kf * np.asarray(If_A, dtype=float)

    def field_from_emf(self, Ef_pu):
        """Inverse: required field current [A] for a target EMF [pu]."""
        return np.asarray(Ef_pu, dtype=float) / self.kf

    # ------------------------------------------------------------------
    # Power-angle characteristics (steady state)
    # ------------------------------------------------------------------
    def active_power(self, Ef, delta, Vt=1.0, salient=False):
        """Active power P [pu]: round-rotor P = Ef*Vt/Xs * sin(delta).

        If salient=True, add the reluctance term using Xd, Xq.
        """
        Ef = np.asarray(Ef, dtype=float)
        delta = np.asarray(delta, dtype=float)
        if not salient:
            return (Ef * Vt / self.Xs) * np.sin(delta)
        P = (Ef * Vt / self.Xd) * np.sin(delta) \
            + (Vt**2 / 2.0) * (1.0 / self.Xq - 1.0 / self.Xd) * np.sin(2.0 * delta)
        return P

    def reactive_power(self, Ef, delta, Vt=1.0):
        """Reactive power Q [pu] = Ef*Vt/Xs * cos(delta) - Vt^2/Xs.

        Q>0 over-excited (delivers VARs, lagging pf);
        Q<0 under-excited (absorbs VARs, leading pf).
        """
        Ef = np.asarray(Ef, dtype=float)
        delta = np.asarray(delta, dtype=float)
        return (Ef * Vt / self.Xs) * np.cos(delta) - Vt**2 / self.Xs

    def pmax(self, Ef, Vt=1.0):
        """Steady-state stability limit (pull-out power) Pmax = Ef*Vt/Xs (at delta=90)."""
        return np.asarray(Ef, dtype=float) * Vt / self.Xs

    def power_angle_for_P(self, Ef, P, Vt=1.0):
        """Solve delta from P = Ef*Vt/Xs * sin(delta). Returns NaN if P > Pmax."""
        Ef = float(Ef)
        ratio = P * self.Xs / (Ef * Vt)
        if abs(ratio) > 1.0:
            return float("nan")
        return float(np.arcsin(ratio))

    # ------------------------------------------------------------------
    # Operating point solver (given P, Q find Ef, delta, If, efficiency)
    # ------------------------------------------------------------------
    def operating_point(self, P, Q, Vt=1.0, P_loss_mech_pu=0.0):
        """Solve full operating point from terminal P, Q [pu].

        Phasor: E_internal = Vt + j*Xs*I  (with Ra small, neglected for angle).
        I = (P - jQ)/Vt  (generator convention, Vt as reference phasor).
        Returns dict with Ef, delta, If, S, pf, P_elec, P_mech, efficiency.
        """
        # Armature current phasor (per-unit), Vt taken as reference (angle 0)
        I = (P - 1j * Q) / Vt
        E = Vt + 1j * self.Xs * I          # internal EMF phasor behind Xs
        Ef = abs(E)
        delta = np.angle(E)                # power (load) angle
        If = self.field_from_emf(Ef)
        S = np.hypot(P, Q)
        pf = P / S if S > 1e-12 else 1.0
        # Loss model: armature copper loss + (small) fixed mechanical/core loss
        I_mag2 = abs(I) ** 2
        P_cu = self.Ra * I_mag2            # pu armature copper loss
        P_mech = P + P_cu + P_loss_mech_pu  # mechanical input from prime mover
        eta = P / P_mech if P_mech > 1e-12 else 0.0
        return {
            "Ef_pu": Ef,
            "delta_rad": delta,
            "delta_deg": np.degrees(delta),
            "If_A": If,
            "S_pu": S,
            "pf": pf,
            "over_excited": Ef > Vt,
            "P_elec_pu": P,
            "Q_pu": Q,
            "P_mech_pu": P_mech,
            "P_cu_pu": P_cu,
            "efficiency": float(np.clip(eta, 0.0, 0.9999)),
            "stable": abs(delta) < np.pi / 2.0,
        }

    # ------------------------------------------------------------------
    # Capability curve (Fitzgerald 2003, Ch.5): armature & field limits
    # ------------------------------------------------------------------
    def capability_curve(self, Vt=1.0, n=200):
        """P-Q capability boundary [pu].

        Armature-current limit: circle of radius S_rated centred at origin -> P^2+Q^2 <= 1.
        Field-current (rotor heating) limit: circle radius Ef_rated*Vt/Xs centred at
            (0, -Vt^2/Xs)  (Fitzgerald eq. 5.61).
        Returns the inner envelope of both for Q >= -Vt^2/Xs side.
        """
        theta = np.linspace(-np.pi / 2, np.pi / 2, n)
        # Armature limit (active-power producing half)
        P_arm = np.cos(theta)
        Q_arm = np.sin(theta)
        # Field limit circle
        r_field = self.Ef_rated * Vt / self.Xs
        c_q = -Vt**2 / self.Xs
        P_field = r_field * np.cos(theta)
        Q_field = c_q + r_field * np.sin(theta)
        return {
            "P_armature": P_arm, "Q_armature": Q_arm,
            "P_field": P_field, "Q_field": Q_field,
            "field_center_Q": c_q, "field_radius": r_field,
        }

    # ------------------------------------------------------------------
    # Rotor swing dynamics (electromechanical ODE)  J*dw/dt = Tm - Te
    # ------------------------------------------------------------------
    def _swing_rhs(self, t, y, Pm, Ef, Vt, avr, Vref):
        """State y = [delta, omega_pu, Ef] (Ef state only used when avr=True)."""
        delta, omega_pu, Ef_state = y
        Ef_use = Ef_state if avr else Ef
        Pe = self.active_power(Ef_use, delta, Vt)
        Pm_t = Pm(t) if callable(Pm) else Pm
        ddelta = self.omega_0 * (omega_pu - 1.0)
        domega = (Pm_t - Pe - self.D * (omega_pu - 1.0)) / (2.0 * self.H)
        if avr:
            # terminal voltage estimate from internal EMF and angle (simplified):
            # Vt_est held at command Vt here; AVR regulates Ef toward Vref tracking.
            Vt_meas = Vt
            dEf = (self.Ka * (Vref - Vt_meas) - (Ef_state - self.Ef_rated)) / self.T_field
        else:
            dEf = 0.0
        return [ddelta, domega, dEf]

    def simulate_swing(self, Pm, Ef, Vt=1.0, delta0=None, omega0_pu=1.0,
                       duration_s=5.0, dt=0.005, avr=False, Vref=1.0,
                       P_step=None, t_step=None):
        """Integrate the rotor swing equation with scipy.solve_ivp.

        Pm        : mechanical power [pu], scalar or callable Pm(t).
        Ef        : field EMF [pu] (held constant if avr=False).
        delta0    : initial power angle [rad]; if None, steady-state for (Pm,Ef).
        P_step,t_step : optional step change in Pm at t_step (disturbance).
        Returns dict of arrays: t, delta_rad, delta_deg, omega_pu, Pe_pu, Ef_pu.
        """
        Pm0 = Pm(0.0) if callable(Pm) else Pm
        if delta0 is None:
            d0 = self.power_angle_for_P(Ef, Pm0, Vt)
            delta0 = 0.0 if np.isnan(d0) else d0

        if P_step is not None and t_step is not None:
            base = Pm
            def Pm_fun(t):
                p = base(t) if callable(base) else base
                return p + (P_step if t >= t_step else 0.0)
            Pm_eff = Pm_fun
        else:
            Pm_eff = Pm

        y0 = [delta0, omega0_pu, Ef]
        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        sol = solve_ivp(
            self._swing_rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(Pm_eff, Ef, Vt, avr, Vref),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )
        delta = sol.y[0]
        omega = sol.y[1]
        Ef_arr = sol.y[2]
        Ef_for_Pe = Ef_arr if avr else np.full_like(delta, Ef)
        Pe = self.active_power(Ef_for_Pe, delta, Vt)
        Q = self.reactive_power(Ef_for_Pe, delta, Vt)
        return {
            "t": sol.t,
            "delta_rad": delta,
            "delta_deg": np.degrees(delta),
            "omega_pu": omega,
            "Pe_pu": Pe,
            "Q_pu": Q,
            "Ef_pu": Ef_for_Pe,
            "stable": bool(np.all(np.abs(delta) < np.pi)),
            "success": sol.success,
        }
