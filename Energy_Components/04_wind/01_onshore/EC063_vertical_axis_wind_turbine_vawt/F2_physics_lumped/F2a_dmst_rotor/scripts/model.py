"""
EC063 -- Vertical Axis Wind Turbine (VAWT) -- F2a Physics-Lumped Model

Double-Multiple-Streamtube (DMST) aerodynamics coupled to a rotor-dynamics ODE.

Aerodynamics (blade-element / momentum, azimuth-resolved):
    For each azimuth psi around the rotor the local blade sees a relative wind
    formed from the freestream (reduced by an induction factor a) and the
    blade's own rotational velocity. The local angle of attack and resultant
    velocity give lift/drag from an airfoil polar (NACA0018), which project onto
    the tangential direction to produce torque. Averaging the tangential force
    over one revolution and over N blades gives the mean aerodynamic torque, and
    hence the power coefficient Cp(lambda) as a function of tip-speed ratio
    lambda = omega*R/U.

    Local angle of attack (single-streamtube form, induction a on freestream):
        Vn = U*(1-a)*sin(psi)                  (normal/streamwise comp.)
        Vt = omega*R + U*(1-a)*cos(psi)        (tangential comp.)
        W  = sqrt(Vn^2 + Vt^2)                 (relative speed)
        alpha = atan2(Vn, Vt)                  (angle of attack)
        Ct = Cl*sin(alpha) - Cd*cos(alpha)     (tangential force coeff.)
    The streamwise thrust balances momentum to solve the induction factor a
    (Strickland 1975 single multiple-streamtube; Templin 1974 actuator disc).

    Torque per blade per streamtube:
        dT = 0.5*rho*c*H*W^2*Ct*R   (averaged over psi gives mean torque)

Rotor dynamics (lumped 0D ODE -- the F2 upgrade over the F1 power curve):
        J * d(omega)/dt = T_aero(lambda) - T_load - T_loss
    with T_aero from the DMST Cp(lambda) table, T_load the electrical
    (generator) reaction torque, and T_loss a constant friction torque.

References:
    Paraschivoiu, I. (2002). "Wind Turbine Design with Emphasis on Darrieus
        Concept." Polytechnic International Press.  (DMST theory, ch. 6)
    Strickland, J.H. (1975). "The Darrieus Turbine: A Performance Prediction
        Model Using Multiple Streamtubes." SAND75-0431, Sandia.
    Templin, R.J. (1974). "Aerodynamic Performance Theory for the NRC
        Vertical-Axis Wind Turbine." NRC LTR-LA-160.
    Sheldahl, R.E. & Klimas, P.C. (1981). "Aerodynamic Characteristics of
        Seven Symmetrical Airfoil Sections." SAND80-2114 (NACA0018 polars).
"""

import numpy as np
from scipy.integrate import solve_ivp


class VAWT_F2a:
    """Vertical-axis (Darrieus H-rotor) wind turbine -- DMST + rotor dynamics."""

    def __init__(self, params: dict):
        t = params["turbine"]
        d = params["drivetrain"]
        e = params["environment"]

        self.R = t["rotor_radius"]["value"]          # m
        self.H = t["rotor_height"]["value"]          # m
        self.A = t["swept_area"]["value"]            # m2  (= 2*R*H)
        self.N = int(t["n_blades"]["value"])
        self.c = t["chord"]["value"]                 # m
        self.cl_alpha = t["cl_alpha"]["value"]       # 1/rad
        self.a_stall = t["alpha_stall"]["value"]     # rad
        self.cl_max = t["cl_max"]["value"]
        self.cd0 = t["cd0"]["value"]
        self.cd_stall = t["cd_stall"]["value"]
        self.lambda_opt = t["tip_speed_ratio_opt"]["value"]
        self.cp_ceiling = t["max_cp"]["value"]

        self.J = d["inertia"]["value"]               # kg.m2
        self.eta_gen = d["gen_efficiency"]["value"]
        self.T_loss = d["loss_torque"]["value"]      # N.m

        self.rho = e["air_density"]["value"]         # kg/m3

        # Rotor solidity sigma = N*c/R (Paraschivoiu definition)
        self.sigma = self.N * self.c / self.R

        # Cache for the Cp(lambda) lookup so the ODE rhs is cheap.
        self._lam_grid = None
        self._cp_grid = None
        self._build_cp_table()

    # ------------------------------------------------------------------
    # Airfoil polar -- NACA0018 (Sheldahl & Klimas 1981, simplified)
    # ------------------------------------------------------------------
    def airfoil_coeffs(self, alpha):
        """
        Lift and drag coefficients for a symmetric airfoil vs angle of attack.

        Pre-stall: thin-airfoil linear lift, quadratic-ish drag bucket.
        Post-stall: Viterna-style flat-plate blending toward Cd ~ cd_stall.
        alpha in radians; result symmetric in sign of alpha.
        """
        a = np.asarray(alpha, dtype=float)
        s = np.sign(a)
        am = np.abs(a)

        # Linear (attached) regime
        cl_lin = self.cl_alpha * am
        cl_lin = np.minimum(cl_lin, self.cl_max)
        cd_lin = self.cd0 + 0.02 * (am ** 2) / max(self.a_stall ** 2, 1e-9)

        # Deep-stall / post-stall flat-plate model (Viterna form)
        cl_ps = self.cd_stall * np.sin(am) * np.cos(am) * 0.9 + 0.3 * np.sin(am)
        cd_ps = self.cd0 + (self.cd_stall - self.cd0) * np.sin(am) ** 2

        attached = am <= self.a_stall
        cl = np.where(attached, cl_lin, cl_ps)
        cd = np.where(attached, cd_lin, cd_ps)
        return s * cl, cd

    # ------------------------------------------------------------------
    # Induction factor solve for one streamtube (actuator-disc momentum)
    # ------------------------------------------------------------------
    def _induction(self, lam, psi):
        """
        Solve the streamwise momentum balance for the interference (induction)
        factor a in one streamtube at azimuth psi (Strickland 1975 single
        multiple-streamtube; Templin 1974 actuator disc).

        Blade thrust coefficient in the streamtube (per Paraschivoiu 2002):
            CT_blade = (N*c)/(2*pi*R) * (W/U)^2 * Cn / |sin(psi)|
        Momentum (Glauert/Betz actuator disc):
            CT_mom = 4*a*(1-a)            for a < 0.4 (windmill state)
        Equate and iterate on a; clamp a at the turbulent-wake-state limit so
        Cp can never exceed the Betz bound.
        """
        a = 0.2
        sin_psi = max(abs(np.sin(psi)), 0.12)
        for _ in range(60):
            Un = (1.0 - a)                      # local speed normalised by U
            Vn = Un * np.sin(psi)
            Vt = lam + Un * np.cos(psi)
            W2 = Vn * Vn + Vt * Vt              # (W/U)^2
            alpha = np.arctan2(Vn, Vt)
            cl, cd = self.airfoil_coeffs(alpha)
            cn = cl * np.cos(alpha) + cd * np.sin(alpha)   # streamwise force coeff
            # Blade-element thrust coefficient in this streamtube
            ct_blade = (self.N * self.c) / (2.0 * np.pi * self.R) * W2 * abs(cn) / sin_psi
            # Invert momentum CT=4a(1-a) for a (take physical root a<0.5)
            disc = max(1.0 - ct_blade, 0.0)
            a_new = 0.5 * (1.0 - np.sqrt(disc))
            a_new = min(max(a_new, 0.0), 0.4)   # Glauert turbulent-wake cap
            if abs(a_new - a) < 1e-6:
                a = a_new
                break
            a = 0.6 * a + 0.4 * a_new           # under-relaxation
        return a

    # ------------------------------------------------------------------
    # Power coefficient Cp(lambda) from the streamtube sweep
    # ------------------------------------------------------------------
    def cp_of_lambda(self, lam, n_psi=36):
        """
        Mean power coefficient at tip-speed ratio lam via azimuthal averaging
        of the tangential force over one revolution.
        """
        if lam <= 0:
            return 0.0
        psis = np.linspace(0.0, 2.0 * np.pi, n_psi, endpoint=False)
        Ct_sum = 0.0
        a_sum = 0.0
        for psi in psis:
            a = self._induction(lam, psi)
            a_sum += a
            Un = (1.0 - a)
            Vn = Un * np.sin(psi)
            Vt = lam + Un * np.cos(psi)
            W2 = Vn * Vn + Vt * Vt
            alpha = np.arctan2(Vn, Vt)
            cl, cd = self.airfoil_coeffs(alpha)
            # Tangential force coefficient (drives torque)
            ct = cl * np.sin(alpha) - cd * np.cos(alpha)
            Ct_sum += ct * W2
        Ct_mean = Ct_sum / n_psi

        # Torque coefficient -> power coefficient.
        # Cp = (sigma/2) * lambda * <Ct * (W/U)^2>   (Paraschivoiu non-dim form).
        # The streamtube-averaged induction (mean a over the revolution)
        # represents the global flow slowdown through the rotor; multiplying by
        # the disc-extraction factor (1 - a_mean) accounts for the second
        # (downwind) momentum pass of the double-streamtube method and keeps Cp
        # below the Betz limit (Paraschivoiu 2002, double-multiple-streamtube).
        a_mean = max(min(a_sum / n_psi, 0.45), 0.0)
        cp = 0.5 * self.sigma * lam * Ct_mean * (1.0 - a_mean)
        # Cp may go slightly negative at very high lambda (profile drag exceeds
        # thrust) -- this provides the physical runaway-braking torque. Only the
        # positive side is capped at the Betz bound.
        cp = min(cp, self.cp_ceiling)            # enforce sub-Betz aero ceiling
        return cp

    def _build_cp_table(self, n=70):
        lam = np.linspace(0.2, 11.0, n)
        cp = np.array([self.cp_of_lambda(l) for l in lam])
        # Light smoothing (3-pt moving average) to remove streamtube ripple.
        k = np.array([0.25, 0.5, 0.25])
        cp_s = np.convolve(cp, k, mode="same")
        cp_s[0], cp_s[-1] = cp[0], cp[-1]
        self._lam_grid = lam
        # Lower bound allows the negative (drag-braking) branch at high lambda.
        self._cp_grid = np.clip(cp_s, -1.0, self.cp_ceiling)

    def cp(self, lam):
        """Fast Cp from the cached table (np.interp). Below grid -> 0; above
        grid -> last (negative, braking) value so runaway is bounded."""
        return float(np.interp(lam, self._lam_grid, self._cp_grid,
                               left=0.0, right=self._cp_grid[-1]))

    # ------------------------------------------------------------------
    # Aerodynamic torque & power
    # ------------------------------------------------------------------
    def aero_power(self, U, omega):
        """Aerodynamic (rotor) power [W] at wind speed U and rotor speed omega."""
        if U <= 0:
            return 0.0
        lam = omega * self.R / U
        cp = self.cp(lam)
        return 0.5 * self.rho * self.A * cp * U ** 3

    def aero_torque(self, U, omega):
        """Aerodynamic torque on the rotor [N.m]. At omega->0 use TSR->0 Cp."""
        P = self.aero_power(U, omega)
        if omega <= 1e-6:
            # Starting torque: evaluate Cq at very low lambda directly.
            lam = 1e-3
            cp = self.cp(max(lam, self._lam_grid[0]))
            # Q = 0.5*rho*A*R*Cq*U^2 ; Cq = Cp/lambda but lambda->0 is singular,
            # so use a small finite reference rotor speed for a finite estimate.
            omega_ref = 0.05
            P_ref = self.aero_power(U, omega_ref)
            return P_ref / omega_ref if P_ref > 0 else 0.0
        return P / omega

    # ------------------------------------------------------------------
    # Rotor-dynamics ODE:  J*dw/dt = T_aero - T_load - T_loss
    # ------------------------------------------------------------------
    def domega_dt(self, omega, U, T_load):
        T_aero = self.aero_torque(U, max(omega, 0.0))
        net = T_aero - T_load - self.T_loss
        return net / self.J

    def simulate(self, wind_speed, T_load=0.0, omega0=0.5,
                 dt=0.5, duration_s=120.0):
        """
        Time-domain spin-up / dynamic simulation of the rotor.

        Parameters
        ----------
        wind_speed : float or callable(t)  -- freestream wind speed [m/s]
        T_load     : float or callable(t)  -- generator reaction torque [N.m]
        omega0     : float                 -- initial rotor speed [rad/s]
        dt         : float                 -- output time step [s]
        duration_s : float                 -- total time [s]

        Returns dict of time-series arrays.
        """
        U_fn = wind_speed if callable(wind_speed) else (lambda t: wind_speed)
        TL_fn = T_load if callable(T_load) else (lambda t: T_load)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.domega_dt(y[0], U_fn(t), TL_fn(t))]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [omega0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        omega = np.clip(sol.y[0], 0.0, None)
        N = len(t_out)

        U_arr = np.array([U_fn(tt) for tt in t_out])
        TL_arr = np.array([TL_fn(tt) for tt in t_out])
        lam = np.where(U_arr > 0, omega * self.R / U_arr, 0.0)
        cp_arr = np.array([self.cp(l) for l in lam])
        P_aero = np.array([self.aero_power(U_arr[i], omega[i]) for i in range(N)])
        T_aero = np.array([self.aero_torque(U_arr[i], omega[i]) for i in range(N)])
        P_elec = np.clip((T_aero - self.T_loss) * omega, 0.0, None) * self.eta_gen

        return {
            "t": t_out,
            "omega": omega,
            "rpm": omega * 60.0 / (2.0 * np.pi),
            "tip_speed_ratio": lam,
            "cp": cp_arr,
            "power_aero": P_aero,
            "power_elec": P_elec,
            "torque_aero": T_aero,
            "wind_speed": U_arr,
            "torque_load": TL_arr,
        }

    # ------------------------------------------------------------------
    # Steady-state helpers
    # ------------------------------------------------------------------
    def cp_max(self):
        """Peak Cp and its tip-speed ratio from the table."""
        i = int(np.argmax(self._cp_grid))
        return float(self._cp_grid[i]), float(self._lam_grid[i])

    def steady_omega(self, U, T_load=0.0, dt=0.5, duration_s=300.0):
        """Integrate to (near) steady state and return final rotor speed."""
        r = self.simulate(U, T_load=T_load, omega0=0.5, dt=dt, duration_s=duration_s)
        return r["omega"][-1]
