"""
EC064 -- Small / Micro Wind Turbine -- F2a BEM Rotor-Dynamics

Physics-lumped first-principles model of a small 3-blade HAWT driving a
direct-drive PMSG.  Couples a BEM-derived power coefficient Cp(lambda, beta)
to a single-DOF rotor-dynamics ODE with a generator (electromagnetic) load.

Aerodynamics (Blade-Element-Momentum reduced to a 0D Cp law):
    P_aero = 1/2 * rho * A * Cp(lambda, beta) * U^3
    T_aero = P_aero / omega
    lambda = omega * R / U                         (tip-speed ratio)
Cp(lambda, beta) closed form (Heier 2014; widely used reduction of BEM):
    Cp = c1 (c2/li - c3*beta - c4) exp(-c5/li) + c6*lambda
    1/li = 1/(lambda + 0.08 beta) - 0.035/(beta^3 + 1)
Cp is scaled so its peak equals Cp_max (small-rotor low-Re value, < Betz 0.593).

Rotor-dynamics ODE (Newton's 2nd law for rotation, Manwell 2009 Ch.4;
Burton 2011 drivetrain):
    J * domega/dt = T_aero(omega, U) - T_gen(omega) - T_loss(omega)
    T_loss = b*omega + T_static*tanh(omega/eps)    (windage + bearing friction)

Generator load (PMSG, optimal torque / TSR-tracking below rated;
constant-power feathering above rated, Bossanyi optimal control):
    below rated:  T_gen = K_opt * omega^2          (tracks lambda_opt)
    above rated:  T_gen = P_rated / (eta * omega)   (power-limited)

Small-turbine specifics captured:
  * low Reynolds number -> reduced Cp_max (~0.40 vs 0.45-0.50 utility).
  * passive furling / over-speed: effective swept area is shed between
    v_furl and v_cut_out, and the rotor is parked above cut-out.
  * higher turbulence intensity (IEC 61400-2) supported via a turbulent
    wind callable.

Conservation / bounds enforced:
  * Cp <= Betz limit 16/27 = 0.5926 at all (lambda, beta).
  * P proportional to U^3 at fixed Cp (verified by tests).
  * Mechanical energy balance: integral(J*omega*domega) = integral of net
    shaft power, i.e. dE_kin/dt = P_aero - P_gen - P_loss.

References:
    Manwell, McGowan & Rogers (2009). Wind Energy Explained, 2nd ed., Wiley.
    Burton, Jenkins, Sharpe & Bossanyi (2011). Wind Energy Handbook, 2nd ed.
    Heier, S. (2014). Grid Integration of Wind Energy, 3rd ed., Wiley.
    Wood, D. (2011). Small Wind Turbines: Analysis, Design, and Application.
    IEC 61400-2:2013, Small wind turbines.
"""

import numpy as np
from scipy.integrate import solve_ivp

BETZ_LIMIT = 16.0 / 27.0  # 0.59259...


class SmallWindTurbineF2a:
    """Small HAWT -- BEM Cp(lambda,beta) coupled to rotor-dynamics ODE + PMSG."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.D = u["D"]["value"]
        self.R = self.D / 2.0
        self.A = np.pi / 4.0 * self.D ** 2
        self.rho = u["rho"]["value"]
        self.P_rated = u["P_rated"]["value"]

        self.Cp_max = u["Cp_max"]["value"]
        self.lambda_opt = u["lambda_opt"]["value"]
        self.c1 = u["c1"]["value"]
        self.c2 = u["c2"]["value"]
        self.c3 = u["c3"]["value"]
        self.c4 = u["c4"]["value"]
        self.c5 = u["c5"]["value"]
        self.c6 = u["c6"]["value"]

        self.J = u["J_rotor"]["value"]
        self.b_loss = u["T_loss_coeff"]["value"]
        self.T_static = u["T_loss_static"]["value"]

        self.kt_gen = u["kt_gen"]["value"]
        self.eta_load = u["kv_load_gain"]["value"]
        self.K_opt = u["K_opt_torque"]["value"]
        self.omega_rated = u["omega_rated"]["value"]

        self.v_cut_in = u["v_cut_in"]["value"]
        self.v_rated = u["v_rated"]["value"]
        self.v_furl = u["v_furl"]["value"]
        self.v_cut_out = u["v_cut_out"]["value"]
        self.furl_exp = u["furl_exponent"]["value"]
        self.TI_ref = u["TI_ref"]["value"]

        # Normalise the raw Heier Cp form so its analytic peak equals Cp_max.
        self._cp_scale = self._calibrate_cp_scale()

    # ------------------------------------------------------------------
    # Cp(lambda, beta)  --  BEM-reduced closed form (Heier 2014)
    # ------------------------------------------------------------------
    def _cp_raw(self, lam, beta_deg):
        lam = np.asarray(lam, dtype=float)
        beta = float(beta_deg)
        # Guard the inverse-lambda term against division by zero.
        lam_safe = np.where(np.abs(lam) < 1e-6, 1e-6, lam)
        inv_li = 1.0 / (lam_safe + 0.08 * beta) - 0.035 / (beta ** 3 + 1.0)
        # inv_li can be <=0 at very low lambda -> Cp clamped to 0 there.
        with np.errstate(over="ignore", invalid="ignore"):
            li = 1.0 / inv_li
            cp = (self.c1 * (self.c2 / li - self.c3 * beta - self.c4)
                  * np.exp(-self.c5 / li) + self.c6 * lam_safe)
        cp = np.where(np.isfinite(cp), cp, 0.0)
        cp = np.maximum(cp, 0.0)
        return cp

    def _calibrate_cp_scale(self):
        lam = np.linspace(0.5, 15.0, 2000)
        raw = self._cp_raw(lam, 0.0)
        peak = float(np.max(raw))
        if peak <= 0:
            return 1.0
        return self.Cp_max / peak

    def Cp(self, lam, beta_deg=0.0):
        """Power coefficient. Scaled to Cp_max, hard-capped at the Betz limit."""
        cp = self._cp_scale * self._cp_raw(lam, beta_deg)
        cp = np.minimum(cp, BETZ_LIMIT)
        return np.maximum(cp, 0.0)

    # ------------------------------------------------------------------
    # Furling / over-speed area factor (small-turbine specific)
    # ------------------------------------------------------------------
    def furl_factor(self, U):
        """Effective swept-area fraction (1 below furl, ->0 above cut-out)."""
        U = float(U)
        if U <= self.v_furl:
            return 1.0
        if U >= self.v_cut_out:
            return 0.0
        x = (U - self.v_furl) / (self.v_cut_out - self.v_furl)
        return float(max(0.0, 1.0 - x ** self.furl_exp))

    # ------------------------------------------------------------------
    # Aerodynamic torque
    # ------------------------------------------------------------------
    def aero_power(self, omega, U, beta_deg=0.0):
        """Aerodynamic (shaft) power [W]."""
        U = float(U)
        if U < self.v_cut_in or U >= self.v_cut_out or omega <= 0:
            # Still allow a small starting torque region handled in aero_torque.
            if U < self.v_cut_in or U >= self.v_cut_out:
                return 0.0
        lam = omega * self.R / max(U, 1e-6)
        cp = float(self.Cp(lam, beta_deg))
        ff = self.furl_factor(U)
        P = 0.5 * self.rho * self.A * ff * cp * U ** 3
        return max(P, 0.0)

    def aero_torque(self, omega, U, beta_deg=0.0):
        """Aerodynamic torque on the rotor shaft [N.m]."""
        U = float(U)
        if U < self.v_cut_in or U >= self.v_cut_out:
            return 0.0
        if omega <= 1e-3:
            # Near standstill: use a small-omega expansion so the rotor can
            # spin up. T = P/omega is indeterminate at omega->0; evaluate Cp
            # at a tiny lambda and take the limiting starting torque.
            lam = 1e-3 * self.R / max(U, 1e-6)
            cp = float(self.Cp(max(lam, 0.05), beta_deg))
            ff = self.furl_factor(U)
            P = 0.5 * self.rho * self.A * ff * cp * U ** 3
            return P / max(omega, 1e-3)
        P = self.aero_power(omega, U, beta_deg)
        return P / omega

    # ------------------------------------------------------------------
    # Generator (PMSG) electromagnetic load torque
    # ------------------------------------------------------------------
    def gen_torque(self, omega):
        """Generator load torque [N.m]: K*omega^2 below rated, P-limited above."""
        if omega <= 0:
            return 0.0
        T_track = self.K_opt * omega ** 2
        # Power-limiting cap above rated speed (constant-power feathering).
        T_cap = self.P_rated / (self.eta_load * max(omega, 1e-3))
        if omega > self.omega_rated:
            return min(T_track, T_cap)
        return T_track

    def gen_power_elec(self, omega):
        """Electrical power delivered to the load [W]."""
        return self.eta_load * self.gen_torque(omega) * max(omega, 0.0)

    # ------------------------------------------------------------------
    # Loss torque (bearing friction + windage)
    # ------------------------------------------------------------------
    def loss_torque(self, omega):
        """Drivetrain loss torque [N.m]."""
        return self.b_loss * omega + self.T_static * np.tanh(omega / 0.5)

    # ------------------------------------------------------------------
    # Rotor-dynamics ODE derivative
    # ------------------------------------------------------------------
    def domega_dt(self, omega, U, beta_deg=0.0):
        """Angular-acceleration [rad/s^2] = (T_aero - T_gen - T_loss)/J."""
        omega = max(omega, 0.0)
        T_a = self.aero_torque(omega, U, beta_deg)
        T_g = self.gen_torque(omega)
        T_l = self.loss_torque(omega)
        return (T_a - T_g - T_l) / self.J

    # ------------------------------------------------------------------
    # Steady-state operating point (root of torque balance) -- diagnostic
    # ------------------------------------------------------------------
    def steady_state(self, U, beta_deg=0.0, omega_guess=None):
        """Find omega where T_aero = T_gen + T_loss via bisection."""
        if U < self.v_cut_in or U >= self.v_cut_out:
            return 0.0
        lo, hi = 1e-3, 40.0

        def net(w):
            return (self.aero_torque(w, U, beta_deg)
                    - self.gen_torque(w) - self.loss_torque(w))

        f_lo, f_hi = net(lo), net(hi)
        if f_lo * f_hi > 0:
            # No sign change: pick the omega that maximises aero power.
            ws = np.linspace(lo, hi, 400)
            nets = np.array([net(w) for w in ws])
            return float(ws[np.argmin(np.abs(nets))])
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = net(mid)
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------
    # Time-domain simulation (rotor spin-up / transient)
    # ------------------------------------------------------------------
    def simulate(self, wind_speed, omega0=None, beta_deg=0.0,
                 dt=0.05, duration_s=60.0, TI=0.0, seed=0):
        """
        Integrate the rotor-dynamics ODE.

        Parameters
        ----------
        wind_speed : float or callable(t)->U   [m/s]
        omega0     : initial rotor speed [rad/s] (default: 1 rad/s spin-up)
        beta_deg   : blade pitch angle [deg]
        dt         : output time step [s]
        duration_s : total time [s]
        TI         : turbulence intensity; if >0 and wind_speed is scalar a
                     turbulent series U(t)=Umean*(1+TI*noise) is synthesised.
        seed       : RNG seed for the turbulent wind.

        Returns
        -------
        dict of time-series arrays: t, omega, rpm, tsr, Cp, P_aero, P_elec,
              T_aero, T_gen, T_loss, efficiency, wind_speed.
        """
        if callable(wind_speed):
            U_of_t = wind_speed
            U_mean = float(wind_speed(0.0))
        elif TI and TI > 0:
            rng = np.random.default_rng(seed)
            n_grid = max(int(duration_s) + 2, 4)
            tgrid = np.linspace(0.0, duration_s, n_grid)
            gust = rng.standard_normal(n_grid)
            # smooth (low-pass) the gust series a little
            gust = np.convolve(gust, np.ones(3) / 3.0, mode="same")
            Useries = float(wind_speed) * (1.0 + TI * gust)
            Useries = np.clip(Useries, 0.1, None)
            U_of_t = lambda t: float(np.interp(t, tgrid, Useries))
            U_mean = float(wind_speed)
        else:
            Uc = float(wind_speed)
            U_of_t = lambda t: Uc
            U_mean = Uc

        if omega0 is None:
            omega0 = 1.0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            w = max(y[0], 0.0)
            return [self.domega_dt(w, U_of_t(t), beta_deg)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [omega0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9,
            max_step=max(dt, 1e-3),
        )

        t = sol.t
        omega = np.maximum(sol.y[0], 0.0)
        N = len(t)

        U = np.array([U_of_t(ti) for ti in t])
        tsr = omega * self.R / np.maximum(U, 1e-6)
        cp = np.array([float(self.Cp(l, beta_deg)) for l in tsr])
        P_aero = np.array([self.aero_power(omega[i], U[i], beta_deg) for i in range(N)])
        P_elec = np.array([self.gen_power_elec(omega[i]) for i in range(N)])
        T_aero = np.array([self.aero_torque(omega[i], U[i], beta_deg) for i in range(N)])
        T_gen = np.array([self.gen_torque(omega[i]) for i in range(N)])
        T_loss = np.array([self.loss_torque(omega[i]) for i in range(N)])
        P_wind = 0.5 * self.rho * self.A * U ** 3
        eff = np.where(P_wind > 1e-9, P_elec / P_wind, 0.0)

        return {
            "t": t,
            "omega": omega,
            "rpm": omega * 60.0 / (2.0 * np.pi),
            "tsr": tsr,
            "Cp": cp,
            "P_aero": P_aero,
            "P_elec": P_elec,
            "T_aero": T_aero,
            "T_gen": T_gen,
            "T_loss": T_loss,
            "efficiency": eff,
            "wind_speed": U,
        }
