"""
EC056 -- Linear Fresnel CSP -- F2a Physics-Lumped Receiver Thermal-Balance ODE

Physics-lumped (0D) first-principles model of a Linear Fresnel collector:
rows of nearly-flat mirrors focus DNI onto a single elevated linear receiver.
The receiver wall is treated as ONE lumped thermal node whose temperature
evolves by an energy-balance ODE (integrated with scipy.solve_ivp); the HTF /
steam outlet temperature follows from the enthalpy carried away by the flow.

Optical chain (per Mills 2004 / Häberle 2002, see EC056 F1b):
    eta_opt(theta_L, theta_T) = rho * gamma * tau_sec * alpha
                                * IAM_L(theta_L) * IAM_T(theta_T) * f_end(theta_L)
    IAM_L(theta_L) = cos(theta_L)                 longitudinal (along receiver axis)
    IAM_T(theta_T) = 1 - b_T * theta_T^2          transversal (across mirror field)
    f_end(theta_L) = 1 - f * tan(theta_L) / L     finite-length end loss

Absorbed solar power on the receiver per unit length:
    q_abs = DNI * W_mirror * eta_opt                                   [W/m]

Lumped receiver-wall energy balance (1 ODE, the F2 upgrade over F1):
    (m_wall*cp_wall) dT_w/dt = q_abs - q_conv - q_rad - q_htf          [W/m]
        q_conv = pi*D_o * h_conv * (T_w - T_amb)        ambient convection
        q_rad  = pi*D_o * eps * sigma * (T_w^4 - T_sky^4)   radiation  (~T^4)
        q_htf  = pi*D_i * h_htf * (T_w - T_htf_mean)    wall->HTF transfer

HTF / steam enthalpy balance gives the outlet temperature from the heat
actually delivered to the fluid over the whole receiver length:
    Q_to_fluid = q_htf * L            [W]
    T_out = T_in + Q_to_fluid / (mdot * cp_htf)

Energy conservation is exact in the steady limit:
    q_abs = q_conv + q_rad + q_htf      (dT_w/dt -> 0)
At DNI = 0 there is no absorbed power, the wall cools toward ambient/HTF and
useful output -> 0 (P = 0 at DNI = 0). Radiative loss scales as T^4.

References:
    Mills, D. (2004). "Advances in solar thermal electricity technology."
        Solar Energy 76 (1-3), 19-31.   (Compact Linear Fresnel Reflector / CLFR)
    Novatec Solar / Häberle, A. et al. (2002). "The Solarmundo line focussing
        Fresnel collector." Eurosun 2002 Proceedings.
    Zhu, G. et al. (2014). "History, current state, and future of linear Fresnel
        concentrating solar collectors." Solar Energy 103, 639-652.
    Forristall, R. (2003). "Heat Transfer Analysis and Modeling of a Parabolic
        Trough Solar Receiver." NREL/TP-550-34169.
    Incropera, F. (2011). Fundamentals of Heat and Mass Transfer, 7th ed.
"""

import numpy as np
from scipy.integrate import solve_ivp


class LinearFresnelF2a:
    """Linear Fresnel CSP -- lumped receiver-wall thermal ODE + HTF enthalpy."""

    SIGMA = 5.670374419e-8   # Stefan-Boltzmann constant [W/m2K4]

    def __init__(self, params: dict):
        u = params["unit"]
        # Geometry / optics
        self.W = u["total_mirror_width"]["value"]        # m
        self.L = u["L_collector"]["value"]               # m
        self.focal = u["focal_length"]["value"]          # m
        self.D_o = u["D_abs_outer"]["value"]             # m
        self.D_i = u["D_abs_inner"]["value"]             # m

        self.rho = u["rho_mirror"]["value"]
        self.gamma = u["intercept_factor"]["value"]
        self.tau_sec = u["tau_secondary"]["value"]
        self.alpha = u["abs_absorptance"]["value"]
        self.b_T = u["b_T"]["value"]

        # Thermal-loss properties
        self.eps = u["eps_abs"]["value"]
        self.h_conv = u["h_conv_cavity"]["value"]        # W/m2K
        self.T_sky_off = u["T_sky_offset"]["value"]      # K

        # Wall node + HTF coupling
        self.h_htf = u["h_htf"]["value"]                 # W/m2K
        self.m_wall = u["m_wall_per_m"]["value"]         # kg/m
        self.cp_wall = u["cp_wall"]["value"]             # J/kgK
        self.mdot = u["mdot_htf"]["value"]               # kg/s
        self.cp_htf = u["cp_htf"]["value"]               # J/kgK
        self.eta_pb = u["eta_powerblock"]["value"]

        self.A_aperture = self.W * self.L                # m2
        self.C_wall = self.m_wall * self.cp_wall         # J/(K.m) heat capacity/length

    # ------------------------------------------------------------------
    # Optics
    # ------------------------------------------------------------------
    def iam_longitudinal(self, theta_L_deg):
        """Longitudinal IAM = cos(theta_L). Clipped to [0,1]."""
        return float(np.clip(np.cos(np.radians(theta_L_deg)), 0.0, 1.0))

    def iam_transversal(self, theta_T_deg):
        """Transversal IAM = 1 - b_T*theta_T^2 (Haberle 2002). 0 beyond 60 deg."""
        tt = float(theta_T_deg)
        val = 1.0 - self.b_T * tt * tt
        if tt >= 60.0:
            val = 0.0
        return float(np.clip(val, 0.0, 1.0))

    def end_loss_factor(self, theta_L_deg):
        """Finite-length end loss f_end = 1 - focal*tan(theta_L)/L."""
        tl = float(theta_L_deg)
        tan_t = np.tan(np.radians(tl)) if tl < 85.0 else 100.0
        return float(np.clip(1.0 - self.focal * tan_t / self.L, 0.0, 1.0))

    def optical_efficiency(self, theta_L_deg, theta_T_deg):
        """Total peak-corrected optical efficiency (dimensionless, in [0,1])."""
        eta = (self.rho * self.gamma * self.tau_sec * self.alpha
               * self.iam_longitudinal(theta_L_deg)
               * self.iam_transversal(theta_T_deg)
               * self.end_loss_factor(theta_L_deg))
        return float(np.clip(eta, 0.0, 1.0))

    def absorbed_power_per_m(self, dni, theta_L_deg, theta_T_deg):
        """Solar power absorbed at the receiver per metre of length [W/m]."""
        eta = self.optical_efficiency(theta_L_deg, theta_T_deg)
        return max(0.0, float(dni) * self.W * eta)

    # ------------------------------------------------------------------
    # Loss terms (per metre of receiver) -- functions of wall temperature
    # ------------------------------------------------------------------
    def q_conv_per_m(self, T_w_K, T_amb_K):
        """Convective loss to ambient [W/m]."""
        return np.pi * self.D_o * self.h_conv * (T_w_K - T_amb_K)

    def q_rad_per_m(self, T_w_K, T_amb_K):
        """Radiative loss [W/m], proportional to (T_w^4 - T_sky^4)."""
        T_sky = T_amb_K - self.T_sky_off
        return np.pi * self.D_o * self.eps * self.SIGMA * (T_w_K**4 - T_sky**4)

    def q_htf_per_m(self, T_w_K, T_htf_mean_K):
        """Heat conducted from wall into the HTF/steam [W/m]."""
        return np.pi * self.D_i * self.h_htf * (T_w_K - T_htf_mean_K)

    # ------------------------------------------------------------------
    # HTF outlet temperature from delivered heat (enthalpy balance)
    # ------------------------------------------------------------------
    def htf_outlet_temp(self, q_htf_per_m_val, T_in_K):
        """
        Outlet HTF temperature from the heat delivered over the full length.
        Q_fluid = q_htf_per_m * L ; T_out = T_in + Q_fluid/(mdot*cp).
        Bounded below at T_in only when heat is non-negative; the wall may
        also cool the fluid (q<0) which lowers T_out, conserving energy.
        """
        Q_fluid = q_htf_per_m_val * self.L
        return T_in_K + Q_fluid / (self.mdot * self.cp_htf)

    def htf_mean_temp(self, q_htf_per_m_val, T_in_K):
        """Mean HTF temperature along the tube (arithmetic in-out mean)."""
        T_out = self.htf_outlet_temp(q_htf_per_m_val, T_in_K)
        return 0.5 * (T_in_K + T_out)

    # ------------------------------------------------------------------
    # Lumped wall energy-balance derivative
    # ------------------------------------------------------------------
    def dTw_dt(self, T_w_K, dni, theta_L_deg, theta_T_deg, T_amb_K, T_in_K):
        """
        Receiver-wall temperature rate [K/s].
        C_wall dT_w/dt = q_abs - q_conv - q_rad - q_htf   (all per metre).
        The HTF mean temperature is found from the current wall-to-fluid heat
        flux (implicit-consistent within the step via a light fixed-point).
        """
        q_abs = self.absorbed_power_per_m(dni, theta_L_deg, theta_T_deg)

        # Light fixed-point for the wall->HTF flux (couples T_htf_mean to q_htf)
        T_htf_mean = T_in_K
        for _ in range(3):
            q_htf = self.q_htf_per_m(T_w_K, T_htf_mean)
            T_htf_mean = self.htf_mean_temp(q_htf, T_in_K)

        q_conv = self.q_conv_per_m(T_w_K, T_amb_K)
        q_rad = self.q_rad_per_m(T_w_K, T_amb_K)
        return (q_abs - q_conv - q_rad - q_htf) / self.C_wall

    # ------------------------------------------------------------------
    # Diagnostics at a given wall temperature (steady-evaluation helper)
    # ------------------------------------------------------------------
    def _diagnostics(self, T_w_K, dni, theta_L_deg, theta_T_deg, T_amb_K, T_in_K):
        q_abs = self.absorbed_power_per_m(dni, theta_L_deg, theta_T_deg)
        T_htf_mean = T_in_K
        for _ in range(3):
            q_htf = self.q_htf_per_m(T_w_K, T_htf_mean)
            T_htf_mean = self.htf_mean_temp(q_htf, T_in_K)
        q_conv = self.q_conv_per_m(T_w_K, T_amb_K)
        q_rad = self.q_rad_per_m(T_w_K, T_amb_K)
        Q_fluid = q_htf * self.L                              # W
        T_out = self.htf_outlet_temp(q_htf, T_in_K)
        Q_incident = dni * self.A_aperture                    # W
        eta_thermal = Q_fluid / Q_incident if Q_incident > 1.0 else 0.0
        P_elec = max(0.0, Q_fluid) * self.eta_pb
        return {
            "q_abs_per_m": q_abs,
            "q_conv_per_m": q_conv,
            "q_rad_per_m": q_rad,
            "q_htf_per_m": q_htf,
            "Q_to_fluid_W": Q_fluid,
            "T_htf_out_K": T_out,
            "eta_thermal": float(np.clip(eta_thermal, -1.0, 1.0)),
            "P_electric_W": P_elec,
            "eta_optical": self.optical_efficiency(theta_L_deg, theta_T_deg),
        }

    # ------------------------------------------------------------------
    # Steady-state wall temperature (root of the energy balance)
    # ------------------------------------------------------------------
    def steady_wall_temp(self, dni, theta_L_deg, theta_T_deg, T_amb_K, T_in_K,
                         T_guess=None):
        """
        Long-time wall temperature by integrating to steady state.
        Returns (T_w_steady_K, diagnostics_dict).
        """
        if T_guess is None:
            T_guess = T_in_K + 30.0
        res = self.simulate(dni, theta_L_deg, theta_T_deg, T_amb_K, T_in_K,
                            T_w0_K=T_guess, dt=5.0, duration_s=3000.0)
        Tw = res["T_wall_K"][-1]
        diag = self._diagnostics(Tw, dni, theta_L_deg, theta_T_deg, T_amb_K, T_in_K)
        return Tw, diag

    # ------------------------------------------------------------------
    # Time-domain simulation (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def simulate(self, dni, theta_L_deg, theta_T_deg, T_amb_K, T_in_K,
                 T_w0_K=None, dt=5.0, duration_s=1800.0):
        """
        Integrate the lumped receiver-wall ODE in time.

        Any of dni, theta_L_deg, theta_T_deg, T_amb_K, T_in_K may be a scalar
        or a callable(t) for time-varying forcing.

        Returns dict of time series:
            t, T_wall_K, T_htf_out_K, q_abs_per_m, q_conv_per_m, q_rad_per_m,
            q_htf_per_m, Q_to_fluid_W, eta_thermal, eta_optical, P_electric_W
        """
        def _f(x):
            return x if callable(x) else (lambda t: x)
        f_dni, f_tL, f_tT = _f(dni), _f(theta_L_deg), _f(theta_T_deg)
        f_amb, f_in = _f(T_amb_K), _f(T_in_K)

        if T_w0_K is None:
            T_w0_K = f_in(0.0) + 20.0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTw_dt(y[0], f_dni(t), f_tL(t), f_tT(t),
                                f_amb(t), f_in(t))]

        sol = solve_ivp(rhs, (0.0, duration_s), [T_w0_K], t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-7, max_step=dt)

        t_out = sol.t
        Tw = sol.y[0]
        N = len(t_out)

        T_htf_out = np.zeros(N)
        q_abs = np.zeros(N); q_conv = np.zeros(N); q_rad = np.zeros(N)
        q_htf = np.zeros(N); Q_fluid = np.zeros(N)
        eta_th = np.zeros(N); eta_op = np.zeros(N); P_el = np.zeros(N)

        for i in range(N):
            d = self._diagnostics(Tw[i], f_dni(t_out[i]), f_tL(t_out[i]),
                                   f_tT(t_out[i]), f_amb(t_out[i]), f_in(t_out[i]))
            T_htf_out[i] = d["T_htf_out_K"]
            q_abs[i] = d["q_abs_per_m"]; q_conv[i] = d["q_conv_per_m"]
            q_rad[i] = d["q_rad_per_m"]; q_htf[i] = d["q_htf_per_m"]
            Q_fluid[i] = d["Q_to_fluid_W"]; eta_th[i] = d["eta_thermal"]
            eta_op[i] = d["eta_optical"]; P_el[i] = d["P_electric_W"]

        return {
            "t": t_out,
            "T_wall_K": Tw,
            "T_htf_out_K": T_htf_out,
            "q_abs_per_m": q_abs,
            "q_conv_per_m": q_conv,
            "q_rad_per_m": q_rad,
            "q_htf_per_m": q_htf,
            "Q_to_fluid_W": Q_fluid,
            "eta_thermal": eta_th,
            "eta_optical": eta_op,
            "P_electric_W": P_el,
        }
