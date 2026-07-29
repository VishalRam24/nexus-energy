"""
EC173 -- Distribution Transformer -- F2a Equivalent-Circuit + Lumped Thermal ODE

Physics-lumped (0D) first-principles model of an oil-immersed distribution
transformer. Two coupled blocks:

(1) STEADY-STATE EQUIVALENT CIRCUIT (per-unit, referred to one side)
    ----------------------------------------------------------------------
    Standard single-phase-equivalent T/L-model of a 2-winding transformer
    (Fitzgerald, Kingsley & Umans, "Electric Machinery", 6th ed., 2003, Ch. 2):

         o---[ R_eq + jX_eq ]---+--------------o   secondary
                                |
                       [ R_c ] || [ jX_m ]   (shunt excitation branch)
                                |
         o----------------------+--------------o

      * Series branch  R_eq + jX_eq : winding resistance (copper / I^2R loss)
        and leakage reactance.  R_eq is fixed by the rated load loss P_k,
        X_eq by the short-circuit (impedance) voltage u_k.
      * Shunt branch   R_c || jX_m  : R_c carries the core (no-load / hysteresis
        + eddy) loss; X_m carries the magnetizing current. Fixed by P_0 and i_0.

    From this circuit we get, per unit:
      - Voltage regulation  VR = (V_noload - V_full) / V_full  (approx formula)
      - Copper (load) loss   P_cu = P_k * PLR^2 * R(T)      [temperature-corrected]
      - Core  (no-load) loss P_core = P_0 * (V/V_rated)^n_B [Steinmetz, ~V^1.8]
      - Efficiency  eta = P_out / (P_out + P_cu + P_core)
        peaks at PLR_opt = sqrt(P_0 / P_k) (core loss = copper loss),
        which for distribution units is a PARTIAL load (typically 30-50%).

(2) LUMPED THERMAL ODE (top-oil + hot-spot), integrated with scipy.solve_ivp
    ----------------------------------------------------------------------
    IEEE Std C57.91-2011, "IEEE Guide for Loading Mineral-Oil-Immersed
    Transformers", Clause 7 / Annex G exponential model:

      Top-oil rise over ambient:
        tau_oil * d(theta_oil)/dt = theta_oil_ult(K) - theta_oil
        theta_oil_ult = theta_oil_rated * ((K^2 * R_pk + 1)/(R_pk + 1))^n_oil
        where K = PLR (per-unit load), R_pk = P_k/P_0 (load/no-load loss ratio)

      Hot-spot gradient over top-oil:
        tau_w * d(theta_hs)/dt = theta_hs_ult(K) - theta_hs
        theta_hs_ult = theta_hs_grad_rated * K^(2m),  m ~ 0.8-1.0

      Winding hot-spot temperature:
        T_hs = T_ambient + theta_oil + theta_hs

    The two-exponential form (slow oil tau ~ hours, fast winding tau ~ minutes)
    is the standard C57.91 lumped representation. Energy balance: at steady
    state the integrated loss heat input equals the heat rejected to ambient
    (verified in the test suite via the ultimate-rise consistency check).

References:
    IEEE Std C57.91-2011, Clause 7 & Annex G.
    Fitzgerald, Kingsley & Umans (2003), Electric Machinery, 6th ed., McGraw-Hill, Ch. 2.
    IEC 60076-7:2018, Loading guide for oil-immersed power transformers.
    Kulkarni & Khaparde (2004), Transformer Engineering, CRC Press.
"""

import numpy as np
from scipy.integrate import solve_ivp


class DistributionTransformerF2a:
    """Equivalent-circuit + lumped thermal ODE distribution transformer."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_rated = u["S_rated_kVA"]["value"] * 1000.0     # VA
        self.V_hv = u["V_hv"]["value"]                        # V
        self.V_lv = u["V_lv"]["value"]                        # V
        self.P_0 = u["P_no_load_W"]["value"]                  # W (core/no-load)
        self.P_k = u["P_load_loss_W"]["value"]                # W (copper at rated, T_ref)
        self.u_k = u["u_k_pu"]["value"]                       # pu impedance voltage
        self.i_0 = u["i_0_pu"]["value"]                       # pu no-load current
        self.T_ref = u["T_ref_winding"]["value"]              # degC
        self.alpha_Cu = u["alpha_Cu"]["value"]                # 1/K
        self.n_B = u["n_B"]["value"]                          # core loss V exponent

        # Thermal
        self.theta_oil_rated = u["theta_oil_rise_rated_K"]["value"]   # K
        self.theta_hs_grad_rated = u["theta_hs_grad_rated_K"]["value"]  # K
        self.tau_oil = u["tau_oil_min"]["value"] * 60.0       # s
        self.tau_w = u["tau_winding_min"]["value"] * 60.0     # s
        self.n_oil = u["n_oil"]["value"]
        self.m_oil = u["m_oil_kg"]["value"]
        self.m_core = u["m_core_coil_kg"]["value"]
        self.cp_oil = u["cp_oil"]["value"]
        self.cp_steel = u["cp_steel"]["value"]

        # Per-unit equivalent-circuit parameters --------------------------------
        # Series branch (from short-circuit test):
        #   R_eq_pu = P_k / S_rated   (load loss in pu of rating)
        #   X_eq_pu = sqrt(u_k^2 - R_eq_pu^2)
        self.R_eq_pu = self.P_k / self.S_rated
        self.X_eq_pu = float(np.sqrt(max(self.u_k**2 - self.R_eq_pu**2, 0.0)))

        # Shunt branch (from open-circuit / no-load test):
        #   conductance G_c_pu = P_0 / S_rated  -> R_c_pu = 1/G_c_pu
        #   no-load current i_0 splits into core-loss + magnetizing components
        self.G_c_pu = self.P_0 / self.S_rated
        self.R_c_pu = 1.0 / self.G_c_pu if self.G_c_pu > 0 else np.inf
        b_m_sq = max(self.i_0**2 - self.G_c_pu**2, 0.0)  # susceptance^2
        self.B_m_pu = float(np.sqrt(b_m_sq))
        self.X_m_pu = (1.0 / self.B_m_pu) if self.B_m_pu > 0 else np.inf

        # Loss ratio R_pk used by the thermal ultimate-rise expression.
        self.R_pk = self.P_k / self.P_0 if self.P_0 > 0 else 0.0

    # ======================================================================
    # Equivalent-circuit / steady-state electrical quantities
    # ======================================================================

    def core_loss(self, voltage_pu=1.0):
        """No-load (core) loss [W]. Steinmetz: P_0 * V_pu^n_B (~V^1.8)."""
        v = np.asarray(voltage_pu, dtype=float)
        return self.P_0 * np.abs(v) ** self.n_B

    def copper_loss(self, load_fraction, winding_temp=75.0):
        """
        Load (copper / I^2R) loss [W], temperature-corrected to winding_temp.
        P_cu = P_k * PLR^2 * (1 + alpha_Cu*(T_w - T_ref))
        """
        plr = np.asarray(load_fraction, dtype=float)
        T_w = np.asarray(winding_temp, dtype=float)
        R_ratio = 1.0 + self.alpha_Cu * (T_w - self.T_ref)
        return self.P_k * plr ** 2 * R_ratio

    def total_loss(self, load_fraction, voltage_pu=1.0, winding_temp=75.0):
        """Total loss [W] = core + copper."""
        return self.core_loss(voltage_pu) + self.copper_loss(load_fraction, winding_temp)

    def output_power(self, load_fraction, power_factor=1.0):
        """Output active power [W] = PLR * S_rated * pf."""
        plr = np.asarray(load_fraction, dtype=float)
        return plr * self.S_rated * np.asarray(power_factor, dtype=float)

    def efficiency(self, load_fraction, voltage_pu=1.0, winding_temp=75.0,
                   power_factor=1.0):
        """
        Efficiency = P_out / (P_out + losses). Returns 0 at zero load
        (only no-load loss, no output). Bounded to (0, 1).
        """
        P_out = self.output_power(load_fraction, power_factor)
        P_loss = self.total_loss(load_fraction, voltage_pu, winding_temp)
        eta = np.where(P_out > 0, P_out / (P_out + P_loss), 0.0)
        return np.clip(eta, 0.0, 1.0)

    def optimal_load_fraction(self):
        """
        PLR for maximum efficiency = sqrt(P_0 / P_k) (core loss = copper loss).
        For distribution transformers this is a PARTIAL load.
        """
        return float(np.sqrt(self.P_0 / self.P_k))

    def voltage_regulation(self, load_fraction=1.0, power_factor=1.0,
                           leading=False):
        """
        Per-unit voltage regulation referred to the secondary, from the series
        equivalent-circuit impedance (Fitzgerald Ch. 2 approximate formula):

            VR ~ PLR * (R_eq_pu*cos(phi) +/- X_eq_pu*sin(phi))

        '+' for lagging pf (inductive load, the usual case -> positive VR),
        '-' for leading pf. Exact phasor form also computed and returned.
        """
        plr = float(load_fraction)
        pf = float(power_factor)
        sin_phi = float(np.sqrt(max(1.0 - pf**2, 0.0)))
        sign = -1.0 if leading else 1.0
        vr_approx = plr * (self.R_eq_pu * pf + sign * self.X_eq_pu * sin_phi)

        # Exact: |V1| for |V2| = 1.0 pu at given load current angle.
        I = plr  # pu current magnitude (|S| basis)
        q_sign = -1.0 if leading else 1.0
        # V1 = V2 + I*(R+jX), with I at angle -phi (lagging)
        Ir = I * pf
        Ii = -q_sign * I * sin_phi
        V1r = 1.0 + Ir * self.R_eq_pu - Ii * self.X_eq_pu
        V1i = Ir * self.X_eq_pu + Ii * self.R_eq_pu
        vr_exact = float(np.hypot(V1r, V1i) - 1.0)
        return {"vr_approx_pu": vr_approx, "vr_exact_pu": vr_exact}

    def equivalent_circuit(self):
        """Return the per-unit equivalent-circuit parameter dict."""
        return {
            "R_eq_pu": self.R_eq_pu,
            "X_eq_pu": self.X_eq_pu,
            "Z_eq_pu": self.u_k,
            "R_c_pu": self.R_c_pu,
            "X_m_pu": self.X_m_pu,
            "i_0_pu": self.i_0,
        }

    # ======================================================================
    # Thermal model (IEEE C57.91 exponential, lumped, solve_ivp)
    # ======================================================================

    def _theta_oil_ult(self, K):
        """Ultimate top-oil rise [K] at per-unit load K (C57.91 Annex G)."""
        K = np.asarray(K, dtype=float)
        ratio = (K**2 * self.R_pk + 1.0) / (self.R_pk + 1.0)
        return self.theta_oil_rated * np.maximum(ratio, 0.0) ** self.n_oil

    def _theta_hs_ult(self, K, m=0.8):
        """Ultimate hot-spot-to-oil gradient [K] at per-unit load K."""
        K = np.asarray(K, dtype=float)
        return self.theta_hs_grad_rated * np.abs(K) ** (2.0 * m)

    def hot_spot_steady(self, load_fraction, ambient_temperature=20.0):
        """Steady-state hot-spot temperature [degC] (closed form, no ODE)."""
        K = np.asarray(load_fraction, dtype=float)
        return (np.asarray(ambient_temperature, dtype=float)
                + self._theta_oil_ult(K) + self._theta_hs_ult(K))

    def simulate_thermal(self, load_fraction, ambient_temperature=20.0,
                         dt=60.0, duration=14400.0, theta_oil0=None,
                         theta_hs0=None, m=0.8):
        """
        Integrate the two coupled thermal ODEs with scipy.solve_ivp.

        load_fraction : float OR callable f(t)->K (per-unit load profile)
        ambient_temperature : float OR callable f(t)->T_amb [degC]
        Returns dict of time arrays: t, theta_oil, theta_hs, T_top_oil,
        T_hot_spot, load, ambient, p_core, p_copper, p_total.
        """
        K_fn = load_fraction if callable(load_fraction) else (lambda t: load_fraction)
        Ta_fn = (ambient_temperature if callable(ambient_temperature)
                 else (lambda t: ambient_temperature))

        if theta_oil0 is None:
            theta_oil0 = float(self._theta_oil_ult(K_fn(0.0)))
        if theta_hs0 is None:
            theta_hs0 = float(self._theta_hs_ult(K_fn(0.0), m))

        def rhs(t, y):
            theta_oil, theta_hs = y
            K = float(K_fn(t))
            d_oil = (self._theta_oil_ult(K) - theta_oil) / self.tau_oil
            d_hs = (self._theta_hs_ult(K, m) - theta_hs) / self.tau_w
            return [d_oil, d_hs]

        t_eval = np.arange(0.0, duration + 0.5 * dt, dt)
        sol = solve_ivp(rhs, (0.0, t_eval[-1]), [theta_oil0, theta_hs0],
                        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

        t = sol.t
        theta_oil = sol.y[0]
        theta_hs = sol.y[1]
        Ta = np.array([Ta_fn(ti) for ti in t])
        K_arr = np.array([float(K_fn(ti)) for ti in t])

        T_top_oil = Ta + theta_oil
        T_hot_spot = Ta + theta_oil + theta_hs

        # Loss arrays evaluated at the instantaneous hot-spot temperature.
        p_core = self.core_loss(1.0) * np.ones_like(t)
        p_cu = self.copper_loss(K_arr, T_hot_spot)
        p_total = p_core + p_cu

        return {
            "t": t,
            "theta_oil": theta_oil,
            "theta_hs": theta_hs,
            "T_top_oil": T_top_oil,
            "T_hot_spot": T_hot_spot,
            "load": K_arr,
            "ambient": Ta,
            "p_core": p_core,
            "p_copper": p_cu,
            "p_total": p_total,
        }

    # ======================================================================
    # Daily loading convenience (low average load factor profile)
    # ======================================================================

    @staticmethod
    def residential_daily_profile():
        """
        Hourly per-unit load for a typical residential distribution feeder.
        Average load factor ~0.45 with morning + evening peaks. Returns a
        callable f(t_seconds) -> K via step interpolation (24 h period).
        IEEE C57.91 Annex G uses such varying-load profiles.
        """
        hourly = np.array([
            0.30, 0.27, 0.25, 0.25, 0.28, 0.38,   # 00-05 night/early
            0.55, 0.70, 0.62, 0.50, 0.45, 0.45,   # 06-11 morning peak
            0.48, 0.46, 0.44, 0.46, 0.55, 0.75,   # 12-17 afternoon rise
            0.92, 0.88, 0.78, 0.62, 0.48, 0.36,   # 18-23 evening peak
        ])

        def profile(t):
            hr = int((t / 3600.0) % 24)
            return float(hourly[hr])

        return profile, hourly
