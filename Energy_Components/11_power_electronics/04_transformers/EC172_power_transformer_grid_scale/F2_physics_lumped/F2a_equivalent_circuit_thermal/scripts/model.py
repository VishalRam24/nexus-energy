"""
EC172 -- Power Transformer (Grid-Scale) -- F2a Equivalent-Circuit + Thermal ODE

Physics-lumped model coupling the classical transformer equivalent circuit with a
two-state lumped oil/winding thermal model integrated by scipy.solve_ivp.

----------------------------------------------------------------------------------
1. ELECTRICAL  -- per-unit equivalent circuit (Fitzgerald, Kingsley & Umans 2003,
   "Electric Machinery", 6th ed., Ch.2; the "cantilever" / approximate equivalent
   circuit referred to the rating base):

       o---[ R + jX ]---+----o          R, X : series leakage impedance (pu)
                        |               Rc   : shunt core-loss resistance (pu)
       V_s            [Rc] [jXm]        Xm   : shunt magnetizing reactance (pu)
                        |
       o----------------+----o

   With the load drawing current  I = (S/V) at angle -acos(pf) (lagging),
   the voltage regulation (referred to the secondary base) is the rise in
   receiving-end voltage when load is removed, computed from the series drop:

       V_send = V_recv + I * (R + jX)
       Regulation = (|V_send| - |V_recv|) / |V_recv|

   The standard approximate (Fitzgerald Eq. 2.x) closed form for a load at
   power factor pf (lagging -> +sin, leading -> -sin):

       VR ~ PLR * (R_pu*cos_phi + X_pu*sin_phi)
            + 0.5 * PLR^2 * (X_pu*cos_phi - R_pu*sin_phi)^2

2. LOSSES & EFFICIENCY:
       P_core  = P_no_load * V_pu^n             (no-load, voltage dependent; B^2~V^2)
       P_cu    = P_load_loss * PLR^2 * R(T_w)   (load loss, I^2 * temp-corrected R)
       R(T_w)  = 1 + alpha_Cu*(T_w - T_ref)
       P_out   = PLR * S_rated * pf
       eta     = P_out / (P_out + P_core + P_cu)
   Efficiency peaks where variable (copper) loss equals fixed (core) loss
   (maximum-efficiency theorem, Fitzgerald Ch.2):  PLR* = sqrt(P_core/P_cu_rated).

3. THERMAL  -- lumped two-state ODE (IEEE Std C57.91-2011 loading guide,
   Clause 7 differential / exponential model; also IEC 60076-7:2018):

   Top-oil state (theta_o) :
       tau_o dtheta_o/dt = [delta_theta_o_rated * ((1 + R*K^2)/(1+R))^n_oil] - theta_o
   Hot-spot gradient state (delta_theta_h) :
       tau_w d(delta_theta_h)/dt = [delta_theta_h_rated * K^(2*m)] - delta_theta_h

   where K = PLR (per-unit load current), R = P_cu_rated/P_core (loss ratio),
   theta_o is top-oil rise above ambient, delta_theta_h is hot-spot-to-top-oil
   gradient. Hot-spot temperature:
       theta_hs = T_ambient + theta_o + delta_theta_h

   Energy consistency: at steady state the dissipated loss P_core+P_cu maps to the
   top-oil rise through the rated loss-to-rise calibration, so the thermal rise
   scales monotonically with total loss (and hence with load).

References:
    Fitzgerald, Kingsley & Umans (2003), Electric Machinery, 6th ed., McGraw-Hill, Ch.2.
    IEEE Std C57.91-2011, Guide for Loading Mineral-Oil-Immersed Transformers, Clause 7.
    IEC 60076-7:2018, Power transformers -- Loading guide for oil-immersed transformers.
    Kulkarni & Khaparde (2004), Transformer Engineering, CRC Press.
"""

import numpy as np
from scipy.integrate import solve_ivp


class PowerTransformerF2a:
    """Grid-scale power transformer: pu equivalent circuit + lumped thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_rated = u["S_rated_VA"]["value"]            # VA
        self.V_pri = u["V_primary_V"]["value"]             # V
        self.V_sec = u["V_secondary_V"]["value"]           # V
        self.freq = u["freq_Hz"]["value"]                  # Hz

        # per-unit equivalent circuit
        self.R_pu = u["R_pu"]["value"]
        self.X_pu = u["X_pu"]["value"]
        self.Rc_pu = u["Rc_pu"]["value"]
        self.Xm_pu = u["Xm_pu"]["value"]

        # losses
        self.P_no_load = u["P_no_load_W"]["value"]         # W (core, at rated V)
        self.P_load_loss = u["P_load_loss_W"]["value"]     # W (copper, rated I, T_ref)
        self.n_core = u["core_voltage_exponent"]["value"]
        self.T_ref_winding = u["T_ref_winding"]["value"]   # degC
        self.alpha_Cu = u["alpha_Cu"]["value"]

        # thermal (IEEE C57.91)
        self.dth_oil_rated = u["delta_theta_oil_rated_K"]["value"]    # K
        self.dth_hs_grad = u["delta_theta_hs_gradient_K"]["value"]    # K
        self.n_oil = u["n_oil"]["value"]
        self.m_wind = u["m_wind"]["value"]
        self.tau_oil = u["tau_oil_min"]["value"]           # min
        self.tau_wind = u["tau_wind_min"]["value"]         # min

        # loss ratio R = rated load loss / no-load loss (IEEE C57.91 symbol)
        self.loss_ratio = self.P_load_loss / self.P_no_load

    # ----------------------------------------------------------------- electrical
    def base_impedance_secondary(self):
        """Base impedance on the secondary side [ohm] (3-phase: Z_b = V_LL^2/S)."""
        return self.V_sec ** 2 / self.S_rated

    def series_impedance_secondary(self):
        """Series leakage impedance referred to secondary [ohm]: (R + jX)."""
        Zb = self.base_impedance_secondary()
        return complex(self.R_pu * Zb, self.X_pu * Zb)

    def voltage_regulation(self, load_fraction, power_factor=1.0, leading=False):
        """
        Voltage regulation [fraction] for a given load and power factor.

        Computed exactly from the phasor series drop with the receiving-end
        voltage taken as 1.0 pu reference (Fitzgerald Ch.2):
            V_send = 1.0 + I_pu * (R + jX),  I_pu = PLR at angle -phi (lagging)
            VR = |V_send| - 1.0
        """
        plr = np.asarray(load_fraction, dtype=float)
        pf = float(np.clip(power_factor, 0.0, 1.0))
        cos_phi = pf
        sin_phi = np.sqrt(max(0.0, 1.0 - pf * pf))
        # lagging pf -> current lags -> +sin_phi adds inductive drop
        q_sign = -1.0 if leading else 1.0
        # I_pu phasor (receiving-end V reference at 0 deg): I = PLR*(cos - j*q*sin)
        I_re = plr * cos_phi
        I_im = -q_sign * plr * sin_phi
        # series drop = I * Z
        drop_re = I_re * self.R_pu - I_im * self.X_pu
        drop_im = I_re * self.X_pu + I_im * self.R_pu
        V_send_re = 1.0 + drop_re
        V_send_im = drop_im
        V_send_mag = np.sqrt(V_send_re ** 2 + V_send_im ** 2)
        return V_send_mag - 1.0

    def magnetizing_current_pu(self, voltage_pu=1.0):
        """No-load magnetizing branch current [pu] = V_pu*(1/Rc + 1/jXm)."""
        v = float(voltage_pu)
        I_c = v / self.Rc_pu           # core-loss (real) component
        I_m = v / self.Xm_pu           # magnetizing (reactive) component
        return np.hypot(I_c, I_m)

    # -------------------------------------------------------------------- losses
    def core_loss(self, voltage_pu=1.0):
        """Core (no-load) loss [W] = P_no_load * V_pu^n_core."""
        v = np.asarray(voltage_pu, dtype=float)
        return self.P_no_load * v ** self.n_core

    def copper_loss(self, load_fraction, winding_temperature=75.0):
        """Copper (load) loss [W], temperature-corrected, ~ I^2."""
        plr = np.asarray(load_fraction, dtype=float)
        Tw = np.asarray(winding_temperature, dtype=float)
        R_ratio = 1.0 + self.alpha_Cu * (Tw - self.T_ref_winding)
        return self.P_load_loss * plr ** 2 * R_ratio

    def total_losses(self, load_fraction, voltage_pu=1.0, winding_temperature=75.0):
        return self.core_loss(voltage_pu) + self.copper_loss(load_fraction, winding_temperature)

    # ---------------------------------------------------------- power/efficiency
    def output_power(self, load_fraction, power_factor=1.0):
        """Active power delivered to load [W] = PLR * S_rated * pf."""
        return np.asarray(load_fraction, dtype=float) * self.S_rated * float(power_factor)

    def efficiency(self, load_fraction, voltage_pu=1.0, power_factor=1.0,
                   winding_temperature=75.0):
        """Efficiency = P_out / (P_out + losses), clipped to (0,1)."""
        P_out = self.output_power(load_fraction, power_factor)
        P_loss = self.total_losses(load_fraction, voltage_pu, winding_temperature)
        eta = np.where(P_out + P_loss > 0, P_out / (P_out + P_loss), 0.0)
        return np.clip(eta, 0.0, 1.0)

    def max_efficiency_load(self):
        """
        Per-unit load at which efficiency peaks (copper loss == core loss):
            PLR* = sqrt(P_core / P_cu_rated)   (Fitzgerald Ch.2).
        """
        return float(np.sqrt(self.P_no_load / self.P_load_loss))

    # ----------------------------------------------------------------- thermal SS
    def steady_top_oil_rise(self, load_fraction):
        """Steady-state top-oil rise above ambient [K] (IEEE C57.91 Clause 7)."""
        K = np.asarray(load_fraction, dtype=float)
        ratio = (1.0 + self.loss_ratio * K ** 2) / (1.0 + self.loss_ratio)
        return self.dth_oil_rated * ratio ** self.n_oil

    def steady_hotspot_gradient(self, load_fraction):
        """Steady-state hot-spot-to-top-oil gradient [K]."""
        K = np.asarray(load_fraction, dtype=float)
        return self.dth_hs_grad * K ** (2.0 * self.m_wind)

    def steady_hotspot_temperature(self, load_fraction, ambient_temperature=20.0):
        """Steady-state hot-spot temperature [degC]."""
        return (np.asarray(ambient_temperature, dtype=float)
                + self.steady_top_oil_rise(load_fraction)
                + self.steady_hotspot_gradient(load_fraction))

    # ------------------------------------------------------------- thermal ODE
    def _rhs(self, t, y, K_func, T_amb):
        """
        Two-state thermal ODE (IEEE C57.91 differential model). State:
            y[0] = theta_o  (top-oil rise above ambient, K)
            y[1] = dtheta_h (hot-spot-to-top-oil gradient, K)
        """
        K = float(K_func(t))
        theta_o_ult = self.steady_top_oil_rise(K)
        dth_h_ult = self.steady_hotspot_gradient(K)
        dtheta_o = (theta_o_ult - y[0]) / self.tau_oil
        ddth_h = (dth_h_ult - y[1]) / self.tau_wind
        return [dtheta_o, ddth_h]

    def simulate(self, load_profile, ambient_temperature=20.0, dt=1.0,
                 duration=240.0, power_factor=0.9, voltage_pu=1.0,
                 theta_o0=None, dtheta_h0=0.0):
        """
        Integrate the lumped thermal transient (time in MINUTES) via solve_ivp,
        and report electrical quantities along the trajectory.

        load_profile : float or callable f(t_min)->K (per-unit load current)
        Returns dict of arrays over time.
        """
        if callable(load_profile):
            K_func = load_profile
        else:
            Kval = float(load_profile)
            K_func = lambda t: Kval

        T_amb = float(ambient_temperature)
        if theta_o0 is None:
            theta_o0 = float(self.steady_top_oil_rise(K_func(0.0)))

        t_eval = np.arange(0.0, duration + 1e-9, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration), [theta_o0, dtheta_h0],
            t_eval=t_eval, args=(K_func, T_amb),
            method="RK45", rtol=1e-7, atol=1e-9, max_step=dt,
        )

        t = sol.t
        theta_o = sol.y[0]
        dtheta_h = sol.y[1]
        theta_hs = T_amb + theta_o + dtheta_h
        theta_top_oil = T_amb + theta_o

        K_arr = np.array([K_func(ti) for ti in t])
        # winding temp drives copper-loss resistance correction
        Tw = theta_hs
        P_core = self.core_loss(voltage_pu) * np.ones_like(t)
        P_cu = self.copper_loss(K_arr, Tw)
        P_out = self.output_power(K_arr, power_factor)
        eta = np.where(P_out + P_core + P_cu > 0,
                       P_out / (P_out + P_core + P_cu), 0.0)
        eta = np.clip(eta, 0.0, 1.0)
        VR = np.array([self.voltage_regulation(k, power_factor) for k in K_arr])

        return {
            "t": t,                              # min
            "load_fraction": K_arr,              # pu
            "top_oil_rise": theta_o,             # K
            "hotspot_gradient": dtheta_h,        # K
            "top_oil_temperature": theta_top_oil,  # degC
            "hotspot_temperature": theta_hs,     # degC
            "p_core": P_core,                    # W
            "p_copper": P_cu,                    # W
            "p_total_loss": P_core + P_cu,       # W
            "p_output": P_out,                   # W
            "efficiency": eta,                   # -
            "voltage_regulation": VR,            # fraction
        }
