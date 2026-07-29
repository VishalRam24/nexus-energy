"""
EC055 -- Solar Tower / Central Receiver CSP -- F2a Physics-Lumped Model

Physics-lumped (0D) central-receiver model. The heliostat field concentrates
DNI onto a single external cylindrical receiver, whose metal+salt thermal mass
is treated as ONE lumped node with an energy-balance ODE integrated by
scipy.integrate.solve_ivp. The heated molten salt leaves the receiver and the
enthalpy gain of the salt is delivered as thermal power to the Rankine power
block.

--------------------------------------------------------------------------
1. Heliostat-field optical efficiency (algebraic, from F1 of this component)
--------------------------------------------------------------------------
    eta_field(z) = eta_field_peak * f_cos(z) * f_atm(z)
    f_cos(z) = 1 - cos_coeff * z^2          (field-averaged cosine + spillage proxy)
    f_atm(z) = 1 - atm_coeff * z            (atmospheric slant-path attenuation)

    Concentrated power intercepted at the receiver aperture:
        Q_field = DNI * A_field * eta_field(z) * (1 - spillage_frac)

    Absorbed solar flux on the receiver surface:
        Q_abs = absorptivity * Q_field

   Reference: Stine & Geyer (2001) "Power From The Sun", central-receiver
   chapter; Wagner & Wendelin (2018) SolarPILOT, Solar Energy 171:185-196.

--------------------------------------------------------------------------
2. Lumped receiver energy balance ODE (state = receiver/HTF temperature T)
--------------------------------------------------------------------------
    (m_metal*cp_metal + m_salt*cp_salt) dT/dt
        =  Q_abs                                    absorbed concentrated solar
         - Q_rad(T)                                 IR re-radiation  ∝ (T^4 - T_amb^4)
         - Q_conv(T)                                convective loss  ∝ (T - T_amb)
         - Q_htf(T)                                 enthalpy carried off by the salt

    Q_rad  = emissivity * view_factor * sigma * A_recv * (T^4 - T_amb^4)
    Q_conv = h_conv(v_wind) * A_recv * (T - T_amb)
    Q_htf  = mdot_salt * cp_salt * (T - T_HTF_in)        (T is the salt outlet temp)

    The Stefan-Boltzmann T^4 term makes radiative loss grow fast with T, so the
    receiver thermal efficiency DROPS as the receiver runs hotter -- a defining
    feature of high-temperature central receivers (Stine & Geyer 2001).

   Reference: Falcone (1986) SAND86-8009 "A handbook for solar central receiver
   design"; Zavoico (2001) SAND2001-2120 (Solar Two receiver); Wagner (2008)
   MS thesis, central-receiver lumped receiver energy balance.

--------------------------------------------------------------------------
3. Thermal power to power block
--------------------------------------------------------------------------
    Q_thermal_to_PB = Q_htf = mdot_salt * cp_salt * (T_out - T_HTF_in)
    P_electric_gross = eta_powerblock * Q_thermal_to_PB

Conservation / limit guarantees enforced by construction:
  * Energy balance: Q_abs = dE_stored/dt + Q_rad + Q_conv + Q_htf  (checked in tests).
  * Radiative loss strictly ∝ (T^4 - T_amb^4).
  * Receiver efficiency monotonically decreases with receiver temperature.
  * DNI = 0  =>  Q_abs = 0  =>  no useful heat, P_block = 0.

References:
    Stine, W.B. & Geyer, M. (2001). Power From The Sun. www.powerfromthesun.net
    Wagner, M.J. & Wendelin, T. (2018). SolarPILOT. Solar Energy 171, 185-196.
    Falcone, P.K. (1986). SAND86-8009, Sandia National Laboratories.
    Zavoico, A.B. (2001). SAND2001-2120 (Solar Two central receiver design).
    Kolb, G.J. (2011). SAND2011-2419 (Gemasolar performance analysis).
    Siebers, D.L. & Kraabel, J.S. (1984). SAND84-8717 (convective loss).
"""

import numpy as np
from scipy.integrate import solve_ivp


class SolarTowerF2a:
    """Central-receiver CSP: heliostat optics + lumped receiver thermal ODE."""

    sigma = 5.670374419e-8  # Stefan-Boltzmann constant [W/m2K4]

    def __init__(self, params: dict):
        u = params["unit"]
        # Optics
        self.A_field = u["A_field"]["value"]
        self.eta_field_peak = u["eta_field_peak"]["value"]
        self.atm_coeff = u["atm_atten_coeff"]["value"]
        self.cos_coeff = u["cosine_loss_coeff"]["value"]
        self.spillage = u["spillage_frac"]["value"]
        self.alpha = u["absorptivity"]["value"]

        # Receiver surface / losses
        self.A_recv = u["A_receiver"]["value"]
        self.eps = u["emissivity"]["value"]
        self.F_view = u["view_factor"]["value"]
        self.h_base = u["h_conv_base"]["value"]
        self.h_wind = u["h_conv_wind_coeff"]["value"]

        # Lumped thermal capacitance (metal + resident salt)
        self.m_metal = u["m_receiver_metal"]["value"]
        self.cp_metal = u["cp_receiver_metal"]["value"]
        self.m_salt = u["m_salt_holdup"]["value"]
        self.cp_salt = u["cp_salt"]["value"]
        self.C_th = self.m_metal * self.cp_metal + self.m_salt * self.cp_salt  # J/K

        # HTF / power block
        self.T_HTF_in_design = u["T_HTF_in_C"]["value"] + 273.15  # K
        self.mdot_design = u["mdot_salt_design"]["value"]
        self.eta_pb = u["eta_powerblock"]["value"]

    # ------------------------------------------------------------------
    # 1. Heliostat-field optics
    # ------------------------------------------------------------------
    def field_efficiency(self, zenith_deg):
        """Field optical efficiency (cosine + atmospheric attenuation)."""
        z = np.asarray(zenith_deg, dtype=float)
        f_cos = np.clip(1.0 - self.cos_coeff * z * z, 0.0, 1.0)
        f_atm = np.clip(1.0 - self.atm_coeff * z, 0.0, 1.0)
        return np.clip(self.eta_field_peak * f_cos * f_atm, 0.0, 1.0)

    def Q_field(self, dni, zenith_deg):
        """Concentrated solar power intercepted at receiver aperture [W]."""
        G = np.asarray(dni, dtype=float)
        eta = self.field_efficiency(zenith_deg)
        return G * self.A_field * eta * (1.0 - self.spillage)

    def Q_absorbed(self, dni, zenith_deg):
        """Absorbed concentrated solar flux on receiver surface [W]."""
        return self.alpha * self.Q_field(dni, zenith_deg)

    # ------------------------------------------------------------------
    # 2. Receiver loss terms
    # ------------------------------------------------------------------
    def h_conv(self, wind_speed=0.0):
        """Wind-dependent convective HT coefficient [W/m2K] (Siebers & Kraabel 1984)."""
        v = np.maximum(np.asarray(wind_speed, dtype=float), 0.0)
        return self.h_base + self.h_wind * np.sqrt(v)

    def Q_rad(self, T_K, T_amb_K):
        """Radiative re-radiation loss [W], strictly ∝ (T^4 - T_amb^4)."""
        return self.eps * self.F_view * self.sigma * self.A_recv * (T_K ** 4 - T_amb_K ** 4)

    def Q_conv(self, T_K, T_amb_K, wind_speed=0.0):
        """Convective loss [W]."""
        return self.h_conv(wind_speed) * self.A_recv * (T_K - T_amb_K)

    def Q_htf(self, T_K, mdot, T_HTF_in_K):
        """Enthalpy carried off by the molten salt [W] (clamped >= 0)."""
        return np.maximum(0.0, mdot * self.cp_salt * (T_K - T_HTF_in_K))

    def receiver_efficiency(self, dni, zenith_deg, T_K, T_amb_K, wind_speed=0.0):
        """Steady receiver efficiency = (Q_abs - losses) / Q_field. Drops as T rises."""
        Qf = self.Q_field(dni, zenith_deg)
        Qa = self.alpha * Qf
        Qloss = self.Q_rad(T_K, T_amb_K) + self.Q_conv(T_K, T_amb_K, wind_speed)
        Quse = np.maximum(0.0, Qa - Qloss)
        Qf_safe = np.where(Qf > 1e-6, Qf, 1.0)
        return np.where(Qf > 1e-6, np.clip(Quse / Qf_safe, 0.0, 1.0), 0.0)

    # ------------------------------------------------------------------
    # 3. Lumped energy-balance ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, dni_f, zen_f, Tamb_f, wind_f, mdot_f, Tin_f):
        """dT/dt for the lumped receiver node."""
        T = y[0]
        T_amb = Tamb_f(t) + 273.15
        Tin = Tin_f(t) + 273.15
        Qa = self.Q_absorbed(dni_f(t), zen_f(t))
        Qr = self.Q_rad(T, T_amb)
        Qc = self.Q_conv(T, T_amb, wind_f(t))
        Qh = self.Q_htf(T, mdot_f(t), Tin)
        dTdt = (Qa - Qr - Qc - Qh) / self.C_th
        return [dTdt]

    @staticmethod
    def _as_callable(x, t_grid):
        """Return a time-function for a scalar, array (over t_grid), or callable."""
        if callable(x):
            return x
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            val = float(arr)
            return lambda t: val
        return lambda t: float(np.interp(t, t_grid, arr))

    def simulate(self, dni, zenith_deg=30.0, T_amb_C=25.0, wind_speed=5.0,
                 mdot_salt=None, T_HTF_in_C=290.0, T0_C=290.0,
                 dt=10.0, duration_s=3600.0):
        """
        Integrate the lumped receiver thermal ODE.

        Parameters
        ----------
        dni        : W/m2  -- scalar, array (sampled on t-grid) or callable f(t)
        zenith_deg : deg   -- solar zenith angle (scalar/array/callable)
        T_amb_C    : degC  -- ambient temperature
        wind_speed : m/s   -- wind speed at receiver
        mdot_salt  : kg/s  -- HTF mass flow (default = design flow)
        T_HTF_in_C : degC  -- cold-salt inlet temperature
        T0_C       : degC  -- initial receiver/salt temperature
        dt         : s     -- output sample interval
        duration_s : s     -- total simulated time

        Returns dict of time-series arrays.
        """
        if mdot_salt is None:
            mdot_salt = self.mdot_design

        n = int(round(duration_s / dt)) + 1
        t_grid = np.linspace(0.0, duration_s, n)

        dni_f = self._as_callable(dni, t_grid)
        zen_f = self._as_callable(zenith_deg, t_grid)
        Tamb_f = self._as_callable(T_amb_C, t_grid)
        wind_f = self._as_callable(wind_speed, t_grid)
        mdot_f = self._as_callable(mdot_salt, t_grid)
        Tin_f = self._as_callable(T_HTF_in_C, t_grid)

        T0 = T0_C + 273.15
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [T0],
            t_eval=t_grid, method="BDF",
            args=(dni_f, zen_f, Tamb_f, wind_f, mdot_f, Tin_f),
            rtol=1e-6, atol=1e-3,
        )
        T_K = sol.y[0]

        # Reconstruct power terms along the solution
        dni_arr = np.array([dni_f(t) for t in t_grid])
        zen_arr = np.array([zen_f(t) for t in t_grid])
        Tamb_arr = np.array([Tamb_f(t) for t in t_grid]) + 273.15
        wind_arr = np.array([wind_f(t) for t in t_grid])
        mdot_arr = np.array([mdot_f(t) for t in t_grid])
        Tin_arr = np.array([Tin_f(t) for t in t_grid]) + 273.15

        Q_field = self.Q_field(dni_arr, zen_arr)
        Q_abs = self.alpha * Q_field
        Q_rad = self.Q_rad(T_K, Tamb_arr)
        Q_conv = self.Q_conv(T_K, Tamb_arr, wind_arr)
        Q_htf = self.Q_htf(T_K, mdot_arr, Tin_arr)
        Q_loss = Q_rad + Q_conv

        # stored-energy rate (numerical) for conservation diagnostics
        dEdt = np.gradient(self.C_th * T_K, t_grid)

        eta_field = self.field_efficiency(zen_arr)
        Q_inc = dni_arr * self.A_field
        eta_recv = np.where(Q_field > 1e-6,
                            np.clip(Q_htf / np.where(Q_field > 1e-6, Q_field, 1.0), 0.0, 1.0),
                            0.0)
        eta_overall = np.where(Q_inc > 1e-6,
                               np.clip(Q_htf / np.where(Q_inc > 1e-6, Q_inc, 1.0), 0.0, 1.0),
                               0.0)
        P_block = self.eta_pb * Q_htf  # gross electric [W]

        return {
            "t": t_grid,
            "T_receiver_K": T_K,
            "T_receiver_C": T_K - 273.15,
            "Q_field_W": Q_field,
            "Q_absorbed_W": Q_abs,
            "Q_rad_loss_W": Q_rad,
            "Q_conv_loss_W": Q_conv,
            "Q_loss_W": Q_loss,
            "Q_thermal_to_PB_W": Q_htf,
            "Q_thermal_to_PB_MWth": Q_htf / 1e6,
            "P_electric_W": P_block,
            "P_electric_MWe": P_block / 1e6,
            "dE_stored_dt_W": dEdt,
            "field_efficiency": eta_field,
            "receiver_efficiency": eta_recv,
            "overall_efficiency": eta_overall,
            "solver_success": sol.success,
        }
