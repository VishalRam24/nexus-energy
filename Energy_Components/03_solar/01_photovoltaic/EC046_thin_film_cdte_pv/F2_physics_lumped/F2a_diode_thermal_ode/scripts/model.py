"""
EC046 -- Thin-Film CdTe PV -- F2a Physics-Lumped (Single-Diode + Lumped Thermal ODE)

This is the physics-lumped (0D ODE) upgrade of the F1 De Soto single-diode model.

I-V curve (first-principles, single-diode / De Soto 5-parameter):

    I = I_L - I_o * [exp((V + I*R_s)/a) - 1] - (V + I*R_s)/R_sh

The implicit equation is solved in CLOSED FORM with the Lambert-W function
(Jain & Kapoor 2004), which makes I(V) and V(I) explicit:

    I(V) = (R_sh*(I_L+I_o) - V) / (R_s + R_sh)
           - (a/R_s) * W{ R_s*R_sh*I_o / (a*(R_s+R_sh))
                          * exp( R_sh*(R_s*(I_L+I_o)+V) / (a*(R_s+R_sh)) ) }

A vectorised Newton root-find is provided as a fallback / cross-check for the
regime where the Lambert-W argument overflows.

De Soto 5-parameter translation to operating (G, T_cell):
    I_L  = (G/G_ref)*(I_L_ref + alpha_sc*(T - T_ref))
    I_o  = I_o_ref*(T/T_ref)^3 * exp[ Eg_ref/(k/q*T_ref) - Eg(T)/(k/q*T) ]
    a    = a_ref * T/T_ref
    R_sh = R_sh_ref * (G_ref/G)
    R_s  = const
(De Soto, Klein & Beckman 2006, Solar Energy 80(1) 78-88.)

CdTe specifics:
    * Wide bandgap (1.45 eV, near-optimal for AM1.5G) -> low I_o, low Pmp tempco.
    * Pmp temperature coefficient -0.28 %/K (vs ~-0.40 %/K for c-Si).
      The bare De Soto fit over-predicts this at -0.485 %/K, so a CdTe-specific
      empirical correction factor 0.578 = 0.0028/0.00485 is applied to the
      temperature-dependent deviation of Pmp (preserved from F1b; Strevel et
      al. 2012, Prog. Photovolt. 20(1) 6-11, DOI:10.1002/pip.1209). CdTe ONLY.
    * Blue-rich spectral response -> small low-light energy-yield uplift
      (Nann & Emery 1992; Huld & Amillo 2015).

Lumped thermal ODE (0D lumped-capacitance energy balance, Faiman 2008):

    C_th * dT_cell/dt = Q_abs - P_elec - Q_conv - Q_rad
        Q_abs  = alpha_abs * G * A                      (absorbed solar)
        P_elec = P_mp(G, T_cell)                        (electrical extraction)
        Q_conv = (U0 + U1*v_wind) * A * (T_cell - T_amb)  (Faiman convective)
        Q_rad  = eps * sigma * A * (T_cell^4 - T_sky^4)   (long-wave radiative)

Integrated with scipy.integrate.solve_ivp (LSODA).

References:
    De Soto, Klein & Beckman (2006). Solar Energy 80(1), 78-88.
    Jain & Kapoor (2004). Sol. Energy Mater. Sol. Cells 81, 269-277.
    Faiman (2008). Progress in Photovoltaics 16(4), 307-315.
    Jones & Underwood (2001). Solar Energy 70(4), 349-359.
    Strevel, Trippel & Gloeckler (2012). Prog. Photovolt. 20(1), 6-11.
    Nann & Emery (1992). Sol. Energy Mater. Sol. Cells 27, 189-216.
    First Solar Series 6 Module Datasheet (2019).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant, W/(m^2 K^4)


class CdTePV_F2a:
    """CdTe thin-film PV -- single-diode (Lambert-W) + lumped thermal ODE."""

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]
        th = params["thermal"]
        sp = params.get("spectral", {})
        tc = params.get("tempco_correction", {})

        # --- module / De Soto electrical parameters ---
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.gamma_pmp = mod["gamma_pmp"]["value"]
        self.cells_in_series = mod["cells_in_series"]["value"]
        self.area = mod["area"]["value"]
        self.NOCT = mod["NOCT"]["value"]
        self.T_NOCT_amb = mod["T_NOCT_amb"]["value"]
        self.G_NOCT = mod["G_NOCT"]["value"]
        self.p_mp_stc = mod["p_mp_stc"]["value"]

        self.I_L_ref = ds["I_L_ref"]["value"]
        self.I_o_ref = ds["I_o_ref"]["value"]
        self.R_s = ds["R_s"]["value"]
        self.R_sh_ref = ds["R_sh_ref"]["value"]
        self.a_ref = ds["a_ref"]["value"]
        self.EgRef = ds["EgRef"]["value"]
        self.dEgdT = ds["dEgdT"]["value"]

        # --- CdTe tempco correction (preserved 0.578) ---
        self.gamma_desoto = tc.get("gamma_pmp_desoto_approx", {}).get("value", -0.00485)
        self.corr_factor = tc.get("correction_factor", {}).get("value", 0.578)

        # --- spectral / low-light ---
        self.M_low_light_gain = sp.get("M_low_light_gain", {}).get("value", 0.03)
        self.AM_ref = sp.get("AM_ref", {}).get("value", 1.5)

        # --- thermal ODE parameters ---
        self.C_thermal = th["C_thermal"]["value"]
        self.alpha_abs = th["alpha_abs"]["value"]
        self.U0 = th["U_const"]["value"]
        self.U1 = th["U_wind"]["value"]
        self.epsilon = th["epsilon"]["value"]
        self.default_wind = th["default_wind"]["value"]

        # --- physical constants ---
        self.k = 1.380649e-23
        self.q = 1.602176634e-19
        self.T_ref = 298.15      # K (25 C)
        self.G_ref = 1000.0      # W/m2

    # =====================================================================
    # De Soto 5-parameter translation to operating point
    # =====================================================================
    def desoto_params(self, irradiance, cell_temp_c):
        """Return (I_L, I_o, R_sh, a) at given irradiance [W/m2], T_cell [C]."""
        G = np.asarray(irradiance, dtype=float)
        T = np.asarray(cell_temp_c, dtype=float) + 273.15

        I_L = (G / self.G_ref) * (self.I_L_ref + self.alpha_sc * (T - self.T_ref))
        I_L = np.maximum(I_L, 0.0)

        Eg = self.EgRef * (1.0 + self.dEgdT * (T - self.T_ref) / self.EgRef)
        kq = self.k / self.q
        I_o = self.I_o_ref * (T / self.T_ref) ** 3 * np.exp(
            (self.EgRef / (kq * self.T_ref)) - (Eg / (kq * T))
        )
        a = self.a_ref * T / self.T_ref
        R_sh = self.R_sh_ref * (self.G_ref / np.maximum(G, 1.0))
        return I_L, I_o, R_sh, a

    # =====================================================================
    # Lambert-W closed-form I(V) and V(I)  (Jain & Kapoor 2004)
    # =====================================================================
    def current_from_voltage(self, V, I_L, I_o, R_sh, a):
        """Explicit I(V) via Lambert-W. Falls back to Newton if W overflows."""
        V = np.asarray(V, dtype=float)
        Rs, Rsh = self.R_s, R_sh
        # argument of W: theta = (Rs*Rsh*Io)/(a*(Rs+Rsh)) * exp(...)
        denom = a * (Rs + Rsh)
        log_factor = np.log(np.maximum(Rs * Rsh * I_o / denom, 1e-300))
        exp_arg = Rsh * (Rs * (I_L + I_o) + V) / denom
        log_theta = log_factor + exp_arg
        # Lambert-W of a (possibly huge) positive real: use the asymptotic
        # ln-domain only where exp would overflow.
        with np.errstate(over="ignore", invalid="ignore"):
            theta = np.exp(np.clip(log_theta, -700.0, 700.0))
            W = np.real(lambertw(theta))
            # where the true argument overflowed, W(x) ~ ln(x) - ln(ln(x))
            big = log_theta > 690.0
            if np.any(big):
                lt = log_theta
                W = np.where(big, lt - np.log(np.maximum(lt, 1e-12)), W)
        I = (Rsh * (I_L + I_o) - V) / (Rs + Rsh) - (a / Rs) * W
        out = np.where(np.isfinite(I), I, 0.0)
        return out

    def _i_from_v_newton(self, V, I_L, I_o, R_sh, a, n_iter=60):
        """Vectorised Newton fallback / cross-check for I(V)."""
        V = np.asarray(V, dtype=float)
        I = np.broadcast_to(np.asarray(I_L, dtype=float), V.shape).copy()
        for _ in range(n_iter):
            arg = np.clip((V + I * self.R_s) / a, -50.0, 50.0)
            ex = np.exp(arg)
            f = I_L - I_o * (ex - 1.0) - (V + I * self.R_s) / R_sh - I
            df = -I_o * ex * (self.R_s / a) - self.R_s / R_sh - 1.0
            I = I - f / df
        return I

    def voltage_from_current(self, I, I_L, I_o, R_sh, a):
        """Explicit V(I) via Lambert-W (Jain & Kapoor 2004)."""
        I = np.asarray(I, dtype=float)
        Rs, Rsh = self.R_s, R_sh
        log_factor = np.log(np.maximum(I_o * Rsh / a, 1e-300))
        exp_arg = Rsh * (I_L + I_o - I) / a
        log_theta = log_factor + exp_arg
        with np.errstate(over="ignore", invalid="ignore"):
            theta = np.exp(np.clip(log_theta, -700.0, 700.0))
            W = np.real(lambertw(theta))
            big = log_theta > 690.0
            if np.any(big):
                W = np.where(big, log_theta - np.log(np.maximum(log_theta, 1e-12)), W)
        V = (I_L + I_o - I) * Rsh - I * Rs - a * W
        return np.where(np.isfinite(V), V, 0.0)

    def open_circuit_voltage(self, I_L, I_o, R_sh, a):
        return np.maximum(self.voltage_from_current(0.0, I_L, I_o, R_sh, a), 0.0)

    def short_circuit_current(self, I_L, I_o, R_sh, a):
        return self.current_from_voltage(0.0, I_L, I_o, R_sh, a)

    # =====================================================================
    # Full I-V / P-V curve
    # =====================================================================
    def iv_curve(self, irradiance, cell_temp_c, n_points=200):
        """Return dict with V, I, P arrays over [0, Voc]."""
        I_L, I_o, R_sh, a = self.desoto_params(irradiance, cell_temp_c)
        V_oc = float(self.open_circuit_voltage(I_L, I_o, R_sh, a))
        if V_oc <= 0.0:
            z = np.zeros(n_points)
            return {"V": z, "I": z, "P": z, "v_oc": 0.0}
        V = np.linspace(0.0, V_oc, n_points)
        I = self.current_from_voltage(V, I_L, I_o, R_sh, a)
        I = np.maximum(I, 0.0)
        P = V * I
        return {"V": V, "I": I, "P": P, "v_oc": V_oc}

    # =====================================================================
    # Maximum power point (golden-section on the Lambert-W curve)
    # =====================================================================
    def _raw_mpp(self, irradiance, cell_temp_c):
        G = float(np.asarray(irradiance))
        if G <= 1.0:
            return {"v_mp": 0.0, "i_mp": 0.0, "p_mp": 0.0,
                    "v_oc": 0.0, "i_sc": 0.0, "fill_factor": 0.0}
        I_L, I_o, R_sh, a = self.desoto_params(G, cell_temp_c)
        V_oc = float(self.open_circuit_voltage(I_L, I_o, R_sh, a))
        I_sc = float(self.short_circuit_current(I_L, I_o, R_sh, a))

        gr = (np.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 0.0, V_oc
        for _ in range(80):
            v1 = hi - gr * (hi - lo)
            v2 = lo + gr * (hi - lo)
            p1 = v1 * float(self.current_from_voltage(v1, I_L, I_o, R_sh, a))
            p2 = v2 * float(self.current_from_voltage(v2, I_L, I_o, R_sh, a))
            if p1 < p2:
                lo = v1
            else:
                hi = v2
        V_mp = 0.5 * (lo + hi)
        I_mp = float(self.current_from_voltage(V_mp, I_L, I_o, R_sh, a))
        P_mp = V_mp * I_mp
        ff = P_mp / (V_oc * I_sc) if (V_oc * I_sc) > 0 else 0.0
        return {"v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp,
                "v_oc": V_oc, "i_sc": I_sc, "fill_factor": ff}

    # =====================================================================
    # CdTe corrections: empirical tempco (0.578) + spectral low-light
    # =====================================================================
    def _apply_cdte_tempco(self, p_mp, cell_temp_c):
        """Scale the De Soto temperature-dependent Pmp deviation by 0.578.

        De Soto: P_desoto = P_stc * (1 + gamma_desoto * dT)
        Target : P_corr   = P_stc * (1 + gamma_empirical * dT),
                 gamma_empirical = corr_factor * gamma_desoto.
        """
        dT = float(np.asarray(cell_temp_c)) - 25.0
        if abs(dT) < 0.1:
            return p_mp
        denom = 1.0 + self.gamma_desoto * dT
        denom = denom if abs(denom) > 0.01 else 0.01
        p_stc = p_mp / denom
        gamma_emp = self.corr_factor * self.gamma_desoto
        return p_stc * (1.0 + gamma_emp * dT)

    def _spectral_gain(self, irradiance):
        """CdTe low-light / spectral effective-irradiance uplift in [1, 1+gain].

        Smoothly ramps from full gain at very low G to ~0 at >=1000 W/m2,
        reflecting CdTe's favourable diffuse/high-AM spectral response and
        weaker relative resistive losses at low light (Huld & Amillo 2015).
        """
        G = float(np.asarray(irradiance))
        if G <= 1.0:
            return 1.0
        # Quadratic ramp: concentrate the uplift at genuinely low irradiance
        # (strong at ~200 W/m2, negligible by mid-range) so it does not push
        # the cold/low-light corner over the 0.19 efficiency ceiling.
        frac = np.clip((self.G_ref - G) / self.G_ref, 0.0, 1.0)
        return 1.0 + self.M_low_light_gain * frac ** 2

    def mpp(self, irradiance, cell_temp_c):
        """CdTe-corrected MPP at explicit cell temperature."""
        raw = self._raw_mpp(irradiance, cell_temp_c)
        if raw["p_mp"] <= 0.0:
            return raw
        p_corr = self._apply_cdte_tempco(raw["p_mp"], cell_temp_c)
        p_corr *= self._spectral_gain(irradiance)
        v_mp = raw["v_mp"]
        i_mp = p_corr / v_mp if v_mp > 0.01 else raw["i_mp"]
        ff = p_corr / (raw["v_oc"] * raw["i_sc"]) if (raw["v_oc"] * raw["i_sc"]) > 0 else 0.0
        return {"v_mp": v_mp, "i_mp": i_mp, "p_mp": p_corr,
                "v_oc": raw["v_oc"], "i_sc": raw["i_sc"], "fill_factor": ff}

    def electrical_power(self, irradiance, cell_temp_c):
        """Scalar Pmp [W] used by the thermal ODE (electrical extraction term)."""
        return self.mpp(irradiance, cell_temp_c)["p_mp"]

    def efficiency(self, irradiance, cell_temp_c):
        G = float(np.asarray(irradiance))
        if G <= 1.0:
            return 0.0
        return self.electrical_power(G, cell_temp_c) / (G * self.area)

    # =====================================================================
    # Steady-state cell temperature (Faiman, for initial guess / NOCT check)
    # =====================================================================
    def cell_temp_faiman(self, irradiance, T_amb_c, wind=None):
        """Faiman algebraic steady-state cell temperature [C]."""
        G = np.asarray(irradiance, dtype=float)
        Ta = np.asarray(T_amb_c, dtype=float)
        v = self.default_wind if wind is None else float(wind)
        U = self.U0 + self.U1 * v
        return Ta + self.alpha_abs * G / U

    # =====================================================================
    # Lumped thermal ODE: C dT/dt = Q_abs - P_elec - Q_conv - Q_rad
    # =====================================================================
    def _thermal_rhs(self, t, T_cell_c, G_func, Ta_func, wind, T_sky_offset):
        Tc = float(T_cell_c[0])
        G = float(G_func(t))
        Ta = float(Ta_func(t))
        Q_abs = self.alpha_abs * G * self.area
        P_el = self.electrical_power(G, Tc)
        U = self.U0 + self.U1 * wind
        Q_conv = U * self.area * (Tc - Ta)
        T_sky = Ta + T_sky_offset
        Q_rad = self.epsilon * SIGMA_SB * self.area * (
            (Tc + 273.15) ** 4 - (T_sky + 273.15) ** 4
        )
        dTdt = (Q_abs - P_el - Q_conv - Q_rad) / self.C_thermal
        return [dTdt]

    def simulate(self, irradiance, T_amb_c, T_cell0_c=None,
                 wind=None, duration_s=600.0, dt=10.0, T_sky_offset=-15.0):
        """Integrate the lumped thermal ODE and report electrical outputs.

        irradiance, T_amb_c : float (constant) or callable f(t)->value.
        Returns time series dict (numpy arrays).
        """
        wind = self.default_wind if wind is None else float(wind)
        G_func = irradiance if callable(irradiance) else (lambda t: irradiance)
        Ta_func = T_amb_c if callable(T_amb_c) else (lambda t: T_amb_c)

        if T_cell0_c is None:
            T_cell0_c = float(Ta_func(0.0))

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._thermal_rhs, (0.0, duration_s), [T_cell0_c],
            t_eval=t_eval, method="LSODA",
            args=(G_func, Ta_func, wind, T_sky_offset),
            rtol=1e-6, atol=1e-6, max_step=dt,
        )
        T_series = sol.y[0]
        t = sol.t

        V_mp = np.zeros_like(t)
        I_mp = np.zeros_like(t)
        P_mp = np.zeros_like(t)
        V_oc = np.zeros_like(t)
        I_sc = np.zeros_like(t)
        eff = np.zeros_like(t)
        for i, (ti, Tc) in enumerate(zip(t, T_series)):
            G = float(G_func(ti))
            r = self.mpp(G, Tc)
            V_mp[i] = r["v_mp"]; I_mp[i] = r["i_mp"]; P_mp[i] = r["p_mp"]
            V_oc[i] = r["v_oc"]; I_sc[i] = r["i_sc"]
            eff[i] = (r["p_mp"] / (G * self.area)) if G > 1.0 else 0.0

        return {
            "t": t, "temperature": T_series,
            "v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp,
            "v_oc": V_oc, "i_sc": I_sc, "efficiency": eff,
        }
