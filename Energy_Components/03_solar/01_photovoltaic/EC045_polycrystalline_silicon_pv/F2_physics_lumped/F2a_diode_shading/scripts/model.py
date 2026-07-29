"""
EC045 -- Polycrystalline Silicon PV -- F2a Physics-Lumped Single-Diode Model

Physics-lumped (0D) upgrade of the F1 single-diode model. Three additions make
this an F2 model rather than an algebraic F1:

  1. The full I-V / P-V curve is solved *closed-form* with the Lambert-W function
     (Jain & Kapoor 2004) instead of point Newton iterations, then MPP is located
     on the analytic curve.
  2. Cell temperature is a *dynamic state* governed by a lumped first-order
     thermal energy-balance ODE integrated with scipy.integrate.solve_ivp:
         C * dT_cell/dt = alpha*tau*G - U(w)*(T_cell - T_amb) - P_elec/A
     (Duffie & Beckman 2013; Faiman 2008 heat-loss coefficients).
  3. Partial shading of one bypass-diode substring is modelled: the shaded
     substring's photocurrent collapses, its bypass diode clamps it, and the
     module curve is reconstructed by series-adding substring voltages at common
     current (Bishop 1988 / standard bypass-diode treatment).

Single-diode equation (De Soto 5-parameter form):

    I = I_L - I_o * [exp((V + I*R_s) / a) - 1] - (V + I*R_s) / R_sh

with the De Soto reference-to-operating translation:

    I_L  = (G/G_ref) * [I_L_ref + alpha_sc*(T - T_ref)]
    I_o  = I_o_ref * (T/T_ref)^3 * exp[(Eg_ref/(k/q*T_ref)) - (Eg/(k/q*T))]
    a    = a_ref * T / T_ref
    R_sh = R_sh_ref * (G_ref / G)
    R_s  = constant

References:
    De Soto, W., Klein, S.A., Beckman, W.A. (2006). "Improvement and validation
        of a model for photovoltaic array performance." Solar Energy 80(1):78-88.
    Duffie, J.A., Beckman, W.A. (2013). "Solar Engineering of Thermal Processes,"
        4th ed., Wiley, Ch. 23 (Photovoltaic systems / cell energy balance).
    Faiman, D. (2008). "Assessing the outdoor operating temperature of
        photovoltaic modules." Prog. Photovolt. 16:307-315.
    Jain, A., Kapoor, A. (2004). "Exact analytical solutions of the parameters of
        real solar cells using Lambert W-function." Sol. Energy Mater. 81:269-277.
"""

import numpy as np
from scipy.special import lambertw
from scipy.integrate import solve_ivp


class PolySiPVF2a:
    """Poly-Si PV -- physics-lumped single-diode + thermal ODE + partial shading."""

    # Physical constants
    k = 1.380649e-23        # J/K  Boltzmann
    q = 1.602176634e-19     # C    elementary charge

    def __init__(self, params: dict):
        u = params["unit"]
        self.Ns = u["cells_in_series"]["value"]
        self.alpha_sc = u["alpha_sc"]["value"]
        self.area = u["area"]["value"]

        # De Soto reference parameters
        self.I_L_ref = u["I_L_ref"]["value"]
        self.I_o_ref = u["I_o_ref"]["value"]
        self.R_s = u["R_s"]["value"]
        self.R_sh_ref = u["R_sh_ref"]["value"]
        self.a_ref = u["a_ref"]["value"]
        self.EgRef = u["EgRef"]["value"]
        self.dEgdT = u["dEgdT"]["value"]

        # Thermal / NOCT
        self.NOCT = u["NOCT"]["value"]
        self.T_NOCT_amb = u["T_NOCT_amb"]["value"]
        self.G_NOCT = u["G_NOCT"]["value"]
        self.tau_alpha = u["tau_alpha"]["value"]
        self.C_thermal = u["C_thermal"]["value"]     # J/(m2.K)
        self.U0 = u["U0"]["value"]
        self.U1 = u["U1"]["value"]

        # Shading / bypass
        self.n_substrings = u["n_substrings"]["value"]
        self.V_bypass = u["V_bypass"]["value"]

        self.T_ref = 298.15      # K (STC)
        self.G_ref = 1000.0      # W/m2

    # ------------------------------------------------------------------
    # De Soto 5-parameter translation to operating (G, T)
    # ------------------------------------------------------------------
    def calc_params(self, G, T_cell_c):
        """Return (I_L, I_o, R_sh, a) at irradiance G [W/m2] and T_cell [degC]."""
        G = np.asarray(G, dtype=float)
        T = np.asarray(T_cell_c, dtype=float) + 273.15

        I_L = (G / self.G_ref) * (self.I_L_ref + self.alpha_sc * (T - self.T_ref))
        I_L = np.maximum(I_L, 0.0)

        Eg = self.EgRef * (1.0 + self.dEgdT * (T - self.T_ref))
        I_o = self.I_o_ref * (T / self.T_ref) ** 3 * np.exp(
            (self.EgRef / (self.k / self.q * self.T_ref))
            - (Eg / (self.k / self.q * T))
        )
        a = self.a_ref * T / self.T_ref
        R_sh = self.R_sh_ref * (self.G_ref / np.maximum(G, 1.0))
        return I_L, I_o, R_sh, a

    # ------------------------------------------------------------------
    # Exact I(V) via Lambert-W  (Jain & Kapoor 2004)
    #   I = (R_sh*(I_L+I_o) - V) / (R_s + R_sh)
    #       - (a/R_s) * W{ z }
    #   z = (R_s*R_sh*I_o)/(a*(R_s+R_sh))
    #         * exp[ R_sh*(R_s*I_L + R_s*I_o + V)/(a*(R_s+R_sh)) ]
    # ------------------------------------------------------------------
    def i_from_v(self, V, I_L, I_o, R_sh, a):
        V = np.asarray(V, dtype=float)
        Rs, Rsh = self.R_s, R_sh
        denom = Rs + Rsh
        arg = (Rsh * (Rs * I_L + Rs * I_o + V)) / (a * denom)
        arg = np.clip(arg, -700.0, 700.0)
        z = (Rs * Rsh * I_o) / (a * denom) * np.exp(arg)
        # principal branch is real & non-negative for z >= 0
        W = np.real(lambertw(z, 0))
        I = (Rsh * (I_L + I_o) - V) / denom - (a / Rs) * W
        return I

    # ------------------------------------------------------------------
    # Terminal voltage at a given terminal current I.
    #
    # The dual Lambert-W form V(I) is numerically ill-conditioned for the
    # large shunt resistances of a full module (the W term must cancel a huge
    # linear R_sh term). We instead invert the *stable* forward Lambert-W
    # current curve i_from_v() by a robust bracketed bisection on V. This is
    # still a closed-form-per-evaluation curve (no global iteration over the
    # whole I-V sweep), and matches the single-diode equation to ~1e-9.
    # ------------------------------------------------------------------
    def v_from_i(self, I_target, I_L, I_o, R_sh, a, vmax=None):
        """Voltage such that i_from_v(V) == I_target, by bisection on V."""
        if vmax is None:
            # generous upper bound: open-circuit of ideal diode
            vmax = a * np.log(max(I_L / max(I_o, 1e-30), 1.0) + 1.0) + I_L * self.R_s
        lo, hi = -2.0, float(vmax) * 1.5 + 2.0
        f_lo = float(self.i_from_v(lo, I_L, I_o, R_sh, a)) - I_target
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = float(self.i_from_v(mid, I_L, I_o, R_sh, a)) - I_target
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo = mid
                f_lo = f_mid
        return 0.5 * (lo + hi)

    def v_oc(self, I_L, I_o, R_sh, a):
        return max(self.v_from_i(0.0, I_L, I_o, R_sh, a), 0.0)

    # ------------------------------------------------------------------
    # I-V / P-V curve and MPP for a single (un-shaded) module
    # ------------------------------------------------------------------
    def iv_curve(self, G, T_cell_c, n_points=200):
        I_L, I_o, R_sh, a = self.calc_params(G, T_cell_c)
        if np.asarray(G).item() <= 1.0:
            V = np.linspace(0.0, 0.0, n_points)
            return V, np.zeros_like(V), np.zeros_like(V)
        Voc = float(self.v_oc(I_L, I_o, R_sh, a))
        V = np.linspace(0.0, Voc, n_points)
        I = self.i_from_v(V, I_L, I_o, R_sh, a)
        I = np.maximum(I, 0.0)
        P = V * I
        return V, I, P

    def mpp(self, G, T_cell_c):
        """Maximum-power point of the unshaded module. Returns dict of scalars."""
        G = float(np.asarray(G).item()) if np.asarray(G).ndim else float(G)
        if G <= 1.0:
            return {"v_mp": 0.0, "i_mp": 0.0, "p_mp": 0.0, "v_oc": 0.0, "i_sc": 0.0}

        I_L, I_o, R_sh, a = self.calc_params(G, T_cell_c)
        Voc = float(self.v_oc(I_L, I_o, R_sh, a))
        Isc = float(self.i_from_v(0.0, I_L, I_o, R_sh, a))

        # Golden-section search for V_mp on the analytic Lambert-W curve
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 0.0, Voc
        for _ in range(80):
            v1 = hi - gr * (hi - lo)
            v2 = lo + gr * (hi - lo)
            p1 = v1 * float(self.i_from_v(v1, I_L, I_o, R_sh, a))
            p2 = v2 * float(self.i_from_v(v2, I_L, I_o, R_sh, a))
            if p1 < p2:
                lo = v1
            else:
                hi = v2
        Vmp = 0.5 * (lo + hi)
        Imp = float(self.i_from_v(Vmp, I_L, I_o, R_sh, a))
        Pmp = Vmp * Imp
        return {"v_mp": Vmp, "i_mp": Imp, "p_mp": Pmp, "v_oc": Voc, "i_sc": Isc}

    def efficiency(self, G, T_cell_c):
        G = float(G)
        if G <= 1.0:
            return 0.0
        return self.mpp(G, T_cell_c)["p_mp"] / (G * self.area)

    # ------------------------------------------------------------------
    # Partial shading of one bypass-diode substring
    # ------------------------------------------------------------------
    def mpp_partial_shade(self, G, T_cell_c, shade_fraction, n_points=400):
        """
        Module split into n_substrings series substrings, each bypass-protected.
        One substring receives reduced irradiance G_shaded = (1-shade_fraction)*G.

        Reconstruct the module I-V curve by sweeping module current I and summing
        substring voltages at that common current; a substring whose own curve
        cannot carry I is clamped at its bypass-diode drop (-V_bypass).
        Returns dict with module v_mp, i_mp, p_mp plus per-substring info.
        """
        G = float(G)
        ns = int(self.n_substrings)
        cells_per_sub = self.Ns / ns

        if G <= 1.0 or shade_fraction <= 0.0:
            base = self.mpp(G, T_cell_c)
            base["shaded"] = bool(shade_fraction > 0.0 and G > 1.0)
            return base

        G_lit = G
        G_shaded = max((1.0 - shade_fraction) * G, 0.0)

        # Per-substring single-diode params: scale a, I_L, R_sh, I_o by cells_per_sub
        def sub_params(Gx):
            I_L, I_o, R_sh, a = self.calc_params(max(Gx, 1.0), T_cell_c)
            frac = cells_per_sub / self.Ns
            return I_L, I_o, R_sh * frac, a * frac

        lit = sub_params(G_lit)
        sh = sub_params(G_shaded) if G_shaded > 1.0 else None
        Rs_sub = self.R_s * (cells_per_sub / self.Ns)

        def sub_i_from_v(V, p):
            """Forward current of one substring at terminal voltage V (Lambert-W)."""
            I_L, I_o, R_sh, a = p
            denom = Rs_sub + R_sh
            arg = np.clip((R_sh * (Rs_sub * I_L + Rs_sub * I_o + V)) / (a * denom),
                          -700.0, 700.0)
            z = (Rs_sub * R_sh * I_o) / (a * denom) * np.exp(arg)
            W = np.real(lambertw(z, 0))
            return (R_sh * (I_L + I_o) - V) / denom - (a / Rs_sub) * W

        def sub_voltage(I, p):
            """Voltage of a substring carrying current I, by bisection on V."""
            I_L, I_o, R_sh, a = p
            vmax = a * np.log(max(I_L / max(I_o, 1e-30), 1.0) + 1.0) + I_L * Rs_sub
            lo, hi = -2.0, float(vmax) * 1.5 + 2.0
            f_lo = float(sub_i_from_v(lo, p)) - I
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                f_mid = float(sub_i_from_v(mid, p)) - I
                if f_lo * f_mid <= 0.0:
                    hi = mid
                else:
                    lo = mid
                    f_lo = f_mid
            return 0.5 * (lo + hi)

        # Module short-circuit current = that of the strongest (lit) substring
        Isc_lit = float(sub_i_from_v(0.0, lit))
        Isc_sh = float(sub_i_from_v(0.0, sh)) if sh is not None else 0.0
        I_sweep = np.linspace(0.0, Isc_lit, n_points)

        n_lit = ns - 1
        V_mod = np.zeros_like(I_sweep)
        for idx, I in enumerate(I_sweep):
            # lit substrings
            v_lit = float(sub_voltage(I, lit))
            v_lit = max(v_lit, -self.V_bypass)
            # shaded substring: if its short-circuit current < I, bypass clamps it
            if sh is None:
                v_sh = -self.V_bypass
            else:
                if I > Isc_sh:
                    v_sh = -self.V_bypass
                else:
                    v_sh = max(float(sub_voltage(I, sh)), -self.V_bypass)
            V_mod[idx] = n_lit * v_lit + v_sh

        V_mod = np.maximum(V_mod, 0.0)
        P_mod = V_mod * I_sweep
        imax = int(np.argmax(P_mod))
        return {
            "v_mp": float(V_mod[imax]),
            "i_mp": float(I_sweep[imax]),
            "p_mp": float(P_mod[imax]),
            "v_oc": float(np.max(V_mod)),
            "i_sc": float(Isc_lit),
            "shaded": True,
            "V_curve": V_mod,
            "I_curve": I_sweep,
            "P_curve": P_mod,
        }

    # ------------------------------------------------------------------
    # Cell-temperature ODE (lumped energy balance)  -- Duffie & Beckman
    #   C * dT/dt = alpha*tau*G - U(w)*(T - T_amb) - P_elec/A
    # U(w) = U0 + U1*wind   (Faiman 2008)
    # ------------------------------------------------------------------
    def _U(self, wind):
        return self.U0 + self.U1 * wind

    def _steady_cell_temp(self, G, T_amb, wind):
        """Algebraic steady-state cell temp (initial condition / NOCT-consistent)."""
        if G <= 1.0:
            return T_amb
        U = self._U(wind)
        # ignore the small electrical extraction term for the IC estimate
        return T_amb + (self.tau_alpha * G) / U

    def _dTdt(self, t, T_cell, G_func, Tamb_func, wind_func):
        Tc = float(T_cell[0])
        G = float(G_func(t))
        Tamb = float(Tamb_func(t))
        wind = float(wind_func(t))
        U = self._U(wind)
        if G <= 1.0:
            P_elec_area = 0.0
        else:
            P_elec_area = self.mpp(G, Tc)["p_mp"] / self.area  # W/m2 extracted
        q_in = self.tau_alpha * G
        q_loss = U * (Tc - Tamb)
        return [(q_in - q_loss - P_elec_area) / self.C_thermal]

    def simulate(self, G, T_amb=25.0, wind=1.0, dt=60.0, duration_s=3600.0,
                 T_cell0=None, shade_fraction=0.0):
        """
        Dynamic simulation: integrate the cell-temperature ODE with solve_ivp and
        evaluate MPP / efficiency at each output time. G, T_amb, wind may be
        scalars or callables of t (seconds).

        Returns dict of time-series arrays.
        """
        G_func = G if callable(G) else (lambda t, g=G: g)
        Tamb_func = T_amb if callable(T_amb) else (lambda t, a=T_amb: a)
        wind_func = wind if callable(wind) else (lambda t, w=wind: w)

        if T_cell0 is None:
            T_cell0 = self._steady_cell_temp(G_func(0.0), Tamb_func(0.0), wind_func(0.0))

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(
            self._dTdt, (0.0, duration_s), [T_cell0],
            t_eval=t_eval, args=(G_func, Tamb_func, wind_func),
            method="RK45", rtol=1e-6, atol=1e-6, max_step=dt,
        )
        T_cell = sol.y[0]
        t = sol.t

        v_mp = np.zeros_like(t)
        i_mp = np.zeros_like(t)
        p_mp = np.zeros_like(t)
        v_oc = np.zeros_like(t)
        i_sc = np.zeros_like(t)
        eff = np.zeros_like(t)
        for n, (ti, Tci) in enumerate(zip(t, T_cell)):
            Gi = float(G_func(ti))
            if shade_fraction > 0.0:
                r = self.mpp_partial_shade(Gi, Tci, shade_fraction)
            else:
                r = self.mpp(Gi, Tci)
            v_mp[n] = r["v_mp"]; i_mp[n] = r["i_mp"]; p_mp[n] = r["p_mp"]
            v_oc[n] = r["v_oc"]; i_sc[n] = r["i_sc"]
            eff[n] = (p_mp[n] / (Gi * self.area)) if Gi > 1.0 else 0.0

        return {
            "t": t,
            "T_cell": T_cell,
            "v_mp": v_mp,
            "i_mp": i_mp,
            "p_mp": p_mp,
            "v_oc": v_oc,
            "i_sc": i_sc,
            "efficiency": eff,
            "G": np.array([float(G_func(ti)) for ti in t]),
        }
