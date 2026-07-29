"""
EC052 -- Bifacial PV Module -- F2a Physics-Lumped (Single-Diode + Thermal ODE)

Physics-lumped (0D) upgrade of the F1a bifacial-gain model. Two coupled pieces:

1. ELECTRICAL -- bifacial effective irradiance feeding a single-diode cell.

   Effective irradiance (Marion et al. 2017 NREL view-factor model):
       G_rear  = albedo * G_front * F_view      (if rear not supplied directly)
       G_eff   = G_front + phi * G_rear
   where phi = bifaciality factor (rear/front nameplate response, ~0.7-0.85)
   and F_view is the geometric view factor (fraction of ground-reflected
   irradiance intercepted by the rear face; tilt/height/pitch dependent).

   The single-diode (5-parameter De Soto) equation
       I = I_L - I_o*(exp((V + I*Rs)/a) - 1) - (V + I*Rs)/Rsh
   is solved in CLOSED FORM with the Lambert-W function (Jain & Kapoor 2004),
   giving I(V) without iteration. Voc is found by Lambert-W on I=0, and the
   maximum-power point (MPP) by a 1-D root-find on d(P)/dV = 0 (Brent).

2. THERMAL -- lumped 0D module-temperature ODE (Faiman 2008 / Jones &
   Underwood 2001):
       C * dT_cell/dt = alpha * G_poa - U(v_wind)*(T_cell - T_amb) - P_elec/A
   with U(v) = U0 + U1*v_wind (Faiman wind-dependent loss coefficient).
   The electrical power extracted is removed from the heat budget (the part
   of absorbed sunlight that leaves as electricity does not heat the module).
   Integrated with scipy.integrate.solve_ivp.

The single-diode parameters translate from STC to operating (G_eff, T_cell)
exactly as in De Soto et al. (2006).

References:
    De Soto, Klein & Beckman (2006), "Improvement and validation of a model
        for photovoltaic array performance", Solar Energy 80(1), 78-88.
    Jain & Kapoor (2004), "Exact analytical solutions of the parameters of
        real solar cells using Lambert W-function", Sol. Energy Mater. 81, 269.
    Marion et al. (2017), "A Practical Irradiance Model for Bifacial PV
        Modules", NREL / IEEE PVSC-44.
    Faiman (2008), "Assessing the outdoor operating temperature of PV
        modules", Prog. Photovolt. 16, 307-315.
    Jones & Underwood (2001), "A thermal model for photovoltaic systems",
        Solar Energy 70(4), 349-359.
"""

import numpy as np
from scipy.special import lambertw
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class BifacialPV_F2a:
    """Bifacial PV -- Lambert-W single-diode + lumped thermal ODE."""

    k = 1.380649e-23        # Boltzmann constant [J/K]
    q = 1.602176634e-19     # elementary charge [C]
    T_ref = 298.15          # STC cell temperature [K]
    G_ref = 1000.0          # STC irradiance [W/m2]

    def __init__(self, params: dict):
        mod = params["module"]
        ds = params["desoto_params"]
        th = params["thermal"]

        self.area = mod["area"]["value"]
        self.alpha_sc = mod["alpha_sc"]["value"]
        self.phi = mod["bifaciality_factor"]["value"]
        self.F_view = mod["rear_view_factor"]["value"]

        self.I_L_ref = ds["I_L_ref"]["value"]
        self.I_o_ref = ds["I_o_ref"]["value"]
        self.R_s = ds["R_s"]["value"]
        self.R_sh_ref = ds["R_sh_ref"]["value"]
        self.a_ref = ds["a_ref"]["value"]
        self.EgRef = ds["EgRef"]["value"]
        self.dEgdT = ds["dEgdT"]["value"]

        self.C_th = th["C_thermal"]["value"]      # J/(m2.K)
        self.U0 = th["U0"]["value"]               # W/(m2.K)
        self.U1 = th["U1"]["value"]               # W.s/(m3.K)
        self.absorptance = th["absorptance"]["value"]

    # ------------------------------------------------------------------
    # Bifacial effective irradiance (Marion 2017 view-factor model)
    # ------------------------------------------------------------------
    def effective_irradiance(self, G_front, G_rear=None, albedo=None):
        G_f = float(G_front)
        if G_rear is not None:
            G_r = float(G_rear)
        elif albedo is not None:
            G_r = float(albedo) * G_f * self.F_view
        else:
            G_r = 0.0
        return G_f + self.phi * G_r, G_r

    def poa_irradiance(self, G_front, G_rear):
        """Total plane-of-array irradiance hitting the module (for heat budget)."""
        return float(G_front) + float(G_rear)

    # ------------------------------------------------------------------
    # Single-diode parameter translation STC -> (G_eff, T_cell) [De Soto 2006]
    # ------------------------------------------------------------------
    def _calc_params(self, G_eff, T_cell_K):
        G = max(float(G_eff), 0.0)
        T = float(T_cell_K)

        I_L = (G / self.G_ref) * (self.I_L_ref + self.alpha_sc * (T - self.T_ref))
        I_L = max(I_L, 0.0)

        Eg = self.EgRef * (1.0 + self.dEgdT * (T - self.T_ref) / self.EgRef)
        Vt_q = self.k / self.q
        I_o = self.I_o_ref * (T / self.T_ref) ** 3 * np.exp(
            (self.EgRef / (Vt_q * self.T_ref)) - (Eg / (Vt_q * T))
        )

        a = self.a_ref * T / self.T_ref
        R_sh = self.R_sh_ref * (self.G_ref / max(G, 1e-6))
        return I_L, I_o, R_sh, a

    # ------------------------------------------------------------------
    # Numerically stable Lambert-W of exp(ln_z), i.e. W(z) given ln(z).
    # For PV translation z is astronomically large (ln z ~ 3000), so we never
    # form z directly. Solve w + ln(w) = ln_z by Newton (principal branch).
    # ------------------------------------------------------------------
    @staticmethod
    def _lambertw_from_log(ln_z):
        ln_z = float(ln_z)
        if ln_z < 20.0:
            # small/moderate argument: form z and use scipy directly
            return float(np.real(lambertw(np.exp(ln_z))))
        # large argument asymptotic seed: W ~ L1 - L2 + L2/L1
        L1 = ln_z
        L2 = np.log(ln_z)
        w = L1 - L2 + L2 / L1
        for _ in range(60):
            # f(w) = w + ln(w) - ln_z = 0
            f = w + np.log(w) - ln_z
            df = 1.0 + 1.0 / w
            w_new = w - f / df
            if abs(w_new - w) < 1e-12 * abs(w_new):
                w = w_new
                break
            w = w_new
        return float(w)

    # ------------------------------------------------------------------
    # Lambert-W closed-form I(V)  [Jain & Kapoor 2004]
    # ------------------------------------------------------------------
    def current_from_voltage(self, V, I_L, I_o, R_sh, a):
        """
        Exact single-diode current at terminal voltage V via Lambert-W.

        Solving I = I_L - I_o(exp((V+I*Rs)/a)-1) - (V+I*Rs)/Rsh gives
            I = (Rsh*(I_L+I_o) - V) / (Rs+Rsh)
                - (a/Rs) * W( z )
        with
            z = (Rs*Io*Rsh / (a*(Rs+Rsh))) * exp( Rsh*(Rs*(I_L+Io)+V)
                                                   / (a*(Rs+Rsh)) ).
        We pass ln(z) to a stable Lambert-W to avoid overflow.
        """
        Rs, Rsh = self.R_s, R_sh
        if I_o <= 0.0:
            return 0.0
        arg = (Rsh * (Rs * (I_L + I_o) + V)) / (a * (Rs + Rsh))
        ln_z = np.log(Rs * I_o * Rsh / (a * (Rs + Rsh))) + arg
        w = self._lambertw_from_log(ln_z)
        I = (Rsh * (I_L + I_o) - V) / (Rs + Rsh) - (a / Rs) * w
        return I

    def voc(self, I_L, I_o, R_sh, a):
        """Open-circuit voltage: solve I(V)=0 with Lambert-W on the I=0 branch."""
        if I_L <= 0.0 or I_o <= 0.0:
            return 0.0
        # At I=0: 0 = I_L - I_o(exp(V/a)-1) - V/Rsh -> Lambert-W form (stable).
        ln_z = np.log(I_o * R_sh / a) + R_sh * (I_L + I_o) / a
        w = self._lambertw_from_log(ln_z)
        V_oc = R_sh * (I_L + I_o) - a * w
        return max(float(V_oc), 0.0)

    def isc(self, I_L, I_o, R_sh, a):
        """Short-circuit current: I at V=0."""
        return float(self.current_from_voltage(0.0, I_L, I_o, R_sh, a))

    # ------------------------------------------------------------------
    # I-V / P-V curves and MPP
    # ------------------------------------------------------------------
    def iv_curve(self, G_eff, T_cell_C, n=200):
        I_L, I_o, R_sh, a = self._calc_params(G_eff, T_cell_C + 273.15)
        V_oc = self.voc(I_L, I_o, R_sh, a)
        if V_oc <= 0.0:
            V = np.zeros(n)
            return V, np.zeros(n), V
        V = np.linspace(0.0, V_oc, n)
        I = np.array([self.current_from_voltage(v, I_L, I_o, R_sh, a) for v in V])
        I = np.maximum(I, 0.0)
        P = V * I
        return V, I, P

    def mpp(self, G_front, T_cell_C, G_rear=None, albedo=None):
        """Maximum-power point via root-find on dP/dV = 0. T_cell_C in degC."""
        G_eff, G_rear_used = self.effective_irradiance(
            G_front, G_rear=G_rear, albedo=albedo)
        I_L, I_o, R_sh, a = self._calc_params(G_eff, T_cell_C + 273.15)
        V_oc = self.voc(I_L, I_o, R_sh, a)
        I_sc = self.isc(I_L, I_o, R_sh, a)

        if G_eff <= 1.0 or V_oc <= 1e-9:
            return {"v_mp": 0.0, "i_mp": 0.0, "p_mp": 0.0,
                    "v_oc": 0.0, "i_sc": 0.0,
                    "G_effective": G_eff, "G_rear_used": G_rear_used}

        def dPdV(v):
            h = max(V_oc * 1e-6, 1e-9)
            p_plus = (v + h) * self.current_from_voltage(v + h, I_L, I_o, R_sh, a)
            p_minus = (v - h) * self.current_from_voltage(v - h, I_L, I_o, R_sh, a)
            return (p_plus - p_minus) / (2.0 * h)

        lo, hi = 0.30 * V_oc, 0.999 * V_oc
        try:
            if dPdV(lo) > 0 and dPdV(hi) < 0:
                V_mp = brentq(dPdV, lo, hi, xtol=1e-6)
            else:
                # fallback: dense scan
                V = np.linspace(0.0, V_oc, 400)
                P = V * np.array(
                    [self.current_from_voltage(v, I_L, I_o, R_sh, a) for v in V])
                V_mp = V[int(np.argmax(P))]
        except ValueError:
            V = np.linspace(0.0, V_oc, 400)
            P = V * np.array(
                [self.current_from_voltage(v, I_L, I_o, R_sh, a) for v in V])
            V_mp = V[int(np.argmax(P))]

        I_mp = max(self.current_from_voltage(V_mp, I_L, I_o, R_sh, a), 0.0)
        P_mp = V_mp * I_mp
        return {"v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp,
                "v_oc": V_oc, "i_sc": I_sc,
                "G_effective": G_eff, "G_rear_used": G_rear_used}

    def efficiency(self, G_front, T_cell_C, G_rear=None, albedo=None):
        r = self.mpp(G_front, T_cell_C, G_rear=G_rear, albedo=albedo)
        G_f = float(G_front)
        if G_f <= 1.0:
            return 0.0
        return r["p_mp"] / (G_f * self.area)

    def bifacial_gain(self, G_front, T_cell_C, G_rear=None, albedo=None):
        """(P_bifacial - P_front_only) / P_front_only. T_cell_C in degC."""
        r_bi = self.mpp(G_front, T_cell_C, G_rear=G_rear, albedo=albedo)
        r_mono = self.mpp(G_front, T_cell_C, G_rear=0.0, albedo=0.0)
        P_mono, P_bi = r_mono["p_mp"], r_bi["p_mp"]
        if P_mono <= 1.0:
            return 0.0
        return (P_bi - P_mono) / P_mono

    # ------------------------------------------------------------------
    # Lumped thermal ODE  [Faiman 2008 / Jones & Underwood 2001]
    # ------------------------------------------------------------------
    def _heat_balance(self, T_cell_K, G_poa, G_front, T_amb_K, v_wind,
                      G_rear, albedo):
        """RHS components of C*dT/dt [W/m2]."""
        U = self.U0 + self.U1 * v_wind
        Q_abs = self.absorptance * G_poa
        Q_loss = U * (T_cell_K - T_amb_K)
        # power extracted electrically removed from heat budget (per m2)
        r = self.mpp(G_front, T_cell_K - 273.15, G_rear=G_rear, albedo=albedo)
        Q_elec = r["p_mp"] / self.area
        return Q_abs - Q_loss - Q_elec

    def simulate(self, G_front, T_amb_C=25.0, v_wind=1.0,
                 G_rear=None, albedo=None, T_cell0_C=None,
                 dt=60.0, duration_s=3600.0):
        """
        Transient simulation. G_front / T_amb_C / v_wind may be scalars or
        callables of time t [s]. Integrates the lumped thermal ODE and reports
        the coupled electrical MPP at each output step.
        """
        T_amb_f = T_amb_C if callable(T_amb_C) else (lambda t: T_amb_C)
        G_f = G_front if callable(G_front) else (lambda t: G_front)
        v_f = v_wind if callable(v_wind) else (lambda t: v_wind)

        if T_cell0_C is None:
            T_cell0_C = T_amb_f(0.0)
        T0 = T_cell0_C + 273.15

        def rhs(t, y):
            T_cell = y[0]
            Gfr = G_f(t)
            Tamb = T_amb_f(t) + 273.15
            vw = v_f(t)
            _, G_r = self.effective_irradiance(Gfr, G_rear=G_rear, albedo=albedo)
            G_poa = self.poa_irradiance(Gfr, G_r)
            dT = self._heat_balance(T_cell, G_poa, Gfr, Tamb, vw, G_rear, albedo)
            return [dT / self.C_th]

        n_steps = max(int(round(duration_s / dt)), 1)
        t_eval = np.linspace(0.0, duration_s, n_steps + 1)
        sol = solve_ivp(rhs, (0.0, duration_s), [T0], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-6, max_step=dt)

        t = sol.t
        T_cell_K = sol.y[0]
        T_cell_C = T_cell_K - 273.15

        v_mp = np.zeros_like(t)
        i_mp = np.zeros_like(t)
        p_mp = np.zeros_like(t)
        v_oc = np.zeros_like(t)
        i_sc = np.zeros_like(t)
        eff = np.zeros_like(t)
        G_eff = np.zeros_like(t)
        bgain = np.zeros_like(t)
        for k_i, (tk, Tk_C) in enumerate(zip(t, T_cell_C)):
            Gfr = G_f(tk)
            r = self.mpp(Gfr, Tk_C, G_rear=G_rear, albedo=albedo)
            v_mp[k_i] = r["v_mp"]
            i_mp[k_i] = r["i_mp"]
            p_mp[k_i] = r["p_mp"]
            v_oc[k_i] = r["v_oc"]
            i_sc[k_i] = r["i_sc"]
            G_eff[k_i] = r["G_effective"]
            eff[k_i] = (r["p_mp"] / (Gfr * self.area)) if Gfr > 1.0 else 0.0
            bgain[k_i] = self.bifacial_gain(Gfr, Tk_C, G_rear=G_rear, albedo=albedo)

        return {
            "t": t,
            "temperature_C": T_cell_C,
            "temperature_K": T_cell_K,
            "v_mp": v_mp, "i_mp": i_mp, "p_mp": p_mp,
            "v_oc": v_oc, "i_sc": i_sc,
            "efficiency": eff,
            "G_effective": G_eff,
            "bifacial_gain": bgain,
        }
