"""
EC047 -- Thin-Film CIGS PV -- F2a Physics-Lumped Single-Diode + Thermal ODE

Physics-lumped (0D) upgrade of the F1 De Soto single-diode device. Two pieces:

1. ELECTRICAL -- De Soto 5-parameter single-diode equation, solved in closed
   form with the Lambert-W function (Jain & Kapoor 2004) and a Halley root-find
   fallback. For a load voltage V the cell current satisfies

       I = I_L - I_o*[exp((V + I*R_s)/a) - 1] - (V + I*R_s)/R_sh

   which is transcendental in I. Jain & Kapoor (2004) showed the exact solution is

       I = (R_sh*(I_L + I_o) - V)/(R_s + R_sh)
           - (a/R_s) * W0( z )
       z = (R_s*I_o*R_sh / (a*(R_s+R_sh)))
           * exp( R_sh*(R_s*I_L + R_s*I_o + V) / (a*(R_s+R_sh)) )

   where W0 is the principal branch of the Lambert-W function (scipy.special.lambertw).

2. THERMAL -- lumped 1-node energy balance integrated with scipy.solve_ivp:

       C_th * dT/dt = alpha*G - P_elec/A - (U0 + U1*v_wind)*(T - T_amb)

   i.e. absorbed solar minus electrical power extracted minus convective/radiative
   loss (Faiman 2008 wind-dependent U model). C_th is the areal heat capacity of
   the glass/CIGS/glass laminate. Steady state of this ODE reproduces the Faiman
   module-temperature correlation used in F1b, but the transient captures the
   thermal lag (minutes) of the laminate -- the F2 value-add over algebraic F1.

De Soto temperature/irradiance dependence of the five parameters:
       I_L  = (G/G_ref)*(I_L_ref + alpha_sc*(T - T_ref))
       I_o  = I_o_ref*(T/T_ref)^3 * exp( (q/k)*(Eg_ref/T_ref - Eg/T) )
       a    = a_ref * T/T_ref
       R_sh = R_sh_ref * G_ref/G
       R_s  = R_s (constant)
       Eg   = Eg_ref*(1 + dEgdT*(T - T_ref)/Eg_ref)

CIGS specifics
--------------
  * Bandgap ~1.15 eV, tunable 1.0-1.7 eV through the Ga/(In+Ga) ratio
    (CuInSe2 ~1.0 eV -> CuGaSe2 ~1.7 eV). Exposed as `set_bandgap()`.
  * Low temperature coefficient: gamma_pmp ~ -0.31 %/K, smaller in magnitude
    than poly-Si (-0.39 %/K) -> better hot-climate yield (Green et al. 2019).
  * Metastability / light-soaking: CIGS gains a few % in efficiency after light
    exposure (Vbi increase, shallow-defect relaxation). Captured here only as an
    optional multiplicative `light_soak_gain` on I_L (default 1.0); see note in
    `light_soaking_note()`. NOT a full metastability ODE (that would be F3).

References
----------
  De Soto, Klein & Beckman (2006). Improvement and validation of a model for
      photovoltaic array performance. Solar Energy 80(1), 78-88.
  Jain & Kapoor (2004). Exact analytical solutions of the parameters of real
      solar cells using Lambert W-function. Sol. Energy Mater. Sol. Cells 81(2).
  Faiman (2008). Assessing the outdoor operating temperature of photovoltaic
      modules. Progress in Photovoltaics 16(4), 307-315.
  Jordan & Kurtz (2013). Photovoltaic degradation rates -- an analytical review.
      Progress in Photovoltaics 21(1), 12-29.
  Green et al. (2019). Solar cell efficiency tables (v53). Prog. Photovolt. 27(1).
"""

import numpy as np
from scipy.special import lambertw
from scipy.integrate import solve_ivp

# Physical constants
_K_B = 1.380649e-23      # Boltzmann constant [J/K]
_Q = 1.602176634e-19     # elementary charge [C]


class CIGSPvF2a:
    """CIGS thin-film PV -- physics-lumped single-diode + thermal ODE."""

    T_REF = 298.15        # STC cell temperature [K]
    G_REF = 1000.0        # STC irradiance [W/m2]

    def __init__(self, params: dict):
        u = params["unit"]
        self.cells_in_series = u["cells_in_series"]["value"]
        self.area = u["area"]["value"]
        self.alpha_sc = u["alpha_sc"]["value"]

        self.I_L_ref = u["I_L_ref"]["value"]
        self.I_o_ref = u["I_o_ref"]["value"]
        self.R_s = u["R_s"]["value"]
        self.R_sh_ref = u["R_sh_ref"]["value"]
        self.a_ref = u["a_ref"]["value"]
        self.EgRef = u["EgRef"]["value"]
        self.dEgdT = u["dEgdT"]["value"]

        # thermal
        self.C_th = u["thermal_mass"]["value"]      # J/(m2.K)
        self.U0 = u["U_const"]["value"]             # W/(m2.K)
        self.U1 = u["U_wind"]["value"]              # W.s/(m3.K)
        self.absorptance = u["absorptance"]["value"]

        # CIGS metastability / light-soaking multiplier on I_L (1.0 = as-fitted)
        self.light_soak_gain = 1.0

    # ------------------------------------------------------------------
    # CIGS bandgap tunability (Ga/(In+Ga) grading)
    # ------------------------------------------------------------------
    def set_bandgap(self, Eg_eV):
        """Set the CIGS bandgap [eV]; physical window 1.0-1.7 eV.

        Re-grading the Ga/(In+Ga) ratio shifts the bandgap. The diode reverse-
        saturation current scales as I_o ~ exp(-Eg/(k*T)) (Sze, Physics of
        Semiconductor Devices), so a wider gap lowers I_o and raises Voc. We
        rescale I_o_ref consistently with the change relative to the current gap.
        """
        if not (0.9 <= Eg_eV <= 1.8):
            raise ValueError("CIGS bandgap out of physical window [1.0, 1.7] eV")
        kq = _K_B / _Q  # k/q [V/K]
        self.I_o_ref *= np.exp(-(Eg_eV - self.EgRef) / (kq * self.T_REF))
        self.EgRef = float(Eg_eV)

    def light_soaking_note(self):
        return ("CIGS exhibits beneficial metastability: under light soaking the "
                "open-circuit voltage and fill factor recover/improve by a few "
                "percent (built-in potential increase, shallow-defect relaxation). "
                "Modelled here as an optional multiplicative gain on I_L "
                "(self.light_soak_gain); full transition kinetics are F3-level.")

    # ------------------------------------------------------------------
    # De Soto temperature/irradiance translation of the 5 parameters
    # ------------------------------------------------------------------
    def _calc_params(self, G, T_cell_K):
        G = np.asarray(G, dtype=float)
        T = np.asarray(T_cell_K, dtype=float)
        Geff = np.maximum(G, 1e-9)

        I_L = (G / self.G_REF) * (self.I_L_ref + self.alpha_sc * (T - self.T_REF))
        I_L = np.maximum(I_L, 0.0) * self.light_soak_gain

        Eg = self.EgRef * (1.0 + self.dEgdT * (T - self.T_REF) / self.EgRef)
        kq = _K_B / _Q  # k/q in V/K
        I_o = self.I_o_ref * (T / self.T_REF) ** 3 * np.exp(
            (self.EgRef / (kq * self.T_REF)) - (Eg / (kq * T))
        )
        a = self.a_ref * T / self.T_REF
        R_sh = self.R_sh_ref * (self.G_REF / Geff)
        return I_L, I_o, R_sh, a

    # ------------------------------------------------------------------
    # Exact I(V) via Lambert-W  (Jain & Kapoor 2004)
    # ------------------------------------------------------------------
    def current_from_voltage(self, V, G, T_cell_K):
        """Cell/module current [A] at load voltage V via Lambert-W."""
        I_L, I_o, R_sh, a = self._calc_params(G, T_cell_K)
        return self._i_from_v(np.asarray(V, dtype=float), I_L, I_o, R_sh, a)

    def _i_from_v(self, V, I_L, I_o, R_sh, a):
        Rs, Rsh = self.R_s, R_sh
        # argument of Lambert-W
        arg = (Rsh * (Rs * I_L + Rs * I_o + V)) / (a * (Rs + Rsh))
        arg = np.clip(arg, -np.inf, 500.0)  # guard overflow in exp
        z = (Rs * I_o * Rsh / (a * (Rs + Rsh))) * np.exp(arg)
        W = np.real(lambertw(z))
        I = (Rsh * (I_L + I_o) - V) / (Rs + Rsh) - (a / Rs) * W
        return I

    def _voc(self, I_L, I_o, R_sh, a):
        """Open-circuit voltage (I=0): solve I_L - I_o(e^{V/a}-1) - V/Rsh = 0."""
        V = a * np.log(np.maximum(I_L / np.maximum(I_o, 1e-30), 1.0) + 1.0)
        for _ in range(60):
            ex = np.exp(np.clip(V / a, -50.0, 50.0))
            f = I_L - I_o * (ex - 1.0) - V / R_sh
            df = -I_o * ex / a - 1.0 / R_sh
            V = V - f / df
        return np.maximum(V, 0.0)

    # ------------------------------------------------------------------
    # Full I-V / P-V curve
    # ------------------------------------------------------------------
    def iv_curve(self, G, T_cell_K, n=200):
        """Return (V, I, P) arrays sweeping 0..Voc."""
        I_L, I_o, R_sh, a = self._calc_params(G, T_cell_K)
        I_L = float(I_L); I_o = float(I_o); R_sh = float(R_sh); a = float(a)
        if I_L <= 0.0:
            V = np.zeros(n); return V, V.copy(), V.copy()
        V_oc = float(self._voc(I_L, I_o, R_sh, a))
        V = np.linspace(0.0, V_oc, n)
        I = self._i_from_v(V, I_L, I_o, R_sh, a)
        I = np.maximum(I, 0.0)
        P = V * I
        return V, I, P

    # ------------------------------------------------------------------
    # Maximum power point (golden-section on the smooth P(V))
    # ------------------------------------------------------------------
    def mpp(self, G, T_cell_K):
        """Maximum-power-point operating point at given G, cell temperature."""
        G = float(G)
        I_L, I_o, R_sh, a = self._calc_params(G, T_cell_K)
        I_L = float(I_L); I_o = float(I_o); R_sh = float(R_sh); a = float(a)

        if G <= 1.0 or I_L <= 0.0:
            return {"v_mp": 0.0, "i_mp": 0.0, "p_mp": 0.0,
                    "v_oc": 0.0, "i_sc": 0.0, "fill_factor": 0.0}

        V_oc = float(self._voc(I_L, I_o, R_sh, a))
        I_sc = float(self._i_from_v(0.0, I_L, I_o, R_sh, a))

        gr = (np.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 0.0, V_oc
        for _ in range(80):
            v1 = hi - gr * (hi - lo)
            v2 = lo + gr * (hi - lo)
            p1 = v1 * float(self._i_from_v(v1, I_L, I_o, R_sh, a))
            p2 = v2 * float(self._i_from_v(v2, I_L, I_o, R_sh, a))
            if p1 < p2:
                lo = v1
            else:
                hi = v2
        V_mp = 0.5 * (lo + hi)
        I_mp = float(self._i_from_v(V_mp, I_L, I_o, R_sh, a))
        P_mp = V_mp * I_mp
        ff = P_mp / (V_oc * I_sc) if (V_oc * I_sc) > 0 else 0.0
        return {"v_mp": V_mp, "i_mp": I_mp, "p_mp": P_mp,
                "v_oc": V_oc, "i_sc": I_sc, "fill_factor": ff}

    def efficiency(self, G, T_cell_K):
        """Module MPP conversion efficiency (-)."""
        if G <= 1.0:
            return 0.0
        return self.mpp(G, T_cell_K)["p_mp"] / (G * self.area)

    # ------------------------------------------------------------------
    # Faiman steady-state cell temperature (used for ODE init / cross-check)
    # ------------------------------------------------------------------
    def steady_cell_temperature(self, G, T_amb_c, wind=1.0):
        """Faiman (2008) steady module temperature [degC].

        Algebraic root of the thermal balance with electrical power extracted
        at MPP. Solved by a short fixed-point iteration.
        """
        U = self.U0 + self.U1 * wind
        T = T_amb_c + G * self.absorptance / max(U, 1e-6)  # no-power guess
        for _ in range(20):
            mpp = self.mpp(G, T + 273.15)
            p_area = mpp["p_mp"] / self.area
            T = T_amb_c + (self.absorptance * G - p_area) / max(U, 1e-6)
        return T

    # ------------------------------------------------------------------
    # Lumped thermal ODE  (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def _dTdt(self, T_cell_K, G, T_amb_c, wind):
        T_amb_K = T_amb_c + 273.15
        U = self.U0 + self.U1 * wind
        p_area = self.mpp(G, T_cell_K)["p_mp"] / self.area  # W/m2 extracted
        q_in = self.absorptance * G
        q_loss = U * (T_cell_K - T_amb_K)
        return (q_in - p_area - q_loss) / self.C_th

    def simulate(self, irradiance, T_amb_c, wind=1.0, T_cell0_c=None,
                 dt=60.0, duration_s=3600.0):
        """
        Transient lumped-thermal simulation of the CIGS module.

        Parameters
        ----------
        irradiance : float or callable(t)->float   plane-of-array G [W/m2]
        T_amb_c    : float or callable(t)->float   ambient temperature [degC]
        wind       : float or callable(t)->float   wind speed [m/s]
        T_cell0_c  : float or None                 initial cell temp [degC]
                                                    (default = T_amb at t=0)
        dt         : float                          output step [s]
        duration_s : float                          total duration [s]

        Returns
        -------
        dict of time-series arrays: t, irradiance, T_amb, T_cell_c, v_mp, i_mp,
            p_mp, v_oc, i_sc, fill_factor, efficiency, power_W
        """
        G_f = irradiance if callable(irradiance) else (lambda t: irradiance)
        Ta_f = T_amb_c if callable(T_amb_c) else (lambda t: T_amb_c)
        w_f = wind if callable(wind) else (lambda t: wind)

        if T_cell0_c is None:
            T_cell0_c = Ta_f(0.0)
        T0_K = T_cell0_c + 273.15

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self._dTdt(y[0], G_f(t), Ta_f(t), w_f(t))]

        sol = solve_ivp(rhs, (0.0, duration_s), [T0_K], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-6, max_step=dt)

        t_out = sol.t
        T_K = sol.y[0]
        N = len(t_out)

        G_arr = np.array([G_f(t) for t in t_out])
        Ta_arr = np.array([Ta_f(t) for t in t_out])
        v_mp = np.zeros(N); i_mp = np.zeros(N); p_mp = np.zeros(N)
        v_oc = np.zeros(N); i_sc = np.zeros(N); ff = np.zeros(N); eff = np.zeros(N)

        for i in range(N):
            r = self.mpp(G_arr[i], T_K[i])
            v_mp[i] = r["v_mp"]; i_mp[i] = r["i_mp"]; p_mp[i] = r["p_mp"]
            v_oc[i] = r["v_oc"]; i_sc[i] = r["i_sc"]; ff[i] = r["fill_factor"]
            eff[i] = r["p_mp"] / (G_arr[i] * self.area) if G_arr[i] > 1.0 else 0.0

        return {
            "t": t_out,
            "irradiance": G_arr,
            "T_amb": Ta_arr,
            "T_cell_c": T_K - 273.15,
            "v_mp": v_mp, "i_mp": i_mp, "p_mp": p_mp,
            "v_oc": v_oc, "i_sc": i_sc,
            "fill_factor": ff,
            "efficiency": eff,
            "power_W": p_mp,
        }
