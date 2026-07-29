"""
EC145 -- Pyrolysis Reactor -- F2a Lumped Arrhenius Kinetics + Reactor Energy Balance

Physics-lumped (0D) model of biomass fast/slow pyrolysis: thermal decomposition of
biomass in the absence of oxygen producing bio-oil (tar/condensable vapours), biochar
and non-condensable gas (syngas). The kinetic scheme is the lumped multi-component
Broido--Shafizadeh / competing-reactions network in which virgin biomass decomposes
through three parallel first-order Arrhenius pathways to char, tar (bio-oil) and gas,
and the primary tar can further crack to secondary gas:

    Kinetic scheme (Broido-Shafizadeh / Di Blasi competing reactions):

        Biomass (B) --k1--> Gas   (G)
        Biomass (B) --k2--> Tar   (T, bio-oil)
        Biomass (B) --k3--> Char  (C)
        Tar     (T) --k4--> Gas   (G)            (secondary cracking)

        k_i(T) = A_i * exp(-E_i / (R*T))         Arrhenius

    Species mass ODEs (per unit feed mass, fractions y in [0,1]):

        dy_B/dt = -(k1 + k2 + k3) * y_B
        dy_G/dt =   k1*y_B + k4*y_T
        dy_T/dt =   k2*y_B - k4*y_T
        dy_C/dt =   k3*y_B
        sum(y_B + y_G + y_T + y_C) = 1   (mass conservation, enforced by construction)

    Reactor energy balance (lumped, endothermic decomposition + sensible heating):

        m*cp * dT/dt = Q_ext - dH_rxn * (d(mass decomposed)/dt) - Q_loss
        Q_ext  = external heating duty [W]  (constant-power or ramp)
        Q_loss = hA*(T - T_amb)
        dH_rxn = endothermic heat of pyrolysis [J/kg of biomass decomposed]

The yield--temperature dependence emerges from the relative Arrhenius rates: the tar
(bio-oil) pathway (k2) has the highest pre-exponential and dominates around 700-800 K
(~500 degC) which is exactly where fast pyrolysis maximises bio-oil; secondary cracking
(k4) becomes significant at higher temperature, reducing bio-oil in favour of gas, while
slow heating / lower temperature favours char.

Kinetic and thermodynamic parameters are taken from the Di Blasi (2008) review of
biomass pyrolysis kinetics (the classic three-reaction Chan/Thurner-Mann/Di Blasi
competing scheme) and Bridgwater (2012).

References:
    Di Blasi, C. (2008). "Modeling chemical and physical processes of wood and biomass
        pyrolysis." Prog. Energy Combust. Sci. 34(1):47-90.
    Bridgwater, A.V. (2012). "Review of fast pyrolysis of biomass and product upgrading."
        Biomass Bioenergy 38:68-94.
    Shafizadeh, F. & Chin, P.P.S. (1977). Broido-Shafizadeh cellulose pyrolysis scheme.
    Thurner, F. & Mann, U. (1981). Ind. Eng. Chem. Process Des. Dev. 20:482 (three-rxn rates).
"""

import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314  # J/(mol.K) universal gas constant


class PyrolysisReactorF2a:
    """Lumped Arrhenius pyrolysis kinetics coupled to a reactor energy balance."""

    def __init__(self, params: dict):
        u = params["unit"]
        k = params["kinetics"]

        # --- Competing primary reactions B -> Gas / Tar / Char ---
        self.A1 = float(k["A_gas"]["value"]);   self.E1 = float(k["E_gas"]["value"])
        self.A2 = float(k["A_tar"]["value"]);   self.E2 = float(k["E_tar"]["value"])
        self.A3 = float(k["A_char"]["value"]);  self.E3 = float(k["E_char"]["value"])
        # --- Secondary tar cracking T -> Gas ---
        self.A4 = float(k["A_crack"]["value"]); self.E4 = float(k["E_crack"]["value"])

        # --- Vapour residence time (controls secondary cracking extent) ---
        self.tau_vap = float(u["vapour_residence_s"]["value"]) # s, hot-vapour residence

        # --- Reactor energy balance ---
        self.dH_rxn = float(u["dH_pyrolysis_J_kg"]["value"])   # J/kg endothermic (>0)
        self.cp_solid = float(u["cp_biomass_J_kgK"]["value"])  # J/(kg.K)
        self.m_feed = float(u["feed_mass_kg"]["value"])        # kg charge basis
        self.hA_loss = float(u["hA_loss_W_K"]["value"])        # W/K reactor loss
        self.T_amb = float(u["T_ambient_K"]["value"])          # K

        # --- Product heating values (for energy reporting) ---
        self.LHV_oil = float(u["LHV_bio_oil_MJ_kg"]["value"])
        self.LHV_char = float(u["LHV_char_MJ_kg"]["value"])
        self.LHV_gas = float(u["LHV_gas_MJ_kg"]["value"])
        self.LHV_feed = float(u["LHV_feed_MJ_kg"]["value"])

    # ------------------------------------------------------------------
    # Arrhenius rate constants
    # ------------------------------------------------------------------
    def rate_constants(self, T):
        """Return (k1_gas, k2_tar, k3_char, k4_crack) [1/s] at temperature T [K]."""
        k1 = self.A1 * np.exp(-self.E1 / (R_GAS * T))
        k2 = self.A2 * np.exp(-self.E2 / (R_GAS * T))
        k3 = self.A3 * np.exp(-self.E3 / (R_GAS * T))
        k4 = self.A4 * np.exp(-self.E4 / (R_GAS * T))
        return k1, k2, k3, k4

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side.
    # State = [y_B, y_C, y_Tvap, y_oil, y_gas, T] where:
    #   y_B     = unreacted biomass (solid)
    #   y_C     = char (solid, retained)
    #   y_Tvap  = tar present in the hot vapour phase (subject to cracking)
    #   y_oil   = bio-oil collected after the hot zone (quenched, no cracking)
    #   y_gas   = non-condensable gas collected
    # Tar vapour is removed from the hot zone by (a) secondary cracking -> gas
    # (rate k4) and (b) outflow/quench -> collected bio-oil (rate 1/tau_vap).
    # The shorter the vapour residence (fast pyrolysis), the more tar is
    # quenched to bio-oil before it cracks -> bio-oil maximised, as observed.
    # Mass is conserved: y_B + y_C + y_Tvap + y_oil + y_gas = 1 for all t.
    # ------------------------------------------------------------------
    def _rhs(self, t, state, Q_ext_func):
        y_B, y_C, y_Tvap, y_oil, y_gas, T = state
        y_B = max(y_B, 0.0)
        y_Tvap = max(y_Tvap, 0.0)
        k1, k2, k3, k4 = self.rate_constants(T)
        k_out = 1.0 / self.tau_vap                       # 1/s vapour outflow

        # Primary decomposition of biomass
        decomp_B = (k1 + k2 + k3) * y_B
        crack = k4 * y_Tvap                              # tar -> gas (secondary)
        quench = k_out * y_Tvap                          # tar -> collected bio-oil

        dyB = -decomp_B
        dyC = k3 * y_B
        dyTvap = k2 * y_B - crack - quench
        dyOil = quench
        dyGas = k1 * y_B + crack

        # Energy balance.  Rate of biomass mass decomposed -> endothermic sink.
        m_decomp_rate = self.m_feed * decomp_B          # kg/s of biomass reacting
        Q_rxn = self.dH_rxn * m_decomp_rate             # W absorbed (endothermic)
        Q_ext = Q_ext_func(t)                           # W supplied externally
        Q_loss = self.hA_loss * (T - self.T_amb)        # W lost to surroundings
        m_cp = self.m_feed * self.cp_solid              # J/K lumped thermal mass
        dT = (Q_ext - Q_rxn - Q_loss) / m_cp

        return [dyB, dyC, dyTvap, dyOil, dyGas, dT]

    # ------------------------------------------------------------------
    # Time-domain simulation via solve_ivp
    # ------------------------------------------------------------------
    def simulate(self, Q_ext_W, T0_K, dt=0.5, duration_s=120.0):
        """
        Integrate the coupled kinetics + energy balance.

        Parameters
        ----------
        Q_ext_W : float or callable(t)
            External heating duty [W].
        T0_K : float
            Initial reactor (biomass) temperature [K].
        dt : float
            Output sampling step [s].
        duration_s : float
            Total simulated time [s].

        Returns
        -------
        dict of time-series arrays plus final-state scalars.
        """
        Qf = Q_ext_W if callable(Q_ext_W) else (lambda t: Q_ext_W)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        # initial state: all biomass, nothing converted
        # [y_B, y_C, y_Tvap, y_oil, y_gas, T]
        s0 = [1.0, 0.0, 0.0, 0.0, 0.0, T0_K]

        sol = solve_ivp(
            lambda t, s: self._rhs(t, s, Qf),
            (0.0, duration_s), s0,
            t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        y_B = np.clip(sol.y[0], 0.0, 1.0)
        y_C = np.clip(sol.y[1], 0.0, 1.0)
        y_Tvap = np.clip(sol.y[2], 0.0, 1.0)
        y_oil = np.clip(sol.y[3], 0.0, 1.0)
        y_G = np.clip(sol.y[4], 0.0, 1.0)
        T = sol.y[5]
        conversion = 1.0 - y_B

        # Total condensable bio-oil = collected oil + tar still in vapour phase
        # (any remaining tar vapour quenches on collection, so it counts as oil).
        y_oil_total = y_oil + y_Tvap

        # Mass-conservation residual (should be ~0 at all times)
        total = y_B + y_C + y_Tvap + y_oil + y_G
        mass_residual = np.abs(total - 1.0)

        # Energy content of products [MJ per kg feed]
        e_oil = y_oil_total * self.LHV_oil
        e_char = y_C * self.LHV_char
        e_gas = y_G * self.LHV_gas

        return {
            "t": sol.t,
            "temperature": T,
            "y_biomass": y_B,
            "y_gas": y_G,
            "y_bio_oil": y_oil_total,
            "y_char": y_C,
            "y_tar_vapour": y_Tvap,
            "conversion": conversion,
            "mass_residual": mass_residual,
            "energy_bio_oil_MJ_kg": e_oil,
            "energy_char_MJ_kg": e_char,
            "energy_gas_MJ_kg": e_gas,
            # final-state convenience scalars
            "bio_oil_yield": float(y_oil_total[-1]),
            "char_yield": float(y_C[-1]),
            "gas_yield": float(y_G[-1]),
            "final_conversion": float(conversion[-1]),
            "final_temperature": float(T[-1]),
        }

    # ------------------------------------------------------------------
    # Isothermal equilibrium yields at fixed T (long-time limit)
    # ------------------------------------------------------------------
    def equilibrium_yields(self, T_K, hold_s=600.0):
        """
        Isothermal product split at constant temperature T_K, integrated until
        the solid biomass is fully decomposed.  Tar vapour is withdrawn with the
        configured vapour residence time (secondary cracking competes with
        quench/outflow), so the bio-oil yield exhibits the characteristic peak
        near fast-pyrolysis conditions (~500 degC) and falls off at higher T as
        cracking to gas dominates -- exactly the Di Blasi (2008) / Bridgwater
        (2012) yield-temperature behaviour.

        State = [yB, yC, yTvap, yOil, yGas].
        Returns dict: bio_oil_yield, char_yield, gas_yield, residual_biomass.
        """
        k1, k2, k3, k4 = self.rate_constants(T_K)
        k_out = 1.0 / self.tau_vap

        # Integrate long enough for the solid biomass to fully decompose at this
        # temperature (so we compare ultimate product splits, not kinetic
        # snapshots).  Primary decomposition time-scale ~ 1/(k1+k2+k3).
        k_prim = k1 + k2 + k3
        hold = max(hold_s, 25.0 / k_prim) if k_prim > 0 else hold_s

        def rhs(t, s):
            yB, yC, yTvap, yOil, yGas = s
            yB = max(yB, 0.0); yTvap = max(yTvap, 0.0)
            crack = k4 * yTvap
            quench = k_out * yTvap
            return [
                -(k1 + k2 + k3) * yB,        # biomass
                k3 * yB,                     # char
                k2 * yB - crack - quench,    # tar vapour
                quench,                      # collected bio-oil
                k1 * yB + crack,             # gas
            ]

        sol = solve_ivp(rhs, (0.0, hold), [1.0, 0.0, 0.0, 0.0, 0.0],
                        method="LSODA", rtol=1e-9, atol=1e-12)
        yB, yC, yTvap, yOil, yGas = sol.y[:, -1]
        # any tar vapour remaining at the end quenches to oil on collection
        oil = yOil + yTvap
        return {
            "bio_oil_yield": float(np.clip(oil, 0.0, 1.0)),
            "char_yield": float(np.clip(yC, 0.0, 1.0)),
            "gas_yield": float(np.clip(yGas, 0.0, 1.0)),
            "residual_biomass": float(np.clip(yB, 0.0, 1.0)),
        }
