"""
EC148 -- Bioethanol Fermentation -- F2a Physics-Lumped Kinetic Model

Physics-lumped (0D) model of yeast (Saccharomyces cerevisiae) batch / fed-batch
fermentation of glucose to ethanol in a stirred bioreactor. Coupled species
mass-balance ODEs (substrate S, biomass X, product P = ethanol) with Monod
growth kinetics modulated by ethanol PRODUCT INHIBITION (Luong 1985), plus a
lumped energy balance for the exothermic fermentation heat with jacket cooling.

State vector  y = [S, X, P, T]
    S : glucose concentration            [g/L]
    X : viable biomass concentration     [g/L]
    P : ethanol concentration            [g/L]
    T : broth temperature                [K]

Specific growth rate (Monod x Luong product inhibition x temperature factor):
    mu(S,P,T) = mu_max * S/(Ks + S) * (1 - P/P*)^n * f_T(T)
    (Luong: power-law cessation of growth at the critical ethanol P*; n>1 toxic.)

Substrate uptake combines growth, product formation and maintenance:
    -dS/dt = (1/Yxs) mu X + (1/Yps) qP X + ms X        (Pirt maintenance)
Ethanol formation -- Luedeking-Piret (growth + non-growth associated):
    qP = alpha * mu + beta
    dP/dt = qP * X
Biomass:
    dX/dt = mu X
Energy balance (lumped, well-mixed):
    rho V cp dT/dt = (-dS/dt_growth-consumed) * (dHf/M) * V_L  - UA (T - Tcool)
    heat released is proportional to glucose catabolised to ethanol+CO2.

Conservation / physical bounds enforced:
    * Carbon/mass: ethanol produced <= Yps_theoretical * glucose consumed (0.511 g/g).
    * Substrate never goes negative; growth halts when S -> 0 or P -> P*.
    * Product inhibition: mu -> 0 as P -> P* (Luong), monotone decreasing in P.

References:
    Bai, F.W., Anderson, W.A., Moo-Young, M. (2008). "Ethanol fermentation
        technologies from sugar and starch feedstocks." Biotechnol. Adv.
        26(1):89-105.
    Luong, J.H.T. (1985). "Kinetics of ethanol inhibition in alcohol
        fermentation." Biotechnol. Bioeng. 27(3):280-285.
    Levenspiel, O. (1980). "The Monod equation: a revisit and a generalization
        to product inhibition situations." Biotechnol. Bioeng. 22:1671-1687.
    Monod, J. (1949). "The growth of bacterial cultures." Annu. Rev.
        Microbiol. 3:371-394.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BioethanolFermentationF2a:
    """S. cerevisiae glucose->ethanol fermentation: Monod + Luong inhibition + thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.mu_max = float(u["mu_max"]["value"])             # 1/h
        self.Ks = float(u["Ks"]["value"])                     # g/L
        self.Yxs = float(u["Yxs"]["value"])                   # g/g
        self.Yps = float(u["Yps"]["value"])                   # g/g (observed)
        self.Yps_th = float(u["Yps_theoretical"]["value"])    # g/g (0.511 max)
        self.ms = float(u["ms"]["value"])                     # g/(g.h)
        self.alpha = float(u["alpha_lp"]["value"])            # g/g
        self.beta = float(u["beta_lp"]["value"])              # g/(g.h)
        self.P_star = float(u["P_star"]["value"])             # g/L (product)
        self.P_star_g = float(u["P_star_growth"]["value"])    # g/L (growth)
        self.n_luong = float(u["n_luong"]["value"])           # - (growth)
        self.n_luong_p = float(u["n_luong_p"]["value"])       # - (product)
        # initial / reactor
        self.S0 = float(u["S0"]["value"])
        self.X0 = float(u["X0"]["value"])
        self.P0 = float(u["P0"]["value"])
        self.V = float(u["V"]["value"])                       # m3
        self.rho = float(u["rho_broth"]["value"])             # kg/m3
        self.cp = float(u["cp_broth"]["value"])               # J/(kg.K)
        self.dHf = float(u["dH_ferment"]["value"])            # J/mol glucose
        self.M_glc = float(u["M_glucose"]["value"])           # g/mol
        self.UA = float(u["UA_cool"]["value"])                # W/K
        self.T_cool = float(u["T_coolant_K"]["value"])        # K
        self.T_opt = float(u["T_opt_K"]["value"])             # K
        self.T_width = float(u["T_width_K"]["value"])         # K

    # ------------------------------------------------------------------
    # Temperature activity factor (Gaussian about optimum) in [0,1]
    # ------------------------------------------------------------------
    def temperature_factor(self, T):
        """Dimensionless growth activity vs temperature, peak 1.0 at T_opt."""
        return float(np.exp(-((T - self.T_opt) / self.T_width) ** 2))

    # ------------------------------------------------------------------
    # Specific growth rate: Monod x Luong product inhibition x f_T
    # ------------------------------------------------------------------
    def specific_growth_rate(self, S, P, T):
        """Specific growth rate mu [1/h] (Monod + Luong + temperature)."""
        S = max(S, 0.0)
        monod = S / (self.Ks + S) if (self.Ks + S) > 0 else 0.0
        # Luong product inhibition on GROWTH: (1 - P/P*_growth)^n, clipped at 0.
        # Growth is more ethanol-sensitive than fermentation (P*_growth < P*).
        frac = 1.0 - P / self.P_star_g
        inhib = frac ** self.n_luong if frac > 0.0 else 0.0
        mu = self.mu_max * monod * inhib * self.temperature_factor(T)
        return max(mu, 0.0)

    # ------------------------------------------------------------------
    # Specific ethanol production rate -- Luedeking-Piret
    # ------------------------------------------------------------------
    def specific_product_rate(self, mu, S, P):
        """
        Specific ethanol production rate qP [g_EtOH/(g_cells.h)].

        Luedeking-Piret form: qP = alpha*mu + beta, with BOTH terms gated by
        a smooth Monod substrate availability factor S/(Ks+S) and the Luong
        ethanol-inhibition factor, so production stops continuously as glucose
        is exhausted (S -> 0) or as ethanol approaches P* -- this keeps the
        ODE right-hand side smooth (no hard switches -> non-stiff for LSODA).
        """
        if S <= 0.0:
            return 0.0
        s_avail = S / (self.Ks + S)
        frac = 1.0 - P / self.P_star
        inhib = frac ** self.n_luong_p if frac > 0.0 else 0.0
        qP_nongrowth = self.beta * s_avail * inhib
        return self.alpha * mu + qP_nongrowth

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def rhs(self, t, y):
        """d[S,X,P,T]/dt for the lumped fermenter (smooth, conservative)."""
        S, X, P, T = y
        S = max(S, 0.0)
        X = max(X, 0.0)

        mu = self.specific_growth_rate(S, P, T)
        qP = self.specific_product_rate(mu, S, P)

        dXdt = mu * X

        # --- Ethanol formation (Luedeking-Piret, growth + non-growth) ---
        dPdt = qP * X

        # --- Substrate balance (anaerobic catabolic structure, Bai 2008) ---
        #   Glucose flows to: (i) biomass via Yxs, (ii) ethanol+CO2 catabolism
        #   at the observed product yield Yps, (iii) maintenance (Pirt).
        #   Coupling catabolism through Yps guarantees overall yield -> Yps
        #   (<= 0.511 theoretical). Maintenance is smoothly gated by S so the
        #   field is continuous through substrate exhaustion.
        s_avail = S / (self.Ks + S) if (self.Ks + S) > 0 else 0.0
        rS_growth = (mu / self.Yxs) * X if self.Yxs > 0 else 0.0
        rS_prod = (dPdt / self.Yps) if self.Yps > 0 else 0.0
        rS_maint = self.ms * X * s_avail
        dSdt = -(rS_growth + rS_prod + rS_maint)

        # Thermal balance: heat released ~ glucose catabolised (mol/h * dHf)
        glucose_consumed_g_per_h = -dSdt  # g/L/h
        # convert to total mol/s in the reactor (V in m3 = 1000 L)
        V_L = self.V * 1000.0
        mol_per_s = glucose_consumed_g_per_h * V_L / self.M_glc / 3600.0
        Q_gen = mol_per_s * self.dHf                     # W
        Q_cool = self.UA * (T - self.T_cool)             # W
        m_broth = self.rho * self.V                      # kg
        dTdt = (Q_gen - Q_cool) / (m_broth * self.cp)    # K/s
        dTdt *= 3600.0                                   # K/h  (time base is hours)

        return [dSdt, dXdt, dPdt, dTdt]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, S0=None, X0=None, P0=None, T0=None,
                 dt=0.25, duration_h=48.0):
        """
        Integrate the batch fermentation.

        Parameters
        ----------
        S0, X0, P0 : float
            Initial glucose, biomass, ethanol [g/L] (defaults from params).
        T0 : float
            Initial broth temperature [K] (default = T_opt).
        dt : float
            Output sampling interval [h].
        duration_h : float
            Total fermentation time [h].

        Returns
        -------
        dict of time-series arrays plus scalar performance metrics.
        """
        S0 = self.S0 if S0 is None else float(S0)
        X0 = self.X0 if X0 is None else float(X0)
        P0 = self.P0 if P0 is None else float(P0)
        T0 = self.T_opt if T0 is None else float(T0)

        t_eval = np.arange(0.0, duration_h + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_h]

        sol = solve_ivp(
            self.rhs, (0.0, duration_h), [S0, X0, P0, T0],
            t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9,
            max_step=dt,
        )

        S = np.clip(sol.y[0], 0.0, None)
        X = np.clip(sol.y[1], 0.0, None)
        P = np.clip(sol.y[2], 0.0, None)
        T = sol.y[3]
        t = sol.t
        N = len(t)

        mu = np.array([self.specific_growth_rate(S[i], P[i], T[i]) for i in range(N)])

        # Performance metrics
        glucose_consumed = S0 - S[-1]                         # g/L
        ethanol_yield = P[-1] / glucose_consumed if glucose_consumed > 1e-9 else 0.0  # g/g
        productivity = P[-1] / t[-1] if t[-1] > 0 else 0.0    # g/(L.h)
        # fermentation (sugar->ethanol) efficiency vs theoretical
        ferment_efficiency = ethanol_yield / self.Yps_th if self.Yps_th > 0 else 0.0

        return {
            "t": t,
            "glucose": S,
            "biomass": X,
            "ethanol": P,
            "temperature": T,
            "mu": mu,
            "glucose_consumed_g_L": float(glucose_consumed),
            "ethanol_final_g_L": float(P[-1]),
            "biomass_final_g_L": float(X[-1]),
            "ethanol_yield_g_g": float(ethanol_yield),
            "ferment_efficiency": float(ferment_efficiency),
            "productivity_g_L_h": float(productivity),
            "T_final_K": float(T[-1]),
        }
