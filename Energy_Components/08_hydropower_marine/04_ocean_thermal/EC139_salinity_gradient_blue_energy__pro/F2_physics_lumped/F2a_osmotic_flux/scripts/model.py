"""
EC139 -- Salinity Gradient Blue Energy (PRO) -- F2a Osmotic Flux (physics-lumped)

Pressure-Retarded Osmosis (PRO) extracts power from the salinity gradient between
a concentrated DRAW solution (seawater) and a dilute FEED (river water) separated
by a semipermeable membrane. Water permeates from feed -> draw against an applied
hydraulic pressure DeltaP; the pressurized permeate spins a hydro-turbine.

Governing equations (first-principles, lumped over the module):

  1. van't Hoff osmotic pressure of each stream:
        pi = nu * R * T * c_mol          [Pa]   (c_mol in mol/m3)

  2. Water flux through the membrane (Loeb 1976; Lee, Baker & Lonsdale 1981):
        Jw = A * (Delta_pi_eff - DeltaP)       [m/s]
     where Delta_pi_eff is the EFFECTIVE osmotic pressure difference at the active
     layer, reduced by concentration polarization (CP) and reverse salt flux.

  3. Concentration polarization model (Achilli, Cath & Childress 2009;
     Yip & Elimelech 2011) -- the workhorse PRO flux equation with internal
     concentration polarization (ICP) in the porous support and external
     concentration polarization (ECP) on the draw side:

        Jw = A * [ pi_D * exp(-Jw/k)  -  pi_F * exp(Jw*S/D) ] / ... (implicit)
                 - B*[exp(Jw*S/D) - exp(-Jw/k)] coupling   - DeltaP

     We solve this implicitly for Jw at each operating point (fixed point /
     Brent root-find), which is the lumped (0-D across membrane) closure.

  4. Reverse salt flux (Js, salt leaking draw->feed), from solution-diffusion
     coupled to Jw (Yip & Elimelech 2011):
        Js = B * (c_D,m - c_F,m)              [mol/(m2.s)]

  5. Power density (the central PRO result, Loeb 1976):
        W = Jw * DeltaP                       [W/m2]
     For an ideal membrane (no CP, no salt flux) W is maximized at DeltaP = Delta_pi/2,
     giving W_max = A*Delta_pi^2/4. CP and Js lower and left-shift the real peak.

  6. Lumped module concentration / volume ODE (scipy.solve_ivp) -- as water and
     salt cross the membrane the draw stream is diluted along the module:
        V dC_draw/dt = Q*(C_in - C_draw) - Jw*A_mem*C_draw + Js*A_mem    (salt balance)
     integrated over the module residence time to get an outlet-averaged state.

  7. Net power = turbine power on permeate  -  pumping power (feed + draw),
     with pressure-exchanger energy recovery (Statkraft Tofte pilot 2009).

References:
  Loeb, S. (1976). J. Membr. Sci., 1, 49-63.  (PRO concept; W peaks at DeltaP=Dpi/2)
  Lee, K.L., Baker, R.W., Lonsdale, H.K. (1981). J. Membr. Sci., 8, 141-171. (ICP / S)
  Achilli, A., Cath, T.Y., Childress, A.E. (2009). J. Membr. Sci., 343, 42-52.
  Yip, N.Y., Elimelech, M. (2011). Environ. Sci. Technol., 45, 10273-10282.
  Straub, A.P., Deshmukh, A., Elimelech, M. (2016). Energy Environ. Sci., 9, 31-48.
  Statkraft (2009). World's first osmotic power plant, Tofte, Norway.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


class SalinityGradientPRO_F2a:
    """PRO salinity-gradient power -- lumped osmotic-flux model with CP, reverse
    salt flux, and a module dilution ODE."""

    R = 8.314          # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_perm"]["value"]          # m/(s.Pa)  water permeability
        self.B = u["B_salt"]["value"]          # m/s       salt permeability
        self.S = u["S_struct"]["value"]        # m         structural parameter
        self.k = u["k_ecp"]["value"]           # m/s       external MT coefficient
        self.D = u["D_salt"]["value"]          # m2/s      salt diffusivity
        self.A_mem = u["A_mem"]["value"]       # m2
        self.C_draw0 = u["C_draw_g_per_L"]["value"]
        self.C_feed0 = u["C_feed_g_per_L"]["value"]
        self.T_K = u["T_K"]["value"]
        self.M = u["M_NaCl"]["value"]          # g/mol
        self.nu = u["nu_NaCl"]["value"]
        self.V_draw = u["V_draw"]["value"]     # m3
        self.Q_draw = u["Q_draw"]["value"]     # m3/s
        self.eta_turb = u["eta_turbine"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_px = u["eta_px"]["value"]
        self.dP_pump_feed = u["dP_pump_feed_Pa"]["value"]

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------
    def conc_mol_m3(self, C_gL):
        """Convert g/L NaCl to mol/m3."""
        return np.asarray(C_gL, float) / self.M * 1000.0

    def osmotic_pressure(self, C_gL, T_K=None):
        """van't Hoff osmotic pressure [Pa] for a NaCl stream at C_gL."""
        if T_K is None:
            T_K = self.T_K
        return self.nu * self.R * T_K * self.conc_mol_m3(C_gL)

    # ------------------------------------------------------------------
    # Flux closure with concentration polarization + reverse salt flux
    # ------------------------------------------------------------------
    def water_flux(self, delta_P_Pa, C_draw_gL=None, C_feed_gL=None, T_K=None):
        """
        Solve the implicit PRO water-flux equation for Jw [m/s].

        Active-layer effective osmotic pressures (Yip & Elimelech 2011):
            pi_D,eff = pi_D * exp(-Jw/k)        (draw-side ECP -> dilutive)
            pi_F,eff = pi_F * exp(+Jw*S/D)      (feed-side ICP -> concentrative)
        Reverse-salt-flux correction enters through the B-term:

            Jw = A * [ pi_D*exp(-Jw/k) - pi_F*exp(Jw*S/D) ]
                 / ( 1 + (B/Jw)*(exp(Jw*S/D) - exp(-Jw/k)) )      (implicit form)
                 - A*delta_P

        We rearrange to residual f(Jw)=0 and root-find with Brent.
        """
        if C_draw_gL is None: C_draw_gL = self.C_draw0
        if C_feed_gL is None: C_feed_gL = self.C_feed0
        if T_K is None: T_K = self.T_K

        pi_D = float(self.osmotic_pressure(C_draw_gL, T_K))
        pi_F = float(self.osmotic_pressure(C_feed_gL, T_K))
        dP = float(delta_P_Pa)
        A, B, S, k, D = self.A, self.B, self.S, self.k, self.D

        def residual(Jw):
            if Jw <= 1e-12:
                # limit: no CP, classic Jw = A*(dpi - dP)
                return A * ((pi_D - pi_F) - dP) - Jw
            ecp = np.exp(-Jw / k)            # <1, dilutes draw
            icp = np.exp(Jw * S / D)         # >1, concentrates feed
            num = pi_D * ecp - pi_F * icp
            den = 1.0 + (B / Jw) * (icp - ecp)
            dpi_eff = num / den
            return A * (dpi_eff - dP) - Jw

        # Bracket: Jw between ~0 and the ideal flux A*(pi_D - pi_F)
        Jw_hi = max(A * (pi_D - pi_F), 1e-9)
        f_lo = residual(1e-12)
        f_hi = residual(Jw_hi)
        if f_lo * f_hi > 0:
            # no sign change -> flux is non-positive (DeltaP >= effective dpi); return <=0
            # estimate residual flux directly (can be negative => reverse osmosis regime)
            Jw_guess = A * ((pi_D - pi_F) - dP)
            return Jw_guess
        Jw = brentq(residual, 1e-12, Jw_hi, xtol=1e-12, rtol=1e-10, maxiter=200)
        return Jw

    def reverse_salt_flux(self, Jw, C_draw_gL=None, C_feed_gL=None, T_K=None):
        """
        Reverse salt flux Js [mol/(m2.s)] (draw -> feed leakage), evaluated at the
        membrane active layer with the same CP profiles (Yip & Elimelech 2011):

            Js = B * (c_D,m - c_F,m)
        with c_D,m = c_D*exp(-Jw/k), c_F,m = c_F*exp(Jw*S/D).
        """
        if C_draw_gL is None: C_draw_gL = self.C_draw0
        if C_feed_gL is None: C_feed_gL = self.C_feed0
        cD = float(self.conc_mol_m3(C_draw_gL))
        cF = float(self.conc_mol_m3(C_feed_gL))
        if Jw <= 1e-12:
            return self.B * (cD - cF)
        cD_m = cD * np.exp(-Jw / self.k)
        cF_m = cF * np.exp(Jw * self.S / self.D)
        return self.B * (cD_m - cF_m)

    def power_density(self, delta_P_Pa, C_draw_gL=None, C_feed_gL=None, T_K=None):
        """PRO power density W = Jw * DeltaP  [W/m2]."""
        Jw = self.water_flux(delta_P_Pa, C_draw_gL, C_feed_gL, T_K)
        return Jw * float(delta_P_Pa)

    def optimal_delta_P(self, C_draw_gL=None, C_feed_gL=None, T_K=None,
                        n_scan=200):
        """
        Find DeltaP that maximizes power density by scanning 0..Delta_pi.
        Returns (dP_opt_Pa, W_max_Wm2). For an ideal membrane this lands at
        DeltaP = Delta_pi/2 (Loeb 1976).
        """
        if C_draw_gL is None: C_draw_gL = self.C_draw0
        if C_feed_gL is None: C_feed_gL = self.C_feed0
        if T_K is None: T_K = self.T_K
        dpi = float(self.osmotic_pressure(C_draw_gL, T_K) -
                    self.osmotic_pressure(C_feed_gL, T_K))
        dP_grid = np.linspace(0.0, dpi, n_scan)
        W = np.array([self.power_density(dP, C_draw_gL, C_feed_gL, T_K)
                      for dP in dP_grid])
        i = int(np.argmax(W))
        return dP_grid[i], W[i]

    # ------------------------------------------------------------------
    # Lumped module concentration / volume ODE
    # ------------------------------------------------------------------
    def simulate(self, delta_P_Pa, C_draw_gL=None, C_feed_gL=None, T_K=None,
                 dt=1.0, duration_s=600.0):
        """
        Integrate the lumped draw-side salt/volume balance over the module with
        scipy.solve_ivp. As permeate water enters the draw side it dilutes it,
        while reverse salt flux adds salt back. The draw stream is continuously
        replenished at flow Q_draw with inlet concentration C_draw0.

        State: x = [V (m3), m_salt (mol)] in the draw control volume.

          dV/dt     = Q_in - Q_out + Jw*A_mem
          dm_salt/dt= Q_in*c_in - Q_out*c_out + Js*A_mem - Jw*A_mem*c_out_perm
        We hold V ~ constant by setting Q_out = Q_in + Jw*A_mem (overflow), so the
        permeate is harvested. c_draw = m_salt/V evolves to a steady outlet state.

        Returns dict of time series + final power metrics.
        """
        if C_draw_gL is None: C_draw_gL = self.C_draw0
        if C_feed_gL is None: C_feed_gL = self.C_feed0
        if T_K is None: T_K = self.T_K

        V0 = self.V_draw
        cD_in = float(self.conc_mol_m3(C_draw_gL))   # mol/m3 inlet
        m0 = cD_in * V0                              # mol salt initially
        Q_in = self.Q_draw

        def rhs(t, x):
            V, m = x
            V = max(V, 1e-6)
            c_draw = m / V                            # mol/m3 current draw conc
            C_draw_now = c_draw * self.M / 1000.0     # back to g/L
            Jw = self.water_flux(delta_P_Pa, C_draw_now, C_feed_gL, T_K)
            Js = self.reverse_salt_flux(Jw, C_draw_now, C_feed_gL, T_K)
            perm_vol = Jw * self.A_mem                # m3/s water entering draw
            Q_out = Q_in + perm_vol                   # overflow keeps V ~ const
            dV = Q_in - Q_out + perm_vol              # = 0 by construction
            dm = Q_in * cD_in - Q_out * c_draw + Js * self.A_mem
            return [dV, dm]

        t_eval = np.arange(0.0, duration_s + dt, dt)
        sol = solve_ivp(rhs, (0.0, duration_s), [V0, m0],
                        t_eval=t_eval, method="LSODA",
                        rtol=1e-7, atol=1e-9, max_step=dt)

        V = sol.y[0]
        m = sol.y[1]
        c_draw = m / np.maximum(V, 1e-6)             # mol/m3
        C_draw_gL_t = c_draw * self.M / 1000.0       # g/L

        # Recompute flux/power/salt-flux time series at each state
        Jw_t = np.array([self.water_flux(delta_P_Pa, c, C_feed_gL, T_K)
                         for c in C_draw_gL_t])
        Js_t = np.array([self.reverse_salt_flux(jw, c, C_feed_gL, T_K)
                         for jw, c in zip(Jw_t, C_draw_gL_t)])
        W_t = Jw_t * float(delta_P_Pa)               # W/m2

        # Power balance (final / averaged steady state)
        P_perm = W_t * self.A_mem                    # W hydraulic on permeate
        P_turb = P_perm * self.eta_turb              # W electrical out
        # Pumping: feed must be pressurized to ~DeltaP via PX (recovered),
        # plus low-grade circulation/filtration head.
        Q_feed = Jw_t * self.A_mem                   # feed throughput ~ permeate
        P_pump_draw = (delta_P_Pa * (Q_feed)) * (1.0 - self.eta_px) / self.eta_pump
        P_pump_feed = (self.dP_pump_feed * Q_feed) / self.eta_pump
        P_pump = P_pump_draw + P_pump_feed
        P_net = P_turb - P_pump                      # W

        return {
            "t": sol.t,
            "C_draw_gL": C_draw_gL_t,
            "Jw": Jw_t,                              # m/s
            "Jw_LMH": Jw_t * 1000.0 * 3600.0,        # L/(m2.h)
            "Js": Js_t,                              # mol/(m2.s)
            "power_density": W_t,                    # W/m2
            "P_turbine_W": P_turb,
            "P_pump_W": P_pump,
            "P_net_W": P_net,
            "delta_P_Pa": float(delta_P_Pa),
        }
