"""
EC189 -- Natural Gas Pipeline -- F2a Physics-Lumped Line-Pack Dynamics

0D (single-control-volume) ISOTHERMAL COMPRESSIBLE flow model.

Two coupled pieces of physics:

  (1) STEADY FLOW EQUATION (algebraic, evaluated each step) -- general isothermal
      pipe-flow equation in Menon (2005) SI form (Gas Pipeline Hydraulics, Eq. 2.x):

          Q = 1.1494e-3 * (T_b/P_b) * sqrt( (P1^2 - P2^2) / (G * T_f * L * Z * f) ) * D^2.5

      with Q [std m3/day], T_b [K], P_b [kPa], P [kPa], D [mm], L [km].
      f is the Darcy friction factor from the Colebrook-White equation, which makes
      this the "general" (fully-rough/transitional) flow equation; setting
      f = 0.032/D^(1/3) recovers the classic Weymouth law (consistency check with F1).
      Flow scales as Q ~ sqrt(P1^2 - P2^2): the squared-pressure structure is the
      signature of isothermal compressible flow with friction (Menon 2005, Ch. 2).

  (2) LINE-PACK MASS/PRESSURE ODE (lumped, integrated by scipy.solve_ivp) --
      the pipe is one lumped control volume of geometric volume V_pipe. The stored
      ("line-pack") gas mass is m = rho_avg * V_pipe with isothermal real-gas density

          rho_avg = P_avg * M / (Z * R * T)            (M = G * M_air)

      Conservation of mass on the control volume (Mohitpour 2003, transient line-pack):

          dm/dt = m_in - m_out
       => dP_avg/dt = (Z * R * T) / (M * V_pipe) * (m_in - m_out)

      Inflow m_in is set by the upstream (compressor/supply) boundary; outflow m_out
      is the steady flow equation driven by the instantaneous (P_avg -> P_out) drop.
      Line-pack therefore acts as a pressure-storage buffer: a supply/demand imbalance
      charges or discharges the pipe inventory and moves P_avg, exactly the mechanism
      transmission operators use for short-term gas storage.

Conservation & physical guarantees enforced:
  * Mass conservation:  integral(m_in - m_out) dt = delta(line-pack mass)  (to solver tol)
  * Pressure-drop-with-flow:  Q = 0 iff P1 = P2;  Q monotone increasing in (P1^2-P2^2)
  * Friction: Darcy f > 0 always (Colebrook), so finite drop for any finite flow
  * Line-pack storage:  surplus inflow raises P_avg; deficit lowers it

Pure Python + NumPy + SciPy.  Natural-gas properties hardcoded with citations
(see data/parameters.json).

References:
  Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press. (general/Weymouth flow eq.)
  Mohitpour, M., Golshan, H., Murray, A. (2003). Pipeline Design & Construction.
      ASME Press. (line-pack, transient mass balance)
  Colebrook, C.F. (1939). J. Inst. Civ. Eng., 11(4), 133-156. (friction factor)
"""

import numpy as np
from scipy.integrate import solve_ivp

M_AIR = 0.028966  # kg/mol, molar mass of dry air


class NGPipelineF2a:
    """Isothermal lumped line-pack natural-gas pipeline model."""

    # Menon (2005) SI general-flow constant: Q[std m3/day], P[kPa], D[mm], L[km]
    GENERAL_K = 1.1494e-3

    def __init__(self, params: dict):
        u = params["unit"]
        g = params["gas"]

        # Geometry (user-facing SI: km, m)
        self.L_km = u["length"]["value"]
        self.D_m = u["diameter"]["value"]
        self.eps_m = u["roughness"]["value"]
        self.E = u["efficiency_factor"]["value"]

        # Thermo / base conditions
        self.T_flow = u["T_flow"]["value"]          # K
        self.T_b = u["T_base"]["value"]             # K
        self.P_b_kPa = u["P_base"]["value"]         # kPa

        # Gas properties
        self.G = g["specific_gravity"]["value"]
        self.Z = g["Z_avg"]["value"]
        self.M = g["molar_mass"]["value"]           # kg/mol (= G * M_air)
        self.R = g["R_universal"]["value"]          # J/(mol K)
        self.mu = g["viscosity"]["value"]           # Pa.s

        # Standard-condition density (ideal gas at base, Z_b ~ 1): kg/std-m3
        self.rho_std = (self.P_b_kPa * 1e3 * self.M) / (self.R * self.T_b)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def pipe_volume_m3(self):
        """Geometric internal volume of the lumped control volume [m3]."""
        return np.pi * (self.D_m ** 2) / 4.0 * (self.L_km * 1000.0)

    def gas_density(self, P_Pa, T=None, Z=None):
        """Real-gas density [kg/m3] at absolute pressure P_Pa (isothermal)."""
        T = self.T_flow if T is None else T
        Z = self.Z if Z is None else Z
        return (P_Pa * self.M) / (Z * self.R * T)

    def linepack_mass_kg(self, P_avg_Pa, T=None, Z=None):
        """Stored line-pack mass at mean pressure P_avg_Pa [kg]."""
        return self.gas_density(P_avg_Pa, T, Z) * self.pipe_volume_m3()

    # ------------------------------------------------------------------
    # Friction factor -- Colebrook-White (general flow equation)
    # ------------------------------------------------------------------
    def reynolds(self, Q_std_m3_day, D=None):
        """Reynolds number from standard-volumetric flow (Menon 2005, Eq. 2.34)."""
        D = self.D_m if D is None else D
        if Q_std_m3_day <= 0:
            return 0.0
        m_dot = Q_std_m3_day / 86400.0 * self.rho_std        # kg/s
        # Re = 4 m_dot / (pi D mu)
        return 4.0 * m_dot / (np.pi * D * self.mu)

    def friction_factor(self, Q_std_m3_day, D=None):
        """Darcy friction factor.

        Laminar: f = 64/Re.  Turbulent: Colebrook-White (implicit, solved by
        fixed-point iteration) using absolute roughness eps.  Falls back to the
        Weymouth fully-turbulent value when flow/Re is negligible so the steady
        equation degrades gracefully to F1 behaviour.
        """
        D = self.D_m if D is None else D
        Re = self.reynolds(Q_std_m3_day, D)
        # Weymouth fully-turbulent reference (Menon 2005): f_w = 0.032 / D_mm^(1/3)
        f_wey = 0.032 / (D * 1000.0) ** (1.0 / 3.0)
        if Re < 1.0:
            return f_wey
        if Re < 2100.0:
            return 64.0 / Re
        # Colebrook-White: 1/sqrt(f) = -2 log10( eps/(3.7 D) + 2.51/(Re sqrt(f)) )
        rr = self.eps_m / (3.7 * D)
        inv_sqrt_f = 1.0 / np.sqrt(f_wey)  # initial guess
        for _ in range(40):
            inv_sqrt_f = -2.0 * np.log10(rr + 2.51 * inv_sqrt_f / Re)
        f = 1.0 / inv_sqrt_f ** 2
        return f

    # ------------------------------------------------------------------
    # Steady isothermal flow equation  Q ~ sqrt(P1^2 - P2^2)
    # ------------------------------------------------------------------
    def flow_rate_std_m3_day(self, P1_Pa, P2_Pa, f=None, L_km=None, D=None):
        """General isothermal flow equation (Menon 2005, SI).

        Returns volumetric flow at standard conditions [std m3/day].
        Sign convention: positive = P1 -> P2 (forward).  Handles reverse flow.
        """
        L_km = self.L_km if L_km is None else L_km
        D = self.D_m if D is None else D
        P1_kPa = P1_Pa / 1e3
        P2_kPa = P2_Pa / 1e3
        D_mm = D * 1000.0

        dp2 = P1_kPa ** 2 - P2_kPa ** 2
        sign = 1.0 if dp2 >= 0.0 else -1.0
        adp2 = abs(dp2)
        if adp2 <= 0.0:
            return 0.0

        if f is None:
            # Iterate once on f(Q): start from Weymouth, refine with Colebrook.
            f = 0.032 / D_mm ** (1.0 / 3.0)
            for _ in range(3):
                Q_try = (self.E * self.GENERAL_K * (self.T_b / self.P_b_kPa) *
                         np.sqrt(adp2 / (self.G * self.T_flow * L_km * self.Z * f)) *
                         D_mm ** 2.5)
                f = self.friction_factor(Q_try, D)

        Q = (self.E * self.GENERAL_K * (self.T_b / self.P_b_kPa) *
             np.sqrt(adp2 / (self.G * self.T_flow * L_km * self.Z * f)) *
             D_mm ** 2.5)
        return sign * Q

    def flow_rate_kg_s(self, P1_Pa, P2_Pa, f=None, L_km=None, D=None):
        """Mass flow [kg/s] from the steady flow equation."""
        Q = self.flow_rate_std_m3_day(P1_Pa, P2_Pa, f, L_km, D)
        return Q / 86400.0 * self.rho_std

    # ------------------------------------------------------------------
    # Line-pack pressure ODE
    # ------------------------------------------------------------------
    def dPavg_dt(self, P_avg_Pa, m_in_kg_s, m_out_kg_s):
        """d(P_avg)/dt [Pa/s] from lumped mass balance.

        From m = rho_avg V_pipe and rho_avg = P_avg M /(Z R T):
            dP_avg/dt = (Z R T)/(M V_pipe) * (m_in - m_out)
        """
        V = self.pipe_volume_m3()
        return (self.Z * self.R * self.T_flow) / (self.M * V) * (m_in_kg_s - m_out_kg_s)

    def simulate(self, P_avg0_bar, P_out_bar, m_in_kg_s, dt, duration_s,
                 P_in_bar=None):
        """Transient line-pack simulation (isothermal).

        The pipe is one control volume at mean pressure P_avg(t).
          * Inflow m_in_kg_s (supply boundary) -- float or callable(t).
          * Outflow is the steady flow equation from the *mean* pressure
            P_avg(t) down to the fixed downstream delivery pressure P_out_bar,
            over half the pipe length (mean-pressure node sits at the midpoint).
          * If P_in_bar is given, it instead fixes the upstream pressure and the
            inflow is computed from the flow equation P_in -> P_avg (driven model).

        Returns dict of time-series:
          t, P_avg [bar], linepack_mass [kg], m_in [kg/s], m_out [kg/s],
          Q_out [std m3/day], friction_factor, reynolds.
        """
        _min = m_in_kg_s if callable(m_in_kg_s) else (lambda t: m_in_kg_s)
        P_out_Pa = P_out_bar * 1e5
        half_L = self.L_km / 2.0

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def m_out_of(P_avg_Pa):
            # outflow from mean node to delivery, over downstream half-length
            return self.flow_rate_kg_s(P_avg_Pa, P_out_Pa, f=None, L_km=half_L)

        def m_in_of(t, P_avg_Pa):
            if P_in_bar is not None:
                # driven by fixed upstream pressure over upstream half-length
                return self.flow_rate_kg_s(P_in_bar * 1e5, P_avg_Pa, f=None,
                                           L_km=half_L)
            return _min(t)

        def rhs(t, y):
            P_avg_Pa = y[0]
            m_in = m_in_of(t, P_avg_Pa)
            m_out = m_out_of(P_avg_Pa)
            return [self.dPavg_dt(P_avg_Pa, m_in, m_out)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [P_avg0_bar * 1e5],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1.0,
            max_step=dt,
        )

        t_out = sol.t
        P_avg_Pa = sol.y[0]
        N = len(t_out)

        P_avg_bar = P_avg_Pa / 1e5
        lp_mass = np.array([self.linepack_mass_kg(P) for P in P_avg_Pa])
        m_in_arr = np.array([m_in_of(t_out[i], P_avg_Pa[i]) for i in range(N)])
        m_out_arr = np.array([m_out_of(P_avg_Pa[i]) for i in range(N)])
        Q_out = np.array([self.flow_rate_std_m3_day(P_avg_Pa[i], P_out_Pa,
                          f=None, L_km=half_L) for i in range(N)])
        f_arr = np.array([self.friction_factor(abs(q)) for q in Q_out])
        Re_arr = np.array([self.reynolds(abs(q)) for q in Q_out])

        return {
            "t": t_out,
            "P_avg": P_avg_bar,
            "linepack_mass": lp_mass,
            "m_in": m_in_arr,
            "m_out": m_out_arr,
            "Q_out": Q_out,
            "friction_factor": f_arr,
            "reynolds": Re_arr,
        }
