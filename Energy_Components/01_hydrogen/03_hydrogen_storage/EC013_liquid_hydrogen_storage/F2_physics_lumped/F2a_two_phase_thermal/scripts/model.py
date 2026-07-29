"""
EC013 -- Liquid Hydrogen (LH2) Storage -- F2a Two-Phase Lumped Cryogenic Tank

First-principles 0D lumped model of a vacuum-jacketed, MLI-insulated LH2 dewar.
Heat ingress through the multi-layer insulation drives boil-off; the tank is
treated as two control volumes (saturated liquid + saturated vapor ullage) that
share a common saturation interface.  The coupled mass + energy balances yield
self-pressurization, boil-off rate (BOR, %/day), pressure rise and venting.

State vector  y = [m_tot, U_tot]   (homogeneous-equilibrium tank model)
    m_tot total fluid mass       [kg]
    U_tot total internal energy  [J]   (liquid + vapor + wall sensible)

Governing balances (homogeneous two-phase, well-mixed equilibrium):

  Mass:    dm_tot/dt = -m_dot_vent
  Energy:  dU_tot/dt =  Q_in - m_dot_vent * h_V    (Q_in = heat leak through MLI)

  Closure -- the FIXED tank volume is what couples liquid and vapor.  At every
  instant the contents are saturated, so given (m_tot, U_tot) the equilibrium
  temperature T (hence P, rho_L, rho_V, latent heat) and the vapor quality x
  are obtained by simultaneously enforcing:
        V_L + V_V = V_tank          (volume constraint, fixed)
        m_L + m_V = m_tot           (mass)
        m_L u_L(T) + m_V u_V(T) + C_wall(T) = U_tot   (energy)
        P = P_sat(T)                (saturation line, Antoine)
  This 2-equation root-find (in T and x) recovers the full state each step.
  The split between liquid and vapor masses is therefore an OUTPUT of the
  volume constraint -- never imposed -- so V_L + V_V == V_tank holds exactly.

Self-pressurization driver (Van Dresar & Stochl 1993, NASA TM-106033):
    When sealed (m_dot_vent = 0), Q_in raises U_tot at fixed m_tot and fixed V;
    the saturated mixture must climb the saturation line, so T and P rise
    monotonically while liquid evaporates just enough to keep V_L+V_V=V_tank.
    When P >= P_vent the relief valve opens and saturated vapor is expelled
    (m_dot_vent > 0) to hold P at the set-point.  In open-vent mode the valve
    is always open and the tank sits at the NBP boil-off condition.

MLI effective conductance (Johnson 2010, NREL/TP-560-47503):
    k_eff(T_mean) = k0 * (T_mean / T0_ref)^n_mli ,  T_mean = (T_amb + T_sat)/2
    U_eff = k_eff / d_mli ,   Q_in = U_eff * A_surf * (T_amb - T_sat)

References
----------
Van Dresar, N.T. & Stochl, M.J. (1993) "Coefficients for the boiloff of liquid
    hydrogen in a partially-filled tank," NASA TM-106033.
Barron, R.F. (1985) Cryogenic Systems, 2nd ed., Oxford Univ. Press
    (saturation line, Clausius-Clapeyron, Antoine tabulation).
Sherif, S.A. et al. (1997) Int. J. Hydrogen Energy 22(7):683.
Johnson, W.L. (2010) NREL/TP-560-47503 -- MLI apparent conductivity k(T).
Leachman, J.W. et al. (2009) J. Phys. Chem. Ref. Data 38(3):721 -- H2 EOS;
    saturated-property anchor values taken from the NIST Chemistry WebBook.
"""

import numpy as np
from scipy.integrate import solve_ivp


class LH2TwoPhaseTank:
    """Lumped two-phase (liquid + vapor) cryogenic LH2 tank with boil-off ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        h = params["hydrogen"]
        ant = params["antoine"]
        amb = params["ambient"]

        # --- geometry / insulation ---
        self.V_tank   = u["V_tank"]["value"]               # m3
        self.A_surf   = u["A_surf"]["value"]               # m2
        self.m_tank   = u["m_tank"]["value"]               # kg
        self.cp_wall  = u["cp_wall"]["value"] * 1e3        # J/(kg.K)
        self.U_ref    = u["U_ref"]["value"]                # W/(m2.K) at T_amb=298
        self.d_mli    = u["MLI_thickness"]["value"]        # m
        self.n_mli    = u["n_mli"]["value"]
        self.T0_mli   = u["T0_mli_ref"]["value"]           # K
        self.P_vent   = u["P_vent"]["value"]               # bar
        self.P_max    = u["P_max"]["value"]                # bar
        self.fill_max = u["fill_fraction_max"]["value"]

        # --- hydrogen saturated properties (anchored at NBP) ---
        self.T_nbp    = h["T_sat_1atm"]["value"]           # K
        self.T_crit   = h["T_crit"]["value"]               # K
        self.rho_L0   = h["rho_liquid"]["value"]           # kg/m3 at NBP
        self.rho_V0   = h["rho_vapor"]["value"]            # kg/m3 at NBP
        self.h_vap0   = h["h_vap"]["value"] * 1e3          # J/kg at NBP
        self.cp_L     = h["cp_liquid"]["value"] * 1e3      # J/(kg.K)
        self.cv_V     = h["cv_vapor"]["value"] * 1e3       # J/(kg.K)
        self.cp_V     = h["cp_vapor"]["value"] * 1e3       # J/(kg.K)
        self.LHV      = h["LHV"]["value"]                  # MJ/kg
        self.R_H2     = h["R_H2"]["value"]                 # J/(kg.K)
        self.dhvap_dT = h["dhvap_dT"]["value"] * 1e3       # J/(kg.K)

        # --- Antoine: log10(P_bar) = A - B/(T + C) ---
        self.A_ant = ant["A"]["value"]
        self.B_ant = ant["B"]["value"]
        self.C_ant = ant["C"]["value"]

        self.T_amb_default = amb["T_ambient_default"]["value"]

        # reference enthalpy datum: define internal energy of saturated liquid
        # u_L(T) = cp_L * (T - T_ref); vapor u_V = u_L + (h_vap - P*(1/rhoV - 1/rhoL))
        self._T_ref_u = self.T_nbp  # datum at NBP

    # ------------------------------------------------------------------
    # Saturation line  (Antoine vapor pressure + inverse)
    # ------------------------------------------------------------------
    def p_sat(self, T):
        """Saturation pressure [bar] from Antoine fit (valid ~14-32 K)."""
        T = np.asarray(T, dtype=float)
        return 10.0 ** (self.A_ant - self.B_ant / (T + self.C_ant))

    def t_sat(self, P_bar):
        """Saturation temperature [K] -- inverse Antoine."""
        P_bar = np.asarray(P_bar, dtype=float)
        return self.B_ant / (self.A_ant - np.log10(P_bar)) - self.C_ant

    # ------------------------------------------------------------------
    # Temperature-dependent saturated properties (linearized along dome)
    # ------------------------------------------------------------------
    def h_vap(self, T):
        """Latent heat [J/kg]; declines toward critical point. NIST slope."""
        hv = self.h_vap0 + self.dhvap_dT * (np.asarray(T, dtype=float) - self.T_nbp)
        return np.maximum(hv, 1.0e3)

    def rho_liquid(self, T):
        """Saturated-liquid density [kg/m3]. Linear contraction with T (NIST)."""
        # ~ -1.9 kg/m3 per K near NBP (70.85 @20.28K -> ~64 @24K)
        return self.rho_L0 - 1.9 * (np.asarray(T, dtype=float) - self.T_nbp)

    def rho_vapor_ideal(self, T, P_bar):
        """Saturated-vapor density via ideal gas [kg/m3]."""
        P = np.asarray(P_bar, dtype=float) * 1e5
        return P / (self.R_H2 * np.asarray(T, dtype=float))

    # ------------------------------------------------------------------
    # MLI conductance and heat leak
    # ------------------------------------------------------------------
    def u_eff(self, T_amb, T_sat):
        """Effective overall U [W/(m2.K)] via Johnson (2010) k(T) power law."""
        T_amb = np.asarray(T_amb, dtype=float)
        T_sat = np.asarray(T_sat, dtype=float)
        k0 = self.U_ref * self.d_mli            # k_eff at reference (T_amb=298)
        T_mean = 0.5 * (T_amb + T_sat)
        T_mean_ref = 0.5 * (self.T_amb_default + self.T_nbp)
        k_eff = k0 * (T_mean / T_mean_ref) ** self.n_mli
        return k_eff / self.d_mli

    def heat_leak(self, T_amb, T_sat):
        """Heat ingress through MLI [W]."""
        U = self.u_eff(T_amb, T_sat)
        return U * self.A_surf * (np.asarray(T_amb, dtype=float) -
                                  np.asarray(T_sat, dtype=float))

    # ------------------------------------------------------------------
    # Internal energy bookkeeping
    # ------------------------------------------------------------------
    def u_liquid(self, T):
        """Specific internal energy of saturated liquid [J/kg], datum at NBP."""
        return self.cp_L * (np.asarray(T, dtype=float) - self._T_ref_u)

    def u_vapor(self, T):
        """Specific internal energy of saturated vapor [J/kg].
        u_V = u_L + h_vap - P*(v_V - v_L) ~ u_L + h_vap - R*T  (ideal-gas flow work)."""
        T = np.asarray(T, dtype=float)
        return self.u_liquid(T) + self.h_vap(T) - self.R_H2 * T

    def h_vapor(self, T):
        """Specific enthalpy of saturated vapor [J/kg] (for vent enthalpy flux)."""
        T = np.asarray(T, dtype=float)
        return self.u_vapor(T) + self.R_H2 * T  # h = u + Pv = u + R T (ideal)

    def total_internal_energy(self, m_L, m_V, T, include_wall=True):
        """Total internal energy of tank contents (+ wall) [J]."""
        U = m_L * self.u_liquid(T) + m_V * self.u_vapor(T)
        if include_wall:
            U += self.m_tank * self.cp_wall * (T - self._T_ref_u)
        return U

    # ------------------------------------------------------------------
    # Equilibrium state recovery:  (m_tot, U_tot) -> (T, P, m_L, m_V)
    # ------------------------------------------------------------------
    def _T_from_mass_energy(self, m_tot, U_tot, m_V):
        """Given total mass, total energy and a vapor mass m_V, invert the
        (linearized) energy relation for T.  Used inside the volume root-find.

        U_tot = (m_L+m_V)*cp_L*(T-Tref) + m_wall*cp_wall*(T-Tref)
                + m_V*(h_vap0 - dhvap_dT*T_nbp) + m_V*(dhvap_dT - R)*T
        with m_L = m_tot - m_V.
        """
        cap = m_tot * self.cp_L + self.m_tank * self.cp_wall \
              + m_V * (self.dhvap_dT - self.R_H2)
        const = -m_tot * self.cp_L * self._T_ref_u \
                - self.m_tank * self.cp_wall * self._T_ref_u \
                + m_V * (self.h_vap0 - self.dhvap_dT * self.T_nbp)
        return (U_tot - const) / cap

    def equilibrium_state(self, m_tot, U_tot):
        """Recover (T, P, m_L, m_V) enforcing the fixed-volume constraint.

        Two unknowns (T, m_V) constrained by:
          (a) energy:  T = T(m_tot, U_tot, m_V)            [_T_from_mass_energy]
          (b) volume:  m_L/rho_L(T) + m_V/rho_V(T,P) = V_tank
        Solved by 1-D bracketing on m_V via a volume residual (monotone).
        """
        m_tot = max(float(m_tot), 1e-9)

        def vol_residual(m_V):
            m_V = min(max(m_V, 1e-12), m_tot - 1e-12)
            m_L = m_tot - m_V
            T = self._T_from_mass_energy(m_tot, U_tot, m_V)
            T = float(np.clip(T, self.T_nbp - 5.0, self.T_crit - 0.5))
            P = float(self.p_sat(T))
            rhoL = float(self.rho_liquid(T))
            rhoV = float(self.rho_vapor_ideal(T, P))
            V = m_L / rhoL + m_V / rhoV
            return V - self.V_tank, T, P

        lo, hi = 1e-9, m_tot - 1e-9
        r_lo = vol_residual(lo)[0]   # almost all liquid -> V small (negative residual)
        r_hi = vol_residual(hi)[0]   # almost all vapor  -> V huge  (positive residual)
        if r_lo * r_hi > 0:
            # no sign change (e.g. fully vapor or fully liquid): pick nearest end
            m_V = lo if abs(r_lo) < abs(r_hi) else hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                rmid = vol_residual(mid)[0]
                if r_lo * rmid <= 0:
                    hi = mid
                else:
                    lo = mid
                    r_lo = rmid
            m_V = 0.5 * (lo + hi)

        _, T, P = vol_residual(m_V)
        m_V = min(max(m_V, 0.0), m_tot)
        m_L = m_tot - m_V
        return T, P, m_L, m_V

    # ------------------------------------------------------------------
    # Initial condition builder
    # ------------------------------------------------------------------
    def initial_state(self, fill_fraction, P0_bar):
        """Build (m_tot, U_tot) from fill fraction and initial pressure."""
        f = float(np.clip(fill_fraction, 0.0, self.fill_max))
        T0 = float(self.t_sat(P0_bar))
        rhoL = float(self.rho_liquid(T0))
        rhoV = float(self.rho_vapor_ideal(T0, P0_bar))
        V_L = self.V_tank * f
        V_V = self.V_tank * (1.0 - f)
        m_L = rhoL * V_L
        m_V = rhoV * V_V
        m_tot = m_L + m_V
        U0 = float(self.total_internal_energy(m_L, m_V, T0))
        return np.array([m_tot, U0]), T0

    # ------------------------------------------------------------------
    # Right-hand side of the ODE system
    # ------------------------------------------------------------------
    def _rhs(self, t, y, T_amb_fn, venting):
        """ODE RHS.  `venting` is a fixed bool for the current integration phase
        (no internal P-crossing branch -> smooth RHS, no solver chatter)."""
        m_tot, U_tot = y
        m_tot = max(m_tot, 1e-9)

        T, P, m_L, m_V = self.equilibrium_state(m_tot, U_tot)
        T_amb = float(T_amb_fn(t))
        Q_in = float(self.heat_leak(T_amb, T))     # W

        if venting:
            # relief valve open: expel saturated vapor at the rate that holds
            # the saturation state -> NBP / set-point boil: m_dot = Q_in/h_vap.
            hv = float(self.h_vap(T))
            m_dot_vent = max(Q_in, 0.0) / hv
        else:
            m_dot_vent = 0.0

        h_V = float(self.h_vapor(T))
        dm_tot = -m_dot_vent
        dU = Q_in - m_dot_vent * h_V
        return [dm_tot, dU]

    def _vent_event(self, t, y, T_amb_fn, venting):
        """Terminal event: P crosses the vent set-point (sealed phase only)."""
        m_tot, U_tot = y
        _, P, _, _ = self.equilibrium_state(max(m_tot, 1e-9), U_tot)
        return P - self.P_vent
    _vent_event.terminal = True
    _vent_event.direction = 1.0

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def simulate(self, fill_fraction=0.90, T_amb_K=298.15, P0_bar=1.01325,
                 duration_s=86400.0, n_steps=400, sealed=True):
        """Integrate the two-phase tank for `duration_s`.

        Parameters
        ----------
        fill_fraction : initial liquid fill (0.05-0.95)
        T_amb_K       : ambient temperature; scalar or callable(t)->K
        P0_bar        : initial tank pressure [bar]
        duration_s    : simulation horizon [s]
        n_steps       : output samples
        sealed        : True = closed tank (self-pressurizes until P_vent);
                        False = vent always open (NBP boil-off, P constant)

        Returns dict of time-series arrays.
        """
        if callable(T_amb_K):
            T_amb_fn = T_amb_K
        else:
            T_amb_fn = lambda t, _T=float(T_amb_K): _T

        y0, T0 = self.initial_state(fill_fraction, P0_bar)
        m0_total = y0[0]

        t_eval = np.linspace(0.0, duration_s, n_steps)

        if sealed:
            # Phase 1: sealed self-pressurization until P hits the vent set-point
            # (terminal event), Phase 2: venting at the set-point.  Splitting at
            # the event keeps each RHS smooth -> no stiff-discontinuity chatter.
            sol1 = solve_ivp(self._rhs, (0.0, duration_s), y0, t_eval=t_eval,
                             args=(T_amb_fn, False), method="LSODA",
                             events=self._vent_event,
                             rtol=1e-7, atol=1e-9, max_step=duration_s / 50.0)
            t_all = list(sol1.t)
            y_all = [sol1.y[0].tolist(), sol1.y[1].tolist()]

            if sol1.status == 1 and len(sol1.t_events[0]) > 0:
                # vent reached -> continue venting to the end of the horizon
                t_vent = float(sol1.t_events[0][0])
                y_vent = sol1.y_events[0][0]
                t_eval2 = t_eval[t_eval > t_vent]
                if t_eval2.size > 0:
                    sol2 = solve_ivp(self._rhs, (t_vent, duration_s), y_vent,
                                     t_eval=t_eval2, args=(T_amb_fn, True),
                                     method="LSODA", rtol=1e-7, atol=1e-9,
                                     max_step=duration_s / 50.0)
                    t_all += list(sol2.t)
                    y_all[0] += sol2.y[0].tolist()
                    y_all[1] += sol2.y[1].tolist()
                    success = sol2.success
                else:
                    success = sol1.success
            else:
                success = sol1.success

            sol_t = np.array(t_all)
            m_tot = np.maximum(np.array(y_all[0]), 1e-9)
            U_tot = np.array(y_all[1])
        else:
            sol = solve_ivp(self._rhs, (0.0, duration_s), y0, t_eval=t_eval,
                            args=(T_amb_fn, True), method="LSODA",
                            rtol=1e-7, atol=1e-9, max_step=duration_s / 50.0)
            sol_t = sol.t
            m_tot = np.maximum(sol.y[0], 1e-9)
            U_tot = sol.y[1]
            success = sol.success

        T = np.empty_like(m_tot)
        P = np.empty_like(m_tot)
        m_L = np.empty_like(m_tot)
        m_V = np.empty_like(m_tot)
        for i, (mt, u) in enumerate(zip(m_tot, U_tot)):
            T[i], P[i], m_L[i], m_V[i] = self.equilibrium_state(mt, u)

        Q_in = self.heat_leak(np.array([T_amb_fn(tt) for tt in sol_t]), T)
        hv = self.h_vap(T)
        m_dot = np.maximum(Q_in, 0.0) / hv               # kg/s instantaneous boil-off
        fill = m_L / (self.rho_liquid(T) * self.V_tank)

        # BOR %/day relative to instantaneous liquid mass
        safe_mL = np.where(m_L > 1e-6, m_L, 1.0)
        BOR = np.where(m_L > 1e-6, m_dot * 86400.0 / safe_mL * 100.0, 0.0)

        m_total = m_L + m_V
        E_stored = m_L * self.LHV                          # MJ (usable liquid)

        return {
            "t": sol_t,
            "m_liquid": m_L,
            "m_vapor": m_V,
            "m_total": m_total,
            "temperature": T,
            "pressure": P,
            "heat_leak_W": Q_in,
            "boiloff_rate_kg_s": m_dot,
            "BOR_pct_day": BOR,
            "fill_fraction": fill,
            "U_eff_W_m2_K": self.u_eff(np.array([T_amb_fn(tt) for tt in sol_t]), T),
            "energy_stored_MJ": E_stored,
            "m0_total": m0_total,
            "success": success,
        }

    # ------------------------------------------------------------------
    # Convenience steady-state metrics (for quick screening / tests)
    # ------------------------------------------------------------------
    def steady_boiloff(self, fill_fraction, T_amb_K, P_bar=1.01325):
        """Instantaneous NBP boil-off metrics at given conditions."""
        T_sat = float(self.t_sat(P_bar))
        Q = float(self.heat_leak(T_amb_K, T_sat))
        hv = float(self.h_vap(T_sat))
        m_dot = max(Q, 0.0) / hv
        rhoL = float(self.rho_liquid(T_sat))
        m_L = rhoL * self.V_tank * float(np.clip(fill_fraction, 0, self.fill_max))
        BOR = (m_dot * 86400.0 / m_L * 100.0) if m_L > 0 else 0.0
        return {
            "heat_leak_W": Q,
            "boiloff_rate_kg_s": m_dot,
            "BOR_pct_day": BOR,
            "T_sat_K": T_sat,
            "stored_mass_kg": m_L,
        }
