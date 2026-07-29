"""
EC067 -- Airborne Wind Energy (AWE) -- F2a Crosswind Pumping-Cycle Model

Physics-lumped (0D/1D) first-principles model of a ground-generation
("yo-yo" / pumping-mode) crosswind kite power system.

------------------------------------------------------------------------
1. Loyd crosswind power limit  (Loyd 1980)
------------------------------------------------------------------------
For a kite flying crosswind on a tether, the wing sweeps the air at an
apparent speed v_a >> v_wind, generating an aerodynamic force
proportional to (CL/CD)^2.  The IDEAL reel-out (drag-power) limit is

        P_loyd = (2/27) * rho * A * CL * (CL/CD)^2 * v_w^3

(Loyd 1980, Eq. for reel-out / "drag" mode; equivalently the lift-mode
ideal P_max = (4/27) rho A v^3 CL^3/CD^2 differs by the reeling-factor
optimisation 4/27 vs 2/27 -- we use the reel-out optimum 2/27 as the
hard upper bound on extractable mechanical power).  No real cycle may
exceed this.

------------------------------------------------------------------------
2. Reeling-factor / traction model  (Schmehl 2018; Fagiano 2012)
------------------------------------------------------------------------
During reel-out the ground-station pays out tether at v_reel = f * v_w
(f = reeling factor, Loyd optimum f = 1/3).  The wind component felt by
the kite is reduced to v_w*(1 - f).  The crosswind traction force is

        F_t = 0.5 * rho * A * CL * G_eff^2 * (v_w (1-f))^2

with effective glide ratio G_eff = CL / (CD_kite + CD_tether_eff),
where the tether contributes an equivalent drag area (1/4 * d * L * CD_t)
referenced to the wing (Schmehl 2018, Eq. tether-drag lumping; factor
1/4 from integrating the linear apparent-speed profile along the line).

Reel-out mechanical power           P_out_mech = F_t * v_reel_out
Reel-out electrical power            P_out_elec = eta_gen * P_out_mech

------------------------------------------------------------------------
3. Reel-in (retraction) phase
------------------------------------------------------------------------
The kite is depowered (CL -> gamma_in * CL, low drag) and pulled back in
at v_in = f_in * v_w.  The ground station MOTORS the tether in, so the
required electrical power is

        P_in_mech = F_in * v_reel_in,  F_in = 0.5 rho A (gamma_in CL) v_in^2
        P_in_elec = P_in_mech / eta_mot      (consumed, negative)

------------------------------------------------------------------------
4. Tether-length ODE  (solve_ivp)
------------------------------------------------------------------------
        dL/dt = +v_reel_out   during reel-out  (L: L_min -> L_max)
        dL/dt = -v_reel_in    during reel-in   (L: L_max -> L_min)

The instantaneous tether drag (hence traction) depends on L, so the ODE
is genuinely coupled to the power.  We integrate one full pumping cycle
and energy-balance it:

        E_cycle = E_out_elec - E_in_elec      (net, must be > 0)
        P_avg   = E_cycle / t_cycle
        duty    = t_out / t_cycle

------------------------------------------------------------------------
References
------------------------------------------------------------------------
    Loyd, M.L. (1980). "Crosswind kite power." J. Energy 4(3):106-111.
    Fagiano, L. & Milanese, M. (2012). "Airborne wind energy: basic
        concepts and physical foundations." Proc. American Control Conf.
    Luchsinger, R.H. (2013). "Pumping cycle kite power." In: Ahrens,
        Diehl, Schmehl (eds.) Airborne Wind Energy, Springer, Ch.3.
    Schmehl, R. (ed.) (2018). Airborne Wind Energy: Advances in
        Technology Development and Research. Springer (Green Energy Tech).
"""

import numpy as np
from scipy.integrate import solve_ivp


class AWE_PumpingCycle_F2a:
    """Crosswind kite pumping-cycle power system (ground-generation)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_wing"]["value"]
        self.CL = u["CL"]["value"]
        self.CD_kite = u["CD_kite"]["value"]
        self.CD_tether = u["CD_tether"]["value"]
        self.d_tether = u["d_tether"]["value"]
        self.L_min = u["L_min"]["value"]
        self.L_max = u["L_max"]["value"]
        self.rho = u["rho"]["value"]
        self.eta_gen = u["eta_gen"]["value"]
        self.eta_mot = u["eta_mot"]["value"]
        self.f_out = u["f_out"]["value"]
        self.f_in = u["f_in"]["value"]
        self.gamma_in = u["gamma_in"]["value"]
        self.P_rated = u["P_rated"]["value"]
        self.v_cut_in = u["v_cut_in"]["value"]
        self.v_cut_out = u["v_cut_out"]["value"]

    # ------------------------------------------------------------------
    # Loyd ideal crosswind power limit (hard upper bound)
    # ------------------------------------------------------------------
    def loyd_power_limit(self, v_w):
        """Ideal Loyd (1980) reel-out crosswind power [W].
        P = (2/27) rho A CL (CL/CD)^2 v_w^3 ."""
        G = self.CL / self.CD_kite
        return (2.0 / 27.0) * self.rho * self.A * self.CL * G ** 2 * v_w ** 3

    # ------------------------------------------------------------------
    # Effective glide ratio including tether drag at tether length L
    # ------------------------------------------------------------------
    def tether_drag_area_coeff(self, L):
        """Equivalent kite-referenced drag coefficient added by the tether.
        CD_t,eff = (1/4) * CD_tether * d * L / A   (Schmehl 2018 lumping)."""
        return 0.25 * self.CD_tether * self.d_tether * L / self.A

    def effective_glide_ratio(self, L):
        CD_eff = self.CD_kite + self.tether_drag_area_coeff(L)
        return self.CL / CD_eff

    # ------------------------------------------------------------------
    # Reel-out traction force [N] at wind v_w and tether length L
    # ------------------------------------------------------------------
    def traction_force(self, v_w, L):
        """Crosswind traction force during reel-out [N]."""
        G_eff = self.effective_glide_ratio(L)
        v_app = v_w * (1.0 - self.f_out)          # reduced wind component
        # Crosswind force ~ 0.5 rho A CL * (G_eff*v_app)^2  (apparent speed
        # = G_eff * wind-along-tether; classic crosswind scaling)
        F = 0.5 * self.rho * self.A * self.CL * (G_eff ** 2) * (v_app ** 2)
        return F

    def reel_in_force(self, v_w):
        """Tether tension during depowered reel-in [N] (motor must overcome)."""
        v_in = self.f_in * v_w
        return 0.5 * self.rho * self.A * (self.gamma_in * self.CL) * (v_in ** 2)

    # ------------------------------------------------------------------
    # Instantaneous electrical powers
    # ------------------------------------------------------------------
    def power_reel_out(self, v_w, L):
        """Electrical power generated during reel-out [W] (>0)."""
        F = self.traction_force(v_w, L)
        v_reel = self.f_out * v_w
        return self.eta_gen * F * v_reel

    def power_reel_in(self, v_w):
        """Electrical power consumed during reel-in [W] (>0 magnitude)."""
        F = self.reel_in_force(v_w)
        v_reel = self.f_in * v_w
        return (F * v_reel) / self.eta_mot

    # ------------------------------------------------------------------
    # Tether-length ODEs (solve_ivp) for one pumping cycle
    # ------------------------------------------------------------------
    def simulate(self, v_wind, n_eval=200):
        """
        Integrate one full pumping cycle (reel-out then reel-in) with the
        tether-length ODE dL/dt = +/- v_reel, capping reel-out electrical
        power at P_rated.  Returns time-series + cycle scalars.

        Parameters
        ----------
        v_wind : float
            Wind speed at operating altitude [m/s].
        n_eval : int
            Output samples per phase.

        Returns
        -------
        dict with keys:
            t, L, phase, P_elec, F_traction (time-series arrays);
            P_loyd_limit, P_avg, P_out_avg, P_in_avg, duty,
            t_out, t_in, t_cycle, E_out, E_in, E_net,
            traction_peak, capacity_factor, energy_residual.
        """
        v_w = float(v_wind)

        # below cut-in or above cut-out: parked, no power
        if v_w < self.v_cut_in or v_w > self.v_cut_out or v_w <= 0.0:
            z = np.zeros(2)
            return {
                "t": np.array([0.0, 1.0]), "L": np.array([self.L_min, self.L_min]),
                "phase": np.array([0, 0]), "P_elec": z, "F_traction": z,
                "P_loyd_limit": self.loyd_power_limit(max(v_w, 0.0)),
                "P_avg": 0.0, "P_out_avg": 0.0, "P_in_avg": 0.0, "duty": 0.0,
                "t_out": 0.0, "t_in": 0.0, "t_cycle": 1.0,
                "E_out": 0.0, "E_in": 0.0, "E_net": 0.0,
                "traction_peak": 0.0, "capacity_factor": 0.0,
                "energy_residual": 0.0,
            }

        v_reel_out = self.f_out * v_w
        v_reel_in = self.f_in * v_w
        stroke = self.L_max - self.L_min

        # ---- Phase 1: reel-out, dL/dt = +v_reel_out (L_min -> L_max) ----
        t_out = stroke / v_reel_out

        def rhs_out(t, y):
            return [v_reel_out]

        def stop_out(t, y):
            return y[0] - self.L_max
        stop_out.terminal = True
        stop_out.direction = 1

        t_eval_out = np.linspace(0.0, t_out, n_eval)
        sol_out = solve_ivp(
            rhs_out, (0.0, t_out * 1.01), [self.L_min],
            t_eval=t_eval_out, events=stop_out,
            method="RK45", rtol=1e-8, atol=1e-9, max_step=t_out / 20.0,
        )
        L_out = sol_out.y[0]
        t_out_arr = sol_out.t

        # ---- Phase 2: reel-in, dL/dt = -v_reel_in (L_max -> L_min) ----
        t_in = stroke / v_reel_in

        def rhs_in(t, y):
            return [-v_reel_in]

        def stop_in(t, y):
            return y[0] - self.L_min
        stop_in.terminal = True
        stop_in.direction = -1

        # integrate slightly past t_in so the terminal event (L=L_min) is
        # bracketed; t_eval up to t_in keeps the last sample at the event.
        t_eval_in = np.linspace(0.0, t_in, n_eval)
        sol_in = solve_ivp(
            rhs_in, (0.0, t_in * 1.05), [self.L_max],
            t_eval=t_eval_in, events=stop_in,
            method="RK45", rtol=1e-8, atol=1e-9, max_step=t_in / 20.0,
        )
        L_in = sol_in.y[0]
        t_in_arr = sol_in.t
        # the analytic stroke ends exactly at L_min; snap the final sample
        # (last linspace node coincides with the terminal event time).
        L_in[-1] = self.L_min

        # ---- Powers / forces along the integrated trajectories ----
        F_out = np.array([self.traction_force(v_w, L) for L in L_out])
        P_out_mech = self.eta_gen * F_out * v_reel_out
        # cap electrical power at rated (ground-station / generator limit)
        P_out = np.minimum(P_out_mech, self.P_rated)

        P_in_const = self.power_reel_in(v_w)          # ~constant during reel-in
        F_in_const = self.reel_in_force(v_w)
        P_in = -np.full_like(L_in, P_in_const)        # consumed -> negative
        F_in = np.full_like(L_in, F_in_const)

        # ---- Energies (trapezoidal integration of the ODE-resolved series) ----
        E_out = np.trapezoid(P_out, t_out_arr) if hasattr(np, "trapezoid") \
            else np.trapz(P_out, t_out_arr)
        E_in_signed = np.trapezoid(P_in, t_in_arr) if hasattr(np, "trapezoid") \
            else np.trapz(P_in, t_in_arr)
        E_in = -E_in_signed                            # positive magnitude
        E_net = E_out - E_in

        t_cycle = t_out_arr[-1] + t_in_arr[-1]
        P_avg = E_net / t_cycle
        P_out_avg = E_out / t_out_arr[-1]
        P_in_avg = E_in / t_in_arr[-1]
        duty = t_out_arr[-1] / t_cycle

        # ---- Assemble continuous time-series across the cycle ----
        t_full = np.concatenate([t_out_arr, t_out_arr[-1] + t_in_arr])
        L_full = np.concatenate([L_out, L_in])
        phase_full = np.concatenate([
            np.ones_like(t_out_arr), -np.ones_like(t_in_arr)])
        P_full = np.concatenate([P_out, P_in])
        F_full = np.concatenate([F_out, F_in])

        # energy-conservation residual: integral of P_full vs E_net
        E_check = np.trapezoid(P_full, t_full) if hasattr(np, "trapezoid") \
            else np.trapz(P_full, t_full)
        energy_residual = abs(E_check - E_net) / max(abs(E_net), 1e-9)

        return {
            "t": t_full,
            "L": L_full,
            "phase": phase_full,
            "P_elec": P_full,
            "F_traction": F_full,
            "P_loyd_limit": self.loyd_power_limit(v_w),
            "P_avg": P_avg,
            "P_out_avg": P_out_avg,
            "P_in_avg": P_in_avg,
            "duty": duty,
            "t_out": float(t_out_arr[-1]),
            "t_in": float(t_in_arr[-1]),
            "t_cycle": float(t_cycle),
            "E_out": float(E_out),
            "E_in": float(E_in),
            "E_net": float(E_net),
            "traction_peak": float(np.max(F_out)),
            "capacity_factor": float(min(max(P_avg / self.P_rated, 0.0), 1.0)),
            "energy_residual": float(energy_residual),
        }
