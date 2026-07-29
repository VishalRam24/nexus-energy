"""
EC203 -- Membrane-Based CO2 Separation -- F2a Solution-Diffusion Cross-Flow Module

Physics-lumped (1D-in-area) gas-separation membrane model. The single-stage
module is integrated along cumulative membrane area with scipy.solve_ivp,
giving the classic permeate-purity vs CO2-recovery tradeoff.

------------------------------------------------------------------------------
Permeation (solution-diffusion, Wijmans & Baker 1995):
    Local trans-membrane molar flux of species i per unit area:
        J_i = Q_i * ( p_feed * x_i  -  p_perm * y_i )           [mol/(m2.s)]
    where Q_i is the permeance, p the total pressure on each side, x_i the
    local retentate (feed-side) mole fraction and y_i the local permeate
    mole fraction. Driving force is the PARTIAL-PRESSURE difference.
    Selectivity   alpha = Q_CO2 / Q_N2  controls intrinsic separation.

Cross-flow module mass balance (Baker 2012, Ch.8; Geankoplis 2003, Ch.13):
    Treat membrane as a series of differential area elements. The feed (high
    pressure) flows along the module losing the more-permeable species; on the
    low-pressure side the locally-produced permeate is swept away (cross-flow:
    permeate at a point does not back-mix). State variables integrated vs the
    cumulative area A are the retentate molar flows of each species:
        dF_CO2/dA = -J_CO2
        dF_N2 /dA = -J_N2
    The local permeate composition is set by the flux RATIO at that point,
    which in turn depends on the local retentate composition x_i and the
    pressure ratio -- this is the implicit "permeation equation" solved
    pointwise (Geankoplis Eq. 13.4-x). Collected (mixed-cup) permeate purity
    follows from integrating the species fluxes over the whole area.

Pressure-ratio limit (Baker 2012):
    The permeate CO2 fraction can never exceed the value set by the pressure
    ratio phi = p_feed/p_perm even with infinite selectivity:
        y_CO2  <=  phi * x_CO2          (enrichment ceiling)
    This caps achievable purity at low pressure ratio -- a defining feature
    of membrane CO2 capture from near-atmospheric flue gas.

Purity-recovery tradeoff:
    Increasing area (-> "stage cut" theta = permeate/feed) raises CO2 recovery
    but dilutes permeate with N2, lowering purity. The model reproduces this
    monotonic tradeoff and the selectivity / pressure-ratio dependence.

References:
    Merkel, T.C. et al. (2010). Power plant post-combustion CO2 capture: an
      opportunity for membranes. J. Membr. Sci. 359:126-139.
    Baker, R.W. (2012). Membrane Technology and Applications, 3rd ed., Wiley.
    Wijmans, J.G. & Baker, R.W. (1995). The solution-diffusion model: a review.
      J. Membr. Sci. 107:1-21.
    Geankoplis, C.J. (2003). Transport Processes and Separation Process
      Principles, 4th ed., Ch.13.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Unit conversion: 1 GPU = 1e-6 cm3(STP)/(cm2.s.cmHg)
# Convert to mol/(m2.s.Pa):
#   1 cm3(STP) = 1/22414 mol ; 1 cm2 = 1e-4 m2 ; 1 cmHg = 1333.22 Pa
#   => GPU_to_SI = 1e-6 * (1/22414) / (1e-4) / 1333.22  mol/(m2.s.Pa)
GPU_TO_SI = 1e-6 * (1.0 / 22414.0) / 1e-4 / 1333.22  # mol/(m2.s.Pa)
BAR_TO_PA = 1.0e5


class MembraneF2a:
    """Solution-diffusion cross-flow CO2/N2 membrane module (1D area ODE)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_CO2 = u["Q_CO2_GPU"]["value"] * GPU_TO_SI          # mol/(m2.s.Pa)
        self.alpha = u["selectivity_CO2_N2"]["value"]             # -
        self.Q_N2 = self.Q_CO2 / self.alpha                       # mol/(m2.s.Pa)
        self.area = u["area_m2"]["value"]                         # m2
        self.F_feed = u["feed_flow_mol_s"]["value"]               # mol/s
        self.y_feed = u["y_CO2_feed"]["value"]                    # -
        self.p_feed = u["P_feed_bar"]["value"] * BAR_TO_PA        # Pa
        self.p_perm = u["P_perm_bar"]["value"] * BAR_TO_PA        # Pa
        self.T = u["T_K"]["value"]                                # K

    # ------------------------------------------------------------------
    @property
    def pressure_ratio(self):
        """phi = p_feed / p_perm (enrichment ceiling)."""
        return self.p_feed / self.p_perm

    # ------------------------------------------------------------------
    def local_permeate_fraction(self, x_co2, p_feed=None, p_perm=None):
        """
        Local permeate CO2 mole fraction y produced at a point where the
        retentate (feed-side) CO2 fraction is x_co2.

        From flux balance at the membrane face (cross-flow, no permeate
        back-mixing). The permeate composition equals the ratio of CO2 flux
        to total flux (Geankoplis 2003, Eq. 13.4-x; Baker 2012, Ch.8):
            y = J_CO2 / (J_CO2 + J_N2)
        with J_i = Q_i ( p_f x_i - p_p y_i ). This is implicit in y because
        y appears in the partial-pressure driving force. The residual
            g(y) = y (J_CO2 + J_N2) - J_CO2
        is monotone on [0,1] (g(0) = -J_CO2 <= 0, g(1) = J_N2 >= 0), so it
        is bracketed and solved robustly by bisection using the EXACT clamped
        fluxes (clamping enforces that no species permeates against an adverse
        partial-pressure gradient).
        """
        if p_feed is None:
            p_feed = self.p_feed
        if p_perm is None:
            p_perm = self.p_perm
        x = float(np.clip(x_co2, 0.0, 1.0))
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0

        def fluxes(y):
            J_co2 = self.Q_CO2 * max(p_feed * x - p_perm * y, 0.0)
            J_n2 = self.Q_N2 * max(p_feed * (1.0 - x) - p_perm * (1.0 - y), 0.0)
            return J_co2, J_n2

        def g(y):
            J_co2, J_n2 = fluxes(y)
            tot = J_co2 + J_n2
            if tot <= 0.0:
                return y  # degenerate: drive y toward 0
            return y - J_co2 / tot

        # g is monotone increasing in y on [0, 1]; bisection.
        lo, hi = 0.0, 1.0
        g_lo, g_hi = g(lo), g(hi)
        if g_lo >= 0.0:
            y = 0.0
        elif g_hi <= 0.0:
            y = 1.0
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if g(mid) < 0.0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-10:
                    break
            y = 0.5 * (lo + hi)

        # Pressure-ratio enrichment ceiling: y <= phi * x  (Baker 2012)
        y = min(y, self.pressure_ratio * x)
        return float(np.clip(y, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _flux(self, x_co2):
        """Local species fluxes [mol/(m2.s)] given retentate CO2 fraction."""
        y = self.local_permeate_fraction(x_co2)
        x = float(np.clip(x_co2, 0.0, 1.0))
        J_co2 = self.Q_CO2 * max(self.p_feed * x - self.p_perm * y, 0.0)
        J_n2 = self.Q_N2 * max(self.p_feed * (1.0 - x) - self.p_perm * (1.0 - y), 0.0)
        return J_co2, J_n2

    # ------------------------------------------------------------------
    def simulate(self, area_m2=None, feed_flow_mol_s=None, y_CO2_feed=None,
                 n_eval=200):
        """
        Integrate the cross-flow module mass balance along cumulative area.

        State y = [F_CO2, F_N2] = retentate molar flows [mol/s].
        ODE vs area A:
            dF_CO2/dA = -J_CO2(x)
            dF_N2 /dA = -J_N2(x)      with x = F_CO2/(F_CO2+F_N2)

        Returns dict with area-resolved profiles and module-outlet summary:
            recovery  = permeated CO2 / fed CO2
            purity    = CO2 fraction in collected (mixed) permeate
            stage_cut = total permeate / total feed
        """
        A_tot = self.area if area_m2 is None else float(area_m2)
        F_in = self.F_feed if feed_flow_mol_s is None else float(feed_flow_mol_s)
        y0 = self.y_feed if y_CO2_feed is None else float(y_CO2_feed)

        F_co2_0 = F_in * y0
        F_n2_0 = F_in * (1.0 - y0)

        def rhs(A, F):
            F_co2, F_n2 = F
            tot = F_co2 + F_n2
            if tot <= 1e-12:
                return [0.0, 0.0]
            x = F_co2 / tot
            J_co2, J_n2 = self._flux(x)
            return [-J_co2, -J_n2]

        A_eval = np.linspace(0.0, A_tot, n_eval)

        sol = solve_ivp(
            rhs, (0.0, A_tot), [F_co2_0, F_n2_0],
            t_eval=A_eval, method="LSODA", rtol=1e-8, atol=1e-12,
            max_step=A_tot / 50.0 if A_tot > 0 else np.inf,
        )

        A = sol.t
        F_co2 = np.clip(sol.y[0], 0.0, None)
        F_n2 = np.clip(sol.y[1], 0.0, None)

        # Cumulative permeate (what has passed through up to area A)
        perm_co2 = F_co2_0 - F_co2
        perm_n2 = F_n2_0 - F_n2
        perm_tot = perm_co2 + perm_n2

        with np.errstate(divide="ignore", invalid="ignore"):
            permeate_purity = np.where(perm_tot > 1e-15, perm_co2 / perm_tot, y0)
            retentate_x = np.where((F_co2 + F_n2) > 1e-15,
                                   F_co2 / (F_co2 + F_n2), 0.0)

        recovery = perm_co2[-1] / F_co2_0 if F_co2_0 > 0 else 0.0
        stage_cut = perm_tot[-1] / F_in if F_in > 0 else 0.0

        return {
            "area": A,
            "F_CO2_retentate": F_co2,
            "F_N2_retentate": F_n2,
            "retentate_x_CO2": retentate_x,
            "permeate_purity": permeate_purity,
            "cum_permeate_CO2": perm_co2,
            "cum_permeate_total": perm_tot,
            # module-outlet scalars
            "recovery": float(recovery),
            "purity": float(permeate_purity[-1]),
            "stage_cut": float(stage_cut),
            "retentate_CO2_fraction": float(retentate_x[-1]),
            "permeate_flow_mol_s": float(perm_tot[-1]),
            "retentate_flow_mol_s": float(F_co2[-1] + F_n2[-1]),
            "pressure_ratio": self.pressure_ratio,
        }

    # ------------------------------------------------------------------
    def area_for_recovery(self, target_recovery, feed_flow_mol_s=None,
                          y_CO2_feed=None, A_max_factor=200.0,
                          tol=1e-3, max_iter=80):
        """
        Solve for the membrane area giving a target CO2 recovery.

        Recovery is monotone increasing in area, so a bounded bisection is
        used (deterministic, fixed iteration cap -- avoids the rare
        non-convergence of a generic root finder on the very flat
        high-recovery tail). Returns area [m2].
        """
        F_in = self.F_feed if feed_flow_mol_s is None else float(feed_flow_mol_s)
        y0 = self.y_feed if y_CO2_feed is None else float(y_CO2_feed)

        def rec(A):
            return self.simulate(area_m2=A, feed_flow_mol_s=F_in,
                                 y_CO2_feed=y0, n_eval=60)["recovery"]

        lo, hi = 1e-3, self.area * A_max_factor
        if rec(hi) < target_recovery:
            return hi  # cannot reach target within bound

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            r = rec(mid)
            if abs(r - target_recovery) < tol:
                return float(mid)
            if r < target_recovery:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    # ------------------------------------------------------------------
    def two_stage(self, area1_m2, area2_m2, feed_flow_mol_s=None,
                  y_CO2_feed=None):
        """
        Two-stage cascade: permeate of stage 1 is re-compressed and fed to
        stage 2 to boost purity (Merkel 2010 multi-stage scheme). Returns the
        final (stage-2) permeate purity and overall CO2 recovery.
        """
        F_in = self.F_feed if feed_flow_mol_s is None else float(feed_flow_mol_s)
        y0 = self.y_feed if y_CO2_feed is None else float(y_CO2_feed)

        s1 = self.simulate(area_m2=area1_m2, feed_flow_mol_s=F_in, y_CO2_feed=y0)
        F_perm1 = s1["permeate_flow_mol_s"]
        y_perm1 = s1["purity"]

        if F_perm1 <= 1e-12:
            return {"overall_recovery": 0.0, "final_purity": y0, "stage1": s1,
                    "stage2": None}

        s2 = self.simulate(area_m2=area2_m2, feed_flow_mol_s=F_perm1,
                           y_CO2_feed=y_perm1)
        # overall recovery = (CO2 in stage2 permeate) / (CO2 in original feed)
        co2_feed = F_in * y0
        co2_final = s2["cum_permeate_CO2"][-1]
        overall_recovery = co2_final / co2_feed if co2_feed > 0 else 0.0
        return {
            "overall_recovery": float(overall_recovery),
            "final_purity": float(s2["purity"]),
            "stage1_purity": float(y_perm1),
            "stage1_recovery": float(s1["recovery"]),
            "stage1": s1,
            "stage2": s2,
        }
