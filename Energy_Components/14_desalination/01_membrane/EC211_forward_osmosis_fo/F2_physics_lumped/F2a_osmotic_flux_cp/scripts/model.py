"""
EC211 -- Forward Osmosis (FO) -- F2a Physics-Lumped Osmotic Flux Model

0D lumped model of an osmotically driven membrane module. Water is transported
from a dilute feed to a concentrated draw solution by the osmotic-pressure
difference across a semipermeable membrane -- there is NO applied hydraulic
pressure (this is the defining feature of FO vs RO).

------------------------------------------------------------------------------
Water flux (the FLUX-LIMITED governing equation, McCutcheon & Elimelech 2006):

In an ideal membrane the driving force is  d_pi = pi_draw - pi_feed  and
Jw = A * d_pi. But concentration polarization severely reduces the *effective*
osmotic pressures at the active layer. For the common configuration (active
layer facing draw, AL-DS), the implicit flux equation is:

    Jw = A * [ pi_draw * exp(-Jw/k_ecp)  -  pi_feed * exp(Jw*S/D) ]  - B*...

We use the widely cited explicit form (active layer facing feed, AL-FS, the
typical FO/“PRO-mode-off” orientation for desalination), where the draw side
suffers dilutive INTERNAL concentration polarization in the porous support and
the feed side suffers concentrative EXTERNAL CP:

    Jw = A * [ pi_draw * exp(-Jw * K)  -  pi_feed * exp(Jw / k_ecp) ]

with K = S / D the solute resistivity of the support layer (s/m), S the
membrane structural parameter (S = t*tau/eps) and D the salt diffusivity.
Reverse salt flux: Js = B * (c_draw,m - c_feed,m), with B the salt permeability.
The exp(-Jw*K) term is the ICP factor -- it is the DOMINANT FO flux limiter:
as Jw rises, the effective draw osmotic pressure collapses, which self-limits
the flux. We solve this scalar implicit equation for Jw with Brent's method.

Osmotic pressure (van't Hoff, dilute-electrolyte):  pi = phi * i * c * R * T
  phi = osmotic coefficient, i = van't Hoff dissociation factor, c in mol/m3.

------------------------------------------------------------------------------
Lumped module ODE (scipy.solve_ivp) -- draw-solution dilution & mass balance:

State y = [V_draw (m3), n_salt_draw (mol)].
    dV/dt      = +Qw           (permeate water enters draw tank)
    dn_salt/dt = -Js * Am / 1  (reverse salt flux LEAVES draw -> feed)
where Qw = Jw * Am (m3/s), Jw in m/s, Am = membrane area.
The draw concentration c_draw = n_salt / V decreases as water dilutes it and
salt leaks back -- this progressively kills the osmotic driving force.
Mass is conserved: water gained by draw = water lost by feed; salt lost by
draw (reverse flux) = salt gained by feed.

Specific energy: the FO step itself is nearly energy-free (osmotically driven,
only low-grade circulation pumping), but the draw solution must be RE-
concentrated downstream. Reported SEC is therefore dominated by regeneration:
    SEC_total ~ SEC_regen (kWh per m3 of product water).  -- McGinnis 2008

References:
    McCutcheon, J.R. & Elimelech, M. (2006). Influence of concentrative and
        dilutive internal concentration polarization on flux behavior in
        forward osmosis. J. Membr. Sci. 284, 237-247.
    Cath, T.Y., Childress, A.E. & Elimelech, M. (2006). Forward osmosis:
        Principles, applications, and recent developments. J. Membr. Sci.
        281, 70-87.
    McGinnis, R.L. & Elimelech, M. (2008). Global challenges in energy and
        water supply: the promise of engineered osmosis. Environ. Sci.
        Technol. 42, 8625-8629.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Unit conversions
LMH_BAR_TO_SI = 1.0 / (1000.0 * 3600.0 * 1e5)   # L/(m2.h.bar) -> m/(s.Pa)
LMH_TO_SI = 1.0 / (1000.0 * 3600.0)             # L/(m2.h) -> m/s  (m3/m2/s)
BAR_TO_PA = 1e5


class ForwardOsmosisF2a:
    """Forward-osmosis lumped module: osmotic flux + ICP/ECP + draw dilution ODE."""

    R = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        self.A_perm = u["A_perm"]["value"] * LMH_BAR_TO_SI    # m/(s.Pa)
        self.B_salt = u["B_salt"]["value"] * LMH_TO_SI        # m/s
        self.Am = u["membrane_area_m2"]["value"]              # m2
        self.k_ecp = u["k_ecp"]["value"]                      # m/s
        self.D_salt = u["D_salt"]["value"]                    # m2/s
        self.S_struct = u["S_structural_m"]["value"]          # m
        self.i_vh = u["i_vant_hoff"]["value"]
        self.phi = u["phi_osmotic"]["value"]
        self.T = u["T_op_K"]["value"]                         # K
        self.M_salt = u["M_salt_kg_mol"]["value"]             # kg/mol
        self.V_draw0 = u["V_draw0_m3"]["value"]               # m3
        self.c_draw0 = u["c_draw0_mol_m3"]["value"]           # mol/m3
        self.c_feed = u["c_feed_mol_m3"]["value"]             # mol/m3
        self.SEC_regen = u["SEC_regen_kWh_m3"]["value"]       # kWh/m3

        # support-layer solute resistivity K = S/D  (s/m)
        self.K_icp = self.S_struct / self.D_salt

    # -- osmotic pressure (van't Hoff) ------------------------------------
    def osmotic_pressure(self, c_mol_m3):
        """pi [Pa] for molar concentration c [mol/m3]."""
        c = np.asarray(c_mol_m3, dtype=float)
        return self.phi * self.i_vh * c * self.R * self.T

    # -- implicit water flux with ICP (draw) + ECP (feed) -----------------
    def water_flux(self, c_draw, c_feed=None):
        """
        Solve the McCutcheon-Elimelech implicit flux equation for Jw [m/s].
        AL-FS orientation: dilutive ICP on the draw side (exp(-Jw*K)),
        concentrative ECP on the feed side (exp(+Jw/k_ecp)).
        """
        if c_feed is None:
            c_feed = self.c_feed
        pi_draw = self.osmotic_pressure(c_draw)
        pi_feed = self.osmotic_pressure(c_feed)

        # No driving force -> no flux (or reverse not allowed here)
        if pi_draw <= pi_feed:
            return 0.0

        def residual(Jw):
            pi_draw_eff = pi_draw * np.exp(-Jw * self.K_icp)
            pi_feed_eff = pi_feed * np.exp(Jw / self.k_ecp)
            return self.A_perm * (pi_draw_eff - pi_feed_eff) - Jw

        # Bracket: Jw in (0, A*pi_draw]; ideal (no CP) flux is the upper bound.
        Jw_ideal = self.A_perm * (pi_draw - pi_feed)
        lo, hi = 0.0, max(Jw_ideal, 1e-12)
        # residual(0) = A*(pi_draw - pi_feed) > 0 ; residual(hi) <= 0
        if residual(hi) > 0:
            hi *= 2.0
        try:
            Jw = brentq(residual, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=200)
        except ValueError:
            Jw = 0.0
        return max(Jw, 0.0)

    def salt_flux(self, c_draw, c_feed=None):
        """Reverse salt flux Js [mol/(m2.s)] from draw -> feed."""
        if c_feed is None:
            c_feed = self.c_feed
        # Js = B * (c_draw - c_feed); B in m/s, c in mol/m3 -> mol/(m2.s)
        return self.B_salt * max(c_draw - c_feed, 0.0)

    # -- lumped module ODE -------------------------------------------------
    def _rhs(self, t, y):
        V_draw, n_salt = y
        V_draw = max(V_draw, 1e-9)
        c_draw = n_salt / V_draw                      # mol/m3
        Jw = self.water_flux(c_draw)                  # m/s
        Js = self.salt_flux(c_draw)                   # mol/(m2.s)
        Qw = Jw * self.Am                             # m3/s water into draw
        dV = Qw
        dn = -Js * self.Am                            # mol/s salt leaving draw
        return [dV, dn]

    def simulate(self, duration_s=3600.0, n_points=200,
                 V_draw0=None, c_draw0=None, c_feed=None):
        """
        Integrate the draw-tank dilution ODE over `duration_s`.

        Returns dict with time series of volume, concentration, fluxes,
        permeate produced, and the regeneration specific energy.
        """
        if V_draw0 is None:
            V_draw0 = self.V_draw0
        if c_draw0 is None:
            c_draw0 = self.c_draw0
        if c_feed is not None:
            self.c_feed = c_feed

        n_salt0 = c_draw0 * V_draw0
        t_eval = np.linspace(0.0, duration_s, n_points)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [V_draw0, n_salt0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-10,
            max_step=duration_s / 50.0,
        )

        V = sol.y[0]
        n_salt = sol.y[1]
        c_draw = n_salt / np.maximum(V, 1e-12)

        Jw = np.array([self.water_flux(c) for c in c_draw])      # m/s
        Js = np.array([self.salt_flux(c) for c in c_draw])       # mol/(m2.s)

        # Convenience conversions
        Jw_LMH = Jw / LMH_TO_SI                                   # L/(m2.h)
        Js_gMH = Js * self.M_salt * 1000.0 * 3600.0              # g/(m2.h)
        pi_draw = self.osmotic_pressure(c_draw) / BAR_TO_PA       # bar
        pi_feed = self.osmotic_pressure(self.c_feed) / BAR_TO_PA  # bar

        permeate_m3 = V - V_draw0                                 # cumulative water gained
        # specific reverse salt flux selectivity Js/Jw [g/L]
        with np.errstate(divide="ignore", invalid="ignore"):
            Js_Jw = np.where(Jw > 1e-15,
                             (Js * self.M_salt) / Jw * 1000.0,    # g per L permeate
                             0.0)

        return {
            "t": sol.t,
            "V_draw_m3": V,
            "c_draw_mol_m3": c_draw,
            "Jw_m_s": Jw,
            "Jw_LMH": Jw_LMH,
            "Js_mol_m2_s": Js,
            "Js_gMH": Js_gMH,
            "Js_Jw_gL": Js_Jw,
            "pi_draw_bar": pi_draw,
            "pi_feed_bar": pi_feed,
            "permeate_m3": permeate_m3,
            "SEC_regen_kWh_m3": self.SEC_regen,
        }
