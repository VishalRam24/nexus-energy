"""
EC208 -- CO2 Geological Sequestration -- F2a Physics-Lumped Reservoir / Trapping Model

A 0D (lumped "tank") first-principles model of supercritical CO2 injection into a
deep saline aquifer. Couples injectivity, pressure build-up, plume growth and the
four IPCC trapping mechanisms into a single ODE system integrated with
scipy.integrate.solve_ivp.

------------------------------------------------------------------------------
State vector  y = [M_mobile, M_residual, M_dissolved, M_mineral, M_injected]  (kg CO2)
------------------------------------------------------------------------------
  M_mobile     : free (mobile, buoyant) supercritical CO2 plume         -> structural trapping
  M_residual   : CO2 immobilised as residual saturation (snap-off)      -> residual trapping
  M_dissolved  : CO2 dissolved into the formation brine                 -> solubility trapping
  M_mineral    : CO2 fixed as carbonate minerals                        -> mineral trapping
  M_injected   : cumulative injected mass (book-keeping integral of m_inj)

Mass-conservation invariant (enforced in tests):
    M_mobile + M_residual + M_dissolved + M_mineral == M_injected == integral(m_inj dt)

------------------------------------------------------------------------------
Governing equations
------------------------------------------------------------------------------
1. INJECTIVITY -- steady radial Darcy flow (Bachu 2003; Nordbotten & Celia 2006):

        Q_res = 2*pi*k*h*(P_bh - P_res) / (mu_co2 * (ln(r_e/r_w) + S))      [m3/s]
        m_inj = Q_res * rho_co2                                             [kg/s]

   Bottomhole pressure adds the hydrostatic CO2 column to the wellhead pressure:
        P_bh = P_wh + rho_co2 * g * depth
   Injection is shut whenever P_bh would exceed the fracture pressure
        P_frac = frac_gradient * depth
   (the model caps the realised injection so P_bh <= P_frac), and during the
   post-injection period m_inj = 0.

2. PRESSURE BUILD-UP -- compressible tank (material balance, van der Meer 1993):

        P_res = P0 + (V_co2_inplace_eq) / (ct * V_pore)        [Pa]
   where the *effective* reservoir-volume increment driving pressure is the
   pore volume currently occupied by mobile + residual CO2 minus the volume of
   brine displaced; lumped here as dP = V_free_CO2 / (ct * V_pore) using the
   volume of free (mobile+residual) CO2 at reservoir density. Dissolved and
   mineralised CO2 no longer pressurise the system (they occupy brine/solid).

3. PLUME / SATURATION:
        S_co2_avg = V_free_CO2 / V_pore         (average CO2 saturation, lumped)
        plume_radius = sqrt( V_free_CO2 / (pi * h * phi * (1 - Swi)) )   (Nordbotten 2006)

4. TRAPPING DYNAMICS (the four mechanisms evolving in time):
   (a) Structural  = mobile plume held beneath caprock (state M_mobile itself).
   (b) Residual    : as the plume migrates / is imbibed, a fraction of the swept
                     mobile CO2 snaps off at the residual saturation. Modelled as a
                     first-order transfer proportional to the *trailing* mobile CO2:
                         dM_residual/dt = k_res * M_mobile
                     with k_res derived from Sgr and the characteristic imbibition
                     timescale (Nordbotten & Celia 2006; Juanes et al. 2006).
   (c) Solubility  : convective dissolution of mobile CO2 into under-saturated brine,
                     bounded by the brine solubility capacity:
                         dM_dissolved/dt = k_sol * M_mobile * (1 - C_diss/C_sat_cap)
                     (Pruess & Spycher 2007; Ennis-King & Paterson 2005).
   (d) Mineral     : slow carbonation of dissolved CO2 (centuries):
                         dM_mineral/dt = k_min * M_dissolved
                     (Gunter et al. 1997; IPCC 2005 Ch.5).

   Security increases over time: free/structural (least secure) -> residual ->
   dissolved -> mineral (most secure), matching the IPCC (2005) trapping-storage
   security pyramid.

------------------------------------------------------------------------------
References
------------------------------------------------------------------------------
  IPCC (2005). Special Report on Carbon Dioxide Capture and Storage, Ch.5
      "Underground Geological Storage". Cambridge Univ. Press.
  Bachu, S. (2003). Environ. Geol. 44, 277-289.
  Nordbotten, J.M. & Celia, M.A. (2006). J. Fluid Mech. 561, 307-327.
  Nordbotten, J.M., Celia, M.A. & Bachu, S. (2005). Transp. Porous Media 58, 339-360.
  van der Meer, L.G.H. (1993). Energy Convers. Mgmt 34, 959-966.
  Pruess, K. & Spycher, N. (2007). Energy Convers. Mgmt 48, 1761-1767.
  Juanes, R. et al. (2006). Water Resour. Res. 42, W12418.
"""

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81                       # m/s2
MD_TO_M2 = 9.869233e-16        # 1 mD -> m2
SEC_PER_YEAR = 365.25 * 86400.0


class CO2SequestrationF2a:
    """Lumped reservoir + trapping ODE model for CO2 geological storage."""

    def __init__(self, params: dict):
        r = params["reservoir"]
        c = params["co2"]
        b = params["brine"]
        tr = params["trapping"]
        inj = params["injection"]

        # --- reservoir geometry / petrophysics ---
        self.depth = r["depth_m"]["value"]
        self.thickness = r["thickness_m"]["value"]
        self.area_m2 = r["area_km2"]["value"] * 1e6
        self.porosity = r["porosity"]["value"]
        self.k = r["permeability_mD"]["value"] * MD_TO_M2          # m2
        self.P0 = r["P_reservoir_initial_bar"]["value"] * 1e5      # Pa
        self.T_res = r["T_reservoir_K"]["value"]
        self.frac_grad = r["fracture_gradient_bar_per_m"]["value"] * 1e5  # Pa/m
        self.ct = r["compressibility_total_per_Pa"]["value"]       # 1/Pa
        self.skin = r["skin_factor"]["value"]
        self.r_w = r["wellbore_radius_m"]["value"]
        self.r_e = r["drainage_radius_m"]["value"]
        self.Sgr = r["residual_gas_saturation"]["value"]
        self.Swi = r["irreducible_water_saturation"]["value"]

        # --- fluids ---
        self.rho_co2 = c["rho_injection"]["value"]                 # kg/m3
        self.mu_co2 = c["viscosity_injection"]["value"]            # Pa.s
        self.rho_brine = b["rho_brine"]["value"]
        self.C_sat = b["co2_solubility_kg_per_m3"]["value"]        # kg CO2 / m3 brine

        # --- trapping rate constants (per second) ---
        self.k_sol = tr["k_solubility_per_year"]["value"] / SEC_PER_YEAR
        self.k_min = tr["k_mineral_per_year"]["value"] / SEC_PER_YEAR
        # residual-trapping rate from Sgr; characteristic imbibition timescale ~ 10 yr
        self.k_res = (self.Sgr / (1.0 - self.Swi)) / (10.0 * SEC_PER_YEAR)

        # --- injection defaults ---
        self.P_wh_default = inj["P_wellhead_bar"]["value"] * 1e5
        self.target_rate_kg_s = (inj["target_rate_Mt_per_year"]["value"] * 1e9
                                 / SEC_PER_YEAR)
        self.inj_years_default = inj["injection_years"]["value"]

        # --- derived ---
        self.V_pore = self.area_m2 * self.thickness * self.porosity   # m3
        # max free-CO2 brine volume available for dissolution (whole pore brine)
        self.brine_volume = self.V_pore * (1.0 - 0.0)                 # m3 (lumped)
        self.C_cap = self.C_sat * self.brine_volume                    # kg CO2 dissolvable

    # ------------------------------------------------------------------
    # Pressures
    # ------------------------------------------------------------------
    def fracture_pressure_pa(self):
        """Caprock fracture pressure at reservoir depth [Pa]."""
        return self.frac_grad * self.depth

    def bottomhole_pressure_pa(self, P_wh_pa):
        """Bottomhole = wellhead + hydrostatic supercritical CO2 column [Pa]."""
        return P_wh_pa + self.rho_co2 * G * self.depth

    def reservoir_pressure_pa(self, M_mobile, M_residual):
        """
        Lumped tank pressure [Pa]. Free (mobile + residual) CO2 occupies pore
        volume and pressurises the compressible reservoir:
            dP = V_free_CO2 / (ct * V_pore)
        """
        V_free = (M_mobile + M_residual) / self.rho_co2     # m3 of free CO2
        dP = V_free / (self.ct * self.V_pore)
        return self.P0 + dP

    # ------------------------------------------------------------------
    # Injectivity (radial Darcy)
    # ------------------------------------------------------------------
    def injection_rate_kg_s(self, P_wh_pa, P_res_pa):
        """
        Realised mass injection rate [kg/s] from radial Darcy flow, capped so
        bottomhole pressure never exceeds the fracture pressure (injection is
        choked / shut if the formation cannot accept more).
        """
        P_bh = self.bottomhole_pressure_pa(P_wh_pa)
        P_frac = self.fracture_pressure_pa()
        # never allow the driving bottomhole pressure above fracture pressure
        P_bh_eff = min(P_bh, P_frac)
        ln_term = np.log(self.r_e / self.r_w) + self.skin
        dP = max(P_bh_eff - P_res_pa, 0.0)
        Q = (2.0 * np.pi * self.k * self.thickness * dP) / (self.mu_co2 * ln_term)
        m_dot = Q * self.rho_co2
        # cap at target rate (compressor / surface-facility limit)
        return min(m_dot, self.target_rate_kg_s)

    def max_wellhead_pressure_bar(self):
        """Wellhead pressure that brings P_bh exactly to fracture pressure [bar]."""
        P_wh_max = self.fracture_pressure_pa() - self.rho_co2 * G * self.depth
        return P_wh_max / 1e5

    # ------------------------------------------------------------------
    # Plume / saturation geometry
    # ------------------------------------------------------------------
    def avg_saturation(self, M_mobile, M_residual):
        """Average CO2 saturation in the swept pore volume [-]."""
        V_free = (M_mobile + M_residual) / self.rho_co2
        return min(V_free / self.V_pore, 1.0)

    def plume_radius_m(self, M_mobile, M_residual):
        """
        Equivalent plume radius [m] (Nordbotten & Celia 2006 sharp-interface):
            r = sqrt( V_free / (pi * h * phi * (1 - Swi)) )
        """
        V_free = (M_mobile + M_residual) / self.rho_co2
        denom = np.pi * self.thickness * self.porosity * (1.0 - self.Swi)
        return np.sqrt(max(V_free, 0.0) / denom)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, P_wh_pa, t_inj_end):
        M_mobile, M_residual, M_dissolved, M_mineral, _M_inj = y
        M_mobile = max(M_mobile, 0.0)
        M_residual = max(M_residual, 0.0)
        M_dissolved = max(M_dissolved, 0.0)

        # source term: injection only during the active period
        if t <= t_inj_end:
            P_res = self.reservoir_pressure_pa(M_mobile, M_residual)
            m_inj = self.injection_rate_kg_s(P_wh_pa, P_res)
        else:
            m_inj = 0.0

        # trapping fluxes (kg/s)
        f_res = self.k_res * M_mobile                                    # residual snap-off
        sol_factor = max(0.0, 1.0 - M_dissolved / max(self.C_cap, 1e-9))
        f_sol = self.k_sol * M_mobile * sol_factor                       # dissolution
        f_min = self.k_min * M_dissolved                                 # mineralisation

        dM_mobile = m_inj - f_res - f_sol
        dM_residual = f_res
        dM_dissolved = f_sol - f_min
        dM_mineral = f_min
        dM_inj = m_inj                                                   # cumulative injected
        return [dM_mobile, dM_residual, dM_dissolved, dM_mineral, dM_inj]

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------
    def simulate(self, P_wellhead_bar=None, injection_years=None,
                 sim_years=None, n_points=200):
        """
        Integrate the coupled reservoir/trapping ODEs.

        Returns a dict of time-series arrays (time in years) plus scalars.
        """
        P_wh_pa = (self.P_wh_default if P_wellhead_bar is None
                   else P_wellhead_bar * 1e5)
        t_inj_years = self.inj_years_default if injection_years is None else injection_years
        if sim_years is None:
            sim_years = max(t_inj_years * 3.0, 200.0)

        t_inj_end = t_inj_years * SEC_PER_YEAR
        t_end = sim_years * SEC_PER_YEAR
        t_eval = np.linspace(0.0, t_end, n_points)

        y0 = [0.0, 0.0, 0.0, 0.0, 0.0]
        sol = solve_ivp(
            self._rhs, (0.0, t_end), y0,
            t_eval=t_eval, args=(P_wh_pa, t_inj_end),
            method="LSODA", rtol=1e-7, atol=1e-3, max_step=t_inj_end / 50.0,
        )

        M_mobile = np.maximum(sol.y[0], 0.0)
        M_residual = np.maximum(sol.y[1], 0.0)
        M_dissolved = np.maximum(sol.y[2], 0.0)
        M_mineral = np.maximum(sol.y[3], 0.0)
        M_injected = np.maximum(sol.y[4], 0.0)
        M_total = M_mobile + M_residual + M_dissolved + M_mineral

        # derived series
        P_res = np.array([self.reservoir_pressure_pa(mm, mr)
                          for mm, mr in zip(M_mobile, M_residual)])
        S_avg = np.array([self.avg_saturation(mm, mr)
                          for mm, mr in zip(M_mobile, M_residual)])
        r_plume = np.array([self.plume_radius_m(mm, mr)
                            for mm, mr in zip(M_mobile, M_residual)])
        # realised injection-rate series
        m_inj = []
        for ti, mm, mr in zip(sol.t, M_mobile, M_residual):
            if ti <= t_inj_end:
                m_inj.append(self.injection_rate_kg_s(P_wh_pa,
                             self.reservoir_pressure_pa(mm, mr)))
            else:
                m_inj.append(0.0)
        m_inj = np.array(m_inj)

        eps = 1e-30
        trap_frac = {
            "structural": M_mobile / (M_total + eps),
            "residual": M_residual / (M_total + eps),
            "solubility": M_dissolved / (M_total + eps),
            "mineral": M_mineral / (M_total + eps),
        }
        return {
            "t_years": sol.t / SEC_PER_YEAR,
            "M_mobile_t": M_mobile,
            "M_residual_t": M_residual,
            "M_dissolved_t": M_dissolved,
            "M_mineral_t": M_mineral,
            "M_total_t": M_total,
            "injected_cumulative_t": M_injected,
            "reservoir_pressure_bar": P_res / 1e5,
            "fracture_pressure_bar": self.fracture_pressure_pa() / 1e5,
            "bottomhole_pressure_bar": self.bottomhole_pressure_pa(P_wh_pa) / 1e5,
            "saturation_avg": S_avg,
            "plume_radius_m": r_plume,
            "injection_rate_kg_s": m_inj,
            "trapping_fraction": trap_frac,
            "success": sol.success,
        }
