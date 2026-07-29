"""
EC082 -- Ice Thermal Storage -- F2a Stefan-Problem Lumped Enthalpy Model

Physics-lumped (0D) latent thermal-energy-storage model of an ice-on-coil
tank.  The state is the total enthalpy H [J] of the storage water relative to
liquid water at the fusion temperature T_f = 0 C.  The enthalpy method (Voller
& Cross 1981) recovers temperature and ice fraction from H without tracking the
freeze/melt front explicitly, so the discontinuous latent plateau at 0 C is
handled cleanly by a single ODE through scipy.integrate.solve_ivp.

State variable:
    H  [J]  -- enthalpy of the water mass relative to liquid at T_f.
              H = 0           => all liquid at exactly T_f (ice fraction 0)
              H = -L          => all ice   at exactly T_f (ice fraction 1)
              where L = m_water * h_fusion is the total latent capacity.

Enthalpy <-> (Temperature, ice fraction) closure (Alexiades & Solomon 1993):
    H >= 0                 : liquid sensible region, T = T_f + H/(m*cp_water), f_ice = 0
    -L < H < 0             : mushy / phase change, T = T_f, f_ice = -H/L
    H <= -L                : solid sub-cooled region, T = T_f + (H+L)/(m*cp_ice), f_ice = 1

Stefan moving-front conduction resistance.  As ice builds on the coil, the
growing ice layer adds a conduction resistance in series with the coil-water
film.  For a cylindrical ice annulus of outer radius r_o around a coil of
radius r_i (Alexiades & Solomon 1993, cylindrical Stefan problem):

    R_ice(f_ice) = ln(r_o / r_i) / (2*pi*k_ice*Lcoil)

with the ice volume V_ice = f_ice * m_water / rho_ice distributed over the coil
length, giving r_o(f_ice).  The effective conductance UA therefore *decreases*
as ice thickness grows during charging -- the central nonlinearity of ice-on-coil
storage.  During discharge (melting from the outside) the same annulus resistance
is recovered as the ice shrinks.

Energy balance ODE (first law, lumped):

    dH/dt = -Q_coil(H) - Q_loss(T_amb)

    Q_coil = UA_eff(f_ice) * (T_water - T_brine)     [W, >0 removes heat -> freezes]
    Q_loss = UA_ambient   * (T_water - T_amb)         [W, ambient ingress melts ice]

Sign convention: charging supplies cold brine (T_brine < T_f) so Q_coil > 0 and
H decreases (more ice).  Discharging supplies warm brine (T_brine > T_f) so
Q_coil < 0 and H increases (ice melts, delivering cooling).

Conservation guarantees enforced by construction:
  * Latent + sensible energy are a single conserved quantity H; the cooling
    energy delivered equals the enthalpy change minus ambient losses.
  * ice_fraction is clamped to [0, 1] by the enthalpy closure.
  * During phase change the temperature is pinned exactly at T_f = 0 C.

References:
    Voller, V.R. & Cross, M. (1981). Accurate solutions of moving boundary
        problems using the enthalpy method. Int. J. Heat Mass Transfer 24(3),
        545-556.
    Alexiades, V. & Solomon, A.D. (1993). Mathematical Modeling of Melting and
        Freezing Processes. Hemisphere.
    ASHRAE (2020). Handbook -- HVAC Systems and Equipment, ch.51 Thermal Storage.
    Dincer, I. & Rosen, M.A. (2021). Thermal Energy Storage, 3rd ed., Wiley.
"""

import numpy as np
from scipy.integrate import solve_ivp


class IceTES_F2a:
    """Ice thermal storage -- lumped Stefan/enthalpy-method dynamic model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.m_water = u["mass_water_kg"]["value"]              # kg
        self.T_f     = u["T_phase_change"]["value"]             # degC
        self.h_fus   = u["h_fusion"]["value"] * 1000.0          # kJ/kg -> J/kg
        self.cp_w    = u["cp_water"]["value"]                   # J/(kg.K)
        self.cp_i    = u["cp_ice"]["value"]                     # J/(kg.K)
        self.k_ice   = u["k_ice"]["value"]                      # W/(m.K)
        self.rho_ice = u["rho_ice"]["value"]                    # kg/m3
        self.A_coil  = u["coil_area_m2"]["value"]               # m2
        self.r_i     = u["coil_radius_m"]["value"]              # m
        self.h_cw    = u["h_coil_water"]["value"]               # W/(m2.K)
        self.h_br    = u["h_brine_film"]["value"]               # W/(m2.K)
        self.UA_amb  = u["UA_ambient"]["value"]                 # W/K
        self.eta_rt  = u["round_trip_efficiency"]["value"]      # -

        # Total latent capacity [J]: H ranges over [-L, +inf)
        self.L = self.m_water * self.h_fus

        # Coil geometry: total tube length implied by area = 2*pi*r_i*Lcoil
        self.L_coil = self.A_coil / (2.0 * np.pi * self.r_i)   # m

        # Clean-coil (no ice) conductance: two films in series over coil area.
        self.UA_clean = 1.0 / (1.0 / (self.h_br * self.A_coil)
                               + 1.0 / (self.h_cw * self.A_coil))  # W/K

    # ------------------------------------------------------------------
    # Enthalpy <-> (T, ice fraction) closure  (Alexiades & Solomon 1993)
    # ------------------------------------------------------------------
    def ice_fraction(self, H):
        """Frozen mass fraction in [0, 1] from enthalpy H [J]."""
        H = np.asarray(H, dtype=float)
        # -H/L over the mushy band, clamped to [0,1] for the sensible regions.
        return np.clip(-H / self.L, 0.0, 1.0)

    def temperature(self, H):
        """Storage temperature [degC] from enthalpy H [J] (pinned at T_f in mush)."""
        H = np.asarray(H, dtype=float)
        T = np.full_like(H, self.T_f, dtype=float)
        # liquid sensible: H > 0
        liq = H > 0.0
        T = np.where(liq, self.T_f + H / (self.m_water * self.cp_w), T)
        # solid sub-cooled: H < -L
        sol = H < -self.L
        T = np.where(sol, self.T_f + (H + self.L) / (self.m_water * self.cp_i), T)
        return T

    def state_of_charge(self, H):
        """SOC = stored cooling fraction = ice fraction (0 empty .. 1 full)."""
        return self.ice_fraction(H)

    # ------------------------------------------------------------------
    # Stefan moving-front conduction resistance through the ice annulus
    # ------------------------------------------------------------------
    def ice_outer_radius(self, f_ice):
        """Outer radius [m] of the cylindrical ice annulus around the coil."""
        f_ice = float(np.clip(f_ice, 0.0, 1.0))
        V_ice = f_ice * self.m_water / self.rho_ice            # m3 of ice
        # V_ice = pi*(r_o^2 - r_i^2)*L_coil  ->  solve for r_o
        r_o = np.sqrt(self.r_i**2 + V_ice / (np.pi * self.L_coil))
        return r_o

    def UA_effective(self, f_ice):
        """Effective coil conductance [W/K] including the growing ice layer.

        Series resistance: clean-coil films + cylindrical ice-annulus
        conduction R_ice = ln(r_o/r_i)/(2*pi*k_ice*L_coil).
        UA *decreases* monotonically as ice thickness grows (Stefan effect).
        """
        f_ice = float(np.clip(f_ice, 0.0, 1.0))
        if f_ice <= 1e-9:
            return self.UA_clean
        r_o = self.ice_outer_radius(f_ice)
        R_ice = np.log(r_o / self.r_i) / (2.0 * np.pi * self.k_ice * self.L_coil)
        R_clean = 1.0 / self.UA_clean
        return 1.0 / (R_clean + R_ice)

    # ------------------------------------------------------------------
    # Heat-flow terms
    # ------------------------------------------------------------------
    def q_coil(self, H, T_brine):
        """Heat removed by the coil [W]. >0 freezes (charging), <0 melts."""
        f_ice = self.ice_fraction(H)
        T_w = self.temperature(H)
        UA = self.UA_effective(f_ice)
        flow = UA * (T_w - T_brine)
        # Round-trip / parasitic effectiveness applied on the freezing path only.
        if flow > 0:
            flow *= self.eta_rt
        return flow

    def q_loss(self, H, T_amb):
        """Ambient heat ingress [W] (positive when T_amb > T_water -> melts ice)."""
        T_w = self.temperature(H)
        return self.UA_amb * (T_w - T_amb)

    # ------------------------------------------------------------------
    # ODE right-hand side:  dH/dt = -Q_coil - Q_loss
    # ------------------------------------------------------------------
    def _rhs(self, t, y, brine_fn, amb_fn):
        H = y[0]
        # Clamp enthalpy to physical band to avoid integrator overshoot driving
        # ice_fraction outside [0,1]; reflect by zeroing the offending flux.
        T_brine = brine_fn(t)
        T_amb = amb_fn(t)
        Qc = self.q_coil(H, T_brine)
        Ql = self.q_loss(H, T_amb)
        dH = -Qc - Ql
        # Hard physical bounds on H in [-L, H_liquid_cap]. Block flux that would
        # push past full ice or fully-melted-and-warmed beyond ambient drive.
        if H <= -self.L and dH < 0:
            dH = 0.0       # already full ice, cannot freeze further
        return [dH]

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def simulate(self, T_brine, T_amb=None, ice_fraction0=0.0,
                 dt=60.0, duration_s=3600.0):
        """Integrate the lumped enthalpy ODE over [0, duration_s].

        Parameters
        ----------
        T_brine : float or callable(t)->degC
            Coil supply (brine) temperature. < T_f charges (builds ice),
            > T_f discharges (melts ice / delivers cooling).
        T_amb : float or callable(t)->degC, optional
            Ambient temperature for shell losses (default 20 C).
        ice_fraction0 : float
            Initial ice fraction in [0, 1].
        dt : float
            Output sampling interval [s].
        duration_s : float
            Total simulated time [s].

        Returns a dict of time-series arrays.
        """
        if T_amb is None:
            T_amb = 20.0
        brine_fn = T_brine if callable(T_brine) else (lambda t: float(T_brine))
        amb_fn = T_amb if callable(T_amb) else (lambda t: float(T_amb))

        f0 = float(np.clip(ice_fraction0, 0.0, 1.0))
        H0 = -f0 * self.L                      # start in the mushy band

        t_eval = np.arange(0.0, duration_s + 0.5 * dt, dt)
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [H0],
            t_eval=t_eval, args=(brine_fn, amb_fn),
            method="LSODA", rtol=1e-7, atol=1e-2, max_step=dt,
        )

        H = sol.y[0]
        # Clamp enthalpy to the physical band for reporting.
        H = np.clip(H, -self.L, None)

        f_ice = self.ice_fraction(H)
        T_w = self.temperature(H)
        UA = np.array([self.UA_effective(f) for f in f_ice])
        T_br = np.array([brine_fn(t) for t in sol.t])
        T_am = np.array([amb_fn(t) for t in sol.t])
        q_coil = np.array([self.q_coil(h, tb) for h, tb in zip(H, T_br)])
        q_loss = np.array([self.q_loss(h, ta) for h, ta in zip(H, T_am)])

        # Cooling energy delivered to the load = positive enthalpy gain via coil
        # during discharge (melting). Integrate -q_coil when q_coil<0.
        cooling_W = np.where(q_coil < 0, -q_coil, 0.0)
        charge_W = np.where(q_coil > 0, q_coil, 0.0)

        return {
            "t": sol.t,
            "enthalpy_J": H,
            "ice_fraction": f_ice,
            "soc": f_ice,
            "temperature_C": T_w,
            "UA_eff_W_per_K": UA,
            "ice_radius_m": np.array([self.ice_outer_radius(f) for f in f_ice]),
            "q_coil_W": q_coil,
            "q_loss_W": q_loss,
            "cooling_power_W": cooling_W,
            "charge_power_W": charge_W,
            "energy_stored_kwh": f_ice * (self.L / 3.6e6),
            "T_brine_C": T_br,
            "T_amb_C": T_am,
            "success": sol.success,
        }
