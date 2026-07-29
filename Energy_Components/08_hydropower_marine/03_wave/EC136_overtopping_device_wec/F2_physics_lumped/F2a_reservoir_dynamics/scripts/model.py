"""
EC136 -- Overtopping Device WEC (Wave Dragon) -- F2a Physics-Lumped Reservoir Dynamics

Physics-lumped (0D) dynamic model of an overtopping wave energy converter.

Energy pathway:
    incident waves -> run-up a curved ramp -> OVERTOP the crest (freeboard Rc)
    -> fill an ELEVATED reservoir held above mean sea level
    -> water drains through low-head Kaplan turbines back to the sea
    -> electricity.

Two coupled first-principles relations + one lumped ODE:

1. Overtopping discharge (Van der Meer & Janssen 1995 / EurOtop 2018 mean form):
       q* = q / sqrt(g * Hm0^3)
       q* = a * exp( -b * Rc / Hm0 )                          (dimensionless)
   so the volumetric overtopping rate per metre crest width is
       q [m^3/s/m] = a * sqrt(g * Hm0^3) * exp( -b * Rc_eff / Hm0 )
   where the EFFECTIVE freeboard Rc_eff = crest_freeboard + reservoir water level
   (instantaneous level above the crest reduces the relative freeboard the wave
   must clear; as the reservoir fills, effective freeboard rises and overtopping
   falls -- the physical self-limiting feedback). Total inflow:
       Q_in [m^3/s] = q * W
   q increases strongly with wave height Hm0 (monotone) -- enforced in tests.

2. Low-head Kaplan turbine flow (orifice / energy law) and power:
       Q_out = K * sqrt(2 * g * H)              H = head = freeboard + level
       P_hyd = rho * g * Q_out * H               (gross hydraulic power)
       P_elec = P_hyd * eta_turbine * eta_generator       (P = rho g Q H eta)
   Power is zero when the reservoir empties (H <= crest, level <= 0).

3. Lumped reservoir level ODE (mass conservation, free-surface area A_res):
       A_res * d(level)/dt = Q_in(level, Hm0, Tz) - Q_out(level)
   integrated with scipy.integrate.solve_ivp (RK45). Reservoir level is bounded
   in [0, depth_max]; overflow above depth_max spills back to sea (capped inflow).

Outputs: time series of reservoir level, inflow, turbine flow, head, electrical
power; plus mean power, captured wave power, and overall wave-to-wire efficiency.

Conservation:
    Volume:  integral(Q_in - Q_out) dt = A_res * (level_end - level_0)  (closed)
    Energy:  P_elec <= P_hyd <= rho*g*Q_out*H ; eta_overall in (0,1).

Hard-coded properties (cited in parameters.json):
    rho = 1025 kg/m3 (seawater, UNESCO 1981 EOS / ITTC)
    g   = 9.81 m/s2  (standard gravity, ISO 80000-3)

References:
    Van der Meer, J.W. & Janssen, J.P.F.M. (1995). Wave run-up and overtopping
        at dikes. ASCE, Wave Forces on Inclined and Vertical Wall Structures.
    Kofoed, J.P. (2002). Wave Overtopping of Marine Structures -- Utilization of
        Wave Energy. PhD Thesis, Aalborg University.
    Kofoed, J.P., Frigaard, P., Friis-Madsen, E. & Sorensen, H.C. (2006).
        Prototype testing of the wave energy converter Wave Dragon.
        Coastal Engineering, 53, 859-867.
    Tedd, J. & Kofoed, J.P. (2009). Measurements of overtopping flow time
        series on the Wave Dragon. Renewable Energy, 34, 711-717.
    EurOtop (2018). Manual on wave overtopping of sea defences, 2nd ed.
"""

import numpy as np
from scipy.integrate import solve_ivp


class OvertoppingWEC_F2a:
    """Overtopping WEC -- lumped reservoir-level ODE with Van der Meer overtopping
    inflow and low-head Kaplan turbine outflow."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.W          = u["ramp_width_m"]["value"]          # m
        self.A_res      = u["A_reservoir_m2"]["value"]        # m2
        self.Rc         = u["crest_freeboard_m"]["value"]     # m (crest above MSL)
        self.depth_max  = u["reservoir_depth_max_m"]["value"] # m (storage cap)
        self.n_turb     = u["n_turbines"]["value"]            # -
        self.K          = u["turbine_K_m2"]["value"]          # m2 (per array)
        self.eta_turb   = u["eta_turbine"]["value"]           # -
        self.eta_gen    = u["eta_generator"]["value"]         # -
        self.a          = u["vdm_a"]["value"]                 # -
        self.b          = u["vdm_b"]["value"]                 # -
        self.rho        = u["rho_water"]["value"]             # kg/m3
        self.g          = u["g"]["value"]                     # m/s2

    # ------------------------------------------------------------------ physics
    def overtopping_q_per_m(self, Hm0, level):
        """Van der Meer/EurOtop overtopping discharge per metre crest [m^3/s/m].

        q = a * sqrt(g * Hm0^3) * exp(-b * Rc_eff / Hm0)
        Rc_eff = crest freeboard + instantaneous reservoir level above crest.
        """
        Hm0 = max(float(Hm0), 1e-6)
        Rc_eff = self.Rc + max(float(level), 0.0)
        q = self.a * np.sqrt(self.g * Hm0 ** 3) * np.exp(-self.b * Rc_eff / Hm0)
        return max(q, 0.0)

    @staticmethod
    def _smoothstep(x):
        """C1-continuous smootherstep on [0,1] (0 below, 1 above) -- used to taper
        flows smoothly at the reservoir bounds so the ODE RHS stays continuous
        (avoids RK45 step-size chatter at hard min/max-level boundaries)."""
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)

    def inflow_Q(self, Hm0, level):
        """Total overtopping inflow into the reservoir [m^3/s].
        Tapers smoothly to zero over the top 10% of storage (spillage to sea
        once the reservoir is full)."""
        q = self.overtopping_q_per_m(Hm0, level) * self.W
        # full-reservoir taper: 1 below 90% depth, 0 at depth_max
        band = 0.1 * self.depth_max
        full_factor = 1.0 - self._smoothstep((level - (self.depth_max - band)) / band)
        return q * full_factor

    def head(self, level):
        """Hydraulic head driving the turbines [m] = crest freeboard + reservoir
        level above MSL (water surface in elevated reservoir relative to sea)."""
        return self.Rc + max(float(level), 0.0)

    def turbine_Q(self, level):
        """Turbine discharge [m^3/s] via orifice/energy law Q = K*sqrt(2 g H).
        Tapers smoothly to zero as the reservoir empties over a thin near-empty
        band (no stored water above crest -> turbines shut), keeping the ODE
        RHS continuous at the lower bound."""
        if level <= 0.0:
            return 0.0
        H = self.head(level)
        Q = self.K * np.sqrt(2.0 * self.g * H)
        # near-empty taper: 0 at level=0, 1 once level exceeds a thin empty band
        band = 0.03 * self.depth_max          # 0.09 m for depth_max=3 m
        empty_factor = self._smoothstep(level / band)
        return Q * empty_factor

    def turbine_power_W(self, level):
        """Electrical power P = rho * g * Q * H * eta_turbine * eta_generator [W]."""
        if level <= 0.0:
            return 0.0
        Q = self.turbine_Q(level)
        H = self.head(level)
        return self.rho * self.g * Q * H * self.eta_turb * self.eta_gen

    def incident_wave_power_W(self, Hm0, Tz):
        """Incident wave power resource on the device crest width [W].
        J = rho g^2 Hm0^2 Te / (64 pi) per metre, with Te ~ 1.2*Tz (JONSWAP).
        Used only for the overall wave-to-wire efficiency denominator."""
        Te = 1.2 * float(Tz)
        J = (self.rho * self.g ** 2 * float(Hm0) ** 2 * Te) / (64.0 * np.pi)
        return J * self.W

    # ------------------------------------------------------------------ ODE
    def _rhs(self, t, y, Hm0_fn, Tz):
        # Smooth, continuous RHS: inflow tapers to 0 at depth_max, turbine
        # outflow tapers to 0 at empty -> level naturally stays in [0, depth_max]
        # without any discontinuous clamp (which would stall the RK45 stepper).
        level = min(max(y[0], 0.0), self.depth_max)
        Hm0 = Hm0_fn(t)
        Qin = self.inflow_Q(Hm0, level)
        Qout = self.turbine_Q(level)
        return [(Qin - Qout) / self.A_res]

    def simulate(self, Hm0, Tz, level0=0.5, dt=5.0, duration_s=1800.0):
        """Integrate the lumped reservoir ODE.

        Hm0 : significant wave height [m] (scalar or callable Hm0(t) for sea-state
              transients).
        Tz  : mean wave period [s].
        level0 : initial reservoir level above crest [m].
        Returns dict of time series + scalar summaries.
        """
        Hm0_fn = Hm0 if callable(Hm0) else (lambda t, h=float(Hm0): h)
        level0 = min(max(float(level0), 0.0), self.depth_max)
        t_eval = np.arange(0.0, duration_s + dt, dt)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [level0],
            t_eval=t_eval, args=(Hm0_fn, Tz),
            method="RK45", rtol=1e-6, atol=1e-8,
        )

        t = sol.t
        level = np.clip(sol.y[0], 0.0, self.depth_max)

        Qin   = np.array([self.inflow_Q(Hm0_fn(ti), lv) for ti, lv in zip(t, level)])
        Qout  = np.array([self.turbine_Q(lv) for lv in level])
        Hd    = np.array([self.head(lv) for lv in level])
        P_el  = np.array([self.turbine_power_W(lv) for lv in level])
        P_hyd = self.rho * self.g * Qout * Hd

        # Mean values (trapezoidal over the record).
        P_mean = float(np.trapezoid(P_el, t) / (t[-1] - t[0])) if t[-1] > t[0] else float(P_el.mean())
        Qin_mean = float(np.trapezoid(Qin, t) / (t[-1] - t[0])) if t[-1] > t[0] else float(Qin.mean())

        # Incident wave resource (time-averaged) and wave-to-wire efficiency.
        Pinc = np.array([self.incident_wave_power_W(Hm0_fn(ti), Tz) for ti in t])
        Pinc_mean = float(Pinc.mean())
        eta_overall = P_mean / Pinc_mean if Pinc_mean > 0 else 0.0

        # Mass-conservation residual (volume balance closure).
        vol_in  = float(np.trapezoid(Qin, t))
        vol_out = float(np.trapezoid(Qout, t))
        dvol_store = self.A_res * (level[-1] - level[0])
        mass_residual = vol_in - vol_out - dvol_store  # ~0 if conserved

        return {
            "t": t,
            "level": level,                 # m above crest
            "Q_in": Qin,                    # m3/s
            "Q_out": Qout,                  # m3/s
            "head": Hd,                     # m
            "power_elec_W": P_el,           # W
            "power_hyd_W": P_hyd,           # W
            "P_mean_W": P_mean,             # W
            "P_mean_kW": P_mean / 1e3,      # kW
            "Q_in_mean": Qin_mean,          # m3/s
            "P_incident_mean_W": Pinc_mean, # W
            "eta_overall": eta_overall,     # -
            "mass_residual_m3": mass_residual,
            "success": bool(sol.success),
        }
