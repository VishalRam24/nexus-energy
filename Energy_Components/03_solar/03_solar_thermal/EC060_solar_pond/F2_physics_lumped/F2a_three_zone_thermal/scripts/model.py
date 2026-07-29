"""
EC060 -- Solar Pond (Salinity-Gradient) -- F2a Three-Zone Lumped Energy Balance

Physics-lumped (0D-per-zone) dynamic model of a non-convecting salt-gradient
solar pond. Three zones are resolved:

    UCZ  Upper Convective Zone   (surface mixed layer, ~ambient)
    NCZ  Non-Convecting Zone     (salinity gradient; acts as transparent insulation)
    LCZ  Lower Convective Zone   (storage; the useful hot reservoir, 70-90 C)

KEY PHYSICS
-----------
1. Solar attenuation with depth (Beer-Lambert / Bouguer law).
   Sunlight is absorbed exponentially as it penetrates the brine. The fraction
   of surface-incident radiation that survives to be deposited in the LCZ is

       f_LCZ = tau_surface * exp(-mu * (h_ucz + h_ncz))

   Because the NCZ suppresses convection, the radiation that reaches the LCZ is
   trapped as heat (the "transparent insulation" greenhouse effect of the pond).
   (Rabl & Nielsen 1975; Hull 1980; Tabor 1981.)

2. Lumped energy balance ODE for LCZ storage temperature:

       (rho*cp*V_lcz) dT_lcz/dt = Q_solar - Q_top_path - Q_ground - Q_extract

   where
     Q_solar     = A * G * f_LCZ                 absorbed solar reaching LCZ
     Q_top_path  = A * U_ncz * (T_lcz - T_ucz)   conductive loss up through NCZ
     Q_ground    = A * (k_ground/L_ground)*(T_lcz - T_ground)   downward ground loss
     Q_extract   = user heat withdrawal (heat exchanger in LCZ)

   The NCZ conductive series resistance is
     U_ncz = 1 / (h_ncz/k_water)      [W/m2K]
   so the NCZ thickness directly sets the insulating quality of the pond.

3. UCZ tracks ambient closely (thin, wind-mixed). It is modelled with its own
   small lumped capacitance and a top loss to ambient
     Q_top = A * U_top * (T_ucz - T_amb)
   plus the upward conductive gain from the LCZ through the NCZ. This couples
   the two ODEs.

The two coupled ODEs (T_lcz, T_ucz) are integrated with scipy.integrate.solve_ivp.

CONSERVATION
------------
Energy is conserved: the stored-energy change of each zone equals the time
integral of (in - out) fluxes for that zone (verified in the test suite).
At night G = 0 so Q_solar = 0 and the pond can only cool / be discharged.

References:
    Tabor, H. (1981). Solar ponds. Solar Energy 27(3), 181-194.
    Hull, J.R. (1980). Computer simulation of solar pond thermal behavior.
        Solar Energy 25, 33-40.
    Rabl, A. & Nielsen, C.E. (1975). Solar ponds for space heating.
        Solar Energy 17, 1-12.
    Wang, Y.F. & Akbarzadeh, A. (1982). A parametric study on solar ponds.
        Solar Energy 30(6), 555-562.
    Duffie, J.A. & Beckman, W.A. (2013). Solar Engineering of Thermal
        Processes, 4th ed., ch.9.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SolarPondF2a:
    """Three-zone lumped energy-balance solar pond model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A_pond"]["value"]
        self.h_ucz = u["h_ucz"]["value"]
        self.h_ncz = u["h_ncz"]["value"]
        self.h_lcz = u["h_lcz"]["value"]
        self.rho = u["rho_brine"]["value"]
        self.cp = u["cp_brine"]["value"]
        self.k_water = u["k_water"]["value"]
        self.tau_surface = u["tau_surface"]["value"]
        self.mu = u["mu_extinction"]["value"]
        self.T_ground = u["T_ground"]["value"]
        self.k_ground = u["k_ground"]["value"]
        self.L_ground = u["L_ground"]["value"]
        self.U_top = u["U_top"]["value"]

        # Derived geometric / thermal quantities
        self.V_lcz = self.A * self.h_lcz                 # m3
        self.V_ucz = self.A * self.h_ucz                 # m3
        self.C_lcz = self.rho * self.cp * self.V_lcz     # J/K  storage heat capacity
        # UCZ uses near-pure-water properties (dilute surface layer)
        self.C_ucz = 1000.0 * 4180.0 * self.V_ucz        # J/K
        # NCZ conductive (series) loss coefficient referenced to pond area
        self.U_ncz = self.k_water / self.h_ncz           # W/(m2.K)
        # Downward ground loss coefficient
        self.U_ground = self.k_ground / self.L_ground    # W/(m2.K)

    # ------------------------------------------------------------------ optics
    def solar_fraction_to_lcz(self):
        """Beer-Lambert fraction of surface-incident radiation reaching the LCZ.

        f = tau_surface * exp(-mu * (h_ucz + h_ncz))
        """
        depth = self.h_ucz + self.h_ncz
        return self.tau_surface * np.exp(-self.mu * depth)

    def transmitted_fraction(self, depth_m):
        """Beer-Lambert transmitted fraction at an arbitrary depth (incl. surface)."""
        return self.tau_surface * np.exp(-self.mu * np.asarray(depth_m, dtype=float))

    def Q_solar_lcz(self, G):
        """Absorbed solar power deposited in LCZ [W] for surface irradiance G [W/m2]."""
        G = np.asarray(G, dtype=float)
        return self.A * G * self.solar_fraction_to_lcz()

    # ----------------------------------------------------------------- losses
    def Q_ncz_path(self, T_lcz, T_ucz):
        """Conductive heat flux up from LCZ to UCZ through the NCZ [W]."""
        return self.A * self.U_ncz * (T_lcz - T_ucz)

    def Q_ground(self, T_lcz):
        """Downward conductive loss to the deep ground sink [W]."""
        return self.A * self.U_ground * (T_lcz - self.T_ground)

    def Q_top(self, T_ucz, T_amb):
        """Top surface loss UCZ -> ambient [W] (convection+evap+radiation lumped)."""
        return self.A * self.U_top * (T_ucz - T_amb)

    # -------------------------------------------------------------------- ODE
    def _rhs(self, t, y, G_fn, Tamb_fn, Qext_fn):
        T_lcz, T_ucz = y
        G = float(G_fn(t))
        T_amb = float(Tamb_fn(t))
        Q_ext = float(Qext_fn(t))

        q_solar = self.A * G * self.solar_fraction_to_lcz()
        q_ncz = self.A * self.U_ncz * (T_lcz - T_ucz)
        q_ground = self.A * self.U_ground * (T_lcz - self.T_ground)
        q_top = self.A * self.U_top * (T_ucz - T_amb)

        # LCZ storage balance
        dT_lcz = (q_solar - q_ncz - q_ground - Q_ext) / self.C_lcz
        # UCZ balance: gains conductive heat from LCZ, loses to ambient.
        # A small fraction of solar is also absorbed in the surface layer, but
        # that is dumped to ambient and does not heat storage; we omit it as it
        # has negligible effect on the LCZ (it leaves through the top quickly).
        dT_ucz = (q_ncz - q_top) / self.C_ucz
        return [dT_lcz, dT_ucz]

    @staticmethod
    def _as_callable(x, default=None):
        """Coerce a scalar / array-with-time / callable into a function of t."""
        if callable(x):
            return x
        if x is None:
            x = default
        val = float(x)
        return lambda t: val

    def simulate(self, G, T_lcz_init=20.0, T_ucz_init=None, T_amb=20.0,
                 Q_extract_W=0.0, duration_days=10.0, dt_hours=1.0,
                 diurnal=False, G_peak=None):
        """Integrate the coupled two-zone ODE system.

        Parameters
        ----------
        G : float or callable(t_seconds)->W/m2
            Surface solar irradiance. If `diurnal=True` and G is scalar, a
            half-sine daytime profile peaking at `G_peak` (default = G) with
            zero output at night is synthesised, which exercises Q_solar=0 at
            night.
        T_lcz_init, T_ucz_init : float [degC]   initial zone temperatures
            (T_ucz_init defaults to T_amb).
        T_amb : float or callable(t)->degC      ambient temperature.
        Q_extract_W : float or callable(t)->W   heat withdrawal from the LCZ.
        duration_days : float                    simulated horizon.
        dt_hours : float                         output sampling interval.

        Returns dict with time series (SI/degC) and instantaneous power terms.
        """
        if T_ucz_init is None:
            T_ucz_init = T_amb if not callable(T_amb) else float(T_amb(0.0))

        total_s = duration_days * 86400.0
        n = max(2, int(round(total_s / (dt_hours * 3600.0))) + 1)
        t_eval = np.linspace(0.0, total_s, n)

        if diurnal and not callable(G):
            Gp = G_peak if G_peak is not None else G

            def G_fn(t):
                # Half-sine daytime (06:00-18:00), zero at night
                frac = (t % 86400.0) / 86400.0
                if 0.25 <= frac <= 0.75:
                    return Gp * np.sin(np.pi * (frac - 0.25) / 0.5)
                return 0.0
        else:
            G_fn = self._as_callable(G)

        Tamb_fn = self._as_callable(T_amb)
        Qext_fn = self._as_callable(Q_extract_W, default=0.0)

        sol = solve_ivp(
            self._rhs, (0.0, total_s), [T_lcz_init, T_ucz_init],
            t_eval=t_eval, args=(G_fn, Tamb_fn, Qext_fn),
            method="LSODA", rtol=1e-6, atol=1e-6, max_step=3600.0,
        )

        T_lcz = sol.y[0]
        T_ucz = sol.y[1]
        t = sol.t

        G_arr = np.array([G_fn(ti) for ti in t])
        Tamb_arr = np.array([Tamb_fn(ti) for ti in t])
        Qext_arr = np.array([Qext_fn(ti) for ti in t])

        q_solar = self.A * G_arr * self.solar_fraction_to_lcz()
        q_ncz = self.A * self.U_ncz * (T_lcz - T_ucz)
        q_ground = self.A * self.U_ground * (T_lcz - self.T_ground)
        q_top = self.A * self.U_top * (T_ucz - Tamb_arr)

        return {
            "t": t,
            "t_days": t / 86400.0,
            "T_lcz": T_lcz,            # storage temperature (degC)
            "T_ucz": T_ucz,            # surface zone temperature (degC)
            "T_amb": Tamb_arr,
            "G": G_arr,
            "Q_solar_W": q_solar,      # solar reaching LCZ
            "Q_ncz_W": q_ncz,          # loss up through NCZ
            "Q_ground_W": q_ground,    # loss to ground
            "Q_top_W": q_top,          # surface loss
            "Q_extract_W": Qext_arr,   # extracted heat
            "f_lcz": self.solar_fraction_to_lcz(),
            "success": sol.success,
        }
