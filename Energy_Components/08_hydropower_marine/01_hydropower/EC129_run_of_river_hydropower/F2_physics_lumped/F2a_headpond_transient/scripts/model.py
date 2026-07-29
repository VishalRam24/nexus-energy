"""
EC129 -- Run-of-River Hydropower -- F2a Physics-Lumped Headpond Transient

Physics-lumped (0D) first-principles model of a low-head run-of-river plant
with an explicit forebay/headpond level ODE, Darcy-Weisbach penstock head loss,
and a Kaplan/Francis hill-chart turbine-efficiency curve.

Hydraulic power balance (steady, at each instant):
    P_hydraulic = rho * g * Q * H_net                       [W]
    P_elec      = eta_turbine(Q) * eta_gen * P_hydraulic    [W]

Net head after penstock losses (Darcy-Weisbach + minor losses):
    v      = Q / A_penstock                 (penstock velocity)
    h_loss = (f * L / D + sum_K) * v^2 / (2 g)
    H_net  = H_gross - h_loss               (head-loss strictly reduces net head)

Turbine efficiency (hill chart, parabolic in flow ratio q = Q / Q_design):
    eta_t(q) = eta_peak * (1 - k * (q - 1)^2),  0 < eta_t < 1
    (zero outside [q_min, q_max] gate limits)

Forebay / headpond mass-balance ODE (lumped reservoir, minimal pondage):
    A_pond * dz/dt = Q_inflow(t) - Q_turbine - Q_spill(z)
    H_gross        = z   (forebay level above tailwater)
    Q_spill(z)     = C_w * max(z - z_max, 0)^1.5     (broad-crested weir)
Integrated with scipy.integrate.solve_ivp.

Energy conservation: the hydraulic head extracted at the turbine equals the
loss of gross potential head minus penstock friction loss; the level ODE is a
strict volumetric mass balance (inflow - outflow = storage rate).

Physical constants are hard-coded with citations:
    rho = 1000 kg/m^3  (fresh water ~10 C; CRC Handbook; Gulliver & Arndt 1991 App. A)
    g   = 9.81 m/s^2   (standard gravity; IEC 60041)

References:
    Gulliver, J.S. & Arndt, R.E.A. (1991), "Hydropower Engineering Handbook",
        McGraw-Hill -- Ch.3 (head & flow), Ch.4 (penstock hydraulics, Darcy-Weisbach),
        Ch.5 (turbine performance / hill charts).
    Penche, C. (1998), "Layman's Guidebook on How to Develop a Small Hydro Site",
        European Commission DG-XVII.
    White, F.M. (2011), "Fluid Mechanics", 7th ed., McGraw-Hill (Darcy-Weisbach).
"""

import numpy as np
from scipy.integrate import solve_ivp


class RunOfRiverF2a:
    """Run-of-river hydropower -- physics-lumped headpond transient model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_design = u["H_design"]["value"]            # m (gross design head)
        self.Q_design = u["Q_design"]["value"]            # m3/s
        self.P_rated = u["P_rated"]["value"]              # kW
        self.eta_peak = u["eta_peak"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.k = u["k_efficiency"]["value"]
        self.q_min = u["q_min"]["value"]
        self.q_max = u["q_max"]["value"]
        self.L = u["penstock_length"]["value"]            # m
        self.D = u["penstock_diameter"]["value"]          # m
        self.f = u["friction_factor"]["value"]
        self.K_minor = u["minor_loss_K"]["value"]
        self.A_pond = u["pond_area"]["value"]             # m2
        self.z0 = u["z0"]["value"]                        # m
        self.z_min = u["z_min"]["value"]                  # m
        self.z_max = u["z_max"]["value"]                  # m
        self.C_w = u["spill_coeff"]["value"]              # m2.5/s
        self.rho = u["rho"]["value"]                      # kg/m3
        self.g = u["g"]["value"]                          # m/s2
        self.A_penstock = np.pi * (self.D ** 2) / 4.0     # m2

    # ------------------------------------------------------------------
    # Penstock hydraulics (Darcy-Weisbach)
    # ------------------------------------------------------------------
    def penstock_velocity(self, Q_m3s):
        """Mean penstock flow velocity [m/s]."""
        return np.asarray(Q_m3s, dtype=float) / self.A_penstock

    def head_loss(self, Q_m3s):
        """Darcy-Weisbach + minor penstock head loss [m] (>= 0, grows as Q^2)."""
        v = self.penstock_velocity(Q_m3s)
        return (self.f * self.L / self.D + self.K_minor) * v * v / (2.0 * self.g)

    def net_head(self, Q_m3s, H_gross):
        """Net head = gross head minus penstock losses [m] (floored at 0)."""
        H_net = np.asarray(H_gross, dtype=float) - self.head_loss(Q_m3s)
        return np.clip(H_net, 0.0, None)

    # ------------------------------------------------------------------
    # Turbine hill chart
    # ------------------------------------------------------------------
    def turbine_efficiency(self, Q_m3s):
        """Hydraulic efficiency vs flow (parabolic Kaplan/Francis hill chart)."""
        Q = np.asarray(Q_m3s, dtype=float)
        q = Q / self.Q_design
        eta_t = self.eta_peak * (1.0 - self.k * (q - 1.0) ** 2)
        eta_t = np.where((q < self.q_min) | (q > self.q_max), 0.0, eta_t)
        return np.clip(eta_t, 0.0, self.eta_peak)

    def overall_efficiency(self, Q_m3s):
        """Overall plant efficiency = eta_turbine * eta_generator (in (0,1))."""
        eta_t = self.turbine_efficiency(Q_m3s)
        return np.where(eta_t > 0.0, eta_t * self.eta_gen, 0.0)

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------
    def power_kw(self, Q_m3s, H_gross):
        """Electrical output power [kW]. P = eta * rho * g * Q * H_net."""
        Q = np.asarray(Q_m3s, dtype=float)
        H_net = self.net_head(Q, H_gross)
        eta = self.overall_efficiency(Q)
        P = eta * self.rho * self.g * Q * H_net / 1000.0  # kW
        return np.clip(P, 0.0, self.P_rated)

    # ------------------------------------------------------------------
    # Headpond hydraulics
    # ------------------------------------------------------------------
    def spill_flow(self, z):
        """Broad-crested weir spill when forebay exceeds spillway crest [m3/s]."""
        z = np.asarray(z, dtype=float)
        return self.C_w * np.clip(z - self.z_max, 0.0, None) ** 1.5

    def turbine_flow(self, z, Q_demand):
        """Actual turbine flow given forebay level and operator demand [m3/s].

        Clamped to gate limits and shut off below the minimum operating level.
        """
        z = float(z)
        Q = float(Q_demand)
        if z <= self.z_min:
            return 0.0
        Q = min(max(Q, 0.0), self.q_max * self.Q_design)
        return Q

    # ------------------------------------------------------------------
    # Forebay level ODE: A_pond * dz/dt = Q_in - Q_turb - Q_spill
    # ------------------------------------------------------------------
    def _rhs(self, t, y, inflow_fn, demand_fn):
        z = y[0]
        Q_in = float(inflow_fn(t))
        Q_dem = float(demand_fn(t))
        Q_turb = self.turbine_flow(z, Q_dem)
        Q_spill = float(self.spill_flow(z))
        dzdt = (Q_in - Q_turb - Q_spill) / self.A_pond
        return [dzdt]

    def simulate(self, Q_inflow, Q_demand, z0=None, dt=10.0, duration_s=3600.0):
        """Integrate the forebay level ODE and derive power.

        Parameters
        ----------
        Q_inflow : float or callable(t)->m3/s   river inflow into the forebay
        Q_demand : float or callable(t)->m3/s   operator turbine-flow demand
        z0       : float                          initial forebay level [m]
        dt       : float                          output sampling step [s]
        duration_s : float                        simulated horizon [s]

        Returns
        -------
        dict with arrays: t, z (level), H_gross, H_net, head_loss, Q_inflow,
        Q_turbine, Q_spill, eta, power_kw.
        """
        if z0 is None:
            z0 = self.z0
        inflow_fn = Q_inflow if callable(Q_inflow) else (lambda t, v=float(Q_inflow): v)
        demand_fn = Q_demand if callable(Q_demand) else (lambda t, v=float(Q_demand): v)

        n = max(2, int(round(duration_s / dt)) + 1)
        t_eval = np.linspace(0.0, duration_s, n)

        sol = solve_ivp(
            self._rhs,
            (0.0, duration_s),
            [z0],
            t_eval=t_eval,
            args=(inflow_fn, demand_fn),
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            max_step=max(dt, 1.0),
        )

        t = sol.t
        z = sol.y[0]
        Q_in = np.array([float(inflow_fn(ti)) for ti in t])
        Q_turb = np.array([self.turbine_flow(zi, demand_fn(ti)) for ti, zi in zip(t, z)])
        Q_spill = self.spill_flow(z)
        H_gross = z.copy()
        h_loss = self.head_loss(Q_turb)
        H_net = self.net_head(Q_turb, H_gross)
        eta = self.overall_efficiency(Q_turb)
        P = self.power_kw(Q_turb, H_gross)

        return {
            "t": t,
            "z": z,
            "H_gross": H_gross,
            "H_net": H_net,
            "head_loss": h_loss,
            "Q_inflow": Q_in,
            "Q_turbine": Q_turb,
            "Q_spill": np.asarray(Q_spill, dtype=float),
            "eta": eta,
            "power_kw": P,
        }
