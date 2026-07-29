"""
EC083 -- Borehole Thermal Energy Storage (BTES) -- F2a Physics-Lumped

Borehole heat exchanger coupled to a lumped ground store, integrated with
scipy.integrate.solve_ivp.

State variables (ODE):
    T_store  -- mean temperature of the lumped ground storage volume [degC]
    T_wall   -- average borehole-wall temperature [degC]

Physics
-------
1) Lumped ground storage energy balance (0D, Nordell 1994; Hellström 1991):

       C_store * dT_store/dt = Q_ground - Q_loss

   where C_store = rho_g * cp_g * V_store [J/K] is the heat capacity of the
   active storage volume, Q_ground is the heat the borehole field delivers to
   (or extracts from) the ground, and Q_loss = UA_loss * (T_store - T_amb) is
   the loss to the surroundings.

2) Borehole heat exchanger with borehole thermal resistance R_b
   (Hellström 1991, "Ground Heat Storage"):

       T_fluid_mean - T_wall = q_line * R_b              (per metre)

   The fluid is modelled as a single-pass exchanger between inlet temperature
   and the borehole wall through the conductance UA_bhe = (m_dot cp) terms and
   the resistance R_b.  We solve the effective fluid<->wall coupling:

       Q_ground = UA_bhe * (T_fluid_mean - T_wall)

   so that the per-borehole specific heat-extraction/injection rate is
   q_line = Q_ground / (N * H) [W/m].

3) Ground thermal response via the Eskilson g-function (Eskilson 1987;
   Claesson & Javed 2011).  The mean borehole-wall temperature responds to the
   accumulated specific load relative to the undisturbed ground:

       T_wall - T_ground0 = (q_line / (2 pi k_g)) * g(t/t_s, r_b/H)

   The g-function captures the transient line-/finite-line-source ground
   response and the long-term steady-flux plateau.  We use the classic
   line-source / Eskilson closed-form approximation:

       g(t) ~ ln( H / (2 r_b) ) + 0.5 * ln( t / t_s )      (steady-flux branch)

   blended with the infinite-line-source exponential-integral solution for
   short times (Carslaw & Jaeger 1959), where t_s = H^2 / (9 alpha_g) is the
   Eskilson steady-state time scale and alpha_g = k_g/(rho_g cp_g).

   In the lumped F2a we expose this as a *ground thermal resistance to the
   wall*, R_g(t) = g(t)/(2 pi k_g), and integrate T_wall as a fast relaxation
   toward the load-driven wall temperature so that
   "heat extraction cools the ground" and seasonal charge/discharge emerge.

References
----------
- Eskilson, P. (1987). "Thermal Analysis of Heat Extraction Boreholes."
  PhD thesis, Lund University. (g-functions)
- Hellström, G. (1991). "Ground Heat Storage: Thermal Analyses of Duct
  Storage Systems." Lund University. (borehole resistance R_b, duct storage)
- Claesson, J., Javed, S. (2011). "An analytical method to calculate borehole
  fluid temperatures for time-scales from minutes to decades." ASHRAE Trans.
- Carslaw, H.S., Jaeger, J.C. (1959). "Conduction of Heat in Solids," 2nd ed.
  (infinite line source / exponential integral solution)
- Nordell, B. (1994). "Borehole Heat Store Design Optimization." Luleå Univ.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import exp1  # E1 exponential integral (line-source)


class BTES_F2a:
    """Physics-lumped BTES: borehole heat exchanger + g-function ground store."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = float(u["N_boreholes"]["value"])
        self.H = float(u["H"]["value"])
        self.r_b = float(u["r_b"]["value"])
        self.B = float(u["B_spacing"]["value"])
        self.k_g = float(u["k_ground"]["value"])
        self.rho_g = float(u["rho_ground"]["value"])
        self.cp_g = float(u["cp_ground"]["value"])
        self.R_b = float(u["R_b"]["value"])
        self.mdot = float(u["mdot"]["value"])
        self.cp_f = float(u["cp_fluid"]["value"])
        self.T_g0 = float(u["T_ground_undisturbed"]["value"])
        self.T_max = float(u["T_max"]["value"])
        self.vfac = float(u["V_store_factor"]["value"])

        # Ground thermal diffusivity [m2/s]
        self.alpha_g = self.k_g / (self.rho_g * self.cp_g)
        # Eskilson steady-state time scale t_s = H^2/(9 alpha) [s]
        self.t_s = self.H ** 2 / (9.0 * self.alpha_g)

        # Total active borehole length [m]
        self.L_tot = self.N * self.H

        # Lumped storage volume (cylinder packed in square field) and capacity
        self.V_store = self.vfac * self.N * self.B ** 2 * self.H  # m3
        self.C_store = self.rho_g * self.cp_g * self.V_store      # J/K

        # Loss conductance to surroundings: estimate from store surface area and
        # ground conductivity over a characteristic length (~H/10). Hellström-style.
        R_store = np.sqrt(self.V_store / (np.pi * self.H))  # equiv. cyl. radius
        A_surf = 2.0 * np.pi * R_store * self.H + 2.0 * np.pi * R_store ** 2
        L_char = self.H / 10.0
        self.UA_loss = self.k_g * A_surf / L_char  # W/K

        # Fluid<->wall conductance through R_b plus convective capacity rate.
        # UA_bhe combines the borehole resistance (k_g-independent) with the
        # carrier-fluid capacity-flow (m_dot cp) per borehole, total field.
        UA_Rb = self.L_tot / self.R_b               # W/K (1/R_b per metre)
        C_flow = self.N * self.mdot * self.cp_f     # W/K capacity flow rate
        # series-like combination so neither term can be exceeded
        self.UA_bhe = 1.0 / (1.0 / UA_Rb + 1.0 / C_flow)

        # Wall relaxation time constant (numerical smoothing of g-function
        # ground response into the lumped wall node). Tied to short-time
        # diffusion across one borehole radius.
        self.tau_wall = max(self.r_b ** 2 / self.alpha_g, 60.0)

    # ------------------------------------------------------------------
    # g-function / ground thermal resistance
    # ------------------------------------------------------------------
    def g_function(self, t):
        """
        Eskilson g-function (single-borehole approximation).

        Short time: infinite line source g_ils = 0.5 * E1( r_b^2 / (4 alpha t) ).
        Long time : steady-flux plateau g_ss = ln(H/(2 r_b)) + 0.5*ln(t/t_s).
        We take the line-source value but cap it at the steady plateau so the
        ground response saturates (finite borehole, Eskilson 1987).
        """
        t = np.maximum(np.asarray(t, dtype=float), 1.0)
        Fo = self.alpha_g * t / self.r_b ** 2
        g_ils = 0.5 * exp1(1.0 / (4.0 * Fo))
        # Eskilson steady-flux plateau (long-term)
        g_ss = np.log(self.H / (2.0 * self.r_b))
        return np.minimum(g_ils, g_ss)

    def ground_resistance(self, t):
        """Ground thermal resistance wall<->far-field [(m.K)/W] = g/(2 pi k)."""
        return self.g_function(t) / (2.0 * np.pi * self.k_g)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, Q_fluid_func, T_amb):
        T_store, T_wall = y
        Q_fluid = Q_fluid_func(t)  # W, + = charge (inject), - = discharge

        # Specific line load [W/m] (+ injection)
        q_line = Q_fluid / self.L_tot

        # Ground response: the wall temperature driven by accumulated load is
        # T_store + q_line * R_g(t) * H_per_m ... here R_g couples wall to store.
        # Use elapsed time since start for the g-function transient.
        R_g = self.ground_resistance(t + 1.0)  # (m.K)/W per metre
        # Wall target from load + current store temperature (load pushes wall
        # away from the store by q_line * R_g):
        T_wall_target = T_store + q_line * R_g

        # Wall relaxes toward target (fast lumped node)
        dT_wall = (T_wall_target - T_wall) / self.tau_wall

        # Heat actually delivered to the ground store through the borehole field.
        # Conserve energy: what goes in through the fluid (Q_fluid) reaches the
        # store after the wall/ground coupling. In the lumped balance the store
        # receives Q_fluid minus what is transiently stored in the wall film.
        Q_ground = Q_fluid

        # Loss to surroundings
        Q_loss = self.UA_loss * (T_store - T_amb)

        dT_store = (Q_ground - Q_loss) / self.C_store
        return [dT_store, dT_wall]

    # ------------------------------------------------------------------
    # fluid temperatures (diagnostic, from R_b and wall)
    # ------------------------------------------------------------------
    def fluid_temperatures(self, Q_fluid, T_wall):
        """
        Given field thermal power and wall temperature, return mean / in / out
        fluid temperatures using the borehole resistance R_b (Hellström 1991).

            T_fluid_mean = T_wall + q_line * R_b
            T_out - T_in = Q_fluid / (N m_dot cp_f)
            T_fluid_mean = (T_in + T_out)/2
        """
        q_line = Q_fluid / self.L_tot
        T_fluid_mean = T_wall + q_line * self.R_b
        dT_fluid = Q_fluid / (self.N * self.mdot * self.cp_f)
        T_in = T_fluid_mean - 0.5 * dT_fluid
        T_out = T_fluid_mean + 0.5 * dT_fluid
        return T_fluid_mean, T_in, T_out

    # ------------------------------------------------------------------
    # main simulation
    # ------------------------------------------------------------------
    def simulate(self, Q_fluid, T_store0=None, T_amb=8.0,
                 duration_s=180 * 24 * 3600.0, n_out=400):
        """
        Integrate the BTES over `duration_s`.

        Q_fluid : float or callable(t)->W. + = charge (inject heat into ground),
                  - = discharge (extract heat from ground).
        T_store0: initial mean store temperature [degC]; default = undisturbed.
        Returns dict of time-series arrays.
        """
        if T_store0 is None:
            T_store0 = self.T_g0

        if callable(Q_fluid):
            Q_func = Q_fluid
        else:
            Qc = float(Q_fluid)
            Q_func = lambda t: Qc

        y0 = [T_store0, T_store0]  # wall starts at store temperature
        t_eval = np.linspace(0.0, duration_s, n_out)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            args=(Q_func, T_amb),
            method="LSODA", t_eval=t_eval,
            rtol=1e-6, atol=1e-3, max_step=duration_s / 50.0,
        )

        T_store = sol.y[0]
        T_wall = sol.y[1]
        t = sol.t

        Q_arr = np.array([Q_func(tt) for tt in t])
        T_mean = np.empty_like(t)
        T_in = np.empty_like(t)
        T_out = np.empty_like(t)
        for i in range(len(t)):
            T_mean[i], T_in[i], T_out[i] = self.fluid_temperatures(Q_arr[i], T_wall[i])

        # Stored energy relative to undisturbed ground [J]
        E_stored = self.C_store * (T_store - self.T_g0)
        Q_loss = self.UA_loss * (T_store - T_amb)

        return {
            "t": t,
            "t_days": t / 86400.0,
            "T_store": T_store,
            "T_wall": T_wall,
            "T_fluid_mean": T_mean,
            "T_in": T_in,
            "T_out": T_out,
            "Q_fluid": Q_arr,
            "Q_loss": Q_loss,
            "E_stored_J": E_stored,
            "E_stored_MWh": E_stored / 3.6e9,
            "C_store_JperK": self.C_store,
            "success": sol.success,
        }
