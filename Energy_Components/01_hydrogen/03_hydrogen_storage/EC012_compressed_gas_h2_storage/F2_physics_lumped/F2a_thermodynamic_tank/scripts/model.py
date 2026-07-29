"""
EC012 -- Compressed Gas H2 Storage -- F2a Thermodynamic Tank (physics-lumped)

A 0D lumped open-system thermodynamic tank model for fill (charge) and
discharge of high-pressure gaseous hydrogen.  Couples a real-gas (truncated
virial) equation of state to a first-law energy balance on the control volume
formed by the gas inside the tank, with lumped heat exchange to a thermal wall
and the ambient.

State vector (integrated with scipy.integrate.solve_ivp):
    y[0] = m_gas   gas mass inside tank        [kg]
    y[1] = T_gas   bulk gas temperature        [K]
    y[2] = T_wall  lumped wall temperature     [K]

Governing equations
--------------------
1. Real-gas equation of state (Leachman et al. 2009 virial, refit to NIST):
       Z(T,P) = 1 + B(T) P_MPa + C(T) P_MPa^2
       B(T)   = A1 + A2 / T^2     [MPa^-1]
       C(T)   = B1 + B2 / T^2     [MPa^-2]
   Density form (P implicit in Z): solve for P from
       P = m * Z(T,P) * R_H2 * T / V        (fixed point in P)

2. Open-system (control-volume) first law, Moran & Shapiro:
       d(m u)/dt = m_dot_in h_in  -  m_dot_out h_out  -  Q_wall
   Expanding d(m u)/dt = m du/dt + u dm/dt and using
   u = cv T,  h = cp T (ideal-gas caloric model for H2; the real-gas
   correction enters via the EOS / Z, not the caloric equation, which is an
   excellent approximation for H2 since its inversion behaviour is weak):
       m cv dT/dt = m_dot_in (cp T_in)  -  m_dot_out (cp T)
                    - cv T (m_dot_in - m_dot_out)  -  Q_wall
   so for a pure fill (m_dot_out = 0):
       m cv dT/dt = m_dot_in cp T_in  -  m_dot_in cv T  -  Q_wall
   The (cp - cv) T_in term is exactly the flow-work that heats the tank on
   fast fill (the classic ~+50 K adiabatic fill rise for H2, Zheng 2012).

3. Mass balance:
       dm/dt = m_dot_in - m_dot_out

4. Lumped two-node wall heat transfer:
       Q_wall   = UA_gw (T_gas - T_wall)         [W]  gas -> wall
       m_w cp_w dT_wall/dt = Q_wall - UA_wa (T_wall - T_amb)
   With UA_gw and UA_wa derived from the single overall UA_ambient by a
   series-resistance split (gas film much less resistive than insulation),
   so the steady state recovers the F1b Newton-cooling time constant.

References
----------
Leachman, J.W. et al. (2009). Fundamental equations of state for parahydrogen,
    normal hydrogen, and orthohydrogen. J. Phys. Chem. Ref. Data 38(3) 721-748.
Moran, M.J. & Shapiro, H.N. Fundamentals of Engineering Thermodynamics
    (open-system / control-volume first law, transient filling of a tank).
Zheng, J. et al. (2012). Standardized thermodynamic ... fast filling of
    hydrogen tanks. Int. J. Hydrogen Energy 37(2) 1048-1057.
Lemmon, E.W. et al. (2008). NIST Chemistry WebBook (H2 properties).
"""

import numpy as np
from scipy.integrate import solve_ivp

R_UNIVERSAL = 8.314   # J/(mol.K)


class CompressedGasH2F2a:
    """Lumped thermodynamic tank model for gaseous H2 fill / discharge."""

    def __init__(self, params: dict):
        tank = params["tank"]["type_IV"]
        h2 = params["hydrogen"]
        z = params["compressibility"]
        amb = params["ambient"]
        inflow = params["inflow"]

        self.V_tank = tank["volume"]["value"]            # m3
        self.P_max = tank["max_pressure"]["value"]       # bar
        self.P_min = tank["min_pressure"]["value"]       # bar
        self.tank_mass = tank["mass_empty"]["value"]     # kg
        self.m_wall = tank["wall_mass"]["value"]         # kg
        self.cp_wall = tank["cp_wall"]["value"]          # J/(kg.K)
        self.UA_amb = tank["UA_ambient"]["value"]        # W/K overall

        self.M_H2 = h2["molar_mass"]["value"]            # kg/mol
        self.LHV = h2["LHV"]["value"]                    # MJ/kg
        self.gamma = h2["gamma"]["value"]
        self.cp_H2 = h2["cp"]["value"]                   # J/(kg.K)
        self.cv_H2 = h2["cv"]["value"]                   # J/(kg.K)

        # Virial coefficients (refit to NIST H2, preserved from F1b). P in MPa.
        self.A1 = z["A1"]
        self.A2 = z["A2"]
        self.B1 = z["B1"]
        self.B2 = z["B2"]

        self.T_amb_default = amb["T_amb_default"]["value"]   # K
        self.T_in_default = inflow["T_in"]["value"]          # K
        self.P_supply = inflow["P_supply"]["value"]          # bar

        self.R_H2 = R_UNIVERSAL / self.M_H2              # J/(kg.K)

        # Series split of the single overall UA into gas->wall and wall->amb.
        # Wall is the dominant resistance; gas film is ~10x more conductive.
        # Series: 1/UA = 1/UA_gw + 1/UA_wa, with UA_gw = 10 * UA_wa.
        self.UA_gw = 11.0 * self.UA_amb     # gas -> wall
        self.UA_wa = (11.0 / 10.0) * self.UA_amb   # wall -> ambient

    # ------------------------------------------------------------------
    # Real-gas EOS
    # ------------------------------------------------------------------
    def _B_virial(self, T_K):
        T = np.asarray(T_K, dtype=float)
        return self.A1 + self.A2 / T ** 2

    def _C_virial(self, T_K):
        T = np.asarray(T_K, dtype=float)
        return self.B1 + self.B2 / T ** 2

    def compressibility_factor(self, P_bar, T_K):
        """Z(T,P) = 1 + B(T) P_MPa + C(T) P_MPa^2 (Leachman 2009 refit)."""
        P = np.asarray(P_bar, dtype=float)
        T = np.asarray(T_K, dtype=float)
        P_MPa = P * 0.1
        B = self._B_virial(T)
        C = self._C_virial(T)
        return np.maximum(1.0 + B * P_MPa + C * P_MPa ** 2, 1e-3)

    def pressure(self, m_kg, T_K):
        """
        Tank pressure [bar] for a given gas mass and temperature, solving the
        implicit real-gas EOS  P = m Z(T,P) R_H2 T / V  by fixed-point
        iteration (Z depends weakly on P, so this converges in a few steps).
        """
        m = float(m_kg)
        T = float(T_K)
        rho = m / self.V_tank                       # kg/m3
        P_ideal_Pa = rho * self.R_H2 * T            # ideal-gas guess
        P_bar = P_ideal_Pa / 1e5
        for _ in range(40):
            Z = float(self.compressibility_factor(P_bar, T))
            P_new = P_ideal_Pa * Z / 1e5            # P = rho Z R T
            if abs(P_new - P_bar) < 1e-8 * max(1.0, P_bar):
                P_bar = P_new
                break
            P_bar = P_new
        return P_bar

    def density_from_PT(self, P_bar, T_K):
        """Gas density [kg/m3] from EOS: rho = P / (Z R_H2 T)."""
        P_Pa = np.asarray(P_bar, dtype=float) * 1e5
        T = np.asarray(T_K, dtype=float)
        Z = self.compressibility_factor(P_bar, T)
        return P_Pa / (Z * self.R_H2 * T)

    def mass_from_PT(self, P_bar, T_K):
        """Gas mass [kg] in the tank at given P, T (real gas)."""
        return self.density_from_PT(P_bar, T_K) * self.V_tank

    # ------------------------------------------------------------------
    # Energy / SOC helpers
    # ------------------------------------------------------------------
    def energy_stored(self, m_kg):
        """Chemical energy content [MJ] = m * LHV."""
        return np.asarray(m_kg, dtype=float) * self.LHV

    def soc(self, m_kg, T_amb_K=None):
        """
        State of charge (0-1): usable mass fraction between the mass at P_min
        and the mass at P_max, both evaluated at ambient temperature.
        """
        if T_amb_K is None:
            T_amb_K = self.T_amb_default
        m_min = self.mass_from_PT(self.P_min, T_amb_K)
        m_max = self.mass_from_PT(self.P_max, T_amb_K)
        m = np.asarray(m_kg, dtype=float)
        return np.clip((m - m_min) / (m_max - m_min), 0.0, 1.0)

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, y, mdot_fn, T_in, T_amb):
        """
        State derivatives for [m_gas, T_gas, T_wall].

        mdot_fn(t) > 0 : filling   (inflow of H2 at T_in)
        mdot_fn(t) < 0 : discharge (outflow of H2 at the current tank T)
        """
        m, T, T_w = y
        m = max(m, 1e-6)
        T = max(T, 1.0)

        mdot = mdot_fn(t)

        # gas <-> wall and wall <-> ambient heat flows
        Q_gw = self.UA_gw * (T - T_w)          # gas loses Q_gw to wall (>0 if hot)
        Q_wa = self.UA_wa * (T_w - T_amb)       # wall loses to ambient

        # Open-system first law:  m cv dT/dt = sum(mdot_i h_i) - u sum(mdot_i) - Q
        # h = cp * T_stream, u = cv * T_gas
        if mdot >= 0.0:                          # filling: stream at T_in
            h_stream = self.cp_H2 * T_in
        else:                                    # discharge: stream leaves at T
            h_stream = self.cp_H2 * T
        u_gas = self.cv_H2 * T

        dm_dt = mdot
        # m cv dT/dt = mdot*h_stream - u_gas*mdot - Q_gw
        dT_dt = (mdot * h_stream - u_gas * mdot - Q_gw) / (m * self.cv_H2)

        dTw_dt = (Q_gw - Q_wa) / (self.m_wall * self.cp_wall)

        return [dm_dt, dT_dt, dTw_dt]

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def simulate(self, mdot, T0_K, T_amb_K=None, T_in_K=None,
                 m0_kg=None, P0_bar=None, dt=1.0, duration_s=60.0):
        """
        Integrate the lumped tank during fill / discharge.

        Parameters
        ----------
        mdot : float or callable(t)
            H2 mass flow [kg/s].  >0 fill, <0 discharge.
        T0_K : float
            Initial gas (and wall) temperature [K].
        T_amb_K : float
            Ambient temperature [K] (default param).
        T_in_K : float
            Inlet H2 temperature for filling [K] (default param).
        m0_kg : float, optional
            Initial gas mass [kg]. If None, derived from P0_bar & T0_K.
        P0_bar : float, optional
            Initial tank pressure [bar] (used if m0_kg is None; default P_min).
        dt : float
            Output sample interval [s].
        duration_s : float
            Total simulated time [s].

        Returns
        -------
        dict with arrays: t, mass, temperature, T_wall, pressure, density,
            soc, energy_MJ, and scalar-ish 'mdot' input echoed per sample.
        """
        if T_amb_K is None:
            T_amb_K = self.T_amb_default
        if T_in_K is None:
            T_in_K = self.T_in_default

        if m0_kg is None:
            if P0_bar is None:
                P0_bar = self.P_min
            m0_kg = float(self.mass_from_PT(P0_bar, T0_K))

        mdot_fn = mdot if callable(mdot) else (lambda t: float(mdot))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        y0 = [m0_kg, T0_K, T0_K]
        sol = solve_ivp(
            self._rhs, (0.0, duration_s), y0,
            t_eval=t_eval, args=(mdot_fn, T_in_K, T_amb_K),
            method="RK45", rtol=1e-8, atol=1e-10, max_step=dt,
        )

        t_out = sol.t
        m_out = sol.y[0]
        T_out = sol.y[1]
        Tw_out = sol.y[2]
        N = len(t_out)

        P_out = np.zeros(N)
        rho_out = np.zeros(N)
        for i in range(N):
            P_out[i] = self.pressure(m_out[i], T_out[i])
            rho_out[i] = m_out[i] / self.V_tank

        soc_out = self.soc(m_out, T_amb_K)
        E_out = self.energy_stored(m_out)
        mdot_out = np.array([mdot_fn(tt) for tt in t_out])

        return {
            "t": t_out,
            "mass": m_out,
            "temperature": T_out,
            "T_wall": Tw_out,
            "pressure": P_out,
            "density": rho_out,
            "soc": soc_out,
            "energy_MJ": E_out,
            "mdot": mdot_out,
        }
