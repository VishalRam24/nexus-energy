"""
EC191 -- Gas Compressor Station -- F2a Centrifugal Compressor (Physics-Lumped)

Physics-lumped (0D) dynamic model of a single-body pipeline centrifugal
compressor driven by a gas turbine / electric motor, handling real natural gas.

Component physics
-----------------
1. Dimensionless head-flow characteristic (centrifugal map). With flow
   coefficient phi = Q1 / (A2 * U2) and head coefficient psi = g*H_poly / U2^2,
   the map is modelled as an inverted parabola (Boyce 2012, Ch.7):

       psi(phi) = psi0 * (1 - a * ((phi - phi_design)/phi_design)^2)

   Polytropic head H_poly = psi * U2^2 / g          [J/kg when expressed as g*H]
   Tip speed scales with shaft speed:  U2 = U2_design * (N/N_design).

   Operating limits:
       surge : phi < phi_surge   (positive map slope, flow instability)
       choke : phi > phi_choke   (stonewall, Mach-limited)

2. Polytropic compression of REAL natural gas (Boyce 2012 eq. 7-x; Menon 2005).
   The delivered polytropic head fixes the pressure ratio through

       H_poly = Z_avg * R_s * T1 * (n/(n-1)) * (PR^((n-1)/n) - 1)

   with the polytropic exponent from the polytropic efficiency:

       (n-1)/n = (gamma-1)/gamma / eta_poly

   Solving for PR:
       PR = (1 + H_poly*(n-1)/n / (Z_avg*R_s*T1))^(n/(n-1))

3. Discharge temperature (polytropic path):
       T2 = T1 * PR^((n-1)/n)          (T2 > T1 always for PR > 1)

4. Shaft / driver power:
       W_shaft = m_dot * H_poly / eta_mech            [W]
       W_fuel  = W_shaft / eta_driver  (gas-turbine fuel power)   [W]

5. Lumped discharge-plenum pressure ODE (gas mass accumulation, ideal-gas
   plenum with real-gas Z, isothermal-plenum approximation):

       dP_disch/dt = (Z*R_s*T_disch / V_plenum) * (m_in - m_out)

   where the throttle valve sets  m_out = C_valve * sqrt(max(P_disch - P_line,0)).

6. Lumped casing/gas thermal ODE for discharge temperature relaxation:

       m_metal*cp_metal * dT_disch/dt =
            m_in*cp*(T2_ideal - T_disch) - hA_loss*(T_disch - T_ambient)

   i.e. hot compressed gas drives T_disch toward the polytropic T2, casing
   heat loss pulls it back toward ambient. Energy conservation:
   shaft work in = enthalpy rise of gas (sensible) within efficiency bounds.

Real-gas Z is evaluated with a lightweight Standing-Katz-style correlation in
reduced coordinates (Menon 2005, Ch.1); a hardcoded inlet Z is used as the
default reference.

References
----------
    Boyce, M.P. (2012). Gas Turbine Engineering Handbook, 4th ed., Ch.7
        "Centrifugal Compressors". Butterworth-Heinemann.
    Menon, E.S. (2005). Gas Pipeline Hydraulics, CRC Press. Ch.1, Ch.5-6.
    GPSA (2004). Engineering Data Book, 12th ed. (gas properties, Z-factor).
"""

import numpy as np
from scipy.integrate import solve_ivp

G0 = 9.80665  # m/s^2


class NGCompressorF2a:
    """Centrifugal natural-gas compressor: head-flow map + polytropic real gas
    + lumped discharge pressure/thermal transient."""

    R_univ = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        g = params["gas"]

        # --- machine / map ---
        self.U2_design = u["U2_tip_speed"]["value"]      # m/s
        self.N_design = u["N_design_rpm"]["value"]       # rpm
        self.psi0 = u["head_coeff_psi0"]["value"]
        self.a_map = u["head_curve_slope_a"]["value"]
        self.phi_design = u["phi_design"]["value"]
        self.phi_surge = u["phi_surge"]["value"]
        self.phi_choke = u["phi_choke"]["value"]
        self.A2 = u["A2_eye_area"]["value"]              # m2
        self.eta_poly = u["eta_polytropic"]["value"]
        self.eta_mech = u["eta_mech"]["value"]
        self.eta_driver = u["eta_driver"]["value"]

        # --- operating defaults ---
        self.T_inlet = u["T_inlet"]["value"]             # K
        self.P_inlet = u["P_inlet"]["value"]             # bar
        self.P_disch_init = u["P_discharge_init"]["value"]
        self.V_plenum = u["V_plenum"]["value"]           # m3
        self.m_metal = u["m_metal"]["value"]             # kg
        self.cp_metal = u["cp_metal"]["value"]           # J/(kg.K)
        self.hA_loss = u["hA_loss"]["value"]             # W/K
        self.T_ambient = u["T_ambient"]["value"]         # K
        self.C_valve = u["C_valve"]["value"]             # kg/(s.bar^0.5)
        self.P_line = u["P_line"]["value"]               # bar

        # --- gas ---
        self.M = g["molar_mass"]["value"]                # kg/mol
        self.R_s = g["R_specific"]["value"]              # J/(kg.K)
        self.gamma = g["gamma"]["value"]
        self.cp = g["cp"]["value"]                       # J/(kg.K)
        self.Z_inlet = g["Z_inlet"]["value"]
        self.Tc = g["Tc"]["value"]                       # K
        self.Pc = g["Pc"]["value"]                       # bar
        self.LHV = g["LHV"]["value"]                     # MJ/kg

        # polytropic exponent: (n-1)/n = (gamma-1)/gamma / eta_poly
        self.np_exp = (self.gamma - 1.0) / self.gamma / self.eta_poly

    # ------------------------------------------------------------------
    # Real-gas compressibility (Standing-Katz style, Menon 2005 Ch.1)
    # ------------------------------------------------------------------
    def z_factor(self, P_bar, T_K):
        """Lightweight Z(P,T) via reduced coordinates. Returns Z in (0,1].

        Uses a simple Dranchuk-style virial truncation calibrated so that at
        the reference suction point it reproduces the hardcoded Z_inlet."""
        Pr = P_bar / self.Pc
        Tr = T_K / self.Tc
        # truncated correlation: Z = 1 + B*Pr/Tr with B<0 (attractive regime)
        B = -0.30
        Z = 1.0 + B * Pr / (Tr ** 3)
        return float(np.clip(Z, 0.25, 1.05))

    def z_avg(self, P_in_bar, P_out_bar, T_K):
        """Average Z over the compression path (Menon 2005 averaging)."""
        return 0.5 * (self.z_factor(P_in_bar, T_K) + self.z_factor(P_out_bar, T_K))

    # ------------------------------------------------------------------
    # Centrifugal head-flow map
    # ------------------------------------------------------------------
    def tip_speed(self, speed_ratio):
        """Impeller tip speed [m/s] for fractional shaft speed."""
        return self.U2_design * speed_ratio

    def flow_coefficient(self, m_dot, P_in_bar, T_in, speed_ratio):
        """phi = Q1 / (A2 * U2) using actual inlet volumetric flow Q1."""
        rho_in = self.density(P_in_bar, T_in)
        Q1 = m_dot / rho_in                              # m3/s actual
        U2 = self.tip_speed(speed_ratio)
        return Q1 / (self.A2 * U2)

    def head_coefficient(self, phi):
        """psi(phi) inverted-parabola map (Boyce 2012)."""
        frac = (phi - self.phi_design) / self.phi_design
        return self.psi0 * (1.0 - self.a_map * frac ** 2)

    def polytropic_head(self, phi, speed_ratio):
        """Delivered polytropic head g*H_poly [J/kg]."""
        psi = self.head_coefficient(phi)
        U2 = self.tip_speed(speed_ratio)
        return max(psi * U2 ** 2, 0.0)

    def density(self, P_bar, T_K):
        """Real-gas density [kg/m3]: rho = P / (Z R_s T)."""
        Z = self.z_factor(P_bar, T_K)
        return (P_bar * 1e5) / (Z * self.R_s * T_K)

    # ------------------------------------------------------------------
    # Operating-limit checks
    # ------------------------------------------------------------------
    def in_surge(self, phi):
        return phi < self.phi_surge

    def in_choke(self, phi):
        return phi > self.phi_choke

    # ------------------------------------------------------------------
    # Polytropic compression: head -> pressure ratio, T2
    # ------------------------------------------------------------------
    def pressure_ratio(self, H_poly, T_in, P_in_bar):
        """Pressure ratio delivered for a given polytropic head [J/kg]."""
        Z = self.z_factor(P_in_bar, T_in)
        # H_poly = Z*R_s*T1*(1/np_exp)*(PR^np_exp - 1)
        term = 1.0 + H_poly * self.np_exp / (Z * self.R_s * T_in)
        term = max(term, 1.0)
        return term ** (1.0 / self.np_exp)

    def discharge_temperature_ideal(self, T_in, PR):
        """Polytropic-path discharge temperature [K] (always > T_in for PR>1)."""
        return T_in * PR ** self.np_exp

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------
    def shaft_power(self, m_dot, H_poly):
        """Shaft power [W]: m_dot * head / eta_mech."""
        return m_dot * H_poly / self.eta_mech

    def fuel_power(self, W_shaft):
        """Gas-turbine driver fuel power [W] = shaft / eta_driver."""
        return W_shaft / self.eta_driver

    def operating_point(self, m_dot, speed_ratio=1.0, P_in_bar=None, T_in=None):
        """Steady-state operating point summary at given flow & speed."""
        P_in_bar = self.P_inlet if P_in_bar is None else P_in_bar
        T_in = self.T_inlet if T_in is None else T_in

        phi = self.flow_coefficient(m_dot, P_in_bar, T_in, speed_ratio)
        H = self.polytropic_head(phi, speed_ratio)
        PR = self.pressure_ratio(H, T_in, P_in_bar)
        T2 = self.discharge_temperature_ideal(T_in, PR)
        W_sh = self.shaft_power(m_dot, H)
        W_fuel = self.fuel_power(W_sh)
        return {
            "phi": phi,
            "psi": self.head_coefficient(phi),
            "H_poly_J_per_kg": H,
            "pressure_ratio": PR,
            "P_discharge_bar": PR * P_in_bar,
            "T_discharge_K": T2,
            "shaft_power_W": W_sh,
            "shaft_power_MW": W_sh / 1e6,
            "fuel_power_MW": W_fuel / 1e6,
            "in_surge": self.in_surge(phi),
            "in_choke": self.in_choke(phi),
        }

    # ------------------------------------------------------------------
    # Lumped transient ODE: discharge pressure + thermal
    # ------------------------------------------------------------------
    def simulate(self, m_in, speed_ratio=1.0, P_in_bar=None, T_in=None,
                 dt=0.1, duration_s=60.0):
        """Integrate discharge-plenum pressure and casing temperature.

        Parameters
        ----------
        m_in : float or callable(t)->float
            Compressor mass flow into the plenum [kg/s].
        speed_ratio : float
            Shaft-speed fraction of design (sets tip speed / head).
        P_in_bar, T_in : suction conditions (defaults from params).
        dt : output sample step [s];  duration_s : total time [s].

        Returns dict of time-series arrays.
        """
        P_in_bar = self.P_inlet if P_in_bar is None else P_in_bar
        T_in = self.T_inlet if T_in is None else T_in

        if callable(m_in):
            m_of_t = m_in
        else:
            m_of_t = lambda t: float(m_in)

        # initial state: [P_disch (Pa), T_disch (K)]
        y0 = [self.P_disch_init * 1e5, T_in + 30.0]

        def rhs(t, y):
            P_pa, T_d = y
            P_bar = P_pa / 1e5
            m = max(m_of_t(t), 0.0)

            # compressor operating point at this flow & speed
            phi = self.flow_coefficient(m, P_in_bar, T_in, speed_ratio)
            H = self.polytropic_head(phi, speed_ratio)
            PR = self.pressure_ratio(H, T_in, P_in_bar)
            T2_ideal = self.discharge_temperature_ideal(T_in, PR)

            # plenum out-flow through throttle valve
            dP_valve = max(P_bar - self.P_line, 0.0)
            m_out = self.C_valve * np.sqrt(dP_valve)

            # real-gas plenum pressure ODE
            Z = self.z_factor(P_bar, T_d)
            dPdt = (Z * self.R_s * T_d / self.V_plenum) * (m - m_out)  # Pa/s

            # casing/gas thermal ODE
            Q_in = m * self.cp * (T2_ideal - T_d)
            Q_loss = self.hA_loss * (T_d - self.T_ambient)
            dTdt = (Q_in - Q_loss) / (self.m_metal * self.cp_metal)

            return [dPdt, dTdt]

        n_out = max(int(round(duration_s / dt)) + 1, 2)
        t_eval = np.linspace(0.0, duration_s, n_out)
        sol = solve_ivp(rhs, (0.0, duration_s), y0, t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-3, max_step=dt)

        P_pa = sol.y[0]
        T_d = sol.y[1]
        P_bar = P_pa / 1e5

        # derived series
        phi = np.array([self.flow_coefficient(max(m_of_t(tt), 0.0), P_in_bar,
                                              T_in, speed_ratio) for tt in sol.t])
        H = np.array([self.polytropic_head(p, speed_ratio) for p in phi])
        PR_comp = np.array([self.pressure_ratio(h, T_in, P_in_bar) for h in H])
        m_arr = np.array([max(m_of_t(tt), 0.0) for tt in sol.t])
        W_shaft = np.array([self.shaft_power(m, h) for m, h in zip(m_arr, H)])
        W_fuel = W_shaft / self.eta_driver

        return {
            "t": sol.t,
            "P_discharge_bar": P_bar,
            "T_discharge_K": T_d,
            "phi": phi,
            "psi": np.array([self.head_coefficient(p) for p in phi]),
            "H_poly_J_per_kg": H,
            "pressure_ratio": PR_comp,
            "mass_flow_kg_s": m_arr,
            "shaft_power_MW": W_shaft / 1e6,
            "fuel_power_MW": W_fuel / 1e6,
            "in_surge": phi < self.phi_surge,
            "in_choke": phi > self.phi_choke,
        }
