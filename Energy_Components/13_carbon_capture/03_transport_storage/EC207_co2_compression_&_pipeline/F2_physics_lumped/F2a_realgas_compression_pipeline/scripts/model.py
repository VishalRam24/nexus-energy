"""
EC207 -- CO2 Compression & Pipeline -- F2a Physics-Lumped Model

Real-gas multistage intercooled compression of captured CO2 to the dense /
supercritical phase (~110-150 bar) followed by dense-phase pipeline transport,
with a lumped pipeline / discharge pressure-transient ODE integrated by
scipy.integrate.solve_ivp.

----------------------------------------------------------------------------
A. REAL-GAS COMPRESSIBILITY  Z(T,P)
----------------------------------------------------------------------------
CO2 near its critical point (T_c = 304.13 K, P_c = 73.77 bar) shows strong
Z deviation from unity (Z drops to ~0.2-0.4 in the dense phase).  We use the
Redlich-Kwong-Soave (Soave 1972) cubic equation of state to get Z(T,P):

    Z^3 - Z^2 + (A - B - B^2) Z - A B = 0
    A = a*alpha*P / (R^2 T^2),  B = b P / (R T)
    a = 0.42748 R^2 T_c^2 / P_c,  b = 0.08664 R T_c / P_c
    alpha = [1 + m (1 - sqrt(T/T_c))]^2,  m = 0.480 + 1.574 w - 0.176 w^2

The real-gas factor multiplies the ideal polytropic stage work (Schultz 1962
real-gas polytropic head uses an average Z over the stage).

----------------------------------------------------------------------------
B. MULTISTAGE INTERCOOLED POLYTROPIC COMPRESSION
----------------------------------------------------------------------------
Equal pressure ratio per stage  PR = (P_out/P_in)^(1/N).
Per-stage real-gas polytropic specific work (Schultz 1962; McCollum & Ogden
2006 use the same staged real-gas head):

    w_stage = Z_avg * (n/(n-1)) * R_s * T_in * (PR^((n-1)/n) - 1) / eta_p

Discharge T per stage (polytropic):  T_out = T_in * PR^((n-1)/n).
Intercooling returns the gas to T_intercool (~35C) before the next stage, so
every stage starts cool and the device stays out of the two-phase region.
Total specific work  w = sum_stages(w_stage) / eta_mech.
Specific energy consumption  SEC = w / 3.6e6 * 1000  [kWh/tCO2].

----------------------------------------------------------------------------
C. DENSE-PHASE PIPELINE PRESSURE DROP (Darcy-Weisbach)
----------------------------------------------------------------------------
Dense-phase CO2 (~800 kg/m3) is treated as a nearly incompressible liquid:

    dP = f (L/D) (rho v^2 / 2),    v = m_dot/(rho A),   Re = rho v D / mu
    f  : Churchill (1977) friction factor (valid all Re, all roughness)

(IEAGHG 2002 pipeline study uses the same Darcy / Fanning approach; the
Beggs-Brill correlation reduces to single-phase Darcy here because dense-phase
CO2 is single-phase, so we use the rigorous single-phase Darcy form.)

----------------------------------------------------------------------------
D. LUMPED PIPELINE / DISCHARGE PRESSURE-TRANSIENT ODE  (solve_ivp)
----------------------------------------------------------------------------
A lumped control volume V (compressor discharge receiver + pipeline inlet
section) is filled by the compressor (m_in) and drained into the pipeline
(m_out = f(P)).  Treating the dense-phase fluid with isothermal bulk modulus K:

    dP/dt = (K / (rho V)) * (m_in - m_out(P))

with the pipeline outflow set by the Darcy pressure-drop relation solved for
m_out given the instantaneous head (P - P_delivery).  This first-order ODE
gives the discharge pressure build-up / relaxation toward the steady operating
point -- the lumped (0D) transient required at F2.

References:
    McCollum, D.L. & Ogden, J.M. (2006). "Techno-Economic Models for Carbon
        Dioxide Compression, Transport, and Storage." UCD-ITS-RR-06-14.
    IEAGHG (2002). "Transmission of CO2 and Energy." Report PH4/6.
    IPCC (2005). Special Report on Carbon Dioxide Capture and Storage, Ch. 4.
    Span, R. & Wagner, W. (1996). J. Phys. Chem. Ref. Data 25(6), 1509-1596.
    Soave, G. (1972). Chem. Eng. Sci. 27, 1197-1203.
    Schultz, J.M. (1962). "The Polytropic Analysis of Centrifugal Compressors."
        J. Eng. Power 84(1), 69-82.
    Churchill, S.W. (1977). Chem. Eng. 84(24), 91-92.
"""

import numpy as np
from scipy.integrate import solve_ivp

R_UNIV = 8.314462  # J/(mol.K)


class CO2CompressionPipelineF2a:
    """Real-gas multistage CO2 compression + dense-phase pipeline (lumped ODE)."""

    def __init__(self, params: dict):
        c = params["compressor"]
        g = params["co2"]
        p = params["pipeline"]

        # Compressor
        self.N = int(c["n_stages"]["value"])
        self.n_poly = c["polytropic_index"]["value"]
        self.eta_p = c["eta_polytropic"]["value"]
        self.eta_m = c["eta_mech"]["value"]
        self.T_inlet_default = c["T_inlet"]["value"]
        self.T_intercool = c["T_intercool"]["value"]
        self.P_inlet_default = c["P_inlet"]["value"]
        self.P_sc = c["P_supercritical"]["value"]

        # CO2 properties
        self.M = g["molar_mass"]["value"]
        self.R_s = g["R_specific"]["value"]
        self.gamma = g["gamma"]["value"]
        self.cp_ideal = g["cp_ideal"]["value"]
        self.T_crit = g["T_critical"]["value"]
        self.P_crit = g["P_critical"]["value"] * 1e5    # bar -> Pa
        self.omega = g["omega"]["value"]
        self.rho_dense = g["rho_dense_phase"]["value"]
        self.mu_dense = g["viscosity_dense"]["value"]
        self.K_bulk = g["bulk_modulus_dense"]["value"]
        self.SEC_ref = g["SEC_ref"]["value"]

        # Pipeline
        self.L_default = p["length_km"]["value"]
        self.D_default = p["diameter_m"]["value"]
        self.roughness = p["roughness"]["value"]
        self.m_dot_default = p["mass_flow"]["value"]
        self.V_lump = p["volume_pipe_inlet"]["value"]

        # RKS EoS constants (Soave 1972)
        self._a = 0.42748 * R_UNIV ** 2 * self.T_crit ** 2 / self.P_crit
        self._b = 0.08664 * R_UNIV * self.T_crit / self.P_crit
        self._m_soave = 0.480 + 1.574 * self.omega - 0.176 * self.omega ** 2

    # === A. Real-gas compressibility Z(T,P) via Redlich-Kwong-Soave ==========

    def z_factor(self, T_K, P_bar):
        """Compressibility factor Z of CO2 via RKS cubic EoS (vapour/dense root)."""
        T = float(T_K)
        P = float(P_bar) * 1e5  # bar -> Pa
        alpha = (1.0 + self._m_soave * (1.0 - np.sqrt(T / self.T_crit))) ** 2
        A = self._a * alpha * P / (R_UNIV ** 2 * T ** 2)
        B = self._b * P / (R_UNIV * T)
        # Z^3 - Z^2 + (A - B - B^2) Z - A B = 0
        coeffs = [1.0, -1.0, (A - B - B ** 2), -A * B]
        roots = np.roots(coeffs)
        real = roots[np.abs(roots.imag) < 1e-8].real
        real = real[real > B]  # physical: Z > B
        if real.size == 0:
            return 1.0
        # gas/supercritical -> largest root; if dense liquid, smallest is liquid
        # use largest real root as the single supercritical/vapour phase
        return float(np.max(real))

    def density_real(self, T_K, P_bar):
        """Real-gas density rho = P / (Z R_s T) [kg/m3]."""
        Z = self.z_factor(T_K, P_bar)
        return float(P_bar) * 1e5 / (Z * self.R_s * float(T_K))

    # === B. Multistage intercooled polytropic compression ====================

    def stage_pressure_ratio(self, P_in, P_out):
        return (float(P_out) / float(P_in)) ** (1.0 / self.N)

    def stage_discharge_temperature(self, T_in, PR):
        """Polytropic discharge temperature of one stage [K]."""
        exponent = (self.n_poly - 1.0) / self.n_poly
        return float(T_in) * PR ** exponent

    def stage_work(self, T_in, P_in, PR):
        """Real-gas polytropic specific shaft work of one stage [J/kg]."""
        exponent = (self.n_poly - 1.0) / self.n_poly
        P_out = P_in * PR
        # Schultz real-gas head: average Z across the stage
        Z_in = self.z_factor(T_in, P_in)
        T_out = self.stage_discharge_temperature(T_in, PR)
        Z_out = self.z_factor(T_out, P_out)
        Z_avg = 0.5 * (Z_in + Z_out)
        w = Z_avg * (self.n_poly / (self.n_poly - 1.0)) * self.R_s * T_in \
            * (PR ** exponent - 1.0) / self.eta_p
        return w

    def compress(self, P_in=None, P_out=None, T_in=None):
        """
        Full multistage intercooled compression train.

        Returns a dict with per-stage temperatures, pressures, work, and the
        total specific work / SEC.  Every stage is intercooled back to
        T_intercool before the next stage.
        """
        P_in = self.P_inlet_default if P_in is None else float(P_in)
        P_out = self.P_sc if P_out is None else float(P_out)
        T_start = self.T_inlet_default if T_in is None else float(T_in)

        PR = self.stage_pressure_ratio(P_in, P_out)
        P_stage_in = P_in
        T_stage_in = T_start

        T_disch, T_in_list, P_in_list, P_out_list, w_list = [], [], [], [], []
        for s in range(self.N):
            P_stage_out = P_stage_in * PR
            Td = self.stage_discharge_temperature(T_stage_in, PR)
            w = self.stage_work(T_stage_in, P_stage_in, PR)
            T_in_list.append(T_stage_in)
            P_in_list.append(P_stage_in)
            P_out_list.append(P_stage_out)
            T_disch.append(Td)
            w_list.append(w)
            # intercool before next stage
            T_stage_in = self.T_intercool
            P_stage_in = P_stage_out

        w_total = sum(w_list) / self.eta_m  # J/kg
        sec = w_total / 3.6e6 * 1000.0       # kWh/tonne
        return {
            "stage_T_in": np.array(T_in_list),
            "stage_T_discharge": np.array(T_disch),
            "stage_P_in": np.array(P_in_list),
            "stage_P_out": np.array(P_out_list),
            "stage_work": np.array(w_list),
            "pressure_ratio": PR,
            "w_specific_J_per_kg": w_total,
            "SEC_kWh_per_tCO2": sec,
            "P_discharge_bar": P_out_list[-1],
            "T_final_discharge_K": T_disch[-1],
            "T_after_intercool_K": self.T_intercool,
        }

    def shaft_power_kw(self, m_dot_kg_s, P_in=None, P_out=None, T_in=None):
        res = self.compress(P_in, P_out, T_in)
        return float(m_dot_kg_s) * res["w_specific_J_per_kg"] / 1000.0

    def is_supercritical(self, T_K, P_bar):
        """True where CO2 is in dense / supercritical phase."""
        return (float(T_K) > self.T_crit) and (float(P_bar) > self.P_crit / 1e5)

    # === C. Dense-phase pipeline pressure drop (Darcy-Weisbach) ==============

    def _friction_factor_churchill(self, Re, D):
        eps = self.roughness
        Re = max(Re, 1.0)
        A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * eps / D))) ** 16
        B = (37530.0 / Re) ** 16
        return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)

    def pipeline_pressure_drop_bar(self, m_dot_kg_s, length_km=None,
                                   diameter_m=None, rho_kg_m3=None):
        """Dense-phase CO2 pipeline pressure drop via Darcy-Weisbach [bar]."""
        m = float(m_dot_kg_s)
        L = (self.L_default if length_km is None else float(length_km)) * 1000.0
        D = self.D_default if diameter_m is None else float(diameter_m)
        rho = self.rho_dense if rho_kg_m3 is None else float(rho_kg_m3)
        A_pipe = np.pi / 4.0 * D ** 2
        v = m / (rho * A_pipe)
        Re = rho * v * D / self.mu_dense
        f = self._friction_factor_churchill(Re, D)
        dP_Pa = f * (L / D) * (rho * v ** 2 / 2.0)
        return dP_Pa / 1e5

    def _pipeline_outflow(self, dP_bar, length_km, diameter_m, rho):
        """Invert Darcy: mass flow [kg/s] that produces a given head dP_bar."""
        if dP_bar <= 0.0:
            return 0.0
        L = length_km * 1000.0
        D = diameter_m
        A_pipe = np.pi / 4.0 * D ** 2
        dP_Pa = dP_bar * 1e5
        # iterate on friction factor (depends on Re which depends on v)
        v = 1.0
        for _ in range(40):
            Re = rho * v * D / self.mu_dense
            f = self._friction_factor_churchill(Re, D)
            v_new = np.sqrt(2.0 * dP_Pa * D / (f * L * rho))
            if abs(v_new - v) < 1e-9:
                v = v_new
                break
            v = v_new
        return rho * v * A_pipe

    # === D. Lumped pressure-transient ODE (solve_ivp) ========================

    def simulate_pressure_transient(self, m_in_kg_s=None, P0_bar=None,
                                    P_delivery_bar=80.0, length_km=None,
                                    diameter_m=None, duration_s=300.0,
                                    n_eval=200):
        """
        Lumped discharge / pipeline-inlet pressure transient.

        dP/dt = (K / (rho V)) (m_in - m_out(P))
        m_out(P) from Darcy pressure drop across the pipeline given head
        (P - P_delivery).  Integrated with solve_ivp (RK45).

        m_in_kg_s : compressor mass-flow feeding the lumped volume.
        P0_bar    : initial discharge pressure (default = delivery pressure).
        Returns dict of time-series arrays.
        """
        m_in = self.m_dot_default if m_in_kg_s is None else float(m_in_kg_s)
        L = self.L_default if length_km is None else float(length_km)
        D = self.D_default if diameter_m is None else float(diameter_m)
        rho = self.rho_dense
        P0 = P_delivery_bar if P0_bar is None else float(P0_bar)

        K = self.K_bulk          # Pa
        V = self.V_lump          # m3
        coeff = K / (rho * V)    # Pa/s per (kg/s); P in Pa here

        def rhs(t, y):
            P_Pa = y[0]
            P_bar = P_Pa / 1e5
            head_bar = max(P_bar - P_delivery_bar, 0.0)
            m_out = self._pipeline_outflow(head_bar, L, D, rho)
            return [coeff * (m_in - m_out)]

        t_eval = np.linspace(0.0, duration_s, n_eval)
        sol = solve_ivp(rhs, (0.0, duration_s), [P0 * 1e5], t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e2, max_step=duration_s / 50.0)

        P_bar = sol.y[0] / 1e5
        m_out = np.array([self._pipeline_outflow(max(p - P_delivery_bar, 0.0), L, D, rho)
                          for p in P_bar])
        # steady-state pressure: head where m_out == m_in
        P_ss = P_delivery_bar + self.pipeline_pressure_drop_bar(m_in, L, D, rho)
        return {
            "t": sol.t,
            "P_discharge_bar": P_bar,
            "m_out_kg_s": m_out,
            "m_in_kg_s": m_in,
            "P_steady_state_bar": P_ss,
            "P_delivery_bar": P_delivery_bar,
        }
