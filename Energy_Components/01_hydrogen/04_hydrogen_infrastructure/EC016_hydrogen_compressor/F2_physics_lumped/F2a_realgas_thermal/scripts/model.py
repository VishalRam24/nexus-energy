"""
EC016 -- Hydrogen Compressor -- F2a Real-Gas Multistage Reciprocating Model

Physics-lumped (0D) first-principles model of a multistage intercooled
reciprocating hydrogen compressor with a coupled lumped thermal-transient ODE.

Each stage performs a real-gas polytropic compression. Hydrogen departs strongly
from ideal-gas behaviour at the high pressures used for refuelling (350-900 bar),
so the compressibility factor Z is evaluated with the Redlich-Kwong cubic
equation of state and folded into both the discharge-temperature and the
specific-work relations.

----------------------------------------------------------------------------
Real-gas compressibility (Redlich-Kwong, 1949)
----------------------------------------------------------------------------
    Z^3 - Z^2 + (A - B - B^2) Z - A B = 0
    A = a P / (R^2 T^2.5),   B = b P / (R T)
    a = 0.42748 R^2 Tc^2.5 / Pc,   b = 0.08664 R Tc / Pc          (per mole)

Z is taken as the largest real root (vapour/supercritical branch). For H2 at
ambient T and 20-900 bar this gives Z ~ 1.01-1.6, i.e. H2 is *less* compressible
than an ideal gas -- exactly the behaviour that drives the well-known >30 %
energy penalty of high-pressure H2 compression (Sdanghi 2019).

----------------------------------------------------------------------------
Stage thermodynamics (polytropic, real gas)
----------------------------------------------------------------------------
Equal stage pressure ratio (minimises total work for equal-T intercooling):
    PR_s = (P_out / P_in)^(1/N)

Discharge temperature, real-gas polytropic (Bloch 2006, eq. for non-ideal n):
    T2 = T1 * PR_s^(Z * (n-1)/n)

Real-gas isentropic discharge temperature (for isentropic-efficiency split):
    T2s = T1 * PR_s^(Z * (gamma-1)/gamma)

Stage specific work (polytropic head / efficiency, real gas):
    w_poly = Z * (n/(n-1)) * R_s * T1 * (PR_s^(Z*(n-1)/n) - 1)
    w_stage = w_poly / eta_poly

Isentropic check (Bloch 2006):
    w_isen_ideal = Z * (gamma/(gamma-1)) * R_s * T1 * (PR_s^(Z*(g-1)/g) - 1)
    w_stage_isen = w_isen_ideal / eta_isen
(both routes bracket the real shaft work; the polytropic route is used for the
energy balance, the isentropic route for the efficiency diagnostic.)

----------------------------------------------------------------------------
Volumetric efficiency (reciprocating clearance, Bloch 2006)
----------------------------------------------------------------------------
    eta_vol = 1 - c * (PR_s^(1/n) - 1)
with c the clearance-volume ratio. This sets the actual induced volume / swept
volume and falls as the stage pressure ratio rises.

----------------------------------------------------------------------------
Intercooler (effectiveness model, Aungier 2000)
----------------------------------------------------------------------------
    T_after_ic = T2 - eps * (T2 - T_coolant)
The cooled gas is the suction state of the next stage.

----------------------------------------------------------------------------
Lumped thermal-transient ODE (cylinder + intercooler metal mass)
----------------------------------------------------------------------------
A fraction f of the dissipated heat (compression irreversibility + mechanical
friction) is absorbed by the lumped metal mass, the rest leaves with the gas /
coolant. The metal exchanges with ambient:

    m_metal * cp_metal * dT_metal/dt = f * Q_dissipated - hA_amb * (T_metal - T_amb)

    Q_dissipated = m_dot * [ (1/eta_mech - 1) * w_poly_sum                  (friction)
                             + sum_stage (w_stage - w_reversible_stage) ]   (irrev.)

solved with scipy.integrate.solve_ivp. This captures cold-start warm-up and the
metal-temperature transient that governs valve/seal thermal stress.

References
----------
Sdanghi, G. et al. (2019). Review of the current technologies and performances
    of hydrogen compression for stationary and automotive applications.
    Renew. Sustain. Energy Rev., 102, 150-170.
Bloch, H. P. (2006). A Practical Guide to Compressor Technology, 2nd ed., Wiley.
Redlich, O. & Kwong, J. N. S. (1949). Chem. Rev., 44(1), 233-244.
Leachman, J. W. et al. (2009). J. Phys. Chem. Ref. Data, 38(3), 721-748.
Bossel, U. (2006). Proc. IEEE, 94(10), 1826-1837.
Aungier, R. H. (2000). Centrifugal Compressors, ASME Press.
"""

import numpy as np
from scipy.integrate import solve_ivp

R_UNIV = 8.314462618  # J/(mol.K)


class H2CompressorRealGasThermal:
    """Multistage real-gas reciprocating H2 compressor with lumped thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        h = params["hydrogen"]

        self.N = int(u["n_stages"]["value"])
        self.n = float(u["polytropic_index"]["value"])
        self.eta_isen = float(u["eta_isentropic"]["value"])
        self.eta_poly = float(u["eta_polytropic"]["value"])
        self.eta_mech = float(u["eta_mech"]["value"])
        self.clearance = float(u["clearance_volume_ratio"]["value"])
        self.T_inlet_default = float(u["T_inlet"]["value"])
        self.P_inlet_default = float(u["P_inlet"]["value"])
        self.P_out_max = float(u["P_outlet_max"]["value"])
        self.eps_ic = float(u["intercooler_effectiveness"]["value"])
        self.T_cool = float(u["T_coolant"]["value"])

        self.m_metal = float(u["m_thermal"]["value"])
        self.cp_metal = float(u["cp_metal"]["value"])
        self.hA_amb = float(u["hA_ambient"]["value"])
        self.T_amb = float(u["T_ambient"]["value"])
        self.f_heat = float(u["frac_heat_to_metal"]["value"])

        self.M = float(h["molar_mass"]["value"])
        self.R_s = float(h["R_specific"]["value"])
        self.gamma = float(h["gamma"]["value"])
        self.cp_H2 = float(h["cp"]["value"])
        self.LHV = float(h["LHV"]["value"])
        self.Tc = float(h["T_crit"]["value"])
        self.Pc = float(h["P_crit"]["value"]) * 1e5  # bar -> Pa

        # Redlich-Kwong molar constants
        self.a_rk = 0.42748 * R_UNIV ** 2 * self.Tc ** 2.5 / self.Pc
        self.b_rk = 0.08664 * R_UNIV * self.Tc / self.Pc

    # ------------------------------------------------------------------
    # Real-gas compressibility factor (Redlich-Kwong)
    # ------------------------------------------------------------------
    def compressibility(self, T, P_bar):
        """H2 compressibility factor Z from Redlich-Kwong EoS (vapour root)."""
        P = float(P_bar) * 1e5  # Pa
        A = self.a_rk * P / (R_UNIV ** 2 * T ** 2.5)
        B = self.b_rk * P / (R_UNIV * T)
        # Z^3 - Z^2 + (A - B - B^2) Z - A B = 0
        coeffs = [1.0, -1.0, (A - B - B ** 2), -A * B]
        roots = np.roots(coeffs)
        real = roots[np.abs(roots.imag) < 1e-9].real
        real = real[real > B]  # physical: Z > B
        if real.size == 0:
            return 1.0
        return float(real.max())  # vapour/supercritical branch

    # ------------------------------------------------------------------
    # Stage pressure ratio (equal split)
    # ------------------------------------------------------------------
    def stage_pressure_ratio(self, P_in, P_out):
        return (float(P_out) / float(P_in)) ** (1.0 / self.N)

    def volumetric_efficiency(self, PR_s):
        """Reciprocating clearance volumetric efficiency (Bloch 2006)."""
        eta_v = 1.0 - self.clearance * (PR_s ** (1.0 / self.n) - 1.0)
        return max(eta_v, 0.0)

    # ------------------------------------------------------------------
    # Stage-by-stage thermodynamic profile (real gas)
    # ------------------------------------------------------------------
    def stage_profile(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """
        Real-gas polytropic profile for every stage.

        Returns a dict of length-N arrays plus scalar aggregates.
        """
        T1 = self.T_inlet_default if T_inlet is None else float(T_inlet)
        T_cool = self.T_cool if T_coolant is None else float(T_coolant)
        eps = self.eps_ic if eps_ic is None else float(eps_ic)

        PRs = self.stage_pressure_ratio(P_in, P_out)
        n = self.n
        g = self.gamma

        T_in_arr = np.zeros(self.N)
        T_disc_arr = np.zeros(self.N)     # polytropic discharge T (pre-cooler)
        T_isen_arr = np.zeros(self.N)     # isentropic discharge T
        T_after_arr = np.zeros(self.N)    # post-intercooler T
        Z_arr = np.zeros(self.N)
        w_poly_arr = np.zeros(self.N)     # reversible polytropic head [J/kg]
        w_stage_arr = np.zeros(self.N)    # actual stage work (poly/eta_poly) [J/kg]
        w_rev_arr = np.zeros(self.N)      # reversible (isentropic) work [J/kg]
        eta_v_arr = np.zeros(self.N)

        # interstage pressures (equal ratio)
        P_lo = float(P_in)
        T_current = T1
        for k in range(self.N):
            P_hi = P_lo * PRs
            Z = self.compressibility(T_current, P_lo)
            Z_arr[k] = Z
            T_in_arr[k] = T_current

            exp_poly = Z * (n - 1.0) / n
            exp_isen = Z * (g - 1.0) / g

            T2 = T_current * PRs ** exp_poly
            T2s = T_current * PRs ** exp_isen
            T_disc_arr[k] = T2
            T_isen_arr[k] = T2s

            # reversible polytropic head and reversible isentropic work
            w_poly = Z * (n / (n - 1.0)) * self.R_s * T_current * (PRs ** exp_poly - 1.0)
            w_rev = Z * (g / (g - 1.0)) * self.R_s * T_current * (PRs ** exp_isen - 1.0)
            w_poly_arr[k] = w_poly
            w_rev_arr[k] = w_rev
            w_stage_arr[k] = w_poly / self.eta_poly
            eta_v_arr[k] = self.volumetric_efficiency(PRs)

            if k < self.N - 1:
                T_after = T2 - eps * (T2 - T_cool)
                T_after_arr[k] = T_after
                T_current = T_after
            else:
                T_after_arr[k] = T2
            P_lo = P_hi

        return {
            "PR_stage": PRs,
            "T_in_stage": T_in_arr,
            "T_discharge": T_disc_arr,
            "T_isentropic": T_isen_arr,
            "T_after_ic": T_after_arr,
            "Z": Z_arr,
            "w_poly_J_kg": w_poly_arr,
            "w_stage_J_kg": w_stage_arr,
            "w_rev_J_kg": w_rev_arr,
            "eta_vol": eta_v_arr,
        }

    # ------------------------------------------------------------------
    # Aggregate steady-state quantities
    # ------------------------------------------------------------------
    def specific_work(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Total specific shaft work [J/kg] incl. mechanical losses."""
        prof = self.stage_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        return prof["w_stage_J_kg"].sum() / self.eta_mech

    def sec_kwh_per_kg(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        return self.specific_work(P_in, P_out, T_inlet, T_coolant, eps_ic) / 3.6e6

    def shaft_power_kw(self, m_dot, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        return float(m_dot) * self.specific_work(P_in, P_out, T_inlet, T_coolant, eps_ic) / 1000.0

    def isentropic_efficiency(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Overall isentropic efficiency = reversible isentropic work / actual work."""
        prof = self.stage_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        w_actual = prof["w_stage_J_kg"].sum() / self.eta_mech
        w_rev = prof["w_rev_J_kg"].sum()
        return w_rev / w_actual if w_actual > 0 else 0.0

    def compression_efficiency(self, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Energy efficiency LHV / (LHV + w_total) on an MJ/kg basis."""
        w_MJ = self.specific_work(P_in, P_out, T_inlet, T_coolant, eps_ic) / 1.0e6
        return self.LHV / (self.LHV + w_MJ)

    def heat_rejected_kw(self, m_dot, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """Total intercooler heat rejection [kW]."""
        prof = self.stage_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        Q = 0.0
        for k in range(self.N - 1):
            Q += float(m_dot) * self.cp_H2 * (prof["T_discharge"][k] - prof["T_after_ic"][k])
        return Q / 1000.0

    def dissipated_heat_w(self, m_dot, P_in, P_out, T_inlet=None, T_coolant=None, eps_ic=None):
        """
        Irreversible heat generation rate [W]: compression irreversibility
        (actual stage work minus reversible isentropic work) + mechanical
        friction. This is the source term for the metal thermal ODE.
        """
        prof = self.stage_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)
        w_stage_sum = prof["w_stage_J_kg"].sum()
        w_rev_sum = prof["w_rev_J_kg"].sum()
        w_friction = (1.0 / self.eta_mech - 1.0) * w_stage_sum
        w_irrev = max(w_stage_sum - w_rev_sum, 0.0)
        return float(m_dot) * (w_irrev + w_friction)

    # ------------------------------------------------------------------
    # Lumped thermal-transient ODE
    # ------------------------------------------------------------------
    def dTmetal_dt(self, T_metal, Q_diss_w):
        """Metal lumped-mass temperature rate [K/s]."""
        Q_in = self.f_heat * Q_diss_w
        Q_out = self.hA_amb * (T_metal - self.T_amb)
        return (Q_in - Q_out) / (self.m_metal * self.cp_metal)

    def simulate(self, m_dot, P_in, P_out, T_inlet=None, T_coolant=None,
                 eps_ic=None, T_metal0=None, dt=10.0, duration_s=1800.0):
        """
        Transient warm-up simulation of the lumped metal temperature plus the
        steady per-stage gas thermodynamics.

        Parameters
        ----------
        m_dot      : float -- mass flow [kg/s] (constant or callable(t))
        P_in,P_out : float -- suction / discharge pressures [bar]
        T_inlet    : float -- first-stage suction T [K]
        T_coolant  : float -- intercooler coolant T [K]
        eps_ic     : float -- intercooler effectiveness
        T_metal0   : float -- initial metal temperature [K] (default = ambient)
        dt         : float -- output step [s]
        duration_s : float -- total horizon [s]

        Returns
        -------
        dict of time series + scalar steady-state diagnostics.
        """
        _m = m_dot if callable(m_dot) else (lambda t: m_dot)
        T0 = self.T_amb if T_metal0 is None else float(T_metal0)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            Q = self.dissipated_heat_w(_m(t), P_in, P_out, T_inlet, T_coolant, eps_ic)
            return [self.dTmetal_dt(y[0], Q)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T0], t_eval=t_eval,
            method="RK45", rtol=1e-7, atol=1e-7, max_step=dt,
        )

        t_out = sol.t
        T_metal = sol.y[0]
        N = len(t_out)

        power_kw = np.zeros(N)
        Q_rej_kw = np.zeros(N)
        Q_diss_kw = np.zeros(N)
        for i in range(N):
            m_i = _m(t_out[i])
            power_kw[i] = self.shaft_power_kw(m_i, P_in, P_out, T_inlet, T_coolant, eps_ic)
            Q_rej_kw[i] = self.heat_rejected_kw(m_i, P_in, P_out, T_inlet, T_coolant, eps_ic)
            Q_diss_kw[i] = self.dissipated_heat_w(m_i, P_in, P_out, T_inlet, T_coolant, eps_ic) / 1000.0

        prof = self.stage_profile(P_in, P_out, T_inlet, T_coolant, eps_ic)

        return {
            "t": t_out,
            "T_metal": T_metal,
            "shaft_power_kW": power_kw,
            "heat_rejected_kW": Q_rej_kw,
            "heat_dissipated_kW": Q_diss_kw,
            "SEC_kWh_kg": self.sec_kwh_per_kg(P_in, P_out, T_inlet, T_coolant, eps_ic),
            "isentropic_efficiency": self.isentropic_efficiency(P_in, P_out, T_inlet, T_coolant, eps_ic),
            "compression_efficiency": self.compression_efficiency(P_in, P_out, T_inlet, T_coolant, eps_ic),
            "T_discharge_final_K": float(prof["T_discharge"][-1]),
            "stage_profile": prof,
        }
