"""
EC049 -- Multi-Junction Concentrator PV (CPV) -- F2a Physics-Lumped Model

Series multi-junction (GaInP / GaInAs / Ge) single-diode model with current
matching, logarithmic concentration boost, and a lumped cell thermal ODE under
high heat flux + active cooling.

PHYSICS
-------
Each subcell j is a single-diode device:
    J_j(V_j) = Jsc_j - J0_j * (exp(q*V_j / (n_j*k*T)) - 1)
where Jsc_j is the (concentration-scaled) photocurrent density and J0_j the
reverse saturation current density.

Concentration scaling (King 2007, Kurtz 2008):
    Jsc_j(C) = C * Jsc_j(1-sun)                      (photocurrent ~ linear in C)
    Voc_j(C) = Voc_j(1-sun) + (n_j k T / q) * ln(C)  (logarithmic Voc gain)
The ln(C) Voc rise falls directly out of the diode equation: at open circuit
Jsc = J0*exp(qVoc/nkT), so multiplying Jsc by C adds (nkT/q)ln(C) to Voc.

Current matching (series constraint):
    The subcells carry a COMMON current J. The string short-circuit current is
    set by the LIMITING (smallest-Jsc) subcell. Excess photocurrent in the
    current-rich subcells (here Ge) is wasted. The cell terminal voltage is the
    SUM of subcell voltages at the shared operating current:
        V_cell(J) = sum_j  V_j(J)
    found by inverting each subcell diode equation:
        V_j(J) = (n_j k T / q) * ln((Jsc_j - J) / J0_j + 1)
    minus a lumped series-resistance drop  J * Rs.

Maximum power point is obtained by sweeping J in [0, Jsc_limit] and maximising
P = V_cell(J) * J * A_cell.

Cell thermal ODE (lumped 0-D, active cooling):
    m*cp dT/dt = Q_abs - P_elec - Q_cool
    Q_abs  = optical_efficiency * DNI * C * A_cell    (concentrated optical input)
    P_elec = electrical power extracted at MPP
    Q_cool = hA_cool * (T - T_coolant)
Integrated with scipy.integrate.solve_ivp. High flux (hundreds of suns) makes
the active-cooling term essential -- without it the junction would run away.

REFERENCES
----------
King, R.R. et al. (2007). "Advances in High-Efficiency III-V Multijunction
    Solar Cells." Advances in OptoElectronics, 2007, 29523.
Kurtz, S. et al. (2008). "Considerations for measuring multijunction CPV cells."
    AIP Conf. Proc. (concentration scaling of Jsc / Voc).
Cotal, H. et al. (2009). "III-V multijunction solar cells for concentrating
    photovoltaics." Energy & Environ. Sci. 2, 174-192 (optics, thermal).
Friedman, D.J. (2010). "Progress and challenges for next-generation high-
    efficiency multijunction solar cells." Curr. Opin. Solid State Mater. Sci.
"""

import numpy as np
from scipy.integrate import solve_ivp


class MultiJunctionCPV_F2a:
    """Series multi-junction CPV cell with current matching + thermal ODE."""

    # Physical constants
    q = 1.602176634e-19      # C
    k = 1.380649e-23         # J/K
    G_ref = 1000.0           # W/m2, 1-sun DNI reference

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_junctions = int(u["n_junctions"]["value"])
        self.A_cell = u["A_cell"]["value"]                 # m2
        self.C_design = u["C_design"]["value"]             # suns
        self.eta_opt = u["optical_efficiency"]["value"]
        self.T_ref = u["T_ref"]["value"]                   # K
        self.Rs = u["Rs"]["value"]                         # ohm.cm2
        self.beta_Jsc = u["beta_Jsc"]["value"]             # 1/K
        self.m_cell = u["m_cell"]["value"]                 # kg
        self.cp_cell = u["cp_cell"]["value"]               # J/(kg.K)
        self.hA_cool = u["hA_cool"]["value"]               # W/K
        self.T_coolant = u["T_coolant"]["value"]           # K

        sub = u["subcells"]["value"]
        # mA/cm2 -> A/cm2 for Jsc; J0 already A/cm2
        self.Jsc_ref = np.array([s["Jsc_1sun_ref"] * 1e-3 for s in sub])  # A/cm2
        self.J0_ref = np.array([s["J0_ref"] for s in sub])                # A/cm2
        self.n_diode = np.array([s["n_diode"] for s in sub])
        self.Eg = np.array([s["Eg_eV"] for s in sub])                     # eV
        self.names = [s["name"] for s in sub]

    # ------------------------------------------------------------------
    # Concentration ratio from DNI
    # ------------------------------------------------------------------
    def concentration(self, dni):
        """Effective concentration ratio C = (DNI/G_ref) * C_design [suns]."""
        dni = np.asarray(dni, dtype=float)
        return np.maximum(dni / self.G_ref, 0.0) * self.C_design

    # ------------------------------------------------------------------
    # Per-subcell concentration- & temperature-scaled photocurrent
    # ------------------------------------------------------------------
    def subcell_Jsc(self, C, T):
        """Photocurrent density of each subcell [A/cm2]. Shape (n_junctions,)."""
        temp_corr = 1.0 + self.beta_Jsc * (T - self.T_ref)
        return C * self.Jsc_ref * temp_corr

    def subcell_J0(self, T):
        """Temperature-scaled saturation current density [A/cm2].
        J0 ~ T^3 exp(-Eg q / (n k T)) (diode dark-current scaling, King 2007)."""
        T = float(T)
        ratio = T / self.T_ref
        Eg_J = self.Eg * self.q
        exp_arg = (Eg_J / (self.n_diode * self.k)) * (1.0 / self.T_ref - 1.0 / T)
        return self.J0_ref * ratio**3 * np.exp(exp_arg)

    # ------------------------------------------------------------------
    # Subcell voltage at a shared series current
    # ------------------------------------------------------------------
    def _Vt(self, T):
        return self.k * T / self.q  # thermal voltage [V]

    def subcell_voltage(self, J, Jsc, J0, T):
        """Invert single-diode eqn for subcell voltage at current J [A/cm2].
        V_j = n_j Vt ln((Jsc_j - J)/J0_j + 1). Clipped at J=Jsc (V->Voc)."""
        Vt = self._Vt(T)
        avail = np.maximum(Jsc - J, 0.0)
        arg = avail / J0 + 1.0
        return self.n_diode * Vt * np.log(arg)

    def cell_voltage(self, J, C, T):
        """Total cell terminal voltage at shared current density J [A/cm2].
        Sum of subcell voltages minus lumped Rs drop. J in A/cm2, Rs in ohm.cm2."""
        Jsc = self.subcell_Jsc(C, T)
        J0 = self.subcell_J0(T)
        Vsub = self.subcell_voltage(J, Jsc, J0, T)
        V = np.sum(Vsub) - J * self.Rs
        return V

    def Voc(self, C, T):
        """Open-circuit voltage: J=0, sum of subcell Voc (no Rs drop)."""
        if C <= 0:
            return 0.0
        Jsc = self.subcell_Jsc(C, T)
        J0 = self.subcell_J0(T)
        Vsub = self.n_diode * self._Vt(T) * np.log(Jsc / J0 + 1.0)
        return float(np.sum(Vsub))

    def limiting_current(self, C, T):
        """Series short-circuit current density = min subcell Jsc [A/cm2]."""
        Jsc = self.subcell_Jsc(C, T)
        return float(np.min(Jsc))

    def limiting_index(self, C, T):
        Jsc = self.subcell_Jsc(C, T)
        return int(np.argmin(Jsc))

    # ------------------------------------------------------------------
    # Maximum power point (steady-state, by current sweep)
    # ------------------------------------------------------------------
    def mpp(self, dni, T, n_pts=400):
        """Maximum power point at given DNI [W/m2] and junction T [K].

        Returns dict with i_mp [A], v_mp [V], p_mp [W], j_sc [A/cm2],
        v_oc [V], fill_factor, efficiency, concentration.
        """
        C = float(self.concentration(dni))
        if C <= 0.0:
            return {"i_mp": 0.0, "v_mp": 0.0, "p_mp": 0.0, "j_sc": 0.0,
                    "v_oc": 0.0, "fill_factor": 0.0, "efficiency": 0.0,
                    "concentration": 0.0, "T_cell": T,
                    "limiting_subcell": None}

        J_lim = self.limiting_current(C, T)        # A/cm2
        v_oc = self.Voc(C, T)

        # Sweep series current from 0 to just below the limiting Jsc
        J_sweep = np.linspace(0.0, J_lim * (1.0 - 1e-6), n_pts)
        V_sweep = np.array([self.cell_voltage(j, C, T) for j in J_sweep])
        V_sweep = np.maximum(V_sweep, 0.0)
        A_cm2 = self.A_cell * 1e4                  # m2 -> cm2
        P_sweep = V_sweep * (J_sweep * A_cm2)      # W
        idx = int(np.argmax(P_sweep))

        v_mp = float(V_sweep[idx])
        i_mp = float(J_sweep[idx] * A_cm2)
        p_mp = float(P_sweep[idx])

        i_sc = J_lim * A_cm2
        ff = p_mp / (v_oc * i_sc) if (v_oc * i_sc) > 0 else 0.0

        # Efficiency on optical input delivered to the cell.
        # Concentrated irradiance on the cell = C * G_ref [W/m2] (C in suns).
        P_opt = self.eta_opt * (C * self.G_ref) * self.A_cell   # W
        eta = p_mp / P_opt if P_opt > 0 else 0.0

        return {
            "i_mp": i_mp, "v_mp": v_mp, "p_mp": p_mp,
            "j_sc": J_lim, "i_sc": i_sc, "v_oc": v_oc,
            "fill_factor": ff, "efficiency": eta,
            "concentration": C, "T_cell": float(T),
            "limiting_subcell": self.names[self.limiting_index(C, T)],
        }

    # ------------------------------------------------------------------
    # Lumped thermal ODE with active cooling
    # ------------------------------------------------------------------
    def _heat_balance(self, T, dni):
        """Net heat rate into the lumped cell mass [W] at junction temp T."""
        C = float(self.concentration(dni))
        # Concentrated optical flux on cell = C * G_ref [W/m2] (C in suns)
        Q_abs = self.eta_opt * (C * self.G_ref) * self.A_cell   # absorbed optical W
        if C > 0:
            mpp = self.mpp(dni, T)
            P_elec = mpp["p_mp"]
        else:
            P_elec = 0.0
        Q_cool = self.hA_cool * (T - self.T_coolant)
        return Q_abs - P_elec - Q_cool, P_elec, Q_abs, Q_cool

    def simulate(self, dni, T0=None, dt=1.0, duration_s=120.0):
        """Transient junction temperature + MPP via scipy.solve_ivp.

        dni : float (constant) or callable dni(t) [W/m2]
        T0  : initial junction temperature [K] (default = coolant temp)
        Returns time series dict.
        """
        if T0 is None:
            T0 = self.T_coolant
        dni_fn = dni if callable(dni) else (lambda t: dni)

        def rhs(t, y):
            T = y[0]
            g = float(dni_fn(t))
            qnet, _, _, _ = self._heat_balance(T, g)
            return [qnet / (self.m_cell * self.cp_cell)]

        t_eval = np.arange(0.0, duration_s + 1e-9, dt)
        sol = solve_ivp(rhs, (0.0, duration_s), [T0], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-6, max_step=dt)

        T_arr = sol.y[0]
        t_arr = sol.t

        # Post-process electrical quantities at each step
        p_mp = np.zeros_like(t_arr)
        v_mp = np.zeros_like(t_arr)
        i_mp = np.zeros_like(t_arr)
        v_oc = np.zeros_like(t_arr)
        eta = np.zeros_like(t_arr)
        C_arr = np.zeros_like(t_arr)
        for i, (t, T) in enumerate(zip(t_arr, T_arr)):
            g = float(dni_fn(t))
            r = self.mpp(g, T)
            p_mp[i] = r["p_mp"]
            v_mp[i] = r["v_mp"]
            i_mp[i] = r["i_mp"]
            v_oc[i] = r["v_oc"]
            eta[i] = r["efficiency"]
            C_arr[i] = r["concentration"]

        return {
            "t": t_arr,
            "temperature": T_arr,
            "p_mp": p_mp,
            "v_mp": v_mp,
            "i_mp": i_mp,
            "v_oc": v_oc,
            "efficiency": eta,
            "concentration": C_arr,
        }
