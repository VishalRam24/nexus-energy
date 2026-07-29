"""
EC051 -- Dye-Sensitized Solar Cell (DSSC) -- F2a Physics-Lumped Single-Diode Model

Physics-lumped electrochemical-PV model of a DSSC. The DSSC is treated as a
photoelectrochemical single-diode device:

    I(V) = I_ph(G, T)
           - I0(T) * (exp((V + I*Rs) / (n*Vt)) - 1)        diode recombination at
                                                            the TiO2 / electrolyte interface
           - (V + I*Rs) / Rsh                              shunt leakage

  * I_ph  -- photocurrent from dye (N719) light absorption; I_ph ~ G with a small
             positive temperature coefficient (red-shift of dye absorption).
  * I0    -- dark saturation current describing recombination of injected
             electrons in TiO2 with I3- in the electrolyte; strongly T-dependent
             via the effective bandgap term (Boltzmann statistics).
  * Rs    -- series resistance dominated by the TCO sheet resistance AND the
             ohmic drop of the I3-/I- redox couple diffusing through the
             electrolyte (the DSSC-specific loss).
  * A redox / triiodide DIFFUSION limit (j_diff_lim) caps the deliverable
     current, mimicking I3- mass-transport saturation at high illumination.

DSSC physics specifics (vs. crystalline Si):
  * High ideality factor n ~ 2 (interfacial / trap-assisted recombination).
  * Lower open-circuit voltage, Voc ~ 0.7 V (set by TiO2 conduction band edge
    minus the I3-/I- redox potential, ~0.9 eV; minus recombination losses).
  * Excellent DIFFUSE / LOW-LIGHT performance: Voc falls only logarithmically
    with G, so efficiency stays high indoors / under cloud.

A lumped 0-D thermal ODE (scipy.integrate.solve_ivp) tracks the cell temperature:

    m*cp * dT/dt = Q_abs - P_elec - Q_loss
    Q_abs  = absorptance * G * A            (absorbed solar power)
    P_elec = V_mpp * I_mpp                  (electrical power extracted)
    Q_loss = hA_loss * (T - T_amb)          (convective+radiative loss)

so irradiance and temperature dependence are fully coupled: G drives both the
photocurrent and the heating, and the resulting T feeds back into I0(T) and Voc.

I-V / P-V curves are obtained by a Newton root-find of the implicit diode
equation on a voltage grid; MPP is the maximum of P = V*I.

References:
    O'Regan, B. & Graetzel, M. (1991). A low-cost, high-efficiency solar cell
        based on dye-sensitized colloidal TiO2 films. Nature 353, 737-740.
    Graetzel, M. (2003). Dye-sensitized solar cells.
        J. Photochem. Photobiol. C 4, 145-153.
    Snaith, H. & Schmidt-Mende, L. (2007). Advances in liquid-electrolyte and
        solid-state dye-sensitized solar cells. Adv. Mater. 19, 3187-3200.
    Ni, M. et al. (2006). An analytical study of the porosity effect on
        dye-sensitized solar cell performance.
        Sol. Energy Mater. Sol. Cells 90, 1331-1344.
    Duffie & Beckman (2013). Solar Engineering of Thermal Processes (thermal balance).
"""

import numpy as np
from scipy.integrate import solve_ivp


class DSSC_F2a:
    """Dye-sensitized solar cell -- physics-lumped single-diode + thermal ODE."""

    # Physical constants
    q = 1.602176634e-19   # C, elementary charge
    k_B = 1.380649e-23    # J/K, Boltzmann constant

    def __init__(self, params: dict):
        u = params["unit"]
        self.I_L0 = u["I_L_density_ref"]["value"]       # A/cm2 at G_ref, T_ref
        self.I0_ref = u["I0_density_ref"]["value"]      # A/cm2 at T_ref
        self.n = u["n_ideality"]["value"]               # -
        self.E_g = u["E_g"]["value"]                    # eV
        self.Rs = u["Rs_area"]["value"]                 # Ohm.cm2
        self.Rsh = u["Rsh_area"]["value"]               # Ohm.cm2
        self.j_diff_lim = u["j_diff_lim"]["value"]      # A/cm2
        self.alpha_isc = u["alpha_isc"]["value"]        # 1/K
        self.A = u["A_cell"]["value"]                   # cm2
        self.T_ref = u["T_ref"]["value"]                # K
        self.G_ref = u["G_ref"]["value"]                # W/m2
        self.m_cell = u["m_cell"]["value"]              # kg
        self.cp_cell = u["cp_cell"]["value"]            # J/(kg.K)
        self.hA_loss = u["hA_loss"]["value"]            # W/K
        self.absorptance = u["absorptance"]["value"]    # -
        self.T_amb_default = u["T_amb"]["value"]        # K

    # ------------------------------------------------------------------ #
    # Temperature-dependent physical quantities
    # ------------------------------------------------------------------ #
    def thermal_voltage(self, T):
        """Vt = k_B*T/q  (V)."""
        return self.k_B * T / self.q

    def photocurrent_density(self, G, T):
        """
        Dye-absorption photocurrent density (A/cm2).

        Linear in irradiance (Beer-Lambert dye absorption in optically-thick
        TiO2 film -> J_ph proportional to photon flux ~ G), with a small
        positive temperature coefficient (dye red-shift). Capped by the
        I3-/I- diffusion limit (mass-transport ceiling).
        """
        if G <= 0.0:
            return 0.0
        jL = self.I_L0 * (G / self.G_ref) * (1.0 + self.alpha_isc * (T - self.T_ref))
        jL = max(jL, 0.0)
        # Triiodide diffusion limit: smooth saturation toward j_diff_lim.
        jL = self.j_diff_lim * jL / (jL + self.j_diff_lim) if jL > 0 else 0.0
        return jL

    def saturation_current_density(self, T):
        """
        Diode dark saturation current density (A/cm2) for TiO2/electrolyte
        recombination. Standard single-diode temperature scaling:

            I0(T) = I0_ref * (T/T_ref)^3
                    * exp[ (q*E_g / (n*k_B)) * (1/T_ref - 1/T) ]

        (Boltzmann recombination statistics; cf. single-diode PV temperature
         models, here with the DSSC effective gap E_g and ideality n.)
        """
        Eg_J = self.E_g * self.q
        term = (Eg_J / (self.n * self.k_B)) * (1.0 / self.T_ref - 1.0 / T)
        term = np.clip(term, -100.0, 100.0)
        return self.I0_ref * (T / self.T_ref) ** 3 * np.exp(term)

    # ------------------------------------------------------------------ #
    # Implicit single-diode current at a voltage (Newton)
    # ------------------------------------------------------------------ #
    def _current_density(self, V, jL, I0, nVt):
        """Solve I = jL - I0*(exp((V+I*Rs)/nVt)-1) - (V+I*Rs)/Rsh for I (A/cm2)."""
        V = np.asarray(V, dtype=float)
        J = np.full(V.shape, jL, dtype=float)
        for _ in range(80):
            arg = np.clip((V + J * self.Rs) / nVt, -50.0, 50.0)
            e = np.exp(arg)
            F = J - jL + I0 * (e - 1.0) + (V + J * self.Rs) / self.Rsh
            dF = 1.0 + I0 * e * self.Rs / nVt + self.Rs / self.Rsh
            step = F / dF
            J = J - step
            if np.max(np.abs(step)) < 1e-12:
                break
        return J

    # ------------------------------------------------------------------ #
    # Full I-V / P-V curve and MPP
    # ------------------------------------------------------------------ #
    def iv_curve(self, G, T, n_points=300):
        """
        Compute I-V and P-V curves and the maximum-power point.

        Returns dict with arrays (per cell, total current scaled by area) and
        scalar Voc, Isc, Vmp, Imp, Pmp, FF, eta.
        """
        G = float(G)
        T = float(T)
        nVt = self.n * self.thermal_voltage(T)
        jL = self.photocurrent_density(G, T)        # A/cm2
        I0 = self.saturation_current_density(T)     # A/cm2

        # Zero-irradiance -> dark cell, no power.
        if jL <= 0.0:
            zeros = np.zeros(n_points)
            return {
                "V": np.linspace(0.0, 0.0, n_points),
                "I": zeros, "P": zeros,
                "Voc_V": 0.0, "Isc_A": 0.0,
                "Vmp_V": 0.0, "Imp_A": 0.0, "Pmp_W": 0.0,
                "FF": 0.0, "eta": 0.0,
                "jL_A_cm2": 0.0, "I0_A_cm2": I0, "T_K": T,
            }

        # Analytic Voc estimate (ignoring Rsh) to bound the voltage grid.
        Voc_est = nVt * np.log(jL / I0 + 1.0)
        V_max = min(max(Voc_est * 1.15, 0.05), 1.2)
        V_pts = np.linspace(0.0, V_max, n_points)

        J_arr = self._current_density(V_pts, jL, I0, nVt)  # A/cm2

        # Open-circuit voltage = where I crosses zero (interpolate).
        Voc = self._find_voc(V_pts, J_arr)

        mask = V_pts <= Voc
        V_c = V_pts[mask]
        J_c = np.clip(J_arr[mask], 0.0, None)
        P_c = V_c * J_c                                      # W/cm2

        Isc = float(self._current_density(np.array([0.0]), jL, I0, nVt)[0])
        Isc = max(Isc, 0.0)

        idx = int(np.argmax(P_c)) if P_c.size else 0
        Vmp = float(V_c[idx]) if V_c.size else 0.0
        Imp = float(J_c[idx]) if J_c.size else 0.0
        Pmp = float(P_c[idx]) if P_c.size else 0.0          # W/cm2

        FF = Pmp / (Voc * Isc) if (Voc * Isc) > 0 else 0.0
        # Efficiency = electrical power out / incident solar power on the cell.
        P_in = (G / 1.0e4)                                   # W/cm2  (G W/m2 -> W/cm2)
        eta = Pmp / P_in if P_in > 0 else 0.0

        return {
            "V": V_c,
            "I": J_c * self.A,
            "P": P_c * self.A,
            "Voc_V": Voc,
            "Isc_A": Isc * self.A,
            "Vmp_V": Vmp,
            "Imp_A": Imp * self.A,
            "Pmp_W": Pmp * self.A,
            "FF": FF,
            "eta": eta,
            "jL_A_cm2": jL,
            "I0_A_cm2": I0,
            "T_K": T,
        }

    @staticmethod
    def _find_voc(V_pts, J_arr):
        """Linear-interpolate the voltage at which current first crosses zero."""
        sign = np.sign(J_arr)
        crossings = np.where(np.diff(sign) != 0)[0]
        if crossings.size > 0:
            i = crossings[0]
            denom = (J_arr[i + 1] - J_arr[i])
            if abs(denom) < 1e-30:
                return float(V_pts[i])
            Voc = V_pts[i] - J_arr[i] * (V_pts[i + 1] - V_pts[i]) / denom
            return float(np.clip(Voc, 0.0, V_pts[-1]))
        return float(V_pts[int(np.argmin(np.abs(J_arr)))])

    def mpp_power(self, G, T):
        """Convenience: maximum-power-point power (W) at (G, T)."""
        return self.iv_curve(G, T)["Pmp_W"]

    # ------------------------------------------------------------------ #
    # Lumped thermal ODE (scipy.solve_ivp)
    # ------------------------------------------------------------------ #
    def simulate(self, G, T0=None, T_amb=None, dt=2.0, duration_s=600.0):
        """
        Dynamic simulation of the lumped thermal ODE coupled to the I-V model.

        Energy balance:
            m*cp dT/dt = absorptance*G*A_m2 - P_elec(G,T) - hA_loss*(T - T_amb)

        G may be a scalar (W/m2) or a callable G(t).

        Returns time series of T, Voc, Isc, Pmp, eta plus the final I-V curve.
        """
        if T0 is None:
            T0 = self.T_amb_default
        if T_amb is None:
            T_amb = self.T_amb_default

        G_fn = G if callable(G) else (lambda t: float(G))
        A_m2 = self.A * 1.0e-4   # cm2 -> m2 for the absorbed-irradiance term

        def rhs(t, y):
            T = float(y[0])
            T = min(max(T, 200.0), 400.0)
            Gt = max(G_fn(t), 0.0)
            Q_abs = self.absorptance * Gt * A_m2          # W
            # Electrical power removed from the thermal balance.
            P_elec = self.iv_curve(Gt, T)["Pmp_W"]        # W
            Q_loss = self.hA_loss * (T - T_amb)           # W
            dTdt = (Q_abs - P_elec - Q_loss) / (self.m_cell * self.cp_cell)
            return [dTdt]

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        sol = solve_ivp(rhs, (0.0, duration_s), [T0], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-6, max_step=dt)

        T_arr = sol.y[0]
        t_arr = sol.t

        Voc = np.zeros_like(t_arr)
        Isc = np.zeros_like(t_arr)
        Pmp = np.zeros_like(t_arr)
        eta = np.zeros_like(t_arr)
        for i, (ti, Ti) in enumerate(zip(t_arr, T_arr)):
            r = self.iv_curve(max(G_fn(ti), 0.0), Ti)
            Voc[i] = r["Voc_V"]
            Isc[i] = r["Isc_A"]
            Pmp[i] = r["Pmp_W"]
            eta[i] = r["eta"]

        final = self.iv_curve(max(G_fn(t_arr[-1]), 0.0), T_arr[-1])

        return {
            "t": t_arr,
            "temperature": T_arr,
            "Voc": Voc,
            "Isc": Isc,
            "Pmp": Pmp,
            "efficiency": eta,
            "iv_curve": final,
        }
