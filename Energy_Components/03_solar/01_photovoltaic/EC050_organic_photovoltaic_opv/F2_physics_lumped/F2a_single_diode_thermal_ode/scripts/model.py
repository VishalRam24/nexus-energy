"""
EC050 -- Organic Photovoltaic (OPV) -- F2a Physics-Lumped Single-Diode + Thermal ODE

Physics-lumped (0D) first-principles model of an OPV module string.

1. Single-diode (Shockley) device equation, solved implicitly for the full I-V
   and P-V curve by root-finding (no curve-fit / no lookup):

       I(V) = I_L - I0 * (exp((V + I*Rs) / (n * Vt_string)) - 1) - (V + I*Rs) / Rsh

   where Vt_string = N_s * k * T / q  is the thermal voltage of the series string.
   OPV specifics:
     - high ideality factor n ~ 1.7 (trap-assisted / charge-transfer-state
       recombination -> "soft" diode knee),
     - significant series resistance Rs and finite shunt resistance Rsh
       -> low fill factor (FF ~ 0.5-0.6, vs ~0.8 for c-Si).

2. Temperature & irradiance dependence of the single-diode parameters
   (De Soto et al. 2006):
       I_L(G,T) = (G/G_ref) * (I_L_ref + alpha_sc * (T - T_ref))
       I0(T)    = I0_ref * (T/T_ref)^3 * exp( Eg_ref/(k/q * T_ref) - Eg(T)/(k/q * T) )
       Eg(T)    = Eg_ref + dEgdT * (T - T_ref)
       Rsh(G)   = Rsh_ref * (G_ref / G)        (shunt grows at low light)

3. Maximum power point (MPP) by golden-section search of P(V)=V*I(V) on [0, Voc].

4. Lumped thermal ODE (energy balance over the module, Faiman 2008 loss model),
   integrated with scipy.integrate.solve_ivp:

       (m*cp) dT/dt = absorptance * G * area - P_elec(G,T) - U_loss * area * (T - T_amb)

   i.e. absorbed solar heat minus extracted electrical power minus convective+
   radiative losses. OPV's lightweight flexible substrate gives a small m*cp,
   hence a fast thermal time constant.

OPV behavioural notes captured by these parameters:
   - low conversion efficiency (~ a few %; bounded < 12 % in tests),
   - excellent low-light / indoor performance (shunt-limited regime),
   - flexible / low thermal mass,
   - fast degradation (flagged in parameters.json; not integrated -- F2c scope).

References:
    Shockley, W. (1949). The theory of p-n junctions in semiconductors.
        Bell Syst. Tech. J. 28(3), 435-489.
    De Soto, W., Klein, S.A., Beckman, W.A. (2006). Improvement and validation
        of a model for photovoltaic array performance. Solar Energy 80(1), 78-88.
    Faiman, D. (2008). Assessing the outdoor operating temperature of PV modules.
        Prog. Photovolt. 16(4), 307-315.
    Brabec, C.J. et al. (2010). Polymer-fullerene bulk-heterojunction solar cells.
        Adv. Mater. 22, 3839-3856.
    Cuce, E. et al. (2013). Effects of low irradiance on OPV performance.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Physical constants
_K_B = 1.380649e-23      # Boltzmann constant [J/K]
_Q = 1.602176634e-19     # Elementary charge [C]
_KB_OVER_Q = _K_B / _Q   # [V/K]


class OPV_F2a:
    """OPV single-diode device with lumped thermal ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.I_L_ref = u["I_L_ref"]["value"]      # A
        self.I0_ref = u["I0_ref"]["value"]        # A
        self.n = u["n"]["value"]                  # -
        self.N_s = u["N_s"]["value"]              # -
        self.Rs = u["Rs"]["value"]                # Ohm
        self.Rsh_ref = u["Rsh"]["value"]          # Ohm
        self.alpha_sc = u["alpha_sc"]["value"]    # A/K
        self.Eg_ref = u["Eg_ref"]["value"]        # eV
        self.dEgdT = u["dEgdT"]["value"]          # eV/K
        self.area = u["area"]["value"]            # m2
        self.T_ref = u["T_ref"]["value"]          # K
        self.G_ref = u["G_ref"]["value"]          # W/m2
        self.absorptance = u["absorptance"]["value"]
        self.m_cp = u["m_cp"]["value"]            # J/K
        self.U_loss = u["U_loss"]["value"]        # W/(m2.K)

    # ------------------------------------------------------------------
    # Single-diode parameters at given irradiance G [W/m2] and temp T [K]
    # ------------------------------------------------------------------
    def diode_params(self, G, T):
        """Return (I_L, I0, a, Rsh) where a = n * N_s * k * T / q [V]."""
        G = float(G)
        T = float(T)
        I_L = (G / self.G_ref) * (self.I_L_ref + self.alpha_sc * (T - self.T_ref))
        I_L = max(I_L, 0.0)

        Eg = self.Eg_ref + self.dEgdT * (T - self.T_ref)
        I0 = (self.I0_ref * (T / self.T_ref) ** 3
              * np.exp(self.Eg_ref / (_KB_OVER_Q * self.T_ref)
                       - Eg / (_KB_OVER_Q * T)))
        I0 = max(I0, 1e-30)

        a = self.n * self.N_s * _KB_OVER_Q * T   # thermal voltage of the string [V]

        # Shunt resistance grows at low light (De Soto): Rsh = Rsh_ref * G_ref/G
        Rsh = self.Rsh_ref * (self.G_ref / max(G, 1.0))
        return I_L, I0, a, Rsh

    # ------------------------------------------------------------------
    # Implicit single-diode solve I(V) by Newton root-finding
    # ------------------------------------------------------------------
    def current_from_voltage(self, V, G, T, params=None, n_iter=30, tol=1e-12):
        """Solve the Shockley single-diode equation for I at voltage V [V]."""
        if params is None:
            params = self.diode_params(G, T)
        I_L, I0, a, Rsh = params
        V = np.asarray(V, dtype=float)
        I = np.full_like(V, I_L)  # warm start at short-circuit
        for _ in range(n_iter):
            arg = np.clip((V + I * self.Rs) / a, -50.0, 50.0)
            exp_term = np.exp(arg)
            f = I_L - I0 * (exp_term - 1.0) - (V + I * self.Rs) / Rsh - I
            df = -I0 * exp_term * (self.Rs / a) - self.Rs / Rsh - 1.0
            step = f / df
            I = I - step
            if np.max(np.abs(step)) < tol:
                break
        return I

    def open_circuit_voltage(self, G, T, params=None, n_iter=40, tol=1e-12):
        """Voc: solve I(V)=0 (Newton on the V-axis with I=0)."""
        if params is None:
            params = self.diode_params(G, T)
        I_L, I0, a, Rsh = params
        if I_L <= 0.0:
            return 0.0
        # initial guess: ideal-diode Voc ignoring shunt
        V = a * np.log(I_L / I0 + 1.0)
        for _ in range(n_iter):
            arg = np.clip(V / a, -50.0, 50.0)
            exp_term = np.exp(arg)
            f = I_L - I0 * (exp_term - 1.0) - V / Rsh        # I at this V (Rs irrelevant at I=0)
            df = -I0 * exp_term / a - 1.0 / Rsh
            step = f / df
            V = V - step
            if abs(step) < tol:
                break
        return float(max(V, 0.0))

    # ------------------------------------------------------------------
    # Full I-V / P-V curve
    # ------------------------------------------------------------------
    def iv_curve(self, G, T, n_points=200):
        """Full I-V and P-V curve from V=0 to Voc. Returns dict of arrays."""
        params = self.diode_params(G, T)
        Voc = self.open_circuit_voltage(G, T, params)
        if Voc <= 0.0:
            z = np.zeros(n_points)
            return {"V": z, "I": z.copy(), "P": z.copy(), "Voc": 0.0}
        V = np.linspace(0.0, Voc, n_points)
        I = np.maximum(self.current_from_voltage(V, G, T, params), 0.0)
        P = V * I
        return {"V": V, "I": I, "P": P, "Voc": Voc}

    # ------------------------------------------------------------------
    # Maximum power point -- golden-section search on P(V)
    # ------------------------------------------------------------------
    def mpp(self, G, T):
        """Maximum power point. Returns dict: Voc, Isc, Vmp, Imp, Pmp, FF, eta."""
        params = self.diode_params(G, T)
        Voc = self.open_circuit_voltage(G, T, params)
        Isc = float(max(self.current_from_voltage(0.0, G, T, params), 0.0))

        if Voc <= 0.0 or G <= 0.0:
            return {"Voc": 0.0, "Isc": 0.0, "Vmp": 0.0, "Imp": 0.0,
                    "Pmp": 0.0, "FF": 0.0, "eta": 0.0}

        gr = (np.sqrt(5.0) - 1.0) / 2.0
        lo, hi = 0.0, Voc
        v1 = hi - gr * (hi - lo)
        v2 = lo + gr * (hi - lo)
        p1 = v1 * float(self.current_from_voltage(v1, G, T, params))
        p2 = v2 * float(self.current_from_voltage(v2, G, T, params))
        for _ in range(50):
            if (hi - lo) < 1e-6 * Voc:
                break
            if p1 < p2:
                lo, v1, p1 = v1, v2, p2
                v2 = lo + gr * (hi - lo)
                p2 = v2 * float(self.current_from_voltage(v2, G, T, params))
            else:
                hi, v2, p2 = v2, v1, p1
                v1 = hi - gr * (hi - lo)
                p1 = v1 * float(self.current_from_voltage(v1, G, T, params))
        Vmp = 0.5 * (lo + hi)
        Imp = float(max(self.current_from_voltage(Vmp, G, T, params), 0.0))
        Pmp = Vmp * Imp

        FF = Pmp / (Voc * Isc) if (Voc * Isc) > 0 else 0.0
        P_in = G * self.area
        eta = Pmp / P_in if P_in > 0 else 0.0
        return {"Voc": Voc, "Isc": Isc, "Vmp": Vmp, "Imp": Imp,
                "Pmp": Pmp, "FF": FF, "eta": eta}

    def power_at_mpp(self, G, T):
        """Electrical power [W] extracted at the MPP (used by the thermal ODE)."""
        return self.mpp(G, T)["Pmp"]

    # ------------------------------------------------------------------
    # Lumped thermal ODE (energy balance), integrated with solve_ivp
    # ------------------------------------------------------------------
    def dTdt(self, T, G, T_amb_K):
        """Temperature rate [K/s] from the lumped energy balance.

        (m*cp) dT/dt = absorptance*G*area - P_elec - U_loss*area*(T - T_amb)
        """
        Q_abs = self.absorptance * G * self.area
        P_elec = self.power_at_mpp(G, T)
        Q_loss = self.U_loss * self.area * (T - T_amb_K)
        return (Q_abs - P_elec - Q_loss) / self.m_cp

    def steady_state_temperature(self, G, T_amb_K, tol=1e-4, max_iter=200):
        """Fixed-point steady-state cell temperature (dT/dt = 0)."""
        T = T_amb_K + 5.0
        for _ in range(max_iter):
            Q_abs = self.absorptance * G * self.area
            P_elec = self.power_at_mpp(G, T)
            T_new = T_amb_K + (Q_abs - P_elec) / (self.U_loss * self.area)
            if abs(T_new - T) < tol:
                return T_new
            T = T_new
        return T

    def simulate(self, irradiance, T_amb_C, T0_C=None, dt=5.0, duration_s=600.0):
        """
        Transient simulation of cell temperature and electrical output.

        Parameters
        ----------
        irradiance : float or callable(t)->G [W/m2]
        T_amb_C    : float or callable(t)->T_amb [degC]
        T0_C       : float, initial cell temperature [degC] (default = T_amb(0))
        dt         : output time step [s]
        duration_s : total duration [s]

        Returns
        -------
        dict of time-series arrays: t, T_cell_C, Voc, Isc, Vmp, Imp,
            power (Pmp [W]), efficiency, FF, G, T_amb_C
        """
        G_fn = irradiance if callable(irradiance) else (lambda t: irradiance)
        Ta_fn = T_amb_C if callable(T_amb_C) else (lambda t: T_amb_C)

        if T0_C is None:
            T0_C = Ta_fn(0.0)
        T0_K = T0_C + 273.15

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            T = y[0]
            G = float(G_fn(t))
            Ta_K = float(Ta_fn(t)) + 273.15
            return [self.dTdt(T, G, Ta_K)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T0_K],
            t_eval=t_eval, method="RK45", rtol=1e-5, atol=1e-7,
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        Voc = np.zeros(N); Isc = np.zeros(N); Vmp = np.zeros(N)
        Imp = np.zeros(N); power = np.zeros(N); eff = np.zeros(N)
        ff = np.zeros(N); G_arr = np.zeros(N); Ta_arr = np.zeros(N)

        for i in range(N):
            G = float(G_fn(t_out[i]))
            m = self.mpp(G, T_out[i])
            Voc[i] = m["Voc"]; Isc[i] = m["Isc"]; Vmp[i] = m["Vmp"]
            Imp[i] = m["Imp"]; power[i] = m["Pmp"]; eff[i] = m["eta"]
            ff[i] = m["FF"]; G_arr[i] = G; Ta_arr[i] = float(Ta_fn(t_out[i]))

        return {
            "t": t_out,
            "T_cell_C": T_out - 273.15,
            "Voc": Voc, "Isc": Isc, "Vmp": Vmp, "Imp": Imp,
            "power": power, "efficiency": eff, "FF": ff,
            "G": G_arr, "T_amb_C": Ta_arr,
        }
