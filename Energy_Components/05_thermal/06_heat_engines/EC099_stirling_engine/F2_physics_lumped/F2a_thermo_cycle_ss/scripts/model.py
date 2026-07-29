"""
EC099 -- Stirling Engine -- F2a Physics-Lumped Ideal-Cycle Model

First-principles (0D) Stirling-cycle model with a lumped warm-up ODE.

CYCLE THERMODYNAMICS -- Schmidt isothermal analysis
---------------------------------------------------
The Schmidt (1871) closed-form solution assumes sinusoidal volume
variation, isothermal expansion/compression spaces, an ideal (perfect-
mixing) regenerator and an ideal gas of fixed total mass. For an
alpha / gamma machine the instantaneous total gas mass is constant, so
the instantaneous pressure is

    p(theta) = p_mean * sqrt(1 - b^2) / (1 - b*cos(theta - phi))

with the Schmidt grouping (Urieli & Berchowitz 1984, Ch. 3):

    tau   = T_c / T_h                       (cold/hot temperature ratio)
    s     = V_swept_comp / V_swept_exp      (swept-volume ratio, kappa)
    X_DV  = dead-volume ratio  (dead/ swept)
    S     = tau + 2*tau*X_DV*... + s   (dimensionless dead-volume term)
    b     = sqrt(tau^2 + 2*tau*s*cos(alpha) + s^2) / (S)   (pressure-swing
            amplitude ratio)
    phi   = phase of the pressure wave.

The indicated work per cycle of the EXPANSION (hot) space is the closed
contour integral W_e = oint p dV_e, which Schmidt evaluates in closed
form to

    W_e = p_mean * V_swept_exp * pi * (1 - tau) * b * sin(phi)
          / (1 + sqrt(1 - b^2))                                   [J/cycle]

and the net indicated cycle work is W_ind = W_e * (1 - tau) grouping
(both spaces).  Indicated power = W_ind * (rev/s).

IDEAL THERMAL EFFICIENCY
------------------------
With a PERFECT regenerator the Schmidt cycle efficiency equals Carnot,
eta_id = 1 - tau.  A regenerator of effectiveness eps < 1 leaves an
un-recovered re-heat duty each cycle that must be supplied externally,
lowering efficiency to (Walker 1980; Urieli & Berchowitz 1984):

    eta_regen = (1 - tau)
              / (1 + (1-eps) * (gamma-1) * ln(...) term / s_work)

We implement the standard regenerator-loss penalty: the external heat
that must replace the un-recovered regenerator duty is
    Q_regen_loss = (1 - eps) * m_gas * cv * (T_h - T_c)  per cycle,
added to the ideal heater duty Q_e.  This always gives eta < 1 - tau
< Carnot, enforcing the second law.

LOSSES (to brake / electrical power)
------------------------------------
    P_indicated -> minus pumping (gas-circuit flow) loss fraction
                -> times mechanical efficiency (friction/windage)
    P_brake = mech_eff * (1 - pumping_coeff) * P_indicated

BEALE / WEST NUMBER CROSS-CHECK
-------------------------------
Empirical similitude (Walker 1980; West 1986):
    P_beale = B * p_mean * V_swept_exp * freq
The West number variant scales by (T_h - T_c)/(T_h + T_c).  Used only as
an independent sanity check on the Schmidt indicated power.

LUMPED WARM-UP ODE  (scipy.solve_ivp)
-------------------------------------
Heater-head temperature transient from cold start:
    C_th * dT_h/dt = Q_burner - UA_gas*(T_h - T_gas_mean) - UA_loss*(T_h - T_amb)
integrated with RK45.  Steady state -> the design hot-end temperature.

References
----------
    Schmidt G. (1871), Z. Ver. Dtsch. Ing. 15, 1-12.
    Urieli I. & Berchowitz D.M. (1984), Stirling Cycle Engine Analysis,
        Adam Hilger, Bristol.
    Walker G. (1980), Stirling Engines, Oxford University Press.
    West C.D. (1986), Principles and Applications of Stirling Engines, Van Nostrand.
"""

import numpy as np
from scipy.integrate import solve_ivp

R_UNIV = 8.314462618  # J/(mol.K)


class StirlingEngineF2a:
    """Physics-lumped Stirling engine: Schmidt ideal cycle + warm-up ODE."""

    def __init__(self, params: dict):
        e = params["engine"]
        self.P_rated      = e["P_rated"]["value"]
        self.f_carnot     = e["f_carnot"]["value"]
        self.V_exp        = e["V_swept_exp"]["value"]        # m3
        self.V_comp       = e["V_swept_comp"]["value"]       # m3
        self.kappa        = e["kappa"]["value"]              # V_comp/V_exp
        self.phase_rad    = np.deg2rad(e["phase_deg"]["value"])
        self.X_dead       = e["dead_volume_ratio"]["value"]
        self.regen_eff    = e["regen_eff"]["value"]
        self.n_rpm        = e["n_rpm"]["value"]
        self.p_mean       = e["p_mean"]["value"]             # Pa
        self.T_h          = e["T_h"]["value"]                # K
        self.T_c          = e["T_c"]["value"]                # K
        self.mech_eff     = e["mech_eff"]["value"]
        self.pump_coeff   = e["pumping_coeff"]["value"]
        self.beale_number = e["beale_number"]["value"]
        self.gas_name     = e["working_gas"]["value"]

        g = params["gas_properties"][self.gas_name]
        self.R_gas  = g["R_specific"]   # J/(kg.K)
        self.gamma  = g["gamma"]
        self.M_gas  = g["M"]            # kg/mol
        self.cv     = self.R_gas / (self.gamma - 1.0)   # J/(kg.K)

        th = params["thermal"]
        self.C_th     = th["C_th_head"]["value"]
        self.Q_burner = th["Q_burner"]["value"]
        self.UA_loss  = th["UA_loss"]["value"]
        self.UA_gas   = th["UA_gas"]["value"]
        self.T_amb    = th["T_amb"]["value"]

    # ------------------------------------------------------------------
    # Schmidt dimensionless groupings
    # ------------------------------------------------------------------
    def _schmidt_groups(self, T_h, T_c):
        """
        Return (tau, b, phi, S) -- the Schmidt isothermal parameters.
        Urieli & Berchowitz (1984), Eqs. 3.x.
        """
        tau = T_c / T_h                       # cold/hot ratio (<1)
        s = self.kappa                        # swept volume ratio
        a = self.phase_rad
        # Mean gas temperature in dead (regenerator) volume (log-mean)
        if abs(T_h - T_c) < 1e-9:
            T_r = T_h
        else:
            T_r = (T_h - T_c) / np.log(T_h / T_c)
        Xr = self.X_dead * (T_c / T_r)        # effective dead-volume term

        # Schmidt amplitude numerator/denominator
        B = np.sqrt(tau**2 + 2.0 * tau * s * np.cos(a) + s**2)
        S = tau + 2.0 * tau * Xr + s          # dimensionless total
        b = B / S
        b = min(b, 0.999999)                  # guard sqrt(1-b^2)
        # Phase of pressure wave
        phi = np.arctan2(s * np.sin(a), tau + s * np.cos(a))
        return tau, b, phi, S, T_r

    # ------------------------------------------------------------------
    # Total working-gas mass (ideal gas, Schmidt)
    # ------------------------------------------------------------------
    def gas_mass(self, T_h, T_c):
        """Total charge mass of working gas [kg] from mean pressure."""
        tau, b, phi, S, T_r = self._schmidt_groups(T_h, T_c)
        # m = p_mean * V_exp * S / (R * T_c)  * sqrt(1-b^2)  (Schmidt closure)
        m = self.p_mean * self.V_exp * S * np.sqrt(1.0 - b**2) / (self.R_gas * T_c)
        return m

    # ------------------------------------------------------------------
    # Indicated work / power -- closed-form Schmidt integral
    # ------------------------------------------------------------------
    def _expansion_heat_per_cycle(self, T_h, T_c):
        """
        Isothermal EXPANSION-space heat per cycle Q_e [J] (Schmidt closed
        form, Urieli & Berchowitz 1984 Eq. 3.27).  Because the expansion is
        isothermal (dU=0), this equals the expansion-space work W_e and is
        the heat the HEATER must supply ideally.

            Q_e = W_e = pi * p_mean * V_exp * b * sin(phi) / (1 + sqrt(1-b^2))
        """
        tau, b, phi, S, T_r = self._schmidt_groups(T_h, T_c)
        delta = np.sqrt(1.0 - b**2)
        W_e = (np.pi * self.p_mean * self.V_exp * b * abs(np.sin(phi))
               / (1.0 + delta))
        return W_e

    def indicated_work_per_cycle(self, T_h=None, T_c=None):
        """
        Net indicated cycle work per cycle [J].

        Compression-space work W_c = -tau * W_e (common pressure trace,
        compression space at T_c), so the net is
            W_net = W_e + W_c = W_e * (1 - tau)
        With a perfect regenerator this gives eta = W_net/Q_e = 1-tau = Carnot.
        """
        T_h = self.T_h if T_h is None else T_h
        T_c = self.T_c if T_c is None else T_c
        tau = T_c / T_h
        W_e = self._expansion_heat_per_cycle(T_h, T_c)
        return W_e * (1.0 - tau)

    def heater_duty_per_cycle(self, T_h=None, T_c=None):
        """
        External heat the heater must supply each cycle [J].
            Q_in = Q_e (isothermal expansion heat)
                 + (1-eps) * regenerator re-heat duty
        The regenerator-loss term raises Q_in above the ideal Q_e, so the
        efficiency W_net/Q_in falls strictly below Carnot (second law).
        """
        T_h = self.T_h if T_h is None else T_h
        T_c = self.T_c if T_c is None else T_c
        Q_e = self._expansion_heat_per_cycle(T_h, T_c)
        # Regenerator loss: un-recovered sensible duty of gas shuttling
        m = self.gas_mass(T_h, T_c)
        Q_regen_loss = (1.0 - self.regen_eff) * m * self.cv * (T_h - T_c)
        return Q_e + Q_regen_loss

    def indicated_power(self, T_h=None, T_c=None, n_rpm=None):
        """Indicated power [W] = W_ind * cycles/s."""
        n = self.n_rpm if n_rpm is None else n_rpm
        freq = n / 60.0
        return self.indicated_work_per_cycle(T_h, T_c) * freq

    def cycle_efficiency(self, T_h=None, T_c=None):
        """Indicated thermal efficiency = W_ind / Q_heater  (< Carnot)."""
        T_h = self.T_h if T_h is None else T_h
        T_c = self.T_c if T_c is None else T_c
        W = self.indicated_work_per_cycle(T_h, T_c)
        Q = self.heater_duty_per_cycle(T_h, T_c)
        return W / Q if Q > 0 else 0.0

    def carnot_efficiency(self, T_h=None, T_c=None):
        T_h = self.T_h if T_h is None else T_h
        T_c = self.T_c if T_c is None else T_c
        return 1.0 - T_c / T_h

    # ------------------------------------------------------------------
    # Brake (shaft) power -- mechanical + pumping losses
    # ------------------------------------------------------------------
    def brake_power(self, T_h=None, T_c=None, n_rpm=None):
        """Net shaft power [W] after pumping and mechanical losses."""
        P_ind = self.indicated_power(T_h, T_c, n_rpm)
        return self.mech_eff * (1.0 - self.pump_coeff) * P_ind

    def brake_efficiency(self, T_h=None, T_c=None, n_rpm=None):
        """Overall (brake) thermal efficiency."""
        T_h = self.T_h if T_h is None else T_h
        T_c = self.T_c if T_c is None else T_c
        Q = self.heater_duty_per_cycle(T_h, T_c) * ((self.n_rpm if n_rpm is None else n_rpm) / 60.0)
        P_b = self.brake_power(T_h, T_c, n_rpm)
        return P_b / Q if Q > 0 else 0.0

    # ------------------------------------------------------------------
    # Beale / West number cross-check
    # ------------------------------------------------------------------
    def beale_power(self, T_h=None, T_c=None, n_rpm=None):
        """Empirical Beale-number power estimate [W] (Walker 1980)."""
        n = self.n_rpm if n_rpm is None else n_rpm
        freq = n / 60.0
        return self.beale_number * self.p_mean * self.V_exp * freq

    def west_power(self, T_h=None, T_c=None, n_rpm=None):
        """West-number power: Beale scaled by (Th-Tc)/(Th+Tc)."""
        T_h = self.T_h if T_h is None else T_h
        T_c = self.T_c if T_c is None else T_c
        n = self.n_rpm if n_rpm is None else n_rpm
        freq = n / 60.0
        Wn = 0.25  # West constant (Walker 1980; 0.25-0.35)
        return Wn * self.p_mean * self.V_exp * freq * (T_h - T_c) / (T_h + T_c)

    # ------------------------------------------------------------------
    # Lumped warm-up ODE for hot-end temperature
    # ------------------------------------------------------------------
    def dThdt(self, T_h, Q_burner=None):
        """Heater-head temperature rate [K/s]."""
        Qb = self.Q_burner if Q_burner is None else Q_burner
        # Heat absorbed by working gas (drives the cycle); proportional to
        # head-to-gas temperature difference toward the gas mean (~T_c..T_h)
        T_gas_ref = self.T_c
        Q_to_gas = self.UA_gas * (T_h - T_gas_ref)
        Q_loss = self.UA_loss * (T_h - self.T_amb)
        return (Qb - Q_to_gas - Q_loss) / self.C_th

    def simulate(self, T_h0=None, T_c=None, n_rpm=None, p_mean=None,
                 Q_burner=None, dt=1.0, duration_s=600.0):
        """
        Transient warm-up: integrate hot-end temperature ODE and report
        the cycle performance time-series as the head heats up.

        Parameters
        ----------
        T_h0 : float   initial heater-head temperature [K] (cold start)
        T_c  : float   cold-side temperature [K] (default param)
        n_rpm: float   engine speed [rpm]
        p_mean: float  mean pressure [Pa] (override charge)
        Q_burner: float burner duty [W]
        dt, duration_s : output step and horizon [s]

        Returns
        -------
        dict with arrays: t, T_h, indicated_power, brake_power,
             efficiency, carnot_eff, beale_power, heat_input
        """
        if T_c is not None:
            self.T_c = T_c
        if n_rpm is not None:
            self.n_rpm = n_rpm
        if p_mean is not None:
            self.p_mean = p_mean
        if T_h0 is None:
            T_h0 = self.T_amb

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dThdt(y[0], Q_burner)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T_h0],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-9,
            max_step=dt,
        )

        t_out = sol.t
        Th_out = sol.y[0]
        N = len(t_out)

        P_ind = np.zeros(N)
        P_brk = np.zeros(N)
        eff = np.zeros(N)
        eff_c = np.zeros(N)
        P_beale = np.zeros(N)
        Q_in = np.zeros(N)

        for i in range(N):
            Th = Th_out[i]
            if Th <= self.T_c:        # no positive cycle below cold side
                continue
            P_ind[i] = self.indicated_power(Th, self.T_c, self.n_rpm)
            P_brk[i] = self.brake_power(Th, self.T_c, self.n_rpm)
            eff[i] = self.cycle_efficiency(Th, self.T_c)
            eff_c[i] = self.carnot_efficiency(Th, self.T_c)
            P_beale[i] = self.beale_power(Th, self.T_c, self.n_rpm)
            Q_in[i] = self.heater_duty_per_cycle(Th, self.T_c) * (self.n_rpm / 60.0)

        return {
            "t": t_out,
            "T_h": Th_out,
            "indicated_power": P_ind,
            "brake_power": P_brk,
            "efficiency": eff,
            "carnot_eff": eff_c,
            "beale_power": P_beale,
            "heat_input": Q_in,
        }
