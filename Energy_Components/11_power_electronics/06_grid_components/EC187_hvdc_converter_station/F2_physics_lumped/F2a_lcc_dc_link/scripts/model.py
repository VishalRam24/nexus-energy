"""
EC187 -- HVDC Converter Station -- F2a Physics-Lumped (LCC, 12-pulse + DC-link ODE)

Line-commutated converter (LCC) point-to-point HVDC link. Two stations
(rectifier and inverter) connected by a lumped DC line. Each station is an
n-bridge series 12-pulse converter described by the classical converter
equations, and the DC-link current is integrated as a first-order ODE.

------------------------------------------------------------------------------
Converter (per pole) average-value equations  (Kundur 1994, Ch.10; Arrillaga 1998)
------------------------------------------------------------------------------
Ideal no-load direct voltage of a 6-pulse Graetz bridge:
    Vd0 = (3*sqrt(2)/pi) * V_LL_secondary

Rectifier average DC voltage (firing/delay angle alpha, commutating reactance Xc):
    Vd_r = n * ( Vd0 * cos(alpha) - (3/pi) * Xc * Id )

Inverter average DC voltage (extinction/advance angle gamma):
    Vd_i = n * ( Vd0 * cos(gamma) - (3/pi) * Xc * Id )       (counter-emf, opposes Id)

The term (3/pi)*Xc*Id is the commutation voltage drop (overlap), NOT a resistive
loss -- it is reactive. The equivalent commutation resistance Rc = (3/pi)*Xc.

Power factor and reactive consumption (both ends absorb Q, supplied by filters):
    cos(phi) ~= Vd / (n*Vd0)            (ratio of actual to ideal DC voltage)
    Q = P * tan(phi)
A LCC always *consumes* reactive power (phi > 0 at both rectifier and inverter);
typically Q ~ 0.5-0.6 * P at rated load.

------------------------------------------------------------------------------
DC-link current ODE  (current-source link, smoothing reactors + line inductance)
------------------------------------------------------------------------------
The DC circuit is rectifier emf -> L (smoothing+line) and R (line) -> inverter emf:

    L_dc * dId/dt = Vd_r(alpha, Id) - R_line * Id - Vd_i(gamma, Id)

with L_dc = L_smoothing_rect + L_line + L_smoothing_inv. Steady state recovers the
classic two-terminal load-flow:
    Id_ss = (Vd_r0 - Vd_i0) / (R_line + Rc_r + Rc_i)
where Vd_r0 = n*Vd0*cos(alpha), Vd_i0 = n*Vd0*cos(gamma),
Rc = n*(3/pi)*Xc per terminal.

------------------------------------------------------------------------------
Power balance / efficiency
------------------------------------------------------------------------------
    P_dc_rect  = Vd_r * Id                 (DC power leaving rectifier)
    P_dc_inv   = Vd_i * Id                 (DC power entering inverter)
    P_line_loss= R_line * Id^2             (resistive line loss)
    P_dc_rect - P_dc_inv = P_line_loss     (DC energy conservation, exact at SS)
Station converter losses (valve conduction, transformer, aux, cooling) are added
on the AC side as a no-load + proportional model (Cigre TB 388):
    P_loss_stn = P_no_load + loss_factor * P_rated * (Id/Id_rated)
    P_ac_in (rect) = P_dc_rect + P_loss_rect
    P_ac_out(inv)  = P_dc_inv  - P_loss_inv
    eta_link = P_ac_out / P_ac_in   in (0,1), ~0.97-0.99 at rated.

References:
    Kundur, P. (1994). Power System Stability and Control. McGraw-Hill, Ch.10.
    Arrillaga, J. (1998). High Voltage Direct Current Transmission, 2nd ed., IET.
    Cigre TB 388 (2009). Impacts of HVDC Lines on the Economics of HVDC Projects.
    Kim, C.-K. et al. (2009). HVDC Transmission: Power Conversion Applications
        in Power Systems. Wiley/IEEE Press (VSC and LCC converter theory).
"""

import numpy as np
from scipy.integrate import solve_ivp

PI = np.pi


class HVDC_LCC_F2a:
    """LCC-HVDC point-to-point link -- 12-pulse converters + DC-link current ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_MW"]["value"] * 1e6          # W
        self.V_dc_rated = u["V_dc_rated_kV"]["value"] * 1e3    # V (pole)
        self.Id_rated = u["Id_rated_kA"]["value"] * 1e3        # A
        self.n = u["n_bridges_series"]["value"]                # bridges/pole
        self.Vd0_r = u["Vd0_rect_kV"]["value"] * 1e3          # V per bridge (rectifier)
        self.Vd0_i = u["Vd0_inv_kV"]["value"] * 1e3           # V per bridge (inverter)
        self.Xc = u["Xc_ohm"]["value"]                         # Ohm per bridge
        self.alpha_rect = np.deg2rad(u["alpha_rect_deg"]["value"])
        self.gamma_inv = np.deg2rad(u["gamma_inv_deg"]["value"])
        self.loss_factor = u["loss_factor_station"]["value"]
        self.P_no_load = u["P_no_load_MW"]["value"] * 1e6      # W
        self.line_len = u["line_length_km"]["value"]           # km
        self.line_R_per_km = u["line_R_ohm_per_km"]["value"]   # Ohm/km
        self.line_L_per_km = u["line_L_mH_per_km"]["value"] * 1e-3  # H/km
        self.L_smooth = u["L_smoothing_H"]["value"]            # H per terminal

        # Equivalent commutation resistance per terminal (n bridges in series)
        self.Rc = self.n * (3.0 / PI) * self.Xc                # Ohm
        # DC line resistance (at reference temperature)
        self.R_line0 = self.line_R_per_km * self.line_len      # Ohm
        # Total DC-link inductance: 2 smoothing reactors + line inductance
        self.L_dc = 2.0 * self.L_smooth + self.line_L_per_km * self.line_len  # H

    # ------------------------------------------------------------------
    # Line resistance (optional temperature dependence)
    # ------------------------------------------------------------------
    def line_resistance(self, T_line_degC=20.0):
        """DC line resistance [Ohm] with copper/aluminium temperature coeff."""
        alpha_R = 0.00393  # 1/K, copper
        return self.R_line0 * (1.0 + alpha_R * (T_line_degC - 20.0))

    # ------------------------------------------------------------------
    # Converter DC voltages (average-value model)
    # ------------------------------------------------------------------
    def Vd_rectifier(self, Id, alpha=None):
        """Rectifier average DC terminal voltage [V]. Vd = n(Vd0 cos a - (3/pi)Xc Id)."""
        a = self.alpha_rect if alpha is None else alpha
        return self.n * (self.Vd0_r * np.cos(a) - (3.0 / PI) * self.Xc * Id)

    def Vd_inverter(self, Id, gamma=None):
        """Inverter average DC counter-voltage [V], extinction angle gamma.

        In the DC load-flow convention (Kundur 1994, eq. 10.x), the inverter is
        written as a back-emf seen by the line whose magnitude *rises* with Id
        because the commutation overlap subtracts from the inverter's own output
        but adds to the voltage the line must push against:
            Vd_i = n ( Vd0_i cos(gamma) + (3/pi) Xc Id ).
        """
        g = self.gamma_inv if gamma is None else gamma
        return self.n * (self.Vd0_i * np.cos(g) + (3.0 / PI) * self.Xc * Id)

    def Vd0_ideal(self, alpha):
        """Ideal (no-overlap) rectifier DC voltage of the n-bridge converter [V]."""
        return self.n * self.Vd0_r * np.cos(alpha)

    # ------------------------------------------------------------------
    # Reactive power consumption (LCC absorbs Q at both ends)
    # ------------------------------------------------------------------
    def power_factor(self, Id, angle, end="rect"):
        """Displacement power factor cos(phi) ~ Vd/(n*Vd0) for given delay angle.

        end : "rect" or "inv" -- selects the converter's no-load Vd0 reference.
        """
        # Displacement pf ~ (ideal DC voltage at the firing angle minus the
        # commutation overlap drop) / ideal no-load DC voltage. The overlap always
        # *reduces* the power factor at BOTH ends, so the -drop sign is physical
        # regardless of the load-flow sign convention used for the back-emf.
        Vd0 = self.Vd0_r if end == "rect" else self.Vd0_i
        Vd_int = self.n * (Vd0 * np.cos(angle) - (3.0 / PI) * self.Xc * Id)
        Vd_ideal_noload = self.n * Vd0
        cphi = np.clip(Vd_int / Vd_ideal_noload, -1.0, 1.0)
        return cphi

    def reactive_consumption(self, Id, P_dc, angle, end="rect"):
        """Reactive power consumed by a converter [W] (Q = P tan phi), always >= 0."""
        cphi = self.power_factor(Id, angle, end)
        cphi = np.clip(cphi, 1e-6, 1.0)
        tan_phi = np.sqrt(max(1.0 - cphi * cphi, 0.0)) / cphi
        return abs(P_dc) * tan_phi

    # ------------------------------------------------------------------
    # Steady-state DC current (classic two-terminal load flow)
    # ------------------------------------------------------------------
    def steady_state_current(self, alpha=None, gamma=None, T_line_degC=20.0):
        """Steady DC current [A]: Id = (Vd_r0 - Vd_i0)/(R_line + Rc_r + Rc_i)."""
        a = self.alpha_rect if alpha is None else alpha
        g = self.gamma_inv if gamma is None else gamma
        Vd_r0 = self.n * self.Vd0_r * np.cos(a)
        Vd_i0 = self.n * self.Vd0_i * np.cos(g)
        R_total = self.line_resistance(T_line_degC) + self.Rc + self.Rc
        Id = (Vd_r0 - Vd_i0) / R_total
        return max(Id, 0.0)

    # ------------------------------------------------------------------
    # DC-link current ODE derivative
    # ------------------------------------------------------------------
    def dId_dt(self, Id, alpha, gamma, T_line_degC=20.0):
        """L_dc dId/dt = Vd_r - R_line*Id - Vd_i   [A/s]."""
        Vd_r = self.Vd_rectifier(Id, alpha)
        Vd_i = self.Vd_inverter(Id, gamma)
        R_line = self.line_resistance(T_line_degC)
        return (Vd_r - R_line * Id - Vd_i) / self.L_dc

    # ------------------------------------------------------------------
    # Station / link power balance at a given operating current
    # ------------------------------------------------------------------
    def power_balance(self, Id, alpha=None, gamma=None, T_line_degC=20.0):
        """Full power balance dict at DC current Id [A]."""
        a = self.alpha_rect if alpha is None else alpha
        g = self.gamma_inv if gamma is None else gamma
        Vd_r = self.Vd_rectifier(Id, a)
        Vd_i = self.Vd_inverter(Id, g)
        R_line = self.line_resistance(T_line_degC)

        P_dc_rect = Vd_r * Id          # W leaving rectifier DC bus
        P_dc_inv = Vd_i * Id           # W entering inverter DC bus
        P_line_loss = R_line * Id * Id

        # Station converter losses (no-load + proportional to loading)
        load_frac = abs(Id) / self.Id_rated
        P_loss_stn = self.P_no_load + self.loss_factor * self.P_rated * load_frac

        P_ac_in = P_dc_rect + P_loss_stn         # AC drawn at rectifier
        P_ac_out = P_dc_inv - P_loss_stn         # AC delivered at inverter
        P_ac_out = max(P_ac_out, 0.0)

        eta = P_ac_out / P_ac_in if P_ac_in > 0 else 0.0

        Q_rect = self.reactive_consumption(Id, P_dc_rect, a, end="rect")
        Q_inv = self.reactive_consumption(Id, P_dc_inv, g, end="inv")

        return {
            "Id_A": Id,
            "Vd_rect_V": Vd_r,
            "Vd_inv_V": Vd_i,
            "P_dc_rect_W": P_dc_rect,
            "P_dc_inv_W": P_dc_inv,
            "P_line_loss_W": P_line_loss,
            "P_loss_station_W": P_loss_stn,
            "P_ac_in_W": P_ac_in,
            "P_ac_out_W": P_ac_out,
            "efficiency": eta,
            "Q_rect_VAR": Q_rect,
            "Q_inv_VAR": Q_inv,
            "pf_rect": self.power_factor(Id, a),
            "pf_inv": self.power_factor(Id, g),
        }

    # ------------------------------------------------------------------
    # Map a power order to the firing angle needed (constant-gamma inverter)
    # ------------------------------------------------------------------
    def alpha_for_power(self, P_order_W, gamma=None, T_line_degC=20.0):
        """Find rectifier alpha so DC power ~ P_order at steady current.

        Uses Id_target = P_order / V_dc_rated (constant-voltage operation) and
        inverts the SS current equation for cos(alpha).
        """
        g = self.gamma_inv if gamma is None else gamma
        Id_t = np.clip(P_order_W / self.V_dc_rated, 0.0, 1.2 * self.Id_rated)
        R_total = self.line_resistance(T_line_degC) + self.Rc + self.Rc
        Vd_i0 = self.n * self.Vd0_i * np.cos(g)
        # Id = (n Vd0_r cos a - Vd_i0)/R_total -> cos a = (Id R_total + Vd_i0)/(n Vd0_r)
        cos_a = (Id_t * R_total + Vd_i0) / (self.n * self.Vd0_r)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        return float(np.arccos(cos_a))

    # ------------------------------------------------------------------
    # Time-domain simulation of the DC-link current
    # ------------------------------------------------------------------
    def simulate(self, alpha, gamma=None, Id0=0.0, dt=1e-4, duration_s=0.2,
                 T_line_degC=20.0):
        """
        Integrate the DC-link current ODE for given firing/extinction schedules.

        Parameters
        ----------
        alpha : float or callable(t)
            Rectifier firing angle [rad].
        gamma : float or callable(t) or None
            Inverter extinction angle [rad]; default from parameters.
        Id0 : float
            Initial DC current [A].
        dt : float
            Output time step [s].
        duration_s : float
            Total duration [s].
        T_line_degC : float
            DC line temperature [degC].

        Returns
        -------
        dict of time-series: t, Id_A, Vd_rect_V, Vd_inv_V, P_dc_rect_W,
            P_dc_inv_W, P_line_loss_W, efficiency, Q_rect_VAR, Q_inv_VAR.
        """
        g_default = self.gamma_inv if gamma is None else gamma
        a_fn = alpha if callable(alpha) else (lambda t: alpha)
        g_fn = g_default if callable(g_default) else (lambda t: g_default)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            Id = y[0]
            return [self.dId_dt(Id, a_fn(t), g_fn(t), T_line_degC)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [Id0],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-4,
            max_step=dt,
        )

        t_out = sol.t
        Id_out = np.maximum(sol.y[0], 0.0)
        N = len(t_out)

        Vd_r = np.zeros(N); Vd_i = np.zeros(N)
        P_dc_r = np.zeros(N); P_dc_i = np.zeros(N)
        P_line = np.zeros(N); eta = np.zeros(N)
        Q_r = np.zeros(N); Q_i = np.zeros(N)

        for k in range(N):
            pb = self.power_balance(Id_out[k], a_fn(t_out[k]), g_fn(t_out[k]),
                                    T_line_degC)
            Vd_r[k] = pb["Vd_rect_V"]; Vd_i[k] = pb["Vd_inv_V"]  # noqa: E702
            P_dc_r[k] = pb["P_dc_rect_W"]; P_dc_i[k] = pb["P_dc_inv_W"]
            P_line[k] = pb["P_line_loss_W"]; eta[k] = pb["efficiency"]
            Q_r[k] = pb["Q_rect_VAR"]; Q_i[k] = pb["Q_inv_VAR"]

        return {
            "t": t_out,
            "Id_A": Id_out,
            "Vd_rect_V": Vd_r,
            "Vd_inv_V": Vd_i,
            "P_dc_rect_W": P_dc_r,
            "P_dc_inv_W": P_dc_i,
            "P_line_loss_W": P_line,
            "efficiency": eta,
            "Q_rect_VAR": Q_r,
            "Q_inv_VAR": Q_i,
        }
