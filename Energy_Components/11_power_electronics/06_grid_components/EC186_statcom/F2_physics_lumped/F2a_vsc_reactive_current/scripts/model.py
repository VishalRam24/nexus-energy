"""
EC186 -- STATCOM (Static Synchronous Compensator) -- F2a
Physics-Lumped VSC Reactive-Current Control with DC-Link Dynamics.

A STATCOM is a shunt-connected Voltage-Sourced Converter (VSC) that exchanges
reactive power with the bus by controlling the magnitude of its AC output
voltage V_conv relative to the bus voltage V_bus across the coupling reactance X:

    Q_inj = V_bus * (V_bus - V_conv) / X            (Hingorani & Gyugyi 2000, Ch.5)

  - V_conv > V_bus  -> Q < 0 absorbed-from-converter convention here means the
    converter LEADS the bus and SUPPLIES capacitive (leading) reactive power.
  - V_conv < V_bus  -> converter absorbs inductive (lagging) reactive power.

Unlike an SVC (a variable shunt susceptance, Q = B*V^2), the STATCOM is a
controlled CURRENT source: its reactive current capability is *independent of
bus voltage* down to very low V (constant-current region), which is the key
operational advantage modelled here.

This F2a model is a *lumped* dynamic (0D ODE) representation in the synchronous
dq reference frame (grid-voltage-oriented, so v_d = V_bus_peak, v_q = 0):

State vector x = [i_d, i_q, Vdc]:

  Inner current loop (closed-loop first-order, fast dq control; Yazdani &
  Iravani 2010, Ch.8 — decoupled current control collapses the L/R converter
  dynamics into a designed closed-loop bandwidth 1/tau_i):

      di_d/dt = (i_d_ref - i_d) / tau_i
      di_q/dt = (i_q_ref - i_q) / tau_i

      i_q_ref = -Q_ref / (1.5 * v_d)        (reactive-current command, dq)
      i_q_ref is clamped to +/- I_max  -> CONSTANT-CURRENT capability at low V.
      i_d_ref from the DC-link voltage PI regulator (draws just enough real
      current to hold Vdc and cover losses).

  DC-link capacitor energy balance (Acha et al. 2002, Ch.6):

      C_dc * Vdc * dVdc/dt = P_ac_into_dc - P_loss
      P_ac_into_dc = 1.5 * (v_d*i_d + v_q*i_q)   (3-phase power, peak-amplitude dq)

Outputs of interest: instantaneous Q delivered to the bus,
Q = -1.5 * v_d * i_q  (with i_q < 0 => Q > 0 capacitive), the equivalent
converter voltage V_conv from the Q-injection relation, the DC-link voltage,
and converter losses.

References:
    Hingorani, N.G. & Gyugyi, L. (2000). Understanding FACTS. IEEE Press. Ch.5.
    Acha, E., Fuerte-Esquivel, C.R., Ambriz-Perez, H., Angeles-Camacho, C.
        (2002). FACTS: Modelling and Simulation in Power Networks. Wiley. Ch.6.
    Yazdani, A. & Iravani, R. (2010). Voltage-Sourced Converters in Power
        Systems. Wiley/IEEE. Ch.8 (current control), Ch.5 (dq frame).
"""

import numpy as np
from scipy.integrate import solve_ivp


class STATCOM_F2a:
    """VSC-based STATCOM -- lumped dq current control + DC-link voltage ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.Q_max = u["Q_max_MVAR"]["value"] * 1e6        # VAR
        self.Q_min = u["Q_min_MVAR"]["value"] * 1e6        # VAR
        self.V_LL = u["V_rated_kV"]["value"] * 1e3         # V (line-line RMS)
        self.X_pu = u["X_pu"]["value"]
        self.R_pu = u["R_pu"]["value"]
        self.f = u["f_grid_Hz"]["value"]
        self.w = 2.0 * np.pi * self.f
        self.Vdc_rated = u["Vdc_rated_kV"]["value"] * 1e3  # V
        self.C_dc = u["C_dc_mF"]["value"] * 1e-3           # F
        self.tau_i = u["tau_i_ms"]["value"] * 1e-3         # s
        self.kp_vdc = u["kp_vdc"]["value"]
        self.ki_vdc = u["ki_vdc"]["value"]
        self.loss_factor = u["loss_factor"]["value"]
        self.P_standby = u["P_standby_MW"]["value"] * 1e6  # W

        # Base quantities (STATCOM rating = max |Q|)
        self.S_base = max(abs(self.Q_max), abs(self.Q_min))   # VA
        self.V_base_phase_pk = self.V_LL * np.sqrt(2.0 / 3.0)  # peak phase volt
        self.Z_base = self.V_LL ** 2 / self.S_base             # ohm (LL/3ph)
        self.X = self.X_pu * self.Z_base                       # ohm
        self.R = self.R_pu * self.Z_base                       # ohm

        # Rated converter line current (peak) and current limit.
        # S = 1.5 * v_d * i_pk  => i_pk = S / (1.5 * v_d)
        self.I_rated_pk = self.S_base / (1.5 * self.V_base_phase_pk)
        self.I_max = 1.05 * self.I_rated_pk   # 5% headroom

    # ------------------------------------------------------------------
    # Static / algebraic helpers
    # ------------------------------------------------------------------
    def vd_from_Vbus(self, V_bus_pu: float) -> float:
        """Grid-oriented dq: v_d = peak phase voltage at the bus, v_q = 0."""
        return V_bus_pu * self.V_base_phase_pk

    def iq_ref_from_Q(self, Q_ref: float, v_d: float) -> float:
        """
        Reactive-current command from desired Q.
        Q = -1.5 * v_d * i_q  (convention: i_q<0 -> Q>0 capacitive).
        Clamped to +/-I_max -> constant-current capability at low voltage.
        """
        Q_ref = float(np.clip(Q_ref, self.Q_min, self.Q_max))
        v_d = max(v_d, 1e-3)
        i_q = -Q_ref / (1.5 * v_d)
        return float(np.clip(i_q, -self.I_max, self.I_max))

    def Q_from_iq(self, v_d: float, i_q: float) -> float:
        """Instantaneous reactive power delivered to the bus [VAR]."""
        return -1.5 * v_d * i_q

    def P_from_id(self, v_d: float, i_d: float) -> float:
        """Instantaneous real power drawn from the bus into the VSC [W]."""
        return 1.5 * v_d * i_d

    def Vconv_from_Q(self, V_bus_pu: float, Q: float) -> float:
        """
        Equivalent converter AC voltage from the Q-injection relation
            Q = V_bus*(V_bus - V_conv)/X    (per-unit form),
        i.e. V_conv = V_bus - Q*X / V_bus.   Returns line-line RMS [V].
        """
        V_bus = V_bus_pu * self.V_LL
        Q_pu = Q / self.S_base
        X = self.X_pu
        V_conv_pu = V_bus_pu - Q_pu * X / max(V_bus_pu, 1e-3)
        return V_conv_pu * self.V_LL

    def losses(self, i_d: float, i_q: float) -> float:
        """Converter conduction/switching losses [W]: I^2 R + standby + switching."""
        i_mag2 = i_d ** 2 + i_q ** 2
        p_cond = 1.5 * self.R * i_mag2                 # 3-phase I^2R on coupling R
        # switching loss approximated as loss_factor share of throughput
        s_app = 1.5 * self.V_base_phase_pk * np.sqrt(i_mag2)
        p_sw = self.loss_factor * s_app
        return p_cond + p_sw + self.P_standby

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------
    def _rhs(self, t, x, Q_ref_fn, V_bus_fn):
        i_d, i_q, Vdc, xi = x   # xi = integral state of DC-link PI [pu*s]
        V_bus_pu = V_bus_fn(t)
        Q_ref = Q_ref_fn(t)
        v_d = self.vd_from_Vbus(V_bus_pu)

        # Reactive-current command (with constant-current clamp)
        i_q_ref = self.iq_ref_from_Q(Q_ref, v_d)

        # DC-link voltage PI -> real-current command i_d_ref.
        # i_d > 0 draws real power from bus to charge the DC capacitor.
        # The integral term drives the steady-state error to zero, so at
        # equilibrium P_in exactly covers losses (true energy balance).
        # error = (rated - Vdc): positive when capacitor is under-charged, so
        # i_d_ref > 0 draws real power from the bus to recharge the DC link.
        e_vdc = (self.Vdc_rated - Vdc) / self.Vdc_rated  # per-unit error
        dxi = e_vdc                                       # integrate error
        i_d_ref = (self.kp_vdc * e_vdc + self.ki_vdc * xi) * self.I_rated_pk
        i_d_ref = float(np.clip(i_d_ref, -self.I_max, self.I_max))

        # Fast inner current loop (designed closed-loop first-order response)
        di_d = (i_d_ref - i_d) / self.tau_i
        di_q = (i_q_ref - i_q) / self.tau_i

        # DC-link capacitor energy balance
        P_ac = self.P_from_id(v_d, i_d)        # real power into converter
        P_loss = self.losses(i_d, i_q)
        Vdc_safe = max(Vdc, 1e-3)
        dVdc = (P_ac - P_loss) / (self.C_dc * Vdc_safe)

        return [di_d, di_q, dVdc, dxi]

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------
    def simulate(self, Q_ref, V_bus_pu=1.0, dt=1e-4, duration_s=0.1,
                 x0=None):
        """
        Integrate the lumped STATCOM ODE.

        Parameters
        ----------
        Q_ref       : float | callable(t)->VAR ... reactive power command [VAR]
                      (or MVAR? -> caller passes VAR; predict() handles MVAR)
        V_bus_pu    : float | callable(t)->pu ... bus voltage in per-unit
        dt          : output time step [s]
        duration_s  : total simulated time [s]
        x0          : optional initial state [i_d, i_q, Vdc, xi]

        Returns dict of time-series arrays.
        """
        Q_ref_fn = Q_ref if callable(Q_ref) else (lambda t: float(Q_ref))
        V_bus_fn = V_bus_pu if callable(V_bus_pu) else (lambda t: float(V_bus_pu))

        if x0 is None:
            x0 = [0.0, 0.0, self.Vdc_rated, 0.0]

        n = max(2, int(round(duration_s / dt)) + 1)
        t_eval = np.linspace(0.0, duration_s, n)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), x0,
            t_eval=t_eval, args=(Q_ref_fn, V_bus_fn),
            method="RK45", rtol=1e-6, atol=1e-9, max_step=dt,
        )

        i_d = sol.y[0]
        i_q = sol.y[1]
        Vdc = sol.y[2]
        t = sol.t

        V_bus_pu_arr = np.array([V_bus_fn(tt) for tt in t])
        Q_ref_arr = np.array([Q_ref_fn(tt) for tt in t])
        v_d_arr = self.vd_from_Vbus(V_bus_pu_arr)

        Q_out = -1.5 * v_d_arr * i_q
        P_in = 1.5 * v_d_arr * i_d
        P_loss = np.array([self.losses(i_d[k], i_q[k]) for k in range(len(t))])
        V_conv = np.array([self.Vconv_from_Q(V_bus_pu_arr[k], Q_out[k])
                           for k in range(len(t))])
        I_mag = np.sqrt(i_d ** 2 + i_q ** 2)

        return {
            "t": t,
            "i_d": i_d,
            "i_q": i_q,
            "I_mag": I_mag,
            "Vdc": Vdc,
            "Q_out_VAR": Q_out,
            "Q_out_MVAR": Q_out / 1e6,
            "Q_ref_VAR": Q_ref_arr,
            "P_in_W": P_in,
            "P_loss_W": P_loss,
            "V_conv_V": V_conv,
            "V_bus_pu": V_bus_pu_arr,
            "I_max": self.I_max,
        }
