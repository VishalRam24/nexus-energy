"""
EC188 -- Superconducting Magnetic Energy Storage (SMES) -- F2a Physics-Lumped

Coil + power-conditioning chopper, 0D first-principles dynamic model.

State variable
--------------
    I  : superconducting coil current [A]

Magnetic energy stored in the coil field:
    E = 0.5 * L * I^2                                   [J]               (1)

Coil terminal (circuit) equation. The chopper imposes a controllable DC
voltage V_chop across the coil. With a tiny equivalent series resistance
R_coil (joint + lead residual; ~0 in the superconducting state):
    V_chop = L * dI/dt + R_coil * I                                       (2)
=>  dI/dt = (V_chop - R_coil * I) / L                                     (ODE)

This is the lumped coil-current ODE integrated by scipy.integrate.solve_ivp.

Sign / power convention (grid-referenced):
    V_chop > 0  -> dI/dt > 0  -> coil current rises -> CHARGING
    V_chop < 0  -> dI/dt < 0  -> coil current falls -> DISCHARGING
    Coil (DC) power:        P_coil = V_chop * I                           (3)
        P_coil > 0  energy flowing INTO the coil (charge)
        P_coil < 0  energy flowing OUT of the coil (discharge)

Converter (chopper) losses -- one efficiency pass between DC coil side and
the AC grid terminals:
    charging   (P_coil > 0):  P_grid_in  = P_coil / eta_conv + P_cryo
    discharging(P_coil < 0):  P_grid_out = |P_coil| * eta_conv - P_cryo

Cryogenic refrigeration is a continuous parasitic electrical load P_cryo,
drawn from the grid whether or not the coil is charging/discharging
(static heat-leak removal through a GM / pulse-tube cooler).

Two driving modes are supported:
  * voltage-command:  V_chop(t) given directly (fundamental form of (2)).
  * power-command:    a target DC coil power P_req is requested; the chopper
                      computes the voltage needed, V_chop = P_req / I, with
                      saturation at +-V_dc_max and current limits.

Energy conservation is exact for the coil itself:  integral(P_coil dt) = dE,
since dE/dt = L*I*dI/dt = I*(V_chop - R_coil*I) = P_coil - I^2*R_coil,
i.e. d(0.5 L I^2)/dt = V_chop*I - R_coil*I^2 (Joule loss term, ~0 supercon).

Round-trip efficiency (charge then discharge the same energy) is strictly
between 0 and 1 because of the two converter passes (eta_conv^2) plus the
cryogenic energy consumed over the cycle time.

References
----------
    Hassenzahl, W.V. (2001). Superconducting magnetic energy storage.
        IEEE Trans. Appl. Supercond. 11(1):1447-1453.
    Buckles, W. & Hassenzahl, W.V. (2000). Superconducting magnetic energy
        storage. IEEE Power Eng. Rev. 20(5):16-20.
    Ali, M.H., Wu, B. & Dougal, R.A. (2010). An overview of SMES applications
        in power and energy systems. IEEE Trans. Sustain. Energy 1(1):38-47.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SMES_F2a:
    """SMES physics-lumped model: coil-current ODE + chopper + cryo load."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.L          = u["L_H"]["value"]            # H
        self.I_max      = u["I_max_A"]["value"]        # A
        self.I_min      = u["I_min_A"]["value"]        # A
        self.P_rated    = u["P_rated_MW"]["value"] * 1e6   # W
        self.V_dc_max   = u["V_dc_max_V"]["value"]     # V
        self.R_coil     = u["R_coil_ohm"]["value"]     # ohm
        self.eta_conv   = u["eta_converter"]["value"]  # -
        self.T_op       = u["T_op_K"]["value"]         # K

        self.P_cryo     = params.get("cryo", {}).get(
            "P_cryo_MW", {"value": 0.0})["value"] * 1e6   # W

        self.E_max = 0.5 * self.L * self.I_max ** 2    # J
        self.E_max_MJ = self.E_max / 1e6

    # ------------------------------------------------------------------
    # Algebraic helpers
    # ------------------------------------------------------------------
    def energy_J(self, I):
        """Magnetic energy E = 0.5 L I^2 [J]."""
        return 0.5 * self.L * np.asarray(I, dtype=float) ** 2

    def current_from_energy(self, E_J):
        """Inverse of (1): I = sqrt(2E/L) [A]."""
        return np.sqrt(np.maximum(2.0 * np.asarray(E_J, dtype=float) / self.L, 0.0))

    def soc(self, I):
        """Energy-based state of charge = (I/I_max)^2."""
        return self.energy_J(I) / (self.E_max + 1e-30)

    def coil_power_W(self, V_chop, I):
        """DC coil power P = V*I [W] (positive into coil)."""
        return np.asarray(V_chop, dtype=float) * np.asarray(I, dtype=float)

    def voltage_for_power(self, P_req_W, I):
        """
        Chopper voltage needed to deliver a requested DC coil power.
        P_coil = V*I  ->  V = P/I, saturated at +-V_dc_max.
        Near I=0 the achievable power -> 0 (can't push power into zero current
        without infinite voltage); clamp to V_dc_max.
        """
        I = float(I)
        if abs(I) < 1e-6:
            V = np.sign(P_req_W) * self.V_dc_max
        else:
            V = P_req_W / I
        return float(np.clip(V, -self.V_dc_max, self.V_dc_max))

    def grid_power_W(self, P_coil_W):
        """
        Grid-side electrical power including converter pass and cryo load.
        Returns dict with signed P_grid (positive = drawn from grid).
        """
        P_coil = float(P_coil_W)
        if P_coil >= 0.0:        # charging: coil absorbs P_coil, losses raise grid draw
            P_grid = P_coil / self.eta_conv + self.P_cryo
        else:                    # discharging: coil delivers |P_coil|, losses cut output
            P_grid = -(abs(P_coil) * self.eta_conv) + self.P_cryo
        return P_grid

    # ------------------------------------------------------------------
    # Coil-current ODE  dI/dt = (V_chop - R*I)/L
    # ------------------------------------------------------------------
    def _rhs(self, t, y, v_func):
        I = y[0]
        V = v_func(t, I)
        dIdt = (V - self.R_coil * I) / self.L
        # Soft current limiting: do not exceed [I_min, I_max]
        if I >= self.I_max and dIdt > 0:
            dIdt = 0.0
        if I <= self.I_min and dIdt < 0:
            dIdt = 0.0
        return [dIdt]

    def simulate(self, I0, command, mode="voltage",
                 dt=0.01, duration_s=1.0):
        """
        Integrate the coil-current ODE.

        Parameters
        ----------
        I0       : initial coil current [A]
        command  : if mode=="voltage": V_chop [V], scalar or callable(t)->V
                   if mode=="power":   P_req DC coil power [W] (positive = charge),
                                       scalar or callable(t)->P
        mode     : "voltage" or "power"
        dt       : output sample step [s]
        duration_s : total simulated time [s]

        Returns dict of time-series arrays.
        """
        I0 = float(np.clip(I0, self.I_min, self.I_max))

        if mode == "voltage":
            if callable(command):
                v_func = lambda t, I: float(np.clip(command(t), -self.V_dc_max, self.V_dc_max))
            else:
                Vc = float(np.clip(command, -self.V_dc_max, self.V_dc_max))
                v_func = lambda t, I: Vc
        elif mode == "power":
            if callable(command):
                v_func = lambda t, I: self.voltage_for_power(command(t), I)
            else:
                Pc = float(command)
                v_func = lambda t, I: self.voltage_for_power(Pc, I)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        n = max(int(round(duration_s / dt)) + 1, 2)
        t_eval = np.linspace(0.0, duration_s, n)

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [I0],
            t_eval=t_eval, args=(v_func,),
            method="RK45", rtol=1e-7, atol=1e-6, max_step=dt,
        )

        t = sol.t
        I = np.clip(sol.y[0], self.I_min, self.I_max)

        V = np.array([v_func(ti, Ii) for ti, Ii in zip(t, I)])
        E_J = self.energy_J(I)
        E_MJ = E_J / 1e6
        SOC = self.soc(I)
        P_coil = V * I                                  # W, signed (into coil)
        P_grid = np.array([self.grid_power_W(p) for p in P_coil])  # W, signed (from grid)
        P_cryo = np.full_like(t, self.P_cryo)

        return {
            "t": t,
            "I_coil_A": I,
            "V_chop_V": V,
            "E_stored_MJ": E_MJ,
            "SOC": SOC,
            "P_coil_W": P_coil,
            "P_coil_MW": P_coil / 1e6,
            "P_grid_W": P_grid,
            "P_grid_MW": P_grid / 1e6,
            "P_cryo_MW": P_cryo / 1e6,
            "mode": mode,
        }

    # ------------------------------------------------------------------
    # Round-trip efficiency over a full charge -> discharge cycle
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, P_W=None, I_start=None, I_target=None):
        """
        Charge the coil from I_start to I_target at constant DC power |P_W|,
        then discharge back to I_start at the same power. Returns dict with
        eta_rt = E_grid_out / E_grid_in (strictly in (0,1)).
        """
        if I_start is None:
            I_start = self.I_min
        if I_target is None:
            I_target = self.I_max
        if P_W is None:
            P_W = 0.5 * self.P_rated

        dE = self.energy_J(I_target) - self.energy_J(I_start)   # J to add
        # Approximate cycle time (energy / power) for each leg.
        t_leg = dE / P_W

        # Grid energy IN during charge (coil power +P_W, converter + cryo).
        P_grid_in = P_W / self.eta_conv + self.P_cryo
        E_grid_in = P_grid_in * t_leg

        # Grid energy OUT during discharge (coil power -P_W).
        P_grid_out = P_W * self.eta_conv - self.P_cryo
        P_grid_out = max(P_grid_out, 0.0)
        E_grid_out = P_grid_out * t_leg

        eta_rt = E_grid_out / (E_grid_in + 1e-30)
        return {
            "eta_rt": eta_rt,
            "E_grid_in_MJ": E_grid_in / 1e6,
            "E_grid_out_MJ": E_grid_out / 1e6,
            "dE_coil_MJ": dE / 1e6,
            "t_leg_s": t_leg,
        }
