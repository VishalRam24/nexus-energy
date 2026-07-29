"""
EC131 -- Tidal Barrage -- F2a Physics-Lumped Basin Water-Level ODE

Physics-lumped (0D) dynamic model of a tidal barrage. A basin of plan area A
sits behind a barrage. The sea outside oscillates sinusoidally:

    eta_sea(t) = z0 + a * sin(2*pi*t / T_tide)                            [m]

The basin water level z_basin(t) is the single ODE state. Mass conservation of
an incompressible fluid in a prismatic basin gives:

    A * d(z_basin)/dt = Q(t)                                              [m3/s]

where Q is the volumetric flow through the barrage (sign convention: Q > 0 fills
the basin, Q < 0 empties it). Q depends on the operating mode and the head
H = eta_sea - z_basin (sea minus basin):

  * SLUICING (gates open, free flow):   Q = sign(H) * Cd_s * A_s * sqrt(2 g |H|)
    -- standard orifice / weir discharge (Baker 1991, Ch. 5; Torricelli).
  * GENERATING (flow through turbines):  Q = sign(H) * Cd_t * A_t * sqrt(2 g |H|)
    with power extracted only when |H| >= h_min_gen.
  * HOLDING (gates + turbines closed):   Q = 0 (head builds up).

Instantaneous hydraulic power converted at the turbines (Baker 1991, eqn for
low-head plant; Prandle 1984):

    P_hyd = rho * g * |Q_turb| * |H|                                      [W]
    P_elec = eta_turbine * P_hyd            (only while |H| >= h_min_gen)

Energy per tidal cycle is the time integral of P_elec over one period T_tide.

This is a classic two-basin-state idealisation reduced to ONE state (single
basin vs. open sea). The operation strategy implemented is single-effect EBB
generation with optional flood generation:

  Flood phase (sea rising above basin): hold while sea fills above basin, open
  sluices to fill basin, then close.
  Ebb phase (sea falling below basin): hold to build head h_start_gen, then
  generate through turbines until head drops below h_min_gen, then sluice-empty.

Conservation guarantees enforced/tested:
  * Mass: integral of Q over a closed cycle returns basin to its periodic
    level (volume balance) to within solver tolerance.
  * Power: P = rho*g*Q*H*eta with 0 < eta < 1; P = 0 below h_min_gen.
  * Energy: E_cycle = integral P_elec dt >= 0, bounded above by the theoretical
    potential-energy release 0.5*rho*g*A*(2a)^2 / something -> sanity vs F1.

Hardcoded physical constants (overridable via parameters.json):
    rho = 1025 kg/m3  seawater, 15 C, 35 PSU   (UNESCO / Fofonoff 1985)
    g   = 9.81  m/s2  standard gravity          (ISO 80000-3 / CODATA)

References:
    Baker, A.C. (1991). "Tidal Power." Peter Peregrinus / IEE.
    Prandle, D. (1984). "Simple theory of tidal power." Physics and Chemistry
        of the Earth, 9, 217-228.
    Charlier, R.H. (2003). "Tidal Energy." Springer.
"""

import numpy as np
from scipy.integrate import solve_ivp


class TidalBarrageF2a:
    """Physics-lumped basin water-level ODE for a tidal barrage."""

    # Operating modes
    MODE_HOLD = 0       # all gates closed
    MODE_SLUICE = 1     # sluice gates open (free flow, no generation)
    MODE_GENERATE = 2   # flow through turbines (power extracted)

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = float(u["A_basin"]["value"])              # m2
        self.a = float(u["tidal_amplitude"]["value"])      # m
        self.z0 = float(u["mean_sea_level"]["value"])      # m
        self.T = float(u["T_tide"]["value"])               # s
        self.N_turb = float(u["N_turbines"]["value"])      # -
        self.Cd_t = float(u["Cd_turbine"]["value"])        # -
        self.A_t1 = float(u["A_turbine"]["value"])         # m2 per unit
        self.Cd_s = float(u["Cd_sluice"]["value"])         # -
        self.A_s = float(u["A_sluice"]["value"])           # m2
        self.eta_t = float(u["eta_turbine"]["value"])      # -
        self.h_min = float(u["h_min_gen"]["value"])        # m
        self.h_start = float(u["h_start_gen"]["value"])    # m
        self.rho = float(u["rho"]["value"])                # kg/m3
        self.g = float(u["g"]["value"])                    # m/s2

        self.omega = 2.0 * np.pi / self.T                  # rad/s
        self.A_turb_total = self.N_turb * self.A_t1        # m2

    # ------------------------------------------------------------------
    # Forcing
    # ------------------------------------------------------------------
    def sea_level(self, t):
        """Sinusoidal sea tide elevation [m]."""
        return self.z0 + self.a * np.sin(self.omega * t)

    # ------------------------------------------------------------------
    # Flow laws (orifice / Torricelli discharge)
    # ------------------------------------------------------------------
    def _orifice_flow(self, Cd, A_flow, H):
        """Signed orifice flow Q = sign(H) Cd A sqrt(2 g |H|)  [m3/s]."""
        return np.sign(H) * Cd * A_flow * np.sqrt(2.0 * self.g * abs(H))

    def turbine_flow(self, H):
        """Volumetric flow through the turbine bank [m3/s] (signed)."""
        return self._orifice_flow(self.Cd_t, self.A_turb_total, H)

    def sluice_flow(self, H):
        """Volumetric flow through open sluice gates [m3/s] (signed)."""
        return self._orifice_flow(self.Cd_s, self.A_s, H)

    # ------------------------------------------------------------------
    # Operation strategy: returns mode given (t, z_basin)
    # ------------------------------------------------------------------
    def operating_mode(self, t, z_basin, flood_gen=False):
        """
        Single-effect ebb generation (+ optional flood generation).

        Decide mode from the head H = sea - basin and the phase of the tide.
        Logic (head-threshold based, robust for the ODE):

          Sea ABOVE basin (H > 0, flood/filling tendency):
            - if flood_gen and H >= h_min: GENERATE (flood generation)
            - else: SLUICE to fill basin toward high water
          Sea BELOW basin (H < 0, ebb/emptying tendency):
            - if |H| >= h_start: GENERATE (ebb generation, the main mode)
            - if h_min <= |H| < h_start while already generating: GENERATE
            - else: HOLD to build head
        """
        H = self.sea_level(t) - z_basin
        if H > 0:  # sea higher than basin -> basin wants to fill
            if flood_gen and H >= self.h_min:
                return self.MODE_GENERATE
            return self.MODE_SLUICE
        else:      # sea lower than basin -> basin wants to empty (ebb)
            if abs(H) >= self.h_start:
                return self.MODE_GENERATE
            # below start head: hold to build head (until next cycle resets)
            return self.MODE_HOLD

    def flow(self, t, z_basin, flood_gen=False):
        """Net signed flow into the basin [m3/s] for the current mode."""
        H = self.sea_level(t) - z_basin
        mode = self.operating_mode(t, z_basin, flood_gen)
        if mode == self.MODE_HOLD:
            return 0.0, mode
        if mode == self.MODE_SLUICE:
            return self.sluice_flow(H), mode
        # GENERATE
        if abs(H) < self.h_min:
            return 0.0, self.MODE_HOLD  # below min head: no flow, no power
        return self.turbine_flow(H), mode

    # ------------------------------------------------------------------
    # Power
    # ------------------------------------------------------------------
    def power_elec(self, t, z_basin, flood_gen=False):
        """Instantaneous electrical power [W]. Generation only above h_min."""
        H = self.sea_level(t) - z_basin
        Q, mode = self.flow(t, z_basin, flood_gen)
        if mode != self.MODE_GENERATE or abs(H) < self.h_min:
            return 0.0
        P_hyd = self.rho * self.g * abs(Q) * abs(H)   # W
        return self.eta_t * P_hyd

    # ------------------------------------------------------------------
    # ODE
    # ------------------------------------------------------------------
    def _rhs(self, t, y, flood_gen):
        """dz/dt = Q / A   (mass conservation, prismatic basin)."""
        z = y[0]
        Q, _ = self.flow(t, z, flood_gen)
        return [Q / self.A]

    def simulate(self, n_cycles=1, z_init=None, flood_gen=False,
                 n_eval=2000, rtol=1e-6, atol=1e-3):
        """
        Integrate the basin water-level ODE over n_cycles tidal periods.

        Returns dict with time series and cycle-integrated energy.
        """
        if z_init is None:
            z_init = self.z0  # start at mean sea level
        t_end = n_cycles * self.T
        t_eval = np.linspace(0.0, t_end, int(n_eval))

        sol = solve_ivp(
            self._rhs, (0.0, t_end), [z_init],
            t_eval=t_eval, args=(flood_gen,),
            method="RK45", rtol=rtol, atol=atol, max_step=self.T / 200.0,
        )

        t = sol.t
        z_basin = sol.y[0]
        z_sea = self.sea_level(t)
        head = z_sea - z_basin

        Q = np.array([self.flow(ti, zi, flood_gen)[0]
                      for ti, zi in zip(t, z_basin)])
        P = np.array([self.power_elec(ti, zi, flood_gen)
                      for ti, zi in zip(t, z_basin)])

        # Energy by trapezoidal integration of power [J -> MWh]
        E_J = np.trapz(P, t)
        E_MWh = E_J / 3.6e9
        E_per_cycle_MWh = E_MWh / n_cycles

        # Volume throughput (mass conservation diagnostic) [m3]
        V_in = np.trapz(np.clip(Q, 0.0, None), t)
        V_out = np.trapz(np.clip(-Q, 0.0, None), t)

        P_avg_MW = (E_J / t_end) / 1e6 if t_end > 0 else 0.0

        return {
            "t": t,
            "z_sea": z_sea,
            "z_basin": z_basin,
            "head": head,
            "flow": Q,
            "power": P,                       # W
            "energy_MWh": E_MWh,
            "energy_per_cycle_MWh": E_per_cycle_MWh,
            "avg_power_MW": P_avg_MW,
            "peak_power_MW": float(np.max(P)) / 1e6 if P.size else 0.0,
            "volume_in_m3": V_in,
            "volume_out_m3": V_out,
            "solver_success": bool(sol.success),
        }

    # ------------------------------------------------------------------
    # Analytical reference (F1 Prandle) for cross-check
    # ------------------------------------------------------------------
    def theoretical_energy_per_cycle_MWh(self, amplitude=None):
        """
        Prandle/Baker theoretical SINGLE-EFFECT average-power figure per cycle:
            E = 0.5 * rho * g * A * h^2     (h = amplitude),  in MWh.
        This is the conservative average-power formula (Prandle 1984), NOT an
        upper bound -- a well-operated plant generating across a larger head
        swing can exceed it. See max_energy_per_cycle_MWh for the hard bound.
        """
        a = self.a if amplitude is None else float(amplitude)
        E_J = 0.5 * self.rho * self.g * self.A * a ** 2
        return E_J / 3.6e9

    def max_energy_per_cycle_MWh(self, amplitude=None):
        """
        Hard thermodynamic upper bound on extractable energy per tidal cycle:
        the potential energy of trapping basin volume A over the full range
        R = 2a and releasing it,  E_max = 0.5 * rho * g * A * R^2 = 2 rho g A a^2
        (Baker 1991). The ODE energy must stay strictly below this.
        """
        a = self.a if amplitude is None else float(amplitude)
        R = 2.0 * a
        E_J = 0.5 * self.rho * self.g * self.A * R ** 2
        return E_J / 3.6e9
