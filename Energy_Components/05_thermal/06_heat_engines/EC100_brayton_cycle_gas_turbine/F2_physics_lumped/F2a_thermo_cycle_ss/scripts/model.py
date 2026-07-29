"""
EC100 -- Brayton Cycle Gas Turbine -- F2a Physics-Lumped Air-Standard Cycle

Physics-lumped (0D) first-principles model of a simple-cycle gas turbine using
the air-standard Brayton cycle with VARIABLE specific heats cp(T), component
isentropic efficiencies, a turbine-inlet-temperature (TIT) metallurgical limit,
and optional regeneration / intercooling. A lumped rotor/shaft transient ODE
gives spool (shaft-speed) dynamics, integrated with scipy.integrate.solve_ivp.

Station numbering (simple cycle):
    1  compressor inlet (ambient)
    2  compressor exit
    3  turbine inlet (combustor exit, TIT)
    4  turbine exit (exhaust)
    5  recuperator cold-side exit (if regeneration)

Cycle equations (Moran & Shapiro 2014, Ch.9; Saravanamuttoo et al. 2009, Ch.2):

  Isentropic relations with variable cp use the relative-pressure / enthalpy
  formulation. Here we use the temperature-ratio form with a locally evaluated
  gamma(T) = cp(T) / (cp(T) - R) so that variable-cp behaviour is captured while
  staying algebraic:

    Compression (1->2):
      T2s = T1 * PR^((gamma-1)/gamma)            ideal (isentropic)
      T2  = T1 + (T2s - T1) / eta_c              real (isentropic efficiency)
      w_c = cp_bar(T1,T2) * (T2 - T1)            specific compressor work [J/kg]

    Combustion (2->3):
      T3  = TIT  (capped at metallurgical limit)
      q_in = cp_bar(T2,T3) * (T3 - T2) / eta_comb   fuel heat [J/kg]
      mdot_fuel = mdot_air * q_in / LHV

    Expansion (3->4):
      PR_t = PR * (1 - dP_comb)                  turbine pressure ratio (with loss)
      T4s = T3 / PR_t^((gamma-1)/gamma)          ideal
      T4  = T3 - eta_t * (T3 - T4s)              real
      w_t = cp_bar(T3,T4) * (T3 - T4)            specific turbine work [J/kg]

    Net specific work / efficiency:
      w_net = w_t - w_c
      eta_th = w_net / q_in
      W_net = mdot_air * w_net                    shaft power [W]

  Regeneration (recuperator, effectiveness eps):
      T2' = T2 + eps * (T4 - T2)                 preheat compressed air with exhaust
      q_in is then computed from T2' -> T3, raising eta_th (Moran & Shapiro Eq.9.??).

Variable specific heat cp(T) of air / combustion gas [J/(kg.K)]:
  Polynomial fit valid ~250-1800 K, from NASA Glenn thermodynamic data
  (McBride, Zehe & Gordon 2002) and Cengel & Boles (2015) ideal-gas air tables:

      cp(T) = 1034.09 - 0.2849*T + 7.817e-4*T^2 - 4.971e-7*T^3 + 1.077e-10*T^4

  This reproduces cp(300K)=1.005, cp(1000K)=1.142, cp(1500K)=1.211 kJ/kg.K to
  within ~1%. R_air = 287.0 J/(kg.K) is taken constant (ideal gas).

Spool (rotor) transient ODE (Saravanamuttoo et al. 2009, Ch.8, shaft dynamics;
Newton's 2nd law for rotation):

      I * dOmega/dt = (W_t - W_c - W_load) / Omega

  where I is the lumped polar moment of inertia, Omega the shaft angular speed,
  W_t turbine power, W_c compressor power and W_load the electrical/load torque
  power demand. A positive net torque accelerates the spool; this is the standard
  single-spool acceleration model used for start-up / load-rejection studies.

References:
  Moran, M.J. & Shapiro, H.N. (2014). Fundamentals of Engineering
      Thermodynamics, 8th ed., Wiley, Ch.9 (Gas Power Systems).
  Saravanamuttoo, H.I.H., Rogers, G.F.C., Cohen, H. & Straznicky, P. (2009).
      Gas Turbine Theory, 6th ed., Pearson, Ch.2-3, Ch.8.
  Cengel, Y.A. & Boles, M.A. (2015). Thermodynamics: An Engineering Approach,
      8th ed., McGraw-Hill (air property tables A-17).
  McBride, B.J., Zehe, M.J. & Gordon, S. (2002). NASA/TP-2002-211556.
"""

import numpy as np
from scipy.integrate import solve_ivp


class Brayton_F2a:
    """Air-standard Brayton cycle gas turbine with variable cp and spool ODE."""

    R_air = 287.0  # J/(kg.K), specific gas constant for air

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_MW"]["value"] * 1e6          # W
        self.eta_rated = u["eta_rated"]["value"]
        self.PR = u["pressure_ratio"]["value"]
        self.eta_c = u["eta_compressor"]["value"]
        self.eta_t = u["eta_turbine"]["value"]
        self.eta_comb = u["eta_combustor"]["value"]
        self.T1 = u["T_inlet_K"]["value"]                      # K
        self.P1 = u["P_inlet_kPa"]["value"]                    # kPa
        self.TIT = u["TIT_K"]["value"]                         # K
        self.mdot_air = u["mdot_air_kg_s"]["value"]            # kg/s
        self.LHV = u["LHV_fuel_MJ_kg"]["value"] * 1e6          # J/kg
        self.dP_comb = u["dP_combustor_frac"]["value"]
        self.regen_eps = u["regen_effectiveness"]["value"]
        self.I_spool = u["I_spool_kg_m2"]["value"]             # kg.m2
        self.N_rated = u["N_rated_rpm"]["value"]               # rpm
        self.omega_rated = self.N_rated * 2.0 * np.pi / 60.0   # rad/s

    # ------------------------------------------------------------------
    # Variable specific heat cp(T) of air / combustion gas [J/(kg.K)]
    # ------------------------------------------------------------------
    def cp(self, T):
        """cp of air [J/(kg.K)] -- NASA Glenn / Cengel polynomial fit."""
        return (1034.09 - 0.2849 * T + 7.817e-4 * T**2
                - 4.971e-7 * T**3 + 1.077e-10 * T**4)

    def cp_bar(self, Ta, Tb, n=8):
        """Mean cp over [Ta,Tb] by composite-trapezoid integration of cp(T)."""
        Ts = np.linspace(Ta, Tb, n)
        return float(np.trapz(self.cp(Ts), Ts) / (Tb - Ta)) if Tb != Ta else float(self.cp(Ta))

    def gamma(self, T):
        """Local ratio of specific heats gamma(T) = cp / (cp - R)."""
        cp = self.cp(T)
        return cp / (cp - self.R_air)

    # ------------------------------------------------------------------
    # Cycle station temperatures
    # ------------------------------------------------------------------
    def compress(self, T1, PR):
        """Return (T2, w_c[J/kg]) for real compression 1->2."""
        g = self.gamma(T1)
        T2s = T1 * PR ** ((g - 1.0) / g)
        T2 = T1 + (T2s - T1) / self.eta_c
        w_c = self.cp_bar(T1, T2) * (T2 - T1)
        return T2, w_c, T2s

    def expand(self, T3, PR_t):
        """Return (T4, w_t[J/kg]) for real expansion 3->4."""
        g = self.gamma(T3)
        T4s = T3 / PR_t ** ((g - 1.0) / g)
        T4 = T3 - self.eta_t * (T3 - T4s)
        w_t = self.cp_bar(T3, T4) * (T3 - T4)
        return T4, w_t, T4s

    # ------------------------------------------------------------------
    # Full steady-state cycle
    # ------------------------------------------------------------------
    def cycle(self, PR=None, TIT=None, T1=None, regen_eps=None, mdot_air=None):
        """
        Evaluate the air-standard Brayton cycle at given operating point.

        Returns dict of station temps, specific works, efficiency, powers,
        fuel flow and a Carnot ceiling for the same T-limits.
        """
        PR = self.PR if PR is None else float(PR)
        TIT = min(self.TIT if TIT is None else float(TIT), self.TIT)
        T1 = self.T1 if T1 is None else float(T1)
        eps = self.regen_eps if regen_eps is None else float(regen_eps)
        mdot = self.mdot_air if mdot_air is None else float(mdot_air)

        # 1->2 compression
        T2, w_c, T2s = self.compress(T1, PR)
        # 3->4 expansion (turbine sees pressure ratio reduced by combustor loss)
        T3 = TIT
        PR_t = PR * (1.0 - self.dP_comb)
        T4, w_t, T4s = self.expand(T3, PR_t)

        # Optional regeneration: preheat air 2->2' with exhaust (only if T4>T2)
        T2_eff = T2
        if eps > 0.0 and T4 > T2:
            T2_eff = T2 + eps * (T4 - T2)

        # 2'->3 combustion heat addition
        q_in = self.cp_bar(T2_eff, T3) * (T3 - T2_eff) / self.eta_comb
        q_in = max(q_in, 1e-9)

        w_net = w_t - w_c
        eta_th = w_net / q_in

        W_net = mdot * w_net
        W_c = mdot * w_c
        W_t = mdot * w_t
        Q_in = mdot * q_in
        mdot_fuel = Q_in / self.LHV
        Q_exhaust = mdot * self.cp_bar(T1, T4) * (T4 - T1)  # rel. to ambient

        eta_carnot = 1.0 - T1 / T3

        return {
            "T1_K": T1, "T2_K": T2, "T2s_K": T2s, "T2_eff_K": T2_eff,
            "T3_K": T3, "T4_K": T4, "T4s_K": T4s,
            "w_compressor_J_kg": w_c,
            "w_turbine_J_kg": w_t,
            "w_net_J_kg": w_net,
            "q_in_J_kg": q_in,
            "eta_thermal": eta_th,
            "eta_carnot": eta_carnot,
            "W_net_W": W_net, "W_compressor_W": W_c, "W_turbine_W": W_t,
            "Q_in_W": Q_in, "Q_exhaust_W": Q_exhaust,
            "mdot_air_kg_s": mdot, "mdot_fuel_kg_s": mdot_fuel,
            "back_work_ratio": w_c / w_t if w_t > 0 else float("inf"),
            "pressure_ratio": PR,
            "regen_effectiveness": eps,
        }

    # ------------------------------------------------------------------
    # Optimal pressure ratio (max specific net work) -- cold-air ideal:
    #   PR_opt = (T3/T1)^(gamma/(2(gamma-1)))   (Moran & Shapiro, Saravanamuttoo)
    # ------------------------------------------------------------------
    def optimal_pressure_ratio(self, TIT=None, T1=None):
        """Analytic PR that maximises specific net work (cold-air-standard)."""
        TIT = self.TIT if TIT is None else float(TIT)
        T1 = self.T1 if T1 is None else float(T1)
        g = self.gamma(0.5 * (T1 + TIT))
        return (TIT / T1) ** (g / (2.0 * (g - 1.0)))

    # ------------------------------------------------------------------
    # Lumped spool / rotor transient ODE via solve_ivp
    #   I * dOmega/dt = (W_turbine - W_compressor - W_load) / Omega
    # ------------------------------------------------------------------
    def simulate_spool(self, load_power, omega0=None, t_end=20.0, n_eval=200,
                       PR=None, TIT=None, T1=None):
        """
        Integrate single-spool shaft dynamics under a (possibly time-varying)
        electrical load. The aero powers are recomputed at each Omega assuming
        mass flow scales linearly with shaft speed (Omega/Omega_rated), a
        standard lumped-parameter approximation (Saravanamuttoo Ch.8).

        Parameters
        ----------
        load_power : float or callable(t)->W  -- electrical load (W)
        omega0     : initial shaft speed [rad/s] (default rated)
        t_end      : integration horizon [s]
        Returns dict: t, omega, rpm, speed_fraction, W_turbine, W_compressor, W_load
        """
        if omega0 is None:
            omega0 = self.omega_rated
        load_fn = load_power if callable(load_power) else (lambda t: float(load_power))
        PR_op = self.PR if PR is None else float(PR)
        TIT_op = self.TIT if TIT is None else float(TIT)
        T1_op = self.T1 if T1 is None else float(T1)

        def powers(omega):
            frac = max(omega / self.omega_rated, 1e-3)
            mdot = self.mdot_air * frac
            st = self.cycle(PR=PR_op, TIT=TIT_op, T1=T1_op, mdot_air=mdot)
            return st["W_turbine_W"], st["W_compressor_W"]

        # Floor: shaft cannot physically spin below ~1% of rated (flame-out /
        # standstill). Below it net torque is held at/below zero so Omega
        # neither diverges nor goes negative through the 1/Omega singularity.
        omega_floor = 0.01 * self.omega_rated

        def rhs(t, y):
            omega = y[0]
            if omega <= omega_floor:
                return [0.0]  # clamp at floor: machine has decelerated to standstill
            W_t, W_c = powers(omega)
            W_load = load_fn(t)
            domega = (W_t - W_c - W_load) / (self.I_spool * omega)
            return [domega]

        # Event: shaft decelerates down to the floor -> stop integration cleanly.
        def hit_floor(t, y):
            return y[0] - omega_floor
        hit_floor.terminal = True
        hit_floor.direction = -1

        t_eval = np.linspace(0.0, t_end, n_eval)
        sol = solve_ivp(rhs, (0.0, t_end), [omega0], t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-4,
                        max_step=t_end / 50.0, events=hit_floor)

        omega = np.clip(sol.y[0], omega_floor, None)
        Wt = np.array([powers(w)[0] for w in omega])
        Wc = np.array([powers(w)[1] for w in omega])
        Wl = np.array([load_fn(t) for t in sol.t])
        return {
            "t": sol.t,
            "omega": omega,
            "rpm": omega * 60.0 / (2.0 * np.pi),
            "speed_fraction": omega / self.omega_rated,
            "W_turbine": Wt,
            "W_compressor": Wc,
            "W_load": Wl,
            "success": bool(sol.success),
        }
