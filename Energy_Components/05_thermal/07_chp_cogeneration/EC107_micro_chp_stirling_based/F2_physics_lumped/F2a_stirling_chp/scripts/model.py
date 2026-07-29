"""
EC107 -- Micro-CHP (Stirling-based) -- F2a Physics-Lumped Model

Burner-driven Stirling-engine micro-CHP. A premixed gas burner heats the
engine heater head; a Stirling cycle converts part of that heat to ~1 kWe
of electricity, and BOTH the rejected cycle heat (cooler) AND the recovered
flue-gas sensible heat are delivered to the domestic heating circuit. This
is a "heat-led" cogeneration unit: electrical efficiency is low (~10-15 %),
thermal efficiency is high (~75-80 %), total (CHP) efficiency ~85-90 %.

Indicated power (two complementary first-principles estimators):
  1. Beale number   :  P_ind = Bn * p_mean * V_swept * f          (West 1986)
  2. Schmidt/Carnot :  eta_ind = carnot_fraction * (1 - T_cold/T_hot)
The Beale correlation sets the size of the machine; the Carnot-fraction sets
the *efficiency ceiling* and is enforced as eta_elec < eta_Carnot.

Energy balance (per unit time, LHV fuel basis):
    Q_fuel  = burner fuel input (LHV)
    Q_head  = eta_burner * Q_fuel          heat delivered to heater head
    Q_flue  = (1 - eta_burner) ... plus combustion sensible heat in flue
    P_ind   = eta_ind * Q_head             indicated (mechanical) power
    P_elec  = eta_mech * P_ind             electrical output
    Q_reject= Q_head - P_ind               heat rejected by Stirling cooler
    Q_flue_rec = eta_flue_recovery * Q_flue_sensible
    Q_th    = Q_reject + Q_flue_rec        useful heat to water circuit
    eta_e   = P_elec / Q_fuel
    eta_th  = Q_th   / Q_fuel
    eta_tot = eta_e + eta_th               (< 1 ; balance = stack/casing loss)

Thermal warm-up ODE (lumped heater-head + block + water jacket):
    m*cp dT/dt = Q_head_to_mass - hA_loss*(T - T_amb) - Q_useful_extracted
The Stirling only produces electricity once the head is hot, so output is
gated on (T - T_cold)/(T_hot_set - T_cold). Integrated with scipy.solve_ivp.

References
----------
Walker, G. (1980). Stirling Engines. Oxford University Press.
Organ, A.J. (1992). Thermodynamics and Gas Dynamics of the Stirling Cycle
    Machine. Cambridge University Press.
West, C.D. (1986). Principles and Applications of Stirling Engines.
    Van Nostrand Reinhold. (Beale-number correlation)
Onovwiona, H.I. & Ugursal, V.I. (2006). Residential cogeneration systems.
    Renewable & Sustainable Energy Reviews 10(5), 389-431.
Conroy, G. et al. (2014). WhisperGen Stirling micro-CHP field trial.
    Applied Energy 122, 194-204.
Cengel, Y. & Boles, M. (2015). Thermodynamics, 8th ed. (cp_water).
Turns, S.R. (2012). An Introduction to Combustion, 3rd ed. (cp_flue-gas).
"""

import numpy as np
from scipy.integrate import solve_ivp


class StirlingCHP_F2a:
    """Physics-lumped burner-driven Stirling micro-CHP with warm-up ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_e_rated = u["P_e_rated_W"]["value"]          # W
        self.T_hot = u["T_hot_K"]["value"]                  # K
        self.T_cold = u["T_cold_K"]["value"]                # K
        self.Bn = u["beale_number"]["value"]                # -
        self.p_mean = u["p_mean_Pa"]["value"]               # Pa
        self.V_swept = u["V_swept_m3"]["value"]             # m3
        self.freq = u["freq_Hz"]["value"]                   # Hz
        self.eta_mech = u["eta_mech"]["value"]              # -
        self.eta_burner = u["eta_burner"]["value"]          # -
        self.eta_flue_rec = u["eta_flue_recovery"]["value"] # -
        self.carnot_frac = u["carnot_fraction"]["value"]    # -
        self.m_th = u["m_thermal_kg"]["value"]              # kg
        self.cp_th = u["cp_thermal_J_kgK"]["value"]         # J/(kg.K)
        self.hA_loss = u["hA_loss_W_K"]["value"]            # W/K
        self.Q_fuel_max = u["Q_fuel_max_W"]["value"]        # W
        self.cp_gas = u["cp_gas_J_kgK"]["value"]            # J/(kg.K)  (cited)
        self.cp_water = u["cp_water_J_kgK"]["value"]        # J/(kg.K)  (cited)
        self.T_amb = u["T_amb_K"]["value"]                  # K

    # ------------------------------------------------------------------
    # Carnot efficiency between cycle hot/cold reservoirs
    # ------------------------------------------------------------------
    def carnot_efficiency(self, T_hot=None, T_cold=None):
        """Carnot upper bound eta_C = 1 - T_cold/T_hot [-]."""
        T_hot = self.T_hot if T_hot is None else T_hot
        T_cold = self.T_cold if T_cold is None else T_cold
        return 1.0 - T_cold / T_hot

    def indicated_efficiency(self, T_hot=None, T_cold=None):
        """Indicated (thermal->mechanical) efficiency of the Stirling cycle.

        eta_ind = carnot_fraction * eta_Carnot  (always < eta_Carnot).
        """
        return self.carnot_frac * self.carnot_efficiency(T_hot, T_cold)

    # ------------------------------------------------------------------
    # Beale-number indicated power (machine sizing, West 1986)
    # ------------------------------------------------------------------
    def beale_power(self):
        """Indicated power from the Beale correlation [W].

        P_ind = Bn * p_mean * V_swept * f
        """
        return self.Bn * self.p_mean * self.V_swept * self.freq

    def rated_fuel_input(self):
        """Burner fuel input (LHV) needed at full fire to reach P_e_rated [W]."""
        eta_e_rated = self.eta_mech * self.indicated_efficiency() * self.eta_burner
        return self.P_e_rated / max(eta_e_rated, 1e-9)

    # ------------------------------------------------------------------
    # Steady-state CHP energy balance at a given thermal load fraction
    # ------------------------------------------------------------------
    def steady_state(self, load_fraction, T_hot=None, T_cold=None):
        """Steady-state electrical/thermal split at a burner load fraction.

        load_fraction in [0,1] scales the burner fuel input from 0..Q_fuel_max.
        Returns a dict of powers [W] and efficiencies [-].
        """
        load = float(np.clip(load_fraction, 0.0, 1.0))
        T_hot = self.T_hot if T_hot is None else T_hot
        T_cold = self.T_cold if T_cold is None else T_cold

        Q_fuel = load * self.Q_fuel_max                       # LHV fuel in
        if Q_fuel <= 0.0:
            return {
                "Q_fuel_W": 0.0, "P_elec_W": 0.0, "Q_th_W": 0.0,
                "Q_reject_W": 0.0, "Q_flue_rec_W": 0.0,
                "eta_elec": 0.0, "eta_th": 0.0, "eta_total": 0.0,
                "eta_carnot": self.carnot_efficiency(T_hot, T_cold),
                "power_to_heat": 0.0, "load_fraction": load,
            }

        eta_C = self.carnot_efficiency(T_hot, T_cold)
        eta_ind = self.carnot_frac * eta_C                    # < eta_C

        # Heat split at the burner: most to heater head, remainder to flue
        Q_head = self.eta_burner * Q_fuel                     # to Stirling head
        Q_flue_sensible = (1.0 - self.eta_burner) * Q_fuel    # flue losses

        # Stirling cycle on the head heat
        P_ind = eta_ind * Q_head                              # indicated (mech)
        P_elec = self.eta_mech * P_ind                        # electrical
        Q_reject = Q_head - P_ind                             # cooler rejection

        # Flue heat recovery (condensing recuperator)
        Q_flue_rec = self.eta_flue_rec * Q_flue_sensible

        Q_th = Q_reject + Q_flue_rec                          # useful heat
        eta_elec = P_elec / Q_fuel
        eta_th = Q_th / Q_fuel
        eta_total = eta_elec + eta_th
        power_to_heat = P_elec / Q_th if Q_th > 0 else 0.0

        return {
            "Q_fuel_W": Q_fuel,
            "P_elec_W": P_elec,
            "Q_th_W": Q_th,
            "Q_reject_W": Q_reject,
            "Q_flue_rec_W": Q_flue_rec,
            "eta_elec": eta_elec,
            "eta_th": eta_th,
            "eta_total": eta_total,
            "eta_carnot": eta_C,
            "power_to_heat": power_to_heat,
            "load_fraction": load,
        }

    # ------------------------------------------------------------------
    # Warm-up gate: electrical output ramps in as the head heats
    # ------------------------------------------------------------------
    def warmup_factor(self, T):
        """Fraction of rated output available given current head temperature.

        Linear gate from T_cold (0 %) to T_hot (100 %), clipped to [0,1].
        Represents the Stirling needing a hot head before it can do work.
        """
        denom = max(self.T_hot - self.T_cold, 1.0)
        return float(np.clip((T - self.T_cold) / denom, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Thermal warm-up ODE derivative
    # ------------------------------------------------------------------
    def thermostat_factor(self, T):
        """Burner modulation toward the heater-head setpoint T_hot [-].

        The premixed burner is thermostatically controlled to hold the head
        at T_hot. As T approaches T_hot the burner throttles down; above the
        setpoint it shuts off. Linear band over [T_cold, T_hot] -> 1 at cold,
        0 at setpoint. This makes T_hot the warm-up asymptote (no overshoot).
        """
        denom = max(self.T_hot - self.T_cold, 1.0)
        return float(np.clip((self.T_hot - T) / denom, 0.0, 1.0))

    def dTdt(self, T, load_fraction):
        """Lumped head temperature rate of change [K/s].

        m*cp dT/dt = theta*Q_head*(1 - eta_ind) - hA_loss*(T - T_amb)

        The burner heat into the head is throttled by the thermostat factor
        theta so the head asymptotes to the setpoint T_hot. The (1 - eta_ind)
        term reflects that the engine's indicated work is exported as
        electricity rather than stored as sensible heat; the remaining head
        heat (cooler reject) is what the lumped metal/jacket mass exchanges
        on its way to thermal equilibrium.
        """
        load = float(np.clip(load_fraction, 0.0, 1.0))
        Q_fuel = load * self.Q_fuel_max
        Q_head = self.eta_burner * Q_fuel

        theta = self.thermostat_factor(T)
        eta_ind = self.indicated_efficiency()

        Q_charge = theta * Q_head * (1.0 - eta_ind)
        Q_loss = self.hA_loss * (T - self.T_amb)

        return (Q_charge - Q_loss) / (self.m_th * self.cp_th)

    # ------------------------------------------------------------------
    # Time-domain warm-up simulation (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def simulate(self, load_fraction, T0_K, dt, duration_s):
        """Simulate cold-start warm-up + CHP output.

        Parameters
        ----------
        load_fraction : float or callable(t) in [0,1]
            Burner firing fraction.
        T0_K : float
            Initial head/block temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Total duration [s].

        Returns
        -------
        dict of time-series arrays: t, temperature, P_elec_W, Q_th_W,
            eta_elec, eta_th, eta_total, warmup_factor.
        """
        _load = (load_fraction if callable(load_fraction)
                 else (lambda t: load_fraction))

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], _load(t))]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [T0_K],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        P_elec = np.zeros(N)
        Q_th = np.zeros(N)
        eta_e = np.zeros(N)
        eta_t = np.zeros(N)
        eta_tot = np.zeros(N)
        gate = np.zeros(N)

        for i in range(N):
            load_i = _load(t_out[i])
            ss = self.steady_state(load_i)
            g = self.warmup_factor(T_out[i])
            gate[i] = g
            # Electrical output is gated by warm-up. When the engine is cold,
            # the indicated work it would have produced is NOT extracted, so
            # that head heat is instead rejected to the water -> energy is
            # conserved (eta_total is invariant to warm-up state).
            P_elec[i] = g * ss["P_elec_W"]
            # Indicated work that was not converted (P_ind = P_elec/eta_mech)
            # stays as heat in the working gas and is rejected by the cooler.
            P_ind_full = ss["P_elec_W"] / max(self.eta_mech, 1e-9)
            Q_unconverted = (1.0 - g) * P_ind_full
            Q_th[i] = ss["Q_th_W"] + Q_unconverted
            Q_fuel_i = ss["Q_fuel_W"]
            eta_e[i] = P_elec[i] / Q_fuel_i if Q_fuel_i > 0 else 0.0
            eta_t[i] = Q_th[i] / Q_fuel_i if Q_fuel_i > 0 else 0.0
            eta_tot[i] = eta_e[i] + eta_t[i]

        return {
            "t": t_out,
            "temperature": T_out,
            "P_elec_W": P_elec,
            "Q_th_W": Q_th,
            "eta_elec": eta_e,
            "eta_th": eta_t,
            "eta_total": eta_tot,
            "warmup_factor": gate,
        }
