"""
EC042 -- Pseudocapacitor -- F2a Physics-Lumped RC-Faradaic Model

Physics-lumped 0D model of a pseudocapacitor (e.g. RuO2 / MnO2 / conducting
polymer) that combines:

  1. Electrostatic double-layer capacitance C_dl (non-faradaic, ~voltage
     independent), and
  2. A voltage-dependent FARADAIC pseudocapacitance C_far(V) arising from fast,
     reversible proton-coupled redox surface reactions, e.g.

         RuO2 + H+ + e- <-> RuO(OH)        (proton-coupled electron transfer)

     RuO2 shows a broad, near-rectangular cyclic-voltammogram because many
     overlapping surface redox transitions span the 0-1 V window. We model the
     differential pseudocapacitance as a Gaussian-broadened hump centred on the
     redox mid-potential (Conway 1999, Ch. 10-11):

         C_far(V) = C_far0 * exp( -((V - V_redox)/V_width)^2 )

     so the total differential capacitance is

         C_diff(V,T) = [C_dl + C_far(V)] * (1 + alpha_C*(T - T_ref))

State / charge conservation
---------------------------
The internal (capacitor-node) voltage V_cap evolves so that charge is conserved.
With current sign convention I>0 = discharge:

     C_diff(V_cap,T) * dV_cap/dt = -(I + I_leak)
     I_leak = V_cap / R_leak                       (faradaic self-discharge)

This is a nonlinear ODE because C_diff depends on V_cap. Energy stored is the
integral of charge over voltage:

     E(V) = integral_0^V C_diff(u) du
          = 0.5*C_dl*V^2
          + C_far0 * 0.5*sqrt(pi)*V_width * [ erf((V-V_redox)/V_width)
                                              - erf((-V_redox)/V_width) ]

RC ladder + redox kinetics (high-rate limitation)
-------------------------------------------------
Pseudocapacitive charge has two access paths (two-branch RC ladder, Conway 1999
Ch. 14 porous-electrode TLM reduced to 2 RC):
  * a FAST branch: accessible outer surface, reached through ESR only;
  * a SLOW branch: porous / inner redox sites reached through an extra
    charge-transfer (redox) resistance R_ct.
At high rate the slow branch cannot keep up, so the *accessible* capacitance
falls -- the hallmark rate-dependence of pseudocapacitors vs EDLCs. We capture
this with a frequency-/rate-dependent accessible-capacitance factor derived from
the slow branch time constant tau_ct = R_ct * C_far:

     f_access(I) = 1 / (1 + (|I| * R_ct * C_far(V_cap,T) / V_max)^2 ) ... (rate fade)

Both R_ct and ESR are temperature-activated (Arrhenius); redox kinetics
(R_ct, E_a_ct ~ 20 kJ/mol) are more strongly activated than ohmic transport.

Terminal voltage
----------------
     V_term = V_cap - I * ESR(T) - I_slow * R_ct(T)
where I_slow is the share of current served by the slow (redox) branch.

Thermal ODE (lumped, scipy.solve_ivp)
-------------------------------------
     m*cp * dT/dt = Q_gen - Q_cool
     Q_gen = I^2*ESR(T) + I_slow^2*R_ct(T)     (irreversible Joule + redox)
           + I * V_cap * |dOCV_dT|             (reversible entropic redox heat)
     Q_cool = hA*(T - T_ambient)

The entropic term is the partial battery-like character of pseudocapacitance
(absent in a pure EDLC). dOCV_dT < 0 for proton insertion, so discharge (I>0)
releases extra heat.

Enforced physics
----------------
  * Charge conservation in the V_cap ODE.
  * Round-trip efficiency strictly in (0,1) (ESR + R_ct + leakage losses).
  * Energy bounds: 0 <= E_stored <= E(V_max).
  * Thermal balance: dT/dt -> 0 when Q_gen == Q_cool.
  * Higher pseudocapacitance than the bare EDLC (C_dl + C_far > C_dl).

References
----------
  Conway, B. E. (1999). Electrochemical Supercapacitors: Scientific Fundamentals
      and Technological Applications. Kluwer/Plenum. (Chs. 10, 11, 14)
  Trasatti, S. & Buzzanca, G. (1971). J. Electroanal. Chem. 29, A1-A5.
  Zheng, J. P., Cygan, P. J. & Jow, T. R. (1995). J. Electrochem. Soc. 142, 2699.
  Sugimoto, W. et al. (2006). Electrochim. Acta 52, 1742-1748.
  Simon, P. & Gogotsi, Y. (2008). Nature Materials 7, 845-854.
"""

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import erf


class PseudocapacitorF2a:
    """Physics-lumped pseudocapacitor: RC ladder + voltage-dependent faradaic C(V)."""

    def __init__(self, params: dict):
        cell = params["cell"]
        therm = params["thermal"]

        # --- capacitance structure ---
        self.C_dl = cell["C_dl"]["value"]              # F, double-layer (non-faradaic)
        self.C_far0 = cell["C_far0"]["value"]          # F, faradaic peak scale
        self.v_redox = cell["v_redox"]["value"]        # V, redox centre
        self.v_width = cell["v_width"]["value"]        # V, redox half-width

        # --- resistances ---
        self.ESR_ref = cell["esr_ref"]["value"]        # Ohm, ohmic series
        self.R_ct_ref = cell["R_ct_ref"]["value"]      # Ohm, redox charge-transfer
        self.R_leak = cell["leakage_resistance"]["value"]  # Ohm

        # --- voltage window ---
        self.v_max = cell["v_max"]["value"]
        self.v_min = cell["v_min"]["value"]

        # --- thermal ---
        self.T_ref = therm["T_ref"]["value"]
        self.E_a_esr = therm["E_a_esr"]["value"]
        self.E_a_ct = therm["E_a_ct"]["value"]
        self.alpha_C = therm["alpha_C"]["value"]
        self.dOCV_dT = therm["dOCV_dT_specific"]["value"]
        self.m_cell = therm["m_cell"]["value"]
        self.cp_cell = therm["cp_cell"]["value"]
        self.hA_cool = therm["hA_cool"]["value"]
        self.T_ambient = therm["T_ambient"]["value"]
        self.R_gas = therm["R_gas"]["value"]

    # ------------------------------------------------------------------
    # Voltage-dependent capacitance
    # ------------------------------------------------------------------
    def faradaic_capacitance(self, v_cap, temperature=None):
        """
        Faradaic (pseudo) differential capacitance C_far(V) [F].
        Gaussian-broadened redox hump centred at v_redox (Conway 1999).
        """
        v = np.asarray(v_cap, dtype=float)
        c = self.C_far0 * np.exp(-((v - self.v_redox) / self.v_width) ** 2)
        if temperature is not None:
            c = c * (1.0 + self.alpha_C * (np.asarray(temperature, dtype=float) - self.T_ref))
        return c

    def differential_capacitance(self, v_cap, temperature):
        """
        Total differential capacitance C_diff(V,T) = [C_dl + C_far(V)] * thermal_factor [F].
        Always > C_dl (pseudocapacitor stores more than a bare EDLC).
        """
        v = np.asarray(v_cap, dtype=float)
        T = np.asarray(temperature, dtype=float)
        thermal_factor = 1.0 + self.alpha_C * (T - self.T_ref)
        return (self.C_dl + self.C_far0 * np.exp(-((v - self.v_redox) / self.v_width) ** 2)) * thermal_factor

    # ------------------------------------------------------------------
    # Resistances (Arrhenius temperature dependence)
    # ------------------------------------------------------------------
    def esr(self, temperature):
        """Series ohmic resistance ESR(T) [Ohm]."""
        T = np.asarray(temperature, dtype=float)
        return self.ESR_ref * np.exp(self.E_a_esr / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    def r_ct(self, temperature):
        """Redox charge-transfer resistance R_ct(T) [Ohm] (slow branch, strongly activated)."""
        T = np.asarray(temperature, dtype=float)
        return self.R_ct_ref * np.exp(self.E_a_ct / self.R_gas * (1.0 / T - 1.0 / self.T_ref))

    # ------------------------------------------------------------------
    # Two-branch RC ladder: rate-dependent accessible pseudocapacitance
    # ------------------------------------------------------------------
    def access_factor(self, v_cap, current, temperature):
        """
        Fraction of faradaic capacitance reachable at the given rate (0,1].

        The slow porous/redox branch has an access time constant
        tau_ct = R_ct*C_far. It can keep up only while the galvanostatic
        discharge timescale t_dis = C_total*V_max / |I| stays long compared to
        tau_ct. The relevant Damkoehler-type group is therefore

            x = tau_ct / t_dis = |I| * R_ct * C_far / (C_total * V_max)

        so the characteristic fade current is I_char ~ C_total*V_max/(R_ct*C_far)
        (~60 A for this 200 F / 1 V module). When x >> 1 the redox sites are
        starved and the accessible faradaic capacitance collapses -- the
        rate-dependence that distinguishes pseudocapacitors from rate-flat EDLCs.
        """
        C_f = self.faradaic_capacitance(v_cap, temperature)
        Rct = self.r_ct(temperature)
        C_total = self.C_dl + C_f
        x = np.abs(np.asarray(current, dtype=float)) * Rct * C_f / (C_total * self.v_max)
        return 1.0 / (1.0 + x ** 2)

    def slow_branch_current(self, v_cap, current, temperature):
        """
        Current served by the slow (redox) branch [A]. The remainder is served
        by the fast (double-layer + outer surface) branch. The slow share scales
        with the fraction of total capacitance that is faradaic *and* accessible.
        """
        I = np.asarray(current, dtype=float)
        C_f = self.faradaic_capacitance(v_cap, temperature)
        f_acc = self.access_factor(v_cap, current, temperature)
        C_total = self.C_dl + C_f
        far_share = (C_f * f_acc) / C_total
        return I * far_share

    # ------------------------------------------------------------------
    # Charge, energy, SOC
    # ------------------------------------------------------------------
    def soc(self, v_cap):
        """SOC = V_cap / V_max in [0,1]."""
        v = np.asarray(v_cap, dtype=float)
        return np.clip(v / self.v_max, 0.0, 1.0)

    def charge(self, v_cap, temperature):
        """
        Stored charge Q(V,T) [C] = integral_0^V C_diff(u) du
            = C_dl*V + C_far0*0.5*sqrt(pi)*V_width*[erf((V-Vr)/w) - erf(-Vr/w)]
        (thermal factor applied multiplicatively).
        """
        v = np.asarray(v_cap, dtype=float)
        T = np.asarray(temperature, dtype=float)
        w = self.v_width
        vr = self.v_redox
        q_dl = self.C_dl * v
        q_far = self.C_far0 * 0.5 * np.sqrt(np.pi) * w * (erf((v - vr) / w) - erf((-vr) / w))
        thermal_factor = 1.0 + self.alpha_C * (T - self.T_ref)
        return (q_dl + q_far) * thermal_factor

    def stored_energy(self, v_cap, temperature):
        """
        Stored electrostatic + faradaic energy E(V,T) [J] = integral_0^V u*C_diff(u) du.
        Numerically integrated (the faradaic part has no simple closed form for
        the u*Gaussian moment over a finite window).
        """
        v = float(np.asarray(v_cap, dtype=float))
        T = float(np.asarray(temperature, dtype=float))
        if v <= 0.0:
            return 0.0
        u = np.linspace(0.0, v, 400)
        c = self.differential_capacitance(u, T)
        return float(trapezoid(u * c, u))

    def energy_max(self, temperature):
        """Maximum storable energy E(V_max,T) [J] -- upper energy bound."""
        return self.stored_energy(self.v_max, temperature)

    # ------------------------------------------------------------------
    # Currents and terminal quantities
    # ------------------------------------------------------------------
    def leakage_current(self, v_cap):
        """Faradaic self-discharge leakage current [A]. I_leak = V_cap / R_leak."""
        return np.asarray(v_cap, dtype=float) / self.R_leak

    def terminal_voltage(self, v_cap, current, temperature):
        """
        Terminal voltage [V]:
            V_term = V_cap - I*ESR(T) - I_slow*R_ct(T)
        clipped to the rated window [v_min, v_max].
        """
        v = np.asarray(v_cap, dtype=float)
        I = np.asarray(current, dtype=float)
        I_slow = self.slow_branch_current(v_cap, current, temperature)
        v_term = v - I * self.esr(temperature) - I_slow * self.r_ct(temperature)
        return np.clip(v_term, self.v_min, self.v_max)

    def power(self, v_cap, current, temperature):
        """Terminal power [W]; positive = discharging."""
        return self.terminal_voltage(v_cap, current, temperature) * np.asarray(current, dtype=float)

    def heat_generation(self, v_cap, current, temperature):
        """
        Heat generation [W]:
            Q = I^2*ESR(T)            (ohmic Joule)
              + I_slow^2*R_ct(T)      (redox charge-transfer Joule)
              + I*V_cap*|dOCV_dT|     (reversible entropic redox heat)
        All terms >= 0 on discharge; the irreversible terms are always >= 0.
        """
        v = np.asarray(v_cap, dtype=float)
        I = np.asarray(current, dtype=float)
        I_slow = self.slow_branch_current(v_cap, current, temperature)
        q_ohmic = I ** 2 * self.esr(temperature)
        q_redox = I_slow ** 2 * self.r_ct(temperature)
        q_entropic = I * v * (-self.dOCV_dT)
        return q_ohmic + q_redox + q_entropic

    # ------------------------------------------------------------------
    # Coupled state derivatives (charge conservation + thermal ODE)
    # ------------------------------------------------------------------
    def vcap_derivative(self, v_cap, current, temperature):
        """
        dV_cap/dt from charge conservation:
            C_diff(V,T) * dV/dt = -(I + I_leak)
        => dV/dt = -(I + V/R_leak) / C_diff(V,T).
        """
        v = np.asarray(v_cap, dtype=float)
        I = np.asarray(current, dtype=float)
        i_total = I + self.leakage_current(v)
        return -i_total / self.differential_capacitance(v, temperature)

    def dTdt(self, v_cap, current, temperature):
        """Lumped thermal ODE: dT/dt = (Q_gen - Q_cool)/(m*cp)."""
        Q_gen = self.heat_generation(v_cap, current, temperature)
        Q_cool = self.hA_cool * (temperature - self.T_ambient)
        return (Q_gen - Q_cool) / (self.m_cell * self.cp_cell)

    # ------------------------------------------------------------------
    # Time-domain simulation (scipy.solve_ivp)
    # ------------------------------------------------------------------
    def simulate(self, current, v_cap0, T0, dt, duration_s):
        """
        Simulate coupled (V_cap, T) dynamics with charge conservation + thermal ODE.

        Parameters
        ----------
        current : float or callable(t)
            Applied current [A]. >0 = discharge, <0 = charge.
        v_cap0 : float
            Initial capacitor-node voltage [V].
        T0 : float
            Initial cell temperature [K].
        dt : float
            Output time step [s].
        duration_s : float
            Total simulation duration [s].

        Returns
        -------
        dict of time-series arrays:
            t, v_cap, terminal_voltage, current, power, soc, temperature,
            stored_energy, capacitance, heat, components(dict)
        """
        _I = current if callable(current) else (lambda t: current)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            v, T = y[0], y[1]
            # clip voltage to the physical window inside the derivative so the
            # integrator cannot push the state out of [v_min, v_max]
            v_eff = min(max(v, self.v_min), self.v_max)
            I = _I(t)
            return [self.vcap_derivative(v_eff, I, T), self.dTdt(v_eff, I, T)]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [v_cap0, T0],
            t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10, max_step=dt,
        )

        t_out = sol.t
        v_out = np.clip(sol.y[0], self.v_min, self.v_max)
        T_out = sol.y[1]
        N = len(t_out)

        v_term = np.zeros(N)
        power = np.zeros(N)
        soc = np.zeros(N)
        E_store = np.zeros(N)
        C_diff = np.zeros(N)
        C_far = np.zeros(N)
        heat = np.zeros(N)
        i_arr = np.zeros(N)
        f_acc = np.zeros(N)

        for i in range(N):
            I = _I(t_out[i])
            v = v_out[i]
            T = T_out[i]
            i_arr[i] = I
            v_term[i] = self.terminal_voltage(v, I, T)
            power[i] = v_term[i] * I
            soc[i] = self.soc(v)
            E_store[i] = self.stored_energy(v, T)
            C_diff[i] = self.differential_capacitance(v, T)
            C_far[i] = self.faradaic_capacitance(v, T)
            heat[i] = self.heat_generation(v, I, T)
            f_acc[i] = self.access_factor(v, I, T)

        return {
            "t": t_out,
            "v_cap": v_out,
            "terminal_voltage": v_term,
            "current": i_arr,
            "power": power,
            "soc": soc,
            "temperature": T_out,
            "stored_energy": E_store,
            "capacitance": C_diff,
            "heat": heat,
            "components": {
                "faradaic_capacitance": C_far,
                "access_factor": f_acc,
            },
        }

    # ------------------------------------------------------------------
    # Round-trip efficiency (galvanostatic charge then discharge)
    # ------------------------------------------------------------------
    def round_trip_efficiency(self, current, v_start, T0, dt=0.01):
        """
        Coulombic-energy round-trip efficiency for a charge then discharge cycle
        at constant |current|. Returns eta in (0,1).

            eta = E_discharge_out / E_charge_in
        """
        I = abs(current)
        # --- charge from v_start up to ~v_max ---
        ch = self._ramp(-I, v_start, T0, dt, target="up")
        E_in = self._energy_throughput(ch, sign_in=True)
        v_top = ch["v_cap"][-1]
        T_top = ch["temperature"][-1]
        # --- discharge from v_top back down toward v_start ---
        dis = self._ramp(I, v_top, T_top, dt, target="down", v_stop=v_start)
        E_out = self._energy_throughput(dis, sign_in=False)
        if E_in <= 0:
            return 0.0
        return float(np.clip(E_out / E_in, 0.0, 1.0))

    def _ramp(self, current, v0, T0, dt, target="up", v_stop=None):
        """Integrate until the voltage crosses a window bound (or v_stop)."""
        if target == "up":
            stop = self.v_max if v_stop is None else v_stop
        else:
            stop = self.v_min if v_stop is None else v_stop

        def event(t, y):
            return y[0] - stop
        event.terminal = True
        event.direction = 1 if target == "up" else -1

        def rhs(t, y):
            v = min(max(y[0], self.v_min), self.v_max)
            return [self.vcap_derivative(v, current, y[1]), self.dTdt(v, current, y[1])]

        # generous time horizon: C*V/I scale
        t_max = 5.0 * (self.C_dl + self.C_far0) * self.v_max / max(abs(current), 1e-6)
        t_eval = np.arange(0.0, t_max, dt)
        sol = solve_ivp(rhs, (0.0, t_max), [v0, T0], t_eval=t_eval,
                        events=event, method="RK45", rtol=1e-7, atol=1e-9, max_step=dt)
        v = np.clip(sol.y[0], self.v_min, self.v_max)
        T = sol.y[1]
        out = {"t": sol.t, "v_cap": v, "temperature": T,
               "current": np.full(len(sol.t), current)}
        return out

    def _energy_throughput(self, traj, sign_in):
        """Integrate terminal power over the trajectory -> energy [J] (>0)."""
        t = traj["t"]
        if len(t) < 2:
            return 0.0
        p = np.array([self.terminal_voltage(traj["v_cap"][i], traj["current"][i], traj["temperature"][i])
                      * traj["current"][i] for i in range(len(t))])
        # charge: current<0 (discharge convention >0). For charge, power<0 ->
        # energy IN is -integral(p). For discharge, power>0 -> energy OUT.
        E = trapezoid(p, t)
        return abs(E)
