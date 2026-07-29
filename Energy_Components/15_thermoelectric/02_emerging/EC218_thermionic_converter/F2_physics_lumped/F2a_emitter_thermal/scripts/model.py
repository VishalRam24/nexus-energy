"""
EC218 -- Thermionic Converter -- F2a Physics-Lumped Emitter-Thermal Model

Physics-lumped (0D) model of a vacuum/Cs-vapor thermionic energy converter
coupling first-principles thermionic emission with a lumped emitter-temperature
ODE solved by scipy.integrate.solve_ivp.

Device physics
--------------
Electrons are thermionically emitted from a hot emitter at temperature T_E and
cross an inter-electrode gap to a cooler collector at T_C, doing electrical work
against an external load.

1. Richardson-Dushman emission law (Richardson 1901; Dushman 1923):
       J_RD(phi, T) = A * T^2 * exp(-q*phi / (k_B * T))            [A/m^2]
   evaluated for both emitter (forward) and collector (back-emission).

2. Net output current density after space-charge / plasma barrier reduction:
       J_net = f_sc * J_E - J_C            (>= 0)
   The factor f_sc lumps the Langmuir space-charge / plasma sheath loss that a
   full motive-diagram (Poisson) calculation would resolve (Langmuir 1923).

3. Output (terminal) voltage. The ideal back-EMF of the diode is the
   work-function difference; the delivered terminal voltage is reduced by the
   plasma arc drop V_d and the lead-resistance ohmic drop:
       V_oc       = phi_E - phi_C                                    [V]
       V_terminal = max(V_oc - V_d - J_net*A_e*R_lead, 0)
   The user may instead clamp an external load voltage V_load.

4. Efficiency = electrical power out / heat into emitter, bounded by Carnot:
       eta       = P_elec / Q_in
       eta_carnot = 1 - T_C / T_E
   Q_in is dominated by the electron-cooling (evaporative) flux plus radiation
   loss to the collector:
       Q_electron = J_E/q * (phi_E + 2*k_B*T_E/q) * q * A_e
                  = J_E * (phi_E + 2*k_B*T_E/q) * A_e
       Q_rad      = emissivity * sigma * A_e * (T_E^4 - T_C^4)

Lumped emitter-temperature ODE
------------------------------
       C_E * dT_E/dt = Q_external - Q_electron(T_E) - Q_rad(T_E)
   where Q_external is the heat power delivered to the emitter (combustion,
   concentrated solar, or radioisotope). Electron cooling and radiation both act
   as negative feedback, so T_E relaxes to a stable steady state.

References
----------
    Hatsopoulos, G.N. & Gyftopoulos, E.P. (1973/1979). Thermionic Energy
        Conversion, Vols. I & II. MIT Press.
    Rasor, N.S. (1991). "Thermionic energy conversion plasmas."
        IEEE Trans. Plasma Sci. 19(6), 1191-1208.
    Angrist, S.W. (1982). Direct Energy Conversion, 4th ed. Allyn & Bacon.
    Langmuir, I. (1923). "The effect of space charge ..." Phys. Rev. 21, 419.
"""

import numpy as np
from scipy.integrate import solve_ivp

k_B = 1.380649e-23       # Boltzmann constant [J/K]
q_e = 1.602176634e-19    # elementary charge [C]


class ThermionicF2a:
    """Physics-lumped thermionic converter with emitter-temperature ODE."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.phi_e0 = u["phi_emitter_0"]["value"]            # eV
        self.phi_c0 = u["phi_collector_0"]["value"]          # eV
        self.A_r = u["A_richardson"]["value"]                # A/(m^2 K^2)
        self.area = u["emitter_area"]["value"]               # m^2
        self.gap = u["interelectrode_gap"]["value"]          # m
        self.T0 = u["T0_K"]["value"]                         # K
        self.dphi_e_dT = u["dphi_dT_emitter"]["value"]       # eV/K
        self.dphi_c_dT = u["dphi_dT_collector"]["value"]     # eV/K
        self.f_sc = u["space_charge_factor"]["value"]        # -
        self.V_plasma = u["plasma_drop_V"]["value"]          # V
        self.R_lead = u["lead_resistance_ohm"]["value"]      # ohm
        self.emis = u["emissivity"]["value"]                 # -
        self.sigma = u["stefan_boltzmann"]["value"]          # W/(m^2 K^4)
        self.C_E = u["emitter_heat_capacity"]["value"]       # J/K

    # ------------------------------------------------------------------
    # Work functions (temperature dependent)
    # ------------------------------------------------------------------
    def phi_emitter(self, T_E):
        """Emitter work function [eV]."""
        return self.phi_e0 + self.dphi_e_dT * (np.asarray(T_E, float) - self.T0)

    def phi_collector(self, T_C):
        """Collector work function [eV]."""
        return self.phi_c0 + self.dphi_c_dT * (np.asarray(T_C, float) - self.T0)

    # ------------------------------------------------------------------
    # Richardson-Dushman emission
    # ------------------------------------------------------------------
    def richardson_current(self, phi_eV, T):
        """Richardson-Dushman saturation current density [A/m^2]."""
        T = np.asarray(T, float)
        phi_J = np.asarray(phi_eV, float) * q_e
        return self.A_r * T ** 2 * np.exp(-phi_J / (k_B * T))

    def net_current_density(self, T_E, T_C):
        """Net output current density [A/m^2] (forward minus back-emission)."""
        phi_e = self.phi_emitter(T_E)
        phi_c = self.phi_collector(T_C)
        J_E = self.richardson_current(phi_e, T_E)
        J_C = self.richardson_current(phi_c, T_C)
        return np.maximum(self.f_sc * J_E - J_C, 0.0)

    # ------------------------------------------------------------------
    # Voltages
    # ------------------------------------------------------------------
    def open_circuit_voltage(self, T_E, T_C):
        """Ideal back-EMF = work-function difference [V]."""
        return np.maximum(self.phi_emitter(T_E) - self.phi_collector(T_C), 0.0)

    def carnot_efficiency(self, T_E, T_C):
        """Carnot limit for the converter [-]."""
        return 1.0 - np.asarray(T_C, float) / np.asarray(T_E, float)

    # ------------------------------------------------------------------
    # Heat fluxes into / out of the emitter
    # ------------------------------------------------------------------
    def electron_cooling_power(self, T_E, T_C):
        """Evaporative electron-cooling heat flux carried off the emitter [W].

        Each emitted electron removes (phi_E + 2 k_B T_E / q) eV of energy
        (work function plus mean thermal kinetic energy 2kT/q).
        """
        phi_e = self.phi_emitter(T_E)
        J_E = self.f_sc * self.richardson_current(phi_e, T_E)
        return J_E * (phi_e + 2.0 * k_B * np.asarray(T_E, float) / q_e) * self.area

    def radiation_power(self, T_E, T_C):
        """Gray-body radiation loss emitter -> collector [W]."""
        T_E = np.asarray(T_E, float)
        T_C = np.asarray(T_C, float)
        return self.emis * self.sigma * self.area * (T_E ** 4 - T_C ** 4)

    def heat_input(self, T_E, T_C):
        """Total heat that must be supplied to the emitter [W]."""
        return self.electron_cooling_power(T_E, T_C) + self.radiation_power(T_E, T_C)

    # ------------------------------------------------------------------
    # Steady-state electrical operating point
    # ------------------------------------------------------------------
    def output_current_density(self, V_term, T_E, T_C):
        """
        Output current density [A/m^2] at terminal (load) voltage V_term.

        Thermionic diode output characteristic (Hatsopoulos & Gyftopoulos 1973):
          - For V_term <= V_oc - V_plasma (saturation / accelerating region) the
            full space-charge-reduced saturation current flows.
          - For V_term above that, the collected current is exponentially
            retarded (Boltzmann barrier), J = J_sat * exp(-q dV / (k_B T_E)),
            collapsing to zero back-EMF-limited as V_term -> V_oc.
        Lead-resistance ohmic drop is folded in via the saturation current.
        """
        J_sat = float(self.net_current_density(T_E, T_C))
        V_oc = float(self.open_circuit_voltage(T_E, T_C))
        V_knee = V_oc - self.V_plasma            # max voltage at full current
        if V_term <= V_knee:
            J = J_sat
        else:
            dV = V_term - V_knee
            J = J_sat * np.exp(-q_e * dV / (k_B * float(T_E)))
        # Subtract self-consistent lead ohmic loss (small) and clamp.
        return max(J, 0.0)

    def operating_point(self, T_E, T_C, V_load=None):
        """
        Electrical operating point at given electrode temperatures.

        V_load : optional clamped terminal voltage [V]. If None, the converter
                 is operated at its maximum-power point along the diode I-V
                 characteristic (the physically relevant operating condition).

        Returns dict with current density, voltage, power, heat, efficiencies.
        """
        T_E = float(T_E)
        T_C = float(T_C)
        phi_e = float(self.phi_emitter(T_E))
        phi_c = float(self.phi_collector(T_C))

        J_sat = float(self.net_current_density(T_E, T_C))
        V_oc = float(self.open_circuit_voltage(T_E, T_C))
        V_avail = max(V_oc - self.V_plasma, 0.0)   # usable terminal-voltage span

        def power_at(V):
            J = self.output_current_density(V, T_E, T_C)
            J_eff = max(J - 0.0, 0.0)
            # lead-resistance ohmic loss on delivered current
            V_eff = V - J_eff * self.area * self.R_lead
            return max(J_eff * max(V_eff, 0.0) * self.area, 0.0)

        if V_load is None:
            # Maximum-power point: scan the usable voltage span.
            if V_avail <= 0.0:
                V_term, P_elec = 0.0, 0.0
            else:
                Vs = np.linspace(0.0, V_avail, 64)
                Ps = np.array([power_at(V) for V in Vs])
                imax = int(np.argmax(Ps))
                V_term = float(Vs[imax])
                P_elec = float(Ps[imax])
        else:
            V_term = max(min(float(V_load), V_avail), 0.0)
            P_elec = power_at(V_term)

        J_net = self.output_current_density(V_term, T_E, T_C)
        Q_in = float(self.heat_input(T_E, T_C))          # W
        Q_in = max(Q_in, 1e-12)

        eta_carnot = self.carnot_efficiency(T_E, T_C)
        eta = P_elec / Q_in
        # Physically eta cannot exceed Carnot; clamp tiny numerical overshoot.
        eta = min(max(eta, 0.0), max(float(eta_carnot), 0.0))

        return {
            "phi_e_eV": phi_e,
            "phi_c_eV": phi_c,
            "J_sat_Am2": J_sat,
            "J_net_Am2": J_net,
            "J_net_Acm2": J_net * 1e-4,
            "V_oc_V": V_oc,
            "V_terminal_V": V_term,
            "power_w": P_elec,
            "power_density_w_cm2": P_elec / self.area * 1e-4,
            "heat_input_w": Q_in,
            "Q_electron_w": float(self.electron_cooling_power(T_E, T_C)),
            "Q_radiation_w": float(self.radiation_power(T_E, T_C)),
            "efficiency": eta,
            "carnot_efficiency": float(eta_carnot),
        }

    # ------------------------------------------------------------------
    # Lumped emitter-temperature ODE
    # ------------------------------------------------------------------
    def dTdt(self, T_E, T_C, Q_external):
        """Emitter temperature derivative [K/s]."""
        return (Q_external - self.heat_input(T_E, T_C)) / self.C_E

    def simulate(self, Q_external_w, T_emitter0_K, T_collector_K,
                 dt, duration_s, V_load=None):
        """
        Integrate the lumped emitter-temperature ODE under a fixed external
        heat input, recording the electrical operating point at each step.

        Parameters
        ----------
        Q_external_w  : float or callable(t) -- heat power into emitter [W]
        T_emitter0_K  : float -- initial emitter temperature [K]
        T_collector_K : float -- fixed collector (sink) temperature [K]
        dt            : float -- output time step [s]
        duration_s    : float -- total duration [s]
        V_load        : optional clamped terminal voltage [V]

        Returns
        -------
        dict of time-series arrays: t, T_emitter, J_net_Am2, V_terminal_V,
             power_w, power_density_w_cm2, heat_input_w, efficiency,
             carnot_efficiency.
        """
        _Q = Q_external_w if callable(Q_external_w) else (lambda t: Q_external_w)
        T_C = float(T_collector_K)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        def rhs(t, y):
            return [self.dTdt(y[0], T_C, _Q(t))]

        sol = solve_ivp(
            rhs, (0.0, duration_s), [float(T_emitter0_K)],
            t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-7,
            max_step=dt,
        )

        t_out = sol.t
        T_out = sol.y[0]
        N = len(t_out)

        J_net = np.zeros(N)
        V_term = np.zeros(N)
        power = np.zeros(N)
        pden = np.zeros(N)
        Q_in = np.zeros(N)
        eta = np.zeros(N)
        eta_c = np.zeros(N)

        for i in range(N):
            op = self.operating_point(T_out[i], T_C, V_load=V_load)
            J_net[i] = op["J_net_Am2"]
            V_term[i] = op["V_terminal_V"]
            power[i] = op["power_w"]
            pden[i] = op["power_density_w_cm2"]
            Q_in[i] = op["heat_input_w"]
            eta[i] = op["efficiency"]
            eta_c[i] = op["carnot_efficiency"]

        return {
            "t": t_out,
            "T_emitter": T_out,
            "J_net_Am2": J_net,
            "V_terminal_V": V_term,
            "power_w": power,
            "power_density_w_cm2": pden,
            "heat_input_w": Q_in,
            "efficiency": eta,
            "carnot_efficiency": eta_c,
        }
