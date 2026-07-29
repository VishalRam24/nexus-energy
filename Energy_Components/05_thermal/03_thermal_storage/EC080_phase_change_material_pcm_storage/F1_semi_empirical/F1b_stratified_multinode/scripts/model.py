"""
EC080 -- Phase-Change Material (PCM) Storage -- F1b Enthalpy Method Model

Builds on F1a (simple three-region) by using the enthalpy method:
    h(T) = cp_s * T                                       for T < T_pc - dT
    h(T) = cp_s * (T_pc - dT) + f * L                     for T_pc - dT <= T <= T_pc + dT (mushy)
    h(T) = cp_s * (T_pc - dT) + L + cp_l * (T - T_pc - dT)  for T > T_pc + dT

Phase fraction f linearly interpolated in mushy zone:
    f = (T - (T_pc - dT)) / (2 * dT)     for T in [T_pc - dT, T_pc + dT]

Heat exchange with HTF via UA_htf:
    Q_htf = UA_htf * (T_htf - T_pcm)    (in charge mode, T_htf > T_pcm)
    Q_htf = UA_htf * (T_pcm - T_htf)    (in discharge mode, T_pcm > T_htf)

PCM material: Paraffin RT58 (T_pc = 58 degC, L = 180 kJ/kg)

References:
    Mehling, H. & Cabeza, L.F. (2008). Heat and Cold Storage with PCM. Springer.
    Voller, V.R. (1990). Fast implicit finite-difference method for the analysis of
        phase change problems. Numerical Heat Transfer B, 17(2), 155-169.
    Rubitherm Technologies GmbH -- RT58 datasheet.
"""

import numpy as np


class PCMStorageF1b:
    """PCM storage using enthalpy method with proper mushy zone interpolation."""

    def __init__(self, params: dict):
        p = params["pcm"]
        self.mass = p["mass_pcm_kg"]["value"]              # kg
        self.T_pc = p["T_pc_degC"]["value"]                 # degC
        self.dT_pc = p["delta_T_pc_degC"]["value"]          # degC half-width
        self.L = p["L_kJ_kg"]["value"] * 1000.0             # J/kg
        self.cp_s = p["cp_solid_kJ_kgK"]["value"] * 1000.0  # J/(kg*K)
        self.cp_l = p["cp_liquid_kJ_kgK"]["value"] * 1000.0 # J/(kg*K)
        self.UA_htf = p["UA_htf_W_K"]["value"]              # W/K
        self.UA_loss = p["UA_loss_W_K"]["value"]             # W/K
        self.T_amb_ref = p["T_ambient_ref_C"]["value"]       # degC

        # Mushy zone boundaries
        self.T_solidus = self.T_pc - self.dT_pc    # 56 degC
        self.T_liquidus = self.T_pc + self.dT_pc   # 60 degC

        # HTF properties (water/glycol)
        self.cp_htf = 4180.0  # J/(kg*K)

    def phase_fraction(self, T_pcm):
        """Liquid phase fraction f [0, 1]."""
        T = np.asarray(T_pcm, dtype=float)
        f = (T - self.T_solidus) / (self.T_liquidus - self.T_solidus)
        return np.clip(f, 0.0, 1.0)

    def enthalpy(self, T_pcm):
        """Specific enthalpy [J/kg] as function of PCM temperature."""
        T = np.asarray(T_pcm, dtype=float)
        h = np.zeros_like(T)

        # Solid region
        solid = T < self.T_solidus
        h[solid] = self.cp_s * T[solid]

        # Mushy region
        mushy = (T >= self.T_solidus) & (T <= self.T_liquidus)
        f_mushy = (T[mushy] - self.T_solidus) / (self.T_liquidus - self.T_solidus)
        h[mushy] = self.cp_s * self.T_solidus + f_mushy * self.L

        # Liquid region
        liquid = T > self.T_liquidus
        h[liquid] = self.cp_s * self.T_solidus + self.L + self.cp_l * (T[liquid] - self.T_liquidus)

        return h

    def temperature_from_enthalpy(self, h):
        """Invert enthalpy to get temperature [degC]."""
        h = np.asarray(h, dtype=float)
        T = np.zeros_like(h)

        h_solidus = self.cp_s * self.T_solidus
        h_liquidus = h_solidus + self.L

        # Solid region
        solid = h < h_solidus
        T[solid] = h[solid] / self.cp_s

        # Mushy region
        mushy = (h >= h_solidus) & (h <= h_liquidus)
        f = (h[mushy] - h_solidus) / self.L
        T[mushy] = self.T_solidus + f * (self.T_liquidus - self.T_solidus)

        # Liquid region
        liquid = h > h_liquidus
        T[liquid] = self.T_liquidus + (h[liquid] - h_liquidus) / self.cp_l

        return T

    def predict(self, T_htf_in_degC, flow_rate_kg_s, mode,
                duration_s=3600.0, T_ambient_degC=None,
                T_pcm_init=None):
        """
        Run the enthalpy-method PCM model for a single step.

        Args:
            T_htf_in_degC:   HTF inlet temperature [degC]
            flow_rate_kg_s:  HTF flow rate [kg/s]
            mode:            'charge', 'discharge', or 'idle'
            duration_s:      Step duration [s]
            T_ambient_degC:  Ambient temperature [degC]
            T_pcm_init:      Initial PCM temperature [degC], default T_pc

        Returns:
            dict with: T_pcm_degC, phase_fraction, energy_stored_kwh,
                       thermal_power_kw, T_outlet_degC
        """
        if T_ambient_degC is None:
            T_ambient_degC = self.T_amb_ref

        if T_pcm_init is None:
            T_pcm_init = self.T_pc

        T_pcm = float(T_pcm_init)

        # Time integration
        dt = 5.0  # 5s substeps
        n_steps = max(1, int(duration_s / dt))
        dt = duration_s / n_steps

        Q_htf_total = 0.0
        T_outlet_sum = 0.0

        for _ in range(n_steps):
            h_current = float(self.enthalpy(np.array([T_pcm]))[0])

            # HTF heat exchange
            if mode == "charge" and flow_rate_kg_s > 0:
                Q_htf = self.UA_htf * (T_htf_in_degC - T_pcm)
            elif mode == "discharge" and flow_rate_kg_s > 0:
                Q_htf = self.UA_htf * (T_htf_in_degC - T_pcm)
            else:
                Q_htf = 0.0

            # HTF outlet temperature (simple effectiveness model)
            if flow_rate_kg_s > 0:
                C_htf = flow_rate_kg_s * self.cp_htf
                epsilon = 1.0 - np.exp(-self.UA_htf / C_htf)
                if mode == "charge":
                    Q_htf = epsilon * C_htf * (T_htf_in_degC - T_pcm)
                    T_outlet = T_htf_in_degC - Q_htf / C_htf
                elif mode == "discharge":
                    Q_htf = epsilon * C_htf * (T_pcm - T_htf_in_degC)
                    Q_htf = -Q_htf  # negative means heat leaving PCM
                    T_outlet = T_htf_in_degC - Q_htf / C_htf
                else:
                    T_outlet = T_htf_in_degC
            else:
                T_outlet = T_htf_in_degC

            # Ambient heat loss
            Q_loss = self.UA_loss * (T_pcm - T_ambient_degC)

            # Update enthalpy
            Q_net = Q_htf - Q_loss  # W (positive = heating PCM)
            h_new = h_current + Q_net * dt / self.mass
            T_pcm = float(self.temperature_from_enthalpy(np.array([h_new]))[0])

            Q_htf_total += Q_htf * dt
            T_outlet_sum += T_outlet

        # Average thermal power [kW]
        thermal_power_kw = Q_htf_total / duration_s / 1000.0

        # Average outlet temperature
        T_outlet_avg = T_outlet_sum / n_steps

        # Phase fraction
        f = float(self.phase_fraction(np.array([T_pcm]))[0])

        # Energy stored [kWh] relative to 20 degC reference
        T_ref = 20.0
        h_ref = float(self.enthalpy(np.array([T_ref]))[0])
        h_final = float(self.enthalpy(np.array([T_pcm]))[0])
        energy_stored_kwh = self.mass * (h_final - h_ref) / 3.6e6

        return {
            "T_pcm_degC": T_pcm,
            "phase_fraction": f,
            "energy_stored_kwh": max(0.0, energy_stored_kwh),
            "thermal_power_kw": thermal_power_kw,
            "T_outlet_degC": T_outlet_avg,
        }

    def simulate(self, T_pcm_init, T_htf_schedule, flow_schedule,
                 mode_schedule, T_ambient_schedule=None, dt_step_s=3600.0):
        """
        Multi-step simulation.

        Returns:
            dict with arrays: T_pcm, phase_fraction, energy_stored_kwh,
                              thermal_power_kw, T_outlet
        """
        N = len(mode_schedule)
        if T_ambient_schedule is None:
            T_amb_arr = np.full(N, self.T_amb_ref)
        else:
            T_amb_arr = np.asarray(T_ambient_schedule, dtype=float)
            if T_amb_arr.ndim == 0:
                T_amb_arr = np.full(N, float(T_amb_arr))

        T_pcm = float(T_pcm_init)
        history = {k: [] for k in ["T_pcm", "phase_fraction", "energy_stored_kwh",
                                    "thermal_power_kw", "T_outlet"]}

        for i in range(N):
            r = self.predict(
                T_htf_in_degC=float(T_htf_schedule[i]),
                flow_rate_kg_s=float(flow_schedule[i]),
                mode=mode_schedule[i],
                duration_s=dt_step_s,
                T_ambient_degC=float(T_amb_arr[i]),
                T_pcm_init=T_pcm,
            )
            T_pcm = r["T_pcm_degC"]
            history["T_pcm"].append(r["T_pcm_degC"])
            history["phase_fraction"].append(r["phase_fraction"])
            history["energy_stored_kwh"].append(r["energy_stored_kwh"])
            history["thermal_power_kw"].append(r["thermal_power_kw"])
            history["T_outlet"].append(r["T_outlet_degC"])

        return {k: np.array(v) for k, v in history.items()}
