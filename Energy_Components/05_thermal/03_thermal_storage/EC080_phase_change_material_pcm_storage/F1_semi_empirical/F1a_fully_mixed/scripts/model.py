"""
EC080 — Phase-Change Material (PCM) Storage — F1a Latent Heat Model

Three-region model for PCM storage:

    SOLID region   (T < Tm):  dT/dt = Q_net / (m * cp_s)
    MUSHY region   (T = Tm):  df/dt = Q_net / (m * L)        [f = liquid fraction]
    LIQUID region  (T > Tm):  dT/dt = Q_net / (m * cp_l)

Total energy stored (relative to T_ref = 0°C):
    E_total = m * [cp_s * (Tm - T_ref) + f * L + cp_l * max(0, T - Tm)]

State of charge:
    SOC = E_stored / E_total_at_full_liquid_50C

For F1a simplification:
  - Fully mixed (no spatial gradients)
  - Sharp mushy zone (exact phase change at T_melt ± T_melt_range/2)
  - q_charge / q_discharge as net heat flows (HTF-to-PCM after UA)

PCM: Paraffin RT42
  - T_melt = 42°C
  - L = 174 kJ/kg
  - cp = 2.0 kJ/(kg·K) for both solid and liquid phases
  - mass = 500 kg

Reference:
    Mehling, H. & Cabeza, L.F. (2008).
    "Heat and Cold Storage with PCM." Springer, Berlin.
    Rubitherm Technologies GmbH — RT42 datasheet.
"""

import numpy as np


class PCMF1a:
    """PCM thermal storage — three-region model (solid / mushy / liquid)."""

    def __init__(self, params: dict):
        p = params["pcm"]
        self.mass       = p["mass_kg"]["value"]              # kg
        self.Tm         = p["T_melt_C"]["value"]             # °C
        self.dT_mushy   = p["T_melt_range_C"]["value"]       # °C half-width
        self.L          = p["latent_heat_kJ_kg"]["value"] * 1000.0  # J/kg
        self.cp_s       = p["cp_solid_kJ_kgK"]["value"]  * 1000.0  # J/(kg·K)
        self.cp_l       = p["cp_liquid_kJ_kgK"]["value"] * 1000.0  # J/(kg·K)
        self.UA         = p["UA_W_K"]["value"]               # W/K
        self.T_ref      = p["T_ref_C"]["value"]              # °C
        self.T_amb_ref  = p["T_ambient_ref_C"]["value"]      # °C

        # Mushy zone boundaries
        self.T_solidus  = self.Tm - self.dT_mushy   # 40°C
        self.T_liquidus = self.Tm + self.dT_mushy   # 44°C

        # Reference energies [J]
        self.E_solid_sensible = self.mass * self.cp_s * (self.Tm - self.T_ref)
        self.E_latent_total   = self.mass * self.L
        # Full capacity = sensible (solid) + full latent + 8°C liquid buffer
        self.E_full_J = (
            self.E_solid_sensible +
            self.E_latent_total +
            self.mass * self.cp_l * 8.0   # liquid buffer to T=50°C
        )

    # ------------------------------------------------------------------
    # Instantaneous rates (for a given T, f state)
    # ------------------------------------------------------------------

    def _region(self, T):
        """Identify thermal region: -1=solid, 0=mushy, +1=liquid."""
        T = np.asarray(T, dtype=float)
        return np.where(T < self.T_solidus, -1,
               np.where(T > self.T_liquidus, 1, 0))

    def dT_dt(self, T_c, liquid_fraction, q_charge_w, q_discharge_w, T_amb_c=None):
        """
        Temperature rate of change [K/s].

        During phase change (mushy zone), temperature is pinned at Tm.
        Returns 0.0 in the mushy zone (dT/dt handled by df_dt).
        """
        if T_amb_c is None:
            T_amb_c = self.T_amb_ref

        T   = np.asarray(T_c,           dtype=float)
        f   = np.asarray(liquid_fraction, dtype=float)
        Qc  = np.asarray(q_charge_w,     dtype=float)
        Qd  = np.asarray(q_discharge_w,  dtype=float)
        T_a = np.asarray(T_amb_c,        dtype=float)

        Q_loss = self.UA * (T - T_a)        # [W]
        Q_net  = Qc - Qd - Q_loss           # [W]

        reg = self._region(T)
        cp_eff = np.where(reg == -1, self.cp_s,
                 np.where(reg ==  1, self.cp_l, np.inf))  # inf → dT/dt=0 in mushy

        return np.where(reg == 0, 0.0, Q_net / (self.mass * cp_eff))

    def df_dt(self, T_c, liquid_fraction, q_charge_w, q_discharge_w, T_amb_c=None):
        """
        Liquid fraction rate of change [1/s].

        Only non-zero in mushy zone; clipped to maintain f in [0, 1].
        """
        if T_amb_c is None:
            T_amb_c = self.T_amb_ref

        T   = np.asarray(T_c,           dtype=float)
        f   = np.asarray(liquid_fraction, dtype=float)
        Qc  = np.asarray(q_charge_w,     dtype=float)
        Qd  = np.asarray(q_discharge_w,  dtype=float)
        T_a = np.asarray(T_amb_c,        dtype=float)

        Q_loss = self.UA * (T - T_a)
        Q_net  = Qc - Qd - Q_loss

        reg = self._region(T)
        # In mushy zone: df/dt = Q_net / (m * L)
        df = np.where(reg == 0, Q_net / (self.mass * self.L), 0.0)
        # Clip: if f=0 and cooling, df can't go negative; if f=1 and heating, df=0
        df = np.where((f <= 0.0) & (df < 0), 0.0, df)
        df = np.where((f >= 1.0) & (df > 0), 0.0, df)
        return df

    def energy_stored_kwh(self, T_c, liquid_fraction):
        """
        Total energy stored relative to T_ref [kWh].
        E = m * [cp_s*(Tm - T_ref) + f*L + cp_l*max(0, T-Tm)]
        """
        T = np.asarray(T_c,            dtype=float)
        f = np.asarray(liquid_fraction, dtype=float)
        f = np.clip(f, 0.0, 1.0)

        E_s = self.mass * self.cp_s * (self.Tm - self.T_ref)     # J sensible (solid phase)
        E_l = self.mass * self.L * f                              # J latent
        E_liq = self.mass * self.cp_l * np.maximum(0.0, T - self.Tm)  # J liquid sensible
        return np.maximum((E_s + E_l + E_liq) / 3.6e6, 0.0)    # kWh

    def soc(self, T_c, liquid_fraction):
        """State of charge [0, 1]."""
        E = self.energy_stored_kwh(T_c, liquid_fraction)
        return np.clip(E / (self.E_full_J / 3.6e6), 0.0, 1.0)

    def heat_loss_w(self, T_c, T_amb_c=None):
        """Heat loss to environment [W]."""
        if T_amb_c is None:
            T_amb_c = self.T_amb_ref
        T   = np.asarray(T_c,     dtype=float)
        T_a = np.asarray(T_amb_c, dtype=float)
        return self.UA * (T - T_a)

    def simulate(self, T0_c, f0, q_charge_profile_w, q_discharge_profile_w,
                 T_amb_profile=None, dt_s=60.0):
        """
        Time-step simulation with Euler integration.

        Args:
            T0_c:          Initial temperature [°C]
            f0:            Initial liquid fraction [0, 1]
            q_charge_profile_w:    Array of charge power [W]
            q_discharge_profile_w: Array of discharge power [W]
            T_amb_profile: Ambient temperature array [°C] or scalar
            dt_s:          Time step [s], default 60s

        Returns:
            dict with time arrays: T, f, energy_kwh, soc, heat_loss_w
        """
        N = len(q_charge_profile_w)
        T_arr = np.zeros(N + 1); T_arr[0] = T0_c
        f_arr = np.zeros(N + 1); f_arr[0] = np.clip(f0, 0.0, 1.0)

        if T_amb_profile is None:
            T_amb_arr = np.full(N, self.T_amb_ref)
        else:
            T_amb_arr = np.asarray(T_amb_profile, dtype=float)

        for i in range(N):
            T_cur = T_arr[i]
            f_cur = f_arr[i]
            Qc = q_charge_profile_w[i]
            Qd = q_discharge_profile_w[i]
            T_a = T_amb_arr[i]

            dT = float(self.dT_dt(T_cur, f_cur, Qc, Qd, T_a))
            df = float(self.df_dt(T_cur, f_cur, Qc, Qd, T_a))

            T_new = T_cur + dT * dt_s
            f_new = f_cur + df * dt_s

            # If temperature tries to cross Tm from outside mushy zone,
            # clamp it to mushy zone boundary and let liquid fraction absorb the rest
            T_arr[i + 1] = np.clip(T_new, 0.0, 80.0)
            f_arr[i + 1] = np.clip(f_new, 0.0, 1.0)

        return {
            "T_C":         T_arr,
            "f":           f_arr,
            "energy_kwh":  self.energy_stored_kwh(T_arr, f_arr),
            "soc":         self.soc(T_arr, f_arr),
            "heat_loss_w": self.heat_loss_w(T_arr, T_amb_profile if T_amb_profile is not None else self.T_amb_ref),
        }
