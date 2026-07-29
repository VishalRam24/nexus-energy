"""
EC182 — Distribution Line — F1b Thermal Ampacity + R(T) + Skin Effect Model

Extends F1a R+jX series model with:
  1. Temperature-dependent resistance:
         R(T) = R_ref * [1 + alpha * (T_cond - T_ref)]   (IEC 60287)
     Distribution conductors run warm (typical 50-70 degC) → up to 20% R increase.
  2. Skin-effect correction:
     At distribution frequencies (50/60 Hz) with typical cable cross-sections (25-400 mm^2),
     the skin factor is close to 1. For overhead: skin_factor ~ 1.01-1.02.
     For underground XLPE cables with large conductors: up to 1.05 (IEC 60287).
  3. Thermal ampacity (IEC 60287-1-1 / IEEE 738 for overhead; Neher-McGrath for cable):
     Overhead: same IEEE 738 approach as transmission.
     Cable (underground): IEC 60287 simplified:
         I_max = sqrt(dT_max / (R_ac * (T_ins + T_soil_therm)))
     where dT_max = T_cond_max - T_amb.
  4. Congestion factor: actual loading as fraction of thermal limit.

References:
    Kersting (2012). Distribution System Modeling and Analysis, 3rd ed.
    IEC 60287-1-1 (2006). Electric cables — current rating calculation.
    IEEE Std 738-2012. Overhead conductor ampacity.
    Neher, J.H. & McGrath, M.H. (1957). The calculation of temperature rise and
        load capability of cable systems. AIEE Trans., 76(3):752-772.
"""

import numpy as np


class DistributionLineF1b:
    """
    Distribution feeder F1b: R+jX model with temperature-dependent R,
    skin-effect, and thermal ampacity limit.
    """

    STEFAN_BOLTZMANN = 5.6704e-8   # W/(m^2 K^4)

    def __init__(self, params: dict):
        u = params["unit"]
        th = params["thermal"]

        self.R_dc_ohm_per_km  = u["R_dc_ohm_per_km"]["value"]
        self.X_ohm_per_km     = u["X_ohm_per_km"]["value"]
        self.V_base_kV        = u["V_base_kV"]["value"]
        self.length_km_def    = u["length_km"]["value"]
        self.alpha_R          = u["alpha_R_per_K"]["value"]
        self.T_ref            = u["T_ref_C"]["value"]
        self.skin_factor      = u["skin_factor"]["value"]

        # Thermal
        self.cable_type       = th["cable_type"]["value"]   # "overhead" or "underground"
        self.D_cond_m         = th["D_conductor_m"]["value"]
        self.T_cond_max       = th["T_conductor_max_C"]["value"]
        self.T_cond_rated     = th["T_conductor_rated_C"]["value"]
        self.emissivity       = th["emissivity"]["value"]
        self.absorptivity     = th["absorptivity"]["value"]
        self.solar_irr        = th["solar_irradiance_W_m2"]["value"]
        self.wind_speed       = th["wind_speed_m_s"]["value"]
        # IEC 60287 underground thermal resistance [K·m/W]
        self.T_ins_thermal    = th.get("T_ins_thermal_Km_W", {}).get("value", 3.5)
        self.T_soil_thermal   = th.get("T_soil_thermal_Km_W", {}).get("value", 1.0)

    # ------------------------------------------------------------------
    # Temperature-dependent AC resistance
    # ------------------------------------------------------------------
    def r_ac_ohm_per_km(self, T_cond_C=None):
        """R_ac(T) [Ohm/km]: DC resistance corrected for temperature and skin effect."""
        T = self.T_cond_rated if T_cond_C is None else np.asarray(T_cond_C, dtype=float)
        return self.R_dc_ohm_per_km * (1.0 + self.alpha_R * (T - self.T_ref)) * self.skin_factor

    # ------------------------------------------------------------------
    # Thermal ampacity
    # ------------------------------------------------------------------
    def _q_convective_overhead(self, T_cond_C, T_amb_C):
        """IEEE 738 convective cooling [W/m]."""
        T_cond = np.asarray(T_cond_C, dtype=float)
        T_amb  = np.asarray(T_amb_C, dtype=float)
        dT     = T_cond - T_amb
        T_film = (T_cond + T_amb) / 2.0 + 273.15
        rho_f  = 1.2929 * 273.15 / T_film
        D = self.D_cond_m
        v = max(self.wind_speed, 0.0)
        if v < 0.05:
            return 2.11e-2 * D ** 0.75 * np.abs(dT) ** 1.25
        return 3.645 * rho_f ** 0.5 * D ** 0.75 * v ** 0.6 * dT

    def _q_radiative(self, T_cond_C, T_amb_C):
        """Radiative cooling [W/m]."""
        T_c = np.asarray(T_cond_C, dtype=float) + 273.15
        T_a = np.asarray(T_amb_C, dtype=float) + 273.15
        return (self.STEFAN_BOLTZMANN * self.emissivity * np.pi * self.D_cond_m *
                (T_c ** 4 - T_a ** 4))

    def thermal_ampacity_A(self, T_amb_C=25.0):
        """
        Maximum continuous current [A].

        Overhead (IEEE 738):
            I_max = sqrt[(q_c + q_r - q_s) / R_ac_Tmax_per_m]
        Underground (IEC 60287):
            I_max = sqrt[dT_max / (R_ac_Tmax_per_m * (T_ins + T_soil))]
        """
        T_amb = np.asarray(T_amb_C, dtype=float)
        R_ac_km = self.r_ac_ohm_per_km(self.T_cond_max)   # Ohm/km
        R_ac_m  = R_ac_km / 1000.0                         # Ohm/m

        if self.cable_type == "overhead":
            q_c  = self._q_convective_overhead(self.T_cond_max, T_amb)
            q_r  = self._q_radiative(self.T_cond_max, T_amb)
            q_s  = self.absorptivity * self.D_cond_m * self.solar_irr
            q_net = np.maximum(q_c + q_r - q_s, 1.0)
            I_max = np.sqrt(q_net / (R_ac_m + 1e-12))
        else:
            # IEC 60287: I_max = sqrt(dT / (R_ac * T_thermal))
            dT = self.T_cond_max - T_amb
            T_total = self.T_ins_thermal + self.T_soil_thermal
            I_max = np.sqrt(np.maximum(dT, 0.0) / (R_ac_m * T_total + 1e-12))

        return I_max

    def ampacity_derating_factor(self, T_amb_C):
        """Derating relative to 25 degC."""
        I_ref = self.thermal_ampacity_A(T_amb_C=25.0)
        I_amb = self.thermal_ampacity_A(T_amb_C=T_amb_C)
        return I_amb / (I_ref + 1e-12)

    # ------------------------------------------------------------------
    # Core distribution line computation
    # ------------------------------------------------------------------
    def compute(self, V_s_kV: float, P_load_kW: float,
                Q_load_kVAR: float, length_km: float = None,
                T_cond_C: float = None, T_amb_C: float = 25.0) -> dict:
        """
        Distribution feeder power flow with thermal correction.

        Parameters
        ----------
        V_s_kV      : Sending-end line-to-line voltage [kV]
        P_load_kW   : Active load at receiving end [kW]
        Q_load_kVAR : Reactive load [kVAR]
        length_km   : Feeder length [km]
        T_cond_C    : Conductor temperature [degC]
        T_amb_C     : Ambient temperature [degC]

        Returns
        -------
        dict with all F1a outputs + R_ac_ohm_km, skin_factor, I_max_A,
        ampacity_margin, derating_factor, congestion_factor
        """
        if length_km is None:
            length_km = self.length_km_def

        V_s_kV      = np.asarray(V_s_kV, dtype=float)
        P_kW        = np.asarray(P_load_kW, dtype=float)
        Q_kVAR      = np.asarray(Q_load_kVAR, dtype=float)
        L = float(length_km)

        R_total = self.r_ac_ohm_per_km(T_cond_C) * L
        X_total = self.X_ohm_per_km * L
        Z = R_total + 1j * X_total

        V_s_phase = V_s_kV * 1000.0 / np.sqrt(3.0)   # V

        S_VA = (P_kW + 1j * Q_kVAR) * 1000.0         # VA

        V_r_phase = V_s_phase.copy() if np.ndim(V_s_phase) > 0 else complex(V_s_phase)
        for _ in range(5):
            safe = np.abs(V_r_phase) > 1.0
            V_r_safe = (np.where(safe, V_r_phase, 1.0 + 0j)
                        if np.ndim(V_r_phase) > 0
                        else (V_r_phase if abs(V_r_phase) > 1.0 else 1.0 + 0j))
            I_line = np.conj(S_VA / (3.0 * V_r_safe))
            V_r_phase = V_s_phase - Z * I_line

        I_mag = np.abs(I_line)
        V_r_phase_mag = np.abs(V_r_phase)
        V_r_kV = V_r_phase_mag * np.sqrt(3.0) / 1000.0

        P_loss_kW   = 3.0 * I_mag ** 2 * float(R_total) / 1000.0
        Q_loss_kVAR = 3.0 * I_mag ** 2 * float(X_total) / 1000.0

        voltage_drop_kV  = V_s_kV - V_r_kV
        safe_V = (np.where(V_s_kV > 0, V_s_kV, 1e-12)
                  if np.ndim(V_s_kV) > 0 else (V_s_kV if V_s_kV > 0 else 1e-12))
        voltage_drop_pct = voltage_drop_kV / safe_V * 100.0

        P_s_kW = P_kW + P_loss_kW
        safe_Ps = (np.where(P_s_kW > 0, P_s_kW, 1e-12)
                   if np.ndim(P_s_kW) > 0 else (P_s_kW if P_s_kW > 0 else 1e-12))
        eta = (np.where(P_s_kW > 0, P_kW / safe_Ps, 0.0)
               if np.ndim(P_s_kW) > 0 else (P_kW / safe_Ps if P_s_kW > 0 else 0.0))

        S_load = np.sqrt(P_kW ** 2 + Q_kVAR ** 2)
        safe_S = (np.where(S_load > 0, S_load, 1e-12)
                  if np.ndim(S_load) > 0 else (S_load if S_load > 0 else 1e-12))
        pf_load = (np.where(S_load > 0, P_kW / safe_S, 1.0)
                   if np.ndim(S_load) > 0 else (P_kW / safe_S if S_load > 0 else 1.0))

        # Thermal ampacity
        I_max_A = self.thermal_ampacity_A(T_amb_C=T_amb_C)
        ampacity_margin = (I_max_A - I_mag) / (I_max_A + 1e-12)
        congestion = I_mag / (I_max_A + 1e-12)
        derating = self.ampacity_derating_factor(T_amb_C)

        return {
            "V_r_kV": V_r_kV,
            "I_line_A": I_mag,
            "P_loss_kW": P_loss_kW,
            "Q_loss_kVAR": Q_loss_kVAR,
            "P_s_kW": P_s_kW,
            "efficiency": eta,
            "voltage_drop_kV": voltage_drop_kV,
            "voltage_drop_pct": voltage_drop_pct,
            "power_factor_load": pf_load,
            "R_ac_ohm_km": float(R_total / L),
            "skin_factor": self.skin_factor,
            "I_max_A": I_max_A,
            "ampacity_margin": ampacity_margin,
            "congestion_factor": congestion,
            "derating_factor": derating,
        }
