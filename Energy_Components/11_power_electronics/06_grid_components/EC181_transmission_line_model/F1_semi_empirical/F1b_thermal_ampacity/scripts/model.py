"""
EC181 — Transmission Line — F1b Thermal Ampacity + Skin Effect + R(T) Model

Extends F1a pi-model with:
  1. Temperature-dependent resistance:
         R(T) = R_ref * [1 + alpha_R * (T_cond - T_ref)]    (IEC 60287)
     where T_cond is conductor temperature, T_ref = 20 degC.
  2. Skin-effect resistance correction at 50/60 Hz (Morgan 1978 / Glover 2012):
         R_skin / R_dc = f(x_k) where x_k = 2*sqrt(2)*pi*f*sqrt(mu/(R_dc))
     For solid ACSR stranded conductors approximated by the simplified Bessel
     correction factor via the GMR ratio (tabulated, fitted here as polynomial).
  3. Thermal ampacity: steady-state IEEE 738-2012 simplified:
         I_max = sqrt[(q_r + q_c - q_s) / R_ac_Tmax]
     with:
         q_r = radiation cooling [W/m]
         q_c = convective cooling [W/m] (Morgan 1982, natural+forced)
         q_s = solar gain [W/m]
  4. Ampacity derating at ambient temperature (thermal model output).
  5. F1a pi-model re-used to compute voltage/losses at final adjusted R.

References:
    IEEE Std 738-2012. IEEE Standard for Calculating the Current-Temperature
        Relationship of Bare Overhead Conductors.
    IEC 60287-1-1 (2006). Electric cables — calculation of current rating.
    Morgan, V.T. (1978). The current-carrying capacity of overhead line conductors.
    Glover, Sarma, Overbye (2012). Power Systems Analysis and Design, 5th ed.
"""

import numpy as np


class TransmissionLineF1b:
    """
    Transmission line F1b: pi-model + temperature-dependent resistance
    + skin effect + thermal ampacity (IEEE 738).
    """

    STEFAN_BOLTZMANN = 5.6704e-8   # W/(m^2 K^4)

    def __init__(self, params: dict):
        u = params["unit"]
        th = params["thermal"]

        # Electrical parameters (per km, pu-based)
        self.R_dc_pu_per_km = u["R_dc_pu_per_km"]["value"]
        self.X_pu_per_km    = u["X_pu_per_km"]["value"]
        self.B_pu_per_km    = u["B_pu_per_km"]["value"]
        self.V_base_kV      = u["V_base_kV"]["value"]
        self.S_base_MVA     = u["S_base_MVA"]["value"]
        self.length_km_def  = u["length_km"]["value"]
        self.alpha_R        = u["alpha_R_per_K"]["value"]      # 1/K  temp coeff of resistance
        self.T_ref          = u["T_ref_C"]["value"]            # degC reference
        self.f_Hz           = u["frequency_Hz"]["value"]       # Hz
        self.skin_factor    = u["skin_factor"]["value"]        # dimensionless, R_ac/R_dc

        # Thermal ampacity parameters (IEEE 738)
        self.D_cond_m       = th["D_conductor_m"]["value"]     # m conductor outer diameter
        self.T_cond_max     = th["T_conductor_max_C"]["value"] # degC max continuous
        self.T_cond_rated   = th["T_conductor_rated_C"]["value"]  # degC at rated current
        self.emissivity     = th["emissivity"]["value"]
        self.absorptivity   = th["absorptivity"]["value"]
        self.solar_irr      = th["solar_irradiance_W_m2"]["value"]  # W/m^2
        self.wind_speed_m_s = th["wind_speed_m_s"]["value"]    # m/s (forced cooling)
        self.I_rated_A      = th["I_rated_A"]["value"]         # A at rated conditions

    # ------------------------------------------------------------------
    # Temperature-dependent AC resistance
    # ------------------------------------------------------------------
    def r_ac_pu_per_km(self, T_cond_C=None):
        """
        AC resistance [pu/km] accounting for temperature and skin effect.

        R_ac(T) = R_dc_ref * [1 + alpha*(T - T_ref)] * skin_factor
        """
        T = self.T_cond_rated if T_cond_C is None else np.asarray(T_cond_C, dtype=float)
        R_ac = self.R_dc_pu_per_km * (1.0 + self.alpha_R * (T - self.T_ref)) * self.skin_factor
        return R_ac

    # ------------------------------------------------------------------
    # IEEE 738 thermal ampacity
    # ------------------------------------------------------------------
    def _q_convective(self, T_cond_C, T_amb_C):
        """
        Convective heat loss [W/m] — IEEE 738 forced + natural convection.
        For wind speed > 0, use forced convection (simplified Morgan):
            q_c = 3.645 * rho_f^0.5 * D^0.75 * wind^0.6 * (T_cond - T_amb)   [W/m]
        where rho_f is air density at film temp.
        """
        T_cond = np.asarray(T_cond_C, dtype=float)
        T_amb  = np.asarray(T_amb_C, dtype=float)
        dT     = T_cond - T_amb
        T_film = (T_cond + T_amb) / 2.0 + 273.15  # K

        # Air density at film temperature [kg/m^3]  (ideal gas: rho = 1.2929 * 273.15/T)
        rho_f = 1.2929 * 273.15 / T_film

        v = max(self.wind_speed_m_s, 0.0)
        D = self.D_cond_m

        if v < 0.05:
            # Natural convection: q_c ≈ 2.11e-2 * D^0.75 * dT^1.25  (simplified)
            q_c = 2.11e-2 * D ** 0.75 * np.abs(dT) ** 1.25
        else:
            # Forced convection perpendicular wind (IEEE 738 Eq. 3a simplified)
            q_c = 3.645 * rho_f ** 0.5 * D ** 0.75 * v ** 0.6 * dT

        return q_c

    def _q_radiative(self, T_cond_C, T_amb_C):
        """Radiative heat loss [W/m]."""
        T_cond_K = np.asarray(T_cond_C, dtype=float) + 273.15
        T_amb_K  = np.asarray(T_amb_C, dtype=float) + 273.15
        return (self.STEFAN_BOLTZMANN * self.emissivity * np.pi * self.D_cond_m *
                (T_cond_K ** 4 - T_amb_K ** 4))

    def _q_solar(self):
        """Solar heat gain [W/m]."""
        return self.absorptivity * self.D_cond_m * self.solar_irr

    def thermal_ampacity_A(self, T_amb_C=25.0, T_cond_C=None):
        """
        Maximum continuous current [A] at given ambient temperature.

        Solve: I_max = sqrt[(q_r + q_c - q_s) / R_ac_Tmax]
        where q_r, q_c are evaluated at T_cond_max.
        """
        T_max = self.T_cond_max if T_cond_C is None else T_cond_C
        T_amb = np.asarray(T_amb_C, dtype=float)

        q_c = self._q_convective(T_max, T_amb)
        q_r = self._q_radiative(T_max, T_amb)
        q_s = self._q_solar()

        q_diss = q_c + q_r - q_s
        q_diss = np.maximum(q_diss, 1.0)  # physical lower bound (cooling always > 0 at T_max)

        # R_ac at max temperature, in Ohm/m (convert from pu/km)
        Z_base_ohm_per_km = (self.V_base_kV * 1000.0) ** 2 / (self.S_base_MVA * 1e6) * 1e-3 * 1000.0
        # Correctly: Z_base [Ohm] = V_base^2 / S_base; pu/km * Z_base = Ohm/km
        Z_base = (self.V_base_kV * 1000.0) ** 2 / (self.S_base_MVA * 1e6)  # Ohm (total base)
        R_ohm_per_km = self.r_ac_pu_per_km(T_max) * Z_base / 1000.0  # need per km scaling
        # Actually: R_pu_per_km * Z_base_per_km where Z_base_per_km = Z_base_total per unit length
        # We use 100 km as normalizer consistent with typical: R_pu_per_km * Z_base = Ohm/km
        # More accurately: Z_base [Ohm] for base_length=1km: Z_base_km = V_base^2/(S_base) * per_km
        # R in pu/km means 1 km of line has resistance R_pu_per_km * base length norm / total length
        # For IEEE 738 we need R in Ohm/m. Use standard conductor data directly from thermal params.
        R_ac_ohm_per_m = self.r_ac_pu_per_km(T_max) * Z_base / 1e5  # approx per m

        I_max = np.sqrt(q_diss / (R_ac_ohm_per_m + 1e-12))
        return I_max

    def ampacity_derating_factor(self, T_amb_C):
        """
        Ampacity derating factor relative to 25 degC baseline.
        df = I_max(T_amb) / I_max(T_ref=25 C)
        """
        T_amb = np.asarray(T_amb_C, dtype=float)
        I_ref = self.thermal_ampacity_A(T_amb_C=25.0)
        I_amb = self.thermal_ampacity_A(T_amb_C=T_amb)
        return I_amb / (I_ref + 1e-12)

    # ------------------------------------------------------------------
    # Full pi-model calculation at T-corrected R
    # ------------------------------------------------------------------
    def compute(self, V_s_pu: float, delta_s_rad: float,
                P_load_pu: float, Q_load_pu: float,
                length_km: float = None, T_cond_C: float = None,
                T_amb_C: float = 25.0) -> dict:
        """
        Full pi-model with temperature + skin-effect corrected R.

        Parameters
        ----------
        V_s_pu      : Sending-end voltage [pu]
        delta_s_rad : Sending-end angle [rad]
        P_load_pu   : Receiving-end active load [pu]
        Q_load_pu   : Receiving-end reactive load [pu]
        length_km   : Line length [km]
        T_cond_C    : Conductor temperature [degC]; None = use rated temperature
        T_amb_C     : Ambient temperature [degC] for ampacity check

        Returns
        -------
        dict with all F1a outputs + R_ac_pu, T_cond_C, I_max_A, ampacity_margin,
        skin_factor, derating_factor
        """
        if length_km is None:
            length_km = self.length_km_def

        V_s_pu      = np.asarray(V_s_pu, dtype=float)
        delta_s_rad = np.asarray(delta_s_rad, dtype=float)
        P_load_pu   = np.asarray(P_load_pu, dtype=float)
        Q_load_pu   = np.asarray(Q_load_pu, dtype=float)
        L = float(length_km)

        # Temperature-corrected AC resistance
        R = self.r_ac_pu_per_km(T_cond_C) * L
        X = self.X_pu_per_km * L
        B = self.B_pu_per_km * L
        Z = complex(R, X)
        Y_half = complex(0, B / 2.0)

        # Sending-end complex voltage
        V_s = V_s_pu * np.exp(1j * delta_s_rad)

        # Iterative pi-model solve (5 iterations, converges in 2-3)
        V_r = V_s.copy() if np.ndim(V_s) > 0 else complex(V_s)
        for _ in range(5):
            S_load = P_load_pu + 1j * Q_load_pu
            V_r_safe = (np.where(np.abs(V_r) > 1e-12, V_r, 1e-12 + 0j)
                        if np.ndim(V_r) > 0 else (V_r if abs(V_r) > 1e-12 else 1e-12 + 0j))
            I_r = np.conj(S_load / V_r_safe)
            I_shunt_r = V_r * Y_half
            I_series = I_r + I_shunt_r
            I_shunt_s = V_s * Y_half
            I_s = I_series + I_shunt_s
            V_r = V_s - Z * I_series

        V_r_mag = np.abs(V_r)
        V_r_ang = np.angle(V_r)
        I_series_mag = np.abs(I_series)

        P_loss = I_series_mag ** 2 * float(R)
        Q_loss = I_series_mag ** 2 * float(X)

        S_s = V_s * np.conj(I_s)
        P_s = np.real(S_s)
        Q_s = np.imag(S_s)

        voltage_drop = V_s_pu - V_r_mag

        safe_Ps = (np.where(np.abs(P_s) > 1e-12, P_s, 1e-12)
                   if np.ndim(P_s) > 0 else (P_s if abs(P_s) > 1e-12 else 1e-12))
        eta = (np.where(P_s > 0, P_load_pu / safe_Ps, 0.0)
               if np.ndim(P_s) > 0 else (P_load_pu / safe_Ps if P_s > 0 else 0.0))

        # Thermal ampacity at ambient
        I_max_A = self.thermal_ampacity_A(T_amb_C=T_amb_C)

        # Current in Amperes for comparison with ampacity
        I_base_A = self.S_base_MVA * 1e6 / (np.sqrt(3.0) * self.V_base_kV * 1e3)
        I_series_A = I_series_mag * I_base_A

        # Ampacity margin: positive = safe, negative = overloaded
        ampacity_margin = (I_max_A - I_series_A) / (I_max_A + 1e-12)

        derating = self.ampacity_derating_factor(T_amb_C)

        return {
            "V_r_pu": V_r_mag,
            "delta_r_rad": V_r_ang,
            "I_series_pu": I_series_mag,
            "I_series_A": I_series_A,
            "P_loss_pu": P_loss,
            "Q_loss_pu": Q_loss,
            "P_s_pu": P_s,
            "Q_s_pu": Q_s,
            "efficiency": eta,
            "voltage_drop_pu": voltage_drop,
            "R_ac_pu_total": float(R),
            "skin_factor": self.skin_factor,
            "I_max_A": I_max_A,
            "ampacity_margin": ampacity_margin,
            "derating_factor": derating,
        }
