"""
EC176 — PMSM — F1b Efficiency + Thermal

Extends F1a loss-separation model with temperature dependence:

1. PM flux demagnetization with temperature (NdFeB):
     Phi_m(T) = Phi_m_ref * (1 + alpha_Br * (T_magnet - T_ref))
   alpha_Br ~ -0.0012/K => flux DECREASES with temperature.
   Lower flux => lower back-EMF => higher current for same torque => more copper loss.

2. Stator resistance increases with temperature (copper):
     R_s(T) = R_s_ref * (1 + alpha_Cu * (T_winding - T_ref))

3. Torque constant degrades with flux:
     k_t(T) = Phi_m(T) * (3/2) * p  (simplified; proportional to Phi_m)
   For prediction: k_t(T) = k_t_ref * Phi_m(T)/Phi_m_ref

4. Back-EMF:
     E_back = Phi_m(T) * omega_elec = Phi_m(T) * p * omega_mech

5. Demagnetization risk flag above 150C magnet temperature.

6. Derating above 40C ambient (similar to IEC 60034 convention).

References:
    Gieras, J.F. (2010). Permanent Magnet Motor Technology, 3rd ed. CRC Press.
    Sebastian, T. (1995). IEEE Trans. Magnetics, 31(4), 2578-2584.
"""

import numpy as np

_RPM_TO_RADS = np.pi / 30.0


class PMSMF1b:
    """PMSM efficiency model with thermal demagnetization and resistance effects."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated = u["P_rated_W"]["value"]                # W
        self.T_rated = u["T_rated_Nm"]["value"]               # Nm
        self.omega_base = u["omega_base_rpm"]["value"]         # rpm
        self.omega_max = u["omega_max_rpm"]["value"]           # rpm
        self.pole_pairs = u["pole_pairs"]["value"]
        self.Phi_m_ref = u["Phi_m_ref"]["value"]               # Wb
        self.T_ref = u["T_ref"]["value"]                       # degC
        self.alpha_Br = u["alpha_Br"]["value"]                 # 1/K (negative)
        self.R_s_ref = u["R_s_ref"]["value"]                   # ohm
        self.alpha_Cu = u["alpha_Cu"]["value"]                 # 1/K
        self.k_e = u["k_e"]["value"]                           # iron loss coeff
        self.k_f = u["k_f"]["value"]                           # mech loss coeff
        self.T_demag = u["T_demag"]["value"]                   # degC
        # Torque constant at reference: k_t = T_rated / I_rated
        # We compute k_t_ref so that at ref conditions the model is consistent
        # k_t is proportional to Phi_m
        self.k_t_ref = (1.5 * self.pole_pairs * self.Phi_m_ref)

    def flux(self, magnet_temperature):
        """PM flux linkage at magnet temperature [Wb]."""
        T_m = np.asarray(magnet_temperature, dtype=float)
        return self.Phi_m_ref * (1.0 + self.alpha_Br * (T_m - self.T_ref))

    def torque_constant(self, magnet_temperature):
        """Torque constant k_t(T) [Nm/A], proportional to PM flux."""
        Phi = self.flux(magnet_temperature)
        return 1.5 * self.pole_pairs * Phi

    def stator_resistance(self, winding_temperature=None, magnet_temperature=None):
        """R_s(T) = R_s_ref * (1 + alpha_Cu*(T-T_ref)).
        Uses magnet_temperature as proxy for winding if winding not given."""
        if winding_temperature is not None:
            T = np.asarray(winding_temperature, dtype=float)
        elif magnet_temperature is not None:
            # Winding temp ~ magnet temp (thermal coupling in same frame)
            T = np.asarray(magnet_temperature, dtype=float)
        else:
            T = np.float64(self.T_ref)
        return self.R_s_ref * (1.0 + self.alpha_Cu * (T - self.T_ref))

    def back_emf(self, speed_rpm, magnet_temperature=80.0):
        """Back-EMF peak voltage [V] = Phi_m(T) * p * omega_mech."""
        Phi = self.flux(magnet_temperature)
        omega_mech = np.asarray(speed_rpm, dtype=float) * _RPM_TO_RADS
        return Phi * self.pole_pairs * omega_mech

    def losses(self, torque_nm, speed_rpm, magnet_temperature=80.0):
        """
        Loss breakdown with thermal effects.

        Returns dict with p_copper_w, p_iron_w, p_mech_w, p_total_w.
        """
        T = np.asarray(torque_nm, dtype=float)
        omega = np.asarray(speed_rpm, dtype=float)

        k_t = self.torque_constant(magnet_temperature)
        R_s = self.stator_resistance(magnet_temperature=magnet_temperature)

        # Current: I = T / k_t (higher at high temp because k_t drops)
        I = np.where(k_t > 1e-9, T / k_t, 0.0)
        P_copper = I ** 2 * R_s

        # Iron loss (temperature-independent approximation)
        P_iron = self.k_e * np.abs(omega) ** 1.5

        # Mechanical loss
        P_mech = self.k_f * np.abs(omega)

        P_total = P_copper + P_iron + P_mech

        return {
            "p_copper_w": P_copper,
            "p_iron_w": P_iron,
            "p_mech_w": P_mech,
            "p_total_w": P_total,
        }

    def output_power(self, torque_nm, speed_rpm):
        """Mechanical output power [W]."""
        T = np.asarray(torque_nm, dtype=float)
        omega_rad = np.asarray(speed_rpm, dtype=float) * _RPM_TO_RADS
        return T * omega_rad

    def input_power(self, torque_nm, speed_rpm, magnet_temperature=80.0):
        """Electrical input power [W]."""
        p_out = self.output_power(torque_nm, speed_rpm)
        loss = self.losses(torque_nm, speed_rpm, magnet_temperature)
        return p_out + loss["p_total_w"]

    def efficiency(self, torque_nm, speed_rpm, magnet_temperature=80.0):
        """Motor efficiency with thermal effects."""
        p_out = self.output_power(torque_nm, speed_rpm)
        p_in = self.input_power(torque_nm, speed_rpm, magnet_temperature)
        eta = np.where(p_in > 1e-6, p_out / p_in, 0.0)
        return np.clip(eta, 0.0, 1.0)

    def derating_factor(self, magnet_temperature=80.0, ambient_temperature=25.0):
        """
        Derating based on:
          1. Flux reduction (automatic via efficiency drop)
          2. Demagnetization risk above T_demag (hard limit)
          3. Ambient derating above 40C
        Returns [0, 1].
        """
        T_m = np.asarray(magnet_temperature, dtype=float)
        T_a = np.asarray(ambient_temperature, dtype=float)

        # Demag risk: sharp cutoff above T_demag
        demag_factor = np.where(T_m < self.T_demag, 1.0,
                                np.clip(1.0 - 0.05 * (T_m - self.T_demag), 0.0, 1.0))

        # Ambient derating above 40C
        ambient_factor = np.clip(1.0 - 0.01 * np.maximum(T_a - 40.0, 0.0), 0.0, 1.0)

        return demag_factor * ambient_factor

    def demagnetization_risk(self, magnet_temperature):
        """Boolean flag: True if magnet temp exceeds demagnetization threshold."""
        T_m = np.asarray(magnet_temperature, dtype=float)
        return T_m >= self.T_demag
