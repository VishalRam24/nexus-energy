"""
EC209 — Reverse Osmosis (RO) — F1b Fouling + Temperature Model

Extends F1a SEC model with:
  1. Membrane fouling: A(t) = A0 * exp(-k_foul * t/8760), k_foul ~ 0.1/year.
     Permeability decreases exponentially with operating hours.
  2. Temperature-dependent flux: J_w(T) = J_ref * exp(2500*(1/T_ref - 1/T)).
     Warmer water → higher flux (lower viscosity).
  3. Osmotic pressure: pi = 0.7 * S/1000 bar (S in ppm).
     Accounts for concentration polarization at membrane surface.
  4. Salt rejection degrades slightly with fouling.

Water flux: J_w = A(t) * T_factor * (P_feed - pi_avg)
Permeate flow: Q_p = J_w * A_membrane
SEC = P_feed / (eta_pump * recovery * 36) with ERD correction

Reference:
    Elimelech, M. & Phillip, W. A. (2011). Science, 333, 712-717.
    Kang, G. & Cao, Y. (2012). Water Research, 46(3), 584-600.
"""

import numpy as np


class ROF1b:
    """Reverse osmosis — fouling + temperature-dependent model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A0 = u["A0"]["value"]  # L/(m2*h*bar)
        self.membrane_area = u["membrane_area_m2"]["value"]
        self.B = u["B"]["value"]  # L/(m2*h)
        self.eta_pump = u["pump_efficiency"]["value"]
        self.k_foul = u["k_foul"]["value"]
        self.eta_ERD = u["eta_ERD"]["value"]
        self.dP_membrane = u["dP_membrane"]["value"]
        self.pi_coeff = u["osmotic_pressure_coeff"]["value"]
        self.T_ref_K = u["T_ref_K"]["value"]
        self.E_factor = u["activation_energy_factor"]["value"]
        self.rejection_clean = u["salt_rejection_clean"]["value"]

    def _fouling_factor(self, operating_hours):
        """Membrane permeability decline due to fouling.
        A(t) = A0 * exp(-k_foul * t/8760)
        Returns factor in [0, 1].
        """
        t = np.asarray(operating_hours, dtype=float)
        return np.exp(-self.k_foul * t / 8760.0)

    def _temperature_factor(self, feed_temperature_degC):
        """Temperature correction for water flux.
        J(T) = J_ref * exp(E*(1/T_ref - 1/T))
        At T > T_ref: factor > 1 (higher flux).
        """
        T = np.asarray(feed_temperature_degC, dtype=float) + 273.15
        return np.exp(self.E_factor * (1.0 / self.T_ref_K - 1.0 / T))

    def osmotic_pressure(self, feed_salinity_ppm):
        """Osmotic pressure (bar) from feed salinity (ppm).
        pi = 0.7 * S/1000
        """
        S = np.asarray(feed_salinity_ppm, dtype=float)
        return self.pi_coeff * S / 1000.0

    def flux_decline_factor(self, operating_hours, feed_temperature_degC):
        """Combined flux factor = fouling * temperature.
        >1 if temperature effect dominates, <1 if fouling dominates.
        """
        return self._fouling_factor(operating_hours) * self._temperature_factor(feed_temperature_degC)

    def permeate_flow_m3_h(self, feed_salinity_ppm, feed_pressure_bar,
                            feed_temperature_degC, recovery_ratio, operating_hours):
        """Permeate flow rate (m3/h).
        J_w = A_eff * (P_feed - pi_avg)   [L/(m2*h)]
        Q_p = J_w * A_membrane / 1000     [m3/h]

        pi_avg = average osmotic pressure (between feed and concentrate sides)
        pi_avg = pi_feed * (1 + 1/(1-recovery)) / 2
        """
        S = np.asarray(feed_salinity_ppm, dtype=float)
        P = np.asarray(feed_pressure_bar, dtype=float)
        r = np.asarray(recovery_ratio, dtype=float)

        # Effective permeability
        foul_f = self._fouling_factor(operating_hours)
        temp_f = self._temperature_factor(feed_temperature_degC)
        A_eff = self.A0 * foul_f * temp_f

        # Average osmotic pressure (feed + concentrate average)
        pi_feed = self.osmotic_pressure(S)
        pi_conc = pi_feed / (1.0 - r)
        pi_avg = (pi_feed + pi_conc) / 2.0

        # Net driving pressure
        NDP = np.clip(P - pi_avg - self.dP_membrane, 0.0, None)

        # Water flux [L/(m2*h)]
        J_w = A_eff * NDP

        # Permeate flow [m3/h]
        Q_p = J_w * self.membrane_area / 1000.0
        return np.clip(Q_p, 0.0, None)

    def sec_kwh_m3(self, feed_salinity_ppm, feed_pressure_bar, feed_temperature_degC,
                    recovery_ratio, operating_hours):
        """Specific energy consumption (kWh/m3 permeate).
        With energy recovery device on brine.
        """
        S = np.asarray(feed_salinity_ppm, dtype=float)
        P = np.asarray(feed_pressure_bar, dtype=float)
        r = np.asarray(recovery_ratio, dtype=float)

        # Brine back-pressure
        P_brine = P - self.dP_membrane

        # Net hydraulic work per m3 permeate
        numerator = P / self.eta_pump - P_brine * (1.0 - r) * self.eta_ERD
        sec_bar = numerator / r
        sec_kwh = sec_bar / 36.0

        # Fouling increases SEC indirectly (need higher pressure or lower recovery)
        # Here we model it as: if flux declines, SEC per actual m3 produced stays same
        # but throughput drops. The SEC itself increases slightly due to concentration polarization.
        foul_f = self._fouling_factor(operating_hours)
        # Degraded membrane: slight SEC penalty (more concentration polarization)
        sec_penalty = 1.0 + 0.1 * (1.0 - foul_f)

        return np.clip(sec_kwh * sec_penalty, 0.5, 20.0)

    def rejection_pct(self, operating_hours):
        """Salt rejection (%) — decreases slightly with fouling/aging.
        R = R_clean * (1 - 0.01 * (1 - foul_factor))
        """
        foul_f = self._fouling_factor(operating_hours)
        R = self.rejection_clean * (1.0 - 0.01 * (1.0 - foul_f))
        return np.clip(R * 100.0, 90.0, 100.0)

    def compute(self, feed_salinity_ppm, feed_pressure_bar, feed_temperature_degC,
                recovery_ratio, operating_hours):
        """Full computation returning all outputs."""
        Q_p = self.permeate_flow_m3_h(feed_salinity_ppm, feed_pressure_bar,
                                       feed_temperature_degC, recovery_ratio,
                                       operating_hours)
        sec = self.sec_kwh_m3(feed_salinity_ppm, feed_pressure_bar,
                               feed_temperature_degC, recovery_ratio, operating_hours)
        rej = self.rejection_pct(operating_hours)
        flux_f = self.flux_decline_factor(operating_hours, feed_temperature_degC)

        return {
            "permeate_flow_m3_h": Q_p,
            "sec_kwh_m3": sec,
            "rejection_pct": rej,
            "flux_decline_factor": flux_f,
        }
