"""
EC017 — Hydrogen Purifier (PSA) — F1a Recovery-Purity Semi-Empirical Model

Pressure Swing Adsorption (PSA) simplified model:

H2 Recovery:
    η_rec = η_rec_nom * (P_feed / P_ref)^α * (1 + β*(y_H2 - y_nom))
    Clipped to [0, 1]. Higher feed pressure → better adsorption selectivity → higher recovery.

Product flow:
    F_product = F_feed * y_H2 * η_rec / purity_product

Tail gas (purge) flow:
    F_tail = F_feed - F_product

Specific energy:
    W_spec = W_nom * (1 + γ * (P_ref/P_feed - 1))   [kWh/kg_H2]
    Higher feed pressure → less recompression overhead → lower specific energy.

Purity is a target input; recovery trades off with purity via:
    η_rec_adj = η_rec * (1 - k_pur * (target_purity - purity_nom))
    where k_pur captures the recovery-purity trade-off.

References:
    Sircar & Golden (2000). Sep. Sci. Technol. 35(5), 667-687.
    Yang, R.T. (1987). Gas Separation by Adsorption Processes. Butterworths.
    DOE Hydrogen Program: https://www.hydrogen.energy.gov
"""

import numpy as np


class HydrogenPurifierPSAF1a:
    """PSA hydrogen purifier semi-empirical model."""

    def __init__(self, params: dict):
        psa = params["psa_unit"]
        perf = params["performance"]
        thermo = params["thermodynamics"]

        self.P_ref = psa["feed_pressure_ref_bar"]["value"]
        self.P_purge = psa["purge_pressure_bar"]["value"]
        self.y_nom = psa["feed_h2_fraction_nominal"]["value"]

        self.eta_nom = perf["recovery_nominal"]["value"]
        self.purity_nom = perf["purity_nominal"]["value"]
        self.W_nom = perf["specific_energy_kWh_per_kg_H2"]["value"]
        self.alpha = perf["recovery_pressure_exponent"]["value"]
        self.beta = perf["recovery_feed_h2_coefficient"]["value"]
        self.gamma = perf["energy_pressure_factor"]["value"]

        self.M_H2 = thermo["M_H2"]["value"]

        # Recovery–purity trade-off coefficient:
        # For every 0.001 increase in target purity above nominal, recovery drops ~2%.
        # Based on typical PSA operating curves. Sircar & Golden 2000.
        self.k_purity = 20.0  # fractional recovery loss per unit purity increase

    def recovery(self, P_feed_bar, y_H2_feed, target_purity=None):
        """
        H2 recovery fraction η_rec (0 < η_rec < 1).

        Args:
            P_feed_bar:    Feed pressure [bar]
            y_H2_feed:     Feed H2 mole fraction (0–1)
            target_purity: Target product purity (optional, adjusts recovery)

        Returns:
            η_rec [dimensionless]
        """
        P = np.asarray(P_feed_bar, dtype=float)
        y = np.asarray(y_H2_feed, dtype=float)

        # Pressure effect: higher pressure improves selectivity
        P_effect = (P / self.P_ref) ** self.alpha

        # Feed composition effect: richer feed → easier separation → higher recovery
        y_effect = 1.0 + self.beta * (y - self.y_nom)
        y_effect = np.clip(y_effect, 0.5, 1.5)

        eta = self.eta_nom * P_effect * y_effect

        # Purity trade-off: demanding higher purity sacrifices recovery
        if target_purity is not None:
            purity = np.asarray(target_purity, dtype=float)
            purity_penalty = 1.0 - self.k_purity * np.maximum(purity - self.purity_nom, 0.0)
            purity_penalty = np.clip(purity_penalty, 0.5, 1.0)
            eta = eta * purity_penalty

        return np.clip(eta, 0.0, 0.999)

    def product_flow(self, feed_flow_kg_s, y_H2_feed, eta_rec, target_purity):
        """
        Product (H2-rich) mass flow rate [kg/s].

        Product contains H2 at purity y_prod; remaining is impurities.
        Mass balance: F_product * y_prod = F_feed * y_H2 * eta_rec
        => F_product = F_feed * y_H2 * eta_rec / y_prod  (approximate, pure H2 product)
        """
        F = np.asarray(feed_flow_kg_s, dtype=float)
        y = np.asarray(y_H2_feed, dtype=float)
        eta = np.asarray(eta_rec, dtype=float)
        y_prod = np.asarray(target_purity, dtype=float)

        # H2 recovered [kg/s]
        F_H2_recovered = F * y * eta
        # Product flow assuming product is essentially pure H2 (purity > 99%)
        F_product = F_H2_recovered / y_prod
        return F_product

    def tail_gas_flow(self, feed_flow_kg_s, product_flow_kg_s):
        """Tail gas (purge/waste) flow [kg/s] = feed - product."""
        return np.asarray(feed_flow_kg_s, dtype=float) - np.asarray(product_flow_kg_s, dtype=float)

    def specific_energy(self, P_feed_bar):
        """
        Specific electric energy consumption [kWh/kg_H2].

        Higher feed pressure → lower specific energy (larger ΔP across the bed
        reduces blower/vacuum-pump energy per kg H2 recovered).

        Scaling: W = W_nom * (P_ref / P_feed)^gamma_P
        where gamma_P is a small positive exponent (~0.15).
        Sircar & Golden (2000): energy decreases with increasing pressure ratio P_H/P_L.
        """
        P = np.asarray(P_feed_bar, dtype=float)
        W = self.W_nom * (self.P_ref / P) ** abs(self.gamma)
        return np.clip(W, 0.51, 5.0)  # physical bounds: >0.5 kWh/kg

    def electric_power(self, product_flow_kg_s, P_feed_bar):
        """
        Electric power consumed [kW].

        Args:
            product_flow_kg_s: H2 product flow rate [kg/s]
            P_feed_bar:        Feed pressure [bar]

        Returns:
            Power [kW]
        """
        W_spec = self.specific_energy(P_feed_bar)          # kWh/kg_H2
        m_dot_h2 = np.asarray(product_flow_kg_s, dtype=float)  # kg/s
        # Power [kW] = specific_energy [kWh/kg] * flow [kg/s] * 3600 s/h
        return W_spec * m_dot_h2 * 3600.0

    def pressure_ratio(self, P_feed_bar):
        """Pressure ratio P_feed / P_purge (key PSA design parameter)."""
        return np.asarray(P_feed_bar, dtype=float) / self.P_purge
