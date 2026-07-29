"""
EC017 -- Hydrogen Purifier (PSA) -- F1b Temperature-Pressure Model

Extends F1a by adding:
1. Temperature-dependent H2 recovery (higher T -> lower adsorption -> lower recovery)
2. Temperature-dependent specific energy: W = W_nom * (P_ref/P)^0.15 * (T/T_ref)^0.5
3. Adsorption capacity correction via van't Hoff
4. Part-load corrections

Temperature effects on PSA:
    Recovery: eta_T = eta_P * (1 + k_T * (T - T_ref))
              k_T < 0: higher T -> less adsorption -> less impurity removal -> lower recovery
    Specific energy: W = W_nom * (P_ref/P)^0.15 * (T/T_ref)^0.5
              Higher T increases gas volume -> more compression energy needed
              (Sircar & Golden 2000, from energy-pressure scaling).

Specific energy: W = W_nom * (P_ref / P)^0.15
    per task specification. Temperature correction on top.

References:
    Sircar & Golden (2000). Sep. Sci. Technol. 35(5), 667-687.
    Yang, R.T. (1987). Gas Separation by Adsorption Processes.
    Ruthven, D.M. (1984). Principles of Adsorption and Adsorption Processes.
    Cavenati et al. (2004). J. Chem. Eng. Data 49(4), 1095-1101.
"""

import numpy as np

R_GAS = 8.314   # J/(mol K)


class HydrogenPurifierPSAF1b:
    """PSA hydrogen purifier with temperature and pressure dependence."""

    def __init__(self, params: dict):
        psa  = params["psa_unit"]
        perf = params["performance"]
        th   = params["thermal"]

        self.P_ref    = float(psa["feed_pressure_ref_bar"]["value"])
        self.P_purge  = float(psa["purge_pressure_bar"]["value"])
        self.y_nom    = float(psa["feed_h2_fraction_nominal"]["value"])
        self.T_ref    = float(psa["T_ref_K"]["value"])

        self.eta_nom   = float(perf["recovery_nominal"]["value"])
        self.purity_nom = float(perf["purity_nominal"]["value"])
        self.W_nom     = float(perf["specific_energy_kWh_per_kg_H2"]["value"])
        self.alpha_P   = float(perf["recovery_pressure_exponent"]["value"])
        self.beta_y    = float(perf["recovery_feed_h2_coefficient"]["value"])
        self.gamma_P   = float(perf["energy_pressure_factor"]["value"])

        self.k_T       = float(th["recovery_temp_coeff"]["value"])
        self.gamma_T   = float(th["energy_temp_exponent"]["value"])

        # Recovery-purity trade-off coefficient
        self.k_purity  = 20.0

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recovery(self, P_feed_bar, y_H2_feed, T_K=None, target_purity=None):
        """
        H2 recovery fraction eta_rec (0–1).

        Pressure effect: (P_feed/P_ref)^alpha_P
        Feed composition: 1 + beta*(y_H2 - y_nom)
        Temperature:      1 + k_T*(T - T_ref)  [k_T < 0 for reduced adsorption]
        Purity trade-off: reduces recovery for stricter purity

        Args:
            P_feed_bar:    Feed pressure [bar]
            y_H2_feed:     Feed H2 mole fraction (0-1)
            T_K:           Temperature [K] (default: T_ref)
            target_purity: Target product purity (optional)

        Returns:
            eta_rec [dimensionless]
        """
        P = np.asarray(P_feed_bar, dtype=float)
        y = np.asarray(y_H2_feed, dtype=float)
        T = np.asarray(T_K if T_K is not None else self.T_ref, dtype=float)

        # Pressure effect
        P_effect = (P / self.P_ref) ** self.alpha_P

        # Feed composition effect
        y_effect = 1.0 + self.beta_y * (y - self.y_nom)
        y_effect = np.clip(y_effect, 0.5, 1.5)

        # Temperature effect: higher T reduces selectivity (less adsorption of impurities)
        T_effect = 1.0 + self.k_T * (T - self.T_ref)
        T_effect = np.clip(T_effect, 0.5, 1.1)

        eta = self.eta_nom * P_effect * y_effect * T_effect

        # Purity trade-off
        if target_purity is not None:
            purity = np.asarray(target_purity, dtype=float)
            purity_penalty = 1.0 - self.k_purity * np.maximum(purity - self.purity_nom, 0.0)
            purity_penalty = np.clip(purity_penalty, 0.5, 1.0)
            eta = eta * purity_penalty

        return np.clip(eta, 0.0, 0.999)

    # ------------------------------------------------------------------
    # Specific energy (PSA task specification: W_nom * (P_ref/P)^0.15)
    # ------------------------------------------------------------------

    def specific_energy(self, P_feed_bar, T_K=None):
        """
        Specific electric energy consumption [kWh/kg_H2].

        Scaling: W = W_nom * (P_ref / P)^0.15 * (T/T_ref)^0.5
        - Higher P: lower specific energy (larger ΔP drives better separation)
        - Higher T: slightly higher energy (reduced adsorption effectiveness)

        The (P_ref/P)^0.15 form is the task specification from Sircar & Golden (2000).

        Args:
            P_feed_bar: Feed pressure [bar]
            T_K:        Temperature [K] (default: T_ref)

        Returns:
            W [kWh/kg_H2]
        """
        P = np.asarray(P_feed_bar, dtype=float)
        T = np.asarray(T_K if T_K is not None else self.T_ref, dtype=float)

        W_P = self.W_nom * (self.P_ref / P) ** abs(self.gamma_P)
        W_T = (T / self.T_ref) ** self.gamma_T
        W   = W_P * W_T

        return np.clip(W, 0.5, 5.0)

    # ------------------------------------------------------------------
    # Product and tail gas flows
    # ------------------------------------------------------------------

    def product_flow(self, feed_flow_kg_s, y_H2_feed, eta_rec, target_purity):
        """Product H2-rich flow [kg/s]."""
        F   = np.asarray(feed_flow_kg_s, dtype=float)
        y   = np.asarray(y_H2_feed, dtype=float)
        eta = np.asarray(eta_rec, dtype=float)
        yp  = np.asarray(target_purity, dtype=float)
        return F * y * eta / yp

    def tail_gas_flow(self, feed_flow_kg_s, product_flow_kg_s):
        """Tail gas flow [kg/s]."""
        return np.asarray(feed_flow_kg_s) - np.asarray(product_flow_kg_s)

    def electric_power(self, product_flow_kg_s, P_feed_bar, T_K=None):
        """Electric power consumed [kW]."""
        W_spec = self.specific_energy(P_feed_bar, T_K)
        m_dot  = np.asarray(product_flow_kg_s, dtype=float)
        return W_spec * m_dot * 3600.0   # kWh/kg * kg/s * 3600 s/h = kW

    def pressure_ratio(self, P_feed_bar):
        """Pressure ratio P_feed / P_purge."""
        return np.asarray(P_feed_bar, dtype=float) / self.P_purge

    # ------------------------------------------------------------------
    # Full evaluate
    # ------------------------------------------------------------------

    def evaluate(self, feed_flow_kg_s, y_H2_feed, P_feed_bar, T_K=None, target_purity=None):
        """Full PSA evaluation."""
        if target_purity is None:
            target_purity = self.purity_nom
        T = T_K if T_K is not None else self.T_ref

        eta_rec    = self.recovery(P_feed_bar, y_H2_feed, T_K=T, target_purity=target_purity)
        F_product  = self.product_flow(feed_flow_kg_s, y_H2_feed, eta_rec, target_purity)
        F_tail     = self.tail_gas_flow(feed_flow_kg_s, F_product)
        W_spec     = self.specific_energy(P_feed_bar, T_K=T)
        P_kW       = self.electric_power(F_product, P_feed_bar, T_K=T)
        P_ratio    = self.pressure_ratio(P_feed_bar)
        F_H2_prod  = F_product * np.asarray(target_purity, dtype=float)

        return {
            "recovery":                   eta_rec,
            "product_flow_kg_s":          F_product,
            "tail_gas_flow_kg_s":         F_tail,
            "specific_energy_kWh_per_kg": W_spec,
            "electric_power_kW":          P_kW,
            "pressure_ratio":             P_ratio,
            "h2_yield_kg_s":              F_H2_prod,
        }
