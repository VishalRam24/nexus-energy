"""
EC172 — Power Transformer (Grid-Scale) — F1b Core+Copper Loss Model

Extends the ideal transformer model (F1a) with:

1. Core loss (no-load loss) — voltage-dependent:
     P_core = P_no_load * (V_pu / V_rated_pu)^n_core
     Typically n_core ~ 2 for eddy-current dominant (sinusoidal)
     This reflects that core loss ~ B^2 ~ V^2 at power frequency.

2. Copper (load) loss — current-dependent, temperature-corrected:
     P_cu_ref at rated current and T_ref_winding (IEC 60076 specifies 75C)
     P_cu(PLR, T_w) = P_cu_ref * PLR^2 * R_ratio(T_w)
     R_ratio(T_w) = (1 + alpha_Cu * (T_w - T_ref_winding))
     This captures temperature rise causing resistance increase.

3. Efficiency:
     P_out = PLR * S_rated * pf     [W, with power factor]
     P_in  = P_out + P_core + P_cu
     eta   = P_out / P_in

4. Temperature rise model (IEC 60076-7 simplified):
     theta_oil_rise = theta_oil_rated * PLR^(2*n_oil / (1+n_oil))
                    (simplified, steady-state hot-spot model)
     theta_winding_rise = theta_winding_gradient * PLR^n_winding
     theta_hot_spot = T_ambient + theta_oil_rise + theta_winding_rise

References:
    IEC 60076-1:2011 — Power transformers: General
    IEC 60076-7:2018 — Loading guide
    Kulkarni & Khaparde (2004). Transformer Engineering. CRC Press.
    Montsinger (1930). Loading transformers by temperature. AIEE Trans.
"""

import numpy as np


class PowerTransformerF1b:
    """Grid-scale power transformer — core + copper loss model with thermal."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_rated = u["S_rated_MVA"]["value"] * 1e6   # VA
        self.P_no_load = u["P_no_load_kW"]["value"] * 1000.0   # W (core loss at rated V)
        self.P_load_loss_ref = u["P_load_loss_kW"]["value"] * 1000.0  # W at rated I, T_ref_winding
        self.T_ref_winding = u["T_ref_winding"]["value"]  # degC (IEC 60076: 75C)
        self.T_ref = u["T_ref"]["value"]
        self.alpha_Cu = u["alpha_Cu"]["value"]
        self.n_core = u["core_voltage_exponent"]["value"]
        self.theta_oil_rated = u["theta_oil_rise_rated"]["value"]   # K
        self.theta_winding_grad = u["theta_winding_gradient"]["value"]  # K
        self.n_oil = u["n_oil"]["value"]
        self.n_winding = u["n_winding"]["value"]

    # --- loss components ---

    def core_loss(self, voltage_pu=1.0):
        """
        Core (no-load) loss [W].
        P_core = P_no_load * (V_pu)^n_core
        """
        v = np.asarray(voltage_pu, dtype=float)
        return self.P_no_load * v ** self.n_core

    def copper_loss(self, load_fraction, winding_temperature=75.0):
        """
        Copper (load) loss [W] — temperature corrected.
        P_cu = P_cu_ref * PLR^2 * (1 + alpha_Cu*(T_w - T_ref_winding))
        """
        plr = np.asarray(load_fraction, dtype=float)
        T_w = np.asarray(winding_temperature, dtype=float)
        R_ratio = 1.0 + self.alpha_Cu * (T_w - self.T_ref_winding)
        return self.P_load_loss_ref * plr**2 * R_ratio

    def total_losses(self, load_fraction, voltage_pu=1.0, winding_temperature=75.0):
        """Total transformer losses [W] = P_core + P_cu."""
        return self.core_loss(voltage_pu) + self.copper_loss(load_fraction, winding_temperature)

    def loss_breakdown(self, load_fraction, voltage_pu=1.0, winding_temperature=75.0):
        """Loss components [W]."""
        return {
            "p_core_w": self.core_loss(voltage_pu),
            "p_copper_w": self.copper_loss(load_fraction, winding_temperature),
            "p_total_w": self.total_losses(load_fraction, voltage_pu, winding_temperature),
        }

    # --- power and efficiency ---

    def output_power(self, load_fraction, power_factor=1.0):
        """Active power output [W] = PLR * S_rated * pf."""
        plr = np.asarray(load_fraction, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        return plr * self.S_rated * pf

    def input_power(self, load_fraction, voltage_pu=1.0, winding_temperature=75.0,
                    power_factor=1.0):
        """Input power [W] = P_out + losses."""
        P_out = self.output_power(load_fraction, power_factor)
        P_loss = self.total_losses(load_fraction, voltage_pu, winding_temperature)
        return P_out + P_loss

    def efficiency(self, load_fraction, voltage_pu=1.0, winding_temperature=75.0,
                   power_factor=1.0):
        """Efficiency = P_out / P_in."""
        P_out = self.output_power(load_fraction, power_factor)
        P_in = self.input_power(load_fraction, voltage_pu, winding_temperature, power_factor)
        eta = np.where(P_in > 0, P_out / P_in, 0.0)
        return np.clip(eta, 0.0, 1.0)

    # --- thermal model ---

    def hot_spot_temperature(self, load_fraction, ambient_temperature=20.0):
        """
        Steady-state hot-spot temperature [degC].
        IEC 60076-7 simplified:
            theta_top_oil = T_amb + theta_oil_rated * (PLR^2)^(n_oil/(1+n_oil))
            theta_hot_spot = theta_top_oil + theta_winding_grad * PLR^n_winding
        """
        plr = np.asarray(load_fraction, dtype=float)
        T_amb = np.asarray(ambient_temperature, dtype=float)

        # Top-oil rise: load-loss ratio is PLR^2, but thermal model uses:
        # theta_oil = theta_oil_rated * ((P_cu + P_core) / (P_cu_rated + P_core_rated))^n_oil
        # Simplified: at rated PLR=1, theta_oil = theta_oil_rated
        # Loss ratio ~ (P_no_load + PLR^2 * P_load_loss) / (P_no_load + P_load_loss)
        P_core = self.P_no_load
        P_cu_rated = self.P_load_loss_ref
        loss_ratio = (P_core + plr**2 * P_cu_rated) / (P_core + P_cu_rated)
        theta_oil = self.theta_oil_rated * loss_ratio**self.n_oil

        # Winding-to-oil gradient at load
        theta_wind = self.theta_winding_grad * np.maximum(plr, 0.0)**self.n_winding

        return T_amb + theta_oil + theta_wind

    def temperature_rise(self, load_fraction, ambient_temperature=20.0):
        """Total winding temperature rise [K] above ambient."""
        T_hot = self.hot_spot_temperature(load_fraction, ambient_temperature)
        return T_hot - np.asarray(ambient_temperature, dtype=float)
