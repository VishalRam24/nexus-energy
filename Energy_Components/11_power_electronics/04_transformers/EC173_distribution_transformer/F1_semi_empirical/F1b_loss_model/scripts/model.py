"""
EC173 -- Distribution Transformer -- F1b Core + Copper Loss Model

Extends F1a (ideal transformation + fixed η) with physically meaningful
loss separation following IEC 60076-1:2011 and IEC 60076-7:2018.

Distribution transformers (typically 25 kVA – 2.5 MVA, 11/0.4 kV) differ from
grid-scale transformers (EC172) by:
  • Smaller size → relatively higher no-load loss fraction
  • ONAN (oil-natural air-natural) or ANAF cooling class
  • IEC-defined test quantities: P_0 (no-load), P_k (short-circuit), u_k (impedance)

Loss model:
──────────────────────────────────────────────────────────────────
1. No-load (core) loss — Steinmetz expression, voltage-dependent:
     P_core = P_0 * (V/V_rated)^n_B
     where n_B ≈ 1.6–2.0 (depends on core lamination type)
     P_0 is the rated no-load loss (given in spec sheet)

2. Load (copper) loss — I^2*R, temperature-corrected:
     P_k(T) = P_k_ref * PLR^2 * (1 + alpha_Cu * (T_w - T_ref))
     PLR = load power ratio (= I_2 / I_2_rated)
     T_ref = 75°C (IEC 60076 reference temperature)

3. Stray loss (additional losses in tank walls, clamps, etc.):
     P_stray = k_stray * P_k_ref * PLR^1.5
     k_stray ~ 0.05–0.15 for distribution transformers

4. Efficiency:
     P_out = PLR * S_rated * pf
     P_in  = P_out + P_core + P_k + P_stray
     eta   = P_out / P_in

5. Part-load efficiency curve:
     Maximum efficiency occurs where P_core = P_k (load loss = no-load loss):
     PLR_opt = sqrt(P_0 / P_k_ref)   [typically 40–60% of rated]

6. Hot-spot temperature (IEC 60076-7 simplified):
     theta_oil_rise = theta_oil_rated * (loss_ratio)^n_oil
     theta_hot_spot = T_amb + theta_oil_rise + theta_winding_grad * PLR^n_wind

References:
    IEC 60076-1:2011. Power transformers — Part 1: General
    IEC 60076-7:2018. Power transformers — Part 7: Loading guide
    Kulkarni, S.V. & Khaparde, S.A. (2004). Transformer Engineering. CRC Press.
    ABB (2016). Distribution Transformer Manual.
"""

import numpy as np


class DistributionTransformerF1b:
    """Distribution transformer — IEC core + copper + stray loss model with thermal."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.S_rated = u["S_rated_kVA"]["value"] * 1000.0   # VA
        self.V_hv = u["V_hv"]["value"]                      # V
        self.V_lv = u["V_lv"]["value"]                      # V
        self.P_0 = u["P_no_load_W"]["value"]                # W  (IEC: P_0)
        self.P_k_ref = u["P_load_loss_W"]["value"]          # W  (IEC: P_k at rated I, 75°C)
        self.T_ref = u["T_ref_winding"]["value"]            # degC = 75
        self.alpha_Cu = u["alpha_Cu"]["value"]              # 1/K
        self.n_B = u["n_B"]["value"]                        # core loss voltage exponent
        self.k_stray = u["k_stray"]["value"]                # stray loss fraction
        self.u_k = u["u_k_pu"]["value"]                     # impedance voltage p.u.
        self.theta_oil_rated = u["theta_oil_rise_K"]["value"]  # K
        self.theta_winding_grad = u["theta_winding_grad_K"]["value"]  # K
        self.n_oil = u["n_oil"]["value"]
        self.n_wind = u["n_wind"]["value"]

    # ------------------------------------------------------------------
    # Core loss
    # ------------------------------------------------------------------

    def core_loss(self, voltage_pu=1.0):
        """
        No-load (core) loss [W].
        P_core = P_0 * (V_pu)^n_B
        """
        v = np.asarray(voltage_pu, dtype=float)
        return self.P_0 * np.abs(v) ** self.n_B

    # ------------------------------------------------------------------
    # Load loss (copper + stray)
    # ------------------------------------------------------------------

    def copper_loss(self, load_fraction, winding_temp=75.0):
        """
        Load (copper) loss [W], temperature-corrected.
        P_k(PLR, T_w) = P_k_ref * PLR^2 * (1 + alpha_Cu * (T_w - T_ref))
        """
        plr = np.asarray(load_fraction, dtype=float)
        T_w = np.asarray(winding_temp, dtype=float)
        R_ratio = 1.0 + self.alpha_Cu * (T_w - self.T_ref)
        return self.P_k_ref * plr ** 2 * R_ratio

    def stray_loss(self, load_fraction):
        """
        Additional stray losses [W] in tank walls, clamps.
        P_stray = k_stray * P_k_ref * PLR^1.5
        (PLR^1.5 accounts for eddy-current nature: between PLR and PLR^2)
        """
        plr = np.asarray(load_fraction, dtype=float)
        return self.k_stray * self.P_k_ref * np.abs(plr) ** 1.5

    # ------------------------------------------------------------------
    # Total losses and efficiency
    # ------------------------------------------------------------------

    def total_losses(self, load_fraction, voltage_pu=1.0, winding_temp=75.0):
        """Total transformer losses [W]."""
        return (self.core_loss(voltage_pu) +
                self.copper_loss(load_fraction, winding_temp) +
                self.stray_loss(load_fraction))

    def loss_breakdown(self, load_fraction, voltage_pu=1.0, winding_temp=75.0):
        """Loss components [W]."""
        p_core = self.core_loss(voltage_pu)
        p_cu = self.copper_loss(load_fraction, winding_temp)
        p_stray = self.stray_loss(load_fraction)
        return {
            "p_core_w": p_core,
            "p_copper_w": p_cu,
            "p_stray_w": p_stray,
            "p_total_w": p_core + p_cu + p_stray,
        }

    def output_power(self, load_fraction, power_factor=1.0):
        """Output active power [W] = PLR * S_rated * pf."""
        plr = np.asarray(load_fraction, dtype=float)
        pf = np.asarray(power_factor, dtype=float)
        return plr * self.S_rated * pf

    def input_power(self, load_fraction, voltage_pu=1.0, winding_temp=75.0, power_factor=1.0):
        """Input power [W] = P_out + losses."""
        return (self.output_power(load_fraction, power_factor) +
                self.total_losses(load_fraction, voltage_pu, winding_temp))

    def efficiency(self, load_fraction, voltage_pu=1.0, winding_temp=75.0, power_factor=1.0):
        """Efficiency = P_out / P_in."""
        P_out = self.output_power(load_fraction, power_factor)
        P_in = self.input_power(load_fraction, voltage_pu, winding_temp, power_factor)
        eta = np.where(P_in > 0, P_out / P_in, 0.0)
        return np.clip(eta, 0.0, 1.0)

    def optimal_load_fraction(self):
        """
        Load fraction for maximum efficiency:
        PLR_opt = sqrt(P_core / P_k_ref)
        (point where core loss = copper loss)
        """
        return np.sqrt(self.P_0 / self.P_k_ref)

    # ------------------------------------------------------------------
    # Thermal model (IEC 60076-7 simplified)
    # ------------------------------------------------------------------

    def hot_spot_temperature(self, load_fraction, ambient_temperature=20.0):
        """
        Hot-spot winding temperature [degC].
        theta_oil_rise = theta_oil_rated * (loss_ratio)^n_oil
        theta_hot_spot = T_amb + theta_oil_rise + theta_winding_grad * PLR^n_wind
        """
        plr = np.asarray(load_fraction, dtype=float)
        T_amb = np.asarray(ambient_temperature, dtype=float)

        # Loss ratio (total losses at PLR vs at rated load)
        loss_at_rated = self.P_0 + self.P_k_ref + self.stray_loss(1.0)
        loss_at_plr = self.P_0 + self.P_k_ref * plr ** 2 + self.stray_loss(plr)
        loss_ratio = np.where(loss_at_rated > 0,
                              loss_at_plr / loss_at_rated,
                              np.abs(plr) ** 2)
        loss_ratio = np.maximum(loss_ratio, 0.0)

        theta_oil = self.theta_oil_rated * loss_ratio ** self.n_oil
        theta_wind = self.theta_winding_grad * np.abs(plr) ** self.n_wind

        return T_amb + theta_oil + theta_wind
