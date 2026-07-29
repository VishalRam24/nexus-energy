"""
EC221 — MHD Generator — F1a Ideal Faraday MHD Model

Faraday-type MHD generator:

    EMF = u * B * h          [V] — induced EMF per unit length (h = channel height)
    J   = sigma * (EMF - V_load) / h   [A/m^2] — current density (simplified 1D)
    K   = V_load / EMF       — load factor (0=short, 1=open, 0.5=max power)

    Power density:
    p = sigma * u^2 * B^2 * K * (1 - K)   [W/m^3]

    Total electrical power:
    P = p * V_channel   where V_channel = L * w * h [m^3]
    P = sigma * u^2 * B^2 * K * (1-K) * L * w * h

    Heat input (kinetic energy of plasma per second):
    Q_in = 0.5 * rho * u^3 * w * h   [W]
    where rho ~ 0.5 kg/m^3 for typical seeded plasma

    MHD efficiency (ideal Faraday):
    eta_mhd = K * (1 - K)   — fraction of ideal maximum
    eta_mhd_max = 0.25 at K=0.5

    Plant efficiency: eta_plant = eta_mhd * additional_factor

References:
    Rosa, R.J. (1987). Magnetohydrodynamic Energy Conversion. McGraw-Hill.
    Messerle, H.K. (1995). Magnetohydrodynamic Electrical Power Generation. Wiley.
"""

import numpy as np


class MHDF1a:
    """MHD Faraday generator — ideal electrical power and efficiency model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.sigma = u["sigma_plasma"]["value"]   # S/m
        self.u0 = u["u_plasma"]["value"]          # m/s
        self.B0 = u["B_field"]["value"]           # T
        self.K0 = u["K_load"]["value"]            # -
        self.L = u["channel_length"]["value"]     # m
        self.w = u["channel_width"]["value"]      # m
        self.h = u["channel_height"]["value"]     # m
        self.V_ch = self.L * self.w * self.h     # m^3

        self.rho_plasma = 0.5  # kg/m^3 — typical seeded combustion gas

    def compute(self, sigma, u, B, K):
        """
        Parameters
        ----------
        sigma : float or array — plasma conductivity [S/m]
        u     : float or array — plasma velocity [m/s]
        B     : float or array — magnetic field [T]
        K     : float or array — load factor [-]

        Returns
        -------
        dict: EMF_V, J_Am2, power_density_Wm3, power_w, heat_input_w, eta_mhd, eta_plant
        """
        sigma = np.asarray(sigma, dtype=float)
        u_v = np.asarray(u, dtype=float)
        B_v = np.asarray(B, dtype=float)
        K_v = np.clip(np.asarray(K, dtype=float), 0.0, 1.0)

        # Induced EMF over channel height
        EMF = u_v * B_v * self.h       # V

        # Power density [W/m^3]
        p_density = sigma * u_v**2 * B_v**2 * K_v * (1.0 - K_v)

        # Total power
        P = p_density * self.V_ch

        # Enthalpy flux (thermodynamic heat input for the MHD channel).
        # For a Faraday MHD generator, the input power is the stagnation
        # enthalpy flux: Q_in = rho * u * h_stag * A_cross
        # For a thermally-dominated plasma (combustion gas at ~2500K),
        # h_stag ~ cp * T_stag + 0.5*u^2. With cp~1200 J/(kg*K), T~2500K:
        # h_stag ~ 3e6 J/kg >> 0.5*u^2 ~3.2e5 J/kg for u=800 m/s.
        # Simplification: h_stag ~ cp_eff * T_plasma with cp_eff*T ~ 4*u^2
        # (order-of-magnitude, seeded combustion gas at M~0.5)
        Q_in = self.rho_plasma * u_v * 4.0 * u_v**2 * self.w * self.h
        Q_in = np.maximum(Q_in, 1e-12)

        # MHD conversion efficiency
        eta_mhd = K_v * (1.0 - K_v)     # max 0.25 at K=0.5

        # Plant efficiency (MHD + bottoming cycle from params)
        eta_plant = P / Q_in
        eta_plant = np.clip(eta_plant, 0.0, 1.0)

        # Current density
        J = sigma * u_v * B_v * (1.0 - K_v)  # A/m^2

        return {
            "EMF_V": EMF,
            "J_Am2": J,
            "power_density_Wm3": p_density,
            "power_w": P,
            "heat_input_w": Q_in,
            "eta_mhd": eta_mhd,
            "eta_plant": eta_plant,
        }
