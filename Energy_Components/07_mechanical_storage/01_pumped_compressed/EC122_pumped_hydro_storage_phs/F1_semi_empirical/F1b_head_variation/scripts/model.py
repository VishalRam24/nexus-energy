"""
EC122 — Pumped Hydro Storage — F1b Head Variation Model

Extends F1a by adding:
1. Variable head with SOC: H(SOC) = H_min + SOC*(H_max - H_min)
2. Penstock friction losses: h_f = f*L*v^2 / (2*D*g) (Darcy-Weisbach)
   where v = Q / A_penstock

Generation (discharge):
    P_gen = eta_turbine * eta_generator * rho * g * Q * (H - h_f) / 1000   [kW]

Pumping (charge):
    P_pump = rho * g * Q * (H + h_f) / (eta_pump * eta_motor * 1000)       [kW]

Effective efficiency includes friction effects:
    eta_gen_eff = eta_turbine * eta_generator * (H - h_f) / H
    eta_pump_eff = eta_pump * eta_motor * H / (H + h_f)
    RTE = eta_gen_eff * eta_pump_eff

References:
    Rehman et al. (2015). RSER, 44, 586-598.
    Mosonyi, E. (1991). Water Power Development. Akademiai Kiado.
    Munson et al. (2013). Fluid Mechanics, 7th ed. Wiley. (Darcy-Weisbach)
"""

import numpy as np


class PHSF1b:
    """Pumped Hydro Storage — head-variation model with penstock friction."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.H_max = u["H_max"]["value"]              # m
        self.H_min = u["H_min"]["value"]              # m
        self.L = u["penstock_length"]["value"]        # m
        self.D = u["penstock_diameter"]["value"]      # m
        self.f = u["friction_factor"]["value"]        # dimensionless
        self.eta_turbine = u["eta_turbine"]["value"]
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_gen = u["eta_generator"]["value"]
        self.eta_motor = u["eta_motor"]["value"]
        self.rho = u["rho"]["value"]                  # kg/m3
        self.g = u["g"]["value"]                      # m/s2
        self.Q_design = u["Q_design"]["value"]        # m3/s
        self.V_res = u["reservoir_volume"]["value"]   # m3
        self.A_pen = np.pi * (self.D / 2.0) ** 2     # penstock cross-section area

    def effective_head(self, soc):
        """Head as a function of SOC: H(SOC) = H_min + SOC*(H_max-H_min)."""
        soc = np.asarray(soc, dtype=float)
        return self.H_min + soc * (self.H_max - self.H_min)

    def friction_loss(self, flow_rate_m3s):
        """
        Penstock friction head loss [m] via Darcy-Weisbach:
            h_f = f * L * v^2 / (2 * D * g)
        where v = Q / A.
        """
        Q = np.asarray(flow_rate_m3s, dtype=float)
        v = np.abs(Q) / self.A_pen
        return self.f * self.L * v ** 2 / (2.0 * self.D * self.g)

    def generation_power(self, soc, flow_rate_m3s):
        """
        Turbine-generator output power [kW] during discharge.
        P_gen = eta_t * eta_g * rho * g * Q * (H - h_f) / 1000
        """
        Q = np.asarray(flow_rate_m3s, dtype=float)
        H = self.effective_head(soc)
        h_f = self.friction_loss(Q)
        H_net = np.maximum(H - h_f, 0.0)
        return self.eta_turbine * self.eta_gen * self.rho * self.g * Q * H_net / 1000.0

    def pumping_power(self, soc, flow_rate_m3s):
        """
        Motor-pump input power [kW] required during charging.
        P_pump = rho * g * Q * (H + h_f) / (eta_pump * eta_motor * 1000)
        """
        Q = np.asarray(flow_rate_m3s, dtype=float)
        H = self.effective_head(soc)
        h_f = self.friction_loss(Q)
        H_total = H + h_f
        return self.rho * self.g * Q * H_total / (self.eta_pump * self.eta_motor * 1000.0)

    def power(self, soc, flow_rate_m3s, mode="discharge"):
        """
        Power [kW] for given mode.
        - discharge: positive power output
        - charge: positive power consumed (input)
        """
        soc = np.asarray(soc, dtype=float)
        Q = np.asarray(flow_rate_m3s, dtype=float)
        if isinstance(mode, str):
            if mode == "discharge":
                return self.generation_power(soc, Q)
            else:
                return self.pumping_power(soc, Q)
        else:
            # Array of modes
            mode = np.asarray(mode)
            P = np.zeros_like(soc, dtype=float)
            dis_mask = mode == "discharge"
            chg_mask = ~dis_mask
            if np.any(dis_mask):
                P[dis_mask] = self.generation_power(soc[dis_mask], Q[dis_mask])
            if np.any(chg_mask):
                P[chg_mask] = self.pumping_power(soc[chg_mask], Q[chg_mask])
            return P

    def efficiency(self, soc, flow_rate_m3s, mode="discharge"):
        """
        Effective efficiency including friction.
        Discharge: eta = eta_t * eta_g * (H-h_f)/H
        Charge:    eta = eta_p * eta_m * H/(H+h_f)
        """
        Q = np.asarray(flow_rate_m3s, dtype=float)
        H = self.effective_head(soc)
        h_f = self.friction_loss(Q)
        if isinstance(mode, str) and mode == "discharge":
            H_net = np.maximum(H - h_f, 0.0)
            return self.eta_turbine * self.eta_gen * H_net / np.maximum(H, 1e-6)
        else:
            return self.eta_pump * self.eta_motor * H / np.maximum(H + h_f, 1e-6)

    def round_trip_efficiency(self, soc, flow_rate_m3s):
        """Round-trip efficiency at given SOC and flow rate."""
        eta_dis = self.efficiency(soc, flow_rate_m3s, "discharge")
        eta_chg = self.efficiency(soc, flow_rate_m3s, "charge")
        return eta_dis * eta_chg

    def energy_capacity(self, soc=1.0):
        """Energy capacity [MWh] at given head."""
        H = self.effective_head(soc)
        return self.rho * self.g * self.V_res * H / 3.6e9  # GWh -> multiply by 1000 for MWh
