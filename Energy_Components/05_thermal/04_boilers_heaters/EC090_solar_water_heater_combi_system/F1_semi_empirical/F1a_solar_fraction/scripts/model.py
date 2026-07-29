"""EC090 -- Solar Water Heater Combi System -- F1a Solar Fraction"""


class SolarCombiModel:
    """
    Solar combi system solar fraction model.

    Q_solar  = eta_collector * G * A_collector    [W]
    Q_solar  = min(Q_solar, Q_demand)             (can't exceed demand)
    Q_aux    = (Q_demand - Q_solar) / eta_boiler  [W fuel input to auxiliary boiler]
    f_solar  = Q_solar / Q_demand                  (solar fraction)

    Parameters
    ----------
    eta_collector  : collector thermal efficiency
    A_collector_m2 : collector aperture area [m^2]
    eta_boiler     : auxiliary boiler efficiency
    Q_demand_W     : baseline heat demand [W]
    """

    def __init__(self, params: dict):
        self.eta_coll = float(params.get("eta_collector", 0.50))
        self.A = float(params.get("A_collector_m2", 6.0))
        self.eta_boiler = float(params.get("eta_boiler", 0.90))
        self.Q_demand_default = float(params.get("Q_demand_W", 10000.0))

    def evaluate(self, G_W_m2: float, Q_demand_W: float = None) -> dict:
        """
        Parameters
        ----------
        G_W_m2    : solar irradiance on collector plane [W/m^2]
        Q_demand_W: heat demand [W] (defaults to parameter value)

        Returns
        -------
        dict: Q_solar_W, Q_aux_input_W, Q_aux_delivered_W, f_solar, Q_demand_W, eta_system
        """
        G = max(0.0, float(G_W_m2))
        Q_demand = float(Q_demand_W) if Q_demand_W is not None else self.Q_demand_default
        Q_demand = max(0.0, Q_demand)

        Q_solar = self.eta_coll * G * self.A
        Q_solar = min(Q_solar, Q_demand)  # clamp to demand

        Q_shortfall = Q_demand - Q_solar
        Q_aux_input = Q_shortfall / self.eta_boiler if Q_shortfall > 0 else 0.0
        Q_aux_delivered = Q_shortfall

        f_solar = Q_solar / Q_demand if Q_demand > 0 else 0.0

        # System efficiency: useful heat / (solar irradiance input + fuel input)
        Q_solar_input = G * self.A
        total_input = Q_solar_input + Q_aux_input
        eta_system = Q_demand / total_input if total_input > 0 else 0.0

        return {
            "Q_solar_W": round(Q_solar, 4),
            "Q_aux_input_W": round(Q_aux_input, 4),
            "Q_aux_delivered_W": round(Q_aux_delivered, 4),
            "f_solar": round(max(0.0, min(1.0, f_solar)), 6),
            "Q_demand_W": round(Q_demand, 4),
            "eta_system": round(max(0.0, min(1.0, eta_system)), 6),
        }
