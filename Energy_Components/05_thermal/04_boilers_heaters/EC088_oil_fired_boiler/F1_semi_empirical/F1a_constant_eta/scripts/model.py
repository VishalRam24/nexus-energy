"""EC088 -- Oil-Fired Boiler -- F1a Constant Efficiency"""


class OilBoilerModel:
    """
    Oil-fired boiler constant efficiency model.

    Q_out = eta * m_fuel * LHV         [W if m_fuel in kg/s]
    m_fuel = Q_out / (eta * LHV)       [kg/s]
    load_fraction = Q_out / P_rated

    Parameters
    ----------
    eta          : thermal efficiency (dimensionless)
    LHV_MJ_per_kg: lower heating value [MJ/kg]
    P_rated_kW   : rated thermal output [kW]
    """

    def __init__(self, params: dict):
        self.eta = float(params.get("eta", 0.87))
        self.LHV = float(params.get("LHV_MJ_per_kg", 42.6)) * 1e6  # J/kg
        self.P_rated = float(params.get("P_rated_kW", 500.0)) * 1e3  # W

    def evaluate(self, m_fuel_kg_s: float = None, Q_demand_W: float = None) -> dict:
        """
        Call with either m_fuel_kg_s OR Q_demand_W (not both).

        Parameters
        ----------
        m_fuel_kg_s : fuel mass flow rate [kg/s]
        Q_demand_W  : heat demand [W] (model back-calculates fuel flow)

        Returns
        -------
        dict: Q_out_W, m_fuel_kg_s, Q_in_W, losses_W, load_fraction, eta
        """
        if m_fuel_kg_s is not None:
            m = float(m_fuel_kg_s)
            Q_in = m * self.LHV
            Q_out = self.eta * Q_in
        elif Q_demand_W is not None:
            Q_out = float(Q_demand_W)
            Q_in = Q_out / self.eta
            m = Q_in / self.LHV
        else:
            raise ValueError("Provide m_fuel_kg_s or Q_demand_W")

        Q_out = min(Q_out, self.P_rated)
        losses = Q_in - Q_out
        load = Q_out / self.P_rated

        return {
            "Q_out_W": round(Q_out, 2),
            "Q_out_kW": round(Q_out / 1e3, 4),
            "m_fuel_kg_s": round(m, 8),
            "Q_in_W": round(Q_in, 2),
            "losses_W": round(losses, 2),
            "load_fraction": round(max(0.0, min(1.0, load)), 6),
            "eta": round(self.eta, 6),
        }
