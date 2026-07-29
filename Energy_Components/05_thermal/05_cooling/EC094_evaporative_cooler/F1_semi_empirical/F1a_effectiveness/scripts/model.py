"""EC094 -- Evaporative Cooler -- F1a Effectiveness Model"""


class EvaporativeCoolerModel:
    """
    Evaporative cooler effectiveness model.

    T_out = T_db - epsilon * (T_db - T_wb)     [degC]
    Q_cool = m_dot_air * Cp_air * (T_db - T_out)  [W]
    COP = Q_cool / P_fan

    Parameters
    ----------
    epsilon      : effectiveness (0-1)
    P_fan_W      : fan electrical power [W]
    Cp_air_J_kgK : air specific heat [J/(kg*K)]
    rho_air_kg_m3: air density [kg/m^3]
    """

    def __init__(self, params: dict):
        self.epsilon = float(params.get("epsilon", 0.85))
        self.P_fan = float(params.get("P_fan_W", 200.0))
        self.Cp = float(params.get("Cp_air_J_kgK", 1005.0))
        self.rho = float(params.get("rho_air_kg_m3", 1.2))

    def evaluate(self, T_db: float, T_wb: float, m_dot_air: float = 1.0) -> dict:
        """
        Parameters
        ----------
        T_db      : dry-bulb temperature [degC]
        T_wb      : wet-bulb temperature [degC]
        m_dot_air : air mass flow rate [kg/s] (default 1.0)

        Returns
        -------
        dict: T_out, Q_cool_W, COP, delta_T, P_fan_W
        """
        T_db = float(T_db)
        T_wb = float(T_wb)
        m_dot = float(m_dot_air)

        if T_wb > T_db:
            raise ValueError("T_wb must be <= T_db")

        T_out = T_db - self.epsilon * (T_db - T_wb)
        delta_T = T_db - T_out
        Q_cool = m_dot * self.Cp * delta_T
        COP = Q_cool / self.P_fan if self.P_fan > 0 else float("inf")

        return {
            "T_out": round(T_out, 4),
            "Q_cool_W": round(Q_cool, 4),
            "COP": round(COP, 4),
            "delta_T": round(delta_T, 4),
            "P_fan_W": round(self.P_fan, 4),
        }
