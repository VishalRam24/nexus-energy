"""EC084 -- Aquifer Thermal Energy Storage (ATES) -- F1a Fully Mixed"""


class ATESModel:
    """
    ATES fully-mixed model.

    E_stored = V_aquifer * rho * Cp * delta_T * eta_recovery  [J]

    For charging/discharging with flow:
        Q_thermal = m_dot * Cp * (T_in - T_ground)  [W]

    Parameters
    ----------
    V_aquifer    : m^3
    rho          : kg/m^3
    Cp           : J/(kg*K)
    eta_recovery : dimensionless (thermal recovery factor)
    T_ground     : degC (undisturbed aquifer temperature)
    T_max        : degC (max storage temperature)
    T_min        : degC (min storage temperature / cold well)
    """

    def __init__(self, params: dict):
        self.V = float(params.get("V_aquifer", 50000.0))
        self.rho = float(params.get("rho", 1000.0))
        self.Cp = float(params.get("Cp", 4186.0))
        self.eta = float(params.get("eta_recovery", 0.70))
        self.T_ground = float(params.get("T_ground", 12.0))
        self.T_max = float(params.get("T_max", 25.0))
        self.T_min = float(params.get("T_min", 5.0))

    def evaluate(self, T_storage: float, m_dot: float = 0.0, T_in: float = None) -> dict:
        """
        Parameters
        ----------
        T_storage : float -- current mean aquifer temperature [degC]
        m_dot     : float -- mass flow rate [kg/s], positive = extraction, negative = injection
        T_in      : float -- inlet temperature [degC] (for injection), defaults to T_max

        Returns
        -------
        dict with E_stored_J, E_stored_kWh, Q_thermal_W, delta_T, SOC, T_storage_new
        """
        T_storage = float(T_storage)
        m_dot = float(m_dot)
        if T_in is None:
            T_in = self.T_max if m_dot < 0 else self.T_ground
        T_in = float(T_in)

        delta_T = T_storage - self.T_ground
        E_stored_J = self.V * self.rho * self.Cp * delta_T * self.eta
        E_stored_kWh = E_stored_J / 3.6e6

        # Thermal power at current flow conditions
        Q_thermal_W = abs(m_dot) * self.Cp * abs(T_in - self.T_ground)

        # SOC: fraction of max storable energy
        delta_T_max = self.T_max - self.T_ground
        E_max = self.V * self.rho * self.Cp * delta_T_max * self.eta
        soc = max(0.0, min(1.0, E_stored_J / E_max)) if E_max > 0 else 0.0

        return {
            "E_stored_J": round(E_stored_J, 0),
            "E_stored_kWh": round(E_stored_kWh, 2),
            "Q_thermal_W": round(Q_thermal_W, 2),
            "delta_T": round(delta_T, 4),
            "SOC": round(soc, 6),
            "T_storage": round(T_storage, 4),
        }
