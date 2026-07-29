"""EC100 -- Brayton Cycle Gas Turbine -- F1a Efficiency Curve"""


class BraytonCycleModel:
    """
    Brayton cycle gas turbine efficiency curve model.

    f(load) = a + b * load_fraction          (part-load correction factor)
    eta(load) = eta_rated * f(load)          (thermal efficiency)
    P_out = load_fraction * P_rated          (output power)
    Q_in = P_out / eta                       (fuel heat input)
    Q_exhaust = Q_in - P_out                 (exhaust heat)

    Parameters
    ----------
    eta_rated  : rated full-load thermal efficiency
    P_rated_MW : rated output power [MW]
    f_load_a   : part-load intercept (0.2 default)
    f_load_b   : part-load slope (0.8 default -> f(1.0)=1.0)
    """

    def __init__(self, params: dict):
        self.eta_rated = float(params.get("eta_rated", 0.38))
        self.P_rated = float(params.get("P_rated_MW", 50.0)) * 1e6  # W
        self.f_a = float(params.get("f_load_a", 0.2))
        self.f_b = float(params.get("f_load_b", 0.8))

    def evaluate(self, load_fraction: float) -> dict:
        """
        Parameters
        ----------
        load_fraction : float in [0, 1]

        Returns
        -------
        dict: eta, P_out_W, P_out_MW, Q_in_W, Q_exhaust_W, load_fraction, f_load
        """
        load = max(0.0, min(1.0, float(load_fraction)))
        f_load = self.f_a + self.f_b * load
        eta = self.eta_rated * f_load

        P_out = load * self.P_rated
        Q_in = P_out / eta if eta > 0 and P_out > 0 else 0.0
        Q_exhaust = Q_in - P_out

        return {
            "eta": round(eta, 6),
            "P_out_W": round(P_out, 2),
            "P_out_MW": round(P_out / 1e6, 6),
            "Q_in_W": round(Q_in, 2),
            "Q_exhaust_W": round(Q_exhaust, 2),
            "load_fraction": round(load, 6),
            "f_load": round(f_load, 6),
        }
