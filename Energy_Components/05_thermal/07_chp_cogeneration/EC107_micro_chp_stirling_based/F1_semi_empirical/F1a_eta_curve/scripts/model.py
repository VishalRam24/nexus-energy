"""EC107 -- Stirling Engine Micro-CHP -- F1a Efficiency Curve"""


class StirlingCHPModel:
    """
    Stirling engine micro-CHP efficiency curve model.

    f(load)  = f_a + f_b * load_fraction
    eta_e    = eta_e_rated  * f(load)    (electrical efficiency)
    eta_th   = eta_th_rated * f(load)    (thermal efficiency)

    P_e      = load * P_e_rated
    Q_in     = P_e / eta_e              (fuel input)
    Q_th     = eta_th * Q_in            (heat output)
    PER      = (P_e + Q_th) / Q_in     (primary energy ratio)

    Stirling: eta_e << eta_th (heat-led CHP)

    Parameters
    ----------
    eta_e_rated  : rated electrical efficiency (0.15)
    eta_th_rated : rated thermal efficiency (0.70)
    P_e_rated_kW : rated electrical output [kW]
    f_a, f_b     : part-load correction coefficients
    """

    def __init__(self, params: dict):
        self.eta_e_rated = float(params.get("eta_e_rated", 0.15))
        self.eta_th_rated = float(params.get("eta_th_rated", 0.70))
        self.P_e_rated = float(params.get("P_e_rated_kW", 1.0)) * 1e3  # W
        self.f_a = float(params.get("f_a", 0.2))
        self.f_b = float(params.get("f_b", 0.8))

    def evaluate(self, load_fraction: float) -> dict:
        load = max(0.0, min(1.0, float(load_fraction)))

        f_load = self.f_a + self.f_b * load
        eta_e = self.eta_e_rated * f_load
        eta_th = self.eta_th_rated * f_load

        P_e = load * self.P_e_rated
        Q_in = P_e / eta_e if eta_e > 0 and P_e > 0 else 0.0
        Q_th = eta_th * Q_in
        PER = (P_e + Q_th) / Q_in if Q_in > 0 else 0.0

        return {
            "eta_e": round(eta_e, 6),
            "eta_th": round(eta_th, 6),
            "P_e_W": round(P_e, 4),
            "P_e_kW": round(P_e / 1e3, 6),
            "Q_th_W": round(Q_th, 4),
            "Q_in_W": round(Q_in, 4),
            "PER": round(PER, 6),
            "load_fraction": round(load, 6),
        }
