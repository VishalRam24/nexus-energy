"""EC106 -- SOFC-Based Fuel Cell CHP -- F1a Efficiency Curve"""


class SOFCCHPModel:
    """
    SOFC CHP efficiency curve model.

    f_e(load)  = f_e_a  + f_e_b  * load
    f_th(load) = f_th_a + f_th_b * load
    eta_e  = eta_e_rated  * f_e(load)    (electrical efficiency)
    eta_th = eta_th_rated * f_th(load)   (thermal efficiency)

    P_e  = load * P_e_rated              (electrical output)
    Q_in = P_e / eta_e                   (fuel input, HHV basis)
    Q_th = eta_th * Q_in                 (heat output)
    PER  = (P_e + Q_th) / Q_in          (primary energy ratio)

    Parameters
    ----------
    eta_e_rated  : rated electrical efficiency
    eta_th_rated : rated thermal efficiency
    P_e_rated_kW : rated electrical output [kW]
    f_e_a, f_e_b : part-load correction for electrical eta
    f_th_a, f_th_b: part-load correction for thermal eta
    """

    def __init__(self, params: dict):
        self.eta_e_rated = float(params.get("eta_e_rated", 0.55))
        self.eta_th_rated = float(params.get("eta_th_rated", 0.30))
        self.P_e_rated = float(params.get("P_e_rated_kW", 5.0)) * 1e3  # W
        self.f_e_a = float(params.get("f_e_a", 0.3))
        self.f_e_b = float(params.get("f_e_b", 0.7))
        self.f_th_a = float(params.get("f_th_a", 0.2))
        self.f_th_b = float(params.get("f_th_b", 0.8))

    def evaluate(self, load_fraction: float) -> dict:
        """
        Parameters
        ----------
        load_fraction : float in [0, 1]

        Returns
        -------
        dict: eta_e, eta_th, P_e_W, P_e_kW, Q_th_W, Q_in_W, PER, load_fraction
        """
        load = max(0.0, min(1.0, float(load_fraction)))

        f_e = self.f_e_a + self.f_e_b * load
        f_th = self.f_th_a + self.f_th_b * load
        eta_e = self.eta_e_rated * f_e
        eta_th = self.eta_th_rated * f_th

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
