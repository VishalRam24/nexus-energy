"""EC102 -- Kalina Cycle -- F1a Efficiency Curve (Carnot * 2nd-law)"""


class KalinaCycleModel:
    """
    Kalina cycle efficiency model (NH3/H2O working fluid).

    eta_Carnot = 1 - T_sink / T_source
    eta = eta_2nd * eta_Carnot

    P_out = Q_source * eta
    Q_rejected = Q_source * (1 - eta)

    Parameters
    ----------
    eta_2nd      : 2nd-law (exergetic) efficiency (0.55 typical for Kalina)
    T_sink_K     : heat sink temperature [K]
    P_rated_kW   : rated output power [kW]
    """

    def __init__(self, params: dict):
        self.eta_2nd = float(params.get("eta_2nd", 0.55))
        self.T_sink = float(params.get("T_sink_K", 298.15))
        self.P_rated = float(params.get("P_rated_kW", 500.0)) * 1e3  # W

    def evaluate(self, T_source_K: float, Q_source_W: float = None) -> dict:
        """
        Parameters
        ----------
        T_source_K : heat source temperature [K]
        Q_source_W : heat source power [W] (optional; if None, uses P_rated as P_out target)

        Returns
        -------
        dict: eta, eta_Carnot, P_out_W, P_out_kW, Q_source_W, Q_rejected_W, T_source_K
        """
        T_source = float(T_source_K)
        if T_source <= self.T_sink:
            raise ValueError("T_source must be > T_sink")

        eta_Carnot = 1.0 - self.T_sink / T_source
        eta = self.eta_2nd * eta_Carnot

        if Q_source_W is not None:
            Q_source = float(Q_source_W)
        else:
            # Back-calculate Q_source from rated power
            Q_source = self.P_rated / eta if eta > 0 else 0.0

        P_out = Q_source * eta
        Q_rejected = Q_source - P_out

        return {
            "eta": round(eta, 6),
            "eta_Carnot": round(eta_Carnot, 6),
            "P_out_W": round(P_out, 2),
            "P_out_kW": round(P_out / 1e3, 6),
            "Q_source_W": round(Q_source, 2),
            "Q_rejected_W": round(Q_rejected, 2),
            "T_source_K": round(T_source, 4),
        }
