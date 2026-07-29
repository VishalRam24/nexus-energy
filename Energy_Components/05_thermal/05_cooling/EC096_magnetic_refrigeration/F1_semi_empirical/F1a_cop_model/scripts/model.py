"""EC096 -- Magnetic Refrigeration -- F1a COP Model"""


class MagneticRefrigerationModel:
    """
    Magnetic refrigeration COP model.

    COP_Carnot = T_cold / (T_hot - T_cold)
    COP = eta_2nd * COP_Carnot
    Q_cool = COP * W_input

    The magnetocaloric effect spans N_stages * delta_T_MCE total temperature lift.

    Parameters
    ----------
    eta_2nd     : 2nd-law (exergetic) efficiency
    delta_T_MCE : magnetocaloric temperature span per stage [K]
    N_stages    : number of AMR stages
    T_cold_K    : cold side temperature [K] (default)
    T_hot_K     : hot side temperature [K] (default)
    """

    def __init__(self, params: dict):
        self.eta_2nd = float(params.get("eta_2nd", 0.4))
        self.delta_T_MCE = float(params.get("delta_T_MCE", 3.0))
        self.N_stages = int(params.get("N_stages", 6))
        self.T_cold_default = float(params.get("T_cold_K", 273.15))
        self.T_hot_default = float(params.get("T_hot_K", 298.15))

    def evaluate(self, W_input_W: float, T_cold_K: float = None, T_hot_K: float = None) -> dict:
        """
        Parameters
        ----------
        W_input_W : electrical work input [W]
        T_cold_K  : cold reservoir temperature [K]
        T_hot_K   : hot reservoir temperature [K]

        Returns
        -------
        dict: COP, COP_Carnot, Q_cool_W, Q_hot_W, T_span_K, W_input_W
        """
        W = max(0.0, float(W_input_W))
        T_cold = float(T_cold_K) if T_cold_K is not None else self.T_cold_default
        T_hot = float(T_hot_K) if T_hot_K is not None else self.T_hot_default

        if T_cold <= 0 or T_hot <= T_cold:
            raise ValueError("Require 0 < T_cold < T_hot")

        delta_T = T_hot - T_cold
        COP_Carnot = T_cold / delta_T
        COP = self.eta_2nd * COP_Carnot

        Q_cool = COP * W
        Q_hot = Q_cool + W
        T_span = self.N_stages * self.delta_T_MCE

        return {
            "COP": round(COP, 6),
            "COP_Carnot": round(COP_Carnot, 6),
            "Q_cool_W": round(Q_cool, 4),
            "Q_hot_W": round(Q_hot, 4),
            "T_span_K": round(T_span, 2),
            "W_input_W": round(W, 4),
        }
