"""EC026 -- Li-Air Battery -- F1a SOC-only Model"""


class LiAirBatteryModel:
    """
    Lithium-Air (Li-O2) battery SOC-only model.

    Reaction:  2Li + O2 -> Li2O2  (E0 = 2.96 V)

    The discharge plateau is nearly flat (~2.96 V), with a slight droop
    near full discharge due to Li2O2 pore clogging.

    OCV(SOC) = ocv_flat - ocv_droop * (1 - SOC)^3

    Very high capacity: Q_nom = 10 Ah (theoretical ~3500 Wh/kg).
    High R_int = 200 mOhm due to Li2O2 film resistance.

    Terminal voltage: V = OCV(SOC) - I * R_int
    """

    def __init__(self, params: dict):
        self.Q_nom     = float(params.get("Q_nom", 10.0))
        self.R_int     = float(params.get("R_int", 0.200))
        self.V_min     = float(params.get("V_min", 2.0))
        self.V_max     = float(params.get("V_max", 3.2))
        self.ocv_flat  = float(params.get("ocv_flat", 2.96))
        self.ocv_droop = float(params.get("ocv_droop", 0.3))

    def _ocv(self, soc: float) -> float:
        soc = max(0.0, min(1.0, soc))
        return self.ocv_flat - self.ocv_droop * (1.0 - soc) ** 3

    def evaluate(self, soc: float, I: float = 0.0, dt: float = 0.0, **kwargs) -> dict:
        """
        Inputs
        ------
        soc : float -- state of charge [0, 1]
        I   : float -- current (A), positive = discharge
        dt  : float -- time step (s)

        Outputs
        -------
        V_terminal, OCV, SOC_new, P, energy_Wh
        """
        soc = max(0.0, min(1.0, float(soc)))
        I   = float(I)
        dt  = float(dt)

        ocv        = self._ocv(soc)
        V_terminal = ocv - I * self.R_int
        V_terminal = max(self.V_min, min(self.V_max, V_terminal))
        P          = V_terminal * I

        soc_new = soc
        if dt > 0.0:
            soc_new = max(0.0, min(1.0, soc - I * dt / (self.Q_nom * 3600.0)))

        energy_Wh = soc * self.Q_nom * self._ocv(soc / 2.0)

        return {
            "V_terminal": round(V_terminal, 6),
            "OCV":        round(ocv, 6),
            "SOC_new":    round(soc_new, 6),
            "P":          round(P, 4),
            "energy_Wh":  round(energy_Wh, 4),
        }
