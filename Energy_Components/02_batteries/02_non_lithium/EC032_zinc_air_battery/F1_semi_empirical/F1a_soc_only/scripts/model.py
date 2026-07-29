"""EC032 -- Zinc-Air Battery -- F1a SOC-only Model"""


class ZincAirBatteryModel:
    """
    Zinc-Air battery SOC-only model.

    Reaction: Zn + 1/2 O2 -> ZnO  (E0 = 1.65 V)

    Discharge plateau is nearly flat (~1.65 V). OCV drops slightly near
    end of discharge as Zn anode is consumed.

    OCV(SOC) = ocv_flat - ocv_droop * (1 - SOC)^2

    Terminal voltage: V = OCV(SOC) - I * R_int
    """

    def __init__(self, params: dict):
        self.Q_nom     = float(params.get("Q_nom", 3.0))
        self.R_int     = float(params.get("R_int", 0.050))
        self.V_min     = float(params.get("V_min", 1.0))
        self.V_max     = float(params.get("V_max", 1.9))
        self.ocv_flat  = float(params.get("ocv_flat", 1.65))
        self.ocv_droop = float(params.get("ocv_droop", 0.1))

    def _ocv(self, soc: float) -> float:
        soc = max(0.0, min(1.0, soc))
        return self.ocv_flat - self.ocv_droop * (1.0 - soc) ** 2

    def evaluate(self, soc: float, I: float = 0.0, dt: float = 0.0, **kwargs) -> dict:
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
