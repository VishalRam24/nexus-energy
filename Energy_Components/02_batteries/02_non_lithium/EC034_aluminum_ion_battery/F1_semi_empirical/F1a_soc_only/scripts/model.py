"""EC034 -- Aluminum-Ion Battery -- F1a SOC-only Model"""


class AluminumIonBatteryModel:
    """
    Aluminum-Ion battery SOC-only model.

    Anode:  Al -> Al3+ + 3e-
    Cathode: graphite intercalation with AlCl4-

    Operating voltage ~2.0 V with graphite cathode. Lower capacity than
    Li-ion (Q_nom=1.0 Ah) but excellent rate capability and safety.

    OCV(SOC) = polynomial fit  (V_min=1.5, V_max=2.3)
    Terminal voltage: V = OCV(SOC) - I * R_int
    """

    def __init__(self, params: dict):
        self.Q_nom      = float(params.get("Q_nom", 1.0))
        self.R_int      = float(params.get("R_int", 0.080))
        self.V_min      = float(params.get("V_min", 1.5))
        self.V_max      = float(params.get("V_max", 2.3))
        self.ocv_coeffs = list(params.get("ocv_coeffs", [2.0, -0.2, 0.1, -0.05]))

    def _ocv(self, soc: float) -> float:
        soc   = max(0.0, min(1.0, soc))
        value = 0.0
        s_pow = 1.0
        for c in self.ocv_coeffs:
            value += c * s_pow
            s_pow *= soc
        return value

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
