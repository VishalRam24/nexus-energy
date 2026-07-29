"""EC024 -- Silicon-Anode Li-ion Battery -- F1a SOC-only Model"""


class SiAnodeBatteryModel:
    """
    Silicon-anode lithium-ion battery SOC-only model.

    Silicon anodes offer ~10x higher theoretical capacity than graphite
    (3590 mAh/g vs 372 mAh/g), enabling Q_nom=4.0 Ah per cell.

    Terminal voltage: V = OCV(SOC) - I * R_int
        I > 0 = discharge
        I < 0 = charge

    OCV(SOC) = polynomial fit calibrated to 2.8-4.2 V range.

    SOC update:  SOC_new = SOC - I * dt / (Q_nom * 3600)
    """

    def __init__(self, params: dict):
        self.Q_nom      = float(params.get("Q_nom", 4.0))
        self.R_int      = float(params.get("R_int", 0.050))
        self.V_min      = float(params.get("V_min", 2.8))
        self.V_max      = float(params.get("V_max", 4.2))
        self.ocv_coeffs = list(params.get("ocv_coeffs", [4.2, -1.0, 0.6, -0.3, 0.15]))

    def _ocv(self, soc: float) -> float:
        soc   = max(0.0, min(1.0, soc))
        value = 0.0
        s_pow = 1.0
        for c in self.ocv_coeffs:
            value += c * s_pow
            s_pow *= soc
        return value

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
