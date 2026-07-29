"""EC030 -- NiCd Battery -- F1a SOC-only Model"""
import math


class NiCdBatteryModel:
    """
    Nickel-Cadmium (NiCd) battery SOC-only model.

    Reaction: Cd + 2NiOOH + 2H2O <-> Cd(OH)2 + 2Ni(OH)2  (E0 = 1.2 V)

    NiCd features a very flat discharge curve in the 1.0-1.35 V range.
    Very low internal resistance (R_int = 20 mOhm), suitable for high rate.

    OCV(SOC) = ocv_flat + ocv_rise * sqrt(SOC) - ocv_droop * (1-SOC)^3

    This creates:
      - slight rise at high SOC
      - flat middle region
      - sharp droop at low SOC

    Terminal voltage: V = OCV(SOC) - I * R_int
    """

    def __init__(self, params: dict):
        self.Q_nom      = float(params.get("Q_nom", 2.0))
        self.R_int      = float(params.get("R_int", 0.020))
        self.V_min      = float(params.get("V_min", 1.0))
        self.V_max      = float(params.get("V_max", 1.55))
        self.ocv_flat   = float(params.get("ocv_flat", 1.2))
        self.ocv_rise   = float(params.get("ocv_rise", 0.08))
        self.ocv_droop  = float(params.get("ocv_droop", 0.12))

    def _ocv(self, soc: float) -> float:
        soc = max(0.0, min(1.0, soc))
        return self.ocv_flat + self.ocv_rise * math.sqrt(soc) - self.ocv_droop * (1.0 - soc) ** 3

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
