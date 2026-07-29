"""EC025 -- Li-S Battery -- F1a SOC-only Model (two-plateau OCV)"""


class LiSBatteryModel:
    """
    Lithium-Sulfur battery SOC-only model.

    Unique two-plateau discharge curve:
      High plateau (~2.3 V): SOC in (soc_transition, 1]
          S8 + 16Li+ + 16e- -> 8Li2S   (upper plateau)
      Low plateau (~2.1 V):  SOC in [0, soc_transition]
          Polysulfide reduction to Li2S (lower plateau)

    Piecewise linear OCV:
        SOC > soc_transition : OCV = V_max linearly to V_high_plateau
        SOC <= soc_transition: OCV = V_high_plateau linearly to V_low_plateau

    Terminal voltage: V = OCV(SOC) - I * R_int
    """

    def __init__(self, params: dict):
        self.Q_nom          = float(params.get("Q_nom", 5.0))
        self.R_int          = float(params.get("R_int", 0.100))
        self.V_min          = float(params.get("V_min", 1.7))
        self.V_max          = float(params.get("V_max", 2.45))
        self.V_high_plateau = float(params.get("V_high_plateau", 2.3))
        self.V_low_plateau  = float(params.get("V_low_plateau", 2.1))
        self.soc_transition = float(params.get("soc_transition", 0.25))

    def _ocv(self, soc: float) -> float:
        """Piecewise linear two-plateau OCV."""
        soc = max(0.0, min(1.0, soc))
        st  = self.soc_transition
        if soc > st:
            # Upper region: linear from V_high_plateau (at soc=st) to V_max (at soc=1)
            frac = (soc - st) / (1.0 - st)
            return self.V_high_plateau + frac * (self.V_max - self.V_high_plateau)
        else:
            if st == 0.0:
                return self.V_low_plateau
            # Lower region: linear from V_low_plateau (at soc=0) to V_high_plateau (at soc=st)
            frac = soc / st
            return self.V_low_plateau + frac * (self.V_high_plateau - self.V_low_plateau)

    def evaluate(self, soc: float, I: float = 0.0, dt: float = 0.0, **kwargs) -> dict:
        """
        Inputs
        ------
        soc : float -- state of charge [0, 1]
        I   : float -- current (A), positive = discharge
        dt  : float -- time step (s)

        Outputs
        -------
        V_terminal, OCV, SOC_new, P, energy_Wh, plateau (str: 'high' or 'low')
        """
        soc = max(0.0, min(1.0, float(soc)))
        I   = float(I)
        dt  = float(dt)

        ocv        = self._ocv(soc)
        V_terminal = ocv - I * self.R_int
        V_terminal = max(self.V_min, min(self.V_max, V_terminal))
        P          = V_terminal * I
        plateau    = "high" if soc > self.soc_transition else "low"

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
            "plateau":    plateau,
        }
