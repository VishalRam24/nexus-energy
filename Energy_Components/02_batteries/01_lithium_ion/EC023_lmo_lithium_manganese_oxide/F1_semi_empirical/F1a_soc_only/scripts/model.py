"""EC023 -- LMO Battery -- F1a SOC-only Model"""


class LMOBatteryModel:
    """
    Lithium Manganese Oxide (LMO) battery, SOC-only model.

    Terminal voltage:  V = OCV(SOC) + I * R_int
        I > 0 = discharge  ->  V drops
        I < 0 = charge     ->  V rises

    OCV(SOC) is a polynomial:
        OCV = sum( ocv_coeffs[i] * SOC^i )

    State update (Coulomb counting):
        SOC_new = SOC - (I * dt) / (Q_nom * 3600)

    Operating range: 3.0 V -- 4.2 V, Q_nom = 2.5 Ah
    """

    def __init__(self, params: dict):
        self.Q_nom      = float(params.get("Q_nom", 2.5))
        self.R_int      = float(params.get("R_int", 0.030))
        self.V_min      = float(params.get("V_min", 3.0))
        self.V_max      = float(params.get("V_max", 4.2))
        self.ocv_coeffs = list(params.get("ocv_coeffs", [4.2, -0.8, 0.4, -0.2, 0.1]))

    def _ocv(self, soc: float) -> float:
        """Polynomial OCV from SOC in [0, 1]."""
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
        dt  : float -- time step (s) for SOC update (0 = no update)

        Outputs
        -------
        V_terminal : terminal voltage (V)
        OCV        : open-circuit voltage (V)
        SOC_new    : updated SOC after dt
        P          : power (W), positive = discharge
        energy_Wh  : available energy (Wh) at current SOC
        """
        soc   = max(0.0, min(1.0, float(soc)))
        I     = float(I)
        dt    = float(dt)

        ocv        = self._ocv(soc)
        V_terminal = ocv - I * self.R_int  # discharge positive -> V drops
        V_terminal = max(self.V_min, min(self.V_max, V_terminal))
        P          = V_terminal * I

        soc_new = soc
        if dt > 0.0:
            delta_soc = (I * dt) / (self.Q_nom * 3600.0)
            soc_new   = max(0.0, min(1.0, soc - delta_soc))

        energy_Wh = soc * self.Q_nom * self._ocv(soc / 2)  # approximate midpoint

        return {
            "V_terminal": round(V_terminal, 6),
            "OCV":        round(ocv, 6),
            "SOC_new":    round(soc_new, 6),
            "P":          round(P, 4),
            "energy_Wh":  round(energy_Wh, 4),
        }
