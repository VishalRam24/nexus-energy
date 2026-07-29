"""EC042 -- Pseudocapacitor -- F1a Capacitor Model"""


class PseudocapacitorModel:
    """
    Pseudocapacitor simple capacitor model.

    V(t) = V0 - I*t/C - I*R_esr   (constant-current discharge)
    E = 0.5 * C * V^2              (stored energy in J)
    P = V * I                       (instantaneous power)

    SOC = V^2 / V_max^2            (energy-based SOC)

    Parameters
    ----------
    C       : capacitance [F]
    R_esr   : equivalent series resistance [Ohm]
    V_max   : maximum cell voltage [V]
    V_min   : minimum cell voltage [V] (cutoff)
    """

    def __init__(self, params: dict):
        self.C = float(params.get("C", 500.0))
        self.R_esr = float(params.get("R_esr", 0.005))
        self.V_max = float(params.get("V_max", 2.7))
        self.V_min = float(params.get("V_min", 0.0))

    def evaluate(self, V0: float, I: float = 0.0, dt: float = 0.0) -> dict:
        """
        Parameters
        ----------
        V0 : float -- initial/current voltage [V]
        I  : float -- discharge current [A] (positive = discharge)
        dt : float -- time step [s] (0 = steady-state snapshot)

        Returns
        -------
        dict with V_terminal, V_new, E_stored_J, E_stored_Wh, P_output, SOC, efficiency
        """
        V0 = max(self.V_min, min(self.V_max, float(V0)))
        I = float(I)
        dt = float(dt)

        # Terminal voltage (instantaneous ESR drop)
        V_terminal = V0 - I * self.R_esr

        # New voltage after time dt (charge redistribution)
        V_new = V0
        if dt > 0.0 and abs(I) > 0.0:
            V_new = V0 - I * dt / self.C
            V_new = max(self.V_min, min(self.V_max, V_new))

        # Stored energy at current voltage
        E_J = 0.5 * self.C * V_terminal ** 2
        E_Wh = E_J / 3600.0

        # Power at terminal voltage
        P_output = V_terminal * I

        # SOC based on stored energy fraction
        E_max = 0.5 * self.C * self.V_max ** 2
        soc = E_J / E_max if E_max > 0 else 0.0

        # Efficiency (power out / power without ESR)
        P_ideal = V0 * I
        eta = P_output / P_ideal if abs(P_ideal) > 0 else 1.0

        return {
            "V_terminal": round(V_terminal, 6),
            "V_new": round(V_new, 6),
            "E_stored_J": round(E_J, 4),
            "E_stored_Wh": round(E_Wh, 6),
            "P_output": round(P_output, 4),
            "SOC": round(max(0.0, min(1.0, soc)), 6),
            "efficiency": round(max(0.0, min(1.0, eta)), 6),
        }
