"""
EC222 — Betavoltaic Cell — F1a Activity Model

Radioactive decay + semiconductor conversion:

    Activity:
    A(t) = A0 * exp(-ln(2) * t / t_half)   [Bq]

    Thermal power from beta emission:
    P_beta(t) = A(t) * E_beta   [W]   (1 Bq * 1 eV = 1.602e-19 W)

    Power output:
    P_out(t) = P_beta(t) * eta_capture * eta_conv   [W]

    Half-life decay: activity halves every t_half years.
    For Ni-63 (t_half=100.2y): after 50 years → 70% of initial power.
    For Tritium (t_half=12.32y): after 50 years → 5.7% of initial power.

    Output is microWatt scale for typical laboratory sources.

References:
    Olsen, L.C. et al. (1993). Nucl. Instrum. Methods Phys. Res. B, 73(1), 139.
    Sychov, M. et al. (2008). Appl. Radiat. Isot. 66(2), 173.
    Blanovsky, A.E. (2012). IEEE Aerospace Conf.
"""

import numpy as np

# Conversion: 1 MeV = 1.602176634e-13 J
MeV_to_J = 1.602176634e-13
ln2 = np.log(2.0)


class BetavoltaicF1a:
    """Betavoltaic cell — isotope activity + semiconductor conversion model."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A0 = u["A0_Bq"]["value"]              # Bq
        self.t_half = u["t_half_years"]["value"]    # years
        self.E_beta = u["E_beta_MeV"]["value"]      # MeV
        self.eta_cap = u["eta_capture"]["value"]    # -
        self.eta_conv = u["eta_conv"]["value"]      # -

    def activity(self, t_years):
        """Isotope activity at time t [Bq]."""
        t = np.asarray(t_years, dtype=float)
        return self.A0 * np.exp(-ln2 * t / self.t_half)

    def compute(self, t_years):
        """
        Parameters
        ----------
        t_years : float or array — time since deployment [years]

        Returns
        -------
        dict: activity_Bq, P_beta_W, P_out_W, P_out_uW, fraction_remaining
        """
        t = np.asarray(t_years, dtype=float)
        t = np.maximum(t, 0.0)

        A = self.activity(t)

        # Beta power [W] = activity [Bq=decays/s] * energy per decay [J]
        P_beta = A * self.E_beta * MeV_to_J   # W

        # Electrical output
        P_out = P_beta * self.eta_cap * self.eta_conv

        fraction = A / self.A0  # fraction of initial activity

        return {
            "activity_Bq": A,
            "P_beta_W": P_beta,
            "P_out_W": P_out,
            "P_out_uW": P_out * 1e6,
            "fraction_remaining": fraction,
        }
