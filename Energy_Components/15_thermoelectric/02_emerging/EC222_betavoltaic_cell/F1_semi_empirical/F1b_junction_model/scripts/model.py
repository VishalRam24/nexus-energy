"""
EC222 — Betavoltaic Cell — F1b P-N Junction Electrical Model

Extends F1a (activity-based power estimate) with a full p-n junction
electrical model:

1. **Isotope decay** (same as F1a):
   A(t) = A0 * exp(-ln2 * t / t_half)
   I_gen(t) = eta_capture * A(t) * q / E_pair
   where E_pair = E_gap / q_e * W_factor (pair creation energy)

2. **Short-circuit current** (temperature- and time-dependent):
   I_sc(t, T) = I_sc_ref * (A(t)/A0) * (1 + alpha_Isc * (T - T_ref))
   The decay of Isc directly mirrors activity decay.

3. **Open-circuit voltage** (temperature-dependent):
   Voc(t, T) = Voc_ref + dVoc_dT * (T - T_ref) + (nkT/q) * ln(A(t)/A0)
   The logarithmic decay of Voc reflects reduced minority carrier injection
   as activity falls.
   Physical constraint: Voc >= 0.

4. **Fill factor** (degradation over time):
   FF(t) = FF_0 * max(1 - FF_decay_rate * t, 0.5)
   Radiation damage slowly degrades FF (mostly contacts/interfaces).

5. **Maximum power output**:
   P_max(t, T) = I_sc(t, T) * Voc(t, T) * FF(t)

6. **Efficiency**:
   eta_cell(t, T) = P_max / P_beta_absorbed
   where P_beta_absorbed = A(t) * E_beta * MeV_to_J * eta_capture

Outputs additionally include P_out_uW, fraction_remaining for comparison
with F1a, and the new junction quantities Isc_uA, Voc_V, FF, eta_junction.

References:
    Olsen, L.C. et al. (1993). Nucl. Instrum. Methods Phys. Res. B, 73(1), 139.
    Sychov, M. et al. (2008). Appl. Radiat. Isot. 66(2), 173.
    Prelas, M. et al. (2014). Progress in Nuclear Energy, 75, 117.
    Sun, W. et al. (2018). Applied Energy, 225, 390.
"""

import numpy as np

MeV_to_J = 1.602176634e-13
q_e = 1.602176634e-19   # C
k_B = 1.380649e-23      # J/K
ln2 = np.log(2.0)


class BetavoltaicF1b:
    """Betavoltaic cell — p-n junction electrical model with decay + T-dependence."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A0 = u["A0_Bq"]["value"]
        self.t_half = u["t_half_years"]["value"]
        self.E_beta = u["E_beta_MeV"]["value"]
        self.eta_cap = u["eta_capture"]["value"]
        self.E_gap = u["E_gap_eV"]["value"]           # eV
        self.T_ref = u["T_ref_K"]["value"]            # K
        self.n_ideality = u["ideality_factor"]["value"]
        self.Voc_ref = u["Voc_ref_V"]["value"]        # V
        self.dVoc_dT = u["dVoc_dT_mV_K"]["value"] * 1e-3  # V/K
        self.FF0 = u["fill_factor_0"]["value"]
        self.FF_decay = u["fill_factor_decay_rate"]["value"]  # 1/year
        self.Isc_ref = u["Isc_ref_uA"]["value"] * 1e-6   # A
        self.alpha_Isc = u["alpha_Isc"]["value"]       # 1/K

    def activity(self, t_years):
        """Activity at time t [Bq]."""
        t = np.asarray(t_years, dtype=float)
        return self.A0 * np.exp(-ln2 * t / self.t_half)

    def beta_power_absorbed(self, A):
        """Absorbed beta power in cell [W] = activity * E_beta * eta_capture."""
        return A * self.E_beta * MeV_to_J * self.eta_cap

    def short_circuit_current(self, t_years, T_K):
        """Short-circuit current [A] — decays with activity, rises slightly with T.
        I_sc(t, T) = I_sc_ref * (A(t)/A0) * (1 + alpha_Isc * (T - T_ref))
        """
        t = np.asarray(t_years, dtype=float)
        T = np.asarray(T_K, dtype=float)
        A = self.activity(t)
        decay_factor = A / self.A0
        T_factor = 1.0 + self.alpha_Isc * (T - self.T_ref)
        T_factor = np.maximum(T_factor, 0.1)   # physical floor
        return self.Isc_ref * decay_factor * T_factor

    def open_circuit_voltage(self, t_years, T_K):
        """Open-circuit voltage [V] — decreases with temperature and decay.
        Voc(t, T) = Voc_ref + dVoc_dT*(T-T_ref) + (n*kT/q)*ln(A(t)/A0)
        The last term accounts for reduced carrier injection at lower activity.
        Physical constraint: Voc >= 0.
        """
        t = np.asarray(t_years, dtype=float)
        T = np.asarray(T_K, dtype=float)
        A = self.activity(t)

        # Temperature term
        V_T_shift = self.dVoc_dT * (T - self.T_ref)

        # Activity decay logarithmic correction
        # As A decreases, Voc drops logarithmically: delta_Voc = (n*kT/q)*ln(A/A0)
        n_kT_q = self.n_ideality * k_B * T / q_e
        V_decay = n_kT_q * np.log(np.maximum(A / self.A0, 1e-15))

        Voc = self.Voc_ref + V_T_shift + V_decay
        return np.maximum(Voc, 0.0)

    def fill_factor(self, t_years):
        """Fill factor at time t — slow radiation damage degradation.
        FF(t) = FF0 * max(1 - FF_decay_rate * t, 0.5)
        Floor at 0.5 * FF0 (radiation damage saturates — interface pinning).
        """
        t = np.asarray(t_years, dtype=float)
        deg = np.maximum(1.0 - self.FF_decay * t, 0.5)
        return self.FF0 * deg

    def compute(self, t_years, T_K=None):
        """
        Parameters
        ----------
        t_years : float or array — time since deployment [years]
        T_K     : float or array — cell temperature [K] (default T_ref from params)

        Returns
        -------
        dict: activity_Bq, P_beta_absorbed_W, Isc_uA, Voc_V, FF,
              P_out_W, P_out_uW, eta_junction, fraction_remaining,
              P_beta_total_W (unabsorbed reference)
        """
        t = np.asarray(t_years, dtype=float)
        t = np.maximum(t, 0.0)

        if T_K is None:
            T = np.full_like(t, self.T_ref) if t.ndim > 0 else self.T_ref
        else:
            T = np.asarray(T_K, dtype=float)

        A = self.activity(t)
        P_absorbed = self.beta_power_absorbed(A)

        Isc = self.short_circuit_current(t, T)
        Voc = self.open_circuit_voltage(t, T)
        FF = self.fill_factor(t)

        P_out = Isc * Voc * FF
        P_out = np.maximum(P_out, 0.0)

        # Cell conversion efficiency (relative to absorbed beta power)
        eta_junction = np.where(P_absorbed > 1e-30, P_out / P_absorbed, 0.0)
        eta_junction = np.clip(eta_junction, 0.0, 1.0)

        # Total unabsorbed beta power (for reference)
        P_beta_total = A * self.E_beta * MeV_to_J

        fraction = A / self.A0

        return {
            "activity_Bq": A,
            "P_beta_total_W": P_beta_total,
            "P_beta_absorbed_W": P_absorbed,
            "Isc_uA": Isc * 1e6,
            "Voc_V": Voc,
            "FF": FF,
            "P_out_W": P_out,
            "P_out_uW": P_out * 1e6,
            "eta_junction": eta_junction,
            "fraction_remaining": fraction,
        }
