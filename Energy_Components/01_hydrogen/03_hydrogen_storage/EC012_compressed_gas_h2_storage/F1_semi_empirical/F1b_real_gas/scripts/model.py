"""
EC012 — Compressed Gas H2 Storage — F1b Real-Gas Model

Extends F1a (ideal-gas compressibility) by adding:
1. Temperature-dependent compressibility Z(T, P) using truncated virial equation
   derived from Leachman et al. (2009):
       Z(T, P) = 1 + B(T)*P_MPa + C(T)*P_MPa^2
       B(T) = A1 + A2 / T^2           [MPa^-1]
       C(T) = B1 + B2 / T^2           [MPa^-2]
   Above 200 bar, Z > 1 for H2 — stored mass is LESS than ideal-gas prediction.

2. Ambient temperature coupling:
   - T_amb affects tank equilibrium temperature (tanks cool/heat toward T_amb)
   - Stored mass at equilibrium: m = P*V / (Z(T_amb, P)*R_H2*T_amb)

3. Heat-of-compression dynamics during fill:
   - Temperature rise during fast compression:
       dT = -h_comp * dm / (Cm_tank)   [K]
   where h_comp is enthalpy of compression per kg H2, Cm_tank = thermal mass [J/K]
   - After fill, tank cools back to T_amb; this reduces pressure (density effect)

4. Compression work with real-gas correction:
   - w_comp = (k/(k-1)) * Z(T1,P1) * R_H2 * T1 * ((P2/P1)^((k-1)/k) - 1) / eta_s

References:
    Leachman, J.W. et al. (2009). Fundamental equations of state for parahydrogen,
    normal hydrogen, and orthohydrogen. J. Phys. Chem. Ref. Data, 38(3), 721-748.
    Zheng, J. et al. (2012). Int. J. Hydrogen Energy, 37(2), 1048-1057.
    Lemmon, E.W. et al. (2008). NIST Chemistry WebBook.
    Sdanghi, G. et al. (2019). Review of hydrogen compression. Renewable and
    Sustainable Energy Reviews, 102, 150-170.
"""

import numpy as np

R_UNIVERSAL = 8.314   # J/(mol·K)


class CompressedGasH2F1b:
    """
    Compressed gas hydrogen storage — real-gas + thermal coupling model.

    Key F1b additions over F1a:
    - Z(T, P): temperature-dependent virial compressibility
    - T_amb coupling for equilibrium stored mass
    - Heat-of-compression temperature transient during fill
    - Real-gas correction in compression work
    """

    def __init__(self, params: dict):
        tank = params["tank"]["type_IV"]
        h2 = params["hydrogen"]
        comp = params["compressor"]
        z = params["compressibility"]
        amb = params["ambient"]

        self.V_tank = tank["volume"]["value"]           # m3
        self.P_max = tank["max_pressure"]["value"]      # bar
        self.P_min = tank["min_pressure"]["value"]      # bar
        self.tank_mass = tank["mass_empty"]["value"]    # kg
        self.Cm_tank = tank["thermal_mass"]["value"]    # J/K  effective thermal mass
        self.UA_amb = tank["UA_ambient"]["value"]       # W/K

        self.M_H2 = h2["molar_mass"]["value"]          # kg/mol
        self.LHV = h2["LHV"]["value"]                  # MJ/kg
        self.gamma = h2["gamma"]["value"]
        self.cp_H2 = h2["cp"]["value"]                 # J/(kg·K)
        self.h_comp = h2["h_comp"]["value"]            # J/kg (heat of compression)

        self.eta_s = comp["eta_isentropic"]["value"]
        self.T_inlet_default = comp["T_inlet"]["value"]   # K
        self.P_inlet_default = comp["P_inlet"]["value"]   # bar

        # Virial coefficients for Z(T, P)  — P in MPa
        self.A1 = z["A1"]   # B(T) = A1 + A2/T^2  [MPa^-1]
        self.A2 = z["A2"]
        self.B1 = z["B1"]   # C(T) = B1 + B2/T^2  [MPa^-2]
        self.B2 = z["B2"]

        self.T_amb_default = amb["T_amb_default"]["value"]   # K

        # Specific gas constant for H2
        self.R_H2 = R_UNIVERSAL / self.M_H2   # J/(kg·K)

    # ------------------------------------------------------------------
    # Real-gas compressibility
    # ------------------------------------------------------------------

    def _B_virial(self, T_K):
        """Second virial coefficient B(T) [MPa^-1]."""
        T = np.asarray(T_K, dtype=float)
        return self.A1 + self.A2 / T ** 2

    def _C_virial(self, T_K):
        """Third virial coefficient C(T) [MPa^-2]."""
        T = np.asarray(T_K, dtype=float)
        return self.B1 + self.B2 / T ** 2

    def compressibility_factor(self, P_bar, T_K):
        """
        Real-gas compressibility Z(T, P).

        Uses truncated virial equation:
            Z = 1 + B(T)*P_MPa + C(T)*P_MPa^2

        For H2: Z > 1 at all pressures above ~10 bar and T > 150 K.
        At 700 bar (70 MPa) and 300 K, Z ≈ 1.40.

        Args:
            P_bar: Pressure [bar]
            T_K:   Temperature [K]
        """
        P = np.asarray(P_bar, dtype=float)
        T = np.asarray(T_K, dtype=float)
        P_MPa = P * 0.1   # bar -> MPa
        B = self._B_virial(T)
        C = self._C_virial(T)
        return np.maximum(1.0 + B * P_MPa + C * P_MPa ** 2, 1e-3)

    # ------------------------------------------------------------------
    # Stored mass and energy
    # ------------------------------------------------------------------

    def stored_mass(self, P_bar, T_K):
        """
        Mass of H2 stored [kg] using real-gas equation of state:
            m = P*V / (Z(T,P) * R_H2 * T)

        Args:
            P_bar: Tank pressure [bar]
            T_K:   Tank gas temperature [K]
        """
        P = np.asarray(P_bar, dtype=float)
        T = np.asarray(T_K, dtype=float)
        P_Pa = P * 1e5
        Z = self.compressibility_factor(P, T)
        return P_Pa * self.V_tank / (Z * self.R_H2 * T)

    def stored_mass_at_Tamb(self, P_bar, T_amb_K=None):
        """
        Mass at thermal equilibrium with ambient temperature.
        Uses T_amb as the tank gas temperature.

        Args:
            P_bar:    Tank pressure [bar]
            T_amb_K:  Ambient temperature [K] (default: self.T_amb_default)
        """
        if T_amb_K is None:
            T_amb_K = self.T_amb_default
        return self.stored_mass(P_bar, np.asarray(T_amb_K, dtype=float))

    def energy_stored(self, P_bar, T_K):
        """Stored energy [MJ] = m_H2 * LHV."""
        return self.stored_mass(P_bar, T_K) * self.LHV

    def fill_fraction(self, P_bar, T_K):
        """Fill fraction (0-1) relative to stored mass at P_max and T_K."""
        m = self.stored_mass(P_bar, T_K)
        m_max = self.stored_mass(self.P_max, T_K)
        return np.clip(m / m_max, 0.0, 1.0)

    def gravimetric_density(self, P_bar, T_K):
        """Gravimetric storage density [wt%]."""
        m_h2 = self.stored_mass(P_bar, T_K)
        return m_h2 / (m_h2 + self.tank_mass) * 100.0

    def volumetric_density(self, P_bar, T_K):
        """Volumetric storage density [kg_H2/m3]."""
        return self.stored_mass(P_bar, T_K) / self.V_tank

    # ------------------------------------------------------------------
    # Heat of compression — tank temperature rise during fill
    # ------------------------------------------------------------------

    def compression_temperature_rise(self, dm_kg):
        """
        Tank temperature rise [K] from adding dm_kg of H2 via compression.

        Model:
            dT = h_comp * dm / Cm_tank

        where h_comp is the specific enthalpy deposited per kg of H2 added
        (includes gas work and thermal energy from compression).

        Args:
            dm_kg: Mass of H2 added [kg]
        Returns:
            dT [K]: positive (tank heats up)
        """
        dm = np.asarray(dm_kg, dtype=float)
        return np.maximum(self.h_comp * dm / self.Cm_tank, 0.0)

    def tank_temperature_after_fill(self, P1_bar, P2_bar, T_amb_K=None):
        """
        Approximate tank temperature after filling from P1 to P2 [K].

        Assumes tank starts at T_amb, mass dm is added, temperature rises by dT.

        Args:
            P1_bar:   Initial pressure [bar]
            P2_bar:   Final pressure [bar]
            T_amb_K:  Ambient temperature [K]
        Returns:
            T_final [K]: tank temperature immediately post-fill
        """
        if T_amb_K is None:
            T_amb_K = self.T_amb_default
        T_amb = np.asarray(T_amb_K, dtype=float)
        m1 = self.stored_mass(P1_bar, T_amb)
        m2 = self.stored_mass(P2_bar, T_amb)
        dm = np.maximum(m2 - m1, 0.0)
        dT = self.compression_temperature_rise(dm)
        return T_amb + dT

    def thermal_equilibration_time(self):
        """
        Characteristic thermal equilibration time [s] = Cm_tank / UA_amb.

        After fill, tank cools back to T_amb with this time constant.
        """
        return self.Cm_tank / self.UA_amb

    def tank_temperature_cooling(self, T_post_fill_K, T_amb_K, t_s):
        """
        Tank temperature [K] at time t_s after fill (Newton cooling).

            T(t) = T_amb + (T_fill - T_amb) * exp(-t / tau)

        Args:
            T_post_fill_K: Tank temperature immediately after fill [K]
            T_amb_K:       Ambient temperature [K]
            t_s:           Time after fill [s]
        """
        T_f = np.asarray(T_post_fill_K, dtype=float)
        T_a = np.asarray(T_amb_K, dtype=float)
        t = np.asarray(t_s, dtype=float)
        tau = self.thermal_equilibration_time()
        return T_a + (T_f - T_a) * np.exp(-t / tau)

    # ------------------------------------------------------------------
    # Compression work
    # ------------------------------------------------------------------

    def compression_work(self, P1_bar, P2_bar, T1_K=None):
        """
        Specific compression work [kJ/kg_H2] with real-gas correction.

        Adds Z(T1, P1) correction factor to isentropic work:
            w = Z(T1,P1) * (k/(k-1)) * R_H2 * T1 * ((P2/P1)^((k-1)/k) - 1) / eta_s

        Args:
            P1_bar: Inlet pressure [bar]
            P2_bar: Outlet pressure [bar]
            T1_K:   Inlet temperature [K] (default: self.T_inlet_default)
        """
        P1 = np.asarray(P1_bar, dtype=float)
        P2 = np.asarray(P2_bar, dtype=float)
        T1 = self.T_inlet_default if T1_K is None else np.asarray(T1_K, dtype=float)
        Z1 = self.compressibility_factor(P1, T1)
        k = self.gamma
        ratio = (P2 / P1) ** ((k - 1.0) / k)
        w = Z1 * (k / (k - 1.0)) * self.R_H2 * T1 * (ratio - 1.0) / self.eta_s
        return w / 1000.0   # J/kg -> kJ/kg

    # ------------------------------------------------------------------
    # T_amb effect on usable capacity
    # ------------------------------------------------------------------

    def usable_mass_vs_Tamb(self, T_amb_K):
        """
        Usable H2 mass [kg] (between P_min and P_max) as function of T_amb.

        Colder ambient -> denser gas -> more mass stored at same pressure.

        Args:
            T_amb_K: Ambient temperature [K] (scalar or array)
        """
        T = np.asarray(T_amb_K, dtype=float)
        m_max = self.stored_mass(self.P_max, T)
        m_min = self.stored_mass(self.P_min, T)
        return m_max - m_min
