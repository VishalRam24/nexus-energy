"""
EC183 -- Circuit Breaker -- F1b Thermal Contact Model

Extends F1a with:
1. Temperature-dependent contact resistance:
       R(T) = R_ref * (1 + alpha_contact * (T_contact - T_ref))
   so that conduction losses increase with contact heating.

2. Ampacity limit (thermal rating per IEC 62271-1):
       I_max_thermal(T_ambient) = I_rated * sqrt((T_max_contact - T_ambient) /
                                                 (T_max_contact - T_ref))
   Linear derating for elevated ambient temperature.

3. Skin-effect correction factor (Bessel function approximation):
       delta = sqrt(rho_Cu / (pi * f * mu0))  [skin depth, m]
       F_skin = R_ac / R_dc = q * ber(q)*bei'(q) - bei(q)*ber'(q)
                               / (ber'(q)^2 + bei'(q)^2)
   where q = r * sqrt(2) / delta and ber/bei are Kelvin functions.
   For engineering use we adopt the IEC correction that approximates
   this as a polynomial for typical conductor sizes and frequencies.
   For MV (11 kV, 630 A, r~8 mm at 50 Hz): F_skin ~1.01 (negligible);
   however at high current/frequency the formula yields meaningful corrections.

4. Standby auxiliary power (always drawn when breaker is energised):
       P_aux = P_aux_W  [W]  (control coils, cubicle heater, indicators)

References:
    ABB (2021). Circuit Breaker Application Guide.
    IEC 62271-100:2021. High-voltage AC circuit-breakers.
    IEC 62271-1:2022. High-voltage switchgear and controlgear.
    Greenwood, A. (1991). Electrical Transients in Power Systems. Wiley.
    Raven, F.H. (1966). Kelvin functions ber/bei polynomial approximation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Kelvin function (ber/bei) approximation for skin effect
# Uses the Abramowitz & Stegun polynomial approximation for small q
# and the asymptotic form for large q.
# ---------------------------------------------------------------------------

def _skin_effect_factor(r_m: float, f_Hz: float, rho_Ohm_m: float = 1.72e-8) -> float:
    """
    Compute AC/DC resistance ratio F_skin = R_ac / R_dc for a solid round
    conductor of radius r_m at frequency f_Hz.

    Uses the Kelvin-function (ber/bei) exact formula for q < 10, and
    asymptotic expansion for q >= 10.

    Parameters
    ----------
    r_m      : conductor radius [m]
    f_Hz     : supply frequency [Hz]
    rho_Ohm_m: conductor resistivity [Ohm m] (default: Cu at 20 C)

    Returns
    -------
    F_skin   : R_ac / R_dc >= 1.0
    """
    mu0 = 4.0 * np.pi * 1e-7
    delta = np.sqrt(rho_Ohm_m / (np.pi * f_Hz * mu0))  # skin depth [m]
    q = r_m * np.sqrt(2.0) / delta                       # normalised radius

    if np.isscalar(q):
        return _ber_bei_ratio(q)
    return np.vectorize(_ber_bei_ratio)(q)


def _ber_bei_ratio(q: float) -> float:
    """
    R_ac/R_dc via Kelvin functions.
    Uses scipy.special.kelvin when q>4; polynomial approx for q<=4.
    Falls back to numpy-only implementation if scipy unavailable.
    """
    if q <= 0.0:
        return 1.0
    try:
        from scipy.special import kelvin
        # kelvin returns (ber, bei, ker, kei)
        ber, bei, _, _ = kelvin(q)
        # Derivatives via Kelvin recurrences:
        #   ber'(q) = (ber1(q) + bei1(q)) / sqrt(2)  where kelvin(sqrt(2)*q)[0]...
        # Use finite-difference approximation for simplicity
        h = 1e-5 * q if q > 1e-3 else 1e-8
        ber_ph, bei_ph, _, _ = kelvin(q + h)
        ber_mh, bei_mh, _, _ = kelvin(q - h)
        ber_p = (ber_ph - ber_mh) / (2.0 * h)
        bei_p = (bei_ph - bei_mh) / (2.0 * h)
        # IEC formula:
        numer = q * (ber * bei_p - bei * ber_p)
        denom = ber_p ** 2 + bei_p ** 2
        if denom < 1e-30:
            return 1.0
        return float(np.sqrt(2.0) * numer / denom)
    except ImportError:
        # Fallback: truncated series valid for q <= 8
        # R_ac/R_dc ~ 1 + q^4/48 - 5*q^8/34560 + ...  (low-q expansion)
        if q <= 8.0:
            return 1.0 + q ** 4 / 48.0 - 5.0 * q ** 8 / 34560.0
        # Asymptotic: R_ac/R_dc ~ q/(2*sqrt(2)) + 3/(64*sqrt(2)*q) + ...
        return q / (2.0 * np.sqrt(2.0)) * (1.0 + 3.0 / (64.0 * q ** 2))


class CircuitBreakerF1b:
    """
    Circuit breaker -- F1b thermal contact + ampacity + skin-effect model.

    Adds to F1a:
    - R(T) temperature-dependent contact resistance
    - Ampacity derating with ambient temperature
    - Skin-effect correction factor F_skin
    - Auxiliary standby power P_aux (always on)
    """

    CLOSED = "closed"
    OPEN = "open"

    def __init__(self, params: dict):
        u = params["unit"]
        self.R_ref = u["R_cb_ohm"]["value"]             # Ohm at T_ref
        self.I_rated = u["I_rated_A"]["value"]           # A
        self.I_interrupt_kA = u["I_interrupt_kA"]["value"]
        self.t_clear_s = u["t_clear_ms"]["value"] / 1000.0
        self.V_rated_kV = u["V_rated_kV"]["value"]
        self.T_ref = u["T_ref"]["value"]                 # degC
        self.alpha = u["alpha_contact"]["value"]         # 1/K
        self.T_max = u["T_max_contact"]["value"]         # degC
        self.P_aux = u["P_aux_W"]["value"]               # W
        self.f_Hz = u["f_system_Hz"]["value"]
        self.r_m = u["r_conductor_mm"]["value"] * 1e-3   # m

        # Pre-compute skin-effect factor (geometry/frequency fixed for a given unit)
        self.F_skin = float(_skin_effect_factor(self.r_m, self.f_Hz))
        self.F_skin = max(self.F_skin, 1.0)  # must be >= 1

    # ------------------------------------------------------------------
    # Sub-models
    # ------------------------------------------------------------------

    def contact_resistance(self, T_contact: float) -> float:
        """
        R(T) = R_ref * (1 + alpha * (T - T_ref)) * F_skin   [Ohm]

        F_skin accounts for AC skin effect (approximately constant
        for a given conductor geometry at one frequency).
        """
        T = np.asarray(T_contact, dtype=float)
        R_dc = self.R_ref * (1.0 + self.alpha * (T - self.T_ref))
        R_dc = np.maximum(R_dc, self.R_ref * 0.5)  # floor: R can't drop below ~50% (oxidation)
        return R_dc * self.F_skin

    def ampacity_limit(self, T_ambient: float = 20.0) -> float:
        """
        IEC 62271-1 Cl. 4.4 thermal derating:
            I_max = I_rated * sqrt((T_max_contact - T_ambient) / (T_max_contact - T_ref))

        Valid for T_ambient < T_max_contact.
        """
        T_a = np.asarray(T_ambient, dtype=float)
        ratio = (self.T_max - T_a) / (self.T_max - self.T_ref)
        ratio = np.maximum(ratio, 0.0)
        return self.I_rated * np.sqrt(ratio)

    def compute(self, I_A: float, state: str = "closed",
                T_contact: float = 50.0, T_ambient: float = 20.0,
                I_fault_kA: float = 0.0) -> dict:
        """
        Parameters
        ----------
        I_A           : Load current [A] (steady-state)
        state         : "closed" or "open"
        T_contact     : Contact temperature [degC] (50 C typical under load)
        T_ambient     : Ambient temperature [degC] (for ampacity derating)
        I_fault_kA    : Prospective fault current [kA] for interrupting check

        Returns
        -------
        dict with:
            P_cond_W          : Conduction losses through R(T) [W]
            P_aux_W           : Auxiliary standby power [W] (always on)
            P_total_W         : Total power dissipation [W]
            R_contact_Ohm     : Temperature + skin-corrected resistance [Ohm]
            F_skin            : AC/DC resistance ratio
            I_max_thermal_A   : Ampacity limit at T_ambient [A]
            thermal_margin    : (I_max - I) / I_max  [0-1]
            is_overloaded     : I > I_max_thermal
            can_interrupt     : I_fault_kA <= I_interrupt_kA
            E_fault_J         : I^2*t energy during fault clearing [J]
            T_contact         : Contact temperature used [degC]
        """
        I = np.asarray(I_A, dtype=float)
        I_fault = np.asarray(I_fault_kA, dtype=float)
        I_fault_A = I_fault * 1000.0

        R = self.contact_resistance(T_contact)

        if state == self.CLOSED:
            P_cond = I ** 2 * R
        else:
            P_cond = np.zeros_like(I)

        P_total = P_cond + self.P_aux

        I_max = self.ampacity_limit(T_ambient)
        thermal_margin = np.clip((I_max - I) / (I_max + 1e-12), -1.0, 1.0)
        is_overloaded = I > I_max

        can_interrupt = I_fault <= self.I_interrupt_kA
        E_fault = I_fault_A ** 2 * R * self.t_clear_s

        return {
            "P_cond_W": P_cond,
            "P_aux_W": np.full_like(P_cond, self.P_aux) if np.ndim(P_cond) > 0 else self.P_aux,
            "P_total_W": P_total,
            "R_contact_Ohm": np.full_like(P_cond, float(R)) if np.ndim(P_cond) > 0 else float(R),
            "F_skin": self.F_skin,
            "I_max_thermal_A": I_max,
            "thermal_margin": thermal_margin,
            "is_overloaded": is_overloaded,
            "can_interrupt": can_interrupt,
            "E_fault_J": E_fault,
            "T_contact": T_contact,
            "state": state,
        }
