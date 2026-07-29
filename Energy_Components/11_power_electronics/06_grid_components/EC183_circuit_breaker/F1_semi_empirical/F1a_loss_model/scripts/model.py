"""
EC183 — Circuit Breaker — F1a Conduction Loss Model

Closed state:
    P_loss = I^2 * R_cb        [W]   — contact resistance heating

Thermal energy dissipated during fault clearing:
    E_fault = I_fault^2 * R_cb * t_clear    [J]   — I^2*t withstand

Interrupting rating check:
    can_interrupt = I_fault <= I_interrupt_rating

State: closed (conducting) or open (no current).

Reference:
    ABB (2021). Circuit Breaker Application Guide.
    IEC 62271-100: High-voltage switchgear — AC circuit-breakers.
"""

import numpy as np


class CircuitBreakerModel:
    """F1a conduction loss + interrupting rating model."""

    CLOSED = "closed"
    OPEN = "open"

    def __init__(self, params: dict):
        u = params["unit"]
        self.R_cb = u["R_cb_ohm"]["value"]           # Ohm
        self.I_rated = u["I_rated_A"]["value"]        # A
        self.I_interrupt_kA = u["I_interrupt_kA"]["value"]  # kA
        self.t_clear_s = u["t_clear_ms"]["value"] / 1000.0  # s
        self.V_rated_kV = u["V_rated_kV"]["value"]    # kV

    def compute(self, I_A: float, state: str = "closed",
                I_fault_kA: float = 0.0) -> dict:
        """
        Parameters
        ----------
        I_A         : Load current [A] (steady-state)
        state       : "closed" or "open"
        I_fault_kA  : Prospective fault current [kA] for interrupting check

        Returns
        -------
        dict with P_loss_W, is_overloaded, can_interrupt,
        E_fault_J, I_fault_kA, thermal_rating_ok
        """
        I_A = np.asarray(I_A, dtype=float)
        I_fault_kA = np.asarray(I_fault_kA, dtype=float)
        I_fault_A = I_fault_kA * 1000.0

        if state == self.CLOSED:
            P_loss = I_A ** 2 * self.R_cb
        else:
            P_loss = np.zeros_like(I_A)

        is_overloaded = I_A > self.I_rated
        can_interrupt = I_fault_kA <= self.I_interrupt_kA

        # Energy during fault clearing (I^2*t)
        E_fault = I_fault_A ** 2 * self.R_cb * self.t_clear_s

        # Thermal rating: continuous current rating check
        thermal_rating_ok = ~is_overloaded

        return {
            "P_loss_W": P_loss,
            "is_overloaded": is_overloaded,
            "can_interrupt": can_interrupt,
            "E_fault_J": E_fault,
            "I_fault_kA": I_fault_kA,
            "thermal_rating_ok": thermal_rating_ok,
            "state": state,
        }
