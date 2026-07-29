"""
EC095 — Thermoelectric Cooler (Peltier) — F1b Multi-Stage Cascade Model

Physics of multi-stage (cascaded) TECs:

  Stage k pumps heat from T_c_k (cold side) to T_h_k (hot side).
  Inter-stage coupling:
      T_c_{k+1} = T_h_k + dT_contact          [contact resistance]
      Q_c_{k+1} = Q_h_k = Q_c_k + W_k         [energy balance]

  So stage k+1 must pump both the cooling load AND the electrical dissipation
  of all lower stages. This is why COP of a cascade falls below a single stage
  operating across the same total temperature span.

Single-stage equations (Goldsmid/Rowe per module):
    Q_c   = alpha * I * T_c - 0.5 * I^2 * R - K * (T_h - T_c)
    W_in  = alpha * I * (T_h - T_c) + I^2 * R
    Q_h   = Q_c + W_in

Cascade algorithm:
  1. Given I_1, I_2, ..., T_cold (external cold), T_hot (external hot):
  2. Set T_c_1 = T_cold.
  3. Compute Q_c_1, W_1, Q_h_1 for stage 1.
  4. T_c_2 = T_h_1 + dT_contact (find T_h_1 from energy balance: see below).
     Note: T_h_1 is unknown — we solve it so that the stage-1 energy balance
     is consistent. Since Q_h_1 = Q_c_2 must be delivered by stage 2, and
     stage 2 rejects to T_hot, we need T_h_1.

  For F1b we use a simplified decoupled approach:
    - Total dT = T_hot - T_cold.
    - Each stage spans dT/n_stages (equal partition) plus inter-stage losses.
    - This is the standard first-pass cascade design approach (Rowe Handbook, ch.6).

  The INTER-STAGE TEMPERATURE is the key output: T_inter = T_cold + dT/n_stages.

  COP cascade:
    COP_cascade = Q_c_1 / (W_1 + W_2 + ... + W_n)
    This is ALWAYS lower than single-stage COP at same dT_total.

  Degradation factor per stage:
    Applied to effective ZT (material property age degradation from
    thermal cycling — Goldsmid 2010 sec.7.3).

References:
    Goldsmid, H.J. (2010). Introduction to Thermoelectricity, Springer, ch.6.
    Rowe, D.M. (Ed.) (2006). CRC Handbook of Thermoelectrics, sec.6.
    Chein, R., Chen, Y. (2005). Int. J. Refrigeration 28, 828-839.
    Riffat, S.B., Ma, X. (2003). Applied Thermal Engineering 23, 913-935.
"""

import numpy as np


class _TECStage:
    """Single TEC stage with N_modules in parallel (same current, same T boundaries)."""

    def __init__(self, alpha, R, K, N, I_max, degradation_factor=1.0):
        # Apply degradation to effective Seebeck coefficient and ZT proxy
        self.alpha  = alpha * degradation_factor
        self.R      = R / degradation_factor   # degraded alpha → higher R (lower ZT)
        self.K      = K
        self.N      = N
        self.I_max  = I_max

    def Q_c(self, I, T_c_K, T_h_K):
        I   = np.asarray(I,     dtype=float)
        Tc  = np.asarray(T_c_K, dtype=float)
        Th  = np.asarray(T_h_K, dtype=float)
        q_mod = self.alpha * I * Tc - 0.5 * I * I * self.R - self.K * (Th - Tc)
        return self.N * q_mod   # W per stack

    def W_in(self, I, T_c_K, T_h_K):
        I   = np.asarray(I,     dtype=float)
        Tc  = np.asarray(T_c_K, dtype=float)
        Th  = np.asarray(T_h_K, dtype=float)
        w_mod = self.alpha * I * (Th - Tc) + I * I * self.R
        return self.N * w_mod

    def Q_h(self, I, T_c_K, T_h_K):
        return self.Q_c(I, T_c_K, T_h_K) + self.W_in(I, T_c_K, T_h_K)

    def optimum_current(self, T_c_K):
        """I_opt = alpha * T_c / R  (maximises Q_c)."""
        Tc = np.asarray(T_c_K, dtype=float)
        return np.minimum(self.alpha * Tc / self.R, self.I_max)


class PeltierTECF1b:
    """Multi-stage cascaded Peltier cooler with inter-stage T matching."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.n_stages = int(u["n_stages"]["value"])
        self.dT_contact = float(u["contact_resistance_K_per_W"]["value"])
        self.cop_stage_factor = float(u["COP_degradation_per_stage"]["value"])

        # Build stage list (stage_1 = bottom/cold, stage_n = top/hot)
        self._stages = []
        for k in range(1, self.n_stages + 1):
            sk = u[f"stage_{k}"]
            stage = _TECStage(
                alpha=sk["alpha_module"]["value"],
                R=sk["R_module"]["value"],
                K=sk["K_module"]["value"],
                N=int(sk["n_modules"]["value"]),
                I_max=sk["I_max"]["value"],
                degradation_factor=sk["degradation_factor"]["value"],
            )
            self._stages.append(stage)

    # ------------------------------------------------------------------
    # Equal-partition cascade solve
    # ------------------------------------------------------------------

    def solve(self, I_stages, T_cold_c, T_hot_c):
        """
        Cascade solve with equal temperature partition between stages.

        Parameters
        ----------
        I_stages  : list or array of length n_stages — current per stage [A]
        T_cold_c  : float or array — external cold-side temperature [degC]
        T_hot_c   : float or array — external hot-side temperature [degC]

        Returns
        -------
        dict with Q_c_kw, W_total_kw, COP, T_inter (inter-stage temperature),
                  Q_c_per_stage, W_per_stage, cop_vs_single_stage
        """
        I_stages = np.asarray(I_stages, dtype=float)
        if I_stages.ndim == 1 and len(I_stages) != self.n_stages:
            raise ValueError(f"I_stages must have {self.n_stages} elements")

        T_c_K = np.asarray(T_cold_c, dtype=float) + 273.15
        T_h_K = np.asarray(T_hot_c,  dtype=float) + 273.15

        # Equal temperature partition across stages
        dT_total = T_h_K - T_c_K
        dT_per_stage = dT_total / self.n_stages

        # Stage boundary temperatures
        T_boundaries = [T_c_K + k * dT_per_stage for k in range(self.n_stages + 1)]

        Q_c_list = []
        W_list   = []

        for k, stage in enumerate(self._stages):
            Tc_k = T_boundaries[k]
            Th_k = T_boundaries[k + 1]
            # Contact resistance adds dT_contact * Q_h_{k-1} to the inter-stage gap
            # For F1b we fold this into the dT_contact parameter as a fixed offset:
            if k > 0:
                Tc_k = Tc_k + self.dT_contact * np.maximum(Q_prev_h, 0.0)
                Th_k = np.maximum(Th_k, Tc_k + 1.0)

            if I_stages.ndim == 0 or I_stages.shape == ():
                I_k = float(I_stages)
            else:
                I_k = float(I_stages[k]) if I_stages.ndim == 1 else I_stages[k]

            Q_c_k = stage.Q_c(I_k, Tc_k, Th_k)
            W_k   = stage.W_in(I_k, Tc_k, Th_k)
            Q_h_k = Q_c_k + W_k
            Q_prev_h = Q_h_k

            Q_c_list.append(Q_c_k)
            W_list.append(W_k)

        Q_c_total = np.maximum(Q_c_list[0], 0.0)   # net cold absorption = stage-1 Q_c
        W_total   = sum(W_list)

        cop = np.where(W_total > 1e-6,
                       Q_c_total / np.where(W_total > 1e-6, W_total, 1.0),
                       0.0)

        # Compare to equivalent single-stage using SAME total N_modules across full dT.
        # This is the physically correct benchmark: cascade vs a hypothetical single
        # stage with all modules stacked, each operating the full temperature span.
        # The cascade wins at large dT because each stage operates at lower dT,
        # but the per-stage energy balance (stage k must pump Q_h_{k-1}) offsets this.
        # Reference: Goldsmid (2010) ch.6 — cascade analysis vs. single-stage equivalent.
        s0 = self._stages[0]
        N_total = sum(stg.N for stg in self._stages)
        I_rep = np.mean([float(I_stages[k]) if I_stages.ndim > 0
                         else float(I_stages) for k in range(self.n_stages)])
        # Build equivalent single stage
        from copy import copy as _copy
        s_equiv = _copy(s0)
        s_equiv.N = N_total
        cop_single = np.where(
            W_total > 1e-6,
            np.maximum(s_equiv.Q_c(I_rep, T_c_K, T_h_K), 0.0) / np.maximum(
                s_equiv.W_in(I_rep, T_c_K, T_h_K), 1e-6), 0.0)

        # Also apply per-stage degradation factor to overall COP
        cop_cascade = cop * (self.cop_stage_factor ** (self.n_stages - 1))

        return {
            "Q_c_kw":               Q_c_total / 1000.0,
            "W_total_kw":           W_total    / 1000.0,
            "COP":                  np.clip(cop_cascade, 0.0, 5.0),
            "T_inter_C":            T_boundaries[1] - 273.15,
            "Q_c_per_stage_W":      Q_c_list,
            "W_per_stage_W":        W_list,
            "COP_single_stage_ref": np.clip(cop_single, 0.0, 5.0),
        }

    def optimum_currents(self, T_cold_c, T_hot_c):
        """Return optimum current for each stage at equal-partition temperatures."""
        T_c_K = np.asarray(T_cold_c, dtype=float) + 273.15
        T_h_K = np.asarray(T_hot_c,  dtype=float) + 273.15
        dT_per_stage = (T_h_K - T_c_K) / self.n_stages
        I_opts = []
        for k, stage in enumerate(self._stages):
            Tc_k = T_c_K + k * dT_per_stage
            I_opts.append(stage.optimum_current(Tc_k))
        return np.array(I_opts)
