"""
EC115 — Integrated Gasification Combined Cycle (IGCC) — F1b Part-Load / Flue-Loss Model

Extends F1a (efficiency curve) with:
  1. Part-load flue/stack gas heat loss
     flue_loss(PLR) = flue_base + flue_partload_coeff * (1 - PLR)
     At part load, combustion temperature drops → stack loss fraction increases.

  2. Auxiliary power correction
     aux_frac(PLR) = aux_base + aux_partload_coeff * (1 - PLR)
     IGCC has a large Air Separation Unit (ASU) whose power draw does NOT scale
     linearly with PLR → auxiliary fraction rises at part load.

  3. Net efficiency including both corrections
     eta_net(PLR, T_amb) = eta_cycle(PLR, T_amb) * (1 - flue_loss(PLR)) * (1 - aux_frac(PLR))
     where eta_cycle is the F1a efficiency (Rankine/CCGT cycle efficiency).

  4. CO2 intensity (restricted to rated load in tests)
     CO2 intensity is well-defined at rated load; at low PLR the efficiency
     degradation raises CO2/kWh well above the published 700-800 g/kWh benchmark.
     See RATIONALE in test_model.py.

Physics notes:
  - IGCC has the worst part-load flexibility of all coal technologies due to
    gasifier turndown constraints (min PLR ~40%).
  - ASU (air separation unit) accounts for ~8-10% of gross output and cannot
    ramp proportionally: auxiliary fraction rises sharply at part load.
  - Flue loss from HRSG is smaller than in pulverized coal because the CCGT
    block recovers heat efficiently, but it still increases at part load.
  - Ambient temperature effect comes through the gas turbine inlet (compressor).

References:
    Cormos, C.-C. (2012). Int. J. Hydrogen Energy, 37(4), 3083-3095.
    IEA GHG R&D Programme (2003). Potential for improvements in gasification
        combined cycle power generation with CO2 capture.
    Higman, C. & van der Burgt, M. (2008). Gasification, 2nd ed. Elsevier.
    Booras, G. & Holt, N. (2004). EPRI Gasification Technologies Conference.
"""

import numpy as np


class IGCCF1b:
    """IGCC — part-load flue/auxiliary loss model (coal-to-electricity, no CCS)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated               = u["rated_power_mw"]["value"]
        self.eta_iso               = u["eta_iso"]["value"]
        self.T_amb_ref             = u["T_amb_ref"]["value"]
        self.k_amb                 = u["k_amb"]["value"]
        self.a0                    = u["plr_coeffs"]["a0"]["value"]
        self.a1                    = u["plr_coeffs"]["a1"]["value"]
        self.a2                    = u["plr_coeffs"]["a2"]["value"]
        self.LHV_coal              = u["LHV_coal"]["value"]
        self.CO2_per_kg_coal       = u["CO2_per_kg_coal"]["value"]
        self.syngas_lhv            = u["syngas_lhv"]["value"]
        self.cge                   = u["gasifier_cold_gas_efficiency"]["value"]
        self.min_plr               = u["min_plr"]["value"]
        self.flue_base             = u["flue_loss_base"]["value"]
        self.flue_plr_coeff        = u["flue_loss_partload_coeff"]["value"]
        self.aux_base              = u["aux_power_base_fraction"]["value"]
        self.aux_plr_coeff         = u["aux_partload_coeff"]["value"]

    # ------------------------------------------------------------------
    # F1a correction factors (cycle only)
    # ------------------------------------------------------------------

    def _f_plr_cycle(self, plr):
        """Part-load PLR correction on cycle (gasifier + CCGT) — same as F1a."""
        plr = np.asarray(plr, dtype=float)
        return self.a0 + self.a1 * plr + self.a2 * plr ** 2

    def _f_amb(self, T_amb):
        """Ambient temperature derating on GT inlet."""
        T_amb = np.asarray(T_amb, dtype=float)
        return 1.0 - self.k_amb * (T_amb - self.T_amb_ref)

    def _eta_cycle(self, plr, T_amb):
        """Gross cycle efficiency before flue/aux corrections (same as F1a)."""
        return self.eta_iso * self._f_plr_cycle(plr) * self._f_amb(T_amb)

    # ------------------------------------------------------------------
    # F1b additional correction factors
    # ------------------------------------------------------------------

    def flue_loss_fraction(self, plr):
        """
        Stack/flue gas heat loss as fraction of coal thermal input.

        flue_loss(PLR) = flue_base + flue_plr_coeff * (1 - PLR)

        At PLR=1.0: flue_loss = 1.5% (HRSG running efficiently at design)
        At PLR=0.4: flue_loss = 1.5% + 1.2% * 0.6 = 2.22% (excess gas, lower T)

        Physics: At part load, combustion temperature drops, exhaust gas volume
        per unit output increases, and HRSG heat recovery degrades → more heat
        leaves via stack.
        """
        plr = np.asarray(plr, dtype=float)
        return self.flue_base + self.flue_plr_coeff * (1.0 - plr)

    def aux_power_fraction(self, plr):
        """
        Auxiliary power consumption fraction.

        aux_frac(PLR) = aux_base + aux_plr_coeff * (1 - PLR)

        At PLR=1.0: aux = 8% (ASU at design, compressors at design point)
        At PLR=0.4: aux = 8% + 2% * 0.6 = 9.2% (ASU cannot scale proportionally)

        Physics: IGCC requires more auxiliary power per unit output at part load
        because the Air Separation Unit has minimum operating requirements and
        does not reduce proportionally when plant output drops.
        """
        plr = np.asarray(plr, dtype=float)
        return self.aux_base + self.aux_plr_coeff * (1.0 - plr)

    # ------------------------------------------------------------------
    # Net efficiency including all F1b losses
    # ------------------------------------------------------------------

    def efficiency(self, plr, T_amb):
        """
        Net LHV efficiency including flue losses and auxiliary power correction.

        eta_net = eta_cycle * (1 - flue_loss) * (1 - aux_frac)

        Units: dimensionless (LHV basis)
        """
        eta_c = self._eta_cycle(plr, T_amb)
        f_l   = self.flue_loss_fraction(plr)
        a_f   = self.aux_power_fraction(plr)
        return eta_c * (1.0 - f_l) * (1.0 - a_f)

    # ------------------------------------------------------------------
    # Power and mass flow outputs
    # ------------------------------------------------------------------

    def power_mw(self, plr):
        """Net electrical output [MW_e]."""
        return self.P_rated * np.asarray(plr, dtype=float)

    def coal_rate_kgs(self, plr, T_amb):
        """
        Coal mass flow rate [kg/s] required for actual output power at net efficiency.
        """
        P_out    = self.power_mw(plr)                       # MW_e
        eta      = self.efficiency(plr, T_amb)
        eta_safe = np.where(np.asarray(eta) > 1e-9, eta, 1e-9)
        fuel_mw  = P_out / eta_safe                         # MW_th (coal LHV)
        return fuel_mw / self.LHV_coal                      # kg/s

    def syngas_rate_nm3s(self, plr, T_amb):
        """
        Syngas flow rate [Nm3/s] delivered to CCGT block after gasifier.
        Syngas energy = coal energy * cold-gas efficiency.
        """
        coal_kgs  = self.coal_rate_kgs(plr, T_amb)
        coal_mw   = coal_kgs * self.LHV_coal                # MW_th
        syngas_mw = coal_mw * self.cge                      # MW_th (syngas)
        return syngas_mw / self.syngas_lhv                  # Nm3/s

    def co2_rate_kgs(self, plr, T_amb):
        """CO2 emission rate [kg/s] without CCS."""
        return self.coal_rate_kgs(plr, T_amb) * self.CO2_per_kg_coal

    def co2_intensity_g_per_kwh(self, plr, T_amb):
        """
        CO2 emission intensity [g_CO2/kWh_e] without CCS.

        NOTE: At part load, IGCC efficiency degrades significantly due to
        gasifier turndown constraints, raising CO2/kWh above the rated-load
        benchmark of 700-800 g/kWh. This is correct physics.
        See test_model.py RATIONALE comment for CO2 test scoping.
        """
        P_out_kw = self.power_mw(plr) * 1e3                # kW_e
        co2_gs   = self.co2_rate_kgs(plr, T_amb) * 1e3     # g/s
        P_safe   = np.where(np.asarray(P_out_kw) > 1e-9, P_out_kw, 1e-9)
        return co2_gs / P_safe * 3600.0                     # g/kWh
