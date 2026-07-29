"""
EC155 -- Geothermal District Heating -- F1b Resource Decline & Part-Load Model

Builds on F1a (heat extraction model) by adding:
  - Geothermal resource temperature decline (aquifer drawdown)
  - Seasonal part-load variation (demand-side PLR)
  - Network supply temperature sensitivity (low-T vs high-T network)
  - Pipe distribution heat loss temperature dependence
  - Reinjection well back-pressure effects (flow-rate limitation at high production)

Aquifer Temperature Decline:
    Direct-use systems extract heat from a hydrothermal aquifer.
    Temperature decline depends on reservoir volume, recharge rate, and extraction rate.
    Model: T_source(t) = T_source_0 - decline_rate * t
    Linear approximation valid for t << reservoir depletion time.
    Rybach & Mongillo (2006): low-enthalpy aquifers 0.1-0.5 degC/yr with balanced reinjection.

Part-Load (seasonal demand):
    District heating load varies seasonally. At part-load, flow is throttled.
    Efficiency-like factor: f_PLR = 1 - k_standby * (1 - PLR) for partial load
    (standby losses increase at low load: pumps run at partial efficiency, HX approach
    temperature increases, distribution losses are a larger fraction of delivered heat).

Network Temperature Sensitivity:
    Lower network supply temperature T_net improves HX effectiveness.
    If T_net approaches T_source, the LMTD decreases, requiring larger HX or accepting
    lower heat transfer. Model: f_T_net = (T_source - T_net) / (T_source - T_net_design)
    Bounded: [0.3, 1.5].

Reinjection Back-Pressure:
    High extraction rates can cause aquifer pressure drawdown, reducing well yield.
    f_flow = 1 - k_pressure * max(0, m_dot/m_dot_design - 1)
    DiPippo (2015); Lund & Toth (2021).

References:
    Lund, J.W. & Toth, A.N. (2021). Direct Utilization of Geothermal Energy 2020
        Worldwide Review. Geothermics, 90, 101915.
    Rybach, L. & Mongillo, M. (2006). Geothermal sustainability: A review with
        identified research needs. GRC Transactions, 30.
    Rybach, L. (2003). Geothermal energy: sustainability and the environment.
        Geothermics, 32(4-6), 463-470.
"""

import numpy as np


class GeothermalDistrictHeatingF1b:
    """
    Geothermal district heating with resource decline, seasonal part-load,
    network temperature sensitivity, and reinjection back-pressure.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.cp_geo          = u["cp_geo"]["value"]                        # J/(kg*K)
        self.eta_hx          = u["heat_transfer_efficiency"]["value"]      # -
        self.f_dist_loss_0   = u["distribution_losses_base"]["value"]      # - base pipe loss
        self.k_dist_T        = u["dist_loss_T_coeff"]["value"]             # extra loss per degC above 60
        self.f_pump          = u["pump_power_fraction"]["value"]           # -
        self.T_source_design = u["T_source_design"]["value"]               # degC
        self.T_return_design = u["T_return_design"]["value"]               # degC
        self.T_net_supply_design = u["T_net_supply_design"]["value"]       # degC district supply T
        self.m_dot_design    = u["m_dot_geo_design"]["value"]              # kg/s
        self.decline_rate    = u["decline_rate_degC_per_yr"]["value"]      # degC/yr aquifer T drop
        self.k_standby       = u["k_standby"]["value"]                     # extra loss fraction at PLR<1
        self.k_pressure      = u["k_pressure_backpressure"]["value"]       # fractional flow reduction
        self.PLR_min         = u["PLR_min"]["value"]

    # ------------------------------------------------------------------
    # Resource temperature decline
    # ------------------------------------------------------------------
    def source_temperature(self, T_source_0: float, years_operation: float) -> float:
        """
        Geothermal source temperature after aquifer drawdown.
        T_source(t) = T_source_0 - decline_rate * t
        Bounded below at T_return_design + 10 (must still have useful delta-T).
        Rybach & Mongillo (2006): 0.1-0.5 degC/yr for well-managed low-enthalpy fields.
        """
        T_0 = float(T_source_0)
        t   = max(0.0, float(years_operation))
        T_t = T_0 - self.decline_rate * t
        return float(max(T_t, self.T_return_design + 5.0))

    # ------------------------------------------------------------------
    # Network temperature factor
    # ------------------------------------------------------------------
    def network_T_factor(self, T_source_c: float, T_net_supply_c: float) -> float:
        """
        HX effectiveness correction for network supply temperature.
        If T_net is close to T_source, less heat can be transferred.
        f_T_net = (T_source - T_net) / (T_source_design - T_net_supply_design)
        Bounded: [0.2, 1.5].
        """
        dT_actual = float(T_source_c) - float(T_net_supply_c)
        dT_design = self.T_source_design - self.T_net_supply_design
        if dT_design <= 0:
            return 1.0
        return float(np.clip(dT_actual / dT_design, 0.2, 1.5))

    # ------------------------------------------------------------------
    # Distribution loss (temperature-dependent)
    # ------------------------------------------------------------------
    def distribution_loss_factor(self, T_net_supply_c: float) -> float:
        """
        Distribution pipe heat losses increase with network temperature.
        f_dist = f_dist_0 + k_dist_T * max(0, T_net - 60)
        Higher T -> larger pipe-to-ambient gradient -> more loss.
        """
        dT = max(0.0, float(T_net_supply_c) - 60.0)
        f = self.f_dist_loss_0 + self.k_dist_T * dT
        return float(np.clip(f, self.f_dist_loss_0, 0.25))

    # ------------------------------------------------------------------
    # Reinjection back-pressure flow limit
    # ------------------------------------------------------------------
    def flow_factor(self, m_dot: float) -> float:
        """
        Flow reduction factor if extraction rate exceeds design.
        f_flow = 1 - k_pressure * max(0, m_dot/m_dot_design - 1)
        Bounded: [0.5, 1.0].
        """
        ratio = float(m_dot) / self.m_dot_design
        f = 1.0 - self.k_pressure * max(0.0, ratio - 1.0)
        return float(np.clip(f, 0.5, 1.0))

    # ------------------------------------------------------------------
    # Part-load efficiency factor
    # ------------------------------------------------------------------
    def part_load_factor(self, PLR: float) -> float:
        """
        Part-load efficiency factor accounting for standby losses at low load.
        f_PL = 1 - k_standby * (1 - PLR)
        At PLR=1: factor=1.0; at PLR=0.5: factor = 1 - k_standby*0.5.
        """
        PLR = np.clip(float(PLR), self.PLR_min, 1.0)
        f   = 1.0 - self.k_standby * (1.0 - PLR)
        return float(np.clip(f, 0.5, 1.0))

    # ------------------------------------------------------------------
    # Main predict
    # ------------------------------------------------------------------
    def predict(self, T_source_degC: float, m_dot_geo_kg_s: float,
                T_return_degC: float, T_net_supply_degC: float,
                PLR: float = 1.0, years_operation: float = 0.0) -> dict:
        """
        Compute district heating system performance.

        Args:
            T_source_degC:      Geothermal supply temperature at wellhead [degC]
            m_dot_geo_kg_s:     Geothermal fluid mass flow rate [kg/s]
            T_return_degC:      Return temperature to reinjection well [degC]
            T_net_supply_degC:  District heating network supply temperature [degC]
            PLR:                Part-load ratio [PLR_min - 1.0]
            years_operation:    Years since start of production [years]

        Returns:
            dict with:
                heat_delivered_kw     : Heat delivered to end users [kW]
                heat_extracted_kw     : Heat extracted from geothermal fluid [kW]
                pump_power_kw         : Circulation pump power [kW]
                system_cop            : Q_delivered / W_pump [-]
                T_source_effective    : Actual source T after aquifer decline [degC]
                resource_factor       : T_source(t)/T_source(0) [-]
                distribution_loss_frac: Fractional distribution pipe loss [-]
        """
        PLR    = np.clip(float(PLR), self.PLR_min, 1.0)
        T_src0 = float(T_source_degC)
        m_dot  = float(m_dot_geo_kg_s)
        T_ret  = float(T_return_degC)
        T_net  = float(T_net_supply_degC)
        years  = max(0.0, float(years_operation))

        # Effective source temperature after aquifer decline
        T_src_eff = self.source_temperature(T_src0, years)

        # Resource factor
        dT_0   = T_src0  - T_ret
        dT_eff = T_src_eff - T_ret
        f_res  = float(np.where(dT_0 > 0, dT_eff / dT_0, 0.0))
        f_res  = float(np.clip(f_res, 0.0, 1.0))

        # Flow factor (back-pressure)
        f_flow = self.flow_factor(m_dot)
        m_eff  = m_dot * f_flow

        # Part-load factor
        f_pl = self.part_load_factor(PLR)

        # Network T factor
        f_T_net = self.network_T_factor(T_src_eff, T_net)

        # Distribution loss factor
        f_dist = self.distribution_loss_factor(T_net)

        # Heat extracted from geothermal fluid
        dT = max(0.0, T_src_eff - T_ret)
        Q_ext = m_eff * self.cp_geo * dT / 1000.0  # kW

        # Heat transferred through HX (apply network-T and part-load factors)
        Q_trans = Q_ext * self.eta_hx * f_T_net * f_pl

        # Heat delivered after pipe losses and PLR throttling
        Q_del = Q_trans * (1.0 - f_dist) * PLR

        # Pump power
        W_pump = self.f_pump * Q_ext

        # System COP
        cop = Q_del / W_pump if W_pump > 0 else 0.0

        return {
            "heat_delivered_kw":     float(max(0.0, Q_del)),
            "heat_extracted_kw":     float(Q_ext),
            "pump_power_kw":         float(W_pump),
            "system_cop":            float(np.clip(cop, 0.0, 200.0)),
            "T_source_effective":    float(T_src_eff),
            "resource_factor":       float(f_res),
            "distribution_loss_frac": float(f_dist),
        }
