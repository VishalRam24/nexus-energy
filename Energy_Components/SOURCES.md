# Sources & References — Priority Components

> Research sources, equations references, parameter sources, and library mappings
> for all 50 priority components. Updated as components are built.

---

## Tier 1 — Core Components

### EC001 — PEM Fuel Cell (PEMFC)
- **F1a Model:** Amphlett polarization curve model
- **Equations:** Amphlett et al. (1995). "Performance modeling of the Ballard Mark IV solid polymer electrolyte fuel cell." J. Electrochem. Soc., 142(1), 1-8. DOI: 10.1149/1.2043866
- **Alt equations:** Larminie & Dicks (2003). "Fuel Cell Systems Explained," 2nd ed. Wiley. ISBN: 978-0-470-84857-9
- **Library:** opem v1.4 (MIT license) — provides Amphlett, Chamberlin-Kim, Larminie-Dicks models
- **Parameters:** Ballard Mark V stack defaults from Amphlett (1995); also OPEM built-in defaults
- **F4 Status:** Complete (AI surrogate, MLP trained on IEEE PHM 2014 dataset)

### EC002 — Solid Oxide Fuel Cell (SOFC)
- **F1a Model:** Nernst + activation/ohmic/concentration loss polarization curve
- **Equations:** Chan et al. (2001). "A complete polarization model of a solid oxide fuel cell and its sensitivity to the change of cell component thickness." J. Power Sources, 93, 130-140. DOI: 10.1016/S0378-7753(00)00556-5
- **Alt equations:** Campanari & Iora (2004). "Definition and sensitivity analysis of a finite volume SOFC model for a tubular cell geometry." J. Power Sources, 132, 113-126
- **Library:** opem v1.4 (MIT) — limited SOFC support; primarily equations-based
- **Parameters:** Siemens-Westinghouse tubular SOFC from Chan (2001), T_op=1073K, 40-cell stack

### EC008 — PEM Electrolyser (PEMEL)
- **F1a Model:** V-I characteristic with activation + ohmic overpotentials
- **Equations:** Ulleberg (2003). "Modeling of advanced alkaline electrolyzers: a system simulation approach." Int. J. Hydrogen Energy, 28(1), 21-33. DOI: 10.1016/S0360-3199(02)00033-2
- **Alt equations:** Garcia-Valverde et al. (2012). "Simple PEM water electrolyser model and experimental validation." Int. J. Hydrogen Energy, 37(2), 1927-1938. DOI: 10.1016/j.ijhydene.2011.09.027
- **Library:** None mature enough; equations-based implementation
- **Parameters:** Proton OnSite/Nel M400 style, Nafion 117 membrane, T=80C, P=30bar

### EC018 — LFP Battery (Lithium Iron Phosphate)
- **F1a Model:** V = OCV(SOC) + I*R_internal (simple SOC-voltage model)
- **Equations:** Tremblay & Dessaint (2009). "Experimental Validation of a Battery Dynamic Model for EV Applications." World Electric Vehicle J., 3(2). DOI: 10.3390/wevj3020289
- **Alt equations:** Shepherd (1965) modified discharge model
- **Library:** PyBaMM v26.3 (BSD-3) — for parameter sets and validation data generation
- **Parameters:** A123 ANR26650 LFP cell: 3.3V nominal, 2.5Ah, Ri~30mOhm. PyBaMM Chen2020 parameter set.
- **OCV data:** PyBaMM built-in LFP OCV vs SOC lookup (Chen et al. 2020)

### EC019 — NMC Battery (Nickel Manganese Cobalt)
- **F1a Model:** V = OCV(SOC) + I*R_internal
- **Equations:** Tremblay & Dessaint (2009) — same framework as LFP with different parameters
- **Library:** PyBaMM v26.3 (BSD-3) — Chen2020 NMC parameter set
- **Parameters:** Samsung INR21700-50E (NMC811): 3.6V nominal, 5.0Ah. Chen et al. (2020) "Development of Experimental Techniques for Parameterization of Multi-scale Lithium-ion Battery Models." J. Electrochem. Soc., 167, 080534
- **OCV data:** PyBaMM built-in NMC OCV vs SOC lookup

### EC028 — Lead-Acid Battery
- **F1a Model:** V = OCV(SOC) + I*R_internal (modified Shepherd model)
- **Equations:** Copetti et al. (1993). "A general battery model for PV system simulation." Progress in Photovoltaics, 1(4), 283-292
- **Alt equations:** Manwell & McGowan (1993). "Lead acid battery storage model for hybrid energy systems." Solar Energy, 50(5), 399-405
- **Library:** None needed; well-established analytical equations
- **Parameters:** Generic 12V/100Ah flooded lead-acid: Vnom=12V, C_nom=100Ah, Ri~10mOhm

### EC036 — Vanadium Redox Flow Battery (VRFB)
- **F1a Model:** V = E_Nernst(SOC) - I*R_cell (Nernst + ohmic model)
- **Equations:** Blanc & Rufer (2010). "Understanding the Vanadium Redox Flow Batteries." In: Paths to Sustainable Energy, InTech. DOI: 10.5772/13338
- **Alt equations:** Shah et al. (2011). "Dynamic modelling of hydrogen evolution effects in VRFB." Electrochimica Acta, 56, 10614-10624
- **Library:** None; equations-based
- **Parameters:** 40-cell stack, 1.26V OCV per cell, membrane area 600cm2, electrolyte 1.6M V

### EC044 — Monocrystalline Silicon PV
- **F1a Model:** Single-diode model (5 parameters)
- **Equations:** De Soto et al. (2006). "Improvement and validation of a model for photovoltaic array performance." Solar Energy, 80(1), 78-88. DOI: 10.1016/j.solener.2005.06.010
- **Library:** pvlib v0.15 (BSD-3) — `pvlib.pvsystem.calcparams_desoto()` + `singlediode()`
- **Parameters:** Canadian Solar CS6K-280M (mono-Si): Pmax=280W, Voc=38.3V, Isc=9.39A at STC
- **Datasheet source:** CEC module database (built into pvlib)

### EC062 — HAWT Onshore Wind Turbine
- **F1a Model:** Power curve P(v) with air density correction
- **Equations:** IEC 61400-12-1 standard power curve methodology
- **Alt equations:** Manwell et al. (2009). "Wind Energy Explained," 2nd ed. Wiley. ISBN: 978-0-470-01500-1
- **Library:** windpowerlib v0.2.2 (MIT) — turbine database + power curve interpolation
- **Parameters:** Enercon E-126 (4.2MW) or Vestas V90-2.0MW from windpowerlib turbine database
- **Power curve data:** windpowerlib built-in turbine library

### EC065 — Offshore Fixed-Bottom Wind Turbine
- **F1a Model:** Power curve P(v) with marine air density correction
- **Equations:** Same IEC 61400-12-1 framework + offshore corrections
- **Library:** windpowerlib v0.2.2 (MIT)
- **Parameters:** Siemens SWT-3.6-120 (3.6MW offshore) or similar from database
- **Note:** F1a identical structure to EC062, different default parameters + hub height

### EC068 — Air-Source Heat Pump (ASHP)
- **F1a Model:** COP = f(T_source, T_sink) using Carnot fraction approach
- **Equations:** Staffell et al. (2012). "A review of domestic heat pumps." Energy Environ. Sci., 5, 9291-9306. DOI: 10.1039/C2EE22653G
- **Alt equations:** EN 14511 standard rating conditions + regression
- **Library:** CoolProp v7.2 (MIT) for refrigerant properties (R410A, R32)
- **Parameters:** Typical 10kW ASHP: COP_rated=3.5 at A7/W35, Carnot_fraction=0.45

### EC074 — Plate Heat Exchanger
- **F1a Model:** Effectiveness-NTU (e-NTU) method
- **Equations:** Incropera & DeWitt (2006). "Fundamentals of Heat and Mass Transfer," 6th ed. Wiley. Ch. 11
- **Alt equations:** Shah & Sekulic (2003). "Fundamentals of Heat Exchanger Design." Wiley
- **Library:** None needed; standard e-NTU correlations
- **Parameters:** Typical gasketed PHE: U=3000 W/m2K (water-water), A=2m2, counter-flow

### EC078 — Sensible Heat Storage (Hot Water Tank)
- **F1a Model:** Fully mixed (0D) energy balance: dT/dt = (Q_in - Q_out - Q_loss) / (m*cp)
- **Equations:** Duffie & Beckman (2013). "Solar Engineering of Thermal Processes," 4th ed. Wiley. Ch. 8
- **Library:** None needed; simple ODE
- **Parameters:** 500L tank, UA_loss=3 W/K, T_set=60C, cp_water=4186 J/kgK

### EC085 — Natural Gas Boiler
- **F1a Model:** eta(load) = eta_nom * f(PLR) — part-load efficiency curve
- **Equations:** EnergyPlus Engineering Reference (2023) — boiler performance curves
- **Alt equations:** Stafford (2009). "Condensing boiler models for system simulation." Building Serv. Eng. Res. Technol., 30(2)
- **Library:** None needed; polynomial regression
- **Parameters:** 50kW condensing gas boiler: eta_nom=0.95 (GCV), minimum PLR=0.1

### EC091 — Vapor Compression Chiller
- **F1a Model:** COP = f(T_evap, T_cond, PLR) using Gordon-Ng universal model
- **Equations:** Gordon & Ng (2000). "Cool Thermodynamics." Cambridge Int. Science Publishing. ISBN: 1898326908
- **Alt equations:** DOE-2 chiller model (regression-based)
- **Library:** CoolProp v7.2 (MIT) for refrigerant properties
- **Parameters:** 500kW centrifugal chiller, R134a, COP_rated=5.5 at ARI conditions

### EC101 — Combined Cycle Gas Turbine (CCGT)
- **F1a Model:** eta(load) from ISO rating + part-load curve
- **Equations:** Kehlhofer et al. (2009). "Combined-Cycle Gas & Steam Turbine Power Plants," 3rd ed. PennWell.
- **Alt equations:** Ganapathy (2003). "Industrial Boilers and Heat Recovery Steam Generators."
- **Library:** CoolProp for gas properties; equations-based
- **Parameters:** GE 9HA-class: rated eta=0.64 (LHV), P_rated=571MW, T_inlet=1600C

### EC109 — Simple Cycle Gas Turbine
- **F1a Model:** eta(load, T_amb) — efficiency vs part-load and ambient temperature
- **Equations:** Walsh & Fletcher (2004). "Gas Turbine Performance," 2nd ed. Blackwell. ISBN: 978-0-632-06434-2
- **Library:** CoolProp for air/gas properties
- **Parameters:** GE LM6000: P_rated=43MW, eta_rated=0.41 (LHV), T_inlet=1260C

### EC122 — Pumped Hydro Storage (PHS)
- **F1a Model:** P = eta * rho * g * Q * H (generation); P_pump = rho * g * Q * H / eta_pump
- **Equations:** Rehman et al. (2015). "Pumped hydro energy storage system: A technological review." Renewable and Sustainable Energy Reviews, 44, 586-598
- **Library:** None needed; fundamental hydraulic equations
- **Parameters:** eta_turbine=0.90, eta_pump=0.88, eta_motor/gen=0.97, H=300m, Q=50m3/s

### EC128 — Conventional Hydroelectric Dam
- **F1a Model:** P = eta * rho * g * Q * H (same structure as PHS generation side)
- **Equations:** Dixon & Hall (2014). "Fluid Mechanics and Thermodynamics of Turbomachinery," 7th ed. Butterworth-Heinemann
- **Library:** None needed
- **Parameters:** Francis turbine: eta=0.93, H=100m, Q=30m3/s, P_rated=27MW

### EC157 — Buck Converter (Step-Down)
- **F1a Model:** Vout = D*Vin, eta = f(D, I_load) with conduction + switching losses
- **Equations:** Erickson & Maksimovic (2020). "Fundamentals of Power Electronics," 3rd ed. Springer. ISBN: 978-3-030-43879-1
- **Library:** None needed; analytical loss model
- **Parameters:** 48V→12V, fsw=100kHz, MOSFET Rds_on=10mOhm, L=100uH, rated 10A

### EC158 — Boost Converter (Step-Up)
- **F1a Model:** Vout = Vin/(1-D), eta = f(D, I_load) with loss model
- **Equations:** Erickson & Maksimovic (2020) — same textbook as buck
- **Library:** None needed
- **Parameters:** 12V→48V, fsw=100kHz, MOSFET Rds_on=15mOhm, L=220uH, rated 5A

### EC164 — Three-Phase DC-AC Inverter
- **F1a Model:** Vac = m * Vdc / sqrt(2), eta = f(P_load) with switching + conduction losses
- **Equations:** Mohan et al. (2003). "Power Electronics: Converters, Applications, and Design," 3rd ed. Wiley
- **Alt equations:** Kolar et al. (2011). "Review of three-phase PWM AC-AC converter topologies." IEEE Trans.
- **Library:** None needed; analytical model
- **Parameters:** 100kW grid-tied inverter, Vdc=800V, fsw=10kHz, IGBT-based, eta_rated=0.98

### EC168 — MPPT Controller
- **F1a Model:** eta_mppt = f(irradiance, V_mpp) — tracking efficiency model
- **Equations:** Hohm & Ropp (2003). "Comparative study of maximum power point tracking algorithms." Progress in Photovoltaics, 11, 47-62. DOI: 10.1002/pip.459
- **Library:** pvlib (BSD-3) for reference MPP calculation
- **Parameters:** P&O algorithm: step_size=0.01V, eta_tracking=0.98 at >200W/m2, 0.92 at 50W/m2

### EC175 — Induction Motor / Generator
- **F1a Model:** Efficiency map eta(T, omega) from nameplate + part-load curve
- **Equations:** IEC 60034-30-1 efficiency classes (IE1-IE4) + scaling laws
- **Alt equations:** Boldea & Nasar (2010). "The Induction Machines Design Handbook," 2nd ed. CRC Press
- **Library:** None needed; efficiency map interpolation
- **Parameters:** 15kW IE3 induction motor: eta_rated=0.917, PF=0.86, 4-pole, 1460rpm

### EC176 — Permanent Magnet Synchronous Motor (PMSM)
- **F1a Model:** Efficiency map eta(T, omega) — loss separation model
- **Equations:** Gieras (2010). "Permanent Magnet Motor Technology," 3rd ed. CRC Press
- **Alt equations:** Hanselman (2006). "Brushless Permanent Magnet Motor Design."
- **Library:** None needed; efficiency map approach
- **Parameters:** 50kW automotive PMSM: eta_peak=0.96, base speed 3000rpm, max 12000rpm

---

## Tier 2 — High Priority Components

### EC009 — Alkaline Electrolyser (AEL)
- **F1a Model:** V-I characteristic (Ulleberg model)
- **Equations:** Ulleberg (2003). Same ref as EC008 — model originally developed for alkaline
- **Library:** None; equations-based
- **Parameters:** Atmospheric alkaline: T=80C, KOH 30wt%, electrode area 0.25m2, 20-cell stack

### EC010 — Solid Oxide Electrolyser Cell (SOEC)
- **F1a Model:** V-I characteristic with Nernst + ASR model
- **Equations:** Ni et al. (2007). "An electrochemical model of a solid oxide steam electrolyser for hydrogen production." Chemical Engineering & Technology, 29(6), 636-642
- **Library:** None; equations-based
- **Parameters:** T=800C, YSZ electrolyte 15um, electrode area 100cm2, ASR=0.3 Ohm.cm2

### EC020 — NCA Battery (Nickel Cobalt Aluminum)
- **F1a Model:** V = OCV(SOC) + I*R_internal
- **Equations:** Same Tremblay & Dessaint framework
- **Library:** PyBaMM (BSD-3) — NCA parameter set
- **Parameters:** Panasonic NCR18650B (NCA): 3.6V nominal, 3.35Ah

### EC031 — Sodium-Ion Battery
- **F1a Model:** V = OCV(SOC) + I*R_internal
- **Equations:** Tremblay & Dessaint framework adapted for Na-ion
- **Library:** None (PyBaMM has experimental Na-ion support)
- **Parameters:** CATL 1st-gen Na-ion: 3.1V nominal, ~160Wh/kg, Ri estimated ~50mOhm

### EC048 — Perovskite Solar Cell
- **F1a Model:** Single-diode model with perovskite-specific parameters
- **Equations:** De Soto (2006) single-diode framework + perovskite-specific parameters
- **Alt equations:** Miyano et al. (2016). "Lead Halide Perovskite Photovoltaic." J. Phys. Chem. Lett., 7, 2240-2245
- **Library:** pvlib (BSD-3) — single-diode solver; custom parameters
- **Parameters:** Lab-scale perovskite: eta=25.7%, Voc=1.18V, Jsc=25.1mA/cm2 (NREL chart 2024)

### EC054 — Parabolic Trough CSP
- **F1a Model:** Q_useful = DNI * A_aperture * eta_optical - Q_loss(T)
- **Equations:** Forristall (2003). "Heat Transfer Analysis and Modeling of a Parabolic Trough Solar Receiver." NREL/TP-550-34169
- **Library:** None; analytical heat loss model
- **Parameters:** Schott PTR70 receiver, Solargenix LS-3 collector: aperture 5.76m, eta_opt=0.75

### EC058 — Flat Plate Solar Collector
- **F1a Model:** Hottel-Whillier equation: Q_u = A * F_R * [S - U_L * (Ti - Ta)]
- **Equations:** Duffie & Beckman (2013). Ch. 6 — Hottel-Whillier-Bliss equation
- **Library:** None needed; classical analytical model
- **Parameters:** Typical glazed flat plate: F_R*tau_alpha=0.75, F_R*U_L=4.5 W/m2K, A=2m2

### EC069 — Ground-Source Heat Pump (GSHP)
- **F1a Model:** COP = f(T_ground, T_sink) — same Carnot fraction approach as ASHP
- **Equations:** Staffell et al. (2012) — same ref as EC068
- **Alt equations:** ASHRAE Handbook — HVAC Applications (2019), Ch. 34
- **Library:** CoolProp (MIT)
- **Parameters:** 15kW GSHP: COP_rated=4.5 at G0/W35, T_ground=10C, Carnot_fraction=0.50

### EC079 — Sensible Heat Storage (Molten Salt)
- **F1a Model:** Fully mixed energy balance (same structure as hot water tank, different fluid)
- **Equations:** Herrmann et al. (2004). "Two-tank molten salt storage for parabolic trough solar power plants." Energy, 29, 883-893
- **Library:** None needed; CoolProp or manual salt properties
- **Parameters:** Solar salt (60% NaNO3 + 40% KNO3): cp=1500 J/kgK, T_hot=565C, T_cold=290C, 1000 MWh_t

### EC080 — Phase Change Material (PCM) Storage
- **F1a Model:** Energy storage with latent heat: E = m * [cp_s*(Tm-T1) + L + cp_l*(T2-Tm)]
- **Equations:** Mehling & Cabeza (2008). "Heat and Cold Storage with PCM." Springer. ISBN: 978-3-540-68556-2
- **Library:** None needed
- **Parameters:** Paraffin RT42: Tm=42C, L=174 kJ/kg, cp=2.0 kJ/kgK, rho=880 kg/m3

### EC092 — Absorption Chiller
- **F1a Model:** COP = f(T_generator, T_condenser, T_evaporator) — characteristic equation
- **Equations:** Herold et al. (2016). "Absorption Chillers and Heat Pumps," 2nd ed. CRC Press. ISBN: 978-1-498-71434-5
- **Alt equations:** Gordon & Ng (2000) thermodynamic model
- **Library:** CoolProp for LiBr-water properties (limited)
- **Parameters:** Single-effect LiBr-H2O: Q_cool=500kW, COP_rated=0.7, T_gen=90C, T_cond=35C

### EC098 — Organic Rankine Cycle (ORC)
- **F1a Model:** eta = eta_Carnot * eta_internal — with part-load correction
- **Equations:** Quoilin et al. (2013). "Techno-economic survey of ORC systems." Renewable and Sustainable Energy Reviews, 22, 168-186
- **Library:** CoolProp (MIT) for organic fluid properties (R245fa, R1233zd)
- **Parameters:** 100kW ORC: T_hot=150C, T_cold=30C, R245fa, eta_expander=0.75, eta_pump=0.65

### EC104 — Gas Engine CHP
- **F1a Model:** eta_el(load) + eta_th(load) = f(PLR)
- **Equations:** ASUE (2011). "BHKW-Kenndaten" (CHP performance data) — German CHP association
- **Alt equations:** US EPA Catalog of CHP Technologies (2017)
- **Library:** None needed
- **Parameters:** 2MW gas engine CHP: eta_el=0.42, eta_th=0.43, PLR_min=0.5

### EC111 — Diesel Generator
- **F1a Model:** Fuel consumption = f(P_load) — Willans line model
- **Equations:** Tuffaha & Gravdahl (2014). "Modeling and Simulation of Diesel Engine." Based on classic Willans line approach
- **Alt equations:** US Army TM 5-811-6 (1996) — diesel fuel consumption curves
- **Library:** None needed
- **Parameters:** 500kW diesel genset: SFC_rated=210 g/kWh, eta_rated=0.40, PLR_min=0.25

### EC116 — Pressurized Water Reactor (PWR)
- **F1a Model:** P_thermal * eta_cycle = P_electric, with load-following constraints
- **Equations:** Todreas & Kazimi (2012). "Nuclear Systems," 2nd ed. CRC Press. Vol I, Ch. 2
- **Alt equations:** Lamarsh & Baratta (2017). "Introduction to Nuclear Engineering," 4th ed.
- **Library:** None needed; simplified steady-state model
- **Parameters:** Typical 1000MWe PWR: eta_thermal=0.33, T_inlet=292C, T_outlet=326C, P=155bar

### EC126 — Flywheel Energy Storage
- **F1a Model:** E = 0.5*J*omega^2, P = T*omega, with self-discharge
- **Equations:** Arani et al. (2017). "Review of Flywheel Energy Storage System Technologies." Energies, 10, 1361. DOI: 10.3390/en10091361
- **Library:** None needed
- **Parameters:** 100kW/25kWh steel flywheel: J=1000 kg.m2, omega_max=16000rpm, self-discharge=1%/h

### EC140 — Anaerobic Digester (Mesophilic)
- **F1a Model:** Biogas yield = f(VS_loading, HRT, T)
- **Equations:** Buswell equation for stoichiometric methane yield + empirical correction
- **Alt equations:** ADM1 simplified (Batstone et al. 2002, IWA)
- **Library:** None for F1a; simplified Buswell
- **Parameters:** Mesophilic: T=37C, HRT=20 days, VS loading 3 kg/m3/day, methane yield 0.35 m3/kgVS

### EC143 — Biomass Gasifier
- **F1a Model:** Syngas composition from equivalence ratio (ER) — equilibrium model
- **Equations:** Zainal et al. (2001). "Prediction of performance of a downdraft gasifier using equilibrium modeling." Energy Conversion and Management, 42(12), 1499-1515
- **Library:** None for F1a; equilibrium chemistry
- **Parameters:** Downdraft: ER=0.25, T_gasification=800C, wood chips (CH1.44O0.66)

### EC153 — Binary Cycle Geothermal Plant
- **F1a Model:** eta = eta_utilization * (1 - T_cold/T_hot) — exergy efficiency model
- **Equations:** DiPippo (2015). "Geothermal Power Plants," 4th ed. Butterworth-Heinemann
- **Library:** CoolProp for binary fluid (isobutane, isopentane)
- **Parameters:** 5MW binary: T_geo=150C, T_reject=25C, isobutane working fluid, eta_util=0.45

### EC193 — Methanation Reactor (Power-to-Gas)
- **F1a Model:** Conversion = f(T, P, H2/CO2 ratio) — Sabatier equilibrium
- **Equations:** Gao et al. (2012). "A thermodynamic analysis of methanation reactions." RSC Advances, 2, 2358-2368. DOI: 10.1039/C2RA00632D
- **Library:** None for F1a
- **Parameters:** Sabatier: T=300C, P=10bar, H2:CO2=4:1, catalyst Ni/Al2O3, conversion=0.95

### EC195 — Ammonia Synthesis (Haber-Bosch)
- **F1a Model:** Conversion = f(T, P) with energy consumption per ton NH3
- **Equations:** Appl (2011). "Ammonia, 2. Production Processes." In: Ullmann's Encyclopedia. DOI: 10.1002/14356007.o02_o11
- **Library:** None for F1a
- **Parameters:** T=450C, P=200bar, Fe catalyst, conversion_per_pass=0.15, E_specific=28 GJ/tNH3

### EC198 — Post-Combustion Capture (Amine Scrubbing)
- **F1a Model:** Capture_rate + specific_energy = f(lean_loading, L/G ratio)
- **Equations:** Abu-Zahra et al. (2007). "CO2 capture from power plants. Part I: A parametric study of the technical performance based on monoethanolamine." Int. J. GHG Control, 1, 37-46
- **Library:** None for F1a
- **Parameters:** MEA 30wt%: capture_rate=90%, reboiler_duty=3.6 GJ/tCO2, L/G=2.5

### EC201 — Direct Air Capture (DAC) — Solid Sorbent
- **F1a Model:** Energy consumption = f(T_regen, humidity, CO2_concentration)
- **Equations:** Fasihi et al. (2019). "Techno-economic assessment of CO2 direct air capture plants." J. Cleaner Production, 224, 957-980. DOI: 10.1016/j.jclepro.2019.03.086
- **Library:** None; simplified energy balance model
- **Parameters:** Climeworks-style: E_thermal=1500 kWh_t/tCO2, E_electric=250 kWh_e/tCO2, T_regen=100C

### EC209 — Reverse Osmosis (RO) Desalination
- **F1a Model:** SEC = f(recovery, salinity, P_feed) — specific energy consumption
- **Equations:** Elimelech & Phillip (2011). "The Future of Seawater Desalination." Science, 333, 712-717
- **Alt equations:** Zhu et al. (2009). "Minimization of energy consumption for a two-pass membrane desalination." J. Membrane Science, 339, 126-137
- **Library:** None for F1a
- **Parameters:** Seawater RO: salinity=35g/L, recovery=45%, SEC=3.5 kWh/m3, Dow FILMTEC SW30HR

### EC216 — Thermoelectric Generator (TEG)
- **F1a Model:** P = eta(T_hot, T_cold) * Q_hot, where eta uses ZT figure of merit
- **Equations:** Rowe (2006). "Thermoelectrics Handbook: Macro to Nano." CRC Press
- **Alt equations:** Snyder & Toberer (2008). "Complex thermoelectric materials." Nature Materials, 7, 105-114
- **Library:** None; analytical model
- **Parameters:** Bi2Te3 module: ZT=1.0, T_hot=200C, T_cold=30C, 40mm x 40mm module
