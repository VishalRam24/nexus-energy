# Energy Components — Master Taxonomy

> **223 components** across **15 sectors**, each with up to **7 fidelity levels**.
> Use the **EC ID** to reference any component. Folder structure mirrors this tree exactly.

> **Build status (2026-06-08, N_Comp_Phase 10):** F0 + F1 + F2 are **COMPLETE for all 223
> components**. Remaining: F3 (distributed PDE, research tier) and F4/F5/F6 (AI surrogates).

---

## Fidelity Levels (apply to every component)

| Code | Name | Description |
|------|------|-------------|
| **F0** | Empirical | Lookup tables, efficiency maps, manufacturer data curves |
| **F1** | Semi-Empirical | Simplified analytical equations with fitted parameters |
| **F2** | Physics-Based (Lumped) | First-principles lumped-parameter ODE models |
| **F3** | Physics-Based (Distributed) | PDE-based spatially-resolved high-fidelity models |
| **F4** | AI Surrogate (Static) | MLP/ANN black-box for steady-state prediction (GPU-ready) |
| **F5** | AI Surrogate (Dynamic) | LSTM/Transformer for transient time-series (GPU-ready) |
| **F6** | PINN | Physics-Informed Neural Network — hybrid physics + AI (GPU-ready) |

---

## Model Status Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: | Complete — trained, tested, verified |
| :construction: | In progress |
| :black_square_button: | Not started |

---

## 01 — Hydrogen & Fuel Cells

### 01.01 Fuel Cells

| ID | Component | Status |
|----|-----------|--------|
| **EC001** | PEM Fuel Cell (PEMFC) | :white_check_mark: F1a, F4 |
| **EC002** | Solid Oxide Fuel Cell (SOFC) | :white_check_mark: F1a |
| **EC003** | Alkaline Fuel Cell (AFC) | :black_square_button: |
| **EC004** | Phosphoric Acid Fuel Cell (PAFC) | :black_square_button: |
| **EC005** | Molten Carbonate Fuel Cell (MCFC) | :black_square_button: |
| **EC006** | Direct Methanol Fuel Cell (DMFC) | :black_square_button: |
| **EC007** | Reversible Fuel Cell (RFC) | :black_square_button: |

### 01.02 Electrolysers

| ID | Component | Status |
|----|-----------|--------|
| **EC008** | PEM Electrolyser (PEMEL) | :white_check_mark: F1a |
| **EC009** | Alkaline Electrolyser (AEL) | :white_check_mark: F1a |
| **EC010** | Solid Oxide Electrolyser (SOEC) | :white_check_mark: F1a |
| **EC011** | Anion Exchange Membrane Electrolyser (AEM) | :black_square_button: |

### 01.03 Hydrogen Storage

| ID | Component | Status |
|----|-----------|--------|
| **EC012** | Compressed Gas H2 Storage | :black_square_button: |
| **EC013** | Liquid Hydrogen Storage | :black_square_button: |
| **EC014** | Metal Hydride H2 Storage | :black_square_button: |
| **EC015** | Chemical H2 Storage (LOHC / Ammonia) | :black_square_button: |

### 01.04 Hydrogen Infrastructure

| ID | Component | Status |
|----|-----------|--------|
| **EC016** | Hydrogen Compressor | :black_square_button: |
| **EC017** | Hydrogen Purifier (PSA) | :black_square_button: |

---

## 02 — Batteries & Electrochemical Storage

### 02.01 Lithium-Ion Batteries

| ID | Component | Status |
|----|-----------|--------|
| **EC018** | LFP (Lithium Iron Phosphate) | :white_check_mark: F1a |
| **EC019** | NMC (Nickel Manganese Cobalt) | :white_check_mark: F1a |
| **EC020** | NCA (Nickel Cobalt Aluminum) | :white_check_mark: F1a |
| **EC021** | LTO (Lithium Titanate) | :black_square_button: |
| **EC022** | LCO (Lithium Cobalt Oxide) | :black_square_button: |
| **EC023** | LMO (Lithium Manganese Oxide) | :black_square_button: |
| **EC024** | Silicon Anode Li-ion | :black_square_button: |
| **EC025** | Lithium-Sulfur (Li-S) | :black_square_button: |
| **EC026** | Lithium-Air (Li-Air) | :black_square_button: |
| **EC027** | Solid-State Lithium | :black_square_button: |

### 02.02 Non-Lithium Batteries

| ID | Component | Status |
|----|-----------|--------|
| **EC028** | Lead-Acid Battery | :white_check_mark: F1a |
| **EC029** | Nickel-Metal Hydride (NiMH) | :black_square_button: |
| **EC030** | Nickel-Cadmium (NiCd) | :black_square_button: |
| **EC031** | Sodium-Ion Battery | :white_check_mark: F1a |
| **EC032** | Zinc-Air Battery | :black_square_button: |
| **EC033** | Iron-Air Battery | :black_square_button: |
| **EC034** | Aluminum-Ion Battery | :black_square_button: |
| **EC035** | Sodium-Sulfur (NaS) Battery | :black_square_button: |

### 02.03 Flow Batteries

| ID | Component | Status |
|----|-----------|--------|
| **EC036** | Vanadium Redox Flow Battery (VRFB) | :white_check_mark: F1a |
| **EC037** | Zinc-Bromine Flow Battery | :black_square_button: |
| **EC038** | Iron-Chromium Flow Battery | :black_square_button: |
| **EC039** | Organic Flow Battery | :black_square_button: |
| **EC040** | Hydrogen-Bromine Flow Battery | :black_square_button: |

### 02.04 Supercapacitors

| ID | Component | Status |
|----|-----------|--------|
| **EC041** | Electric Double-Layer Capacitor (EDLC) | :black_square_button: |
| **EC042** | Pseudocapacitor | :black_square_button: |
| **EC043** | Hybrid Supercapacitor | :black_square_button: |

---

## 03 — Solar Energy

### 03.01 Photovoltaic Cells & Modules

| ID | Component | Status |
|----|-----------|--------|
| **EC044** | Monocrystalline Silicon PV | :white_check_mark: F1a |
| **EC045** | Polycrystalline Silicon PV | :black_square_button: |
| **EC046** | Thin-Film CdTe PV | :black_square_button: |
| **EC047** | Thin-Film CIGS PV | :black_square_button: |
| **EC048** | Perovskite Solar Cell | :white_check_mark: F1a |
| **EC049** | Multi-Junction Concentrator PV | :black_square_button: |
| **EC050** | Organic Photovoltaic (OPV) | :black_square_button: |
| **EC051** | Dye-Sensitized Solar Cell (DSSC) | :black_square_button: |
| **EC052** | Bifacial PV Module | :black_square_button: |
| **EC053** | Thermophotovoltaic (TPV) | :black_square_button: |

### 03.02 Concentrated Solar Power (CSP)

| ID | Component | Status |
|----|-----------|--------|
| **EC054** | Parabolic Trough CSP | :white_check_mark: F1a |
| **EC055** | Solar Tower (Central Receiver) | :black_square_button: |
| **EC056** | Linear Fresnel CSP | :black_square_button: |
| **EC057** | Stirling Dish CSP | :black_square_button: |

### 03.03 Solar Thermal Collectors

| ID | Component | Status |
|----|-----------|--------|
| **EC058** | Flat Plate Solar Collector | :white_check_mark: F1a |
| **EC059** | Evacuated Tube Solar Collector | :black_square_button: |
| **EC060** | Solar Pond | :black_square_button: |
| **EC061** | Unglazed Solar Collector (Pool Heating) | :black_square_button: |

---

## 04 — Wind Energy

### 04.01 Onshore Wind

| ID | Component | Status |
|----|-----------|--------|
| **EC062** | Horizontal Axis Wind Turbine (HAWT) — Onshore | :white_check_mark: F1a |
| **EC063** | Vertical Axis Wind Turbine (VAWT) | :black_square_button: |
| **EC064** | Small / Micro Wind Turbine | :black_square_button: |

### 04.02 Offshore Wind

| ID | Component | Status |
|----|-----------|--------|
| **EC065** | Offshore Fixed-Bottom Wind Turbine | :white_check_mark: F1a |
| **EC066** | Offshore Floating Wind Turbine | :black_square_button: |

### 04.03 Emerging Wind Technologies

| ID | Component | Status |
|----|-----------|--------|
| **EC067** | Airborne Wind Energy (AWE) | :black_square_button: |

---

## 05 — Thermal Energy Systems

### 05.01 Heat Pumps

| ID | Component | Status |
|----|-----------|--------|
| **EC068** | Air-Source Heat Pump (ASHP) | :white_check_mark: F1a |
| **EC069** | Ground-Source Heat Pump (GSHP) | :white_check_mark: F1a |
| **EC070** | Water-Source Heat Pump | :black_square_button: |
| **EC071** | Absorption Heat Pump | :black_square_button: |
| **EC072** | CO2 Transcritical Heat Pump | :black_square_button: |

### 05.02 Heat Exchangers

| ID | Component | Status |
|----|-----------|--------|
| **EC073** | Shell and Tube Heat Exchanger | :black_square_button: |
| **EC074** | Plate Heat Exchanger | :white_check_mark: F1a |
| **EC075** | Finned Tube Heat Exchanger | :black_square_button: |
| **EC076** | Regenerative Heat Exchanger | :black_square_button: |
| **EC077** | Microchannel Heat Exchanger | :black_square_button: |

### 05.03 Thermal Energy Storage

| ID | Component | Status |
|----|-----------|--------|
| **EC078** | Sensible Heat Storage (Hot Water Tank) | :white_check_mark: F1a |
| **EC079** | Sensible Heat Storage (Molten Salt) | :white_check_mark: F1a |
| **EC080** | Phase Change Material (PCM) Storage | :white_check_mark: F1a |
| **EC081** | Thermochemical Energy Storage | :black_square_button: |
| **EC082** | Ice Thermal Storage | :black_square_button: |
| **EC083** | Borehole Thermal Energy Storage (BTES) | :black_square_button: |
| **EC084** | Aquifer Thermal Energy Storage (ATES) | :black_square_button: |

### 05.04 Boilers & Heaters

| ID | Component | Status |
|----|-----------|--------|
| **EC085** | Natural Gas Boiler | :white_check_mark: F1a |
| **EC086** | Electric Boiler / Resistance Heater | :black_square_button: |
| **EC087** | Biomass Boiler | :black_square_button: |
| **EC088** | Oil-Fired Boiler | :black_square_button: |
| **EC089** | Hydrogen Boiler | :black_square_button: |
| **EC090** | Solar Water Heater (Combi System) | :black_square_button: |

### 05.05 Cooling & Refrigeration

| ID | Component | Status |
|----|-----------|--------|
| **EC091** | Vapor Compression Chiller | :white_check_mark: F1a |
| **EC092** | Absorption Chiller | :white_check_mark: F1a |
| **EC093** | Adsorption Chiller | :black_square_button: |
| **EC094** | Evaporative Cooler | :black_square_button: |
| **EC095** | Thermoelectric Cooler (Peltier) | :black_square_button: |
| **EC096** | Magnetic Refrigeration | :black_square_button: |

### 05.06 Heat Engines & Thermodynamic Cycles

| ID | Component | Status |
|----|-----------|--------|
| **EC097** | Rankine Cycle (Steam Turbine) | :black_square_button: |
| **EC098** | Organic Rankine Cycle (ORC) | :white_check_mark: F1a |
| **EC099** | Stirling Engine | :black_square_button: |
| **EC100** | Brayton Cycle (Gas Turbine) | :black_square_button: |
| **EC101** | Combined Cycle Gas Turbine (CCGT) | :white_check_mark: F1a |
| **EC102** | Kalina Cycle | :black_square_button: |
| **EC103** | Supercritical CO2 Brayton Cycle | :black_square_button: |

### 05.07 Combined Heat & Power (CHP)

| ID | Component | Status |
|----|-----------|--------|
| **EC104** | Gas Engine CHP | :white_check_mark: F1a |
| **EC105** | Gas Turbine CHP | :black_square_button: |
| **EC106** | Fuel Cell CHP (SOFC-based) | :black_square_button: |
| **EC107** | Micro-CHP (Stirling-based) | :black_square_button: |
| **EC108** | Steam Turbine CHP | :black_square_button: |

---

## 06 — Conventional Power Generation

### 06.01 Gas & Combustion

| ID | Component | Status |
|----|-----------|--------|
| **EC109** | Simple Cycle Gas Turbine | :white_check_mark: F1a |
| **EC110** | Reciprocating Gas Engine | :black_square_button: |
| **EC111** | Diesel Generator | :white_check_mark: F1a |
| **EC112** | Micro Gas Turbine | :black_square_button: |

### 06.02 Coal & Steam

| ID | Component | Status |
|----|-----------|--------|
| **EC113** | Subcritical Pulverized Coal Plant | :black_square_button: |
| **EC114** | Supercritical / Ultra-Supercritical Coal | :black_square_button: |
| **EC115** | Integrated Gasification Combined Cycle (IGCC) | :black_square_button: |

### 06.03 Nuclear

| ID | Component | Status |
|----|-----------|--------|
| **EC116** | Pressurized Water Reactor (PWR) | :white_check_mark: F1a |
| **EC117** | Boiling Water Reactor (BWR) | :black_square_button: |
| **EC118** | Small Modular Reactor (SMR) | :black_square_button: |
| **EC119** | Molten Salt Reactor (MSR) | :black_square_button: |
| **EC120** | Fast Breeder Reactor (FBR) | :black_square_button: |
| **EC121** | High-Temperature Gas Reactor (HTGR) | :black_square_button: |

---

## 07 — Mechanical Energy Storage

### 07.01 Pumped & Compressed

| ID | Component | Status |
|----|-----------|--------|
| **EC122** | Pumped Hydro Storage (PHS) | :white_check_mark: F1a |
| **EC123** | Compressed Air Energy Storage (CAES) | :black_square_button: |
| **EC124** | Liquid Air Energy Storage (LAES / CES) | :black_square_button: |
| **EC125** | Adiabatic CAES (A-CAES) | :black_square_button: |

### 07.02 Kinetic & Gravitational

| ID | Component | Status |
|----|-----------|--------|
| **EC126** | Flywheel Energy Storage | :white_check_mark: F1a |
| **EC127** | Gravity Energy Storage | :black_square_button: |

---

## 08 — Hydropower & Marine Energy

### 08.01 Hydropower

| ID | Component | Status |
|----|-----------|--------|
| **EC128** | Conventional Hydroelectric Dam | :white_check_mark: F1a |
| **EC129** | Run-of-River Hydropower | :black_square_button: |
| **EC130** | Small / Micro Hydropower | :black_square_button: |

### 08.02 Tidal Energy

| ID | Component | Status |
|----|-----------|--------|
| **EC131** | Tidal Barrage | :black_square_button: |
| **EC132** | Tidal Stream Turbine | :black_square_button: |
| **EC133** | Tidal Lagoon | :black_square_button: |

### 08.03 Wave Energy

| ID | Component | Status |
|----|-----------|--------|
| **EC134** | Oscillating Water Column (OWC) | :black_square_button: |
| **EC135** | Point Absorber WEC | :black_square_button: |
| **EC136** | Overtopping Device WEC | :black_square_button: |
| **EC137** | Oscillating Body / Attenuator WEC | :black_square_button: |

### 08.04 Ocean Thermal

| ID | Component | Status |
|----|-----------|--------|
| **EC138** | Ocean Thermal Energy Conversion (OTEC) | :black_square_button: |
| **EC139** | Salinity Gradient (Blue Energy / PRO) | :black_square_button: |

---

## 09 — Biomass & Bioenergy

### 09.01 Biogas & Anaerobic Digestion

| ID | Component | Status |
|----|-----------|--------|
| **EC140** | Anaerobic Digester (Mesophilic) | :white_check_mark: F1a |
| **EC141** | Anaerobic Digester (Thermophilic) | :black_square_button: |
| **EC142** | Biogas Upgrading (Biomethane) | :black_square_button: |

### 09.02 Thermochemical Conversion

| ID | Component | Status |
|----|-----------|--------|
| **EC143** | Biomass Gasifier | :white_check_mark: F1a, F2a |
| **EC144** | Biomass Combustion CHP | :black_square_button: |
| **EC145** | Pyrolysis Reactor | :black_square_button: |
| **EC146** | Torrefaction Reactor | :black_square_button: |
| **EC147** | Hydrothermal Liquefaction (HTL) | :black_square_button: |

### 09.03 Biofuel Production

| ID | Component | Status |
|----|-----------|--------|
| **EC148** | Bioethanol Fermentation | :black_square_button: |
| **EC149** | Biodiesel Transesterification | :black_square_button: |
| **EC150** | Fischer-Tropsch Synthesis (BtL) | :black_square_button: |

---

## 10 — Geothermal Energy

### 10.01 Geothermal Power Plants

| ID | Component | Status |
|----|-----------|--------|
| **EC151** | Dry Steam Geothermal Plant | :black_square_button: |
| **EC152** | Flash Steam Geothermal Plant | :black_square_button: |
| **EC153** | Binary Cycle Geothermal Plant | :white_check_mark: F1a |
| **EC154** | Enhanced Geothermal System (EGS) | :black_square_button: |

### 10.02 Direct Use

| ID | Component | Status |
|----|-----------|--------|
| **EC155** | Geothermal District Heating | :black_square_button: |
| **EC156** | Geothermal Heat Pump (GHP) | :black_square_button: |

---

## 11 — Power Electronics & Electrical

### 11.01 DC-DC Converters

| ID | Component | Status |
|----|-----------|--------|
| **EC157** | Buck Converter (Step-Down) | :white_check_mark: F1a |
| **EC158** | Boost Converter (Step-Up) | :white_check_mark: F1a |
| **EC159** | Buck-Boost Converter | :black_square_button: |
| **EC160** | Isolated DC-DC (Flyback / Forward) | :black_square_button: |
| **EC161** | Dual Active Bridge (DAB) | :black_square_button: |
| **EC162** | Resonant LLC Converter | :black_square_button: |

### 11.02 Inverters & Rectifiers

| ID | Component | Status |
|----|-----------|--------|
| **EC163** | Single-Phase DC-AC Inverter | :black_square_button: |
| **EC164** | Three-Phase DC-AC Inverter | :white_check_mark: F1a |
| **EC165** | Multilevel Inverter | :black_square_button: |
| **EC166** | AC-DC Rectifier (Diode Bridge) | :black_square_button: |
| **EC167** | Active Front-End Rectifier (PFC) | :black_square_button: |
| **EC168** | MPPT Controller | :white_check_mark: F1a |

### 11.03 AC-AC Converters

| ID | Component | Status |
|----|-----------|--------|
| **EC169** | Variable Frequency Drive (VFD) | :black_square_button: |
| **EC170** | Solid-State Transformer (SST) | :black_square_button: |
| **EC171** | Cycloconverter | :black_square_button: |

### 11.04 Transformers

| ID | Component | Status |
|----|-----------|--------|
| **EC172** | Power Transformer (Grid-Scale) | :black_square_button: |
| **EC173** | Distribution Transformer | :black_square_button: |
| **EC174** | Instrument Transformer (CT/PT) | :black_square_button: |

### 11.05 Electric Motors & Generators

| ID | Component | Status |
|----|-----------|--------|
| **EC175** | Induction Motor / Generator | :white_check_mark: F1a |
| **EC176** | Permanent Magnet Synchronous Motor (PMSM) | :white_check_mark: F1a |
| **EC177** | Brushless DC Motor (BLDC) | :black_square_button: |
| **EC178** | Switched Reluctance Motor (SRM) | :black_square_button: |
| **EC179** | Wound Rotor Synchronous Generator | :black_square_button: |
| **EC180** | Doubly-Fed Induction Generator (DFIG) | :black_square_button: |

### 11.06 Grid & Transmission Components

| ID | Component | Status |
|----|-----------|--------|
| **EC181** | Transmission Line Model | :black_square_button: |
| **EC182** | Distribution Line Model | :black_square_button: |
| **EC183** | Circuit Breaker | :black_square_button: |
| **EC184** | Power Factor Correction (PFC) Unit | :black_square_button: |
| **EC185** | Static VAR Compensator (SVC) | :black_square_button: |
| **EC186** | STATCOM | :black_square_button: |
| **EC187** | HVDC Converter Station | :black_square_button: |
| **EC188** | Superconducting Magnetic Energy Storage (SMES) | :black_square_button: |

---

## 12 — Gas & Fuel Infrastructure

### 12.01 Natural Gas Infrastructure

| ID | Component | Status |
|----|-----------|--------|
| **EC189** | Natural Gas Pipeline | :black_square_button: |
| **EC190** | LNG Regasification Terminal | :black_square_button: |
| **EC191** | Gas Compressor Station | :black_square_button: |
| **EC192** | Gas Pressure Regulator | :black_square_button: |

### 12.02 Synthetic Fuels & Power-to-X

| ID | Component | Status |
|----|-----------|--------|
| **EC193** | Methanation Reactor (Power-to-Gas) | :white_check_mark: F1a, F2a |
| **EC194** | Methanol Synthesis Reactor | :black_square_button: |
| **EC195** | Ammonia Synthesis (Haber-Bosch) | :white_check_mark: F1a, F2a |
| **EC196** | Synthetic Jet Fuel (Power-to-Liquid) | :black_square_button: |
| **EC197** | DME Synthesis Reactor | :black_square_button: |

---

## 13 — Carbon Capture, Utilization & Storage

### 13.01 CO2 Capture Technologies

| ID | Component | Status |
|----|-----------|--------|
| **EC198** | Post-Combustion Capture (Amine Scrubbing) | :white_check_mark: F1a |
| **EC199** | Pre-Combustion Capture (WGS + Separation) | :black_square_button: |
| **EC200** | Oxy-Fuel Combustion | :black_square_button: |
| **EC201** | Direct Air Capture (DAC) — Solid Sorbent | :white_check_mark: F1a |
| **EC202** | Direct Air Capture (DAC) — Liquid Solvent | :black_square_button: |
| **EC203** | Membrane-Based CO2 Separation | :black_square_button: |
| **EC204** | Calcium Looping | :black_square_button: |

### 13.02 CO2 Utilization

| ID | Component | Status |
|----|-----------|--------|
| **EC205** | CO2 Electrolyzer (CO2 to CO/Fuels) | :black_square_button: |
| **EC206** | CO2 Mineralization | :black_square_button: |

### 13.03 CO2 Transport & Storage

| ID | Component | Status |
|----|-----------|--------|
| **EC207** | CO2 Compression & Pipeline | :black_square_button: |
| **EC208** | CO2 Geological Sequestration | :black_square_button: |

---

## 14 — Desalination & Water-Energy Nexus

### 14.01 Membrane Processes

| ID | Component | Status |
|----|-----------|--------|
| **EC209** | Reverse Osmosis (RO) | :white_check_mark: F1a |
| **EC210** | Electrodialysis (ED) | :black_square_button: |
| **EC211** | Forward Osmosis (FO) | :black_square_button: |

### 14.02 Thermal Desalination

| ID | Component | Status |
|----|-----------|--------|
| **EC212** | Multi-Stage Flash Distillation (MSF) | :black_square_button: |
| **EC213** | Multi-Effect Distillation (MED) | :black_square_button: |
| **EC214** | Mechanical Vapor Compression (MVC) | :black_square_button: |
| **EC215** | Solar Still / Humidification-Dehumidification | :black_square_button: |

---

## 15 — Thermoelectric & Emerging Direct Conversion

### 15.01 Thermoelectric Devices

| ID | Component | Status |
|----|-----------|--------|
| **EC216** | Thermoelectric Generator (TEG) | :white_check_mark: F1a |
| **EC217** | Thermoelectric Cooler (TEC) | :black_square_button: |

### 15.02 Emerging Direct Conversion

| ID | Component | Status |
|----|-----------|--------|
| **EC218** | Thermionic Converter | :black_square_button: |
| **EC219** | Piezoelectric Energy Harvester | :black_square_button: |
| **EC220** | Triboelectric Nanogenerator (TENG) | :black_square_button: |
| **EC221** | Magnetohydrodynamic (MHD) Generator | :black_square_button: |
| **EC222** | Betavoltaic Cell | :black_square_button: |
| **EC223** | Radioisotope Thermoelectric Generator (RTG) | :black_square_button: |

---

## Quick Reference — ID Ranges by Sector

| Sector | ID Range | Count |
|--------|----------|-------|
| 01 Hydrogen & Fuel Cells | EC001–EC017 | 17 |
| 02 Batteries & Electrochemical | EC018–EC043 | 26 |
| 03 Solar Energy | EC044–EC061 | 18 |
| 04 Wind Energy | EC062–EC067 | 6 |
| 05 Thermal Energy Systems | EC068–EC108 | 41 |
| 06 Conventional Power Gen | EC109–EC121 | 13 |
| 07 Mechanical Energy Storage | EC122–EC127 | 6 |
| 08 Hydropower & Marine | EC128–EC139 | 12 |
| 09 Biomass & Bioenergy | EC140–EC150 | 11 |
| 10 Geothermal Energy | EC151–EC156 | 6 |
| 11 Power Electronics & Electrical | EC157–EC188 | 32 |
| 12 Gas & Fuel Infrastructure | EC189–EC197 | 9 |
| 13 Carbon Capture & Storage | EC198–EC208 | 11 |
| 14 Desalination | EC209–EC215 | 7 |
| 15 Thermoelectric & Emerging | EC216–EC223 | 8 |
| **TOTAL** | | **223** |
