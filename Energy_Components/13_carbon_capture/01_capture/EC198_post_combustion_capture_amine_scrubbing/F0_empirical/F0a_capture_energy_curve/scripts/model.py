"""F0a empirical capture-rate & energy-penalty model for EC198.

Post-combustion CO2 capture by 30 wt% MEA amine scrubbing.
Empirical lookup: reboiler duty (GJ/tCO2) is read from a tabulated
capture-rate breakpoint curve via 1-D np.interp, plus a fixed electricity
specific energy. Captured CO2 mass follows from flue-gas flow, CO2 fraction
and the design capture rate.

Data source: Abu-Zahra, M.R.M. et al. (2007), Int. J. Greenhouse Gas
Control 1(1), 37-46 (reboiler duty ~3.2 GJ/tCO2 at 90% capture, 30 wt% MEA).
NumPy only.
"""
import numpy as np


class CaptureEnergyCurve:
    def __init__(self, params):
        u = params["unit"]
        self.q_base = u["q_base"]["value"]                 # GJ/tCO2 at 90%
        self.cap_design = u["capture_rate_design"]["value"]
        self.e_elec = u["electricity_specific"]["value"]   # GJ/tCO2
        self.co2_frac_design = u["CO2_fraction_design"]["value"]
        self.mw_co2 = u["MW_CO2"]["value"]
        self.mw_air = u["MW_air"]["value"]
        # Empirical reboiler-duty breakpoints vs capture rate.
        # Duty rises sharply approaching 95% capture (mass-transfer pinch).
        self.cap_pts = np.array(params["reboiler_curve"]["capture_rate"])
        self.q_pts = np.array(params["reboiler_curve"]["reboiler_GJ_tCO2"])

    def reboiler_duty(self, capture_rate):
        cr = np.asarray(capture_rate, dtype=float)
        return np.interp(cr, self.cap_pts, self.q_pts)

    def predict(self, inputs):
        flue = float(inputs.get("flue_gas_rate", 100.0))      # kg/s
        co2_frac = float(inputs.get("co2_fraction", self.co2_frac_design))
        cr = float(inputs.get("capture_rate", self.cap_design))
        # CO2 mass fraction of flue gas from mole fraction
        w_co2 = co2_frac * self.mw_co2 / (
            co2_frac * self.mw_co2 + (1.0 - co2_frac) * self.mw_air)
        co2_in = flue * w_co2                                  # kg/s CO2 in
        co2_captured = co2_in * cr                            # kg/s captured
        q_th = float(self.reboiler_duty(cr))                  # GJ/tCO2
        # power (MW): GJ/tCO2 * (kg/s /1000 t/kg) * 1000 MJ/GJ = GJ/t * t/s *1e3
        captured_t_s = co2_captured / 1000.0
        thermal_MW = q_th * captured_t_s * 1000.0
        elec_MW = self.e_elec * captured_t_s * 1000.0
        return {
            "co2_captured_kg_s": co2_captured,
            "capture_rate": cr,
            "reboiler_duty_GJ_tCO2": q_th,
            "electricity_GJ_tCO2": self.e_elec,
            "thermal_power_MW": thermal_MW,
            "electric_power_MW": elec_MW,
            "total_specific_energy_GJ_tCO2": q_th + self.e_elec,
        }
