"""F0a empirical model for EC199 Pre-Combustion Capture (WGS + Selexol).

Empirical lookup of WGS CO-conversion vs reactor temperature (Gaussian-like
breakpoint curve peaking at the low-T shift optimum) combined with a fixed
Selexol separation efficiency. Total specific energy = separation + CO2
compression (GJ/tCO2). Overall capture = X_WGS * eta_sep.

Source: IEAGHG (2014) Rpt 2012/8; Kunze & Spliethoff (2012) Appl. Energy 94:109.
NumPy only.
"""
import numpy as np


class ConversionEnergyCurve:
    def __init__(self, params):
        u = params["unit"]
        self.eta_sep = u["eta_sep_design"]["value"]
        self.e_sep = u["E_sep_base"]["value"]
        self.e_comp = u["E_compression"]["value"]
        self.T_pts = np.array(params["wgs_curve"]["T_C"])
        self.X_pts = np.array(params["wgs_curve"]["X_WGS"])

    def wgs_conversion(self, T_C):
        return np.interp(np.asarray(T_C, dtype=float), self.T_pts, self.X_pts)

    def predict(self, inputs):
        T = float(inputs.get("T_WGS_C", 250.0))
        co_in = float(inputs.get("co_flow_kg_s", 10.0))  # CO into shift, kg/s
        X = float(self.wgs_conversion(T))
        eta = float(inputs.get("eta_sep", self.eta_sep))
        # CO2 produced from CO: 28.01 g CO -> 44.01 g CO2
        co2_shifted = co_in * X * (44.01 / 28.01)
        co2_captured = co2_shifted * eta
        sec = self.e_sep + self.e_comp
        return {
            "wgs_conversion": X,
            "separation_efficiency": eta,
            "overall_capture_rate": X * eta,
            "co2_captured_kg_s": co2_captured,
            "separation_GJ_tCO2": self.e_sep,
            "compression_GJ_tCO2": self.e_comp,
            "total_specific_energy_GJ_tCO2": sec,
        }
