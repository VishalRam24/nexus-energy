import json
import numpy as np
import joblib

class FuelCellModel:
    def __init__(self, weights_path='model_files/mlp_weights.json', 
                 scaler_X_path='model_files/scaler_X.save',
                 scaler_y_path='model_files/scaler_y.save'):
        
        # Load weights
        with open(weights_path, 'r') as f:
            data = json.load(f)
            
        self.w1 = np.array(data['w1'])
        self.b1 = np.array(data['b1'])
        self.w2 = np.array(data['w2'])
        self.b2 = np.array(data['b2'])
        self.w3 = np.array(data['w3'])
        self.b3 = np.array(data['b3'])
        
        # Load scalers
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
    def _relu(self, x):
        return np.maximum(0, x)
        
    def predict(self, time_s, current_density_set):
        """
        Predicts fuel cell parameters based on time and set current density.
        
        Args:
            time_s (float): Time in seconds
            current_density_set (float): The current density set point
            
        Returns:
            dict: Containing current_density, voltage, anode_flow_nlpm, cathode_flow_nlpm, RUL
        """
        
        # 1. Prepare input
        X = np.array([[time_s, current_density_set]])
        
        # 2. Scale input
        X_scaled = self.scaler_X.transform(X)
        
        # 3. Forward pass
        a1 = self._relu(np.dot(X_scaled, self.w1) + self.b1)
        a2 = self._relu(np.dot(a1, self.w2) + self.b2)
        y_scaled = np.dot(a2, self.w3) + self.b3
        
        # 4. Inverse scale output
        y_pred = self.scaler_y.inverse_transform(y_scaled)[0]
        
        # 5. Format output
        return {
            'current_density': float(y_pred[0]),
            'voltage': float(y_pred[1]),
            'anode_flow_nlpm': float(y_pred[2]),
            'cathode_flow_nlpm': float(y_pred[3]),
            'RUL': max(0.0, float(y_pred[4])) # Ensure RUL isn't conceptually negative
        }

if __name__ == "__main__":
    import pandas as pd
    
    # Test inference against some random rows
    print("Testing ultra-lightweight inference engine...")
    model = FuelCellModel()
    
    df = pd.read_parquet('data/training.parquet')
    sample = df.sample(5, random_state=100)
    
    for idx, row in sample.iterrows():
        print(f"\n--- Instance (time_h: {row['time_h']:.2f}, I_set: {row['current_density_set']:.2f}) ---")
        pred = model.predict(row['time_s'], row['current_density_set'])
        
        print("Predictions vs Actuals:")
        print(f"Voltage:   Pred {pred['voltage']:.4f}, Actual {row['voltage']:.4f}")
        print(f"C_Density: Pred {pred['current_density']:.4f}, Actual {row['current_density']:.4f}")
        print(f"An_Flow:   Pred {pred['anode_flow_nlpm']:.4f}, Actual {row['anode_flow_nlpm']:.4f}")
        print(f"Ca_Flow:   Pred {pred['cathode_flow_nlpm']:.4f}, Actual {row['cathode_flow_nlpm']:.4f}")
        actual_rul = max(745.0 - row['time_h'], 0)
        print(f"RUL:       Pred {pred['RUL']:.4f}, Actual {actual_rul:.4f}")
