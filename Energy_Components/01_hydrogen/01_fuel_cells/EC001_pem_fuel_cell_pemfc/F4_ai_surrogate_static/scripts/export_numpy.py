import joblib
import numpy as np
import os
import json

def export_mlp_to_numpy():
    print("Loading SKLearn Model...")
    mlp = joblib.load('model_files/fuel_cell_mlp.pkl')
    
    # Extract weights and biases
    weights = mlp.coefs_
    biases = mlp.intercepts_
    
    model_data = {
        "w1": weights[0].tolist(),
        "b1": biases[0].tolist(),
        "w2": weights[1].tolist(),
        "b2": biases[1].tolist(),
        "w3": weights[2].tolist(),
        "b3": biases[2].tolist()
    }
    
    os.makedirs('model_files', exist_ok=True)
    with open('model_files/mlp_weights.json', 'w') as f:
        json.dump(model_data, f)
        
    size_mb = os.path.getsize('model_files/mlp_weights.json') / (1024 * 1024)
    print(f"Exported pure weights to model_files/mlp_weights.json. Size: {size_mb:.4f} MB")

if __name__ == "__main__":
    export_mlp_to_numpy()
