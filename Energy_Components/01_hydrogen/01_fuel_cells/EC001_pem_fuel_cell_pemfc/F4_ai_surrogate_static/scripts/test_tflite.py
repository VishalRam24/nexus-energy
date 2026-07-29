import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
import os

def test_tflite_model(tflite_path, scaler_X_path, scaler_y_path, data_path, num_samples=5):
    print(f"Loading TFLite model from {tflite_path}...")
    
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # Load test data and add cumulative load pre-sample
    df = pd.read_parquet(data_path)
    df.sort_values('time_s', inplace=True)
    df['dt_s'] = df['time_s'].diff().fillna(1.0)
    df['cumulative_load'] = (df['current_density_set'] * df['dt_s']).cumsum()

    sample = df.sample(num_samples, random_state=42)
    
    mean_current = df['current_density_set'].mean()
    nominal_limit = mean_current * 745.0 * 3600.0
    
    print("\n--- Running TFLite Inference Tests ---")
    for idx, row in sample.iterrows():
        # Extracted Real target based on our new logic
        frac = row['cumulative_load'] / nominal_limit
        target_rul = max(745.0 - (frac * 745.0), 0.0)
        
        # 1. Prepare and scale input
        X_raw = np.array([[row['time_s'], row['current_density_set'], row['cumulative_load']]], dtype=np.float32)
        X_scaled = scaler_X.transform(X_raw).astype(np.float32)
        
        # 2. Set input tensor
        interpreter.set_tensor(input_details[0]['index'], X_scaled)
        
        # 3. Invoke interpreter
        interpreter.invoke()
        
        # 4. Get and inverse scale output
        y_scaled_pred = interpreter.get_tensor(output_details[0]['index'])
        y_pred = scaler_y.inverse_transform(y_scaled_pred)[0]
        
        print(f"\nTime (s): {row['time_s']:.0f} | Current Density Set: {row['current_density_set']:.4f}")
        print(f"  Voltage   -> Pred: {y_pred[1]:.4f} | Actual: {row['voltage']:.4f}")
        print(f"  C_Density -> Pred: {y_pred[0]:.4f} | Actual: {row['current_density']:.4f}")
        print(f"  An_Flow   -> Pred: {y_pred[2]:.4f} | Actual: {row['anode_flow_nlpm']:.4f}")
        print(f"  Ca_Flow   -> Pred: {y_pred[3]:.4f} | Actual: {row['cathode_flow_nlpm']:.4f}")
        print(f"  RUL       -> Pred: {max(0, y_pred[4]):.2f} | Target Logic: {target_rul:.2f}")

if __name__ == "__main__":
    test_tflite_model(
        tflite_path='model_files/fuel_cell_mlp.tflite',
        scaler_X_path='model_files/scaler_X.save',
        scaler_y_path='model_files/scaler_y.save',
        data_path='data/training.parquet'
    )
