import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def simulate_and_plot(tflite_path, scaler_X_path, scaler_y_path):
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # Simulation Parameters
    total_hours = 800
    time_h = np.linspace(0, total_hours, total_hours)
    time_s = time_h * 3600
    
    # Profile A: Low Current (0.2A)
    # Profile B: High Current (1.2A)
    # At t=600h, we drop Profile B's current to 0.2A to see if the model 
    # magically "forgets" the history or if it stays degraded.
    
    current_A = np.full_like(time_h, 0.2)
    current_B = np.where(time_h < 600, 1.2, 0.2)
    
    dt_s = time_s[1] - time_s[0] if len(time_s) > 1 else 1.0
    cumulative_load_A = np.cumsum(current_A * dt_s)
    cumulative_load_B = np.cumsum(current_B * dt_s)
    
    results_A = []
    results_B = []
    
    for t, c_a, l_a, c_b, l_b in zip(time_s, current_A, cumulative_load_A, current_B, cumulative_load_B):
        # Infer A
        x_a = scaler_X.transform([[t, c_a, l_a]]).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], x_a)
        interpreter.invoke()
        y_a = scaler_y.inverse_transform(interpreter.get_tensor(output_details[0]['index']))[0]
        results_A.append(y_a)
        
        # Infer B
        x_b = scaler_X.transform([[t, c_b, l_b]]).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], x_b)
        interpreter.invoke()
        y_b = scaler_y.inverse_transform(interpreter.get_tensor(output_details[0]['index']))[0]
        results_B.append(y_b)
        
    results_A = np.array(results_A)
    results_B = np.array(results_B)
    
    # Plotting
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Current Density Set (A/cm²)', 'Predicted Voltage (V)', 'Predicted RUL (h)'))
    
    # Current
    fig.add_trace(go.Scatter(x=time_h, y=current_A, name='FC Low Load', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_h, y=current_B, name='FC High Load', line=dict(color='red')), row=1, col=1)
    
    # Voltage (Index 1)
    fig.add_trace(go.Scatter(x=time_h, y=results_A[:, 1], name='Voltage Low Load', line=dict(color='blue', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time_h, y=results_B[:, 1], name='Voltage High Load', line=dict(color='red', dash='dash')), row=2, col=1)
    
    # RUL (Index 4)
    fig.add_trace(go.Scatter(x=time_h, y=np.maximum(0, results_A[:, 4]), name='RUL Low Load', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_h, y=np.maximum(0, results_B[:, 4]), name='RUL High Load', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=900, title_text="Dual Fuel Cell Simulation Test")
    fig.write_html('simulation_report.html')
    print("HTML written to simulation_report.html. Please open it to evaluate.")

if __name__ == "__main__":
    simulate_and_plot(
        tflite_path='model_files/fuel_cell_mlp.tflite',
        scaler_X_path='model_files/scaler_X.save',
        scaler_y_path='model_files/scaler_y.save'
    )
