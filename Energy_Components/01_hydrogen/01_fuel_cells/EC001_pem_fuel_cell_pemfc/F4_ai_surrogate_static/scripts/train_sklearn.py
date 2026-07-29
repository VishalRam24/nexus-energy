import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"Loaded {df.shape[0]} rows.")

    # Calculate temporal integration of current (cumulative load history)
    # This teaches the AI model to "remember" history by passing the current integral state as an input.
    print("Calculating cumulative load history...")
    df.sort_values(by='time_s', inplace=True)
    df['dt_s'] = df['time_s'].diff().fillna(1.0)
    df['cumulative_load'] = (df['current_density_set'] * df['dt_s']).cumsum()

    # Calculate nominal lifetime load limit
    mean_current = df['current_density_set'].mean()
    nominal_life_limit_load = mean_current * 745.0 * 3600.0 # total Amp-seconds over 745h at nominal capacity

    # True State-Based RUL Calculation
    # RUL strictly decreases based on cumulative payload digested.
    print("Computing cumulative target RUL...")
    fraction_used = df['cumulative_load'] / nominal_life_limit_load
    df['RUL'] = np.maximum(745.0 - (fraction_used * 745.0), 0.0)

    # Note: dataset is massive. Scikit-Learn MLP can be slow. 
    # To meet the speed constraint natively, we will sample 500,000 random points.
    df_sampled = df.sample(n=500000, random_state=42)
    print(f"Sampled {df_sampled.shape[0]} rows for fast SKLearn training.")

    input_cols = ['time_s', 'current_density_set', 'cumulative_load']
    output_cols = ['current_density', 'voltage', 'anode_flow_nlpm', 'cathode_flow_nlpm', 'RUL']

    X = df_sampled[input_cols].values
    y = df_sampled[output_cols].values
    
    print("Scaling features...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    os.makedirs('model_files', exist_ok=True)
    joblib.dump(scaler_X, 'model_files/scaler_X.save')
    joblib.dump(scaler_y, 'model_files/scaler_y.save')
    
    print("Splitting train/val sets...")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
    
    return X_train, X_val, y_train, y_val, scaler_X, scaler_y

if __name__ == "__main__":
    start_time = time.time()
    data_path = "data/training.parquet"
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = load_and_preprocess_data(data_path)
    
    print("Initializing MLPRegressor...")
    # Hidden layers: 256, 128
    # Using ReLU for fast convergence
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        batch_size=2048,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=50,
        verbose=True,
        random_state=42
    )
    
    print("Starting training...")
    model.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    preds_scaled = model.predict(X_val)
    
    mse_scaled = mean_squared_error(y_val, preds_scaled)
    mae_scaled = mean_absolute_error(y_val, preds_scaled)
    print(f"Validation Scaled MSE: {mse_scaled:.6f}")
    print(f"Validation Scaled MAE: {mae_scaled:.6f}")
    
    # Save the model
    joblib.dump(model, 'model_files/fuel_cell_mlp.pkl')
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    print("Model saved to model_files/fuel_cell_mlp.pkl.")
