import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"Loaded {df.shape[0]} rows.")

    # Calculate RUL
    # t_EoL is 745h according to the paper
    print("Computing target RUL...")
    df['RUL'] = np.maximum(745.0 - df['time_h'], 0)

    # Inputs and Outputs
    input_cols = ['time_s', 'current_density_set']
    output_cols = ['current_density', 'voltage', 'anode_flow_nlpm', 'cathode_flow_nlpm', 'RUL']

    X = df[input_cols].values
    y = df[output_cols].values

    # Clean extreme anomalies if needed (optional)
    # Using simple StandardScaler
    
    print("Scaling features...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    print("Saving scalers...")
    os.makedirs('model_files', exist_ok=True)
    joblib.dump(scaler_X, 'model_files/scaler_X.save')
    joblib.dump(scaler_y, 'model_files/scaler_y.save')
    
    # Split the data
    print("Splitting train/val sets...")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)
    
    return X_train, X_val, y_train, y_val, scaler_X, scaler_y

def build_model(input_dim, output_dim):
    # MLP with 4 hidden layers
    # ~221k parameters which comfortably meets the <50 MB requirement (<1 MB).
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation='swish')(inputs)
    x = tf.keras.layers.Dense(256, activation='swish')(x)
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse',
                  metrics=['mae'])
    
    model.summary()
    return model

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Force CPU to avoid weird hanging on some Apple Silicon
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    start_time = time.time()
    data_path = "data/training.parquet"
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = load_and_preprocess_data(data_path)

    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = build_model(input_dim, output_dim)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, monitor='val_loss')
    ]
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, # Start with generous epochs, early stopping will catch it
        batch_size=8192, # Large batch size to utilize GPU/CPU efficiently on 3.6M row dataset
        callbacks=callbacks
    )
    
    model.save('model_files/fuel_cell_mlp.keras')
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
