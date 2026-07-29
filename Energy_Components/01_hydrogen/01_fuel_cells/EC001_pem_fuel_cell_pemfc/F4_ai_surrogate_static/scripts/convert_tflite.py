import os
import joblib
import numpy as np
import tensorflow as tf

def convert_sklearn_to_tflite():
    print("Loading SKLearn Model...")
    mlp = joblib.load('model_files/fuel_cell_mlp.pkl')
    
    # MLPRegressor has attributes: coefs_, intercepts_
    print("Building equivalent Keras model...")
    # Input has 2 features
    inputs = tf.keras.Input(shape=(2,))
    
    # Layer 1 (256 units, ReLU)
    x = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(inputs)
    
    # Layer 2 (128 units, ReLU)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
    
    # Output layer (5 units, Linear)
    outputs = tf.keras.layers.Dense(5, activation='linear', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("Setting weights...")
    model.get_layer('dense_1').set_weights([mlp.coefs_[0], mlp.intercepts_[0]])
    model.get_layer('dense_2').set_weights([mlp.coefs_[1], mlp.intercepts_[1]])
    model.get_layer('output').set_weights([mlp.coefs_[2], mlp.intercepts_[2]])
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations to make it as small as possible
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = 'model_files/fuel_cell_mlp.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"TFLite model saved to {tflite_path}. Size: {size_mb:.4f} MB")

if __name__ == "__main__":
    convert_sklearn_to_tflite()
