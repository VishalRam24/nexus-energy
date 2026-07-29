import os
import json
import numpy as np
import tensorflow as tf

def convert_json_to_tflite():
    print("Loading weights from model_files/mlp_weights.json...")
    with open('model_files/mlp_weights.json', 'r') as f:
        data = json.load(f)
        
    w1, b1 = np.array(data['w1'], dtype=np.float32), np.array(data['b1'], dtype=np.float32)
    w2, b2 = np.array(data['w2'], dtype=np.float32), np.array(data['b2'], dtype=np.float32)
    w3, b3 = np.array(data['w3'], dtype=np.float32), np.array(data['b3'], dtype=np.float32)
    
    print("Building equivalent Keras model...")
    # Input has 3 features now (time, current_density, cumulative_load)
    inputs = tf.keras.Input(shape=(3,))
    
    # Layer 1 (256 units, ReLU)
    x = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(inputs)
    
    # Layer 2 (128 units, ReLU)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
    
    # Output layer (5 units, Linear)
    outputs = tf.keras.layers.Dense(5, activation='linear', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("Setting weights...")
    model.get_layer('dense_1').set_weights([w1, b1])
    model.get_layer('dense_2').set_weights([w2, b2])
    model.get_layer('output').set_weights([w3, b3])
    
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
    convert_json_to_tflite()
