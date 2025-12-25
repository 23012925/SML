"""Script to inspect .h5 model files"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras

print("="*60)
print("MODEL 1: emotiondetector.h5")
print("="*60)
try:
    m1 = keras.models.load_model('models/emotiondetector.h5', compile=False)
    print(f"Input shape: {m1.input_shape}")
    print(f"Output shape: {m1.output_shape}")
    print(f"Layers: {len(m1.layers)}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("MODEL 2: facialemotionmodel (1).h5")
print("="*60)
try:
    m2 = keras.models.load_model('models/facialemotionmodel (1).h5', compile=False)
    print(f"Input shape: {m2.input_shape}")
    print(f"Output shape: {m2.output_shape}")
    print(f"Layers: {len(m2.layers)}")
except Exception as e:
    print(f"Error: {e}")
