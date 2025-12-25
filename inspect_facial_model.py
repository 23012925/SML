#!/usr/bin/env python
"""Inspect the facialemotionmodel.h5 architecture"""
from tensorflow import keras
import os

model_path = 'models/facialemotionmodel.h5'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    model.summary()
else:
    print(f"Model not found: {model_path}")
