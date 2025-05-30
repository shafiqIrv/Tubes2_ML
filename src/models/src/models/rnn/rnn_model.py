import numpy as np
from src.models.src.models.base_model.base_model import BaseModel


class RNNModel:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, inputs):
        """
        Perform forward propagation through all layers
        
        Args:
            inputs: numpy array of shape (batch_size, sequence_length)
                  or (batch_size, sequence_length, input_dim) if embedding is done externally
                  
        Returns:
            numpy array of shape (batch_size, num_classes)
        """
        x = inputs
        
        for i, layer in enumerate(self.layers):
            # Print shape before each layer for debugging
            print(f"Layer {i} input shape: {x.shape}")
            x = layer.forward(x)
            
        # Print final output shape
        print(f"Model output shape: {x.shape}")
        return x
    
    def predict(self, inputs):
        """
        Predict class labels for input data
        
        Args:
            inputs: numpy array of shape (batch_size, sequence_length)
                  or (batch_size, sequence_length, input_dim)
                  
        Returns:
            numpy array of shape (batch_size,) containing class predictions
        """
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=1)
    
    def load_weights_from_keras(self, keras_model):
        """Load weights from a Keras model"""
        # Match layers between Keras model and custom model
        if len(self.layers) != len(keras_model.layers):
            print(f"Warning: Layer count mismatch: Custom model has {len(self.layers)} layers, "
                  f"Keras model has {len(keras_model.layers)} layers")
        
        # Load weights for each layer
        for i, layer in enumerate(self.layers):
            if i < len(keras_model.layers):
                if hasattr(layer, 'load_weights_from_keras'):
                    try:
                        layer.load_weights_from_keras(keras_model.layers[i])
                    except Exception as e:
                        print(f"Error loading weights for layer {i}: {e}")
            else:
                print(f"Warning: No corresponding Keras layer for scratch layer {i}")