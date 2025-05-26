import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    @abstractmethod
    def forward(self, inputs):
        pass
    
    def predict(self, inputs):
        outputs = self.forward(inputs)
        return np.argmax(outputs, axis=1)
    
    def load_weights_from_keras(self, keras_model):
        # Match layers between Keras model and custom model
        if len(self.layers) != len(keras_model.layers):
            raise ValueError(
                f"Layer count mismatch: Custom model has {len(self.layers)} layers, "
                f"Keras model has {len(keras_model.layers)} layers"
            )
        
        # Load weights for each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'load_weights_from_keras'):
                layer.load_weights_from_keras(keras_model.layers[i])
    
    def evaluate(self, x_test, y_test, batch_size=32):
        # Predict in batches to avoid memory issues
        num_samples = len(x_test)
        y_pred = np.zeros(num_samples, dtype=np.int32)
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_x = x_test[i:end_idx]
            y_pred[i:end_idx] = self.predict(batch_x)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        # Calculate macro F1-score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, average='macro')
        
        return {
            'accuracy': accuracy,
            'macro_f1': f1
        }