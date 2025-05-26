import numpy as np

class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, inputs, training=True):
        self.training = training
        
        if not training:
            return inputs
        
        # Generate dropout mask (1: keep, 0: drop)
        self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
        
        # Apply mask to inputs
        return inputs * self.mask
    
    def backward(self, grad_output):
        if not self.training:
            return grad_output
        
        # Apply same mask to gradients
        return grad_output * self.mask
    
    def load_weights_from_keras(self, keras_layer):
        self.dropout_rate = keras_layer.rate