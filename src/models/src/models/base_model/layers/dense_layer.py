import numpy as np

# Fully Connected Layer
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        
        # Init weights and biases
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros(output_dim)
        
        self.input = None
        self.output = None
    
    def forward(self, inputs):
        self.input = inputs
        
        self.output = np.dot(inputs, self.weights) + self.biases
        
        if self.activation:
            self.output = self.activation.forward(self.output)
            
        return self.output
    
    def load_weights_from_keras(self, keras_layer):
        self.weights = keras_layer.get_weights()[0]
        self.biases = keras_layer.get_weights()[1]
    
    def backward(self, grad_output):
        if self.activation:
            grad_output = self.activation.backward(grad_output)
        

        grad_weights = np.dot(self.input.T, grad_output)
        
        grad_biases = np.sum(grad_output, axis=0)

        grad_input = np.dot(grad_output, self.weights.T)
        
        self.grad_weights = grad_weights
        self.grad_biases = grad_biases
        
        return grad_input