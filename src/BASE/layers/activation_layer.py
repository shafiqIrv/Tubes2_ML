import numpy as np

# Base Class Activation Layer
class ActivationLayer:
    def forward(self, inputs):
        pass
    
    def backward(self, grad_output):
        pass

class ReLU(ActivationLayer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, grad_output):
        return grad_output * (self.inputs > 0)

class Sigmoid(ActivationLayer):
    def forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-np.clip(inputs, -50, 50)))
        return self.outputs
    
    def backward(self, grad_output):
        return grad_output * self.outputs * (1 - self.outputs)

class Tanh(ActivationLayer):
    def forward(self, inputs):
        self.outputs = np.tanh(inputs)
        return self.outputs
    
    def backward(self, grad_output):
        return grad_output * (1 - self.outputs**2)

class Softmax(ActivationLayer):
    def forward(self, inputs):
        exp_shifted = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self.outputs
    
    def backward(self, grad_output):
        # TODO
        return grad_output