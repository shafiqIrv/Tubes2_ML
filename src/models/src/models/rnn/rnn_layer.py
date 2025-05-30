import numpy as np
import tensorflow as tf


def tanh(x):
    return np.tanh(x)


class RNNLayer:
    def __init__(self, input_dim, hidden_dim, bidirectional=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.initialize_weights(direction="forward")

        if bidirectional:
            self.initialize_weights(direction="backward")

    # Init weight RNN
    def initialize_weights(self, direction="forward"):
        prefix = direction + "_" if self.bidirectional else ""

        # Input weights
        setattr(
            self,
            f"{prefix}W",
            np.random.randn(self.input_dim, self.hidden_dim) * 0.01,
        )

        # Recurrent weights
        setattr(
            self,
            f"{prefix}U",
            np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01,
        )

        # Bias
        setattr(self, f"{prefix}b", np.zeros(self.hidden_dim))

    # Forward propa 1 time step
    def forward_step(self, x_t, h_prev, direction="forward"):
    
        prefix = direction + "_" if self.bidirectional else ""

        # Get the weights for this direction
        W = getattr(self, f"{prefix}W")
        U = getattr(self, f"{prefix}U")
        b = getattr(self, f"{prefix}b")

        # h_t = tanh(W * x_t + U * h_prev + b)
        h_t = tanh(np.dot(x_t, W) + np.dot(h_prev, U) + b)

        return h_t

    # Forward Propa RNN
    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape

        # Initialize hidden state
        h_forward = np.zeros((batch_size, self.hidden_dim))

        # forward direction
        for t in range(sequence_length):
            h_forward = self.forward_step(
                inputs[:, t, :], h_forward, direction="forward"
            )

        if not self.bidirectional:
            return h_forward

        # For bidirectional backward direction
        h_backward = np.zeros((batch_size, self.hidden_dim))

        for t in range(sequence_length - 1, -1, -1):
            h_backward = self.forward_step(
                inputs[:, t, :], h_backward, direction="backward"
            )

        # Merge
        return np.concatenate([h_forward, h_backward], axis=1)

    # Load weight from Keras
    def load_weights_from_keras(self, keras_layer):
        if isinstance(keras_layer, tf.keras.layers.SimpleRNN):
            # Keras weights order: [kernel, recurrent_kernel, bias]
            weights = keras_layer.get_weights()
            
            self.W = weights[0]  # Input weights 
            self.U = weights[1]  # Recurrent weights 
            self.b = weights[2]  # Biase
            
        elif isinstance(keras_layer, tf.keras.layers.Bidirectional):
            # Extract weights from forward and backward layers
            forward_weights = keras_layer.forward_layer.get_weights()
            backward_weights = keras_layer.backward_layer.get_weights()
            
            # Forward direction
            self.forward_W = forward_weights[0]
            self.forward_U = forward_weights[1]
            self.forward_b = forward_weights[2]
            
            # Backward direction
            self.backward_W = backward_weights[0]
            self.backward_U = backward_weights[1]
            self.backward_b = backward_weights[2]
        else:
            raise ValueError(f"Unsupported layer type for weight loading: {type(keras_layer)}")