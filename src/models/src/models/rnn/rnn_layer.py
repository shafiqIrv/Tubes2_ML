import numpy as np
import tensorflow as tf


def tanh(x):
    return np.tanh(x)


class RNNLayer:
    def __init__(self, input_dim, hidden_dim, bidirectional=False, return_sequences=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences

        # Initialize weights for forward direction
        self.initialize_weights(direction="forward")

        # Initialize weights for backward direction if bidirectional
        if bidirectional:
            self.initialize_weights(direction="backward")

    def initialize_weights(self, direction="forward"):
        """Initialize weights for the RNN layer"""
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

    def forward_step(self, x_t, h_prev, direction="forward"):
        """Perform one step of forward propagation"""
        prefix = direction + "_" if self.bidirectional else ""

        # Get the weights for this direction
        W = getattr(self, f"{prefix}W")
        U = getattr(self, f"{prefix}U")
        b = getattr(self, f"{prefix}b")

        # h_t = tanh(W * x_t + U * h_prev + b)
        h_t = np.tanh(np.dot(x_t, W) + np.dot(h_prev, U) + b)

        return h_t

    def forward(self, inputs):
        """
        Perform forward propagation for the RNN layer

        Args:
            inputs: numpy array of shape (batch_size, sequence_length, input_dim)
                   or (batch_size, sequence_length) if coming directly from embedding

        Returns:
            numpy array of shape:
            - If return_sequences=False:
                (batch_size, hidden_dim) if bidirectional=False
                (batch_size, hidden_dim*2) if bidirectional=True
            - If return_sequences=True:
                (batch_size, sequence_length, hidden_dim) if bidirectional=False
                (batch_size, sequence_length, hidden_dim*2) if bidirectional=True
        """
        # Handle different input shapes
        if len(inputs.shape) == 2:
            # If input is 2D, reshape it to 3D by adding a feature dimension
            print(f"Warning: RNN layer received 2D input with shape {inputs.shape}. "
                  f"Expected 3D input. Reshaping to add feature dimension.")
            batch_size, sequence_length = inputs.shape
            inputs = inputs.reshape(batch_size, sequence_length, 1)
        
        batch_size, sequence_length, input_features = inputs.shape
        
        # Check if input dimension matches expected dimension
        if input_features != self.input_dim:
            print(f"Warning: Input feature dimension {input_features} doesn't match "
                  f"expected dimension {self.input_dim}. This may cause issues.")

        # Initialize arrays to store all hidden states if returning sequences
        if self.return_sequences:
            h_forward_seq = np.zeros((batch_size, sequence_length, self.hidden_dim))
            if self.bidirectional:
                h_backward_seq = np.zeros((batch_size, sequence_length, self.hidden_dim))

        # Initialize hidden state for forward pass
        h_forward = np.zeros((batch_size, self.hidden_dim))

        # Process the sequence in forward direction
        for t in range(sequence_length):
            h_forward = self.forward_step(
                inputs[:, t, :], h_forward, direction="forward"
            )
            
            # If returning sequences, store the hidden state at this time step
            if self.return_sequences:
                h_forward_seq[:, t, :] = h_forward

        # If not bidirectional, return the result based on return_sequences
        if not self.bidirectional:
            if self.return_sequences:
                return h_forward_seq
            else:
                return h_forward

        # For bidirectional RNN, process the sequence in backward direction
        h_backward = np.zeros((batch_size, self.hidden_dim))

        for t in range(sequence_length - 1, -1, -1):
            h_backward = self.forward_step(
                inputs[:, t, :], h_backward, direction="backward"
            )
            
            # If returning sequences, store the hidden state at this time step
            if self.return_sequences:
                h_backward_seq[:, t, :] = h_backward

        # Return the result based on return_sequences
        if self.return_sequences:
            # Concatenate forward and backward hidden states for each time step
            return np.concatenate([h_forward_seq, h_backward_seq], axis=2)
        else:
            # Concatenate just the final hidden states
            return np.concatenate([h_forward, h_backward], axis=1)

    def load_weights_from_keras(self, keras_layer):
        """Load weights from a Keras RNN layer"""
        if isinstance(keras_layer, tf.keras.layers.SimpleRNN):
            # Keras weights order: [kernel, recurrent_kernel, bias]
            weights = keras_layer.get_weights()
            
            self.W = weights[0]  # Input weights 
            self.U = weights[1]  # Recurrent weights 
            self.b = weights[2]  # Bias
            
            # Also set the return_sequences attribute to match Keras
            self.return_sequences = keras_layer.return_sequences
            
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
            
            # Also set the return_sequences attribute to match Keras
            self.return_sequences = keras_layer.forward_layer.return_sequences
            
        else:
            raise ValueError(f"Unsupported layer type for weight loading: {type(keras_layer)}")