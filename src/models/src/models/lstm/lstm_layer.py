import numpy as np
import tensorflow as tf


def sigmoid(x):
    """Sigmoid activation function with clipping to prevent overflow"""
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)


class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, bidirectional=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Initialize weights for forward direction
        self.initialize_weights(direction="forward")

        # Initialize weights for backward direction if bidirectional
        if bidirectional:
            self.initialize_weights(direction="backward")

    def initialize_weights(self, direction="forward"):
        """Initialize weights for the LSTM layer"""
        prefix = direction + "_" if self.bidirectional else ""

        # Input weights for the gates (input, forget, cell, output)
        setattr(
            self,
            f"{prefix}W_i",
            np.random.randn(self.input_dim, self.hidden_dim) * 0.01,
        )
        setattr(
            self,
            f"{prefix}W_f",
            np.random.randn(self.input_dim, self.hidden_dim) * 0.01,
        )
        setattr(
            self,
            f"{prefix}W_c",
            np.random.randn(self.input_dim, self.hidden_dim) * 0.01,
        )
        setattr(
            self,
            f"{prefix}W_o",
            np.random.randn(self.input_dim, self.hidden_dim) * 0.01,
        )

        # Recurrent weights for the gates
        setattr(
            self,
            f"{prefix}U_i",
            np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01,
        )
        setattr(
            self,
            f"{prefix}U_f",
            np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01,
        )
        setattr(
            self,
            f"{prefix}U_c",
            np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01,
        )
        setattr(
            self,
            f"{prefix}U_o",
            np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01,
        )

        # Biases for the gates
        setattr(self, f"{prefix}b_i", np.zeros(self.hidden_dim))
        setattr(self, f"{prefix}b_f", np.zeros(self.hidden_dim))
        setattr(self, f"{prefix}b_c", np.zeros(self.hidden_dim))
        setattr(self, f"{prefix}b_o", np.zeros(self.hidden_dim))

    def forward_step(self, x_t, h_prev, c_prev, direction="forward"):
        """Perform one step of forward propagation for LSTM"""
        prefix = direction + "_" if self.bidirectional else ""

        # Get the weights for this direction
        W_i = getattr(self, f"{prefix}W_i")
        W_f = getattr(self, f"{prefix}W_f")
        W_c = getattr(self, f"{prefix}W_c")
        W_o = getattr(self, f"{prefix}W_o")

        U_i = getattr(self, f"{prefix}U_i")
        U_f = getattr(self, f"{prefix}U_f")
        U_c = getattr(self, f"{prefix}U_c")
        U_o = getattr(self, f"{prefix}U_o")

        b_i = getattr(self, f"{prefix}b_i")
        b_f = getattr(self, f"{prefix}b_f")
        b_c = getattr(self, f"{prefix}b_c")
        b_o = getattr(self, f"{prefix}b_o")

        # Input gate
        i_t = sigmoid(np.dot(x_t, W_i) + np.dot(h_prev, U_i) + b_i)

        # Forget gate
        f_t = sigmoid(np.dot(x_t, W_f) + np.dot(h_prev, U_f) + b_f)

        # Cell update
        c_tilde = tanh(np.dot(x_t, W_c) + np.dot(h_prev, U_c) + b_c)

        # Cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Output gate
        o_t = sigmoid(np.dot(x_t, W_o) + np.dot(h_prev, U_o) + b_o)

        # Hidden state
        h_t = o_t * tanh(c_t)

        return h_t, c_t

    def forward(self, inputs):
        """
        Perform forward propagation for the LSTM layer

        Args:
            inputs: numpy array of shape (batch_size, sequence_length, input_dim)

        Returns:
            numpy array of shape (batch_size, hidden_dim) if bidirectional=False
            or (batch_size, hidden_dim*2) if bidirectional=True
        """
        batch_size, sequence_length, _ = inputs.shape

        # Initialize hidden and cell states
        h_forward = np.zeros((batch_size, self.hidden_dim))
        c_forward = np.zeros((batch_size, self.hidden_dim))

        # Process the sequence in forward direction
        for t in range(sequence_length):
            h_forward, c_forward = self.forward_step(
                inputs[:, t, :], h_forward, c_forward, direction="forward"
            )

        if not self.bidirectional:
            return h_forward

        # For bidirectional LSTM, process the sequence in backward direction
        h_backward = np.zeros((batch_size, self.hidden_dim))
        c_backward = np.zeros((batch_size, self.hidden_dim))

        for t in range(sequence_length - 1, -1, -1):
            h_backward, c_backward = self.forward_step(
                inputs[:, t, :], h_backward, c_backward, direction="backward"
            )

        # Concatenate forward and backward hidden states
        return np.concatenate([h_forward, h_backward], axis=1)

    def load_weights_from_keras(self, keras_layer):
        """Load weights from a Keras LSTM layer"""
        if isinstance(keras_layer, tf.keras.layers.LSTM):
            # Keras weights order: [W_i|W_f|W_c|W_o, U_i|U_f|U_c|U_o, b_i|b_f|b_c|b_o]
            weights = keras_layer.get_weights()

            kernel = weights[0]  # Input weights (W)
            recurrent_kernel = weights[1]  # Recurrent weights (U)
            bias = weights[2]  # Biases

            # Extract weights for each gate
            hidden_dim = self.hidden_dim

            # Input weights
            self.W_i = kernel[:, :hidden_dim]
            self.W_f = kernel[:, hidden_dim : 2 * hidden_dim]
            self.W_c = kernel[:, 2 * hidden_dim : 3 * hidden_dim]
            self.W_o = kernel[:, 3 * hidden_dim :]

            # Recurrent weights
            self.U_i = recurrent_kernel[:, :hidden_dim]
            self.U_f = recurrent_kernel[:, hidden_dim : 2 * hidden_dim]
            self.U_c = recurrent_kernel[:, 2 * hidden_dim : 3 * hidden_dim]
            self.U_o = recurrent_kernel[:, 3 * hidden_dim :]

            # Biases
            self.b_i = bias[:hidden_dim]
            self.b_f = bias[hidden_dim : 2 * hidden_dim]
            self.b_c = bias[2 * hidden_dim : 3 * hidden_dim]
            self.b_o = bias[3 * hidden_dim :]

        elif isinstance(keras_layer, tf.keras.layers.Bidirectional):
            # Extract weights from forward and backward layers
            forward_weights = keras_layer.forward_layer.get_weights()
            backward_weights = keras_layer.backward_layer.get_weights()

            hidden_dim = self.hidden_dim

            # Forward direction
            f_kernel = forward_weights[0]
            f_recurrent_kernel = forward_weights[1]
            f_bias = forward_weights[2]

            # Input weights
            self.forward_W_i = f_kernel[:, :hidden_dim]
            self.forward_W_f = f_kernel[:, hidden_dim : 2 * hidden_dim]
            self.forward_W_c = f_kernel[:, 2 * hidden_dim : 3 * hidden_dim]
            self.forward_W_o = f_kernel[:, 3 * hidden_dim :]

            # Recurrent weights
            self.forward_U_i = f_recurrent_kernel[:, :hidden_dim]
            self.forward_U_f = f_recurrent_kernel[:, hidden_dim : 2 * hidden_dim]
            self.forward_U_c = f_recurrent_kernel[:, 2 * hidden_dim : 3 * hidden_dim]
            self.forward_U_o = f_recurrent_kernel[:, 3 * hidden_dim :]

            # Biases
            self.forward_b_i = f_bias[:hidden_dim]
            self.forward_b_f = f_bias[hidden_dim : 2 * hidden_dim]
            self.forward_b_c = f_bias[2 * hidden_dim : 3 * hidden_dim]
            self.forward_b_o = f_bias[3 * hidden_dim :]

            # Backward direction
            b_kernel = backward_weights[0]
            b_recurrent_kernel = backward_weights[1]
            b_bias = backward_weights[2]

            # Input weights
            self.backward_W_i = b_kernel[:, :hidden_dim]
            self.backward_W_f = b_kernel[:, hidden_dim : 2 * hidden_dim]
            self.backward_W_c = b_kernel[:, 2 * hidden_dim : 3 * hidden_dim]
            self.backward_W_o = b_kernel[:, 3 * hidden_dim :]

            # Recurrent weights
            self.backward_U_i = b_recurrent_kernel[:, :hidden_dim]
            self.backward_U_f = b_recurrent_kernel[:, hidden_dim : 2 * hidden_dim]
            self.backward_U_c = b_recurrent_kernel[:, 2 * hidden_dim : 3 * hidden_dim]
            self.backward_U_o = b_recurrent_kernel[:, 3 * hidden_dim :]

            # Biases
            self.backward_b_i = b_bias[:hidden_dim]
            self.backward_b_f = b_bias[hidden_dim : 2 * hidden_dim]
            self.backward_b_c = b_bias[2 * hidden_dim : 3 * hidden_dim]
            self.backward_b_o = b_bias[3 * hidden_dim :]
