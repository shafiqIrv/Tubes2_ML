import numpy as np
import tensorflow as tf


class RNNLayer:
    def __init__(self, input_dim, hidden_dim, bidirectional=False, return_sequences=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        
        self.W = None
        self.U = None
        self.b = None
        
        if bidirectional:
            self.backward_W = None
            self.backward_U = None
            self.backward_b = None
    
    def _rnn_step(self, x_t, h_prev, W, U, b):
        # Ensure all inputs are float32 for exact matching with Keras
        x_t = x_t.astype(np.float32)
        h_prev = h_prev.astype(np.float32)
        W = W.astype(np.float32)
        U = U.astype(np.float32)
        b = b.astype(np.float32)
        
        # Keras SimpleRNN computes: h_t = tanh(x_t @ W + h_prev @ U + b)
        # Make sure to use the same order of operations
        x_proj = np.dot(x_t, W)
        h_proj = np.dot(h_prev, U)
        linear = x_proj + h_proj + b
        
        # Apply tanh activation - use np.tanh for stability
        h_t = np.tanh(linear)
        
        return h_t.astype(np.float32)
    
    def forward(self, inputs):

        inputs = inputs.astype(np.float32)
        
        # Handle input shape 
        if len(inputs.shape) == 2:
            batch_size, sequence_length = inputs.shape
            inputs = inputs.reshape(batch_size, sequence_length, 1)
        
        batch_size, sequence_length, actual_input_dim = inputs.shape
        
        # Check for dimension mismatch 
        if actual_input_dim != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch in RNN layer: expected {self.input_dim}, got {actual_input_dim}. "
                f"Input shape: {inputs.shape}"
            )
        
        # Initialize hidden states with zeros
        h_forward = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        
        if self.return_sequences:
            forward_outputs = np.zeros((batch_size, sequence_length, self.hidden_dim), dtype=np.float32)
        
        # Forward pass through time steps
        for t in range(sequence_length):
            h_forward = self._rnn_step(inputs[:, t, :], h_forward, self.W, self.U, self.b)
            if self.return_sequences:
                forward_outputs[:, t, :] = h_forward
        
        # Handle unidirectional case
        if not self.bidirectional:
            return forward_outputs if self.return_sequences else h_forward
        
        # Bidirectional case - backward pass
        h_backward = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        
        if self.return_sequences:
            backward_outputs = np.zeros((batch_size, sequence_length, self.hidden_dim), dtype=np.float32)
        
        # Process sequence in reverse
        for t in range(sequence_length - 1, -1, -1):
            h_backward = self._rnn_step(
                inputs[:, t, :], 
                h_backward,
                self.backward_W, 
                self.backward_U, 
                self.backward_b
            )
            if self.return_sequences:
                backward_outputs[:, t, :] = h_backward
        
        # Merge
        if self.return_sequences:
            return np.concatenate([forward_outputs, backward_outputs], axis=2)
        else:
            return np.concatenate([h_forward, h_backward], axis=1)
    
    def load_weights_from_keras(self, keras_layer):
        if isinstance(keras_layer, tf.keras.layers.SimpleRNN):
            weights = keras_layer.get_weights()
            
            # Keras weight order: [kernel, recurrent_kernel, bias]
            # kernel: input weights, recurrent_kernel: hidden state weights
            self.W = weights[0].astype(np.float32)  # Input weights
            self.U = weights[1].astype(np.float32)  # Recurrent weights  
            self.b = weights[2].astype(np.float32)  # Bias
            
            self.return_sequences = keras_layer.return_sequences
            
        elif isinstance(keras_layer, tf.keras.layers.Bidirectional):
            # Get forward and backward weights
            forward_weights = keras_layer.forward_layer.get_weights()
            backward_weights = keras_layer.backward_layer.get_weights()
            
            # Forward direction
            self.W = forward_weights[0].astype(np.float32)
            self.U = forward_weights[1].astype(np.float32)
            self.b = forward_weights[2].astype(np.float32)
            
            # Backward direction
            self.backward_W = backward_weights[0].astype(np.float32)
            self.backward_U = backward_weights[1].astype(np.float32)
            self.backward_b = backward_weights[2].astype(np.float32)
            
            self.return_sequences = keras_layer.forward_layer.return_sequences
            self.bidirectional = True
            
        else:
            raise ValueError(f"Unsupported layer type in keras: {type(keras_layer)}")
        
        print(f"Loaded weights - W: {self.W.shape}, U: {self.U.shape}, b: {self.b.shape}")
        
        if self.bidirectional:
            print(f"Backward weights - W: {self.backward_W.shape}, U: {self.backward_U.shape}, b: {self.backward_b.shape}")