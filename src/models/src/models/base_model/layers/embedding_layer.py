import numpy as np


class EmbeddingLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim  # Vocabulary size
        self.output_dim = output_dim  # Embedding dimension
        self.weights = np.random.randn(input_dim, output_dim) * 0.01

    def forward(self, inputs):
        """
        Perform forward propagation for the Embedding layer

        Args:
            inputs: numpy array of shape (batch_size, sequence_length)

        Returns:
            numpy array of shape (batch_size, sequence_length, output_dim)
        """
        # Get the embedding vectors for each input token
        batch_size, sequence_length = inputs.shape
        output = np.zeros((batch_size, sequence_length, self.output_dim))

        for i in range(batch_size):
            for j in range(sequence_length):
                # Get the token index
                token_idx = inputs[i, j]

                # If token index is out of range, use index 0 (usually reserved for padding)
                if token_idx >= self.input_dim:
                    token_idx = 0

                output[i, j] = self.weights[token_idx]

        return output

    def load_weights_from_keras(self, keras_layer):
        """Load weights from a Keras Embedding layer"""
        self.weights = keras_layer.get_weights()[0]
