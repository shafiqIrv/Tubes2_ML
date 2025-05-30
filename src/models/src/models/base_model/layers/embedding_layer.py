import numpy as np

class EmbeddingLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim  # Vocabulary size
        self.output_dim = output_dim  # Embedding dimension
        self.weights = np.random.randn(input_dim, output_dim) * 0.01

    def forward(self, inputs):
        # inputs to integers
        inputs = np.array(inputs, dtype=np.int32)
        
        # Get the embedding vectors for each input token
        batch_size, sequence_length = inputs.shape
        output = np.zeros((batch_size, sequence_length, self.output_dim), dtype=np.float32)

        for i in range(batch_size):
            for j in range(sequence_length):

                token_idx = inputs[i, j]
                # Handle out-of-range indices 
                if token_idx >= self.input_dim:
                    token_idx = 0  
                elif token_idx < 0:
                    token_idx = 0
                
                # Set the embedding vector
                output[i, j] = self.weights[token_idx]

        return output

    def load_weights_from_keras(self, keras_layer):
        self.weights = keras_layer.get_weights()[0]