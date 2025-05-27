import numpy as np


class Conv2DLayer:
    def __init__(
        self, filters, kernel_size, activation=None, input_shape=None, padding="valid"
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape
        self.padding = padding

        # Initialize weights and biases
        self.weights = None
        self.biases = None

        # Store for backward pass
        self.input = None
        self.output = None

    def initialize_weights(self, input_channels):
        self.weights = np.random.randn(
            self.filters, input_channels, self.kernel_size, self.kernel_size
        ) * np.sqrt(2.0 / (self.kernel_size * self.kernel_size * input_channels))

        self.biases = np.zeros(self.filters)

    def load_weights_from_keras(self, keras_layer):
        keras_weights = keras_layer.get_weights()
        if len(keras_weights) >= 2:
            weights_keras = keras_weights[0]
            self.weights = np.transpose(weights_keras, (3, 2, 0, 1))
            self.biases = keras_weights[1]

    def forward(self, inputs):
        self.input = inputs
        batch_size, height, width, input_channels = inputs.shape

        if self.weights is None:
            self.initialize_weights(input_channels)

        # Calculate output dimensions
        if self.padding == "valid":
            output_height = height - self.kernel_size + 1
            output_width = width - self.kernel_size + 1
            padded_inputs = inputs
        else:
            output_height = height
            output_width = width

            # Add padding to input
            pad_h = self.kernel_size // 2
            pad_w = self.kernel_size // 2
            padded_inputs = np.pad(
                inputs,
                ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode="constant",
            )

        # Initialize output
        output = np.zeros((batch_size, output_height, output_width, self.filters))

        # Perform convolution with corrected indexing
        for b in range(batch_size):
            for f in range(self.filters):
                for i in range(output_height):
                    for j in range(output_width):
                        # Extract region for convolution
                        # Region shape: (kernel_size, kernel_size, input_channels)
                        region = padded_inputs[
                            b, i : i + self.kernel_size, j : j + self.kernel_size, :
                        ]

                        # Get filter weights for this filter
                        # Filter shape: (input_channels, kernel_size, kernel_size)
                        filter_weights = self.weights[f, :, :, :]

                        # Transpose filter weights to match region shape
                        # From (input_channels, kernel_size, kernel_size)
                        # To (kernel_size, kernel_size, input_channels)
                        filter_weights_transposed = np.transpose(
                            filter_weights, (1, 2, 0)
                        )

                        # Now both have shape (kernel_size, kernel_size, input_channels)
                        # Perform element-wise multiplication and sum
                        conv_result = (
                            np.sum(region * filter_weights_transposed) + self.biases[f]
                        )
                        output[b, i, j, f] = conv_result

        # Apply activation if specified
        if self.activation:
            output = self.activation.forward(output)

        self.output = output
        return output
