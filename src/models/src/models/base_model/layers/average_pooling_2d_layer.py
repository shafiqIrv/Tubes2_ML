import numpy as np


class AveragePooling2DLayer:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = (
            pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        )
        self.strides = strides if strides else self.pool_size

        self.input = None
        self.output = None

    def load_weights_from_keras(self, keras_layer):
        pass

    def forward(self, inputs):
        self.input = inputs
        batch_size, input_height, input_width, channels = inputs.shape
        pool_height, pool_width = self.pool_size
        stride_h, stride_w = self.strides

        # Calculate output dimensions
        output_height = (input_height - pool_height) // stride_h + 1
        output_width = (input_width - pool_width) // stride_w + 1

        # Initialize output
        output = np.zeros((batch_size, output_height, output_width, channels))

        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Calculate pooling region boundaries
                        start_i = i * stride_h
                        end_i = start_i + pool_height
                        start_j = j * stride_w
                        end_j = start_j + pool_width

                        # Extract pooling region and take average
                        region = inputs[b, start_i:end_i, start_j:end_j, c]
                        output[b, i, j, c] = np.mean(region)

        self.output = output
        return output
