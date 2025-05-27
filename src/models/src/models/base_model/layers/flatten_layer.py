class FlattenLayer:
    def __init__(self):
        self.input = None
        self.input_shape = None
        self.output = None

    def load_weights_from_keras(self, keras_layer):
        pass

    def forward(self, inputs):
        self.input = inputs
        self.input_shape = inputs.shape[
            1:
        ] 

        batch_size = inputs.shape[0]
        self.output = inputs.reshape(batch_size, -1)
        return self.output
