from ..base_model.base_model import BaseModel
import numpy as np


class CNN(BaseModel):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

    def forward(self, inputs):
        output = inputs

        for _, layer in enumerate(self.layers):
            output = layer.forward(output)

        return output

    def predict(self, inputs):
        outputs = self.forward(inputs)

        # If output is from softmax, take argmax
        if len(outputs.shape) == 2 and outputs.shape[1] == self.num_classes:
            return np.argmax(outputs, axis=1)
        else:
            # If not softmax output, assume raw logits
            return np.argmax(outputs, axis=1)

    def load_weights_from_keras(self, keras_model):
        keras_layers = keras_model.layers

        # Skip input layer if it exists in Keras model
        if (
            hasattr(keras_layers[0], "input_shape")
            and len(keras_layers[0].get_weights()) == 0
        ):
            keras_layers = keras_layers[1:]

        # Load weights for each layer
        for _, (custom_layer, keras_layer) in enumerate(zip(self.layers, keras_layers)):
            try:
                if hasattr(custom_layer, "load_weights_from_keras"):
                    custom_layer.load_weights_from_keras(keras_layer)
                else:
                    print(f"No weights to load (layer doesn't have weights)")
            except Exception as e:
                print(f"Error loading weights: {e}")
