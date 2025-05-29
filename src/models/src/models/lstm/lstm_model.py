import numpy as np
from src.models.src.models.base_model.base_model import BaseModel


class LSTMModel(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """
        Perform forward propagation through all layers

        Args:
            inputs: numpy array of shape (batch_size, sequence_length)

        Returns:
            numpy array of shape (batch_size, num_classes)
        """
        x = inputs

        for layer in self.layers:
            x = layer.forward(x)

        return x
