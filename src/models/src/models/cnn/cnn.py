from models.base_model.base_model import BaseModel


class CNN(BaseModel):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        print("Masuk CNN")

    def forward(self, inputs):
        pass
