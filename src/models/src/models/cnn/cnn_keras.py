import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import f1_score

# Import your base classes (adjust paths as needed)
from ..base_model.base_model import BaseModel
from ..base_model.layers.activation_layer import ReLU, Softmax
from ..base_model.layers.dense_layer import DenseLayer
from ..base_model.utils.evaluation import compare_keras_vs_scratch, evaluate_predictions

# Import the completed CNN class
from ..base_model.layers.conv_2d_layer import Conv2DLayer
from ..base_model.layers.max_pooling_2d_layer import MaxPooling2DLayer
from ..base_model.layers.average_pooling_2d_layer import AveragePooling2DLayer
from ..base_model.layers.flatten_layer import FlattenLayer

from .cnn import CNN


class CNNExperiment:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        """Load and preprocess CIFAR-10 data"""
        print("Loading CIFAR-10 data...")

        # Load CIFAR-10
        (x_train_full, y_train_full), (x_test, y_test) = (
            keras.datasets.cifar10.load_data()
        )

        # Normalize pixel values
        x_train_full = x_train_full.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # Create train/validation split (40k train, 10k validation)
        split_idx = 40000

        self.x_train = x_train_full[:split_idx]
        self.y_train = y_train_full[:split_idx].flatten()
        self.x_val = x_train_full[split_idx:]
        self.y_val = y_train_full[split_idx:].flatten()
        self.x_test = x_test
        self.y_test = y_test.flatten()

        print(f"Training set: {self.x_train.shape}")
        print(f"Validation set: {self.x_val.shape}")
        print(f"Test set: {self.x_test.shape}")

    def create_keras_model(
        self, filters_list=[32, 64, 128], kernel_sizes=[3, 3, 3], pooling_type="max"
    ):
        """Create a Keras CNN model for training"""
        model = keras.Sequential()

        # Add convolutional layers
        for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes)):
            if i == 0:
                model.add(
                    keras.layers.Conv2D(
                        filters, kernel_size, activation="relu", input_shape=(32, 32, 3)
                    )
                )
            else:
                model.add(keras.layers.Conv2D(filters, kernel_size, activation="relu"))

            # Add pooling layer
            if pooling_type == "max":
                model.add(keras.layers.MaxPooling2D((2, 2)))
            else:
                model.add(keras.layers.AveragePooling2D((2, 2)))

        # Add dense layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(10, activation="softmax"))

        return model

    def create_scratch_model(
        self, filters_list=[32, 64, 128], kernel_sizes=[3, 3, 3], pooling_type="max"
    ):
        """Create corresponding from-scratch CNN model"""
        cnn = CNN(input_shape=(32, 32, 3), num_classes=10)

        # Add convolutional layers
        for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes)):
            cnn.add(
                Conv2DLayer(filters=filters, kernel_size=kernel_size, activation=ReLU())
            )

            if pooling_type == "max":
                cnn.add(MaxPooling2DLayer(pool_size=(2, 2)))
            else:
                cnn.add(AveragePooling2DLayer(pool_size=(2, 2)))

        # Add dense layers
        cnn.add(FlattenLayer())
        cnn.add(DenseLayer(input_dim=None, output_dim=64, activation=ReLU()))
        cnn.add(DenseLayer(input_dim=64, output_dim=10, activation=Softmax()))

        return cnn

    def train_keras_model(self, model, epochs=10):
        """Train the Keras model"""
        print("Training Keras model...")

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            self.x_train,
            self.y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=1,
        )

        return history

    def calculate_dense_input_dim(self, scratch_model, sample_input):
        """Calculate the input dimension for the first dense layer"""
        # Forward pass through conv layers only
        output = sample_input

        conv_layer_count = 0
        for layer in scratch_model.layers:
            if isinstance(
                layer, (Conv2DLayer, MaxPooling2DLayer, AveragePooling2DLayer)
            ):
                output = layer.forward(output)
                conv_layer_count += 1
            elif isinstance(layer, FlattenLayer):
                output = layer.forward(output)
                # Update the first dense layer's input dimension
                for next_layer in scratch_model.layers[conv_layer_count + 1 :]:
                    if (
                        isinstance(next_layer, DenseLayer)
                        and next_layer.input_dim is None
                    ):
                        next_layer.input_dim = output.shape[1]
                        next_layer.weights = (
                            np.random.randn(next_layer.input_dim, next_layer.output_dim)
                            * 0.01
                        )
                        print(f"Set Dense layer input_dim to: {next_layer.input_dim}")
                        break
                break

    def experiment_filter_numbers(self):
        """Experiment with different filter configurations"""
        print("\n=== Experimenting with Filter Numbers ===")

        filter_configs = [[16, 32, 64], [32, 64, 128], [64, 128, 256]]

        results = {}

        for i, filters in enumerate(filter_configs):
            print(f"\nExperiment {i+1}: Testing filters {filters}")

            # Create and train Keras model
            keras_model = self.create_keras_model(filters_list=filters)
            history = self.train_keras_model(
                keras_model, epochs=5
            )  # Reduced epochs for demo

            # Evaluate Keras model
            keras_pred = keras_model.predict(
                self.x_test[:1000]
            )  # Use subset for faster testing
            keras_classes = np.argmax(keras_pred, axis=1)
            keras_f1 = f1_score(self.y_test[:1000], keras_classes, average="macro")

            # Create corresponding scratch model
            scratch_model = self.create_scratch_model(filters_list=filters)

            # Calculate dense layer input dimensions
            sample_input = self.x_test[:1]  # Single sample for dimension calculation
            self.calculate_dense_input_dim(scratch_model, sample_input)

            # Load weights from Keras to scratch model
            try:
                scratch_model.load_weights_from_keras(keras_model)

                # Test scratch model
                scratch_pred = scratch_model.predict(
                    self.x_test[:100]
                )  # Smaller subset due to slower computation
                scratch_f1 = f1_score(self.y_test[:100], scratch_pred, average="macro")

                # Calculate agreement
                keras_subset = np.argmax(keras_model.predict(self.x_test[:100]), axis=1)
                agreement = np.mean(keras_subset == scratch_pred) * 100

                results[str(filters)] = {
                    "keras_f1": keras_f1,
                    "scratch_f1": scratch_f1,
                    "agreement": agreement,
                    "history": history,
                }

                print(f"Keras F1-Score: {keras_f1:.4f}")
                print(f"Scratch F1-Score: {scratch_f1:.4f}")
                print(f"Agreement: {agreement:.1f}%")

            except Exception as e:
                print(f"Error in scratch implementation: {e}")
                results[str(filters)] = {
                    "keras_f1": keras_f1,
                    "scratch_f1": None,
                    "agreement": None,
                    "history": history,
                    "error": str(e),
                }

        return results

    def experiment_pooling_types(self):
        """Experiment with different pooling types"""
        print("\n=== Experimenting with Pooling Types ===")

        pooling_types = ["max", "average"]
        results = {}

        for pooling in pooling_types:
            print(f"\nTesting {pooling} pooling...")

            # Create and train Keras model
            keras_model = self.create_keras_model(pooling_type=pooling)
            history = self.train_keras_model(keras_model, epochs=5)

            # Evaluate Keras model
            keras_pred = keras_model.predict(self.x_test[:1000])
            keras_classes = np.argmax(keras_pred, axis=1)
            keras_f1 = f1_score(self.y_test[:1000], keras_classes, average="macro")

            # Create scratch model
            scratch_model = self.create_scratch_model(pooling_type=pooling)

            # Calculate dimensions and load weights
            sample_input = self.x_test[:1]
            self.calculate_dense_input_dim(scratch_model, sample_input)

            try:
                scratch_model.load_weights_from_keras(keras_model)

                # Test scratch model
                scratch_pred = scratch_model.predict(self.x_test[:100])
                scratch_f1 = f1_score(self.y_test[:100], scratch_pred, average="macro")

                # Calculate agreement
                keras_subset = np.argmax(keras_model.predict(self.x_test[:100]), axis=1)
                agreement = np.mean(keras_subset == scratch_pred) * 100

                results[pooling] = {
                    "keras_f1": keras_f1,
                    "scratch_f1": scratch_f1,
                    "agreement": agreement,
                    "history": history,
                }

                print(f"Keras F1-Score: {keras_f1:.4f}")
                print(f"Scratch F1-Score: {scratch_f1:.4f}")
                print(f"Agreement: {agreement:.1f}%")

            except Exception as e:
                print(f"Error in scratch implementation: {e}")
                results[pooling] = {
                    "keras_f1": keras_f1,
                    "scratch_f1": None,
                    "agreement": None,
                    "history": history,
                    "error": str(e),
                }

        return results

    def run_all_experiments(self):
        """Run all required experiments"""
        # Load data
        self.load_data()

        # Run experiments
        print("Starting CNN experiments...")

        filter_results = self.experiment_filter_numbers()
        pooling_results = self.experiment_pooling_types()

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        print("\n1. FILTER NUMBERS EXPERIMENT:")
        print("-" * 40)
        for filters, result in filter_results.items():
            if "error" not in result:
                print(
                    f"   {filters}: Keras F1={result['keras_f1']:.4f}, "
                    f"Scratch F1={result['scratch_f1']:.4f}, "
                    f"Agreement={result['agreement']:.1f}%"
                )
            else:
                print(
                    f"   {filters}: Keras F1={result['keras_f1']:.4f}, "
                    f"Scratch Error: {result['error']}"
                )

        print("\n2. POOLING TYPES EXPERIMENT:")
        print("-" * 40)
        for pooling, result in pooling_results.items():
            if "error" not in result:
                print(
                    f"   {pooling}: Keras F1={result['keras_f1']:.4f}, "
                    f"Scratch F1={result['scratch_f1']:.4f}, "
                    f"Agreement={result['agreement']:.1f}%"
                )
            else:
                print(
                    f"   {pooling}: Keras F1={result['keras_f1']:.4f}, "
                    f"Scratch Error: {result['error']}"
                )

        return filter_results, pooling_results
