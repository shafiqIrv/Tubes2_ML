import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

from src.models.src.models.base_model.utils.nusax_loader import NusaXLoader
from src.models.src.models.rnn.rnn_model import RNNModel
from src.models.src.models.rnn.rnn_layer import RNNLayer
from src.models.src.models.base_model.layers.embedding_layer import EmbeddingLayer
from src.models.src.models.base_model.layers.dense_layer import DenseLayer
from src.models.src.models.base_model.layers.dropout_layer import DropoutLayer
from src.models.src.models.base_model.layers.activation_layer import Softmax
from src.models.src.models.base_model.utils.evaluation import compare_keras_vs_scratch


class RNNExperiments:
    def __init__(self, data_loader=None, batch_size=32, epochs=10, embedding_dim=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_dim = embedding_dim

        # Create or use provided data loader
        if data_loader is None:
            self.data_loader = NusaXLoader(
                batch_size=batch_size,
                max_sequence_length=100,
                vocab_size=10000
            )
        else:
            self.data_loader = data_loader

        # Get dataset characteristics
        self.vocab_size = len(self.data_loader.get_vocabulary())
        self.num_classes = self.data_loader.num_classes
        self.max_sequence_length = self.data_loader.max_sequence_length

        # Create output directories if notexist
        os.makedirs("../output/models/rnn", exist_ok=True)
        os.makedirs("../output/results/rnn", exist_ok=True)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Maximum sequence length: {self.max_sequence_length}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def _build_base_model(self, rnn_layers, rnn_units, bidirectional=False):
        """Build a base RNN model with specified parameters"""
        model = Sequential()

        # Add embedding layer
        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
            )
        )

        # Add RNN layers
        for i, units in enumerate(rnn_units):
            # If it's the last RNN layer, don't return sequences
            return_sequences = i < len(rnn_units) - 1

            if bidirectional:
                model.add(Bidirectional(SimpleRNN(units, return_sequences=return_sequences)))
            else:
                model.add(SimpleRNN(units, return_sequences=return_sequences))

            # Dropout after each RNN layer
            model.add(Dropout(0.2))

        # Add final dense layer
        model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=Adam(), 
            loss=SparseCategoricalCrossentropy(), 
            metrics=["accuracy"]
        )

        return model

    def _train_model(self, model, name=None):
        # Return History

        train_dataset = self.data_loader.get_dataset("train")
        val_dataset = self.data_loader.get_dataset("valid")

        # Train 
        history = model.fit(
            train_dataset, validation_data=val_dataset, epochs=self.epochs, verbose=1
        )

        test_dataset = self.data_loader.get_dataset("test")
        test_loss, test_acc = model.evaluate(test_dataset, verbose=1)

        # F1-score
        x_test, y_test = self.data_loader.get_vectorized_data("test")
        y_pred = np.argmax(model.predict(x_test), axis=1)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\nTest metrics for {name if name else 'model'}:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Macro F1-score: {f1:.4f}")

        # Save model 
        if name:
            model.save_weights(f"../output/models/rnn/{name}.weights.h5")
            # Also save the full model for easier loading later
            model.save(f"../output/models/rnn/{name}.keras")

        return model, history

    # Different total RNN Layeers   
    def run_layer_count_experiment(self, layer_count_variants=None):
        if layer_count_variants is None:
            layer_count_variants = [
                (1, "1 RNN Layer"),
                (2, "2 RNN Layers"),
                (3, "3 RNN Layers"),
            ]
            
        models = []
        histories = []

        for layer_count, name in layer_count_variants:
            print(f"\n=== Training model with {name} ===")

            # Define RNN units per layer
            if layer_count == 1:
                rnn_units = [128]
            elif layer_count == 2:
                rnn_units = [128, 64]
            elif layer_count == 3:
                rnn_units = [128, 64, 32]
            else:
                raise ValueError("Unsupported layer count")

            # Build and train model
            model = self._build_base_model(layer_count, rnn_units)
            model, history = self._train_model(model, name=f"rnn_layers_{layer_count}")
            model.summary()

            models.append((model, name))
            histories.append((history, name))

        # Plot comparison
        self._plot_comparison(
            histories, metric="loss", title="Effect of RNN Layer Count"
        )
        self._plot_comparison(
            histories, metric="accuracy", title="Effect of RNN Layer Count"
        )

        return models, histories

    # Different number of cells
    def run_cell_count_experiment(self, cell_count_variants=None):
        if cell_count_variants is None:
            cell_count_variants = [
                ([32], "32 Units"),
                ([64], "64 Units"),
                ([128], "128 Units"),
            ]
            
        models = []
        histories = []

        for cell_units, name in cell_count_variants:
            print(f"\n=== Training model with {name} ===")

            # Build and train model
            model = self._build_base_model(len(cell_units), cell_units)
            model, history = self._train_model(
                model, name=f"rnn_cells_{'_'.join(map(str, cell_units))}"
            )
            model.summary()

            models.append((model, name))
            histories.append((history, name))

        self._plot_comparison(
            histories, metric="loss", title="Effect of RNN Cell Count"
        )
        self._plot_comparison(
            histories, metric="accuracy", title="Effect of RNN Cell Count"
        )

        return models, histories

    # Bidirectional vs Unidirectional
    def run_direction_experiment(self, direction_variants=None):
        if direction_variants is None:
            direction_variants = [
                (False, "Unidirectional"),
                (True, "Bidirectional"),
            ]
            
        models = []
        histories = []

        for bidirectional, name in direction_variants:
            print(f"\n=== Training model with {name} RNN ===")

            rnn_units = [128]

            model = self._build_base_model(1, rnn_units, bidirectional=bidirectional)
            model, history = self._train_model(
                model,
                name=f"rnn_{'bidirectional' if bidirectional else 'unidirectional'}",
            )
            model.summary()

            models.append((model, name))
            histories.append((history, name))

        # Plot comparison
        self._plot_comparison(
            histories, metric="loss", title="Effect of RNN Direction"
        )
        self._plot_comparison(
            histories, metric="accuracy", title="Effect of RNN Direction"
        )

        return models, histories

    # Plotting history comparison
    def _plot_comparison(self, histories, metric="loss", title="Model Comparison"):
        """Plot comparison of training histories with separate train/validation plots"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for history, name in histories:
            ax1.plot(history.history[metric], label=f"{name}")

        ax1.set_title(f"{title} - Training {metric.capitalize()}")
        ax1.set_ylabel(metric.capitalize())
        ax1.set_xlabel("Epoch")
        ax1.legend(loc="best")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Validation plot
        for history, name in histories:
            ax2.plot(history.history[f"val_{metric}"], label=f"{name}")

        ax2.set_title(f"{title} - Validation {metric.capitalize()}")
        ax2.set_ylabel(metric.capitalize())
        ax2.set_xlabel("Epoch")
        ax2.legend(loc="best")
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(
            f"../output/results/rnn/{title.lower().replace(' ', '_')}_{metric}_comparison.png"
        )
        plt.show()
        
    # Build RNN from scratch using keras weight
    def build_scratch_model(self, keras_model_path):
        # Load the Keras model
        keras_model = load_model(keras_model_path)
        
        # Create a new scratch model
        scratch_model = RNNModel()
        
        # Extract model architecture and add corresponding layers
        for i, layer in enumerate(keras_model.layers):
            if isinstance(layer, Embedding):
                # Add embedding layer
                scratch_model.add(
                    EmbeddingLayer(
                        input_dim=layer.input_dim,
                        output_dim=layer.output_dim
                    )
                )
            elif isinstance(layer, SimpleRNN) or isinstance(layer, Bidirectional):
                # Add RNN layer
                if isinstance(layer, Bidirectional):
                    # For bidirectional layer
                    input_dim = layer.input_shape[-1]
                    hidden_dim = layer.forward_layer.units
                    scratch_model.add(
                        RNNLayer(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            bidirectional=True
                        )
                    )
                else:
                    # For unidirectional layer
                    input_dim = layer.input_shape[-1]
                    hidden_dim = layer.units
                    scratch_model.add(
                        RNNLayer(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            bidirectional=False
                        )
                    )
            elif isinstance(layer, Dropout):
                # Add dropout layer
                scratch_model.add(DropoutLayer(dropout_rate=layer.rate))
            elif isinstance(layer, Dense):
                # Add dense layer
                input_dim = layer.input_shape[-1]
                output_dim = layer.units
                
                if layer.activation.__name__ == 'softmax':
                    # If the activation is softmax, add it separately
                    scratch_model.add(
                        DenseLayer(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            activation=None
                        )
                    )
                    scratch_model.add(Softmax())
                else:
                    # Other activations included in the dense layer
                    scratch_model.add(
                        DenseLayer(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            activation=None  
                        )
                    )
        
        # Load weights from Keras model
        scratch_model.load_weights_from_keras(keras_model)
        
        return scratch_model, keras_model
    
    def compare_models(self, keras_model_path):

        # Build the scratch with keras
        scratch_model, keras_model = self.build_scratch_model(keras_model_path)
        
        # Get test data
        x_test, y_test = self.data_loader.get_vectorized_data("test")
        
        # Compare 
        comparison_results = compare_keras_vs_scratch(
            keras_model, scratch_model, x_test, y_test, batch_size=self.batch_size
        )
        
        print("\n=== Model Comparison Results ===")
        print(f"Keras Model Accuracy: {comparison_results['keras_metrics']['accuracy']:.4f}")
        print(f"Keras Model Macro F1: {comparison_results['keras_metrics']['macro_f1']:.4f}")
        print(f"Scratch Model Accuracy: {comparison_results['scratch_metrics']['accuracy']:.4f}")
        print(f"Scratch Model Macro F1: {comparison_results['scratch_metrics']['macro_f1']:.4f}")
        print(f"Model Agreement: {comparison_results['model_agreement']:.4f}")
        
        return comparison_results