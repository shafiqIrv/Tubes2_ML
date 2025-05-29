import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from src.models.src.models.lstm.nusax_loader import NusaXLoader


class LSTMExperiments:
    def __init__(self, data_loader, batch_size=32, epochs=10, embedding_dim=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.embedding_dim = embedding_dim

        # Use provided data loader
        self.data_loader = data_loader

        # Get dataset characteristics
        self.vocab_size = len(self.data_loader.get_vocabulary())
        self.num_classes = self.data_loader.num_classes
        self.max_sequence_length = self.data_loader.max_sequence_length

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Maximum sequence length: {self.max_sequence_length}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Embedding dimension: {self.embedding_dim}")

    def _build_base_model(self, lstm_layers, lstm_units, bidirectional=False):
        """Build a base LSTM model with specified parameters"""
        model = Sequential()

        # Add embedding layer
        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
            )
        )

        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            # If it's the last LSTM layer, don't return sequences
            return_sequences = i < len(lstm_units) - 1

            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))

            # Add dropout after each LSTM layer
            model.add(Dropout(0.2))

        # Add final dense layer
        model.add(Dense(self.num_classes, activation="softmax"))

        # Compile the model
        model.compile(
            optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
        )

        return model

    def _train_model(self, model, name=None):
        """Train the model and return history"""
        # Get datasets
        train_dataset = self.data_loader.get_dataset("train")
        val_dataset = self.data_loader.get_dataset("valid")

        # Train the model
        history = model.fit(
            train_dataset, validation_data=val_dataset, epochs=self.epochs, verbose=1
        )

        # Evaluate on test set
        test_dataset = self.data_loader.get_dataset("test")
        test_loss, test_acc = model.evaluate(test_dataset, verbose=1)

        # Calculate F1-score
        x_test, y_test = self.data_loader.get_vectorized_data("test")
        y_pred = np.argmax(model.predict(x_test), axis=1)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\nTest metrics for {name if name else 'model'}:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Macro F1-score: {f1:.4f}")

        # Save model if name is provided
        if name:
            model.save_weights(f"../output/models/lstm/{name}.weights.h5")

        return model, history

    def run_layer_count_experiment(self, layer_count_variants):
        """Run experiment with different number of LSTM layers"""
        models = []
        histories = []

        for layer_count, name in layer_count_variants:
            print(f"\n=== Training model with {name} ===")

            # Define LSTM units per layer
            if layer_count == 1:
                lstm_units = [128]
            elif layer_count == 2:
                lstm_units = [128, 64]
            elif layer_count == 3:
                lstm_units = [128, 64, 32]
            else:
                raise ValueError("Unsupported layer count")

            # Build and train model
            model = self._build_base_model(layer_count, lstm_units)
            model, history = self._train_model(model, name=f"lstm_layers_{layer_count}")
            model.summary()

            models.append((model, name))
            histories.append((history, name))

        # Plot comparison
        self._plot_comparison(
            histories, metric="loss", title="Effect of LSTM Layer Count"
        )
        self._plot_comparison(
            histories, metric="accuracy", title="Effect of LSTM Layer Count"
        )

        return models, histories

    def run_cell_count_experiment(self, cell_count_variants):
        """Run experiment with different number of cells per LSTM layer"""
        models = []
        histories = []

        for cell_units, name in cell_count_variants:
            print(f"\n=== Training model with {name} ===")

            # Build and train model
            model = self._build_base_model(len(cell_units), cell_units)
            model, history = self._train_model(
                model, name=f"lstm_cells_{'_'.join(map(str, cell_units))}"
            )
            model.summary()

            models.append((model, name))
            histories.append((history, name))

        # Plot comparison
        self._plot_comparison(
            histories, metric="loss", title="Effect of LSTM Cell Count"
        )
        self._plot_comparison(
            histories, metric="accuracy", title="Effect of LSTM Cell Count"
        )

        return models, histories

    def run_direction_experiment(self, direction_variants):
        """Run experiment with bidirectional vs unidirectional LSTM"""
        models = []
        histories = []

        for bidirectional, name in direction_variants:
            print(f"\n=== Training model with {name} ===")

            # Define LSTM units per layer (single layer for simplicity)
            lstm_units = [128]

            # Build and train model
            model = self._build_base_model(1, lstm_units, bidirectional=bidirectional)
            model, history = self._train_model(
                model,
                name=f"lstm_{'bidirectional' if bidirectional else 'unidirectional'}",
            )
            model.summary()

            models.append((model, name))
            histories.append((history, name))

        # Plot comparison
        self._plot_comparison(
            histories, metric="loss", title="Effect of LSTM Direction"
        )
        self._plot_comparison(
            histories, metric="accuracy", title="Effect of LSTM Direction"
        )

        return models, histories

    def _plot_comparison(self, histories, metric="loss", title="Model Comparison"):
        """Plot comparison of training histories with separate train/validation plots"""
        # Create separate plots for training and validation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Training plot
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
            f"../output/results/lstm/{title.lower().replace(' ', '_')}_{metric}_comparison.png"
        )
        plt.show()
