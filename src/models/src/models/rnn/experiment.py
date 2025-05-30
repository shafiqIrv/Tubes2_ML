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
            rnn_units = [64] * layer_count

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
    # returns (scratch_model, keras_model)
    def build_scratch_model(self, keras_model_path):
        # Load the Keras model
        keras_model = load_model(keras_model_path)
        print(f"Loaded Keras model from {keras_model_path}")
        
        keras_model.summary()
        
        scratch_model = RNNModel()
        
        # Get model architecture
        num_layers = len(keras_model.layers)
        print(f"Keras model has {num_layers} layers")
        
        # Track dimensions 
        current_dims = {}  
        
        # Determine input shape from the Keras model
        input_shape = keras_model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Store input shape
        if len(input_shape) >= 2:
            if len(input_shape) == 2:
                current_dims[-1] = {'seq_length': input_shape[1], 'features': 1}
            else:
                current_dims[-1] = {'seq_length': input_shape[1], 'features': input_shape[2]}
        
        # Chek each layer and add same layer to scratch model
        for i, layer in enumerate(keras_model.layers):
            layer_type = layer.__class__.__name__
            print(f"Processing layer {i}: {layer_type}")
            
            if isinstance(layer, Embedding):
                # Add embedding layer
                scratch_model.add(EmbeddingLayer(
                    input_dim=layer.input_dim,
                    output_dim=layer.output_dim
                ))
                print(f"Added Embedding layer: {layer.input_dim} → {layer.output_dim}")
                # Update current dimensions
                current_dims[i] = {'seq_length': current_dims.get(i-1, {}).get('seq_length', None), 
                                'features': layer.output_dim}
                
            elif isinstance(layer, SimpleRNN):
                # Get the input dimension by checkingg previous layers
                input_dim = None
                prev_idx = i - 1
                
                # Search backwards until we find a layer with output dimension
                while prev_idx >= 0 and input_dim is None:
                    prev_layer = keras_model.layers[prev_idx]
                    
                    if isinstance(prev_layer, Embedding):
                        input_dim = prev_layer.output_dim
                    elif isinstance(prev_layer, SimpleRNN):
                        input_dim = prev_layer.units
                    elif isinstance(prev_layer, Bidirectional):
                        input_dim = prev_layer.forward_layer.units * 2
                    elif isinstance(prev_layer, Dropout):
                        # Check layer before droput
                        prev_idx -= 1
                        continue
                    else:
                        if hasattr(prev_layer, 'output_shape'):
                            output_shape = prev_layer.output_shape
                            if len(output_shape) >= 3:  # (batch, seq, features)
                                input_dim = output_shape[-1]
                    
                    prev_idx -= 1
                
                # use saved dimensions
                if input_dim is None:
                    for j in range(i-1, -2, -1):
                        if j in current_dims and 'features' in current_dims[j]:
                            input_dim = current_dims[j]['features']
                            break
                
                # use default
                if input_dim is None:
                    input_dim = 100
                    print(f"Warning: Could not determine input dimension for layer {i}, using default {input_dim}")
                
                # Add RNN layer
                scratch_model.add(RNNLayer(
                    input_dim=input_dim,
                    hidden_dim=layer.units,
                    bidirectional=False,
                    return_sequences=layer.return_sequences
                ))
                print(f"Added RNN layer: {input_dim} → {layer.units} (return_sequences={layer.return_sequences})")
                
                # Update current dimensions
                current_dims[i] = {'seq_length': current_dims.get(i-1, {}).get('seq_length', None) if layer.return_sequences else None, 
                                'features': layer.units}
                
            elif isinstance(layer, Bidirectional):
                input_dim = None
                prev_idx = i - 1
                
                while prev_idx >= 0 and input_dim is None:
                    prev_layer = keras_model.layers[prev_idx]
                    
                    if isinstance(prev_layer, Embedding):
                        input_dim = prev_layer.output_dim
                    elif isinstance(prev_layer, SimpleRNN):
                        input_dim = prev_layer.units
                    elif isinstance(prev_layer, Bidirectional):
                        input_dim = prev_layer.forward_layer.units * 2
                    elif isinstance(prev_layer, Dropout):
                        prev_idx -= 1
                        continue
                    else:
                        if hasattr(prev_layer, 'output_shape'):
                            output_shape = prev_layer.output_shape
                            if len(output_shape) >= 3:
                                input_dim = output_shape[-1]
                    
                    prev_idx -= 1
                
                # use saved dimension
                if input_dim is None:
                    for j in range(i-1, -2, -1):
                        if j in current_dims and 'features' in current_dims[j]:
                            input_dim = current_dims[j]['features']
                            break
                
                # Default
                if input_dim is None:
                    input_dim = 100
                    print(f"Warning: Could not determine input dimension for layer {i}, using default {input_dim}")
                
                # Add bidirectional RNN layer
                scratch_model.add(RNNLayer(
                    input_dim=input_dim,
                    hidden_dim=layer.forward_layer.units,
                    bidirectional=True,
                    return_sequences=layer.forward_layer.return_sequences
                ))
                print(f"Added Bidirectional RNN layer: {input_dim} → {layer.forward_layer.units * 2} "
                    f"(return_sequences={layer.forward_layer.return_sequences})")
                
                # Update current dimensions
                current_dims[i] = {'seq_length': current_dims.get(i-1, {}).get('seq_length', None) if layer.forward_layer.return_sequences else None, 
                                'features': layer.forward_layer.units * 2}
                
            elif isinstance(layer, Dropout):
                print(f"Added Dropout layer: rate={layer.rate}")
                scratch_model.add(DropoutLayer(dropout_rate=layer.rate))
                
                # Dims before
                if i-1 in current_dims:
                    current_dims[i] = current_dims[i-1].copy()
                
            elif isinstance(layer, Dense):

                input_dim = None
                prev_idx = i - 1
                
                while prev_idx >= 0 and input_dim is None:
                    prev_layer = keras_model.layers[prev_idx]
                    
                    if isinstance(prev_layer, SimpleRNN):
                        input_dim = prev_layer.units
                    elif isinstance(prev_layer, Bidirectional):
                        input_dim = prev_layer.forward_layer.units * 2
                    elif isinstance(prev_layer, Dense):
                        input_dim = prev_layer.units
                    elif isinstance(prev_layer, Dropout):
                        prev_idx -= 1
                        continue
                    else:
                        # get from output shape
                        if hasattr(prev_layer, 'output_shape'):
                            output_shape = prev_layer.output_shape
                            if len(output_shape) == 2:  # (batch, features)
                                input_dim = output_shape[-1]
                            elif len(output_shape) > 2:  # Need to flatten

                                input_dim = np.prod(output_shape[1:])
                    
                    prev_idx -= 1
                
                # use saved dimensions
                if input_dim is None:
                    for j in range(i-1, -2, -1):
                        if j in current_dims and 'features' in current_dims[j]:
                            input_dim = current_dims[j]['features']
                            break
                
                # Default
                if input_dim is None:
                    input_dim = 128
                    print(f"Warning: Could not determine input dimension for layer {i}, using default {input_dim}")
                
                has_softmax = (hasattr(layer, 'activation') and 
                            getattr(layer.activation, '__name__', None) == 'softmax')
                
                # Add dense layer with appropriate activation
                if has_softmax:
                    scratch_model.add(DenseLayer(
                        input_dim=input_dim,
                        output_dim=layer.units,
                        activation=Softmax()
                    ))
                    print(f"Added Dense layer with Softmax: {input_dim} → {layer.units}")
                else:
                    scratch_model.add(DenseLayer(
                        input_dim=input_dim,
                        output_dim=layer.units,
                        activation=None
                    ))
                    print(f"Added Dense layer: {input_dim} → {layer.units}")
                
                # Update current dimensions
                current_dims[i] = {'seq_length': None, 'features': layer.units}
        

        print("\nScratch model structure:")
        for i, layer in enumerate(scratch_model.layers):
            layer_type = type(layer).__name__
            print(f"Layer {i}: {layer_type}")
        
        # Load weights from the Keras model
        try:
            scratch_model.load_weights_from_keras(keras_model)
            print("Successfully loaded weights from Keras model")
        except Exception as e:
            print(f"Error loading weights: {e}")
        
        return scratch_model, keras_model

    def compare_models(self, keras_model_path):
        # Use smaller batch size for stable comparison
        comparison_batch_size = 8
        
        try:
            # Build the scratch model and get the Keras model
            scratch_model, keras_model = self.build_scratch_model(keras_model_path)
            
            # Get test data
            x_test, y_test = self.data_loader.get_vectorized_data("test")
            
            # Use a subset for detailed comparison
            subset_size = min(100, len(x_test))
            x_subset = x_test[:subset_size]
            y_subset = y_test[:subset_size]
            
            print(f"\nRunning layer-by-layer comparison on a single sample...")
            sample_x = x_subset[:1]
            
            # Try comparing layer outputs, but handle any errors
            try:
                problem_layer = self.compare_layer_outputs(keras_model, scratch_model, sample_x)
            except Exception as e:
                print(f"Warning: Layer-by-layer comparison failed: {e}")
                print("Continuing with model-level comparison...")
                problem_layer = -1
            
            print(f"\nRunning detailed comparison on {subset_size} samples...")
            
            # Get Keras predictions for subset
            keras_probs = keras_model.predict(x_subset, batch_size=comparison_batch_size, verbose=0)
            keras_preds = np.argmax(keras_probs, axis=1)
            
            # Get scratch model predictions
            scratch_probs = []
            for i in range(0, subset_size, comparison_batch_size):
                end_idx = min(i + comparison_batch_size, subset_size)
                batch_x = x_subset[i:end_idx]
                batch_probs = scratch_model.forward(batch_x)
                scratch_probs.append(batch_probs)
            
            # Combine batch predictions
            scratch_probs = np.vstack(scratch_probs)
            scratch_preds = np.argmax(scratch_probs, axis=1)
            
            # Calculate agreement metrics
            pred_agreement = np.mean(keras_preds == scratch_preds)
            max_prob_diff = np.max(np.abs(keras_probs - scratch_probs))
            avg_prob_diff = np.mean(np.abs(keras_probs - scratch_probs))
            
            print(f"\nPrediction agreement: {pred_agreement:.4f} ({int(pred_agreement * subset_size)}/{subset_size} samples)")
            print(f"Maximum probability difference: {max_prob_diff:.6f}")
            print(f"Average probability difference: {avg_prob_diff:.6f}")
            
            # Find samples with disagreement
            disagreement_indices = np.where(keras_preds != scratch_preds)[0]
            
            if len(disagreement_indices) > 0:
                print(f"\nFound {len(disagreement_indices)} samples with disagreement:")
                
                # Analyze a few disagreements in detail
                for idx in disagreement_indices[:min(3, len(disagreement_indices))]:
                    print(f"\nSample {idx}:")
                    print(f"  Keras pred: {keras_preds[idx]} (confidence: {keras_probs[idx][keras_preds[idx]]:.4f})")
                    print(f"  Scratch pred: {scratch_preds[idx]} (confidence: {scratch_probs[idx][scratch_preds[idx]]:.4f})")
                    print(f"  Abs probability differences: {np.abs(keras_probs[idx] - scratch_probs[idx])}")
            
            # Compare on full test set
            print("\nComparing models on full test set...")
            full_comparison = compare_keras_vs_scratch(keras_model, scratch_model, x_test, y_test, batch_size=self.batch_size)
            
            # Add detailed metrics
            full_comparison['detailed'] = {
                'subset_size': subset_size,
                'pred_agreement': pred_agreement,
                'max_prob_diff': max_prob_diff,
                'avg_prob_diff': avg_prob_diff,
                'num_disagreements': len(disagreement_indices)
            }
            
            return full_comparison
            
        except Exception as e:
            print(f"Error comparing models: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'keras_metrics': {'accuracy': 0, 'macro_f1': 0},
                'scratch_metrics': {'accuracy': 0, 'macro_f1': 0},
                'model_agreement': 0,
                'error': str(e)
            }

    def compare_models_simple(self, keras_model_path):
        try:
            # Build the scratch model and get the Keras model
            scratch_model, keras_model = self.build_scratch_model(keras_model_path)
            
            # test data
            x_test, y_test = self.data_loader.get_vectorized_data("test")
            
            subset_size = min(100, len(x_test))
            x_subset = x_test[:subset_size]
            y_subset = y_test[:subset_size]
            
            # Get Keras predictions
            print("Getting Keras predictions...")
            keras_probs = keras_model.predict(x_subset, batch_size=8, verbose=0)
            keras_preds = np.argmax(keras_probs, axis=1)
            
            # Get scratch model predictions
            print("Getting scratch model predictions...")
            scratch_probs = []
            for i in range(0, subset_size, 8):
                end_idx = min(i + 8, subset_size)
                batch_x = x_subset[i:end_idx]
                batch_probs = scratch_model.forward(batch_x)
                scratch_probs.append(batch_probs)
            
            scratch_probs = np.vstack(scratch_probs)
            scratch_preds = np.argmax(scratch_probs, axis=1)
            
            # Calculate metrics
            pred_agreement = np.mean(keras_preds == scratch_preds)
            max_prob_diff = np.max(np.abs(keras_probs - scratch_probs))
            avg_prob_diff = np.mean(np.abs(keras_probs - scratch_probs))
            
            print(f"\nPrediction agreement: {pred_agreement:.4f} ({int(pred_agreement * subset_size)}/{subset_size} samples)")
            print(f"Maximum probability difference: {max_prob_diff:.6f}")
            print(f"Average probability difference: {avg_prob_diff:.6f}")
            
            # Calculate comparison 
            print("\nComparing on full dataset...")
            result = compare_keras_vs_scratch(keras_model, scratch_model, x_test, y_test, batch_size=32)
            
            result['detailed'] = {
                'subset_agreement': pred_agreement,
                'max_prob_diff': max_prob_diff,
                'avg_prob_diff': avg_prob_diff
            }
            
            return result
        
        except Exception as e:
            print(f"Error comparing models: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'model_agreement': 0
            }


    def compare_layer_outputs(self, keras_model, scratch_model, x_sample):

        import tensorflow as tf
        
        keras_full_output = keras_model.predict(x_sample, verbose=0)
        
        keras_layer_outputs = []
        
        input_shape = keras_model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        new_input = tf.keras.Input(shape=input_shape[1:])
        
        x = new_input
        intermediate_models = []
        
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                new_layer = tf.keras.layers.Embedding(
                    input_dim=layer.input_dim,
                    output_dim=layer.output_dim,
                    weights=layer.get_weights()
                )
            elif isinstance(layer, tf.keras.layers.SimpleRNN):
                new_layer = tf.keras.layers.SimpleRNN(
                    units=layer.units,
                    return_sequences=layer.return_sequences,
                    weights=layer.get_weights()
                )
            elif isinstance(layer, tf.keras.layers.Bidirectional):
                forward_layer = layer.forward_layer
                if isinstance(forward_layer, tf.keras.layers.SimpleRNN):
                    new_forward = tf.keras.layers.SimpleRNN(
                        units=forward_layer.units,
                        return_sequences=forward_layer.return_sequences,
                        weights=forward_layer.get_weights()
                    )
                    new_layer = tf.keras.layers.Bidirectional(
                        new_forward,
                        weights=layer.get_weights()
                    )
                else:
                    new_layer = type(layer).from_config(layer.get_config())
                    new_layer.build(x.shape)
                    new_layer.set_weights(layer.get_weights())
            elif isinstance(layer, tf.keras.layers.Dropout):
                new_layer = tf.keras.layers.Dropout(rate=layer.rate)
            elif isinstance(layer, tf.keras.layers.Dense):
                new_layer = tf.keras.layers.Dense(
                    units=layer.units,
                    activation=layer.activation,
                    weights=layer.get_weights()
                )
            else:
                try:
                    new_layer = type(layer).from_config(layer.get_config())
                    new_layer.build(x.shape)
                    new_layer.set_weights(layer.get_weights())
                except Exception as e:
                    print(f"Could not recreate layer {type(layer).__name__}: {e}")
                    continue
            
            x = new_layer(x)
            
            intermediate_model = tf.keras.Model(inputs=new_input, outputs=x)
            intermediate_models.append(intermediate_model)
        
        for i, model in enumerate(intermediate_models):
            try:
                output = model.predict(x_sample, verbose=0)
                keras_layer_outputs.append(output)
            except Exception as e:
                print(f"Error getting output for layer {i}: {e}")
                keras_layer_outputs.append(None)
        
        scratch_layer_outputs = []
        x = x_sample.copy()
        for layer in scratch_model.layers:
            if isinstance(layer, DropoutLayer):
                # Disable dropout during inference
                x = layer.forward(x, training=False)
            else:
                x = layer.forward(x)
            scratch_layer_outputs.append(x.copy())
        
        # Compare outputs 
        print("\nLayer-by-layer comparison:")
        for i, (keras_out, scratch_out) in enumerate(zip(keras_layer_outputs, scratch_layer_outputs)):
    
            if keras_out is None:
                print(f"Layer {i}: Could not get Keras output - skipping comparison")
                continue
                
            # Get layer types
            keras_layer_type = keras_model.layers[i].__class__.__name__
            scratch_layer_type = type(scratch_model.layers[i]).__name__
            
            # Compare shapes
            shape_match = (keras_out.shape == scratch_out.shape)
            
            print(f"Layer {i} ({keras_layer_type} / {scratch_layer_type}):")
            print(f"  Shape match: {shape_match} (Keras: {keras_out.shape}, Scratch: {scratch_out.shape})")
            
            if shape_match:
                max_diff = np.max(np.abs(keras_out - scratch_out))
                mean_diff = np.mean(np.abs(keras_out - scratch_out))
                
                is_close = max_diff < 1e-5
                
                print(f"  Max difference: {max_diff:.6f}")
                print(f"  Mean difference: {mean_diff:.6f}")
                print(f"  Outputs close: {is_close}")
                

                if not is_close:
                    print(f"\nFound significant difference at layer {i} ({keras_layer_type})")
                    return i
        
        return -1  
        