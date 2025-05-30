import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class RNNModel:
    def __init__(self):
        self.layers = []
        self.compiled = False
    
    # Add layer
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        x = inputs
        
        for i, layer in enumerate(self.layers):
            try:
                x_prev_shape = x.shape
                x = layer.forward(x)
                print(f"Layer {i} ({type(layer).__name__}): {x_prev_shape} -> {x.shape}")
            except Exception as e:
                print(f"ERROR in layer {i} ({type(layer).__name__})")
                print(f"Input shape: {x.shape}")
                print(f"Error: {e}")
                raise e
        
        return x
    
    def predict(self, inputs, batch_size=1):
        num_samples = len(inputs)
        all_predictions = []
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_inputs = inputs[i:end_idx]
            
            # Get probabilities from forward pass
            batch_outputs = self.forward(batch_inputs)
            
            # Convert to class predictions
            batch_predictions = np.argmax(batch_outputs, axis=1)
            all_predictions.extend(batch_predictions)
        
        return np.array(all_predictions)
    
    def predict_proba(self, inputs, batch_size=32):
        num_samples = len(inputs)
        all_probabilities = []
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_inputs = inputs[i:end_idx]
            
            batch_probabilities = self.forward(batch_inputs)
            all_probabilities.append(batch_probabilities)
        
        return np.vstack(all_probabilities)
    
    def evaluate(self, x_test, y_test, batch_size=32):
        print(f"Evaluating on {len(x_test)} samples...")
        
        # Get predictions
        y_pred = self.predict(x_test, batch_size=batch_size)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'num_samples': len(y_test)
        }
    
    def load_weights_from_keras(self, keras_model):
        print("Loading weights from Keras model...")
        print(f"Keras model has {len(keras_model.layers)} layers")
        print(f"Scratch model has {len(self.layers)} layers")
        
        # Print Keras model architecture
        print("\nKeras model architecture:")
        for i, layer in enumerate(keras_model.layers):
            layer_type = type(layer).__name__
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            else:
                output_shape = "Unknown"
            print(f"  Layer {i}: {layer_type} -> {output_shape}")
        
        print("\nScratch model architecture:")
        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__
            print(f"  Layer {i}: {layer_type}")
        
        # Load weights layer by layer
        successful_loads = 0
        for i, scratch_layer in enumerate(self.layers):
            if i >= len(keras_model.layers):
                print(f"WARNING: No Keras layer for scratch layer {i}")
                continue
                
            keras_layer = keras_model.layers[i]
            
            if not hasattr(scratch_layer, 'load_weights_from_keras'):
                print(f"Layer {i}: No weights to load ({type(scratch_layer).__name__})")
                continue
            
            try:
                print(f"\nLoading weights for layer {i}:")
                scratch_layer.load_weights_from_keras(keras_layer)
                successful_loads += 1
                print(f"Successfully loaded weights for layer {i}")
                
            except Exception as e:
                print(f"Failed to load weights for layer {i}: {e}")
                raise e
        
        print(f"\n✅ Successfully loaded weights for {successful_loads}/{len(self.layers)} layers")
        self.compiled = True
    
    def summary(self):
        print("="*60)
        print("SCRATCH MODEL SUMMARY")
        print("="*60)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = type(layer).__name__
            
            # Count parameters
            layer_params = 0
            param_details = []
            
            if hasattr(layer, 'weights') and layer.weights is not None:
                layer_params += layer.weights.size
                param_details.append(f"weights: {layer.weights.shape}")
                
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer_params += layer.bias.size
                param_details.append(f"bias: {layer.bias.shape}")
                
            if hasattr(layer, 'W') and layer.W is not None:  # RNN layers
                layer_params += layer.W.size + layer.U.size + layer.b.size
                param_details.append(f"W: {layer.W.shape}, U: {layer.U.shape}, b: {layer.b.shape}")
                
                if hasattr(layer, 'backward_W') and layer.backward_W is not None:
                    layer_params += layer.backward_W.size + layer.backward_U.size + layer.backward_b.size
                    param_details.append(f"backward W: {layer.backward_W.shape}")
            
            param_str = ", ".join(param_details) if param_details else "no parameters"
            
            print(f"Layer {i:2d}: {layer_name:15s}")
            print(f"         Parameters: {layer_params:8,} ({param_str})")
            
            total_params += layer_params
        
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print("="*60)


def compare_models_detailed(keras_model, scratch_model, x_test, y_test, batch_size=32, num_samples=100):
    print(f"Comparing models on {num_samples} samples...")
    
    # Use subset for detailed comparison
    x_sample = x_test[:num_samples]
    y_sample = y_test[:num_samples]
    
    # Get predictions
    print("Getting Keras predictions...")
    keras_probs = keras_model.predict(x_sample, batch_size=batch_size, verbose=0)
    keras_preds = np.argmax(keras_probs, axis=1)
    
    print("Getting scratch predictions...")
    scratch_probs = scratch_model.predict_proba(x_sample, batch_size=batch_size)
    scratch_preds = np.argmax(scratch_probs, axis=1)
    
    # Calculate detailed metrics
    keras_acc = accuracy_score(y_sample, keras_preds)
    scratch_acc = accuracy_score(y_sample, scratch_preds)
    
    keras_f1 = f1_score(y_sample, keras_preds, average='macro')
    scratch_f1 = f1_score(y_sample, scratch_preds, average='macro')
    
    # Probability differences
    prob_diff = np.abs(keras_probs - scratch_probs)
    max_prob_diff = np.max(prob_diff)
    mean_prob_diff = np.mean(prob_diff)
    
    # Agreement
    pred_agreement = np.mean(keras_preds == scratch_preds)
    
    
    results = {
        'keras_metrics': {'accuracy': keras_acc, 'macro_f1': keras_f1},
        'scratch_metrics': {'accuracy': scratch_acc, 'macro_f1': scratch_f1},
        'probability_differences': {
            'max_diff': max_prob_diff,
            'mean_diff': mean_prob_diff
        },
        'agreement': pred_agreement,
    }
    
    # results
    print("\n" + "="*50)
    print("DETAILED COMPARISON RESULTS")
    print("="*50)
    print(f"Sample size: {num_samples}")
    print(f"Keras    - Accuracy: {keras_acc:.4f}, F1: {keras_f1:.4f}")
    print(f"Scratch  - Accuracy: {scratch_acc:.4f}, F1: {scratch_f1:.4f}")
    print(f"Max probability difference: {max_prob_diff:.6f}")
    print(f"Mean probability difference: {mean_prob_diff:.6f}")
    print(f"Prediction agreement: {pred_agreement:.4f}")
    
    return results