import numpy as np
import tensorflow as tf

def extract_conv_weights(keras_layer):
    weights, biases = keras_layer.get_weights()
    

    weights = np.transpose(weights, (3, 2, 0, 1))
    
    return weights, biases

def extract_dense_weights(keras_layer):
    return keras_layer.get_weights()

def extract_rnn_weights(keras_layer):
    return keras_layer.get_weights()

def extract_lstm_weights(keras_layer):
    weights = keras_layer.get_weights()
    
    input_weights = weights[0]
    recurrent_weights = weights[1]
    biases = weights[2]
    
    return input_weights, recurrent_weights, biases

def extract_bidirectional_weights(keras_layer):
    # Check if layer has forward and backward layers
    if hasattr(keras_layer, 'forward_layer') and hasattr(keras_layer, 'backward_layer'):
        forward_weights = keras_layer.forward_layer.get_weights()
        backward_weights = keras_layer.backward_layer.get_weights()
        return forward_weights, backward_weights
    else:
        raise ValueError("Not a valid Bidirectional layer")

def extract_embedding_weights(keras_layer):
    return keras_layer.get_weights()[0]

def load_weights_by_layer_type(custom_layer, keras_layer):
    # Get the Keras layer class name
    layer_class = keras_layer.__class__.__name__
    
    try:
        if 'Conv' in layer_class:
            weights, biases = extract_conv_weights(keras_layer)
            custom_layer.weights = weights
            custom_layer.biases = biases
            
        elif 'Dense' in layer_class:
            weights, biases = extract_dense_weights(keras_layer)
            custom_layer.weights = weights
            custom_layer.biases = biases
            
        elif 'SimpleRNN' in layer_class:
            weights = extract_rnn_weights(keras_layer)
            # Custom layer should define how to assign these weights
            if hasattr(custom_layer, 'load_weights_from_keras'):
                custom_layer.load_weights_from_keras(keras_layer)
            else:
                return False
                
        elif 'LSTM' in layer_class:
            weights = extract_lstm_weights(keras_layer)
            # Custom layer should define how to assign these weights
            if hasattr(custom_layer, 'load_weights_from_keras'):
                custom_layer.load_weights_from_keras(keras_layer)
            else:
                return False
                
        elif 'Bidirectional' in layer_class:
            weights = extract_bidirectional_weights(keras_layer)
            # Custom layer should define how to assign these weights
            if hasattr(custom_layer, 'load_weights_from_keras'):
                custom_layer.load_weights_from_keras(keras_layer)
            else:
                return False
                
        elif 'Embedding' in layer_class:
            weights = extract_embedding_weights(keras_layer)
            custom_layer.weights = weights
            
        elif 'Dropout' in layer_class or 'Flatten' in layer_class:
            # These layers don't have weights
            pass
            
        else:
            print(f"Unsupported layer type: {layer_class}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error loading weights for layer type {layer_class}: {e}")
        return False

def match_layers(custom_model, keras_model):
    custom_layers = custom_model.layers
    keras_layers = keras_model.layers
    
    # Simple check for number of layers
    if len(custom_layers) != len(keras_layers):
        print(f"Warning: Layer count mismatch. Custom model has {len(custom_layers)} layers, "
              f"Keras model has {len(keras_layers)} layers")
    
    # Match layers by position for now
    # A more sophisticated approach could match by layer type and parameters
    return list(zip(custom_layers, keras_layers))

def load_weights(custom_model, keras_model):
    matched_layers = match_layers(custom_model, keras_model)
    success = True
    
    for i, (custom_layer, keras_layer) in enumerate(matched_layers):
        layer_success = load_weights_by_layer_type(custom_layer, keras_layer)
        if not layer_success:
            print(f"Failed to load weights for layer {i}: {type(custom_layer).__name__}")
            success = False
    
    return success