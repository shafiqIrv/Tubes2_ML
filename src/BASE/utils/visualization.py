import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model

def plot_training_history(history, title='Training History'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='lower right')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def compare_model_variants(histories, metric='loss', title='Model Comparison'):
    plt.figure(figsize=(12, 6))
    
    for history, label in histories:
        plt.plot(history.history[f'val_{metric}'], label=label)
    
    plt.title(f'Validation {metric.capitalize()} Comparison')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return plt.gcf()

def visualize_feature_maps(model, image, layer_names=None):
    
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers 
                       if 'conv' in layer.name.lower()]
    
    # Create feature extraction models for each layer
    feature_models = [Model(inputs=model.input, outputs=model.get_layer(name).output) 
                     for name in layer_names]
    
    # Get feature maps
    feature_maps = [feature_model.predict(image) for feature_model in feature_models]
    
    # Plot feature maps
    fig = plt.figure(figsize=(15, len(layer_names) * 5))
    
    for i, (feature_map, name) in enumerate(zip(feature_maps, layer_names)):
        # Get number of filters
        n_filters = feature_map.shape[-1]
        
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        # Plot each filter
        for j in range(min(n_filters, 16)):
            plt.subplot(len(layer_names), grid_size, i * grid_size + j + 1)
            plt.imshow(feature_map[0, :, :, j], cmap='viridis')
            plt.title(f"{name} - Filter {j}")
            plt.axis('off')
    
    plt.tight_layout()
    return fig