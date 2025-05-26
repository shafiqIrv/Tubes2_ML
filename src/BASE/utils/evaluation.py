import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def evaluate_predictions(y_true, y_pred, num_classes=None):
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate macro F1-score
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return {
        'accuracy': acc,
        'macro_f1': f1
    }

def compare_keras_vs_scratch(keras_model, scratch_model, x_test, y_test, batch_size=32):

    # Get predictions from both models
    keras_preds = np.argmax(keras_model.predict(x_test, batch_size=batch_size), axis=1)
    scratch_preds = scratch_model.predict(x_test)
    
    # Evaluate each model
    keras_metrics = evaluate_predictions(y_test, keras_preds)
    scratch_metrics = evaluate_predictions(y_test, scratch_preds)
    
    # Calculate agreement between the two models
    agreement = np.mean(keras_preds == scratch_preds)
    
    return {
        'keras_metrics': keras_metrics,
        'scratch_metrics': scratch_metrics,
        'model_agreement': agreement
    }