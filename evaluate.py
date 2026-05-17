import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, labels, model_name, output_dir='outputs'):
    """
    Evaluates the model by computing accuracy, classification report, 
    and generating a confusion matrix heatmap.
    
    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
        labels (list): List of class labels.
        model_name (str): Name of the model (e.g., 'Category Classifier').
        output_dir (str): Directory to save the confusion matrix PNG.
    """
    print(f"\n--- Evaluation: {model_name} ---")
    
    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # 2. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save as PNG
    os.makedirs(output_dir, exist_ok=True)
    filename = model_name.lower().replace(' ', '_') + '_confusion_matrix.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")
