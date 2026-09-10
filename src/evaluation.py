# src/evaluation.py

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

def evaluate_metrics(y_true, y_pred, y_probs):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_probs) if y_probs is not None else 0.0,
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "Classification Report": classification_report(y_true, y_pred, target_names=["Control", "ASD"], digits=4)
    }