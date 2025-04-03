import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve

def generate_performance_report(y_true, y_pred):
    """Generates classification report, confusion matrix, PR curve, and ROC curve"""

    # 🛑 Check for empty or invalid inputs
    if not y_true or not y_pred:
        print("⚠️ Error: y_true or y_pred is empty! Cannot generate report.")
        return
    
    try:
        # Generate metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Ensure y_true and y_pred contain binary values (0/1) before generating curves
        if len(set(y_true)) > 1:  # At least two classes needed
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_pred)

        # Plot Classification Report
        plt.figure(figsize=(6, 3))
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Blues")
        plt.title("Classification Report")
        plt.show()

        # Plot Confusion Matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="OrRd", 
                    xticklabels=['Incorrect', 'Correct'], 
                    yticklabels=['Incorrect', 'Correct'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        # Plot Precision-Recall Curve (Only if binary classification is detected)
        if len(set(y_true)) > 1:
            plt.figure(figsize=(5, 4))
            plt.plot(recall, precision, marker='.')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.show()

            # Plot ROC Curve
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, marker='.')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.show()

    except Exception as e:
        print(f"❌ Error while generating performance report: {e}")
