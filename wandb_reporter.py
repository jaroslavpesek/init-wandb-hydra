import numpy as np
import wandb
from sklearn.metrics import classification_report


def wandb_report(true_labels, pred_labels, target_names):
    report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]

    # Get the unique classes present in true_labels and pred_labels
    present_classes = np.unique(np.concatenate([true_labels, pred_labels]))

    # Filter the target_names and true_labels, pred_labels accordingly
    filtered_target_names = [target_names[i] for i in present_classes]
    filtered_true_labels = [present_classes.tolist().index(label) for label in true_labels]
    filtered_pred_labels = [present_classes.tolist().index(label) for label in pred_labels]

    class_report = classification_report(filtered_true_labels, filtered_pred_labels, target_names=filtered_target_names, zero_division=0).splitlines()

    report_table = []
    for line in class_report[2:(len(filtered_target_names) + 2)]:
        report_table.append(line.split())

    wandb.log({
        f"eval/cm": wandb.plot.confusion_matrix(y_true=filtered_true_labels, preds=filtered_pred_labels, class_names=filtered_target_names),
        f"eval/report": wandb.Table(data=report_table, columns=report_columns),
    })

