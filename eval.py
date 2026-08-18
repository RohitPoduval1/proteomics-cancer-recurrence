# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score
#
# def _generate_probabilities(model: torch.nn.Module, X_prot_tensor: torch.Tensor, X_phos_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
#     """Helper to generate probabilities from the multi-omics model."""
#     model.eval()
#     with torch.no_grad():
#         X_prot_tensor = X_prot_tensor.to(device)
#         X_phos_tensor = X_phos_tensor.to(device)
#         
#         # Unpack the 4 outputs from the MOSAE forward pass (we only need logits)
#         _, _, _, logits = model(X_prot_tensor, X_phos_tensor)
#         
#         probabilities = torch.sigmoid(logits).cpu().numpy()
#     return probabilities
#
# def _find_optimal_probability_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
#     """Finds the optimal threshold using Youden's J statistic."""
#     optimal_idx = np.argmax(tpr - fpr)
#     return thresholds[optimal_idx]
#
# def _plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, study_name: str):
#     """Plots the ROC curve."""
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{study_name} ROC curve (AUC = {roc_auc:.3f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
#     plt.xlim([-0.02, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
#     plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
#     plt.title(f'Zero-Shot Cross-Cancer Recurrence Prediction ({study_name})', fontsize=14)
#     plt.legend(loc="lower right", fontsize=12)
#     plt.grid(alpha=0.3)
#     plt.show()
#
# def evaluate_zero_shot(
#         model: torch.nn.Module,
#         X_prot_tensor: torch.Tensor,
#         X_phos_tensor: torch.Tensor,
#         y_tensor: torch.Tensor,
#         device: torch.device,
#         study_name: str = "PDAC"
# ):
#     """
#     Evaluates the multi-omics model on the provided test set, plots ROC, and prints classification metrics.
#     
#     Args:
#         model (torch.nn.Module): The trained MOSAE PyTorch model.
#         X_prot_tensor (torch.Tensor): Evaluation proteomics feature tensor.
#         X_phos_tensor (torch.Tensor): Evaluation phosphoproteomics feature tensor.
#         y_tensor (torch.Tensor): Evaluation label tensor.
#         device (torch.device): The device (CPU/GPU/MPS) to run inference on.
#         study_name (str): The name of the cohort being evaluated for plot labeling.
#     """
#     probabilities = _generate_probabilities(model, X_prot_tensor, X_phos_tensor, device)
#     true_labels = y_tensor.numpy()
#
#     # Calculate ROC AUC
#     fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
#     roc_auc = auc(fpr, tpr)
#     
#     # Calculate PR AUC
#     pr_auc = average_precision_score(true_labels, probabilities)
#
#     # Plot ROC
#     _plot_roc_curve(fpr, tpr, roc_auc, study_name)
#
#     optimal_threshold = _find_optimal_probability_threshold(fpr, tpr, thresholds)
#     
#     # Print unified metrics
#     print(f"\n--- {study_name} Model Metrics ---")
#     print(f"ROC AUC: {roc_auc:.4f}")
#     print(f"PR AUC:  {pr_auc:.4f}")
#     print(f"Optimal Probability Threshold: {optimal_threshold:.4f}")
#
#     # Generate predictions using the optimal cutoff
#     optimal_predicted_classes = (probabilities >= optimal_threshold).astype(int)
#
#     print(f"\nClassification Report (Cutoff = {optimal_threshold:.4f}):")
#     print(classification_report(true_labels, optimal_predicted_classes, target_names=["No Recurrence", "Recurrence"]))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score

def _generate_probabilities(model: torch.nn.Module, X_prot_tensor: torch.Tensor, X_phos_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Helper to generate probabilities from the multi-omics model."""
    model.eval()
    with torch.no_grad():
        X_prot_tensor = X_prot_tensor.to(device)
        X_phos_tensor = X_phos_tensor.to(device)
        
        # Unpack the 4 outputs from the MOSAE forward pass (we only need logits)
        _, _, _, logits = model(X_prot_tensor, X_phos_tensor)
        
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities

def _find_threshold_for_target_recall(tpr: np.ndarray, thresholds: np.ndarray, target_recall: float) -> float:
    """
    Finds the highest probability threshold that achieves at least the target recall.
    scikit-learn's roc_curve returns thresholds in decreasing order, and tpr in increasing order.
    """
    # Find the first index where the True Positive Rate (Recall) meets or exceeds the target
    valid_indices = np.where(tpr >= target_recall)[0]
    
    if len(valid_indices) == 0:
        # Fallback to the lowest threshold (maximum recall) if target is unreachable
        return thresholds[-1]
    
    optimal_idx = valid_indices[0]
    return thresholds[optimal_idx]

def _plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, study_name: str, target_recall: float, selected_fpr: float):
    """Plots the ROC curve and highlights the selected operating point."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{study_name} ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    
    # Highlight the chosen threshold point prioritizing recall
    plt.scatter([selected_fpr], [target_recall], color='red', s=50, zorder=5, label=f'Selected Point (Recall >= {target_recall})')
    plt.axhline(y=target_recall, color='red', linestyle=':', alpha=0.5)
    plt.axvline(x=selected_fpr, color='red', linestyle=':', alpha=0.5)

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title(f'Zero-Shot Cross-Cancer Recurrence Prediction ({study_name})', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

def evaluate_zero_shot(
        model: torch.nn.Module,
        X_prot_tensor: torch.Tensor,
        X_phos_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        device: torch.device,
        study_name: str = "PDAC",
        target_recall: float = 0.90
):
    """
    Evaluates the multi-omics model on the provided test set, optimizing the decision
    threshold to guarantee a minimum target recall for the positive class.
    
    Args:
        model (torch.nn.Module): The trained MOSAE PyTorch model.
        X_prot_tensor (torch.Tensor): Evaluation proteomics feature tensor.
        X_phos_tensor (torch.Tensor): Evaluation phosphoproteomics feature tensor.
        y_tensor (torch.Tensor): Evaluation label tensor.
        device (torch.device): The device (CPU/GPU/MPS) to run inference on.
        study_name (str): The name of the cohort being evaluated for plot labeling.
        target_recall (float): The minimum acceptable recall for the 'Recurrence' class.
    """
    probabilities = _generate_probabilities(model, X_prot_tensor, X_phos_tensor, device)
    true_labels = y_tensor.numpy()

    # Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR AUC
    pr_auc = average_precision_score(true_labels, probabilities)

    # Find optimal threshold driven strictly by the target recall constraint
    optimal_threshold = _find_threshold_for_target_recall(tpr, thresholds, target_recall)
    
    # Identify the exact TPR and FPR at this chosen threshold for plotting
    selected_idx = np.where(thresholds == optimal_threshold)[0][0]
    actual_tpr = tpr[selected_idx]
    actual_fpr = fpr[selected_idx]

    # Plot ROC with the selected threshold marked
    _plot_roc_curve(fpr, tpr, roc_auc, study_name, actual_tpr, actual_fpr)

    # Print unified metrics
    print(f"\n--- {study_name} Model Metrics ---")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC:  {pr_auc:.4f}")
    print(f"Target Minimum Recall: {target_recall:.2f}")
    print(f"Recall-Optimized Probability Threshold: {optimal_threshold:.4f}")

    # Generate predictions using the recall-optimized cutoff
    optimal_predicted_classes = (probabilities >= optimal_threshold).astype(int)

    print(f"\nClassification Report (Cutoff = {optimal_threshold:.4f}):")
    print(classification_report(true_labels, optimal_predicted_classes, target_names=["No Recurrence", "Recurrence"]))
