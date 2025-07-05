import copy
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from TITAN.titan.utils import get_eval_metrics, seed_torch


def train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels,
                                                    test_slide_id, test_patient_id, log_spaced_values=None, max_iter=500):
    # seed_torch(torch.device('cpu'), 0)
    
    metric_dict = {
        'bacc': 'balanced_accuracy',
        'kappa': 'cohen_kappa_score',
        'auroc': 'roc_auc_score',
    }
    
    # Logarithmically spaced values for regularization
    if log_spaced_values is None:
        log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
    
    # loop over log_spaced_values to find the best C
    best_score = -float('inf')
    best_C = None
    logistic_reg_final = None
    for log2_coeff in tqdm(log_spaced_values, desc="Finding best C"):
        # suppress convergence warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        logistic_reg = LogisticRegression(
            C=1/log2_coeff,
            fit_intercept=True,
            max_iter=max_iter,
            random_state=0,
            solver="lbfgs",
        )
        logistic_reg.fit(train_data, train_labels)
        
        # predict on val set
        try:
            val_loss = log_loss(val_labels, logistic_reg.predict_proba(val_data))
        except ValueError as e:
            val_loss = -balanced_accuracy_score(val_labels, logistic_reg.predict(val_data))

        # print("train info", logistic_reg.score(train_data, train_labels))
        score = -val_loss
        
        # score on val set
        if score > best_score:
            best_score = score
            best_C = log2_coeff
            logistic_reg_final = logistic_reg
    print(f"Best C: {best_C}")
    
    # Evaluate the model on the test data
    test_preds = logistic_reg_final.predict(test_data)
    
    num_classes = len(np.unique(train_labels))
    if num_classes == 2:
        test_probs = logistic_reg_final.predict_proba(test_data)[:, 1]
        roc_kwargs = {}
    else:
        test_probs = logistic_reg_final.predict_proba(test_data)
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

    # aggregate the slide_id to patient_id
    if test_slide_id is None and test_patient_id is None:
        pass
    else:
        agg_df = {
            "slide_id": test_slide_id,
            "patient_id": test_patient_id,
            "targets": test_labels,
            "probs": list(test_probs),
        }
        agg_df = pd.DataFrame(agg_df)
        agg_df = agg_df.groupby("patient_id").agg({
            "targets": "first",
            "probs": "mean",
        }).reset_index()
        test_labels = agg_df["targets"].values
        test_probs = np.vstack(agg_df["probs"].values) if test_probs.ndim > 1 else agg_df["probs"].values
        if test_probs.ndim == 1: # Binary classification
            test_preds = (test_probs > 0.5).astype(int)
        else: # Multi-class classification
            test_preds = np.argmax(test_probs, axis=1)

        print(f"agg from slide {len(test_slide_id)} to patient {len(agg_df)}")

    eval_metrics = get_eval_metrics(test_labels, test_preds, test_probs, roc_kwargs=roc_kwargs)
    
    outputs = {
        "targets": test_labels,
        "preds": test_preds,
        "probs": test_probs,
        "lr_model": copy.deepcopy(logistic_reg_final),
    }
        
    return eval_metrics, outputs


def train_and_evaluate_logistic_regression_with_both_metrics(train_data, train_labels, val_data, val_labels,
                                                             val_slide_id, val_patient_id,
                                                             test_data, test_labels,
                                                             test_slide_id, test_patient_id,
                                                             log_spaced_values=None, max_iter=500):
    """Train and evaluate logistic regression, returning both validation and test metrics
    with patient-level aggregation for both."""

    # Train model and get test metrics using the existing function
    test_metrics, outputs = train_and_evaluate_logistic_regression_with_val(
        train_data, train_labels, val_data, val_labels,
        test_data, test_labels, test_slide_id, test_patient_id,
        log_spaced_values, max_iter
    )

    # Compute validation predictions and probabilities
    val_preds = outputs["lr_model"].predict(val_data)

    num_classes = len(np.unique(train_labels))
    if num_classes == 2:
        val_probs = outputs["lr_model"].predict_proba(val_data)[:, 1]
        roc_kwargs = {}
    else:
        val_probs = outputs["lr_model"].predict_proba(val_data)
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}

    # Aggregate validation predictions at patient level
    if val_slide_id is not None and val_patient_id is not None:
        agg_df = {
            "slide_id": val_slide_id,
            "patient_id": val_patient_id,
            "targets": val_labels,
            "probs": list(val_probs),
        }
        agg_df = pd.DataFrame(agg_df)
        agg_df = agg_df.groupby("patient_id").agg({
            "targets": "first",
            "probs": "mean",
        }).reset_index()
        val_labels = agg_df["targets"].values
        val_probs = np.vstack(agg_df["probs"].values) if val_probs.ndim > 1 else agg_df["probs"].values
        if val_probs.ndim == 1:  # Binary classification
            val_preds = (val_probs > 0.5).astype(int)
        else:  # Multi-class classification
            val_preds = np.argmax(val_probs, axis=1)

        print(f"Val: aggregated from {len(val_slide_id)} slides to {len(agg_df)} patients")

    # Compute validation metrics
    val_metrics = get_eval_metrics(val_labels, val_preds, val_probs, roc_kwargs=roc_kwargs)

    return val_metrics, test_metrics, outputs