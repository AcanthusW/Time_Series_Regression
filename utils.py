import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_validate

def evaluate_metrics(est, X_test_pp, y_test):
    """
    Evaluate the performance metrics for the given model and dataset.

    Parameters:
    est (object): The trained model to evaluate.
    X_test_pp (DataFrame): The preprocessed test feature dataset.
    y_test (Series): The true values for the target variable.

    Returns:
    dict: Dictionary containing evaluation metrics.
    """
    y_pred = est.predict(X_test_pp)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'EV': explained_variance_score(y_test, y_pred)
    }
    return metrics

def cross_evaluate_scores(reg, X, y, scoring=None):
    """
    Perform cross-validation and return the evaluation scores.

    Parameters:
    reg (object): The model to use for cross-validation.
    X (DataFrame): The feature dataset.
    y (Series): The target variable.
    scoring (list, optional): List of scoring metrics to evaluate.

    Returns:
    DataFrame: Cross-validation scores for each metric.
    """
    if scoring is None:
        scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'explained_variance']
    
    cv_results = cross_validate(reg, X, y, scoring=scoring, cv=5, return_train_score=False)
    scores = {metric: -cv_results[f'test_{metric}'] if 'neg_' in metric else cv_results[f'test_{metric}'] for metric in scoring}
    return pd.DataFrame(scores).agg(['mean', 'std']).transpose()

def plot_tr_spectrum_with_annotations(freq_vel_tr, ax=None):
    """
    Plot the training spectrum with annotations for peaks.

    Parameters:
    freq_vel_tr (DataFrame): Frequency and velocity data to plot.
    ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis on which to plot the data.

    Returns:
    None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(freq_vel_tr['Frequency'], freq_vel_tr['Velocity'], label='Spectrum')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('Training Spectrum with Annotations')
    ax.legend()
    plt.tight_layout()
    plt.show()

def PFSpectraFeatExtr(agg_spectrum_window_size=35, spectrum_window_size=35, max_no_peaks=20):
    """
    Perform feature extraction on the spectra using specified parameters.

    Parameters:
    agg_spectrum_window_size (int): Aggregation window size for spectrum.
    spectrum_window_size (int): Window size for smoothing the spectrum.
    max_no_peaks (int): Maximum number of peaks to extract features from.

    Returns:
    dict: Feature extraction configuration dictionary.
    """
    return {
        'agg_spectrum_window_size': agg_spectrum_window_size,
        'spectrum_window_size': spectrum_window_size,
        'max_no_peaks': max_no_peaks
    }