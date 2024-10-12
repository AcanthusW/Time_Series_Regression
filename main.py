import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import shap
import joblib
import os
import utils
import time


# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the dataset
dataset_path = 'data/dataset.csv'
df = pd.read_csv(dataset_path)

# Preprocess data
target = 'flow_rate'
X = df.drop(columns=[target])
y = df[target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Define parameter space for Grid Search
param_space = {
    'method': ['mean', 'std', 'median', 'var'],
    'window': ['hann', 'gaussian', 'triang', 'blackman', 'hamming', 'bartlett', 'kaiser'],
    'max_no_peaks': range(25, 46)  # from 25 - 45
}

Best_Parameters = {'method': None, 'window': None, 'max_no_peaks': None}
Best_Score = 0

print('Start grid search, current time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

for method in param_space['method']:
    for window in param_space['window']:
        for max_no_peaks in param_space['max_no_peaks']:
            # Current parameter output
            print('Current parameters:', method, window, max_no_peaks)

            PF_feat_extr = utils.PFSpectraFeatExtr(
                agg_spectrum_window_size=60,
                spectrum_window_size=60,
                max_no_peaks=max_no_peaks,
                fb_peak_freq_power_feat=True,
                fb_dist_shape_auc_feat=False,
                peak_freq_power_feat=False,
                dist_shape_auc_feat=True,
                agg_spectra_method=method,
                signal_window=window,
                peaks_base_level=0.0025
            )

            reg_pipe = Pipeline(
                steps=[
                    ('pp', PF_feat_extr),
                    ('reg', RandomForestRegressor(random_state=0))
                ]
            )
            # Fit the pipeline
            reg_pipe_fitted = reg_pipe.fit(X_train, y_train)

            # Evaluate on the test set
            test_results = utils.evaluate_metrics(reg_pipe_fitted, X_test, y_test)

            # Update best parameters if current score is better
            if test_results['r2'] > Best_Score:
                Best_Score = test_results['r2']
                Best_Parameters['method'] = method
                Best_Parameters['window'] = window
                Best_Parameters['max_no_peaks'] = max_no_peaks


print('Best Parameters: ', Best_Parameters)
print('Best Score(r2): ', Best_Score)
print('End grid search, current time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


# Create a pipeline with the best parameters found
PF_feat_extr_best = utils.PFSpectraFeatExtr(
    agg_spectrum_window_size=60,
    spectrum_window_size=60,
    max_no_peaks=Best_Parameters['max_no_peaks'],
    fb_peak_freq_power_feat=True,
    fb_dist_shape_auc_feat=False,
    peak_freq_power_feat=False,
    dist_shape_auc_feat=True,
    agg_spectra_method=Best_Parameters['method'],
    signal_window=Best_Parameters['window'],
    peaks_base_level=0.0025
)

reg_pipe_best = Pipeline(
    steps=[
        ('pp', PF_feat_extr_best),
        ('reg', RandomForestRegressor(random_state=0))
    ]
)

# Train the model with K-Fold Cross Validation using the best pipeline
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
    X_kf_train, X_kf_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_kf_train, y_kf_val = y_train.iloc[train_index], y_train.iloc[val_index]
    reg_pipe_best.fit(X_kf_train, y_kf_train)
    score = reg_pipe_best.score(X_kf_val, y_kf_val)
    scores.append(score)
    print(f"Fold {fold}: R^2 Score = {score:.4f}")

# Print the average R^2 score across all folds
mean_score = np.mean(scores)
print(f"Average R^2 Score: {mean_score:.4f}")

# Plot K-Fold Cross Validation results using the trained pipeline
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='--', color='b')
plt.xlabel('Fold Number')
plt.ylabel('R^2 Score')
plt.title('K-Fold Cross Validation Results (reg_pipe_best)')
plt.savefig('results/kfold_cross_validation.png')
plt.close()

# Train the final model on the entire training set
reg_pipe_best.fit(X_train, y_train)


# SHAP Analysis
explainer = shap.Explainer(reg_pipe_best['reg'], X_train)
shap_values = explainer(X_test)

# Plot SHAP summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('results/shap_summary.png')
plt.close()

# PCA Analysis
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# SHAP Analysis on PCA-transformed data
explainer_pca = shap.Explainer(reg_pipe_best['reg'], X_pca)
shap_values_pca = explainer_pca(X_pca)

# Plot SHAP summary for PCA-transformed data
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_pca, X_pca, show=False)
plt.savefig('results/shap_summary_pca.png')
plt.close()

# Save the trained model
joblib.dump(reg_pipe_best, 'results/random_forest_model.pkl')