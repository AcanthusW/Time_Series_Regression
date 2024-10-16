# Time_Series_Regression

Notice!!!: Intended for Non-Commercial Use  

## Overview 

This project is a machine learning pipeline designed to predict flow rates based on high-frequency time-series data. The main components include feature extraction, hyperparameter tuning via grid search, model training, K-Fold cross-validation, and analysis using SHAP values. The repository contains Python scripts and dependencies needed to reproduce the workflow.

```plaintext
├── main.py              # Main script
├── utils.py             # Helper functions
├── plot_3d.py           # 3D visualizations
├── requirements.txt     # All about environment
├── README.md            # Readme documentation
├── origianl_analysis_process.ipynb       # original analysis process, including everything
├── data/                # Data storage
└── results/             # Output plots and model
```

## Requirements

You can install all the dependencies using the following command:

   ```sh
   pip install -r requirements.txt
   ```

## Usage Instructions

### Input Data

Ensure you have the input dataset (`dataset.csv`) in the `data` directory. The script reads this dataset for training and evaluation. You can modify the `dataset_path` variable in `main.py` if your dataset is located elsewhere.

### Running the 3D Plotting Script (`plot_3d.py`)

The `plot_3d.py` script provides 3D visualizations of the dataset, which are saved as HTML files for interactive exploration. To run the plotting functions, use the following command:

   ```sh
   python plot_3d.py
   ```

This command will generate and save the following 3D plots to the `results` directory:

- **Raw Data Plot**: A 3D surface visualization of the raw dataset (`raw_data_plot.html`).
- **Truncated Data Plot**: A 3D surface plot with truncated values for better visualization (`truncated_data_plot.html`).
- **Smoothed Data Plot**: A 3D surface plot of the smoothed version of the data (`smoothed_data_plot.html`).


### Running the Project

To run the project, you need to execute the main script. This script will handle the entire pipeline, from loading the dataset to training and evaluating the model, and generating SHAP analysis plots.

   ```sh
   python main.py
   ```

### Output

The following outputs are generated and saved to the results directory:

- **K-Fold Cross Validation Plot**: Shows the R² score for each fold during cross-validation (`results/kfold_cross_validation.png`).

- **SHAP Summary Plot**:: SHAP analysis of feature importance for the trained model (`results/shap_summary.png`).

- **PCA SHAP Summary Plot**: SHAP analysis of PCA-transformed features (`results/shap_summary_pca.png`).

- **Trained Random Forest Model**: The final model saved for future use (`results/random_forest_model.pkl`).

## Detailed Explanation

### Grid Search

The `main.py` script performs a grid search over different feature extraction and window parameters. The search explores combinations of:

- **Method**: Aggregation methods including `mean`, `std`, `median`, and `var`.
- **Window Function**: Windowing methods such as `hann`, `gaussian`, `triang`, `blackman`, `hamming`, `bartlett`, and `kaiser`.
- **Max Number of Peaks**: Ranges from 25 to 45, to define how many peaks are used in feature extraction.

The best parameters are selected based on the R² score on the validation set, and the final model is saved to `results/random_forest_model.pkl`.

### SHAP Analysis

SHAP values are used to interpret the trained model. The script produces:

- A SHAP summary plot of the original features (`shap_summary.png`).
- A SHAP summary plot of the PCA-transformed features (`shap_summary_pca.png`). These plots help in understanding the feature contributions to the model predictions.


### PCA Analysis

Principal Component Analysis (PCA) is applied to reduce the feature dimensionality. The SHAP analysis is also performed on PCA-transformed features to explore feature interactions in reduced dimensions.


## Contribution

Contributions are welcome! Feel free to open issues or pull requests for suggestions or improvements.

## Contact

For any questions or inquiries, please contact the repository owner.