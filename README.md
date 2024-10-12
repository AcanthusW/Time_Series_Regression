# Time_Series_Regression
A repository focused on high-frequency time-series regression with RandomForestRegressor

Notice: Intended for Non-Commercial Use  


├── main.py                 # Main script
├── utils.py                # Helper functions
├── requirements.txt        # All about environment
├── README.md               # Readme documentation
└── data/                   # Data storage


"""
Usage Instructions:

-- plot_3d.py:

1. To create a raw data plot and save it as an HTML file, use the following command:

   ```
   python visualization.py
   ```

   This will generate and save the raw data plot, truncated data plot, and smoothed data plot in the `results` folder.

2. If you want to create only the raw data plot, modify the script in the `__main__` section:

   ```python
   visualization.raw_data_plot()
   ```

   You can specify the file path for your dataset by changing the `file_path` variable.

3. To generate specific plots, comment out or remove the functions you do not need in the `__main__` section.

Example Command Line Execution:

- To generate the raw data plot:
  ```
  python visualization.py
  ```

- Make sure to have the dataset CSV file in the `data` folder, or change `file_path` accordingly.

"""