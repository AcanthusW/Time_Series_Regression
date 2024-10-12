import pandas as pd
import holoviews as hv
from holoviews import opts
import numpy as np
import utils

hv.extension('bokeh')

class Visualization:
    def __init__(self, file_path, target='flow_rate', cmap='viridis', width=800, height=600, xlabel='Frequency (Hz)',
                 ylabel='Flow Rate (m³/h)', zlabel='Velocity (mm/s)', title='Raw Data'):
        """
        Initialize the Visualization class with configuration parameters.

        Parameters:
        file_path (str): Path to the CSV file containing the data.
        target (str): Column name for the target variable (default is 'flow_rate').
        cmap (str): Colormap to use for the plots (default is 'viridis').
        width (int): Width of the plot (default is 800).
        height (int): Height of the plot (default is 600).
        xlabel (str): Label for the x-axis (default is 'Frequency (Hz)').
        ylabel (str): Label for the y-axis (default is 'Flow Rate (m³/h)').
        zlabel (str): Label for the z-axis (default is 'Velocity (mm/s)').
        title (str): Title of the plot (default is 'Raw Data').
        """
        self.file_path = file_path
        self.target = target
        self.cmap = cmap
        self.width = width
        self.height = height
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.title = title

    def raw_data_plot(self):
        """
        Create a 3D surface plot for the raw data and save it as an HTML file.

        Parameters:
        save_path (str): Path to save the HTML file (default is 'results/raw_data_plot.html').

        Returns:
        hv.Surface: The Holoviews surface plot object.
        """
        df = pd.read_csv(self.file_path)

        X, y = df.drop(columns=self.target), df[self.target]
        X.columns = X.columns.astype(float)
        X.columns.name = 'freq'

        x = np.array(X.columns)
        y = np.array(df[self.target])
        z = X.values

        surface = hv.Surface((x, y, z))

        surface.opts(
            cmap=self.cmap,
            width=self.width,
            height=self.height,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            zlabel=self.zlabel,
            title=self.title,
            colorbar=True,
            colorbar_opts={'title': self.zlabel}
        )

        hv.save(surface, save_path, fmt='html')
        print(f"Raw data plot saved to {save_path}")
        return surface
    

    def truncated_data_plot(self, n, save_path='results/truncated_data_plot.html'):
        """
        Create a 3D surface plot for the truncated data and save it as an HTML file.

        Parameters:
        n (float): The maximum value to truncate the data.
        save_path (str): Path to save the HTML file (default is 'results/truncated_data_plot.html').

        Returns:
        hv.Surface: The Holoviews surface plot object.
        """
        df = pd.read_csv(self.file_path)

        target = self.target
        X, y = df.drop(columns=target), df[target]
        X.columns = X.columns.astype(float)
        X.columns.name = 'freq'

        x = np.array(X.columns)
        y = np.array(df[target])
        z = X.values
        z_truncated = np.where(z > n, n, z)

        surface = hv.Surface((x, y, z_truncated))

        surface.opts(
            cmap=self.cmap,
            width=self.width,
            height=self.height,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            zlabel=self.zlabel,
            title='Truncated Data',
            colorbar=True,
            colorbar_opts={'title': self.zlabel}
        )

        hv.save(surface, save_path, fmt='html')
        print(f"Truncated data plot saved to {save_path}")
        return surface
    

    def smoothed_data_plot(self, save_path='results/smoothed_data_plot.html'):
        """
        Create a 3D surface plot for the smoothed data and save it as an HTML file.

        Parameters:
        save_path (str): Path to save the HTML file (default is 'results/smoothed_data_plot.html').

        Returns:
        hv.Surface: The Holoviews surface plot object.
        """
        df = pd.read_csv(self.file_path)
        target = self.target

        X, y = df.drop(columns=target), df[target]
        X.columns = X.columns.astype(float)
        X.columns.name = 'freq'

        X_smoothed = X.copy()
        for i in range(len(X)):
            X_smoothed.iloc[i] = utils.convolve_spectrum(X.iloc[i])

        x = np.array(X.columns)
        y = np.array(df[target])
        z_smoothed = X_smoothed.values

        surface_smoothed = hv.Surface((x, y, z_smoothed))

    
        surface_smoothed.opts(
            cmap='plasma',
            width=self.width,
            height=self.height,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            zlabel='Smoothed Velocity',
            title='Smoothed Data',
            colorbar=True,
            colorbar_opts={'title': 'Smoothed Velocity'}
        )

        hv.save(surface_smoothed, save_path, fmt='html')
        print(f"Smoothed data plot saved to {save_path}")
        return surface_smoothed

if __name__ == "__main__":
    file_path = 'data/dataset.csv'
    visualization = Visualization(file_path)
    visualization.raw_data_plot()
    visualization.truncated_data_plot(n=5)
    visualization.smoothed_data_plot()