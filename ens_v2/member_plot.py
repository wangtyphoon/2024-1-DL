import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re

def plot_paths_from_csv(directory):
    """
    Automatically reads CSV files from the given directory and plots paths based on longitude and latitude with terrain background.
    
    Parameters:
        directory (str): The path to the directory containing CSV files.
    """
    # Get all CSV files and sort them by the numeric value in the filename
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Iterate over sorted CSV files
    for filename in files:
        file_path = os.path.join(directory, filename)
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Ensure required columns are present
        if {'lons', 'lats'}.issubset(df.columns):
            # Create a figure with Cartopy projection for each member
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Set map extent
            ax.set_extent([105, 160, -5, 50], crs=ccrs.PlateCarree())
            
            # Add features to the map
            ax.add_feature(cfeature.LAND, zorder=0)
            ax.add_feature(cfeature.OCEAN, zorder=0)
            ax.add_feature(cfeature.COASTLINE, zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
            ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=1)
            ax.add_feature(cfeature.RIVERS, zorder=1)
            
            # Plot the path
            ax.plot(df['lons'], df['lats'], label=f"Member {filename.split('.')[0]}", transform=ccrs.PlateCarree())
            
            # Configure plot
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f"Typhoon Track - Member {filename.split('.')[0]}")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"../route/20170909/png/track/group2/Typhoon Track - Member {filename.split('.')[0]}.png")
            plt.show()
        else:
            print(f"Skipping {filename}: required columns 'lon' and 'lat' not found.")

# Define the directory containing route CSV files
directory = "../route/20170909/csv/group2"

# Plot paths from CSV files
plot_paths_from_csv(directory)
