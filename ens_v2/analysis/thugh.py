import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import label

def vis_high_pressure(df_path, ds_path, cluster_type=0, min_region_size=100):
    # Read CSV file
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    # Iterate through each row in the cluster column
    for _, row in df.iterrows():
        if row['cluster'] == cluster_type:
            # Get the member name from the 'member' column
            member_name = row['member']
            route = pd.read_csv(f"../../route/20170909/csv/group1/{member_name}.csv")
            try:
                dataset = ds.sel(model=member_name, level=500)
                print(f"Successfully loaded {member_name}")
                
                # Retrieve longitudes, latitudes, and times
                longitudes = dataset.longitude.values
                latitudes = dataset.latitude.values
                times = dataset.time.values
                colors = plt.cm.Reds(np.linspace(0.5, 1, len(times)))  # Use red color map for the contour lines

                # Create a meshgrid for interpolation
                lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

                # Plotting the points with max wind speed
                for i, time in enumerate(times):

                    if time % 24 == 0 and time <= 96 and time >1:
                        geopotential_at_time = dataset['geopotential'].sel(time=time).values

                        # Find regions of geopotential >= 5880 * 9.80665
                        mask = geopotential_at_time >= 5880 * 9.80665
                        labeled_array, num_features = label(mask)

                        # Remove smaller regions based on min_region_size
                        for region in range(1, num_features + 1):
                            region_size = np.sum(labeled_array == region)
                            if region_size < min_region_size:
                                mask[labeled_array == region] = False

                        # Apply mask to geopotential values
                        filtered_geopotential = np.where(mask, geopotential_at_time, np.nan)
                        
                                            # Plot the filtered geopotential data
                        plt.figure(figsize=(10, 6))
                        ax = plt.axes(projection=ccrs.PlateCarree())
                        ax.set_extent([105, 155, 0, 50], crs=ccrs.PlateCarree())
                        ax.coastlines()
                        ax.add_feature(cfeature.BORDERS, linestyle=':')

                        plt.plot(route['lons'][i-4:i], route['lats'][i-4:i], marker='o', linestyle='-', label=f'member {member_name} Average Path')
                        # Plot the interpolated geopotential data
                        contour = ax.contourf(
                            lon_grid.T, lat_grid.T, filtered_geopotential,
                            levels=[5900 * 9.80665, np.nanmax(geopotential_at_time)], colors=[colors[i]],
                            transform=ccrs.PlateCarree()
                        )

                        plt.title(f"Geopotential 5900 for {member_name} at Time {time}")

                        # Save the figure
                        output_filename = f"../../route/20170909/png/height/geopotential_5900_{member_name}_time_{time}.png"
                        plt.savefig(output_filename, bbox_inches='tight')
                        plt.show()
                
            except KeyError:
                print(f"Dataset for {member_name} could not be found in {ds_path}")
            except FileNotFoundError:
                print(f"File not found for {member_name}")

# Call the function with the specified paths
# vis_high_pressure(
#     df_path="../../route/20170909/combined_features.csv",
#     ds_path="../../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc"
# )
# def add_feature(df_path, ds_path, cluster_types, min_region_size=100):
#     df = pd.read_csv(df_path)
#     ds = xr.open_dataset(ds_path)
       
def process_and_compare_clusters(df_path, ds_path, cluster_types, min_region_size=100):
    # Read CSV file
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    cluster_results = {}

    # Calculate average geopotential and average path for each cluster_type
    for cluster_type in cluster_types:
        selected_members = df[df['cluster'] == cluster_type]['member'].values
        aggregated_geopotential = None
        cluster_route = pd.DataFrame()

        print(f"Processing {len(selected_members)} members in cluster {cluster_type}...")

        for member_name in selected_members:
            try:
                member_data = ds.sel(model=member_name, level=500)['geopotential']
                if aggregated_geopotential is None:
                    aggregated_geopotential = member_data
                else:
                    aggregated_geopotential += member_data

                route = pd.read_csv(f"../../route/20170909/csv/group1/{member_name}.csv")
                cluster_route = pd.concat([cluster_route, route], ignore_index=True)

            except KeyError:
                print(f"Dataset for {member_name} could not be found in {ds_path}")
            except FileNotFoundError:
                print(f"File not found for {member_name}")

        avg_geopotential = aggregated_geopotential / len(selected_members)

        if not cluster_route.empty:
            cluster_route['lons'] = pd.to_numeric(cluster_route['lons'], errors='coerce')
            cluster_route['lats'] = pd.to_numeric(cluster_route['lats'], errors='coerce')
            cluster_route = cluster_route.dropna(subset=['lons', 'lats', 'times'])
            cluster_route_avg = cluster_route.groupby('times')[['lons', 'lats']].mean().reset_index()

        cluster_results[cluster_type] = {
            'avg_geopotential': avg_geopotential,
            'route_avg': cluster_route_avg
        }

    # Plot comparison
    times = cluster_results[cluster_types[0]]['avg_geopotential']['time'].values
    longitudes = cluster_results[cluster_types[0]]['avg_geopotential']['longitude'].values
    latitudes = cluster_results[cluster_types[0]]['avg_geopotential']['latitude'].values
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    for i, time in enumerate(times):
        if time % 24 == 0 and time <= 120 and time > 1:
            plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([105, 155, 0, 50], crs=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')

            for cluster_type in cluster_types:
                avg_geopotential = cluster_results[cluster_type]['avg_geopotential']
                cluster_route_avg = cluster_results[cluster_type]['route_avg']
                geopotential_at_time = avg_geopotential.sel(time=time).values

                # Filter geopotential < 5760 * 9.80665
                mask = geopotential_at_time < 5760 * 9.80665
                labeled_array, num_features = label(mask)

                for region in range(1, num_features + 1):
                    region_size = np.sum(labeled_array == region)
                    if region_size < min_region_size:
                        mask[labeled_array == region] = False

                filtered_geopotential = np.where(mask, geopotential_at_time, np.nan)

                contour = ax.contourf(
                    lon_grid.T, lat_grid.T, filtered_geopotential,
                    levels=[np.nanmin(geopotential_at_time), 5760 * 9.80665],
                    alpha=0.5,
                    transform=ccrs.PlateCarree(),
                    cmap='Reds' if cluster_type == cluster_types[1] else 'Blues'
                )

                plt.plot(
                    cluster_route_avg['lons'][i - 4:i],
                    cluster_route_avg['lats'][i - 4:i],
                    marker='o',
                    linestyle='-',
                    label=f'Cluster {cluster_type} Average Path(-24)',
                    color='Red' if cluster_type == cluster_types[1] else 'Blue'
                )

                plt.plot(
                    cluster_route_avg['lons'][i:i + 4],
                    cluster_route_avg['lats'][i:i + 4],
                    marker='^',
                    linestyle='-',
                    label=f'Cluster {cluster_type} Average Path(+24)',
                    color='Red' if cluster_type == cluster_types[1] else 'Blue'
                )

            plt.title(f"Geopotential < 5760 (Filtered) Comparison at Time {time}")
            plt.legend(loc='lower right')            # Save the figure
            output_filename = f"../../route/20170909/cluster/clusters_though_time_{time}.png"
            plt.savefig(output_filename, bbox_inches='tight')
            plt.show()

# Example usage
# process_and_compare_clusters(
#     df_path="../../route/20170909/combined_features.csv",
#     ds_path="../../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc",
#     cluster_types=[0, 1]
# )

def feature_high_pressure(df_path, ds_path, cluster_type=0, min_region_size=100):
    # Read CSV and NetCDF data
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    # Iterate through clusters
    for _, row in df.iterrows():
        if row['cluster'] == cluster_type:
            member_name = row['member']
            route = pd.read_csv(f"../../route/20170909/csv/group1/{member_name}.csv")
            route  = route.drop('nearest_low_pressure_distance', axis=1)
            # Initialize feature columns
            route['average_low_pressure_area'] = 0.0
            route['peak_low_pressure'] = np.nan


            dataset = ds.sel(model=member_name, level=500)
            print(f"Successfully loaded {member_name}")

            # Extract coordinates and time
            longitudes = dataset.longitude.values
            latitudes = dataset.latitude.values

            # Process each timestep
            for i, time in enumerate(route['times'].values):
                geopotential_at_time = dataset['geopotential'].sel(time=time).values

                # Identify high-pressure regions
                mask = geopotential_at_time <= 5760 * 9.80665
                labeled_array, num_features = label(mask)

                # Filter out small regions
                for region in range(1, num_features + 1):
                    region_size = np.sum(labeled_array == region)
                    if region_size < min_region_size:
                        mask[labeled_array == region] = False

                # Re-label regions and calculate areas
                labeled_array, num_features = label(mask)
                areas = [np.sum(labeled_array == region) for region in range(1, num_features + 1) if np.sum(labeled_array == region) >= min_region_size]
                average_area = np.mean(areas) if areas else 0
                route.at[i, 'average_low_pressure_area'] = average_area

                # Calculate peaks in geopotential
                peaks = [np.min(geopotential_at_time[labeled_array == region]) for region in range(1, num_features + 1)]
                if peaks:
                    route.at[i, 'peak_low_pressure'] = max(peaks) / 9.80665
                else:
                    route.at[i, 'peak_low_pressure'] = np.nan

            # Debugging output to inspect the modified DataFrame
            route.to_csv(f"../../route/20170909/csv/group1/{member_name}.csv", index=False)



feature_high_pressure(
    df_path="../../route/20170909/combined_features.csv",
    ds_path="../../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc",
)
