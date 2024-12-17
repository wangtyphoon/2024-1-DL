import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopy.distance

# def plot_wind_speed_contours(df_path, ds_path, member_name, max_distance_km=200): 
#     # Load dataset and route data for the specified member 
#     df = pd.read_csv(df_path) 
#     ds = xr.open_dataset(ds_path) 
     
#     # Check if the member exists in the DataFrame 
#     if member_name not in df['member'].values: 
#         print(f"Member {member_name} not found in the data.") 
#         return 
     
#     # Load member data 
#     dataset = ds.sel(model=member_name)
#     route = pd.read_csv(f"../route/20170909/csv/{member_name}.csv") 
#     print(f"Successfully loaded {member_name}") 
 
#     # Extract times 
#     times = route['times'] 
     
#     # Loop over each time step 
#     for time_index, time in enumerate(times): 
#         # Process only if time index is less than 72 
#         if time_index >= 72: 
#             break 
 
#         data_at_time = dataset.sel(time=time)

#         # Define the center position 
#         lon_center = route["lons"][time_index] 
#         lat_center = route["lats"][time_index] 
 
#         # Calculate wind speed for each grid point in the entire domain
#         lon_grid, lat_grid = np.meshgrid(dataset.longitude.values, dataset.latitude.values)
#         wind_speeds = []
#         max_wind_within_radius = -np.inf
#         max_wind_location_within_radius = None

#         for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten()):
#             u1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
#             v1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
#             wind_speed = np.sqrt(u1000**2 + v1000**2) / 0.51444444  # Convert to knots
#             wind_speeds.append((lon, lat, wind_speed))
            
#             # Calculate distance from the center for the 200 km limit
#             current_distance = geopy.distance.distance((lat_center, lon_center), (lat, lon)).km
#             if current_distance <= max_distance_km and wind_speed > max_wind_within_radius:
#                 max_wind_within_radius = wind_speed
#                 max_wind_location_within_radius = (lon, lat)
#         print(max_wind_within_radius)
#         # Convert to a DataFrame for easier processing 
#         wind_speeds_df = pd.DataFrame(wind_speeds, columns=["lon", "lat", "wind_speed"]) 
#         wind_speeds_pivot = wind_speeds_df.pivot(index="lat", columns="lon", values="wind_speed") 
         
#         # Plotting 
#         plt.figure(figsize=(8, 6)) 
#         plt.contourf(wind_speeds_pivot.columns, wind_speeds_pivot.index, wind_speeds_pivot.values, cmap="viridis") 
#         plt.colorbar(label="Wind Speed (knots)") 
#         plt.scatter(lon_center, lat_center, color="red", marker="x", label="Center") 
#         if max_wind_location_within_radius:
#             plt.scatter(max_wind_location_within_radius[0], max_wind_location_within_radius[1], color="blue", marker="o", label=f"Max Wind ({max_wind_within_radius:.2f} knots)") 
         
#         plt.title(f"Wind Speed Contour at Time Step {time_index + 1}") 
#         plt.xlabel("Longitude") 
#         plt.ylabel("Latitude") 
#         plt.legend() 
#         plt.grid(True) 
#         plt.show() 

# # Usage 
# # plot_wind_speed_contours(df_path="../route/20170909/combined_features.csv",  
# #                          ds_path="../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc", 
# #                          member_name="NeuralGCM_member_1")


def analyze_max_wind_speed(df_path, ds_path, max_distance_km=250, cluster_type=1):
    # Read CSV file
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    cluster_max_speed_sum = 0  # Accumulated max wind speed for members with cluster == cluster_type
    cluster_member_count = 0  # Count of members with cluster == cluster_type

    # Iterate through each row in the cluster column
    for _, row in df.iterrows():
        if row['cluster'] == cluster_type:
            # Get the member name from the 'member' column
            member_name = row['member']
            try:
                dataset = ds.sel(model=member_name)
                route = pd.read_csv(f"../route/20170909/csv/group2/{member_name}.csv")
                print(f"Successfully loaded {member_name}")

                # Retrieve longitudes, latitudes, and times
                longitudes = dataset.longitude.values
                latitudes = dataset.latitude.values
                times = route['times']
                
                # Initialize columns for max wind speed
                route["max_wind_speed"] = np.nan

                member_max_speed = 0
                count = 0

                # Loop over each time step
                for time_index, time in enumerate(times):
                    # Process only if time index is less than 72
                    if time_index < 72:
                        data_at_time = dataset.sel(time=time)

                        # Define the center position
                        lon_center = route["lons"][time_index]
                        lat_center = route["lats"][time_index]

                        max_speed = 0

                        # Loop over all longitudes and latitudes
                        for lon in longitudes:
                            for lat in latitudes:
                                # Calculate distance from the current point to the center
                                current_distance = geopy.distance.distance((lat_center, lon_center), (lat, lon)).km

                                # If within 200 km, calculate wind speed
                                if current_distance <= max_distance_km:
                                    u1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
                                    v1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
                                    
                                    # Calculate wind speed magnitude
                                    wind_speed = np.sqrt(u1000**2 + v1000**2)
                                    
                                    # Track maximum speed within 200 km
                                    max_speed = max(max_speed, wind_speed)

                        # Store the maximum wind speed
                        route["max_wind_speed"][time_index] = max_speed   # Convert from m/s to knots

                        # Accumulate member's max wind speed
                        member_max_speed += max_speed 
                        count += 1

                # Calculate the average maximum wind speed over the 72-hour period if data is available
                if count > 0:
                    average_max_speed = member_max_speed / count
                    print(f"{member_name} average max wind speed within 72 hours: {average_max_speed:.2f} knots")

                    # Add to the cluster's total max wind speed
                    cluster_max_speed_sum += average_max_speed
                    cluster_member_count += 1
                else:
                    print(f"{member_name} has insufficient data for max wind speed calculation within 72 hours.")

                route.to_csv(f"../route/20170909/csv/group2/{member_name}.csv", index=False)
            except FileNotFoundError:
                print(f"File not found for {member_name}")

    # Calculate the average max wind speed for the cluster type
    if cluster_member_count > 0:
        cluster_average_max_speed = cluster_max_speed_sum / cluster_member_count
        print(f"Cluster {cluster_type} average max wind speed within 72 hours: {cluster_average_max_speed:.2f} knots")
    else:
        print(f"No sufficient data for calculating average max wind speed for cluster {cluster_type}.")

# Usage
analyze_max_wind_speed(df_path="../route/20170909/combined_features_group2.csv", ds_path="../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc")

def cluster_max_wind_speed(df_path, ds_path, cluster_list):
    # Read CSV file
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    plt.figure(figsize=(10, 6))
    for cluster in cluster_list:
        cluster_route = pd.DataFrame()
        for _, row in df.iterrows():
            if row['cluster'] == cluster:
                # Get member name and load its route file
                member_name = row['member']
                try:
                    dataset = ds.sel(model=member_name)
                    route = pd.read_csv(f"../route/20170909/csv/group1/{member_name}.csv")
                    
                    # Append route to cluster_route
                    cluster_route = pd.concat([cluster_route, route], ignore_index=True)
                except FileNotFoundError:
                    print(f"File for member {member_name} not found.")
                    continue

        # Calculate max wind speed by time and plot
        if not cluster_route.empty:
            # Ensure the wind speed data type is numeric for aggregation
            cluster_route['max_wind_speed'] = pd.to_numeric(cluster_route['max_wind_speed'], errors='coerce')
            # Drop rows with NaN values to avoid aggregation errors
            cluster_route = cluster_route.dropna(subset=['max_wind_speed', 'times'])
            cluster_route_max = cluster_route.groupby('times')['max_wind_speed'].mean().reset_index()

            # # Group by time and aggregate max wind speed based on the cluster
            # if cluster == 0:
            #     cluster_route_max = cluster_route.groupby('times')['max_wind_speed'].sum().div(35).reset_index()
            # elif cluster == 1:
            #     cluster_route_max = cluster_route.groupby('times')['max_wind_speed'].sum().div(15).reset_index()
            # Plotting the results
            plt.plot(cluster_route_max['times'], cluster_route_max['max_wind_speed'], marker='o', linestyle='-', label=f'Cluster {cluster} Max Wind Speed')

    plt.xlabel('Time')
    plt.ylabel('Max Wind Speed (m/s)')
    plt.title(f'Max Wind Speed Over Time for Clusters {cluster_list}')
    plt.legend()
    plt.grid(True)
    plt.savefig("../route/20170909/cluster/mean_wind_speed.png")
    plt.show()

# Example usage
# cluster_max_wind_speed("../route/20170909/combined_features.csv", "../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc", cluster_list=[0, 1])
