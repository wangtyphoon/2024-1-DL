import pandas as pd
import xarray as xr
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
def analyze_typhoon_size(df_path, ds_path, cluster_type=0):
    # Read CSV file
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    cluster_size_sum = 0  # Accumulated size index for members with cluster == cluster_type
    cluster_member_count = 0  # Count of members with cluster == cluster_type

    # Iterate through each row in the cluster column
    for _, row in df.iterrows():
        if row['cluster'] == cluster_type:
            # Get the member name from the 'member' column
            member_name = row['member']
            try:
                dataset = ds.sel(model=member_name)
                route = pd.read_csv(f"../route/20170909/csv/group1/{member_name}.csv")
                print(f"Successfully loaded {member_name}")

                # Retrieve longitudes, latitudes, and times
                longitudes = dataset.longitude.values
                latitudes = dataset.latitude.values
                times = route['times']
                
                # Initialize columns for size index
                route["size_index"] = np.nan

                member_size_index = 0
                count = 0

                # Loop over each time step
                for time_index, time in enumerate(times):
                    if time_index < 72:
                        data_at_time = dataset.sel(time=time)

                        # Define the center position
                        lon_center = route["lons"][time_index]
                        lat_center = route["lats"][time_index]

                        # Calculate distances from center to all grid points
                        distances = []
                        for lon in longitudes:
                            for lat in latitudes:
                                current_distance = geopy.distance.distance((lat_center, lon_center), (lat, lon)).km
                                distances.append((lat, lon, current_distance))

                        # Sort by distance and select the nearest 20 grid points
                        distances_sorted = sorted(distances, key=lambda x: x[2])
                        nearest_20_points = distances_sorted[:25]

                        # Calculate wind speed squared sum for the 20 nearest grid points
                        wind_speed_squared_sum = 0
                        for lat, lon, _ in nearest_20_points:
                            u1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
                            v1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
                            
                            # Calculate wind speed magnitude and square it
                            wind_speed_squared = np.sqrt(u1000**2 + v1000**2)
                            
                            # Accumulate wind speed squared
                            wind_speed_squared_sum += wind_speed_squared

                        # Store the size index (accumulated wind speed squared) for this time step
                        route.at[time_index, "size_index"] = wind_speed_squared_sum

                        # Accumulate the total size index for the member
                        member_size_index += wind_speed_squared_sum
                        count += 1

                # Calculate the average size index over the 72-hour period if data is available
                if count > 0:
                    average_size_index = member_size_index / count
                    print(f"{member_name} average size index within 72 hours: {average_size_index:.2f}")

                    # Add to the cluster's total size index
                    cluster_size_sum += average_size_index
                    cluster_member_count += 1
                else:
                    print(f"{member_name} has insufficient data for size index calculation within 72 hours.")

                # Save the updated route with size index information
                route.to_csv(f"../route/20170909/csv/group1/{member_name}.csv", index=False)
            except FileNotFoundError:
                print(f"File not found for {member_name}")

    # Calculate the average size index for the cluster type
    if cluster_member_count > 0:
        cluster_average_size_index = cluster_size_sum / cluster_member_count
        print(f"Cluster {cluster_type} average size index within 72 hours: {cluster_average_size_index:.2f}")
    else:
        print(f"No sufficient data for calculating average size index for cluster {cluster_type}.")

# Example usage
# analyze_typhoon_size(df_path="../route/20170909/combined_features.csv", ds_path="../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc")

def cluster_typhoon_speed(df_path, ds_path, cluster_list):
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

        # Calculate max size_index by time and plot
        if not cluster_route.empty:
            # Ensure the size_index data type is numeric for aggregation
            cluster_route['size_index'] = pd.to_numeric(cluster_route['size_index'], errors='coerce')
            
            # Drop rows with NaN values to avoid aggregation errors
            cluster_route = cluster_route.dropna(subset=['size_index', 'times'])
            
            # Group by time and aggregate size_index based on the cluster
            cluster_route_max = cluster_route.groupby('times')['size_index'].mean().reset_index()
            # if cluster == 0:
            #     cluster_route_max = cluster_route.groupby('times')['size_index'].sum().div(35).reset_index()
            # elif cluster == 1:
            #     cluster_route_max = cluster_route.groupby('times')['size_index'].sum().div(15).reset_index()
            
            # Plotting the results
            plt.plot(cluster_route_max['times'], cluster_route_max['size_index'], marker='o', linestyle='-', label=f'Cluster {cluster} Max Size Index')

    plt.xlabel('Time')
    plt.ylabel('Size Index (m/s)')
    plt.title(f'Size Index Over Time for Clusters {cluster_list}')
    plt.legend()
    plt.grid(True)
    plt.savefig("../route/20170909/cluster/mean_size_index.png")
    plt.show()

# Example usage
cluster_typhoon_speed("../route/20170909/combined_features.csv", "../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc", cluster_list=[0, 1])


# import pandas as pd
# import xarray as xr
# import numpy as np
# import geopy.distance

# def analyze_typhoon_size_for_one_member(df_path, ds_path, member_name, max_distance_km=250):
#     # Read CSV file
#     df = pd.read_csv(df_path)
#     ds = xr.open_dataset(ds_path)

#     # Get specific member from the dataset
#     try:
#         dataset = ds.sel(model=member_name)
#         route = pd.read_csv(f"../route/20170909/csv/{member_name}.csv")
#         print(f"Successfully loaded {member_name}")

#         # Retrieve longitudes, latitudes, and times
#         longitudes = dataset.longitude.values
#         latitudes = dataset.latitude.values
#         times = route['times']
        
#         # Initialize columns for size index
#         route["size_index"] = np.nan

#         # Loop over each time step
#         for time_index, time in enumerate(times):
#             if time_index < 72:
#                 data_at_time = dataset.sel(time=time)

#                 # Define the center position
#                 lon_center = route["lons"][time_index]
#                 lat_center = route["lats"][time_index]

#                 # Calculate distances from center to all grid points
#                 distances = []
#                 for lon in longitudes:
#                     for lat in latitudes:
#                         current_distance = geopy.distance.distance((lat_center, lon_center), (lat, lon)).km
#                         distances.append((lat, lon, current_distance))

#                 # Sort by distance and select the nearest 16 grid points
#                 distances_sorted = sorted(distances, key=lambda x: x[2])
#                 nearest_16_points = distances_sorted[:20]

#                 # Calculate wind speed squared sum for the 16 nearest grid points
#                 wind_speed_squared_sum = 0
#                 for lat, lon, _ in nearest_16_points:
#                     u1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
#                     v1000 = data_at_time.sel(level=1000, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
                    
#                     # Calculate wind speed magnitude and square it
#                     wind_speed_squared = np.sqrt(u1000**2 + v1000**2)
                    
#                     # Accumulate wind speed squared
#                     wind_speed_squared_sum += wind_speed_squared

#                 # Store the size index (accumulated wind speed squared) for this time step
#                 route.at[time_index, "size_index"] = wind_speed_squared_sum

#         # Save the updated route with size index information
#         # route.to_csv(f"../route/20170909/csv/{member_name}_size.csv", index=False)
#         print(f"Updated route with typhoon size index saved for {member_name}")
#         print(route)
#     except FileNotFoundError:
#         print(f"File not found for {member_name}")

# # Example usage for a specific member
# analyze_typhoon_size_for_one_member("../route/20170909/combined_features.csv", "../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc", member_name="NeuralGCM_member_0", max_distance_km=250)
