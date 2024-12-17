import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_bearing(delta_lon, delta_lat):
    angle = np.degrees(np.arctan2(delta_lat, delta_lon))
    bearing = (angle + 360) % 360
    return bearing

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points 
    on the Earth specified in decimal degrees using the Haversine formula.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371  # Radius of Earth in kilometers
    return c * r

def has_north_turn(subset):
    subset = subset.sort_values('times')
    subset['delta_lon'] = subset['lons'].diff()
    subset['delta_lat'] = subset['lats'].diff()
    subset['bearing'] = calculate_bearing(subset['delta_lon'], subset['delta_lat'])
    
    west_moving = subset['bearing'].between(135, 210)
    north_moving = subset['bearing'].between(0, 135)
    
    found_west_movement = False
    consecutive_north_count = 0

    for i in range(len(subset)):
        if west_moving.iloc[i]:
            found_west_movement = True
            consecutive_north_count = 0
        elif found_west_movement and north_moving.iloc[i]:
            consecutive_north_count += 1
            if consecutive_north_count >= 4:
                return True
        else:
            consecutive_north_count = 0

    return False

def plot_paths_from_csv(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    all_features = []  # List to collect all features
    df_paths = pd.DataFrame()  # DataFrame to store the path data for plotting

    for filename in files:
        file_path = os.path.join(directory, filename)
        df_sorted = pd.read_csv(file_path)
        df_sorted = df_sorted[df_sorted['times'] <= 144]

        # Calculate the starting and ending points of each member
        start_end = df_sorted.groupby('member').agg(
            start_lon=('lons', 'first'),
            start_lat=('lats', 'first'),
            end_lon=('lons', 'last'),
            end_lat=('lats', 'last'),
            start_time=('times', 'first'),
            end_time=('times', 'last')
        ).reset_index()

        # Calculate the change in longitude and latitude
        start_end['delta_lon'] = start_end['end_lon'] - start_end['start_lon']
        start_end['delta_lat'] = start_end['end_lat'] - start_end['start_lat']
        start_end['bearing'] = calculate_bearing(start_end['delta_lon'], start_end['delta_lat'])

         # Add latitude when time equals 60 for each member
        lat_at_60 = []
        for member in df_sorted['member'].unique():
            member_df = df_sorted[df_sorted['member'] == member]
            lat_60 = member_df[member_df['times'] == 60]['lats']
            if not lat_60.empty:
                lat_at_60.append(lat_60.iloc[0])
            else:
                print(member)
                lat_at_60.append(member_df['lats'][0])

        start_end['lat_at_60'] = lat_at_60

        # Calculate the total traveled distance for each member
        total_distances = []
        for member in df_sorted['member'].unique():
            member_df = df_sorted[df_sorted['member'] == member]
            lon1, lat1 = member_df['lons'].values[:-1], member_df['lats'].values[:-1]
            lon2, lat2 = member_df['lons'].values[1:], member_df['lats'].values[1:]
            distances = haversine(lon1, lat1, lon2, lat2)
            total_distance = np.sum(distances)
            total_distances.append(total_distance)

        # Calculate the total duration for each member
        start_end['total_duration_hours'] = (pd.to_datetime(start_end['end_time']) - pd.to_datetime(start_end['start_time'])).dt.total_seconds() / 3600.0
        start_end['total_distance_km'] = total_distances

        # Apply the function to check for north turn for each typhoon
        start_end['has_north_turn'] = start_end['member'].apply(
            lambda typhoon: has_north_turn(df_sorted[df_sorted['member'] == typhoon])
        )

        # Classify as 'n' (north turn) or 'w' (westward movement)
        start_end['category'] = start_end['has_north_turn'].apply(lambda x: 1 if x else 0)

        # Collect all features including start and end coordinates
        all_features.append(start_end[['member', 'start_lon', 'start_lat', 'end_lon', 'end_lat', 'bearing', 'total_distance_km', 'total_duration_hours', 'category','lat_at_60']])

        # Add member paths for plotting
        df_sorted['file'] = filename  # Track file origin
        df_paths = pd.concat([df_paths, df_sorted], ignore_index=True)

    # Combine all features into a single DataFrame
    combined_features = pd.concat(all_features, ignore_index=True)

    # Perform clustering if there are enough samples
    if len(combined_features) >= 2:
        # Apply K-means clustering with 2 clusters
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(combined_features[[  'start_lon', 'start_lat','end_lon', 'end_lat', 'bearing'
                                                                        , 'total_duration_hours', 'total_distance_km']])
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        combined_features['cluster'] = kmeans.fit_predict(standardized_features)

         # Calculate and display the number of members in each cluster
        cluster_counts = combined_features['cluster'].value_counts()
        print("\nCluster Counts:")
        print(cluster_counts)

        # Map clusters back to the original DataFrame
        cluster_mapping = combined_features[['member', 'cluster']].set_index('member')
        df_paths['cluster'] = df_paths['member'].map(cluster_mapping['cluster'])

        # # Print the DataFrame with clusters
        print(combined_features['cluster'])

        # Plotting the typhoon paths with clusters
        # Plotting the typhoon paths with clusters
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([105, 155, 0, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels

        # Colors for different clusters
        colors = ['red', 'blue', 'green']

        # Plot each path based on its cluster
        for cluster_label in df_paths['cluster'].unique():
            cluster_data = df_paths[df_paths['cluster'] == cluster_label]
            for member in cluster_data['member'].unique():
                member_data = cluster_data[cluster_data['member'] == member]
                ax.plot(
                    member_data['lons'], member_data['lats'], 
                    color=colors[cluster_label], linewidth=1.5, 
                    label=f'Cluster {cluster_label}' if member == cluster_data['member'].unique()[0] else ""
                )

        plt.title('Typhoon Paths with Clustering and Gridlines')
        plt.legend(loc='upper left')
        plt.savefig(f"../route/20170909/cluster.png")
        plt.show()
    else:
        print("Not enough data to perform clustering.")
    combined_features.to_csv("../route/20170909/combined_features.csv",index=False)

# Define the directory containing route CSV files
directory = "../route/20170909/csv"

# Plot paths from CSV files   
plot_paths_from_csv(directory)
