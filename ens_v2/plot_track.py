# Handle different ensemble members
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from distances import haversine  # 如果在同一目錄下
from membercsv import member_to_csv
import numpy as np
def process_track(all_classified_typhoons, ensemble_members):
    """
    繪製兩個模型（ERA5 和 NeuralGCM）的颱風路徑，並分別用不同標記區分。
    
    Parameters:
        all_classified_typhoons (list of tuples): 每個時間步驟的颱風信息和時間。
        longitude (1D array): 經度座標。
        latitude (1D array): 緯度座標。
        time_step_hours (int): 每個步驟的時間間隔，默認為24小時。
        ensemble_members (list of str): NeuralGCM的系集成員列表。
    """
    typhoon_tracks_era5 = {}  # 用於儲存ERA5模型的颱風路徑
    typhoon_tracks_neuralgcm = {}  # 用於儲存NeuralGCM模型的颱風路徑
    track_id_counter_era5 = 1
    track_id_counter_neuralgcm = 1
    previous_typhoons_era5 = {}
    previous_typhoons_neuralgcm = {}

    distance_threshold_km = 350  # 距離閾值
    old_model = "ERA5"
    system_id = 1
    
    # Longitude threshold to separate the two groups
    longitude_threshold = 135
    latitude_threshold =  18.5    # Adjust this threshold as needed

    for time_index, (classified_typhoons, time_hour) in enumerate(all_classified_typhoons):
        current_typhoons_era5 = {}
        current_typhoons_neuralgcm = {}

        for typhoon in classified_typhoons:
            model = typhoon.get('model', 'Unknown')  # 確保模型名稱正確

            # 跳過溫帶氣旋
            if typhoon['cyclone_type'] == 'Extratropical Cyclone':
                continue

            if model != old_model:
                old_model = model
                system_id = 1

            matched = False
            if model == "ERA5":
                # 比較ERA5模型的颱風
                for prev_id, prev_typhoon in previous_typhoons_era5.items():
                    distance = haversine(typhoon['longitude'], typhoon['latitude'], prev_typhoon['longitude'], prev_typhoon['latitude'])
                    if distance < distance_threshold_km:
                        # 更新颱風路徑
                        typhoon_tracks_era5[prev_id]['lons'].append(typhoon['longitude'])
                        typhoon_tracks_era5[prev_id]['lats'].append(typhoon['latitude'])
                        typhoon_tracks_era5[prev_id]['types'].append(typhoon['cyclone_type'])
                        typhoon_tracks_era5[prev_id]['times'].append(time_hour)
                        current_typhoons_era5[prev_id] = typhoon
                        matched = True
                        # break

                if not matched:
                    # 新增ERA5颱風路徑
                    typhoon_tracks_era5[track_id_counter_era5] = {
                        'lons': [typhoon['longitude']],
                        'lats': [typhoon['latitude']],
                        'types': [typhoon['cyclone_type']],
                        'system_id': system_id,
                        'times': [time_hour]
                    }
                    current_typhoons_era5[track_id_counter_era5] = typhoon
                    track_id_counter_era5 += 1
                    system_id += 1

            elif model in ensemble_members:
                # 比較 NeuralGCM 模型的颱風
                for prev_id, prev_typhoon in previous_typhoons_neuralgcm.items():
                    distance = haversine(typhoon['longitude'], typhoon['latitude'], prev_typhoon['longitude'], prev_typhoon['latitude'])
                    if distance < distance_threshold_km:
                        # 更新颱風路徑
                        typhoon_tracks_neuralgcm[prev_id]['lons'].append(typhoon['longitude'])
                        typhoon_tracks_neuralgcm[prev_id]['lats'].append(typhoon['latitude'])
                        typhoon_tracks_neuralgcm[prev_id]['types'].append(typhoon['cyclone_type'])
                        typhoon_tracks_neuralgcm[prev_id]['times'].append(time_hour)
                        current_typhoons_neuralgcm[prev_id] = typhoon
                        matched = True
                        # break

                if not matched:
                    # 新增NeuralGCM颱風路徑
                    typhoon_tracks_neuralgcm[track_id_counter_neuralgcm] = {
                        'lons': [typhoon['longitude']],
                        'lats': [typhoon['latitude']],
                        'types': [typhoon['cyclone_type']],
                        'times': [time_hour],
                        'system_id': system_id,
                        'member': model,
                        'group': 'group1'  if (typhoon['longitude'] > longitude_threshold or typhoon['latitude'] > latitude_threshold) else 'group2'
                    }
                    current_typhoons_neuralgcm[track_id_counter_neuralgcm] = typhoon
                    track_id_counter_neuralgcm += 1
                    system_id += 1

        # 更新前一時間步的颱風信息
        previous_typhoons_era5 = current_typhoons_era5
        previous_typhoons_neuralgcm = current_typhoons_neuralgcm

    return typhoon_tracks_era5,typhoon_tracks_neuralgcm

def plot_tracks(typhoon_tracks_era5, typhoon_tracks_neuralgcm, ensemble_mean_tracks,initial_limit, interval_limit):
    # 繪製ERA5模型的颱風路徑
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title('Typhoon Tracks (ERA5 vs NeuralGCM)', fontsize=16)

    # Adding geographical features for better visualization
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Add gridlines with latitude and longitude labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Define the colors for the two groups
    group_colors = {'group1': 'blue', 'group2': 'green'}

    era5_plotted = False  # 用於控制圖例僅顯示一次
    for typhoon_id, track_data in typhoon_tracks_era5.items():
        lon = track_data['lons']
        lat = track_data['lats']
        times = track_data['times']

        # 忽略生命週期小於24小時的颱風
        if (times[-1] - times[0]) < interval_limit:
            continue
        norm = plt.Normalize(times[0], times[-1])
        colors = plt.cm.Reds_r(norm(times))  # 使用紅色系的深淺表示時間
        for i in range(len(lon)):
            ax.plot(lon[i], lat[i], 'o', color=colors[i], markersize=6, transform=ccrs.PlateCarree(), label='ERA5' if i == 0 and not era5_plotted else "")
            if times[i] % 12 == 0:
                ax.text(lon[i], lat[i], f'{times[i]}h', fontsize=12, va='bottom', transform=ccrs.PlateCarree())
        
        # 繪製連線
        ax.plot(lon, lat, color='red', linestyle='-', linewidth=2, alpha=0.8, transform=ccrs.PlateCarree(), label='ERA5 Track' if not era5_plotted else "")
        era5_plotted = True  # 更新標誌，確保圖例僅顯示一次

    # 繪製NeuralGCM模型的颱風路徑
    neuralgcm_plotted = {'group1': False, 'group2': False}  # 用於控制圖例僅顯示一次

    for typhoon_id, track_data in typhoon_tracks_neuralgcm.items():
        lon = track_data['lons']
        lat = track_data['lats']
        times = track_data['times']
        group = track_data['group']
        system_color = group_colors[group]

        if times[0] > initial_limit: #  or lat[0]<25:    # 濾除經緯度
            continue

        # 忽略生命週期小於24小時的颱風
        if (times[-1] - times[0]) < interval_limit:
            continue
        if track_data['group']=='group2': #輸出各成員路徑資料
            member_to_csv(track_data)
        for i in range(len(lon)):
            ax.plot(lon[i], lat[i], 'o', color=system_color, markersize=3, transform=ccrs.PlateCarree(), label=f'NeuralGCM ({group})' if i == 0 and not neuralgcm_plotted[group] else "")

        # 繪製連線
        ax.plot(lon, lat, color=system_color, linestyle='-', linewidth=1, alpha=0.8, transform=ccrs.PlateCarree(), label=f'NeuralGCM Track ({group})' if not neuralgcm_plotted[group] else "")
        neuralgcm_plotted[group] = True  # 更新標誌，確保圖例僅顯示一次

    # 繪製系集平均路徑
    for group, track_data in ensemble_mean_tracks.items():
        lon = track_data['lons']
        lat = track_data['lats']
        times = track_data['times']
        ax.plot(lon, lat, color='gold', linestyle='--', linewidth=3, alpha=0.9, transform=ccrs.PlateCarree(), label=f'Ensemble Mean Track ({group})')

    # 添加圖例
    ax.legend(loc='upper right')

    ax.set_extent([105, 160, -5, 50], crs=ccrs.PlateCarree())
    # ax.set_extent([150, 160, 10, 20], crs=ccrs.PlateCarree())

    plt.tight_layout()
    # plt.savefig("../img/neuralgcm_ens_2017-09-10_to_2017-09-17.png")
    plt.show()

def calculate_ensemble_mean_track(typhoon_tracks_neuralgcm,initial_limit,interval_limit):
    """
    計算每個group在每個時間步的系集平均路徑，只考慮存在超過24小時的系統。

    Parameters:
        typhoon_tracks_neuralgcm (dict): NeuralGCM模型的颱風路徑資料。
    Returns:
        dict: 每個群組的平均路徑資料。
    """
    # Initialize dictionaries to store group-wise ensemble data
    group_tracks = {'group1': {}, 'group2': {}}
    
    # Organize typhoon tracks by group
    for track_id, track_data in typhoon_tracks_neuralgcm.items():
        if track_data['times'][0] > initial_limit:# or track_data['lats'][0]<25 :  # 濾除經緯度
            continue
        if (track_data['times'][-1] - track_data['times'][0]) < interval_limit:
            continue
        group = track_data['group']
        if group not in group_tracks:
            group_tracks[group] = {}
        group_tracks[group][track_id] = track_data
    
    # Calculate mean track for each group
    ensemble_mean_tracks = {}
    for group, tracks in group_tracks.items():
        if not tracks:
            continue

        # Collect all track points by time
        time_steps = {}
        for track in tracks.values():
            for lon, lat, time in zip(track['lons'], track['lats'], track['times']):
                if time not in time_steps:
                    time_steps[time] = {'lons': [], 'lats': []}
                time_steps[time]['lons'].append(lon)
                time_steps[time]['lats'].append(lat)
        
        # Calculate mean longitude and latitude for each time step
        mean_lons = []
        mean_lats = []
        mean_times = []
        for time, points in sorted(time_steps.items()):
            if len(points['lons']) > 1:  # Only consider if more than one system exists at this time
                mean_lon = np.mean(points['lons'])
                mean_lat = np.mean(points['lats'])
                mean_lons.append(mean_lon)
                mean_lats.append(mean_lat)
                mean_times.append(time)
        
        # Store the mean track for the group
        ensemble_mean_tracks[group] = {
            'lons': mean_lons,
            'lats': mean_lats,
            'times': mean_times
        }
    
    return ensemble_mean_tracks