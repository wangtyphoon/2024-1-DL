import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopy.distance

def analyze_wind_shear(df_path, ds_path, min_distance_km=350, max_distance_km=850, cluster_type=1):
    # 讀取 CSV 文件
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    cluster_shear_sum = 0  # 累加 cluster == cluster_type 的成員的風切強度
    cluster_member_count = 0  # 計算 cluster == cluster_type 的成員數量

    # 遍歷 cluster 欄位
    for _,row in df.iterrows():
        if row['cluster'] == cluster_type:
            # 根據名稱欄位讀取 nc 文件
            member_name = row['member']  # 假設名稱欄位的名稱是 'member'
            try:
                dataset = ds.sel(model=member_name)
                route = pd.read_csv(f"../route/20170909/csv/group2/{member_name}.csv")
                print(f"成功讀取 {member_name}")

                # 提取所有可能的經緯度點和時間
                longitudes = dataset.longitude.values
                latitudes = dataset.latitude.values
                times = route['times']
                # 初始化兩個新的欄位來儲存垂直風切的計算結果
                route["shear_magnitude"] = np.nan
                route["shear_direction"] = np.nan

                member_shear = 0
                count = 0

                # 遍歷所有時間步驟
                for time_index, time in enumerate(times):
                    # 僅處理 time 小於 72 的情況
                    if time_index < 72:
                        data_at_time = dataset.sel(time=time)

                        # 定義中心位置
                        lon_center = route["lons"][time_index]  # 使用時間索引獲取中心經度
                        lat_center = route["lats"][time_index]  # 使用時間索引獲取中心緯度
                        # 遍歷所有經度和緯度
                        average_u200 = 0
                        average_u850 = 0
                        average_v200 = 0
                        average_v850 = 0
                        count_points = 0

                        for lon in longitudes:
                            for lat in latitudes:
                                # 計算當前經緯度到中心的距離
                                current_distance = geopy.distance.distance((lat_center, lon_center), (lat, lon)).km

                                # 如果距離在 250 到 850 公里之間，則計算垂直風切
                                if min_distance_km <= current_distance <= max_distance_km:
                                    # 假設垂直風切需要使用 850 hPa 和 250 hPa 的風速差異
                                    u850 = data_at_time.sel(level=850, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
                                    v850 = data_at_time.sel(level=850, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
                                    u200 = data_at_time.sel(level=200, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
                                    v200 = data_at_time.sel(level=200, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
                                    average_u200 += u200
                                    average_u850 += u850
                                    average_v200 += v200
                                    average_v850 += v850
                                    count_points += 1

                        # 計算垂直風切（矢量差異）
                        if count_points > 1:
                            shear_u = (average_u200 - average_u850) / count_points
                            shear_v = (average_v200 - average_v850) / count_points
                            shear_magnitude = np.sqrt(shear_u**2 + shear_v**2)

                            # 計算垂直風切方向（使用 arctan2 計算角度，單位為弧度，轉換為度數）
                            shear_direction = np.degrees(np.arctan2(shear_v, shear_u)) + 180
                            route["shear_magnitude"][time_index] = shear_magnitude / 0.51444444
                            route["shear_direction"][time_index] = shear_direction

                            # 累加成員的風切強度
                            member_shear += shear_magnitude / 0.51444444
                            count += 1

                # 確保 count 不為零，避免除零錯誤
                if count > 0:
                    average_mean_shear = member_shear / count
                    print(f"{member_name} 在72小時內的平均垂直風切強度為: {average_mean_shear:.2f}")

                    # 累加到 cluster 的風切強度總和中
                    cluster_shear_sum += average_mean_shear
                    cluster_member_count += 1
                else:
                    print(f"{member_name} 在72小時內沒有足夠的數據計算平均垂直風切強度。")

                route.to_csv(f"../route/20170909/csv//group2/{member_name}.csv", index=False)
            except FileNotFoundError:
                print(f"找不到文件 {member_name}")

    # 計算 cluster == cluster_type 類別的平均風切強度
    if cluster_member_count > 0:
        cluster_average_shear = cluster_shear_sum / cluster_member_count
        print(f"cluster == {cluster_type} 類別的 72 小時內平均垂直風切強度為: {cluster_average_shear:.2f}")
    else:
        print(f"cluster == {cluster_type} 類別沒有足夠的成員數據計算平均垂直風切強度。")

# Usage
# analyze_wind_shear(df_path="../route/20170909/combined_features.csv", ds_path="../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc")

# def calculate_wind_shear(df_path, ds_path, min_distance_km=350, max_distance_km=850):
#     # 讀取 CSV 文件
#     df = pd.read_csv(df_path)
#     ds = xr.open_dataset(ds_path)

#     # 遍歷 cluster 欄位
#     for row in df.iterrows():
#         if row['cluster'] == 1:
#             # 根據名稱欄位讀取 nc 文件
#             member_name = row['member']  # 假設名稱欄位的名稱是 'member'
#             try:
#                 dataset = ds.sel(model=member_name)
#                 route = pd.read_csv(f"../route/20170909/csv/{member_name}.csv")
#                 print(f"成功讀取 {member_name}")

#                 # 提取所有可能的經緯度點和時間
#                 longitudes = dataset.longitude.values
#                 latitudes = dataset.latitude.values
#                 times = route['times']
#                 # 初始化兩個新的欄位來儲存垂直風切的計算結果
#                 route["shear_magnitude"] = np.nan
#                 route["shear_direction"] = np.nan

#                 # 遍歷所有時間步驟
#                 for time_index, time in enumerate(times):
#                     # 僅處理 time 小於 72 的情況
#                     if time_index < 72:
#                         data_at_time = dataset.sel(time=time)

#                         # 定義中心位置
#                         lon_center = route["lons"][time_index]  # 使用時間索引獲取中心經度
#                         lat_center = route["lats"][time_index]  # 使用時間索引獲取中心緯度
#                         # 遍歷所有經度和緯度
#                         average_u200 = 0
#                         average_u850 = 0
#                         average_v200 = 0
#                         average_v850 = 0
#                         count = 0

#                         for lon in longitudes:
#                             for lat in latitudes:
#                                 # 計算當前經緯度到中心的距離
#                                 current_distance = geopy.distance.distance((lat_center, lon_center), (lat, lon)).km

#                                 # 如果距離在 250 到 850 公里之間，則計算垂直風切
#                                 if min_distance_km <= current_distance <= max_distance_km:
#                                     # 假設垂直風切需要使用 850 hPa 和 250 hPa 的風速差異
#                                     u850 = data_at_time.sel(level=850, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
#                                     v850 = data_at_time.sel(level=850, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
#                                     u200 = data_at_time.sel(level=200, longitude=lon, latitude=lat, method="nearest")['u_component_of_wind'].values
#                                     v200 = data_at_time.sel(level=200, longitude=lon, latitude=lat, method="nearest")['v_component_of_wind'].values
#                                     average_u200 += u200
#                                     average_u850 += u850
#                                     average_v200 += v200
#                                     average_v850 += v850
#                                     count += 1

#                         # 計算垂直風切（矢量差異）
#                         if count > 1:
#                             shear_u = (average_u200 - average_u850) / count
#                             shear_v = (average_v200 - average_v850) / count
#                             shear_magnitude = np.sqrt(shear_u**2 + shear_v**2)

#                             # 計算垂直風切方向（使用 arctan2 計算角度，單位為弧度，轉換為度數）
#                             shear_direction = np.degrees(np.arctan2(shear_v, shear_u)) + 180
#                             route["shear_magnitude"][time_index] = shear_magnitude / 0.51444444
#                             route["shear_direction"][time_index] = shear_direction

#                             # 將度數轉換到 0 到 360 度之間
#                             # shear_direction = (shear_direction + 360) % 360
#                             # print(shear_magnitude, shear_direction)
#                 route.to_csv(f"../route/20170909/csv/{member_name}.csv", index=False)
#             except FileNotFoundError:
#                 print(f"找不到文件 {member_name}")

# # Usage
# # calculate_wind_shear(df_path="../route/20170909/combined_features.csv", ds_path="../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc")

def vis_wind_shear(df_path, ds_path,  cluster):
    # 讀取 CSV 文件
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)

    # 遍歷 cluster 欄位
    for _,row in df.iterrows():
        if row['cluster'] == cluster:
            # 根據名稱欄位讀取 nc 文件
            member_name = row['member']  # 假設名稱欄位的名稱是 'member'
            try:
                dataset = ds.sel(model=member_name)
                route = pd.read_csv(f"../route/20170909/csv/group2/{member_name}.csv")
                print(f"成功讀取 {member_name}")

                times = route['times']

                # 初始化列表來存儲風切強度和方向
                shear_magnitudes = []
                shear_directions = []

                # 遍歷所有時間步驟
                for time_index, time in enumerate(times):
                    # 僅處理 time 小於 72 的情況
                    if time_index < 72:
                        shear_magnitude = route['shear_magnitude'][time_index]
                        shear_direction = route['shear_direction'][time_index]

                        shear_magnitudes.append(shear_magnitude)
                        shear_directions.append(shear_direction)

                # 繪製風切強度和風切方向隨時間的變化
                if len(shear_magnitudes) > 0:
                    plot_wind_shear(member_name, shear_magnitudes, shear_directions)

            except FileNotFoundError:
                print(f"找不到文件 {member_name}")

def plot_wind_shear(member_name, shear_magnitudes, shear_directions):
    plt.figure(figsize=(10, 4))

    # 繪製風切強度變化
    plt.subplot(1, 2, 1)
    plt.plot(range(len(shear_magnitudes)), shear_magnitudes, label=f'{member_name} - Shear Magnitude', marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Wind Shear Magnitude')
    plt.title('Wind Shear Magnitude over Time')
    plt.legend()
    plt.grid(True)

    # 繪製風切方向變化
    plt.subplot(1, 2, 2)
    plt.plot(range(len(shear_directions)), shear_directions, label=f'{member_name} - Shear Direction', marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Wind Shear Direction (degrees)')
    plt.title('Wind Shear Direction over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"../route/20170909/png/shear/group2/0/Typhoon shear - Member {member_name}.png")
    # 顯示圖形
    plt.tight_layout()
    plt.show()

# Usage
# vis_wind_shear(df_path="../route/20170909/combined_features.csv",cluster=0, ds_path="../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc")


def cluster_wind_shear(df_path, ds_path, cluster_list, min_distance_km=350, max_distance_km=850):
    # 讀取 CSV 文件
    df = pd.read_csv(df_path)
    ds = xr.open_dataset(ds_path)
    
    # 遍歷 cluster 欄位並繪製各 cluster 的平均路徑
    plt.figure(figsize=(10, 6))
    for cluster in cluster_list:
        cluster_route = pd.DataFrame()
        for _, row in df.iterrows():  # 正確地解包 iterrows() 返回的元組
            if row['cluster'] == cluster:
                # 根據名稱欄位讀取 nc 文件
                member_name = row['member']  # 假設名稱欄位的名稱是 'member'
                try:
                    dataset = ds.sel(model=member_name)
                    route = pd.read_csv(f"../route/20170909/csv/group2/{member_name}.csv")
                    
                    # 合併 route 與 cluster_route
                    cluster_route = pd.concat([cluster_route, route], ignore_index=True)
                except FileNotFoundError:
                    print(f"File for member {member_name} not found.")
                    continue

        # 根據時間將各系集成員的經緯度平均後繪製為路徑圖
        if not cluster_route.empty:
            # 確保經緯度的數據類型為數值型，以便能夠計算平均值
            cluster_route['lons'] = pd.to_numeric(cluster_route['lons'], errors='coerce')
            cluster_route['lats'] = pd.to_numeric(cluster_route['lats'], errors='coerce')
            
            # 刪除包含 NaN 的行，以避免聚合錯誤
            cluster_route = cluster_route.dropna(subset=['lons', 'lats', 'times'])
            
            cluster_route_avg = cluster_route.groupby('times')[['lons', 'lats']].mean().reset_index()
            plt.plot(cluster_route_avg['lons'], cluster_route_avg['lats'], marker='o', linestyle='-', label=f'Cluster {cluster} Average Path')

            # 每隔 24 小時標記一次時間
            for _, avg_row in cluster_route_avg.iterrows():
                if avg_row['times'] % 24 == 0:  # 假設每天 00:00 標記
                    plt.text(avg_row['lons'], avg_row['lats'], str(avg_row['times']), fontsize=8, color='red')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Average Paths for Clusters {cluster_list}')
    plt.legend()
    plt.grid(True)
    plt.savefig("../route/20170909/cluster_route_group2.png")
    plt.show()

    # 繪製風切變強度圖
    plt.figure(figsize=(10, 6))
    for cluster in cluster_list:
        cluster_route = pd.DataFrame()
        for _, row in df.iterrows():  # 正確地解包 iterrows() 返回的元組
            if row['cluster'] == cluster:
                # 根據名稱欄位讀取 nc 文件
                member_name = row['member']
                try:
                    dataset = ds.sel(model=member_name)
                    route = pd.read_csv(f"../route/20170909/csv/group2/{member_name}.csv")
                    
                    # 合併 route 與 cluster_route
                    cluster_route = pd.concat([cluster_route, route], ignore_index=True)
                except FileNotFoundError:
                    print(f"File for member {member_name} not found.")
                    continue

        # 根據時間計算風切變強度並繪製圖表
        if not cluster_route.empty:
            # 確保風速的數據類型為數值型，以便能夠計算風切變
            cluster_route['shear_magnitude'] = pd.to_numeric(cluster_route['shear_magnitude'], errors='coerce')
            
            # 刪除包含 NaN 的行，以避免聚合錯誤
            cluster_route = cluster_route.dropna(subset=['shear_magnitude', 'times'])
            
            cluster_route_avg = cluster_route.groupby('times')['shear_magnitude'].mean().reset_index()
            # Group by time and aggregate size_index based on the cluster
    
            plt.plot(cluster_route_avg['times'], cluster_route_avg['shear_magnitude'], marker='o', linestyle='-', label=f'Cluster {cluster} Wind Shear Magnitude')

    plt.xlabel('Time')
    plt.ylabel('Wind Shear Magnitude')
    plt.title(f'Wind Shear Magnitude Over Time for Clusters {cluster_list}')
    plt.legend()
    plt.grid(True)
    plt.savefig("../route/20170909/cluster_shear_group2.png")
    plt.show()

# 示例調用
cluster_wind_shear("../route/20170909/combined_features.csv", "../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc", cluster_list=[0, 1])
