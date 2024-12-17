from scipy.ndimage import maximum_filter, label, generate_binary_structure, find_objects
import numpy as np
from geopy.distance import geodesic

def find_typhoon_centers(vorticity_np, wind_speed_np, longitude, latitude, vorticity_threshold=1.2e-4, wind_speed_threshold=17.2):
    """
    根據涡度場識別潛在的颱風中心。
    
    Parameters:
        vorticity_np (2D array): 涡度場數據。
        longitude (1D array): 經度座標。
        latitude (1D array): 緯度座標。
        vorticity_threshold (float): 涡度閾值。
        wind_speed_threshold (float): 風速閾值。
    
    Returns:
        list of tuples: 颱風中心的經緯度座標。
    """
    # 創建涡度超過閾值的布林掩膜
    vorticity_mask = vorticity_np > vorticity_threshold

    # 定義連通結構元素（8連通）
    struct = generate_binary_structure(2, 2)

    # 找到涡度場中的局部極大值
    local_max = maximum_filter(vorticity_np, footprint=struct, mode='constant') == vorticity_np

    # 識別潛在的颱風中心
    potential_centers = local_max & vorticity_mask
    
    # 使用3x3池化過濾風速，並設置風速條件
    max_wind_speed = maximum_filter(wind_speed_np, size=3)
    wind_speed_mask = max_wind_speed > wind_speed_threshold
    potential_centers = potential_centers & wind_speed_mask

    # 標記連通區域
    labeled_centers, num_features = label(potential_centers, structure=struct)

    typhoon_centers = []

    # 循環識別各區域的最大涡度位置，標示中心
    for region_idx in range(1, num_features + 1):
        slices = find_objects(labeled_centers == region_idx)[0]
        region_vorticity = vorticity_np[slices]
        max_pos = np.unravel_index(np.argmax(region_vorticity), region_vorticity.shape)
        center_x = slices[0].start + max_pos[0]
        center_y = slices[1].start + max_pos[1]
        lon = longitude[center_x]
        lat = latitude[center_y]
        typhoon_centers.append((lon, lat))
    
    return typhoon_centers

def filter_close_centers(classified_typhoons, distance_threshold_km=500):
    """
    過濾距離過近的颱風中心，保留涡度較高者。
    
    Args:
        classified_typhoons (list of dict): 颱風資訊列表。
        distance_threshold_km (float): 距離閾值，單位為公里。
    
    Returns:
        list of dict: 過濾後的颱風資訊列表。
    """
    filtered_typhoons = []
    for i in range(len(classified_typhoons)):
        keep_center = True
        for j in range(len(filtered_typhoons)):  # 與已過濾的中心比較
            lon1, lat1 = classified_typhoons[i]['longitude'], classified_typhoons[i]['latitude']
            lon2, lat2 = filtered_typhoons[j]['longitude'], filtered_typhoons[j]['latitude']
            distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers

            # 比較涡度，保留涡度較高的中心
            if distance < distance_threshold_km:
                if classified_typhoons[i]['max_vorticity'] < filtered_typhoons[j]['max_vorticity']:
                    keep_center = False
                    break
                else:
                    filtered_typhoons.remove(filtered_typhoons[j])
                    break
        if keep_center:
            filtered_typhoons.append(classified_typhoons[i])
    return filtered_typhoons

def calculate_temperature_differences(temperature_np, longitude, latitude, typhoon_centers, grid_size=1):
    """
    計算每個颱風中心及其鄰近網格內的溫度最大值與最小值的差。
    
    Parameters:
        temperature_np (2D array): 溫度場數據。
        longitude (1D array): 經度座標。
        latitude (1D array): 緯度座標。
        typhoon_centers (list of tuples): 颱風中心的經緯度座標。
        grid_size (int): 鄰域範圍大小，默認為1（即3x3網格）。
    
    Returns:
        list of float: 每個颱風中心的溫度差異。
    """
    temp_diffs = []
    n_lon = len(longitude)
    n_lat = len(latitude)

    for lon, lat in typhoon_centers:
        center_x = np.argmin(np.abs(longitude - lon))
        center_y = np.argmin(np.abs(latitude - lat))

        x_start = max(center_x - grid_size, 0)
        x_end = min(center_x + grid_size + 1, n_lon)
        y_start = max(center_y - grid_size, 0)
        y_end = min(center_y + grid_size + 1, n_lat)

        temp_region = temperature_np[x_start:x_end, y_start:y_end]

        max_temp = np.max(temp_region)
        min_temp = np.min(temp_region)
        temp_diff = max_temp - min_temp
        temp_diffs.append(temp_diff)
    
    return temp_diffs

def calculate_max_vorticity(vorticity_np, longitude, latitude, typhoon_centers, grid_size=1):
    """
    計算每個颱風中心及其鄰近網格內的最大涡度。
    
    Parameters:
        vorticity_np (2D array): 涡度場數據。
        longitude (1D array): 經度座標。
        latitude (1D array): 緯度座標。
        typhoon_centers (list of tuples): 颱風中心的經緯度座標。
    
    Returns:
        list of float: 每個颱風中心的最大涡度。
    """
    vor_diffs = []
    n_lon = len(longitude)
    n_lat = len(latitude)

    for lon, lat in typhoon_centers:
        center_x = np.argmin(np.abs(longitude - lon))
        center_y = np.argmin(np.abs(latitude - lat))
        max_vor = vorticity_np[center_x, center_y]
        vor_diffs.append(max_vor)
    
    return vor_diffs

def classify_and_filter_typhoons(typhoon_centers, temp_diffs, max_vorticity, latitude_threshold=(5, 45), temp_diff_threshold=5.5):
    """
    根據溫度差與緯度範圍分類氣旋為熱帶或溫帶氣旋，並過濾不符合條件的中心。
    
    Parameters:
        typhoon_centers (list of tuples): 颱風中心的經緯度座標。
        temp_diffs (list of float): 每個颱風中心的溫度差異。
        max_vorticity (list of float): 每個颱風中心的最大涡度。
        latitude_threshold (tuple): 緯度範圍。
        temp_diff_threshold (float): 溫度差閾值，超過則為溫帶氣旋。
    
    Returns:
        list of dict: 包含過濾後颱風中心信息及其分類結果。
    """
    filtered_typhoons = []
    for (lon, lat), temp_diff in zip(typhoon_centers, temp_diffs):
        if lat < latitude_threshold[0] or lat > latitude_threshold[1]:
            continue
        
        cyclone_type = 'Extratropical Cyclone' if temp_diff > temp_diff_threshold else 'Tropical Cyclone'

        filtered_typhoons.append({
            'longitude': lon,
            'latitude': lat,
            'temp_diff': temp_diff,
            'cyclone_type': cyclone_type,
            'max_vorticity': max_vorticity
        })

    return filtered_typhoons

def weight_typhoon_centers(vorticity, longitude, latitude, centers, grid_size=5):
    """
    根据颱風中心周圍的网格点渦度进行加权中心调整。

    Parameters:
        vorticity (ndarray): 渦度场的数据。
        longitude (ndarray): 经度数据。
        latitude (ndarray): 纬度数据。
        centers (list of dict): 初步识别的颱風中心列表。
        grid_size (int): 周围网格点的大小，默认是5x5。

    Returns:
        list of dict: 调整后的颱風中心列表。
    """
    adjusted_centers = []
    half_grid = grid_size // 2

    for center in centers:
        lon = center['longitude']
        lat = center['latitude']

        # 找到中心点的索引
        lon_idx = (np.abs(longitude - lon)).argmin()
        lat_idx = (np.abs(latitude - lat)).argmin()

        # 确定周围网格的范围
        lon_start = max(lon_idx - half_grid, 0)
        lon_end = min(lon_idx + half_grid + 1, len(longitude))
        lat_start = max(lat_idx - half_grid, 0)
        lat_end = min(lat_idx + half_grid + 1, len(latitude))

        # 提取周围网格点的渦度值和对应的经纬度
        vorticity_subset = vorticity[ lon_start:lon_end,lat_start:lat_end]
        lon_subset = longitude[lon_start:lon_end]
        lat_subset = latitude[lat_start:lat_end]

        # 计算加权平均的位置
        total_vorticity = np.sum(vorticity_subset)
        if total_vorticity != 0:
            weighted_lon = np.sum(vorticity_subset * lon_subset[:, np.newaxis]) / total_vorticity
            weighted_lat = np.sum(vorticity_subset * lat_subset[np.newaxis, :]) / total_vorticity
        else:
            weighted_lon = lon
            weighted_lat = lat

        # 更新中心位置
        adjusted_center = center.copy()
        adjusted_center['longitude'] = weighted_lon
        adjusted_center['latitude'] = weighted_lat

        adjusted_centers.append(adjusted_center)

    return adjusted_centers