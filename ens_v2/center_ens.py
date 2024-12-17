import numpy as np
import xarray as xr
from fliter_centers import *  # Import everything from fliter_centers
from plot_track import *      # Assuming plot_track.py is also in the same directory or already in your PYTHONPATH

def load_and_select_dataset(filepath, time_hour, model='ERA5', lon_slice=(100, 160), lat_slice=(-5, 50)):
    """
    加載資料集並選擇特定的時間、模型和區域。
    
    Parameters:
        filepath (str): 資料集的檔案路徑。
        time_hour (int): 選擇的時間（以小時為單位）。
        model (str): 模型名稱，預設為 'ERA5'。
        lon_slice (tuple): 經度範圍。
        lat_slice (tuple): 緯度範圍。
    
    Returns:
        xr.Dataset: 選擇後的資料集。
    """
    ds = xr.open_dataset(filepath)
    selected_ds = ds.sel(
        time=time_hour,
        model=model,
        longitude=slice(*lon_slice),
        latitude=slice(*lat_slice)
    )
    return selected_ds

def process_time_step(model, filepath, time_index, time_step_hours=24, vorticity_threshold=1.2e-4, grid_size=1,
                     wind_speed_threshold=14, wind_level=850):
    """
    處理單個時間步驟，識別颱風中心、計算溫度差異，並返回分類結果。
    
    Parameters:
        model (str): 模型名稱（例如 'ERA5', 'NeuralGCM'）。
        filepath (str): 資料集的檔案路徑。
        time_index (int): 時間索引。
        time_step_hours (int): 時間間隔，默認為24小時。
        vorticity_threshold (float): 涡度閾值。
        grid_size (int): 鄰域範圍大小。
        wind_speed_threshold (float): 風速閾值。
        wind_level (int): 風速層高度，預設為850 hPa。
    
    Returns:
        list of dict: 已分類的颱風中心。
        int: 當前時間步的時間（小時）。
    """
    time_hour = time_index * time_step_hours

    # 加載資料集並選擇模型和區域
    ds = load_and_select_dataset(filepath, time_hour, model=model)

    # 提取必要變數
    u_wind = ds.u_component_of_wind.sel(level=wind_level)
    v_wind = ds.v_component_of_wind.sel(level=wind_level)
    vorticity = ds.vorticity
    # geoportential = 
    temperature = ds.temperature.sel(level=850)  # 假設溫度在850 hPa

    # 計算風速和涡度
    wind_speed = np.sqrt(u_wind**2 + v_wind**2).values
    vorticity_np = vorticity.values

    longitude = vorticity.longitude.values
    latitude = vorticity.latitude.values

    # 識別颱風中心
    typhoon_centers = find_typhoon_centers(vorticity_np, wind_speed, longitude, latitude, 
                                           vorticity_threshold, wind_speed_threshold)
    temperature_np = temperature.values

    # 計算溫度差異與渦度極值
    temp_diffs = calculate_temperature_differences(temperature_np, longitude, latitude, typhoon_centers, grid_size)
    max_vorticity = calculate_max_vorticity(vorticity_np, longitude, latitude, typhoon_centers, grid_size)
    # 分類並過濾颱風中心
    classified_typhoons = classify_and_filter_typhoons(typhoon_centers, temp_diffs, max_vorticity)

    # 過濾過近的颱風中心
    classified_typhoons = filter_close_centers(classified_typhoons)

    #加權颱風中心
    weighted_center = weight_typhoon_centers(vorticity_np, longitude, latitude, classified_typhoons, grid_size=5)
    return weighted_center , time_hour


# Handle different ensemble members
def main():
    filepath = '../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc'
    num_time_steps = 33
    ds = xr.open_dataset(filepath)
    ensemble_members = [member for member in ds.model.values if 'NeuralGCM_member' in member]

    all_classified_typhoons = []

    # 處理 ERA5 資料
    for i in range(num_time_steps):
        classified_typhoons, time_hour = process_time_step(
            model="ERA5",
            filepath=filepath,
            time_index=i,
            time_step_hours=6,
            vorticity_threshold=1.2e-4,
            grid_size=1,
            wind_speed_threshold=10.5,
            wind_level=925
        )
        for typhoon in classified_typhoons:
            typhoon['model'] = 'ERA5'
        all_classified_typhoons.append((classified_typhoons, time_hour))

    ensemble_members =ensemble_members[:]
    # Handle NeuralGCM ensemble members
    for member in ensemble_members:
        for i in range(num_time_steps):
            classified_typhoons, time_hour = process_time_step(
                model=member,
                filepath=filepath,
                time_index=i,
                time_step_hours=6,
                vorticity_threshold=1.0e-4,
                grid_size=1,
                wind_speed_threshold=10.5,
                wind_level=850
            )
            for typhoon in classified_typhoons:
                typhoon['model'] = member
            all_classified_typhoons.append((classified_typhoons, time_hour))

    longitude = ds.longitude.values
    latitude = ds.latitude.values
    
    #篩選設定
    initial_limit = 96
    interval_limit = 48
    typhoon_tracks_era5,typhoon_tracks_neuralgcm = process_track(all_classified_typhoons,ensemble_members )
    ensemble_mean_tracks = calculate_ensemble_mean_track(typhoon_tracks_neuralgcm,initial_limit,interval_limit )
    plot_tracks(typhoon_tracks_era5,typhoon_tracks_neuralgcm,ensemble_mean_tracks,initial_limit,interval_limit)
if __name__ == "__main__":
    main()