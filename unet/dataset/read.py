import xarray as xr
import numpy as np

# 打开 .nc 文件
dataset = xr.open_dataset('neuralgcm_ens_with_vorticity_2017-09-07_to_2017-09-17.nc')

# 定义所需的层次
levels = [200, 300, 500, 700, 850, 925]  # hPa

# 选取对应层次的数据
dataset = dataset.sel(level=levels)

# 提取 u 和 v 风分量
u_wind = dataset['u_component_of_wind']
v_wind = dataset['v_component_of_wind']

# 选取 200 hPa 和 850 hPa 的风场
u_200 = u_wind.sel(level=200)
v_200 = v_wind.sel(level=200)
u_850 = u_wind.sel(level=850)
v_850 = v_wind.sel(level=850)

# 计算垂直风切变的大小
u_shear = u_200 - u_850
v_shear = v_200 - v_850
wind_shear_magnitude = np.sqrt(u_shear**2 + v_shear**2)

dataset['wind_shear'] = wind_shear_magnitude

# 使用 np.expand_dims 扩展维度，然后广播到 (51, 33, 39, 40)
# 提取 sim_time 变量
sim_time = dataset['sim_time']  # Shape: (51, 33)
dataset = dataset.drop_vars('sim_time')
expanded_sim_time = np.expand_dims(sim_time, axis=(-2, -1))  # Shape: (51, 33, 1, 1)
expanded_sim_time = np.tile(expanded_sim_time, (1, 1, 39, 40))  # Broadcasting to (51, 33, 39, 40)
dataset['sim_time'] = (('member', 'times', 'longitude', 'latitude'), expanded_sim_time)

# 初始化一个列表来存储所有合并的数据
merged_array_list = []

# 提取 level 的数据
levels = dataset['level'].values

# 遍历所有变量
for var in dataset.data_vars:
    data_var = dataset[var]

    # 检测并填补 NaN 值，使用最近邻插值
    if data_var.isnull().any():
        data_var = data_var.ffill(dim='latitude').bfill(dim='latitude')
        data_var = data_var.ffill(dim='longitude').bfill(dim='longitude')

    # 如果变量符合条件 (5 维)，处理其数据
    if len(data_var.shape) == 5:
        level_arrays = []
        for level in levels:
            # 提取当前 level 的数据并转换为 numpy array
            data_at_level = data_var.sel(level=level).values
            level_arrays.append(data_at_level)
        
        # 合并不同 level 的数据 (增加 level 维度)
        merged_array = np.stack(level_arrays, axis=2)
        merged_array_list.append(merged_array)
    elif len(data_var.shape) == 4:
        # 将四维变量添加一个维度以符合目标数组的维度
        data_array = data_var.values
        expanded_array = np.expand_dims(data_array, axis=2)

        merged_array_list.append(expanded_array)

# 最终将所有变量数据合并为单一大 numpy array
# 合并的维度需根据需求调整，这里假设新维度为变量
final_array = np.concatenate(merged_array_list, axis=2)
# 检查是否存在 NaN 值
if np.isnan(final_array).any():
    print("Final merged array contains NaN values.")
else:
    print("Final merged array does not contain NaN values.")

print(f"Final merged array shape: {final_array.shape}")


# 創建輸出目錄
import os
output_dir = "model_time_data"
os.makedirs(output_dir, exist_ok=True)

# 遍歷 model 和 time 維度
for model_idx in range(final_array.shape[0]):  # 遍歷 model 維度
    for time_idx in range(final_array.shape[1]):  # 遍歷 time 維度
        # 提取特定 model 和 time 的數據
        data_slice = final_array[model_idx, time_idx, :, :, :]  # Shape: (45, 39, 40)
        # 將第45個變數除以10000
        data_slice[44, :, :] /= 100000.0
        # 保存為單獨的文件
        filename = f"model0907_{model_idx}_time_{time_idx}.npy"
        filepath = os.path.join(output_dir, filename)
        # np.save(filepath, data_slice)

print(f"Data has been saved to {output_dir}.")
