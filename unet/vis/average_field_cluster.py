import numpy as np
import pandas as pd

# 讀取數據集
ds = pd.read_csv("combined_features.csv")

# 定義函數來處理每個 cluster
def process_cluster_data(cluster_value, time_range, output_prefix):
    cluster_list = []
    for i in range(len(ds)):
        if ds['cluster'][i] == cluster_value:
            cluster_list.append(i + 1)
    
    for time_step in time_range:
        # 初始化一個空列表來存儲讀取的數據
        all_data = []
        
        for i in cluster_list:
            file_path = f"../dataset/model_time_data/model0909_{i}_time_{time_step}.npy"
            try:
                data = np.load(file_path)
                all_data.append(data)
                print(f"Data from {file_path} loaded successfully with shape: {data.shape}")
            except FileNotFoundError:
                print(f"Warning: {file_path} not found, skipping this file.")
        
        # # 加載 time_0 的數據
        # file_path = f"../dataset/model_time_data/model0909_0_time_0.npy"
        # try:
        #     data = np.load(file_path)
        #     all_data.append(data)
        # except FileNotFoundError:
        #     print(f"Warning: {file_path} not found, skipping this file.")
        
        # 堆疊數據並計算平均值
        if all_data:
            combined_data = np.stack(all_data, axis=0)
            print(f"Combined data shape for time step {time_step}: {combined_data.shape}")
            
            # 計算平均值
            mean_data = np.mean(combined_data, axis=0)
            
            # 保存數據
            output_file = f"{output_prefix}_{time_step}.npy"
            np.save(output_file, mean_data)
            print(f"Saved data to {output_file}")
        else:
            print(f"No data loaded for time step {time_step}.")

# 設置時間範圍和處理每個 cluster
time_range = range(12)  # 0-11
process_cluster_data(cluster_value=0, time_range=time_range, output_prefix="westward")
process_cluster_data(cluster_value=1, time_range=time_range, output_prefix="eastward")
