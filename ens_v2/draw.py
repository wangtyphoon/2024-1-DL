import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# 讀取 CSV 文件
df = pd.read_csv("../route/20170909/combined_features.csv")
filepath = '../output/neuralgcm_ens_with_vorticity_2017-09-09_to_2017-09-17.nc'
ds = xr.open_dataset(filepath)

# 選擇指定的模型
subset = ds.sel(model="NeuralGCM_member_34")

# 繪製每個時間步的渦度圖
for time in subset.time:
    vorticity = subset.sel(time=time)['vorticity']  # 假設渦度變量名稱是 'vorticity'

    plt.figure(figsize=(10, 6))
    vorticity.plot(x='longitude', y='latitude', cmap='viridis', cbar_kwargs={'label': 'Vorticity'})  # 明確設置經度和緯度，並加入色帶標籤
    plt.title(f"渦度圖 - 時間: {str(time.values)}")
    plt.xlabel('經度')
    plt.ylabel('緯度')
    plt.grid(True)
    plt.show()

    # # 保存圖像到文件
    # plt.savefig(f'vorticity_{str(time.values)}.png')
    # plt.close()
