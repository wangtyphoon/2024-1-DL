import gcsfs
import jax
import numpy as np
import pickle
import xarray
import xarray as xr
import os
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm
import matplotlib.pyplot as plt
from vorticity import calculate_vorticity
import time

# Start the timer
start_time = time.time()
gcs = gcsfs.GCSFileSystem(token='anon')

weights_folder = 'weight'
model_name = 'neural_gcm_dynamic_forcing_stochastic_1_4_deg.pkl'
model_save_path = os.path.join(weights_folder, model_name)

with open(model_save_path, 'rb') as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

demo_start_time = '2017-09-08'
demo_end_time = '2017-09-15'
data_inner_steps = 6  # process every 24th hour

# local_era5_filename = f'era5/sliced_era5_data_{demo_start_time}_to_{demo_end_time}.nc'
local_era5_filename = f'era5/sliced_era5_data6_2017-09-08_to_2017-09-10.nc'

# Load the locally saved ERA5 data from NetCDF format
if os.path.exists(local_era5_filename):
    sliced_era5 = xr.open_dataset(local_era5_filename)
    print(f"ERA5 data loaded successfully from {local_era5_filename}")
else:
    print(f"Error: {local_era5_filename} not found!")

era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)
# eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
# eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
eval_era5 = xr.open_dataset("era5/combined_era5_2017-09-08_to_2017-09-17.nc")
inner_steps = 6  # Save model outputs every 6 hours
outer_steps = 7 * 24 // inner_steps  # Total of 7 days, output every 6 hours
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # Time axis in hours

# initialize model state
add_time = 4
inputs = model.inputs_from_xarray(eval_era5.isel(time=0+add_time))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0+add_time))

rng_key = jax.random.key(3)  # optional for deterministic models
initial_state = model.encode(inputs, input_forcings,rng_key)

# use persistence for forcing variables (SST and sea ice cover)
all_forcings = model.forcings_from_xarray(eval_era5.head(time=1+add_time))

# make forecast
final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=timedelta,
    start_with_input=True,
)
predictions_ds = model.data_to_xarray(predictions, times=times)

#fix_the time domain
eval_era5 = eval_era5.sel(time=slice(eval_era5.time[4], None))
# Selecting ERA5 targets from exactly the same time slice
target_trajectory = model.inputs_from_xarray(
    eval_era5
    .thin(time=(inner_steps // data_inner_steps))
    .isel(time=slice(outer_steps))
)
target_data_ds = model.data_to_xarray(target_trajectory, times=times)

combined_ds = xarray.concat([target_data_ds, predictions_ds], 'model')
combined_ds.coords['model'] = ['ERA5', 'NeuralGCM']
combined_ds_wind = np.sqrt(
    combined_ds.u_component_of_wind.sel(level=850)**2 + 
    combined_ds.v_component_of_wind.sel(level=850)**2
)

# Subset the data for the specified latitude and longitude ranges
combined_ds_wind_subset = combined_ds_wind.sel(
    longitude=slice(105, 160), 
    latitude=slice(-5, 50)
)

end_time = time.time()
total_runtime = end_time - start_time

print(f"Total runtime: {total_runtime:.2f} seconds")
# Increase the size of the plot
combined_ds_wind_subset.plot(
    x='longitude', 
    y='latitude', 
    row='time', 
    col='model', 
    robust=True, 
    aspect=2, 
    size=4  # Increase the size to make the plot larger
)
plt.show()
# 假設你已經有了 u_wind 和 v_wind 的資料，以及經緯度資料
u_wind = combined_ds.u_component_of_wind
v_wind = combined_ds.v_component_of_wind
latitude = combined_ds.latitude
longitude = combined_ds.longitude



vorticity = calculate_vorticity(
    u_wind.sel(level=850), v_wind.sel(level=850), latitude, longitude
)

# Add vorticity to the combined dataset
combined_ds['vorticity'] = vorticity

# Specify the path to save the combined dataset with vorticity
output_filename = f'output/neuralgcm_sto03_with_vorticity_{demo_start_time}_to_{demo_end_time}.nc'
combined_ds = combined_ds.sel(
    longitude=slice(105, 160), 
    latitude=slice(-5, 50)
)
# Save the combined dataset to a NetCDF file
combined_ds.to_netcdf(output_filename)

print(f"Dataset saved successfully to {output_filename}")

# Subset the vorticity data for the specified latitude and longitude ranges
vorticity_subset = combined_ds.vorticity.sel(
    longitude=slice(105, 160), 
    latitude=slice(-5, 50)
)

# Plot vorticity
vorticity_subset.plot(
    x='longitude', 
    y='latitude', 
    row='time', 
    col='model', 
    robust=True, 
    aspect=2, 
    size=4,
    cmap='RdBu_r'  # Use a diverging colormap for vorticity
)

plt.show()