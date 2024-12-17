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

demo_start_time = '2022-08-30'
demo_end_time = '2022-09-06'
data_inner_steps = 6  # process every 24th hour

eval_era5 = xr.open_dataset(f"era5/combined_era5_2022-08-29_to_2022-09-06.nc")
inner_steps = 6  # Save model outputs every 6 hours
outer_steps = (7 * 24 + 6 ) // inner_steps  # Total of 7 days, output every 6 hours
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # Time axis in hours

# initialize model state
add_time = 4
inputs = model.inputs_from_xarray(eval_era5.isel(time=0+add_time))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0+add_time))

predictions_ds = []
for i in range(50):
    rng_key = jax.random.key(i+1)  # optional for deterministic models
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
    prediction = model.data_to_xarray(predictions, times=times).sel(
    longitude=slice(105, 160), 
    latitude=slice(-5, 50)
        )
    
    predictions_ds.append(prediction)
    print(f'member{i+1} finished')

# Fix the time domain for ERA5 targets
eval_era5 = eval_era5.sel(time=slice(eval_era5.time[add_time], None))

# Selecting ERA5 targets from exactly the same time slice for comparison
target_trajectory = model.inputs_from_xarray(
    eval_era5
    .thin(time=(inner_steps // data_inner_steps))
    .isel(time=slice(outer_steps))
)
target_data_ds = model.data_to_xarray(target_trajectory, times=times)
target_data_ds = target_data_ds.sel(
    longitude=slice(105, 160), 
    latitude=slice(-5, 50)
        )
# Combine ERA5 target data with NeuralGCM predictions into a single dataset
combined_ds = xr.concat([target_data_ds] + predictions_ds, dim='model')
combined_ds.coords['model'] = ['ERA5'] + [f'NeuralGCM_member_{i}' for i in range(len(predictions_ds))]

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
output_filename = f'output/neuralgcm_ens_with_vorticity_{demo_start_time}_to_{demo_end_time}.nc'

# Save the combined dataset to a NetCDF file
combined_ds.to_netcdf(output_filename)

print(f"Dataset saved successfully to {output_filename}")

