import netCDF4 as nc
import numpy as np
import os
from sklearn.decomposition import PCA
import random
import xarray as xr
from datetime import datetime

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
# Define the number of principal components to retain
n_components = 15

def save_to_netcdf(pred_T, pred_S, depth, sss, sst, aviso, time, lat, lon, file_name='output.nc',
                   temp_pca=None, temp_pcs=None, temp_pca_variances=None, temp_mean=None,
                   sal_pca=None, sal_pcs=None, sal_pca_variances=None, sal_mean=None,
                   seed=None, n_components=None):
    profile_number = np.arange(pred_T.shape[0])
    depth = depth.astype(np.float32)

    data_vars = {
        'Temperature': (('profile_number', 'depth'), pred_T.data),
        'T_mean_pca': (('depth'), temp_mean),
        'Salinity': (('profile_number', 'depth'), pred_S.data),
        'S_mean_pca': (('depth'), sal_mean),
        'SSS': (('profile_number'), sss.data),
        'SST': (('profile_number'), sst.data),
        'AVISO': (('profile_number'), aviso.data),
        'time': (('profile_number'), time.data),
        'lat': (('profile_number'), lat.data),
        'lon': (('profile_number'), lon.data)
    }

    # Include PCA results if available
    if temp_pca is not None:
        data_vars['Temperature_PC'] = (('n_components', 'depth'), temp_pca)
    if temp_pcs is not None:
        data_vars['Temperature_PCS'] = (('profile_number', 'n_components'), temp_pcs.data)  # Use .data here
    if temp_pca_variances is not None:
        data_vars['Temperature_PCA_variances'] = (('n_components'), temp_pca_variances)
    if sal_pca is not None:
        data_vars['Salinity_PC'] = (('n_components', 'depth'), sal_pca)
    if sal_pcs is not None:
        data_vars['Salinity_PCS'] = (('profile_number', 'n_components'), sal_pcs.data)  # Use .data here
    if sal_pca_variances is not None:
        data_vars['Salinity_PCA_variances'] = (('n_components'), sal_pca_variances)

    coords = {
    'profile_number': profile_number,
    'depth': depth
    }
        
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add units and attributes
    ds['Temperature'].attrs['units'] = 'Temperature (degrees Celsius)'
    ds['Salinity'].attrs['units'] = 'Salinity (practical salinity units)'
    ds['SSS'].attrs['units'] = 'Satellite sea surface salinity (psu)'
    ds['SST'].attrs['units'] = 'Satellite sea surface temperature (degrees Kelvin)'
    ds['AVISO'].attrs['units'] = 'Adjusted absolute dynamic topography (meters)'
    ds['lat'].attrs['units'] = 'Latitude'
    ds['lon'].attrs['units'] = 'Longitude'
    if n_components is not None:
        ds.attrs['n_components'] = n_components
    if temp_pca is not None and temp_pcs is not None:
        ds['Temperature_PC'].attrs['units'] = 'Principal Components of Temperature'
        ds['Temperature_PCS'].attrs['units'] = 'Scores of Temperature Principal Components'
    if sal_pca is not None and sal_pcs is not None:
        ds['Salinity_PC'].attrs['units'] = 'Principal Components of Salinity'
        ds['Salinity_PCS'].attrs['units'] = 'Scores of Salinity Principal Components'

    ds.attrs['description'] = 'Synthetic temperature and salinity profiles with PCA results for NeSPReSO'
    ds.attrs['institution'] = 'COAPS, FSU'
    ds.attrs['author'] = 'Jose Roberto Miranda'
    ds.attrs['contact'] = 'jrm22n@fsu.edu'
    ds.attrs['random_seed'] = seed
    ds.attrs['date_of_creation'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    

    comp = dict(zlib=True, complevel=5)
    if type(ds) is xr.Dataset:
        for var in ds: 
            ds[var].encoding.update(comp)
    if type(ds) is xr.DataArray:
        ds.encoding.update(comp)

    ds.to_netcdf(file_name, format='NETCDF4')

file_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/model_input/ARGO_inputs_for_nespreso1.nc'
dataset = xr.open_dataset(file_path, chunks={'profile_number': 1000})

# Get profile numbers
profile_numbers = np.arange(dataset['profile_number'].size)
np.random.shuffle(profile_numbers)

# Define split
split_index = int(0.85 * len(profile_numbers))
train_val_indices = profile_numbers[:split_index]
test_indices = profile_numbers[split_index:]

# Extract data for training and validation
# pred_T_train_val = dataset.variables['Temperature'][train_val_indices, :]
# pred_S_train_val = dataset.variables['Salinity'][train_val_indices, :]
# sss_train_val = dataset.variables['SSS'][train_val_indices]
# sst_train_val = dataset.variables['SST'][train_val_indices]
# aviso_train_val = dataset.variables['AVISO'][train_val_indices]
# time_train_val = dataset.variables['time'][train_val_indices]
# lat_train_val = dataset.variables['lat'][train_val_indices]
# lon_train_val = dataset.variables['lon'][train_val_indices]
pred_T_train_val = dataset['Temperature'].isel(profile_number = train_val_indices)
pred_S_train_val = dataset['Salinity'].isel(profile_number = train_val_indices)
sss_train_val = dataset['SSS'].isel(profile_number = train_val_indices)
sst_train_val = dataset['SST'].isel(profile_number = train_val_indices)
aviso_train_val = dataset['AVISO'].isel(profile_number = train_val_indices)
time_train_val = dataset['time'].isel(profile_number = train_val_indices)
lat_train_val = dataset['lat'].isel(profile_number = train_val_indices)
lon_train_val = dataset['lon'].isel(profile_number = train_val_indices)
depth_data = dataset['depth'].load()

# Handle missing values
pred_T_train_val = np.nan_to_num(pred_T_train_val)
pred_S_train_val = np.nan_to_num(pred_S_train_val)

# Perform PCA
def perform_pca(data, n_components=15):
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(data)
    pca_components = pca.components_
    captured_variances = pca.explained_variance_ratio_
    mean = pca.mean_
    return pca_scores, pca_components, captured_variances, mean

temp_pca_scores, temp_pca_components, temp_pca_variances, temp_mean = perform_pca(pred_T_train_val, n_components)
sal_pca_scores, sal_pca_components, sal_pca_variances, sal_mean = perform_pca(pred_S_train_val, n_components)

# Save training and validation data
train_val_file_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/model_input/nespreso1_train_val_data_PCA_42.nc'
save_to_netcdf(
    pred_T_train_val, pred_S_train_val, depth_data, sss_train_val, sst_train_val, aviso_train_val,
    time_train_val, lat_train_val, lon_train_val, file_name=train_val_file_path,
    temp_pca=temp_pca_components, temp_pcs=temp_pca_scores, temp_pca_variances=temp_pca_variances,
    sal_pca=sal_pca_components, sal_pcs=sal_pca_scores, sal_pca_variances=sal_pca_variances,
    seed=seed, n_components=n_components, temp_mean=temp_mean, sal_mean=sal_mean
)

# Extract data for test set
# pred_T_test = dataset.variables['Temperature'][test_indices, :]
# pred_S_test = dataset.variables['Salinity'][test_indices, :]
# sss_test = dataset.variables['SSS'][test_indices]
# sst_test = dataset.variables['SST'][test_indices]
# aviso_test = dataset.variables['AVISO'][test_indices]
# time_test = dataset.variables['time'][test_indices]
# lat_test = dataset.variables['lat'][test_indices]
# lon_test = dataset.variables['lon'][test_indices]
# depth_data = dataset.variables['depth'][:]
# Extract data for test set using xarray (Dask will handle computations lazily)
pred_T_test = dataset['Temperature'].isel(profile_number=test_indices)
pred_S_test = dataset['Salinity'].isel(profile_number=test_indices)
sss_test = dataset['SSS'].isel(profile_number=test_indices)
sst_test = dataset['SST'].isel(profile_number=test_indices)
aviso_test = dataset['AVISO'].isel(profile_number=test_indices)
time_test = dataset['time'].isel(profile_number=test_indices)
lat_test = dataset['lat'].isel(profile_number=test_indices)
lon_test = dataset['lon'].isel(profile_number=test_indices)
depth_data = dataset['depth'].load()

#let's make the pcs for the test set
pred_T_test = np.nan_to_num(pred_T_test)
pred_S_test = np.nan_to_num(pred_S_test)


# Save test data
test_file_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/model_input/nespreso1_test_data_42.nc'
save_to_netcdf(
    pred_T_test, pred_S_test, depth_data, sss_test, sst_test, aviso_test, time_test, lat_test, lon_test,
    temp_pca=temp_pca_components, sal_pca=sal_pca_components, file_name=test_file_path, seed=seed, n_components=n_components,
    temp_mean=temp_mean, sal_mean=sal_mean
)