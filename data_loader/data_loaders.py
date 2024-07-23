# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from netCDF4 import Dataset as NetCDFDataset
# import pandas as pd
# from base import BaseDataLoader
# from utils import time_sine_cosine, lat_sine_cosine, lon_sine_cosine

# class NeSPReSO1DataLoader(BaseDataLoader):
#     """
#     DataLoader for the NeSPReSO1 dataset, with dynamic input selection.
#     """
#     data_loaded = False
#     nc_data = None
#     n_components = None
#     T = None
#     temp_pca = None
#     S = None
#     sal_pca = None
#     range = None
#     lat = None
#     lon = None
#     sss = None
#     sst = None
#     sst_units = None
#     aviso = None
#     time = None
#     depth = None
#     temp_pcs = None
#     sal_pcs = None
#     t_pca_variances = None
#     sal_pca_variances = None
#     t_mean = None
#     s_mean = None
#     dates = None
#     annual_cycle_cos = None
#     annual_cycle_sin = None
#     lat_sine = None
#     lat_cosine = None
#     lon_sine = None
#     lon_cosine = None

#     def __init__(self, data_dir, batch_size, ignore_params=[], shuffle=True, validation_split=0.1765, num_workers=1):
#         self.data_file = data_dir
#         self.ignore_params = ignore_params

#         if not NeSPReSO1DataLoader.data_loaded:
#             self._load_data()

#         self.dataset = NeSPReSO1Dataset(self.temp_pcs, self.sal_pcs, self.lat, self.lon, self.sss, self.sst, self.aviso,
#                                         self.annual_cycle_cos, self.annual_cycle_sin, self.lat_cosine, self.lat_sine,
#                                         self.lon_cosine, self.lon_sine, self.ignore_params, self.t_pca_variances, self.sal_pca_variances, self.range)

#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#     def _load_data(self):
#         """
#         Load data from NetCDF file and store it in class-level variables.
#         """
#         self.nc_data = NetCDFDataset(self.data_file, 'r')
        
#         # Extract relevant data
#         self.n_components = np.array(self.nc_data.getncattr('n_components'))
#         self.T = np.array(self.nc_data.variables['Temperature'][:])
#         self.temp_pca = np.array(self.nc_data.variables['Temperature_PC'][:])
#         self.S = np.array(self.nc_data.variables['Salinity'][:])
#         self.sal_pca = np.array(self.nc_data.variables['Salinity_PC'][:])
#         self.range = np.array(self.T.max() - self.T.min()), np.array(self.S.max() - self.S.min())
#         self.lat = np.array(self.nc_data.variables['lat'][:])
#         self.lon = np.array(self.nc_data.variables['lon'][:])
#         self.sss = np.array(self.nc_data.variables['SSS'][:])
#         self.sst = np.array(self.nc_data.variables['SST'][:])
#         self.sst_units = self.nc_data.variables['SST'].units
#         if "Kelvin" in self.sst_units:
#             self.sst = self.sst - 273
#         self.aviso = np.array(self.nc_data.variables['AVISO'][:])
#         self.time  = np.array(self.nc_data.variables['time'][:])
#         self.depth = np.array(self.nc_data.variables['depth'][:])
#         try:
#             self.temp_pcs = self.nc_data.variables['Temperature_PCS'][:]
#             self.sal_pcs = self.nc_data.variables['Salinity_PCS'][:]
#             self.t_pca_variances = self.nc_data.variables['Temperature_PCA_variances'][:]
#             self.sal_pca_variances = self.nc_data.variables['Salinity_PCA_variances'][:]
#         except KeyError:
#             # Create NaN arrays
#             self.temp_pcs = np.empty((self.lat.shape[0], self.n_components))
#             self.sal_pcs = np.empty((self.lat.shape[0], self.n_components))
#             self.t_pca_variances = np.empty((self.lat.shape[0], self.n_components))
#             self.sal_pca_variances = np.empty((self.lat.shape[0], self.n_components))
#             self.t_mean = self.nc_data.variables['T_mean_pca'][:]
#             self.s_mean = self.nc_data.variables['S_mean_pca'][:]

#         # Compute annual cycle
#         self.dates = pd.to_datetime("2015-03-31") + pd.to_timedelta(self.time, unit='D')
#         self.annual_cycle_cos, self.annual_cycle_sin = time_sine_cosine(self.dates, 'annual')

#         self.lat_sine, self.lat_cosine = lat_sine_cosine(self.lat)
#         self.lon_sine, self.lon_cosine = lon_sine_cosine(self.lon)

#         NeSPReSO1DataLoader.data_loaded = True

#         # Close the NetCDF file to release resources
#         self.nc_data.close()

#     def get_input_dim(self):
#         """
#         Provide input dimension for the model.
#         """
#         return self._calculate_input_dim()

#     def get_output_dim(self):
#         """
#         Provide output dimension for the model.
#         """
#         return 2 * self.n_components  # 2 times n_components for TEMP and SAL

#     def _calculate_input_dim(self):
#         """
#         Dynamically calculate the input dimension based on selected parameters.
#         """
#         input_dim = 0

#         if 'time' not in self.ignore_params:
#             input_dim += 2  # annual cycle cos and sin

#         if 'lat' not in self.ignore_params:
#             input_dim += 2  # lat cos and sin

#         if 'lon' not in self.ignore_params:
#             input_dim += 2  # lon cos and sin

#         if 'sss' not in self.ignore_params:
#             input_dim += 1

#         if 'sst' not in self.ignore_params:
#             input_dim += 1

#         if 'ssh' not in self.ignore_params:
#             input_dim += 1

#         return input_dim


# class NeSPReSO1Dataset(Dataset):
#     """
#     Custom dataset for NeSPReSO1, compatible with torch DataLoader.
#     """
#     def __init__(self, temp_pcs, sal_pcs, lat, lon, sss, sst, aviso, annual_cycle_cos,
#                  annual_cycle_sin, lat_cosine, lat_sine, lon_cosine, lon_sine,
#                  ignore_params, t_pca_variances, sal_pca_variances, range):
#         self.temp_pcs = temp_pcs
#         self.sal_pcs = sal_pcs
#         self.lat = lat
#         self.lon = lon
#         self.sss = sss
#         self.sst = sst
#         self.aviso = aviso
#         self.annual_cycle_cos = annual_cycle_cos
#         self.annual_cycle_sin = annual_cycle_sin
#         self.lat_cosine = lat_cosine
#         self.lat_sine = lat_sine
#         self.lon_cosine = lon_cosine
#         self.lon_sine = lon_sine
#         self.ignore_params = ignore_params
#         self.t_pca_variances = t_pca_variances
#         self.sal_pca_variances = sal_pca_variances
#         self.range = range
        
#     def __len__(self):
#         return self.temp_pcs.shape[0]

#     def __getitem__(self, idx):
#         inputs = []

#         if 'time' not in self.ignore_params:
#             inputs.extend([self.annual_cycle_cos[idx], self.annual_cycle_sin[idx]])

#         if 'lat' not in self.ignore_params:
#             inputs.extend([self.lat_cosine[idx], self.lat_sine[idx]])

#         if 'lon' not in self.ignore_params:
#             inputs.extend([self.lon_cosine[idx], self.lon_sine[idx]])

#         if 'sss' not in self.ignore_params:
#             inputs.append(self.sss[idx])

#         if 'sst' not in self.ignore_params:
#             inputs.append(self.sst[idx])

#         if 'ssh' not in self.ignore_params:
#             inputs.append(self.aviso[idx])

#         inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
#         profiles = torch.tensor(np.hstack([self.temp_pcs[idx, :], self.sal_pcs[idx, :]]), dtype=torch.float32)

#         return inputs_tensor, profiles
    
#     def get_PCS(self, idx):
#         return torch.tensor(np.hstack([self.temp_pcs[idx, :], self.sal_pcs[idx, :]]), dtype=torch.float32)

#     def get_pcs(self, idx):
#         return self.temp_pcs[idx, :], self.sal_pcs[idx, :]
    
#     def get_pca_variances(self):
#         # Concatenate the variances and convert them directly to a PyTorch tensor
#         concatenated_variances = np.concatenate([self.t_pca_variances[:], self.sal_pca_variances[:]])
#         # Convert the NumPy array to a PyTorch tensor
#         return torch.from_numpy(concatenated_variances).float()
    
#     def get_temp_pca_components(self):
#         return torch.from_numpy(self.temp_pcs).float()

#     def get_sal_pca_components(self):
#         return torch.from_numpy(self.sal_pcs).float()
    
#     def get_n_components(self):
#         return self.temp_pcs.shape[1]
    
#     def get_surface_T(self):
#         return torch.from_numpy(self.sst).float()
    
#     def get_surface_S(self):
#         return torch.from_numpy(self.sss).float()
    
#     def get_range(self):
#         return self.range

import numpy as np
import torch
from torch.utils.data import Dataset
from netCDF4 import Dataset as NetCDFDataset
import pandas as pd
from sklearn.decomposition import PCA
from base import BaseDataLoader
from utils import time_sine_cosine, lat_sine_cosine, lon_sine_cosine

class NeSPReSO1DataLoader(BaseDataLoader):
    """
    DataLoader for the NeSPReSO1 dataset, with dynamic input selection.
    """
    data_loaded = False
    nc_data = None
    n_components = None
    T = None
    temp_pca = None
    S = None
    sal_pca = None
    range = None
    lat = None
    lon = None
    sss = None
    sst = None
    sst_units = None
    aviso = None
    time = None
    depth = None
    temp_pcs = None
    sal_pcs = None
    t_pca_variances = None
    sal_pca_variances = None
    t_mean = None
    s_mean = None
    dates = None
    annual_cycle_cos = None
    annual_cycle_sin = None
    lat_sine = None
    lat_cosine = None
    lon_sine = None
    lon_cosine = None

    def __init__(self, data_dir, batch_size, ignore_params=[], shuffle=True, validation_split=0.1765, num_workers=1):
        self.data_file = data_dir
        self.ignore_params = ignore_params

        if not NeSPReSO1DataLoader.data_loaded:
            self._load_data()

        self.dataset = NeSPReSO1Dataset(self.temp_pcs, self.sal_pcs, self.lat, self.lon, self.sss, self.sst, self.aviso,
                                        self.annual_cycle_cos, self.annual_cycle_sin, self.lat_cosine, self.lat_sine,
                                        self.lon_cosine, self.lon_sine, self.ignore_params, self.t_pca_variances, self.sal_pca_variances, self.range)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _load_data(self):
        """
        Load data from NetCDF file and store it in class-level variables.
        """
        self.nc_data = NetCDFDataset(self.data_file, 'r')
        
        # Extract relevant data
        self.n_components = int(self.nc_data.getncattr('n_components'))
        self.T = np.array(self.nc_data.variables['Temperature'][:])
        self.S = np.array(self.nc_data.variables['Salinity'][:])
        self.lat = np.array(self.nc_data.variables['lat'][:])
        self.lon = np.array(self.nc_data.variables['lon'][:])
        self.sss = np.array(self.nc_data.variables['SSS'][:])
        self.sst = np.array(self.nc_data.variables['SST'][:])
        self.sst_units = self.nc_data.variables['SST'].units
        if "Kelvin" in self.sst_units:
            self.sst = self.sst - 273
        self.aviso = np.array(self.nc_data.variables['AVISO'][:])
        self.time  = np.array(self.nc_data.variables['time'][:])
        self.depth = np.array(self.nc_data.variables['depth'][:])

        # Compute annual cycle
        self.dates = pd.to_datetime("2015-03-31") + pd.to_timedelta(self.time, unit='D')
        self.annual_cycle_cos, self.annual_cycle_sin = time_sine_cosine(self.dates, 'annual')

        self.lat_sine, self.lat_cosine = lat_sine_cosine(self.lat)
        self.lon_sine, self.lon_cosine = lon_sine_cosine(self.lon)

        # Apply PCA
        pca_temp = PCA(n_components=self.n_components)
        self.temp_pcs = pca_temp.fit_transform(self.T)
        self.temp_pca = pca_temp.components_
        self.t_pca_variances = pca_temp.explained_variance_ratio_

        pca_sal = PCA(n_components=self.n_components)
        self.sal_pcs = pca_sal.fit_transform(self.S)
        self.sal_pca = pca_sal.components_
        self.sal_pca_variances = pca_sal.explained_variance_ratio_

        self.t_mean = pca_temp.mean_
        self.s_mean = pca_sal.mean_
        self.range = (self.T.max() - self.T.min(), self.S.max() - self.S.min())

        NeSPReSO1DataLoader.data_loaded = True

        # Close the NetCDF file to release resources
        self.nc_data.close()

    def get_input_dim(self):
        """
        Provide input dimension for the model.
        """
        return self._calculate_input_dim()

    def get_output_dim(self):
        """
        Provide output dimension for the model.
        """
        return 2 * self.n_components  # 2 times n_components for TEMP and SAL

    def _calculate_input_dim(self):
        """
        Dynamically calculate the input dimension based on selected parameters.
        """
        input_dim = 0

        if 'time' not in self.ignore_params:
            input_dim += 2  # annual cycle cos and sin

        if 'lat' not in self.ignore_params:
            input_dim += 2  # lat cos and sin

        if 'lon' not in self.ignore_params:
            input_dim += 2  # lon cos and sin

        if 'sss' not in self.ignore_params:
            input_dim += 1

        if 'sst' not in self.ignore_params:
            input_dim += 1

        if 'ssh' not in self.ignore_params:
            input_dim += 1

        return input_dim

class NeSPReSO1Dataset(Dataset):
    """
    Custom dataset for NeSPReSO1, compatible with torch DataLoader.
    """
    def __init__(self, temp_pcs, sal_pcs, lat, lon, sss, sst, aviso, annual_cycle_cos,
                 annual_cycle_sin, lat_cosine, lat_sine, lon_cosine, lon_sine,
                 ignore_params, t_pca_variances, sal_pca_variances, range):
        self.temp_pcs = temp_pcs
        self.sal_pcs = sal_pcs
        self.lat = lat
        self.lon = lon
        self.sss = sss
        self.sst = sst
        self.aviso = aviso
        self.annual_cycle_cos = annual_cycle_cos
        self.annual_cycle_sin = annual_cycle_sin
        self.lat_cosine = lat_cosine
        self.lat_sine = lat_sine
        self.lon_cosine = lon_cosine
        self.lon_sine = lon_sine
        self.ignore_params = ignore_params
        self.t_pca_variances = t_pca_variances
        self.sal_pca_variances = sal_pca_variances
        self.range = range
        
    def __len__(self):
        return self.temp_pcs.shape[0]

    def __getitem__(self, idx):
        inputs = []

        if 'time' not in self.ignore_params:
            inputs.extend([self.annual_cycle_cos[idx], self.annual_cycle_sin[idx]])

        if 'lat' not in self.ignore_params:
            inputs.extend([self.lat_cosine[idx], self.lat_sine[idx]])

        if 'lon' not in self.ignore_params:
            inputs.extend([self.lon_cosine[idx], self.lon_sine[idx]])

        if 'sss' not in self.ignore_params:
            inputs.append(self.sss[idx])

        if 'sst' not in self.ignore_params:
            inputs.append(self.sst[idx])

        if 'ssh' not in self.ignore_params:
            inputs.append(self.aviso[idx])

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        profiles = torch.tensor(np.hstack([self.temp_pcs[idx, :], self.sal_pcs[idx, :]]), dtype=torch.float32)

        return inputs_tensor, profiles
    
    def get_PCS(self, idx):
        return torch.tensor(np.hstack([self.temp_pcs[idx, :], self.sal_pcs[idx, :]]), dtype=torch.float32)

    def get_pcs(self, idx):
        return self.temp_pcs[idx, :], self.sal_pcs[idx, :]

    def get_pca_variances(self):
        # Concatenate the variances and convert them directly to a PyTorch tensor
        concatenated_variances = np.concatenate([self.t_pca_variances[:], self.sal_pca_variances[:]])
        # Convert the NumPy array to a PyTorch tensor
        return torch.from_numpy(concatenated_variances).float()
    
    def get_temp_pca_components(self):
        return torch.from_numpy(self.temp_pcs).float()

    def get_sal_pca_components(self):
        return torch.from_numpy(self.sal_pcs).float()
    
    def get_n_components(self):
        return self.temp_pcs.shape[1]
    
    def get_surface_T(self):
        return torch.from_numpy(self.sst).float()
    
    def get_surface_S(self):
        return torch.from_numpy(self.sss).float()
    
    def get_range(self):
        return self.range
