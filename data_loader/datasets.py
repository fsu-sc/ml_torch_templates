import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class NeSPReSO_nc_Dataset(Dataset):
    def __init__(self, file_path, n_components=15, transform=None):
        self.data = xr.open_dataset(file_path)
        self.n_components = n_components
        self.transform = transform
        
        # Load and preprocess data
        self.number_of_profiles = len(self.data['profile_number'].values)
        self.depth = self.data['depth'].values
        self.T_profiles = self.data['Temperature'].values
        self.S_profiles = self.data['Salinity'].values
        self.SSS = self.data['SSS'].values
        self.SST = self.data['SST'].values
        self.AVISO = self.data['AVISO'].values
        self.time = self.data['time'].values
        self.lat = self.data['lat'].values
        self.lon = self.data['lon'].values

        # TODO: implement flexible transform and inverse_transform methods (PCA, autoencoder, etc.)

    def __len__(self):
        return self.number_of_profiles
    
    def __getitem__(self, index):
        """
        Args:
        - idx (int): Index of the profile.

        Returns:
        - tuple: input values and concatenated PCA components for temperature and salinity.
        """
        
        # inputs = []
        
        # if self.input_params["timecos"]:
        #     inputs.append(np.cos(2*np.pi*(self.TIME[idx]%365)/365)) 
            # day_of_year = datetime.now().timetuple().tm_yday  # returns 1 for January 1st
            
        # if self.input_params["timesin"]:
        #     inputs.append(np.sin(2*np.pi*(self.TIME[idx]%365)/365))  
        
        # if self.input_params["latcos"]:
        #     inputs.append(np.cos(2*np.pi*(self.LAT[idx]/180)))

        # if self.input_params["latsin"]:
        #     inputs.append(np.sin(2*np.pi*(self.LAT[idx]/180)))  

        # if self.input_params["loncos"]:
        #     inputs.append(np.cos(2*np.pi*(self.LON[idx]/360)))  
            
        # if self.input_params["loncos"]:
        #     inputs.append(np.sin(2*np.pi*(self.LON[idx]/360)))
            
        # if self.input_params["sat"]:                
        #     if self.input_params["sss"]:
        #         # inputs.append(self.SAL[0, idx])
        #         inputs.append(self.SSS[idx])

        #     if self.input_params["sst"]:
        #         inputs.append(self.SST[idx] - 273.15) # convert from Kelvin to Celsius
                
        #     if self.input_params["ssh"]:
        #         # inputs.append(self.SH1950[idx]) #Uses profile SSH
        #         inputs.append(self.AVISO_ADT[idx]) #Uses satellite SSH
        # else:
        #     if self.input_params["sss"]:
        #         inputs.append(self.SAL[0, idx])
        #         # inputs.append(self.SSS[idx])

        #     if self.input_params["sst"]:
        #         inputs.append(self.TEMP[0, idx])  # First value of temperature profile
        #         # inputs.append(self.SST[idx])
                
        #     if self.input_params["ssh"]:
        #         inputs.append(self.SH1950[idx]) #Uses profile SSH
        #         # inputs.append(self.AVISO_ADT[idx]) #Uses satellite SSH
            
        # inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        # profiles = torch.tensor(np.hstack([self.temp_pcs[:, idx], self.sal_pcs[:, idx]]), dtype=torch.float32)
        # return inputs_tensor, profiles