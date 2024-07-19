import mat73
from datetime import datetime, timedelta

def datenum_to_datetime(matlab_datenum):
    # MATLAB's datenum (1) is equivalent to January 1, year 0000, but Python's datetime minimal year is 1
    # There are 366 days for year 0 in MATLAB (it's a leap year in proleptic ISO calendar)
    days_from_year_0_to_year_1 = 366
    python_datetime = datetime.fromordinal(int(matlab_datenum) - days_from_year_0_to_year_1) + timedelta(days=matlab_datenum % 1)
    return python_datetime

data_path =  "/unity/g2/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat"
aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
sss_folder = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"

data = mat73.loadmat(data_path)
TIME = [datenum_to_datetime(datenum) for datenum in data['TIME']]
