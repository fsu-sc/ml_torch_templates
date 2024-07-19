import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import pandas as pd
import datetime

def inverse_pca_transform(pca, pcs, mean):
    mult = pcs @ pca
    #broadcast mean
    return mult + mean

def time_sine_cosine(dates, cycle, multiple=1):
    """
    Compute the sine and cosine values for specified cycles.

    Parameters:
    - dates: list of datetime objects or single datetime object
    - cycle: string name of the cycle (e.g., 'diurnal', 'lunar')
    - multiple: integer multiplier for the cycle (e.g., 2 for double the period)

    Returns:
    - sine_vals: array of sine values
    - cosine_vals: array of cosine values
    """
    
    # Define the periods of different cycles in days
    CYCLE_PERIODS = {
        'diurnal': 1,
        'lunar': 29.53,
        'annual': 365.25,
        'solar': 11 * 365.25,  # converting solar cycle to days
        'milankovitch_eccentricity': 100000 * 365.25,
        'milankovitch_axial_tilt': 41000 * 365.25,
        'milankovitch_precession': 23000 * 365.25
    }
    
    if cycle not in CYCLE_PERIODS:
        raise ValueError(f"Cycle {cycle} is not recognized. Available cycles: {list(CYCLE_PERIODS.keys())}")
    
    # Convert input dates to pandas DatetimeIndex for easy manipulation
    if not isinstance(dates, (list, pd.DatetimeIndex)):
        dates = pd.DatetimeIndex([dates])
    else:
        dates = pd.DatetimeIndex(dates)
    
    # Get the period in days and adjust for multiple
    period_days = CYCLE_PERIODS[cycle] * multiple
    
    # Calculate days since a reference date (e.g., the Unix epoch)
    days_since_epoch = (dates - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    
    # Calculate the phase angle in radians
    phase_angle = 2 * np.pi * (days_since_epoch / period_days)
    
    # Calculate sine and cosine values
    sine_vals = np.sin(phase_angle)
    cosine_vals = np.cos(phase_angle)
    
    return sine_vals, cosine_vals

# # Example usage
# # Define a list of dates or a single date
# dates = [datetime.datetime(2023, 6, 21), datetime.datetime(2024, 6, 21)]
# cycle_name = 'annual'
# multiple = 1

# sine_vals, cosine_vals = time_sine_cosine(dates, cycle_name, multiple)

# # Print results
# for date, sine_val, cosine_val in zip(dates, sine_vals, cosine_vals):
#     print(f"Date: {date}, Sine: {sine_val:.4f}, Cosine: {cosine_val:.4f}")

def lat_sine_cosine(lat):
    """
    Compute the corresponding sine and cosine values for global latitude.

    Parameters:
    - lat: latitude in degrees
    """
    return np.sin(2*np.pi*(lat/180)), np.cos(2*np.pi*(lat/180))

def lon_sine_cosine(lon):
    """
    Compute the corresponding sine and cosine values for global longitude.

    Parameters:
    - lon: longitude in degrees
    """
    return np.sin(2*np.pi*(lon/360)), np.cos(2*np.pi*(lon/360))

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
