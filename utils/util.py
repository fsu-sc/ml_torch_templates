import yaml
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from typing import Any, Union


class _OrderedLoader(yaml.SafeLoader):
    """YAML loader that preserves mapping key order in OrderedDict."""


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> OrderedDict:
    """Build an OrderedDict from a YAML mapping node."""
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


_OrderedLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def ensure_dir(dirname: Union[str, Path]) -> None:
    """Create directory if it does not exist."""
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_yaml(fname: Union[str, Path]) -> OrderedDict:
    """
    Load a YAML file and return its contents as an OrderedDict.

    :param fname: Path to the YAML file.
    :return: Parsed configuration mapping.
    """
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        data = yaml.load(handle, Loader=_OrderedLoader)
    return data if data is not None else OrderedDict()


def write_yaml(content: Any, fname: Union[str, Path]) -> None:
    """
    Write a mapping to a YAML file.

    :param content: Configuration object to serialize.
    :param fname: Destination file path.
    """
    fname = Path(fname)
    with fname.open('wt', encoding='utf-8') as handle:
        yaml.dump(
            content,
            handle,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

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
