import h5py
import numpy as np
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from visione import CliProgress


def peek_features_attributes(h5file: Path) -> Tuple[int, str]:
    """Return dimensionality and name of the features stored in ``h5file``.

    Works with both dataset and per-record group layouts.
    """
    with h5py.File(h5file, "r") as f:
        features_name = f.attrs.get("features_name")
        if "data" in f:
            dim = f["data"].shape[1]
            return dim, features_name
        # per-record groups
        first_key = next(iter(f.keys()))
        group = f[first_key]
        if "feature_vector" in group:
            ds = group["feature_vector"]
        elif "feature" in group:
            ds = group["feature"]
        else:
            ds = next(iter(group.values()))
        dim = ds.shape[-1]
        return dim, features_name


def load_features_compat(hdf5_files: Iterable[Path]) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield ``(id, feature_vector)`` from a sequence of HDF5 files.

    Compatible with files storing features either as ``ids``/``data`` datasets
    or as one group per record.
    """
    progress = CliProgress(total=0)

    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, "r") as f:
            if "ids" in f and "data" in f:
                ids = f["ids"].asstr()[:]
                features = f["data"][:]
                progress.total += len(features)
                for item in progress(zip(ids, features)):
                    yield item
            else:
                keys = list(f.keys())
                progress.total += len(keys)
                for record_id in progress(keys):
                    group = f[record_id]
                    if "feature_vector" in group:
                        ds = group["feature_vector"][:]
                    elif "feature" in group:
                        ds = group["feature"][:]
                    else:
                        dsname = next(iter(group.keys()))
                        ds = group[dsname][:]
                    yield record_id, np.asarray(ds)
