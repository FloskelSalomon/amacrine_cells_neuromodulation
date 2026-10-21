import numpy as np

def pad_to_length(arr, target_length, pad_value=np.nan, n_last=500):
    """
    Pad a 1D array at the end up to target_length.

    pad_value : float or "mean"
        A constant (e.g. np.nan, 0) or "mean" to pad with the mean
        of the last `n_last` timepoints of the array.
    n_last : int
        Number of final timepoints used when pad_value="mean".
    """
    arr = np.asarray(arr, dtype=float)
    pad_width = target_length - arr.shape[0]
    if pad_width < 0:
        raise ValueError(f"Array longer than target length: {arr.shape[0]} > {target_length}")

    if isinstance(pad_value, str):
        if pad_value == "mean":
            pad_value = np.nanmean(arr[-n_last:])
        else:
            raise ValueError(f"Unknown pad_value option: {pad_value!r}")

    return np.pad(arr, (0, pad_width), mode="constant", constant_values=pad_value)

def check_array_lengths(array1, array2):
    arrays = list(array1) + list(array2)

    lengths = np.array([len(arr) for arr in arrays])

    return {
        "unique_lengths": np.unique(lengths),
        "min_length": lengths.min(),
        "max_length": lengths.max()
    }