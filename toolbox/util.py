import numpy as np
import matplotlib.pyplot as plt


def get_freqs(freq_start_hz=100, freq_stop_hz=100_000, num_steps=31, plot=True):
    """
    Generates a list of logarithmically spaced frequencies and their indices.

    Parameters
    ----------
    freq_start_hz : float, optional
        Starting frequency in Hertz. Default is 100 Hz.
    freq_stop_hz : float, optional
        Ending frequency in Hertz. Default is 100,000 Hz.
    num_steps : int, optional
        Number of frequency steps (including `freq_stop_hz` as the final point).
        Default is 31.
    plot : bool, optional
        If True, plots the frequency values on a logarithmic y-axis. Default is True.

    Returns
    -------
    freq_idx : np.ndarray
        Array of frequency indices (e.g., [0, 1, 2, ..., num_steps-1]).
    freq_list : list of float
        List of logarithmically spaced frequencies in Hertz, rounded to 2 decimal places.

    Notes
    -----
    - Frequencies are computed using `np.logspace` between `freq_start_hz` and `freq_stop_hz`,
      with the final frequency explicitly set to `freq_stop_hz`.
    - The frequencies are suitable for use in impedance spectroscopy or other frequency-sweep measurements.

    """
    freq_list = np.logspace(
        start=np.log10(freq_start_hz),
        stop=np.log10(freq_stop_hz),
        num=num_steps - 1,
        endpoint=False,
        base=10,
    )
    freq_list = np.append(freq_list, freq_stop_hz).tolist()
    freq_idx = np.arange(len(freq_list))

    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(freq_list, "x")
        plt.grid()
        plt.yscale("log")
        plt.ylabel("frequency (Hz)")
        plt.xlabel("frequency index")

    freq_list = [np.round(freq, 2) for freq in freq_list]
    freq_list = np.array(freq_list, dtype=float)
    return freq_idx, freq_list


def squeeze1shape(arr):
    """
    Removes all singleton dimensions (dimensions of size 1) from the input array.

    Parameters
    ----------
    arr : np.ndarray
        Input array from which singleton dimensions should be removed.

    Returns
    -------
    np.ndarray
        A squeezed array with all dimensions of size 1 removed.

    Notes
    -----
    - This function is similar to `np.squeeze()`, but explicitly removes *all* singleton dimensions,
      regardless of their position.
    - Useful for cleaning up shapes after operations that add extra dimensions, e.g., (1, N, 1, M) -> (N, M).

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((1, 10, 1, 5))
    >>> squeeze1shape(a).shape
    (10, 5)
    """
    del_axs = np.where(np.array(arr.shape) == 1)[0]
    return np.squeeze(arr, axis=tuple(del_axs))


def z_score(data, along="global", print_info: bool = True):
    """
    Applies z-score normalization to the input data.

    Parameters
    ----------
    data : np.ndarray
        The input data array to be normalized. Can be multi-dimensional.
    along : str, optional
        Specifies the axis along which normalization should be applied:
        - "global" (default): Normalizes across all elements of `data`.
        - "freq": Normalizes along the frequency axis, i.e., per slice `data[:, freq, :]`.
    print_info : bool, optional
        If True (default), prints the mean and standard deviation before and after normalization.

    Returns
    -------
    data_z : np.ndarray
        The z-score normalized data array of the same shape as `data`.

    Notes
    -----
    - NaN values in `data` are ignored during mean and standard deviation computation.
    - The function assumes that the second axis (`axis=1`) represents frequency when `along="freq"` is used.

    """
    if print_info:
        print("before:", np.nanmean(data), np.nanstd(data))

    if along == "global":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        data_z = (data - mean) / std

    if along == "freq":
        data_z = np.zeros(data.shape)
        if print_info:
            print("Normalized along the frequency axis.")
        for freq in range(data.shape[1]):
            mean = np.nanmean(data[:, freq, :])
            std = np.nanstd(data[:, freq, :])
            data_z[:, freq, :] = (data[:, freq, :] - mean) / std

    if print_info:
        print("\nafter:", np.nanmean(data_z), np.nanstd(data_z))
    return data_z
