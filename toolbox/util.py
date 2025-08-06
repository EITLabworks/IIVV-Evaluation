import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from os.path import join
import glob
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def computs_PCA(
    signal: np.ndarray,
    n_cmp: int = 2,
    feature: list = [],
    feature_label: str = "Feature",
    plot: bool = True,
):
    pca = PCA(n_cmp)
    P = pca.fit(signal).transform(signal)

    if n_cmp == 2 and plot:
        plt.figure(figsize=(6, 6))
        if len(feature) == 0:
            plt.scatter(P[:, 0], P[:, 1])
        else:
            norm = mcolors.Normalize(vmin=np.min(feature), vmax=np.max(feature))
            plt.scatter(P[:, 0], P[:, 1], c=feature, cmap="viridis", norm=norm)
            plt.colorbar(label=feature_label)
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.show()

    if n_cmp == 3 and plot:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        if len(feature) == 0:
            sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2])
        else:
            norm = mcolors.Normalize(vmin=np.min(feature), vmax=np.max(feature))
            sc = ax.scatter(
                P[:, 0], P[:, 1], P[:, 2], c=feature, cmap="viridis", norm=norm
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(feature_label)

        # Achsentitel setzen
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.set_zlabel("component 3")
        plt.show()


def compute_TSNE(
    X: np.ndarray,
    feature: list,
    feature_label: str = "Feature",
    plot: bool = True,
):

    X_reshaped = X.reshape(X.shape[0], -1)
    X_reshaped = np.nan_to_num(X_reshaped)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_reshaped)

    plt.figure(figsize=(6, 4))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=feature, cmap="jet", alpha=0.7)
    plt.colorbar(label=feature_label)
    plt.title("t-SNE")
    plt.show()


def visualize_iivv(iivv: list, save_name=None):
    # PCB
    # "gold" -> #d0b56e
    # "green" -> #1a7d3c
    i0, i1 = iivv[0], iivv[1]
    v0, v1 = iivv[2], iivv[3]
    # create meshgrid
    n_el_vv = 5
    x, y = np.meshgrid(np.arange(n_el_vv), np.arange(n_el_vv))

    # Enumerate electrodes for vv
    numbers = np.arange(1, n_el_vv**2 + 1).reshape((n_el_vv, n_el_vv))

    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"{iivv=}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2, n_el_vv + 1)
    ax.set_ylim(-2, n_el_vv + 1)
    ax.set_aspect("equal")

    # Set outer ii electrodes in a circle around the inner vv electrode matrix
    circle_radius = 3.5
    num_outer_electrodes = 8
    theta = np.linspace(0, 2 * np.pi, num_outer_electrodes, endpoint=False)
    x_outer = circle_radius * np.cos(theta - 2.5 * 2 * np.pi / 8) + (n_el_vv - 1) / 2
    y_outer = circle_radius * np.sin(theta - 2.5 * 2 * np.pi / 8) + (n_el_vv - 1) / 2

    # ii electrode areas
    for i in range(num_outer_electrodes):
        if i + 26 == i0:
            facecolor = "C1"
            edgecolor = "C3"  # sink <-> source
        elif i + 26 == i1:
            facecolor = "C1"
            edgecolor = "C3"  # sink <-> source
        else:
            facecolor = "#d0b56e"
            edgecolor = "black"
        ax.text(
            x_outer[i],
            y_outer[i],
            str(n_el_vv**2 + i + 1),
            fontsize=12,
            ha="center",
            va="center",
            color="white",
            bbox=dict(facecolor=facecolor, edgecolor=edgecolor, boxstyle="square"),
        )

    # vv electrode areas
    for i in range(n_el_vv):
        for j in range(n_el_vv):
            if numbers[i, j] == v0:
                txt = str(numbers[i, j])
                facecolor = "C0"
                edgecolor = "black"  # sink <-> source
            elif numbers[i, j] == v1:
                txt = str(numbers[i, j])
                facecolor = "C0"
                edgecolor = "black"  # sink <-> source
            else:
                txt = str(numbers[i, j])
                facecolor = "#d0b56e"
                edgecolor = "black"
            ax.text(
                x[i, j],
                y[i, j],
                txt,
                fontsize=12,
                ha="center",
                va="center",
                color="white",
                bbox=dict(facecolor=facecolor, edgecolor=edgecolor, boxstyle="circle"),
            )
    # inverse y-axis for right alignment
    plt.gca().invert_yaxis()
    plt.gca().set_facecolor("#1a7d3c")
    if save_name != None:
        plt.tight_layout()
        plt.savefig(save_name)
    plt.show()


def visualize_vv_grid(n_el_vv: int = 5):
    X, Y = np.meshgrid(np.arange(n_el_vv), np.arange(n_el_vv))
    x, y = X.flatten(), Y.flatten()

    numbers = np.arange(1, n_el_vv**2 + 1).reshape((n_el_vv, n_el_vv))

    fig, ax = plt.subplots()
    ax.scatter(x, y, c="C1", s=130)
    for i in range(n_el_vv):
        for j in range(n_el_vv):
            ax.text(
                X[i, j],
                Y[i, j],
                str(numbers[i, j]),
                fontsize=12,
                ha="center",
                va="center",
            )
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlim(-1, 5)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_ylim(-1, 5)
    ax.set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()


def visualize_Category_hist(Category):
    Category = [str(c) for c in Category]
    df = pd.DataFrame.from_dict(Counter([str(c) for c in Category]), orient="index")
    df.plot(kind="bar", legend=False)
    plt.xlabel("Category")
    plt.ylabel("Number of samples")
    plt.grid(True)
    plt.show()


### Conversion from .mat to .npz
def get_freqs(freq_start_hz=100, freq_stop_hz=100_000, num_steps=31, plot=True):
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
    return freq_idx, freq_list


def squeeze1shape(arr):
    del_axs = np.where(np.array(arr.shape) == 1)[0]
    return np.squeeze(arr, axis=tuple(del_axs))


def load_data(
    path: str,
    py_idx: bool,
    Zdata: bool,
    patient_no:bool,
    patient_sample:bool,
    Bx_Case: bool,
    Case: bool,
    Avg_Cutpats: bool,
    Sample: bool,
    Category: bool,
    print_info: bool = True,
):
    load_dict = dict()

    load_config = {
        "py_idx": py_idx,
        "Zdata": Zdata,
        "patient_no" : patient_no,
        "patient_sample" : patient_sample,
        "Bx_Case": Bx_Case,
        "Case": Case,
        "Avg_Cutpats": Avg_Cutpats,
        "Sample": Sample,
        "Category": Category,
    }

    # init lists
    for key, should_load in load_config.items():
        if should_load:
            load_dict[key] = list()
    if print_info:
        print(f"Loading {load_dict.keys()} ...")
    # load data
    for ele in np.sort(glob.glob(join(path, "*.npz"))):
        tmp = np.load(ele, allow_pickle=True)
        for key, should_load in load_config.items():
            if should_load and key in tmp:
                load_dict[key].append(tmp[key])

    try:
        for key, vals in load_dict.items():
            load_dict[key] = np.array(load_dict[key])
    except:
        print("Shape error.")
    return load_dict


def z_score(data, along="global", print_info: bool = True):
    """
    along : the axis, we want to apply z-score normalization.
        - global ... all data (default).
        - freq ... along each frequency.
        - print_info ... print norm info
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


def plot_vv_grid(emphasize_els: list = [], n_el_vv: int = 5):
    X, Y = np.meshgrid(np.arange(n_el_vv), np.arange(n_el_vv))
    x, y = X.flatten(), Y.flatten()
    numbers = np.arange(1, n_el_vv**2 + 1).reshape((n_el_vv, n_el_vv))

    fig, ax = plt.subplots()

    for i in range(n_el_vv**2):
        color = "green" if (i + 1) in emphasize_els else "red"
        ax.scatter(x[i], y[i], c=color, s=130)

    for i in range(n_el_vv):
        for j in range(n_el_vv):
            ax.text(
                X[i, j],
                Y[i, j],
                str(numbers[i, j]),
                fontsize=12,
                ha="center",
                va="center",
            )

    ax.set_xticks(range(n_el_vv))
    ax.set_yticks(range(n_el_vv))
    ax.set_xlim(-1, n_el_vv)
    ax.set_ylim(-1, n_el_vv)
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()
