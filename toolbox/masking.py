## Masking class specialized for surface electrode array by Dartmouth

import matplotlib.pyplot as plt
import numpy as np
from typing import Union

vv_distance_close_skip_0 = {
    26: [1, 2, 6, 7],
    27: [4, 5, 9, 10],
    28: [4, 5, 9, 10],
    29: [19, 20, 24, 25],
    30: [19, 20, 24, 25],
    31: [16, 17, 21, 22],
    32: [16, 17, 21, 22],
    33: [1, 2, 6, 7],
}

vv_distance_medium_skip_0 = {
    26: [3, 4, 8, 9, 11, 12, 14, 16, 17, 18, 19],
    27: [2, 3, 7, 8, 12, 14, 15, 17, 18, 19, 20],
    28: [2, 3, 7, 8, 12, 14, 15, 17, 18, 19, 20],
    29: [7, 8, 9, 10, 12, 14, 15, 17, 18, 22, 23],
    30: [7, 8, 9, 10, 12, 14, 15, 17, 18, 22, 23],
    31: [6, 7, 8, 9, 11, 12, 14, 18, 19, 23, 24],
    32: [6, 7, 8, 9, 11, 12, 14, 18, 19, 23, 24],
    33: [3, 4, 8, 9, 11, 12, 14, 16, 17, 18, 19],
}

vv_distance_far_skip_0 = {
    26: [4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    27: [1, 2, 6, 7, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    28: [1, 2, 6, 7, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    29: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 21, 22],
    30: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 21, 22],
    31: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19, 20, 24, 25],
    32: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19, 20, 24, 25],
    33: [4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
}

vv_distance_close_adj = {
    "26_27": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "27_28": [3, 4, 5, 8, 9, 10, 13, 14, 15],
    "28_29": [4, 5, 9, 10, 14, 15, 19, 20, 24, 25],
    "29_30": [13, 14, 15, 18, 19, 20, 23, 24, 25],
    "30_31": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "31_32": [11, 12, 13, 16, 17, 18, 21, 22, 23],
    "32_33": [1, 2, 6, 7, 11, 12, 16, 17, 21, 22],
    "33_26": [1, 2, 3, 6, 7, 8, 11, 12, 13],
}

vv_distance_medium_adj = {
    "26_27": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "27_28": [2, 3, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20],
    "28_29": [3, 4, 8, 9, 13, 14, 18, 19, 23, 24],
    "29_30": [7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 22, 23],
    "30_31": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "31_32": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 23, 24],
    "32_33": [2, 3, 7, 8, 12, 13, 17, 18, 22, 23],
    "33_26": [3, 4, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19],
}

vv_distance_far_adj = {
    "26_27": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "27_28": [1, 2, 6, 7, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "28_29": [1, 2, 6, 7, 11, 12, 16, 17, 21, 22],
    "29_30": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 21, 22],
    "30_31": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "31_32": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19, 20, 24, 25],
    "32_33": [4, 5, 9, 10, 14, 15, 19, 20, 24, 25],
    "33_26": [4, 5, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
}

vv_distance_close_skip_1 = {
    "26_28": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "27_29": [4, 5, 9, 10, 14, 15, 19, 20, 24, 25],
    "28_30": [4, 5, 9, 10, 14, 15, 19, 20, 24, 25],
    "29_31": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "30_32": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "31_33": [1, 2, 6, 7, 11, 12, 16, 17, 21, 22],
    "32_26": [1, 2, 6, 7, 11, 12, 16, 17, 21, 22],
    "33_27": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

vv_distance_medium_skip_1 = {
    "26_28": [6, 7, 8, 9, 11, 12, 13, 14, 18, 19, 23, 24],
    "27_29": [6, 7, 8, 9, 11, 12, 13, 14, 18, 19, 23, 24],
    "28_30": [3, 4, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19],
    "29_31": [3, 4, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19],
    "30_32": [2, 3, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20],
    "31_33": [2, 3, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20],
    "32_26": [7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 22, 23],
    "33_27": [7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 22, 23],
}

vv_distance_far_skip_1 = {
    "26_28": [11, 12, 13, 16, 17, 18, 21, 22, 23],
    "27_29": [11, 12, 13, 16, 17, 18, 21, 22, 23],
    "28_30": [1, 2, 3, 6, 7, 8, 11, 12, 13],
    "29_31": [1, 2, 3, 6, 7, 8, 11, 12, 13],
    "30_32": [3, 4, 5, 8, 9, 10, 13, 14, 15],
    "31_33": [3, 4, 5, 8, 9, 10, 13, 14, 15],
    "32_26": [13, 14, 15, 18, 19, 20, 23, 24, 25],
    "33_27": [13, 14, 15, 18, 19, 20, 23, 24, 25],
}

vv_distance_close_opp = {
    "26_30": [1, 2, 3, 7, 8, 12, 13, 14, 18, 19, 23, 24, 25],
    "27_31": [3, 4, 5, 8, 9, 12, 13, 14, 17, 18, 21, 22, 23],
    "28_32": [5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21],
    "29_33": [1, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 25],
    "30_26": [1, 2, 3, 7, 8, 12, 13, 14, 18, 19, 23, 24, 25],
    "31_27": [3, 4, 5, 8, 9, 12, 13, 14, 17, 18, 21, 22, 23],
    "32_28": [5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21],
    "33_29": [1, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 25],
}

# this is a list, because of multiple specifically defined ii electrodes
vv_long_across = np.array(
    [
        [26, 31, 1, 22],
        [26, 31, 1, 21],
        [26, 31, 2, 21],
        [26, 31, 2, 22],
        [27, 30, 4, 24],
        [27, 30, 4, 25],
        [27, 30, 5, 24],
        [27, 30, 5, 25],
        [28, 33, 1, 5],
        [28, 33, 1, 10],
        [28, 33, 5, 6],
        [28, 33, 6, 10],
        [29, 32, 16, 20],
        [29, 32, 16, 25],
        [29, 32, 20, 21],
        [29, 32, 21, 25],
    ]
)

vv_long_across_extended = {
    "26_31": [1, 2, 3, 21, 22, 23],
    "27_30": [3, 4, 5, 23, 24, 25],
    "28_33": [1, 5, 6, 10, 11, 15],
    "29_32": [11, 15, 16, 20, 21, 25],
    "30_26": [1, 2, 3, 23, 24, 25],
    "31_27": [3, 4, 5, 21, 22, 23],
    "32_28": [5, 10, 11, 15, 16, 21],
    "33_29": [1, 6, 11, 15, 20, 25],
}

vv_medium_across_extended = {
    "26_31": [6, 7, 8, 16, 17, 18],
    "27_30": [8, 9, 10, 18, 19, 20],
    "28_33": [2, 4, 7, 9, 12, 14],
    "29_32": [12, 14, 17, 19, 22, 24],
    "30_26": [6, 7, 8, 18, 19, 20],
    "31_27": [8, 9, 10, 16, 17, 18],
    "32_28": [12, 14, 17, 19, 22, 24],
    "33_29": [2, 7, 12, 14, 19, 24],
}

vv_distance_medium_opp = {
    "26_30": [5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21],
    "27_31": [1, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 25],
    "28_32": [1, 2, 3, 7, 8, 12, 13, 14, 18, 19, 23, 24, 25],
    "29_33": [3, 4, 5, 8, 9, 12, 13, 14, 17, 18, 21, 22, 23],
    "30_26": [5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21],
    "31_27": [1, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 25],
    "32_28": [1, 2, 3, 7, 8, 12, 13, 14, 18, 19, 23, 24, 25],
    "33_29": [3, 4, 5, 8, 9, 12, 13, 14, 17, 18, 21, 22, 23],
}

vv_distance_far_opp = {
    "26_30": [4, 5, 9, 10, 14, 15, 11, 12, 16, 17, 21, 22],
    "27_31": [1, 2, 6, 7, 11, 12, 14, 15, 19, 20, 24, 25],
    "28_32": [1, 2, 3, 6, 7, 8, 18, 19, 20, 23, 24, 25],
    "29_33": [3, 4, 5, 8, 9, 10, 16, 17, 18, 21, 22, 23],
    "30_26": [4, 5, 9, 10, 14, 15, 11, 12, 16, 17, 21, 22],
    "31_27": [1, 2, 6, 7, 11, 12, 14, 15, 19, 20, 24, 25],
    "32_28": [1, 2, 3, 6, 7, 8, 18, 19, 20, 23, 24, 25],
    "33_29": [3, 4, 5, 8, 9, 10, 16, 17, 18, 21, 22, 23],
}

def euclidean_distance(el1, el2):
    """
    Calculate Euclidean distance for single or arrays of electrodes.

    Parameters
    ----------
    el1 : int or array-like of ints
        Electrode indices (can be scalar or numpy array)
    el2 : int or array-like of ints
        Electrode indices (same shape as el1)

    Returns
    -------
    np.ndarray or float
        Distances between electrode pairs. If inputs are arrays, returns array
        of distances; else returns single float distance.
    """
    def index_to_coordinates(index):
        spacing = 1.0  # 1 mm spacing
        if np.isscalar(index):
            # Single electrode index
            if 1 <= index <= 25:
                row = (index - 1) // 5 + 1
                col = (index - 1) % 5 + 1
                x = (col - 3) * spacing
                y = (3 - row) * spacing
                return np.array([x, y])
            elif 26 <= index <= 33:
                current_coords = {
                    26: np.array([-3,  1]),
                    27: np.array([-3, -1]),
                    28: np.array([-1,  3]),
                    29: np.array([ 1,  3]),
                    30: np.array([ 3,  1]),
                    31: np.array([ 3, -1]),
                    32: np.array([ 1, -3]),
                    33: np.array([-1, -3]),
                }
                return current_coords.get(index, np.array([np.nan, np.nan]))
            else:
                return np.array([np.nan, np.nan])
        else:
            # Vectorized coordinates
            coords = np.empty((len(index), 2), dtype=float)
            for i, idx in enumerate(index):
                coords[i] = index_to_coordinates(idx)
            return coords

    # Convert to numpy arrays
    el1 = np.atleast_1d(el1)
    el2 = np.atleast_1d(el2)

    coords1 = index_to_coordinates(el1)
    coords2 = index_to_coordinates(el2)

    # Compute Euclidean distances element-wise
    diffs = coords1 - coords2
    dists = np.linalg.norm(diffs, axis=1)

    # If input is scalar, return scalar
    if dists.size == 1:
        return dists.item()
    return dists

class Masking:
    def __init__(self, IIVV, plot: bool = True, prints: bool = True):

        self.IIVV = np.array(IIVV, dtype=int)
        self.n_el_vv = 5
        self.ii_el = np.unique(IIVV[:, :2])
        self.vv_el = np.unique(IIVV[:, 2:])
        self.plot = plot
        self.prints = prints

    def count_true_false(self):

        self.true_count = sum(self.mask)
        self.false_count = len(self.mask) - self.true_count
        self.true_perc = self.true_count / len(self.mask) * 100
        self.false_perc = self.false_count / len(self.mask) * 100
        if self.prints:
            print(
                f"True: {self.true_count} ({self.true_perc:,.2f}%)\tFalse: {self.false_count}({self.false_perc:,.2f}%)"
            )

    def plot_mask(self, mode: int = 0):
        if mode == 0:
            plt.figure(figsize=(6, 2))
            plt.plot(self.mask, "C0", lw=2, label="Current self.mask")
            plt.xlabel("Mask index")
            plt.yticks(ticks=[0, 1], labels=["False", "True"])
            plt.legend()
            plt.show()
        else:
            plt.figure(figsize=(6, 2))
            plt.plot(self.IIVV[self.mask][:, 0], ".", markersize=1, label="$i_0$")
            plt.plot(self.IIVV[self.mask][:, 1], ".", markersize=1, label="$i_1$")
            plt.plot(self.IIVV[self.mask][:, 2], ".", markersize=1, label="$v_0$")
            plt.plot(self.IIVV[self.mask][:, 3], ".", markersize=1, label="$v_1$")
            plt.xlabel("Mask index")
            plt.ylabel("Electrode numbering")
            plt.grid()
            plt.legend()
            plt.show()

    def ii_mask(
        self, ii: list, f_i_u: str = "u", plot: bool = True, update_mask: bool = True
    ):
        """
        f_i_u : str (default u)
            f ... returns only forward injection ii[0], ii[1]
            i ... returns only inverse injection ii[1], ii[0]
            u ... returns the union injection ii[0], ii[1] | ii[1], ii[0]

        update_mask : bool (default True)
            True ... update self.mask to ii_mask
            False ... return the mask instead of updating self.mask
        """
        assert ii[0] in self.ii_el, "ii[0] is not a valid current injection electrode."
        assert ii[1] in self.ii_el, "ii[1] is not a valid current injection electrode."
        self.ii = ii
        # ii +/- with _f...forward
        mask_f = (self.IIVV[:, 0] == self.ii[0]) & (self.IIVV[:, 1] == self.ii[1])
        # ii -/+ with _i...inverse
        mask_i = (self.IIVV[:, 0] == self.ii[1]) & (self.IIVV[:, 1] == self.ii[0])
        # OR connection of True entries with _u...union
        mask_u = np.ma.mask_or(mask_f, mask_i)
        if plot:
            plt.figure(figsize=(6, 2))
            plt.plot(mask_u, "C2", lw=4, label="mask_f OR mask_i")
            plt.plot(mask_f, "C3", label="mask_f")
            plt.plot(mask_i, "C1", label="mask_i")

            plt.xlabel("Mask index")
            plt.yticks(ticks=[0, 1], labels=["False", "True"])
            plt.legend()
            plt.show()

        if f_i_u == "u" and update_mask:
            self.mask = mask_u
        elif f_i_u == "i" and update_mask:
            self.mask = mask_i
        elif f_i_u == "f" and update_mask:
            self.mask = mask_f
        if update_mask:
            if self.prints:
                print("Mask set by ii rule.")
            self.count_true_false()
        else:
            return mask_u

    def II_mask(self, skip: int, plot: bool = False, update_mask: bool = True):
        """
        Create a mask for all skip II Pattern
        """
        self.skip = skip
        self.ii_skip_pattern(skip)
        mask_ii_selection = list()
        for ii_inj in self.ii_pattern:
            # update_mask must be False here!
            tmp_mask = self.ii_mask(ii_inj, plot=plot, update_mask=False)
            mask_ii_selection.append(tmp_mask)

            num_true = np.count_nonzero(tmp_mask)
            num_false = tmp_mask.size - num_true
            assert num_true + num_false == len(tmp_mask), "Error"

        mask_ii_selection = np.array(mask_ii_selection)
        assert mask_ii_selection.shape[0] == self.ii_pattern.shape[0]

        II_mask = np.any(mask_ii_selection, axis=0)
        del mask_ii_selection

        if plot:
            plt.figure(figsize=(6, 2))
            plt.plot(II_mask, "C0", lw=2, label="II pattern mask")
            plt.xlabel("Mask index")
            plt.yticks(ticks=[0, 1], labels=["False", "True"])
            plt.legend()
            plt.show()

        if update_mask:
            self.mask = II_mask
            if self.prints:
                print("Mask set by II pattern rule.")
            self.count_true_false()
        else:
            return II_mask

    def vv_mask_distance(self, orthogonal_distance: int, operator="leq"):
        """
        operator
            - "leq" ... for distances less equal orthogonal_distance.
            - "less" ... for distances less orthogonal_distance.
        """
        if operator == "less":
            self.mask = (
                euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3])
                < orthogonal_distance
            )
        elif operator == "leq":
            self.mask = (
                euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3])
                <= orthogonal_distance
            )
        if self.prints:
            print("Mask set by vv rule.")
        self.count_true_false()

    def vv_mask_interval(self, orthogonal_distances: list, operator="[]"):
        """
        operator
            - "[]" ... closed interval [a,b] with a,b included.
            - "()" ... open interval (a,b) with a,b excluded.

            - "[)" ... half-open interval [a,b) includes a but not b.
            - "(]" ... half-open interval (a,b] excludes a but includes b.
        """
        a = orthogonal_distances[0]
        b = orthogonal_distances[1]
        assert a < b, "Please use a < b."

        if operator == "[]":
            if self.prints:
                print(f"Applied interval: [{a},{b}].")
            self.mask = (euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) >= a) & (
                euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) <= b
            )
        elif operator == "()":
            if self.prints:
                print(f"Applied interval: ({a},{b}).")
            self.mask = (euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) > a) & (
                euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) < b
            )
        elif operator == "[)":
            if self.prints:
                print(f"Applied interval: [{a},{b}).")
            self.mask = (euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) >= a) & (
                euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) < b
            )
        elif operator == "(]":
            if self.prints:
                print(f"Applied interval: ({a},{b}).")
            self.mask = (euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) > a) & (
                euclidean_distance(self.IIVV[:, 2], self.IIVV[:, 3]) <= b
            )
        else:
            if self.prints:
                print(f"Operator: {operator} unknown.")
            return
        if self.prints:
            print("Mask set by vv rule.")
        self.count_true_false()

    def ii_skip_pattern(self, skip: int):
        """
        skip ... defines the number of skipped electrodes between the injecting electrodes.
            skip = 0 ... adjacent ([26,27],[27,28]...[33,26])
            skip = 1 ... ([26,28],[27,29]...[33,27])
            skip = 2 ... ([26,29],[27,30]...[33,27])
            skip = 3 ... opposite ([26,30],[27,31]...[33,29])
            ...
        """
        self.skip = skip
        ii_pattern = list()
        for el in range(self.ii_el.shape[0]):
            i1 = np.take(self.ii_el, el, mode="wrap")
            i2 = np.take(self.ii_el, el + self.skip + 1, mode="wrap")
            ii_pattern.append([i1, i2])

        self.ii_pattern = np.array(ii_pattern, dtype=int)
        if self.prints:
            print(f"{ii_pattern=}")

    def IIVV_mask(
        self, vv_distance: str, skip=None, plot=True, update_mask: bool = True
    ):
        """
        vv_distance : takes the predefined dict vv_dictance dict (or list)

            available for 'skip == None'
            - "adjacent_close"
            - "adjacent_medium"
            - "adjacent_far"
            - "skip1_close"
            - "skip1_medium"
            - "skip1_far"
            - "opposite_close"
            - "opposite_medium"
            - "opposite_far"

            - "vv_long_across" (list)
            - "vv_long_across_extended"
            - "vv_medium_across_extended"
            - "vv_distance_close_skip_1"

            available for 'skip != None'
            - "close"
            - "medium"
            - "far"
        """

        def create_mask(dd_cond: Union[list, np.ndarray], mode="fromdict"):
            if mode == "fromdict":
                mask_f = (
                    (self.IIVV[:, 0] == self.i0) & (self.IIVV[:, 1] == self.i1)
                ) & (
                    np.isin(self.IIVV[:, 2], dd_cond)
                    & np.isin(self.IIVV[:, 3], dd_cond)
                )
                mask_i = (
                    (self.IIVV[:, 0] == self.i1) & (self.IIVV[:, 1] == self.i0)
                ) & (
                    np.isin(self.IIVV[:, 2], dd_cond)
                    & np.isin(self.IIVV[:, 3], dd_cond)
                )
                mask_u = np.ma.mask_or(mask_f, mask_i)
            elif mode == "fromarray":
                i0, i1, v0, v1 = dd_cond
                mask_f = ((self.IIVV[:, 0] == i0) & (self.IIVV[:, 1] == i1)) & (
                    np.isin(self.IIVV[:, 2], v0) & np.isin(self.IIVV[:, 3], v1)
                )
                mask_i = ((self.IIVV[:, 0] == i1) & (self.IIVV[:, 1] == i0)) & (
                    np.isin(self.IIVV[:, 2], v1) & np.isin(self.IIVV[:, 3], v0)
                )
                mask_u = np.ma.mask_or(mask_f, mask_i)
            return mask_u

        mask_IIVV_selection = list()
        ii_distances = [] 
        vv_distances = []
     

        if skip is None:
            if vv_distance == "all":
                self.mask = np.ones(7728, dtype=bool)
                if self.prints:
                    print("All set to True.")
                    self.count_true_false()
            elif vv_distance == "vv_long_across":
                self.ii_pattern = np.unique(vv_long_across[:, :2])
                if self.prints:
                    print(f"config: {vv_long_across=}")
                for dd in vv_long_across:
                    print(f" dd: {dd}")
                    tmp_mask = create_mask(dd, mode="fromarray")
                    mask_IIVV_selection.append(tmp_mask)
                    i0, i1, v0, v1 = dd 
                    ii_d = euclidean_distance(i0,i1)
                    vv_d = euclidean_distance(v0,v1)
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)

            elif vv_distance == "vv_long_across_extended":
                self.ii_pattern = [
                    list(map(int, key.split("_")))
                    for key in vv_long_across_extended.keys()
                ]
                if self.prints:
                    print(f"config: {vv_long_across_extended=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_long_across_extended[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)

            elif vv_distance == "vv_medium_across_extended":
                self.ii_pattern = [
                    list(map(int, key.split("_")))
                    for key in vv_medium_across_extended.keys()
                ]
                if self.prints:
                    print(f"config: {vv_medium_across_extended=}")

                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_medium_across_extended[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)

            elif vv_distance == "vv_distance_close_skip_1":
                self.ii_pattern = [
                    list(map(int, key.split("_")))
                    for key in vv_distance_close_skip_1.keys()
                ]
                if self.prints:
                    print(f"config: {vv_distance_close_skip_1=}")

                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_close_skip_1[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
            # --- NEW ---

            elif vv_distance == "adjacent_close":
                self.ii_skip_pattern(skip=0)
                if self.prints:
                    print(f"config: {vv_distance_close_adj=}")

                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_close_adj[f"{self.i0}_{self.i1}"]
                    print(dd)
                    print("type(dd)", type(dd))
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "vv_distance_medium_adj":
                self.ii_skip_pattern(skip=0)
                if self.prints:
                    print(f"config: {vv_distance_medium_adj=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_medium_adj[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "adjacent_far":
                self.ii_skip_pattern(skip=0)
                if self.prints:
                    print(f"config: {vv_distance_far_adj=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_far_adj[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "skip1_close":
                self.ii_skip_pattern(skip=1)
                if self.prints:
                    print(f"config: {vv_distance_close_skip_1=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_close_skip_1[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "skip1_medium":
                self.ii_skip_pattern(skip=1)
                if self.prints:
                    print(f"config: {vv_distance_medium_skip_1=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_medium_skip_1[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "skip1_far":
                self.ii_skip_pattern(skip=1)
                if self.prints:
                    print(f"config: {vv_distance_far_skip_1=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_far_skip_1[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "opposite_close":
                self.ii_skip_pattern(skip=3)
                if self.prints:
                    print(f"config: {vv_distance_close_opp=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_close_opp[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "opposite_medium":
                self.ii_skip_pattern(skip=3)
                if self.prints:
                    print(f"config: {vv_distance_medium_opp=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_medium_opp[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)
                    
            elif vv_distance == "opposite_far":
                self.ii_skip_pattern(skip=3)
                if self.prints:
                    print(f"config: {vv_distance_far_opp=}")

                for self.i0, self.i1 in self.ii_pattern:
                    dd = vv_distance_far_opp[f"{self.i0}_{self.i1}"]
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)

        if skip != None:
            self.ii_skip_pattern(skip)
            if vv_distance == "close":

                if self.prints:
                    print(f"config: {vv_distance_close_skip_0=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = np.concatenate(
                        [
                            vv_distance_close_skip_0[self.i0],
                            vv_distance_close_skip_0[self.i1],
                        ]
                    )
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)

            elif vv_distance == "medium":
                if self.prints:
                    print(f"config: {vv_distance_medium_skip_0=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = np.concatenate(
                        [
                            vv_distance_medium_skip_0[self.i0],
                            vv_distance_medium_skip_0[self.i1],
                        ]
                    )
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)

            elif vv_distance == "far":
                if self.prints:
                    print(f"config: {vv_distance_far_skip_0=}")
                for self.i0, self.i1 in self.ii_pattern:
                    dd = np.concatenate(
                        [
                            vv_distance_far_skip_0[self.i0],
                            vv_distance_far_skip_0[self.i1],
                        ]
                    )
                    tmp_mask = create_mask(dd)
                    mask_IIVV_selection.append(tmp_mask)
                    ii_d = euclidean_distance(self.i0,self.i1)
                    v0s = self.IIVV[tmp_mask,2]
                    v1s = self.IIVV[tmp_mask,3]
                    if len(v0s) > 0:
                        vv_d_array = euclidean_distance(v0s,v1s)
                        vv_d = np.mean(vv_d_array)
                    else:       
                        vv_d = np.nan
                    ii_distances.append(ii_d)
                    vv_distances.append(vv_d)

        # if non of the created cases were selected
        if len(mask_IIVV_selection) == 0:
            if self.prints:
                print(f"For {skip=} the {vv_distance=} is not implemented yet.")
            return

        # returning part
        mask_IIVV_selection = np.any(mask_IIVV_selection, axis=0)
        if update_mask:
            self.mask = mask_IIVV_selection
            if plot:
                self.plot_mask(mode=1)
            if self.prints:
                print("Mask set by IIVV pattern rule.")
            self.count_true_false()
        else:
            if plot and self.prints:
                print(
                    "Plotting currently not available. Please use the config:\n\t'plot=True, update_mask = True'"
                )
                
        ii_distances_clean = [d for d in ii_distances if not np.isnan(d)]
        vv_distances_clean = [d for d in vv_distances if not np.isnan(d)]
        avg_ii_distance = np.mean(ii_distances_clean) if ii_distances_clean else 0
        avg_vv_distance = np.mean(vv_distances_clean) if vv_distances_clean else 0

        if self.prints:
                print(f"Average II (current injection) distance across pattern sets: {avg_ii_distance:.3f} mm")
                print(f"Average VV (voltage electrode) distance across pattern sets: {avg_vv_distance:.3f} mm")


        return mask_IIVV_selection, avg_ii_distance, avg_vv_distance
