import os
from glob import glob

import numpy as np


SIGMA_O = 1.0
EPSILON_O = 1.0
CHARGE_O = 1.0
EXTRA_CONST = 0.0


def sample_single_rdf_file(filepath: str, width: float, temperature: float):
    """
    Read a rdf_static_result_*.txt file and build samples:
    features: [Row, width, temperature, EXTRA_CONST, SIGMA_O, EPSILON_O]
    label   : c_myRDF[2]
    Return array shape (n_rows, 7).
    """
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # data line: Row c_myRDF[1] c_myRDF[2] c_myRDF[3]
            if len(parts) < 4:
                continue
            try:
                row_val = float(parts[0])
                label_val = float(parts[2])
            except ValueError:
                continue

            # if label_val > 150:
            #     continue

            features = [
                row_val,
                float(width),
                float(temperature),
                EXTRA_CONST,
                SIGMA_O,
                EPSILON_O,
                label_val,
            ]
            samples.append(features)

    if not samples:
        return None

    return np.array(samples, dtype=float)


def choose_middle(values, ratio: float = 0.2):

    if not values:
        return set()
    if len(values) <= 2:
        return set(values)

    values_sorted = sorted(values, key=lambda x: float(x))
    n = len(values_sorted)
    k = max(1, int(round(n * ratio)))
    if k >= n:
        return set(values_sorted)

    start = (n - k) // 2
    end = start + k
    return set(values_sorted[start:end])


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    use_data_dir = os.path.join(base_dir, "use_data")
    out_data_dir = os.path.join(base_dir, "data")
    os.makedirs(out_data_dir, exist_ok=True)

    file_records = []  # list of (width_str, temp_str, data_array)

    # collect all (width, temperature) data
    width_folders = [
        d for d in os.listdir(use_data_dir) if d.startswith("rdf_results_width_")
    ]

    for width_folder in width_folders:
        folder_path = os.path.join(use_data_dir, width_folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            width_str_from_folder = width_folder.split("_")[-1]
            width_value = float(width_str_from_folder)
        except (IndexError, ValueError):
            continue

        pattern = os.path.join(folder_path, "rdf_static_result_*_*.txt")
        for filepath in glob(pattern):
            filename = os.path.basename(filepath)
            name_core = filename.replace(".txt", "")
            parts = name_core.split("_")
            if len(parts) < 4:
                continue

            width_str = parts[-2]
            temp_str = parts[-1]

            try:
                temp_value = float(temp_str)
            except ValueError:
                continue

            data = sample_single_rdf_file(
                filepath=filepath,
                width=width_value,
                temperature=temp_value,
            )
            if data is None:
                continue

            file_records.append((width_str, temp_str, data))

    if not file_records:
        print("No RDF data found in use_data.")
        raise SystemExit(0)

# Temperature Division:

# 1) The temperatures on both sides (low and high end) are used as the validation set;

# 2) Of the 8 temperatures in the middle, 2/8 of the middle section is used as the test set, and the rest are used as the training set.
    unique_temps = sorted({t for _, t, _ in file_records}, key=lambda x: float(x))
    n_temp = len(unique_temps)

    if n_temp == 0:
        print("No temperatures found in RDF data.")
        raise SystemExit(0)

    # Take a portion of the temperature from each side as the validation set (approximately 10% on each side).
    if n_temp >= 5:
        k_val = max(1, int(round(n_temp * 0.1)))
        if 2 * k_val >= n_temp:
            k_val = 1
        val_low = unique_temps[:k_val]
        val_high = unique_temps[-k_val:]
    else:
        val_low = [unique_temps[0]]
        val_high = [unique_temps[-1]] if n_temp > 1 else []

    val_temps = set(val_low + val_high)

    # The remaining temperature in the middle is used for train/test partitioning.
    mid_temp_list = [t for t in unique_temps if t not in val_temps]

    # Within the intermediate temperature range, approximately 2/8 ≈ 25% of the middle section is taken as the test set.
    mid_test_temps = choose_middle(mid_temp_list, ratio=0.4)
    train_temps = set(mid_temp_list) - mid_test_temps

    unique_widths = sorted({w for w, _, _ in file_records}, key=lambda x: float(x))
    mid_widths = choose_middle(unique_widths, ratio=0.5)

    print("Validation temperatures:", sorted(val_temps, key=float))
    print("Test temperatures (middle of remaining):", sorted(mid_test_temps, key=float))
    print("Train temperatures:", sorted(train_temps, key=float))
    print("Middle widths (for test):", sorted(mid_widths, key=float))

    train_data = []
    val_data = []
    test_data = []

    for width_str, temp_str, data in file_records:
        if temp_str in val_temps:
            val_data.append(data)

        elif (temp_str in mid_test_temps) and (width_str in mid_widths):
            test_data.append(data)
        else:
            train_data.append(data)

    if train_data:
        train_array = np.vstack(train_data)
        np.savez(os.path.join(out_data_dir, "train.npz"), data=train_array)

    if val_data:
        val_array = np.vstack(val_data)
        np.savez(os.path.join(out_data_dir, "valid.npz"), data=val_array)

    if test_data:
        test_array = np.vstack(test_data)
        np.savez(os.path.join(out_data_dir, "test.npz"), data=test_array)

