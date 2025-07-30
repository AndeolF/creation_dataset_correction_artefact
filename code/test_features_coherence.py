import numpy as np

import pycatch22 as catch22  # type: ignore
import os, gc
from joblib import Parallel, delayed
import numpy as np
from sklearn.preprocessing import StandardScaler
import mne  # type: ignore
import matplotlib.pyplot as plt


def plot_time_series(raw):
    raw.plot(duration=5, title="Raw", picks="meg")
    plt.show()


def compute_features(x):
    """OBTAINED VIA THE CATCH22 WEBSITE"""
    res = catch22.catch22_all(x, catch24=True)
    return res["values"]


def get_values_w_multithread_from_3D(threeD_ts):
    """OBTAINED VIA THE CATCH22 WEBSITE"""
    all_values = []
    for twoD_ts in threeD_ts:
        threads_to_use = os.cpu_count()
        results_list = Parallel(n_jobs=threads_to_use)(
            delayed(compute_features)(twoD_ts[i]) for i in range(len(twoD_ts))
        )
        all_values.append(results_list)
        del results_list
        gc.collect()
    all_values = np.array(all_values)
    return all_values


def get_feature_triplet_from_data(data_triplet_ts):
    data_triplet_feature = []
    for triplet_ts in data_triplet_ts:
        threads_to_use = os.cpu_count()
        triplet_feature = Parallel(n_jobs=threads_to_use)(
            delayed(compute_features)(triplet_ts[i]) for i in range(len(triplet_ts))
        )
        triplet_feature = np.array(triplet_feature)
        data_triplet_feature.append(triplet_feature)
    data_triplet_feature = np.array(data_triplet_feature)
    print(data_triplet_feature.shape)
    return data_triplet_feature


def extract_windows_from_onsets(onset, size):
    """
    Génère un tableau 2D contenant les fenêtres de points consécutifs à partir des indices de départ.

    Args:
        onset (np.ndarray): tableau 1D des indices de départ
        size (int): taille de chaque fenêtre

    Returns:
        np.ndarray: tableau 2D de forme (len(onset), size)
    """
    onset = np.asarray(onset).reshape(-1, 1)
    offsets = np.arange(size)
    return np.array(onset + offsets)


def extract_bad_windows(raw):
    annotations = raw.annotations
    sfreq = raw.info["sfreq"]

    bad_indices = []

    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if description.startswith("BAD_"):
            start_bad = int(onset * sfreq)
            stop_bad = int((onset + duration) * sfreq)
            bad_indices.append(np.array(range(start_bad, stop_bad)))

    return bad_indices


def get_epoch_from_event(raw, event_id, verbose=False):
    # ALL EVENT ID : {np.str_('blink'): 1, np.str_('button'): 2, np.str_('cardiac'): 3,
    # np.str_('deviant'): 4, np.str_('standard'): 5}
    events, event_id = mne.events_from_annotations(
        raw, event_id=event_id, verbose=False
    )

    annotations = raw.annotations

    if events.size > 0:
        events_name = list(event_id.keys())

        for event_name in events_name:
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description
            ):
                if str(description) == str(event_name):
                    tmax = round(float(duration), 2)
                    break

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
            event_repeated="drop",
        )
        name_event = str(next(iter(event_id.keys())).item())
        event_data = epochs[name_event].get_data()

        list_epoch_drop = []
        for i, log in enumerate(epochs.drop_log):
            if log:
                list_epoch_drop.append(i)
                if verbose:
                    print(f"Epoch {i} was dropped because: {log}")

        epoch_onsets = epochs.events[:, 0]

        return event_data, epoch_onsets, list_epoch_drop
    else:
        return [], [], []


def extract_valid_triplets(all_data, all_windows_to_avoid, length, step):
    """
    all_data : np.array shape (270, 720000)
    all_windows_to_avoid : list or np.array of indices to avoid (between 0 and 720000)
    length : length of each window
    step : step between each triplet (not between windows)
    """
    num_channels, total_length = all_data.shape

    valid_triplets = []
    current_start = 0

    # Check on channel 0 only
    while current_start + 3 * length <= total_length:
        w1_start = current_start
        w2_start = current_start + length
        w3_start = current_start + 2 * length

        # Create ranges for the three windows
        w1_range = range(w1_start, w1_start + length)
        w2_range = range(w2_start, w2_start + length)
        w3_range = range(w3_start, w3_start + length)

        # Check if any point in any window is in the avoid list
        flag = True
        for windows_to_avoid in all_windows_to_avoid:
            set_avoid = set(tuple(windows_to_avoid))

            if (
                any(i in set_avoid for i in w1_range)
                or any(i in set_avoid for i in w2_range)
                or any(i in set_avoid for i in w3_range)
            ):
                flag = False
                break
                # Valid triplet: collect the indices
        if flag:
            valid_triplets.append((w1_start, w2_start, w3_start))

        # Move to next triplet
        current_start += step

    print(len(valid_triplets))
    # Extract triplets from all channels
    all_triplet_data = []
    for w1_start, w2_start, w3_start in valid_triplets:
        # triplet = np.array(
        #     [
        #         all_data[:, w1_start : w1_start + length],
        #         all_data[:, w2_start : w2_start + length],
        #         all_data[:, w3_start : w3_start + length],
        #     ]
        # )
        triplet = np.stack(
            [
                all_data[:, w1_start : w1_start + length],
                all_data[:, w2_start : w2_start + length],
                all_data[:, w3_start : w3_start + length],
            ],
            axis=1,
        )  # shape: (270, 3, length)
        all_triplet_data.append(triplet)

    # Final array shape: (num_valid_triplets, 270, 3 * length)
    all_triplet_data = np.array(all_triplet_data)
    all_triplet_data = all_triplet_data.reshape(
        -1, all_triplet_data.shape[2], all_triplet_data.shape[3]
    )
    print(f"all_triplet_data.shape : {all_triplet_data.shape}")

    return all_triplet_data


def extract_data_triplet(raw, length_seconde=0.5, step_seconde=0.25):
    raw_copy = raw.copy()
    picks = mne.pick_types(
        raw_copy.info,
        meg=True,
        eeg=False,
        eog=False,
        stim=False,
        ref_meg=False,
    )
    raw_copy = raw_copy.pick(picks)
    raw_copy.filter(l_freq=0.5, h_freq=150, verbose=False)

    try:
        _, all_onset_blink, _ = get_epoch_from_event(
            raw_copy, {np.str_("blink"): 1}, verbose=False
        )
    except:
        all_onset_blink = []
    try:
        _, all_onset_cardiac, _ = get_epoch_from_event(
            raw_copy, {np.str_("cardiac"): 3}, verbose=False
        )
    except:
        all_onset_cardiac = []
    sfreq = raw_copy.info["sfreq"]

    blink_windows_to_avoid = extract_windows_from_onsets(all_onset_blink, 0.5 * sfreq)
    cardiac_windows_to_avoid = extract_windows_from_onsets(
        all_onset_cardiac, 0.1 * sfreq
    )
    windows_bad = extract_bad_windows(raw_copy)
    blink_windows_to_avoid_list = list(blink_windows_to_avoid)
    cardiac_windows_to_avoid_list = list(cardiac_windows_to_avoid)
    windows_bad_list = list(windows_bad)
    all_windows_to_avoid = blink_windows_to_avoid_list
    # all_windows_to_avoid += cardiac_windows_to_avoid_list
    all_windows_to_avoid += windows_bad_list

    all_data, _ = raw_copy[:, :]

    print(all_data.shape)

    print(f"all_windows_to_avoid : {len(all_windows_to_avoid)}\n")

    length = int(length_seconde * sfreq)
    step = int(step_seconde * sfreq)

    triplet_data = extract_valid_triplets(all_data, all_windows_to_avoid, length, step)
    print(triplet_data.shape)

    return triplet_data


def get_distance_from_triplet_features(data_triplet_features):
    # LOAD DATA
    features_before = data_triplet_features[:, 0, :]
    features_middle = data_triplet_features[:, 1, :]
    features_after = data_triplet_features[:, 2, :]

    # STD DATA
    big_numpy_features = np.vstack((features_before, features_middle, features_after))
    print(big_numpy_features.shape)
    scaler = StandardScaler()
    scaler.fit(big_numpy_features)
    features_before_std = scaler.transform(features_before)
    features_middle_std = scaler.transform(features_middle)
    features_after_std = scaler.transform(features_after)

    features_shuffle_std = features_before_std.copy()
    np.random.shuffle(features_shuffle_std)

    # COMPARE DISTANCE
    features_mean_std = (features_before_std + features_after_std) / 2
    distance_middle_mean = np.linalg.norm(features_middle_std - features_mean_std)
    print(f"distance_middle_mean : {distance_middle_mean:.02f}")
    distance_before_middle = np.linalg.norm(features_middle_std - features_before_std)
    print(f"distance_before_middle : {distance_before_middle:.02f}")
    distance_before_after = np.linalg.norm(features_after_std - features_before_std)
    print(f"distance_before_after : {distance_before_after:.02f}")
    distance_shuffle = np.linalg.norm(features_shuffle_std - features_before_std)
    print(f"distance_shuffle : {distance_shuffle:.02f}")
    return


if __name__ == "__main__":
    sub = "sub-0001_ses-02_run-05"
    # GET THE TIME SERIES TRIPLET
    if False:
        raw_path = "../data/" + sub + "/raw_with_annotations.fif"
        raw = mne.io.read_raw_fif(
            raw_path,
            preload=True,
            verbose=False,
        )
        # plot_time_series(raw)
        triplet_ts = extract_data_triplet(raw)
        print(f"\n\n{triplet_ts.shape}\n\n")
        np.save("../data/" + sub + "/triplet_ts.npy", triplet_ts)

    # GET THE FEATURES TRIPLET
    if False:
        triplet_ts = np.load("../data/" + sub + "/triplet_ts.npy")
        print(triplet_ts.shape)
        triplet_features = get_feature_triplet_from_data(triplet_ts)
        np.save("../data/" + sub + "/triplet_features.npy", triplet_features)

    # COMPARE THE FEATURES
    if True:
        triplet_features = np.load("../data/" + sub + "/triplet_features.npy")
        get_distance_from_triplet_features(triplet_features)
