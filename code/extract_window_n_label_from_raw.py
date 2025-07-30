import mne  # type: ignore
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pycatch22 as catch22  # type: ignore
import os, gc
from joblib import Parallel, delayed


"""
Ce script sert a prendre en input les raw_with_annotation créer dans 'subject_omega' est a créer les deux
fichiers 'features_vectors' et 'time_series_windows' qui contiennent les windows de TS et les features 
pour l'entrainement, et cela pour chaque run importé. Ensuite le script 'get_one_big_dataset' va créer 
le dataset a partir de tout cela.
"""


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


def get_features_vector_from_one_channel_data(data_windows_one_channel):
    threads_to_use = os.cpu_count()
    results_list = Parallel(n_jobs=threads_to_use)(
        delayed(compute_features)(data_windows_one_channel[i])
        for i in range(len(data_windows_one_channel))
    )
    results_list = np.array(results_list)
    return results_list


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


def extract_window_from_indice_one_channel(data_one_channel, indices):
    X = []
    for indice in indices:
        X.append(data_one_channel[indice])
    return np.array(X)


def extract_good_indice_from_one_channel(
    data_one_channel, length, step, all_point_to_avoid
):
    n_total = data_one_channel.shape[0]
    all_indice = []

    for start in range(0, n_total - length + 1, step):
        potential_indice = [i for i in range(start, start + length)]
        set_potential_indice = set(potential_indice)
        flag = True
        for windows_to_avoid in all_point_to_avoid:
            set_avoid = set(tuple(windows_to_avoid))
            if set_avoid.intersection(set_potential_indice):
                flag = False
        if flag:
            all_indice.append(potential_indice)

    return all_indice


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


# NOTE : J'ai fait un filtrage de freq de 0.5 à 150Hz sur les données d'entrainement pour le dataset data_numpy_05s_windows
# NOTE : J'ai fait un resample à 1200Hz et un filtrage de 0.5 à 200Hz sur les données du dataset data_05s_downsample
def get_data(raw, lengh_seconde=0.5, step_seconde=0.25):
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
    raw_copy.filter(l_freq=0.5, h_freq=200, verbose=False)

    # ACTION ON THE RAW
    new_sfreq = 1200
    raw_copy = raw_copy.copy().resample(new_sfreq)

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
    all_windows_to_avoid += cardiac_windows_to_avoid_list
    all_windows_to_avoid += windows_bad_list

    all_data, _ = raw_copy[:, :]

    lengh = int(lengh_seconde * sfreq)
    step = int(step_seconde * sfreq)

    all_X = []

    for i, channel in enumerate(all_data):
        if i == 0:
            indice_windows = extract_good_indice_from_one_channel(
                channel, lengh, step, all_windows_to_avoid
            )
        X = extract_window_from_indice_one_channel(channel, indice_windows)
        all_X.append(X)

    all_X = np.array(all_X).astype(np.float32)

    return all_X


def get_features(X):
    all_features_vector = []
    for data_one_channel in X:
        features_vector_one_channel = get_features_vector_from_one_channel_data(
            data_one_channel
        )
        all_features_vector.append(features_vector_one_channel)
    all_features_vector = np.array(all_features_vector).astype(np.float32)
    print(all_features_vector.shape)
    return all_features_vector


def merge_all_channel(tab_all_channel):
    print(tab_all_channel.shape)

    # In the case where there's no good windows
    if len(tab_all_channel.shape) > 2:
        tab_all_channel = tab_all_channel.reshape(-1, tab_all_channel.shape[-1])

    print(tab_all_channel.shape)
    return tab_all_channel


def list_folder(chemin):
    chemin_objet = Path(chemin)
    try:
        dossiers_ds = [elem for elem in chemin_objet.iterdir() if elem.is_dir()]
        return [dossier.name for dossier in dossiers_ds]
    except FileNotFoundError:
        return f"Le chemin '{chemin}' n'existe pas."
    except PermissionError:
        return f"Accès refusé au chemin '{chemin}'."


def lister_fichiers(dossier):
    try:
        # Liste tous les fichiers et dossiers dans le chemin donné
        fichiers = os.listdir(dossier)

        # Filtrer pour ne garder que les fichiers
        fichiers = [f for f in fichiers if os.path.isfile(os.path.join(dossier, f))]

        return fichiers
    except FileNotFoundError:
        return "Le dossier spécifié n'existe pas."
    except PermissionError:
        return "Permission refusée pour accéder au dossier."
    except Exception as e:
        return f"Une erreur est survenue: {e}"


def nom_sans_extension(fichier):
    # Récupère le nom du fichier sans l'extension
    nom_sans_ext = os.path.splitext(fichier)[0]
    return nom_sans_ext


if __name__ == "__main__":
    path_to_data = "../data"
    all_sub_name = list_folder(path_to_data)
    compteur = 0
    for sub in all_sub_name:
        path_to_sub = path_to_data + "/" + sub

        path_to_save = "../data_05s_downsample/" + sub
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        file_in_save_sub = lister_fichiers(path_to_save)

        if (
            "time_series_windows.npy" in file_in_save_sub
            or "features_vectors.npy" in file_in_save_sub
        ):
            continue
        else:
            raw = mne.io.read_raw_fif(
                path_to_sub + "/raw_with_annotations.fif",
                preload=True,
                verbose=False,
            )
            X = get_data(raw, lengh_seconde=0.5, step_seconde=0.25)
            X_all_channel_merge = merge_all_channel(X)
            np.save(path_to_save + "/time_series_windows.npy", X_all_channel_merge)

            features_vector = get_features(X)
            features_vector_all_channel_merge = merge_all_channel(features_vector)
            np.save(
                path_to_save + "/features_vectors.npy",
                features_vector_all_channel_merge,
            )
        compteur += 1
        print(compteur)
        if compteur == 50:
            break
