import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split


"""
Va récupérer tous les TS et features vectors contneue dans le dossier des sub et en faire un gros tableau où
tout est merge, puis va créer les set de train, val et eval.
small dataset contient les données de 5 run
medium dataset les données de 30 run
"""


def merge_all_sub_array(list_path_to_all_array, num_max_run=100):
    list_of_arrays = []
    list_files_use = []
    print(f"num max file to merge : {len(list_path_to_all_array)}")

    for i, data_path in enumerate(list_path_to_all_array):
        print(f"{i} ; {data_path}")
        try:
            array = np.load(data_path)
            list_files_use.append(data_path)
        except Exception as e:
            print(f"Erreur chargement {data_path}: {e}")
            array = np.empty((0,))

        if array.size > 0:
            list_of_arrays.append(array)

        if i == num_max_run:
            break

    if len(list_of_arrays) == 0:
        return np.empty((0,))

    print("stack")
    big_one_array = np.vstack(list_of_arrays)
    print(big_one_array.shape)
    return big_one_array, list_files_use


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


# def train_test_split_custom(array_data, array_label):
#     X_train_n_validation, X_test, y_train_n_validation, y_test = train_test_split(
#         array_data, array_label, test_size=0.05, random_state=11
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_n_validation, y_train_n_validation, test_size=0.2, random_state=11
#     )
#     return X_train, X_val, X_test, y_train, y_val, y_test


def train_test_split_custom(array_data, array_label):
    indices = np.arange(len(array_data))

    idx_train_val, idx_test = train_test_split(indices, test_size=0.01, random_state=11)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.07, random_state=11
    )

    return (
        array_data[idx_train],
        array_data[idx_val],
        array_data[idx_test],
        array_label[idx_train],
        array_label[idx_val],
        array_label[idx_test],
    )


def get_big_array(path_to_dataset, num_max_run=100):
    all_sub_name = list_folder(path_to_dataset)
    list_path_to_all_sub_times_series = []
    list_path_to_all_sub_features = []
    for sub in all_sub_name:
        list_path_to_all_sub_times_series.append(
            path_to_dataset + "/" + sub + "/time_series_windows.npy"
        )
        list_path_to_all_sub_features.append(
            path_to_dataset + "/" + sub + "/features_vectors.npy"
        )
    big_array_time_series, list_ts_files_use = merge_all_sub_array(
        list_path_to_all_sub_times_series, num_max_run=num_max_run
    )
    big_array_features_vectors, list_features_files_use = merge_all_sub_array(
        list_path_to_all_sub_features, num_max_run=num_max_run
    )

    return big_array_time_series, big_array_features_vectors, list_ts_files_use


def save_train_test_val(
    path_to_dataset, X_train, X_val, X_test, y_train, y_val, y_test, verbose=True
):
    if not os.path.exists(path_to_dataset + "/dataset"):
        os.makedirs(path_to_dataset + "/dataset")
    if verbose:
        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_val.shape)
        print(y_test.shape)
    np.save(path_to_dataset + "/dataset/train_time_series.npy", X_train)
    np.save(path_to_dataset + "/dataset/train_features.npy", y_train)
    np.save(path_to_dataset + "/dataset/val_time_series.npy", X_val)
    np.save(path_to_dataset + "/dataset/val_features.npy", y_val)
    np.save(path_to_dataset + "/dataset/test_time_series.npy", X_test)
    np.save(path_to_dataset + "/dataset/test_features.npy", y_test)


if __name__ == "__main__":
    path_to_dataset = "../data_05s_downsample"
    num_max_run = 50
    specific_dataset_folder_name = path_to_dataset + f"/{num_max_run}_run"

    # Pour obtenir les deux gros tableaux contenant les data
    if True:
        big_array_time_series, big_array_features_vectors, list_files_use = (
            get_big_array(path_to_dataset, num_max_run=num_max_run)
        )
        np_list_files_use = np.array(list_files_use)
        if not os.path.exists(specific_dataset_folder_name):
            os.makedirs(specific_dataset_folder_name)
        np.save(
            specific_dataset_folder_name + f"/big_array_{num_max_run}_run_time_series",
            big_array_time_series,
        )
        np.save(
            specific_dataset_folder_name
            + f"/big_array_{num_max_run}_run_features_vectors",
            big_array_features_vectors,
        )
        np.save(
            specific_dataset_folder_name + f"/list_files_use_{num_max_run}_run",
            np_list_files_use,
        )

    if True:
        big_array_time_series = np.load(
            specific_dataset_folder_name
            + f"/big_array_{num_max_run}_run_time_series.npy"
        )
        big_array_features_vectors = np.load(
            specific_dataset_folder_name
            + f"/big_array_{num_max_run}_run_features_vectors.npy"
        )
        print("feature and ts loaded")
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_custom(
            big_array_time_series, big_array_features_vectors
        )
        save_train_test_val(
            specific_dataset_folder_name,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            verbose=True,
        )

    if True:
        Y_train = np.load(specific_dataset_folder_name + "/dataset/train_features.npy")
        print(Y_train.shape)
        Y_test = np.load(specific_dataset_folder_name + "/dataset/test_features.npy")
        print(Y_test.shape)
        Y_val = np.load(specific_dataset_folder_name + "/dataset/val_features.npy")
        print(Y_val.shape)
        Y_train_small = np.load(
            "../data_numpy_05s_windows/small_dataset_05s_windows/features_train.npy"
        )
        print(Y_train_small.shape)
