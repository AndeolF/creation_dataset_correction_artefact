# CREATION_DATASET_CORRECTION_ARTEFACT

This is a repository created by Andéol FOURNIER during an internship under the supervision of Sylvain Baillet, PhD.

# 📦 Requirements

For a simplified and reproducible setup, it is recommended to use **[Poetry](https://python-poetry.org/)** to manage dependencies and the virtual environment. (files .toml and .lock are located at the root of the project)

---

## 🗂 Repository Structure

### `data/`

This folder contains the initial data (from the `subject_omega` repository). Each file is named `raw_with_annotations.fif` and stored in subfolders following the naming convention: `sub-ses-run`.

---

### `code/`

This folder includes the scripts used to process the data and build the final dataset:

---

#### `extract_windows_n_label_from_raw.py`

This script:

- Iterates through all `raw_with_annotations.fif` files in the `data/` folder,
- Extracts clean, artifact-free epochs from the raw files,
- Computes the **Catch22** features for each epoch,
- Saves the extracted epochs as `time_series_windows.npy`,
- Saves the corresponding features as `features_vectors.npy` in the same folder as the source data.

> **Note:** The raw files are usually preprocessed (e.g., filtering, downsampling), which is specified in the `get_data` function for the corresponding dataset.

---

#### `get_one_big_dataset.py`

This script:

- Collects all `time_series_windows.npy` and `features_vectors.npy` files across the folder data,
- Concatenates them into two large arrays: `big_array_time_series.npy` and `big_array_features_vectors.npy`,
- Builds the final dataset structure used for training and evaluation by splitting the data into train, validation, and evaluation sets.

It also saves:

- A summary file listing the data files used in the dataset creation,
- A `dataset/` folder containing the structured dataset ready for model training.

> **Note:** The data split proportions are adjusted due to the large dataset size and the generative nature of the task.

---

#### `test_features_coherence.py`

This script is used to validate the hypothesis that time series extracted close to an artifact-contaminated epoch produce more similar features.

