from pathlib import Path

import numpy as np
import pandas as pd


def load_new_synthetic(csv_path):
    """
    Load one of the contributor's new synthetic LIBS datasets.

    Returns
    -------
    X : np.ndarray
        Spectral intensity matrix with shape
        (n_spectra, n_wavelengths).

    y : np.ndarray
        Material labels with shape (n_spectra,).

    wavelengths : np.ndarray
        Wavelength grid in nm with shape (n_wavelengths,).
    """

    csv_path = Path(csv_path)

    df = pd.read_csv(
        csv_path,
        dtype={"material_grade_name": str},
        low_memory=False,
    )

    if "material_grade_name" not in df.columns:
        raise ValueError(
            f"{csv_path.name}: missing 'material_grade_name' column."
        )

    # Wavelength columns are the columns whose names can be
    # interpreted as floating-point wavelength values.
    wavelength_columns = []
    wavelengths = []

    for column in df.columns:
        try:
            wavelength = float(column)
        except (TypeError, ValueError):
            continue

        wavelength_columns.append(column)
        wavelengths.append(wavelength)

    wavelengths = np.asarray(wavelengths, dtype=float)

    X = df[wavelength_columns].to_numpy(dtype=float)

    y = (
        df["material_grade_name"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    # Basic validation
    if X.shape[0] != len(y):
        raise ValueError("Number of spectra and labels do not match.")

    if X.shape[1] != len(wavelengths):
        raise ValueError(
            "Number of spectral columns and wavelengths do not match."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("Spectral matrix contains NaN or infinite values.")

    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Wavelengths are not strictly increasing.")

    return X, y, wavelengths


def print_dataset_summary(csv_path):
    X, y, wavelengths = load_new_synthetic(csv_path)

    labels, counts = np.unique(y, return_counts=True)

    print(f"\nDataset: {Path(csv_path).name}")
    print(f"X shape:           {X.shape}")
    print(f"y shape:           {y.shape}")
    print(f"Wavelength shape:  {wavelengths.shape}")
    print(
        f"Wavelength range:  "
        f"{wavelengths[0]:.3f} - {wavelengths[-1]:.3f} nm"
    )
    print(f"Number of classes: {len(labels)}")

    print("\nClass counts:")
    for label, count in zip(labels, counts):
        print(f"  {label:20s} {count}")


if __name__ == "__main__":

    data_dir = Path(
        "data/SyntheticLIBS/New_Syntethic_data"
    )

    datasets = [
        data_dir / "lsa_reference_grades_1z_Te_varied_std-0.20.csv",
        data_dir / "lsa_reference_grades_2z_Te_varied_std-0.20..csv",
    ]

    for dataset in datasets:
        print_dataset_summary(dataset)