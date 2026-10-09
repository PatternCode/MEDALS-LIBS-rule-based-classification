from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# =============================================================================
# Configuration
# =============================================================================

SATURATION_THRESHOLD = 65535

BASELINE_WIDTH_NM = 10.0

# Search around each nominal atomic line.
LINE_HALF_WINDOW_NM = 0.50

# Local background is estimated from the neighbourhood outside
# the immediate peak-search region.
BACKGROUND_INNER_NM = 0.80
BACKGROUND_OUTER_NM = 2.00

# Use the strongest N lines for each element.
TOP_K_LINES = 3


DATA_ROOT = Path("data")

SYNTHETIC_FILE = (
    DATA_ROOT
    / "SyntheticLIBS"
    / "classification_dataset.csv"
)

COMPOSITION_FILE = (
    DATA_ROOT
    / "SyntheticLIBS"
    / "compositions.csv"
)

FEB_ROOT = (
    DATA_ROOT
    / "Medals_Samples_Feb26"
)

STEEL_GRADES_ROOT = (
    DATA_ROOT
    / "SteelGrades"
)

OUTPUT_ROOT = (
    Path("outputs")
    / "rule_based_fe_cu"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Spectral lines
# =============================================================================

# Selected Fe I lines within the measured wavelength range.
#
# These are intentionally distributed across the spectrum rather than
# relying on a single line.

FE_LINES_NM = np.asarray(
    [
        358.26,
        372.13,
        382.18,
        414.61,
        418.95,
        426.42,
        470.74,
        561.52,
    ],
    dtype=float,
)


# Selected Cu I lines within the measured wavelength range.
#
# The very strong Cu I resonance lines near 324.75 and 327.40 nm
# cannot be used because they are outside the MEDALS wavelength range.

CU_LINES_NM = np.asarray(
    [
        510.55,
        515.32,
        521.82,
        570.02,
        578.21,
    ],
    dtype=float,
)


# =============================================================================
# Wavelength-domain morphological baseline correction
# =============================================================================

def build_wavelength_neighborhoods(
    wavelengths: np.ndarray,
    width_nm: float,
) -> tuple[np.ndarray, np.ndarray]:

    half_width = (
        width_nm / 2.0
    )

    left = np.searchsorted(
        wavelengths,
        wavelengths - half_width,
        side="left",
    )

    right = np.searchsorted(
        wavelengths,
        wavelengths + half_width,
        side="right",
    )

    return left, right


def wavelength_erosion(
    intensity: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:

    output = np.empty_like(
        intensity,
        dtype=float,
    )

    for i, (start, stop) in enumerate(
        zip(left, right)
    ):

        output[i] = np.min(
            intensity[start:stop]
        )

    return output


def wavelength_dilation(
    intensity: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:

    output = np.empty_like(
        intensity,
        dtype=float,
    )

    for i, (start, stop) in enumerate(
        zip(left, right)
    ):

        output[i] = np.max(
            intensity[start:stop]
        )

    return output


def baseline_correct(
    intensity: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:

    eroded = wavelength_erosion(
        intensity,
        left,
        right,
    )

    baseline = wavelength_dilation(
        eroded,
        left,
        right,
    )

    corrected = (
        intensity
        - baseline
    )

    # Negative values are intentionally retained.
    return corrected


# =============================================================================
# Line evidence
# =============================================================================

def calculate_line_evidence(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    line_nm: float,
) -> dict:
    """
    Calculate evidence for one expected atomic line.

    Peak intensity:
        maximum intensity inside ± LINE_HALF_WINDOW_NM

    Local background:
        median intensity in the two side regions between
        BACKGROUND_INNER_NM and BACKGROUND_OUTER_NM.

    Evidence:
        positive local peak contrast divided by the maximum
        positive intensity in the complete spectrum.
    """

    distance = np.abs(
        wavelengths
        - line_nm
    )

    peak_mask = (
        distance
        <= LINE_HALF_WINDOW_NM
    )

    background_mask = (
        (distance >= BACKGROUND_INNER_NM)
        & (distance <= BACKGROUND_OUTER_NM)
    )

    if not np.any(
        peak_mask
    ):
        raise ValueError(
            f"No wavelength channels near "
            f"{line_nm:.3f} nm."
        )

    if not np.any(
        background_mask
    ):
        raise ValueError(
            f"No background channels near "
            f"{line_nm:.3f} nm."
        )

    peak_region = spectrum[
        peak_mask
    ]

    background_region = spectrum[
        background_mask
    ]

    peak_index_local = np.argmax(
        peak_region
    )

    peak_wavelengths = wavelengths[
        peak_mask
    ]

    peak_intensity = float(
        peak_region[
            peak_index_local
        ]
    )

    observed_peak_wavelength = float(
        peak_wavelengths[
            peak_index_local
        ]
    )

    local_background = float(
        np.median(
            background_region
        )
    )

    contrast = (
        peak_intensity
        - local_background
    )

    # Negative evidence has no meaning for presence of an emission line.
    positive_contrast = max(
        0.0,
        contrast,
    )

    global_maximum = float(
        np.max(
            spectrum
        )
    )

    if global_maximum <= 0:

        relative_evidence = 0.0

    else:

        relative_evidence = (
            positive_contrast
            / global_maximum
        )

    return {
        "nominal_wavelength_nm": line_nm,
        "observed_peak_wavelength_nm": observed_peak_wavelength,
        "peak_intensity": peak_intensity,
        "local_background": local_background,
        "contrast": contrast,
        "evidence": relative_evidence,
    }


def calculate_element_score(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    lines_nm: np.ndarray,
) -> tuple[float, list[dict]]:
    """
    Calculate line evidence for every candidate line and return
    the average of the TOP_K_LINES strongest evidences.
    """

    evidence_records = []

    for line_nm in lines_nm:

        record = calculate_line_evidence(
            wavelengths=wavelengths,
            spectrum=spectrum,
            line_nm=float(
                line_nm
            ),
        )

        evidence_records.append(
            record
        )

    values = np.asarray(
        [
            record["evidence"]
            for record
            in evidence_records
        ],
        dtype=float,
    )

    number_to_use = min(
        TOP_K_LINES,
        len(values),
    )

    strongest = np.sort(
        values
    )[
        -number_to_use:
    ]

    score = float(
        np.mean(
            strongest
        )
    )

    return (
        score,
        evidence_records,
    )


def classify_spectrum(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
) -> dict:
    """
    Completely rule-based Fe-vs-Cu classification.
    """

    (
        fe_score,
        fe_records,
    ) = calculate_element_score(
        wavelengths,
        spectrum,
        FE_LINES_NM,
    )

    (
        cu_score,
        cu_records,
    ) = calculate_element_score(
        wavelengths,
        spectrum,
        CU_LINES_NM,
    )

    if fe_score > cu_score:

        prediction = "Fe"

    else:

        prediction = "Cu"

    denominator = (
        fe_score
        + cu_score
    )

    if denominator > 0:

        margin = (
            fe_score
            - cu_score
        ) / denominator

    else:

        margin = 0.0

    return {
        "prediction": prediction,
        "fe_score": fe_score,
        "cu_score": cu_score,
        "margin": margin,
        "fe_lines": fe_records,
        "cu_lines": cu_records,
    }


# =============================================================================
# Ground-truth family mapping
# =============================================================================

def load_family_mapping() -> dict[str, str]:

    compositions = pd.read_csv(
        COMPOSITION_FILE,
        sep=";",
        dtype={
            "label": "string"
        },
        low_memory=False,
    )

    mapping = {}

    for label, group in compositions.groupby(
        "label"
    ):

        matrix_elements = (
            group[
                "matrix_element"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(
            matrix_elements
        ) != 1:

            raise ValueError(
                f"Unexpected matrix-element "
                f"definition for {label}."
            )

        mapping[
            str(label)
        ] = matrix_elements[
            0
        ]

    return mapping


# =============================================================================
# Synthetic dataset
# =============================================================================

def load_synthetic_fe_cu(
    family_mapping: dict[str, str],
):

    df = pd.read_csv(
        SYNTHETIC_FILE,
        sep=";",
        dtype={
            "label": "string"
        },
        low_memory=False,
    )

    feature_columns = list(
        df.columns[
            1:
        ]
    )

    wavelengths = np.asarray(
        feature_columns,
        dtype=float,
    )

    families = (
        df["label"]
        .astype(str)
        .map(
            family_mapping
        )
    )

    keep = families.isin(
        [
            "Fe",
            "Cu",
        ]
    )

    filtered = df.loc[
        keep
    ].reset_index(
        drop=True
    )

    filtered_families = (
        families.loc[
            keep
        ]
        .reset_index(
            drop=True
        )
    )

    X = filtered[
        feature_columns
    ].to_numpy(
        dtype=float
    )

    labels = (
        filtered["label"]
        .astype(str)
        .to_numpy()
    )

    true_family = (
        filtered_families
        .astype(str)
        .to_numpy()
    )

    return (
        wavelengths,
        X,
        labels,
        true_family,
    )


# =============================================================================
# Real dataset discovery
# =============================================================================

def iter_real_files():

    # February dataset

    for sample_dir in sorted(
        path
        for path in FEB_ROOT.iterdir()
        if path.is_dir()
    ):

        label = (
            sample_dir.name
        )

        for path in sorted(
            sample_dir.glob(
                "*.csv"
            )
        ):

            yield (
                label,
                "Medals_Samples_Feb26",
                path,
            )

    # SteelGrades dataset

    for path in sorted(
        STEEL_GRADES_ROOT.glob(
            "*.csv"
        )
    ):

        if "_LSA_" not in path.name:

            raise ValueError(
                f"Cannot infer label "
                f"from {path.name}"
            )

        label = path.name.split(
            "_LSA_",
            maxsplit=1,
        )[0]

        yield (
            label,
            "SteelGrades",
            path,
        )


# =============================================================================
# Evaluate one domain
# =============================================================================

def evaluate_records(
    records: list[dict],
    domain: str,
):
    """
    Calculate binary Fe/Cu metrics.
    """

    results = pd.DataFrame(
        records
    )

    y_true = results[
        "true_family"
    ].to_numpy()

    y_pred = results[
        "predicted_family"
    ].to_numpy()

    labels = [
        "Fe",
        "Cu",
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    ).astype(float)

    row_sums = cm.sum(
        axis=1,
        keepdims=True,
    )

    normalized_cm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm
        ),
        where=row_sums != 0,
    )

    print()
    print("=" * 70)
    print(
        f"{domain.upper()} RESULTS"
    )
    print("=" * 70)

    print(
        f"Spectra evaluated : "
        f"{len(results)}"
    )

    print(
        f"Accuracy          : "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{macro_f1:.4f}"
    )

    print()

    for i, label in enumerate(
        labels
    ):

        print(
            f"{label}: "
            f"precision={precision[i]:.4f}, "
            f"recall={recall[i]:.4f}, "
            f"F1={f1[i]:.4f}, "
            f"support={support[i]}"
        )

    summary = pd.DataFrame(
        [
            {
                "domain": domain,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "number_of_spectra": len(
                    results
                ),
            }
        ]
    )

    per_class = pd.DataFrame(
        {
            "domain": domain,
            "family": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    return (
        results,
        summary,
        per_class,
        normalized_cm,
    )


# =============================================================================
# Confusion matrix plotting
# =============================================================================

def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    output_file: Path,
):

    labels = [
        "Fe-based",
        "Cu-based",
    ]

    fig, ax = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    image = ax.imshow(
        matrix,
        vmin=0,
        vmax=1,
        cmap="Blues",
    )

    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_yticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Predicted family"
    )

    ax.set_ylabel(
        "True family"
    )

    ax.set_title(
        title
    )

    for i in range(
        2
    ):

        for j in range(
            2
        ):

            value = float(
                matrix[
                    i,
                    j,
                ]
            )

            rgba = image.cmap(
                image.norm(
                    value
                )
            )

            luminance = (
                0.2126
                * rgba[0]
                + 0.7152
                * rgba[1]
                + 0.0722
                * rgba[2]
            )

            text_color = (
                "white"
                if luminance < 0.5
                else "black"
            )

            ax.text(
                j,
                i,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=text_color,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Fraction of true family"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# Main
# =============================================================================

def main():

    family_mapping = (
        load_family_mapping()
    )

    # =========================================================================
    # Synthetic data
    # =========================================================================

    (
        wavelengths,
        X_synthetic,
        synthetic_labels,
        synthetic_true_family,
    ) = load_synthetic_fe_cu(
        family_mapping
    )

    synthetic_records = []
    synthetic_line_records = []

    print()
    print("=" * 70)
    print(
        "PROCESSING SYNTHETIC Fe/Cu MATERIALS"
    )
    print("=" * 70)

    print(
        "Synthetic spectra:",
        len(
            X_synthetic
        ),
    )

    for index, spectrum in enumerate(
        X_synthetic
    ):

        result = classify_spectrum(
            wavelengths,
            spectrum,
        )

        synthetic_records.append(
            {
                "domain": "synthetic",
                "spectrum_id": index,
                "label": synthetic_labels[
                    index
                ],
                "true_family": (
                    synthetic_true_family[
                        index
                    ]
                ),
                "predicted_family": result[
                    "prediction"
                ],
                "fe_score": result[
                    "fe_score"
                ],
                "cu_score": result[
                    "cu_score"
                ],
                "margin": result[
                    "margin"
                ],
            }
        )

        for (
            element,
            line_records,
        ) in [
            (
                "Fe",
                result[
                    "fe_lines"
                ],
            ),
            (
                "Cu",
                result[
                    "cu_lines"
                ],
            ),
        ]:

            for line_result in line_records:

                synthetic_line_records.append(
                    {
                        "domain": "synthetic",
                        "spectrum_id": index,
                        "label": (
                            synthetic_labels[
                                index
                            ]
                        ),
                        "true_family": (
                            synthetic_true_family[
                                index
                            ]
                        ),
                        "element": element,
                        **line_result,
                    }
                )

    # =========================================================================
    # Real spectra
    # =========================================================================

    baseline_left, baseline_right = (
        build_wavelength_neighborhoods(
            wavelengths,
            BASELINE_WIDTH_NM,
        )
    )

    real_records = []
    real_line_records = []
    saturation_records = []

    total_real_fe_cu = 0
    saturated_real_fe_cu = 0

    real_id = 0

    print()
    print("=" * 70)
    print(
        "PROCESSING REAL Fe/Cu MATERIALS"
    )
    print("=" * 70)

    for (
        label,
        source_group,
        path,
    ) in iter_real_files():

        family = (
            family_mapping.get(
                label
            )
        )

        # Exclude Al- and Ni-matrix materials.
        if family not in [
            "Fe",
            "Cu",
        ]:
            continue

        df = pd.read_csv(
            path,
            sep=";",
            low_memory=False,
        )

        measured_columns = list(
            df.columns[
                2:
            ]
        )

        measured_wavelengths = np.asarray(
            measured_columns,
            dtype=float,
        )

        if not np.allclose(
            wavelengths,
            measured_wavelengths,
        ):

            raise ValueError(
                f"Wavelength grid "
                f"differs in {path}"
            )

        X_file = df[
            measured_columns
        ].to_numpy(
            dtype=float
        )

        for row_index, spectrum in enumerate(
            X_file
        ):

            total_real_fe_cu += 1

            saturated = bool(
                np.any(
                    spectrum
                    >= SATURATION_THRESHOLD
                )
            )

            saturation_records.append(
                {
                    "label": label,
                    "family": family,
                    "source_group": (
                        source_group
                    ),
                    "source_file": (
                        path.name
                    ),
                    "spectrum_number": (
                        row_index + 1
                    ),
                    "saturated": saturated,
                    "maximum_intensity": float(
                        np.max(
                            spectrum
                        )
                    ),
                }
            )

            if saturated:

                saturated_real_fe_cu += 1

                continue

            corrected = baseline_correct(
                spectrum,
                baseline_left,
                baseline_right,
            )

            result = classify_spectrum(
                wavelengths,
                corrected,
            )

            real_records.append(
                {
                    "domain": "real",
                    "spectrum_id": real_id,
                    "label": label,
                    "true_family": family,
                    "predicted_family": (
                        result[
                            "prediction"
                        ]
                    ),
                    "fe_score": result[
                        "fe_score"
                    ],
                    "cu_score": result[
                        "cu_score"
                    ],
                    "margin": result[
                        "margin"
                    ],
                    "source_group": (
                        source_group
                    ),
                    "source_file": (
                        path.name
                    ),
                    "spectrum_number": (
                        row_index + 1
                    ),
                }
            )

            for (
                element,
                line_records,
            ) in [
                (
                    "Fe",
                    result[
                        "fe_lines"
                    ],
                ),
                (
                    "Cu",
                    result[
                        "cu_lines"
                    ],
                ),
            ]:

                for line_result in line_records:

                    real_line_records.append(
                        {
                            "domain": "real",
                            "spectrum_id": (
                                real_id
                            ),
                            "label": label,
                            "true_family": (
                                family
                            ),
                            "element": (
                                element
                            ),
                            **line_result,
                        }
                    )

            real_id += 1

    # =========================================================================
    # Evaluation
    # =========================================================================

    (
        synthetic_results,
        synthetic_summary,
        synthetic_per_class,
        synthetic_cm,
    ) = evaluate_records(
        synthetic_records,
        "synthetic",
    )

    (
        real_results,
        real_summary,
        real_per_class,
        real_cm,
    ) = evaluate_records(
        real_records,
        "real",
    )

    # =========================================================================
    # Save numerical outputs
    # =========================================================================

    synthetic_results.to_csv(
        OUTPUT_ROOT
        / "predictions_synthetic.csv",
        index=False,
    )

    real_results.to_csv(
        OUTPUT_ROOT
        / "predictions_real.csv",
        index=False,
    )

    pd.DataFrame(
        synthetic_line_records
    ).to_csv(
        OUTPUT_ROOT
        / "line_evidence_synthetic.csv",
        index=False,
    )

    pd.DataFrame(
        real_line_records
    ).to_csv(
        OUTPUT_ROOT
        / "line_evidence_real.csv",
        index=False,
    )

    pd.DataFrame(
        saturation_records
    ).to_csv(
        OUTPUT_ROOT
        / "saturation_report.csv",
        index=False,
    )

    summary = pd.concat(
        [
            synthetic_summary,
            real_summary,
        ],
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_ROOT
        / "summary_metrics.csv",
        index=False,
    )

    per_class = pd.concat(
        [
            synthetic_per_class,
            real_per_class,
        ],
        ignore_index=True,
    )

    per_class.to_csv(
        OUTPUT_ROOT
        / "per_family_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        synthetic_cm,
        index=[
            "Fe",
            "Cu",
        ],
        columns=[
            "Fe",
            "Cu",
        ],
    ).to_csv(
        OUTPUT_ROOT
        / "confusion_synthetic_normalized.csv"
    )

    pd.DataFrame(
        real_cm,
        index=[
            "Fe",
            "Cu",
        ],
        columns=[
            "Fe",
            "Cu",
        ],
    ).to_csv(
        OUTPUT_ROOT
        / "confusion_real_normalized.csv"
    )

    # =========================================================================
    # Confusion-matrix figures
    # =========================================================================

    plot_confusion_matrix(
        synthetic_cm,
        (
            "Rule-Based Fe/Cu Classification "
            "- Synthetic Spectra"
        ),
        OUTPUT_ROOT
        / "confusion_synthetic_normalized.png",
    )

    plot_confusion_matrix(
        real_cm,
        (
            "Rule-Based Fe/Cu Classification "
            "- Real Spectra"
        ),
        OUTPUT_ROOT
        / "confusion_real_normalized.png",
    )

    # =========================================================================
    # Final summary
    # =========================================================================

    print()
    print("=" * 70)
    print(
        "RULE-BASED Fe/Cu CLASSIFICATION SUMMARY"
    )
    print("=" * 70)

    print(
        "Fe lines (nm):",
        ", ".join(
            f"{value:.2f}"
            for value in FE_LINES_NM
        ),
    )

    print(
        "Cu lines (nm):",
        ", ".join(
            f"{value:.2f}"
            for value in CU_LINES_NM
        ),
    )

    print()

    print(
        "Line search half-width:",
        f"{LINE_HALF_WINDOW_NM:.2f} nm",
    )

    print(
        "Top lines used per element:",
        TOP_K_LINES,
    )

    print()

    print(
        "Total real Fe/Cu spectra:",
        total_real_fe_cu,
    )

    print(
        "Saturated real spectra excluded:",
        saturated_real_fe_cu,
    )

    print(
        "Usable real Fe/Cu spectra:",
        len(
            real_records
        ),
    )

    print()

    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.4f}"
            ),
        )
    )

    print()

    print(
        "Results directory:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()