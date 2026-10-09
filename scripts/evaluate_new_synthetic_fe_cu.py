from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from load_new_synthetic import load_new_synthetic

from rule_based_fe_cu_classifier import (
    classify_spectrum,
    evaluate_records,
    FE_LINES_NM,
    CU_LINES_NM,
    LINE_HALF_WINDOW_NM,
    BACKGROUND_INNER_NM,
    BACKGROUND_OUTER_NM,
    TOP_K_LINES,
)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "SyntheticLIBS"
    / "New_Syntethic_data"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "new_synthetic_evaluation"
)


DATASETS = {
    "1z": DATA_ROOT
    / "lsa_reference_grades_1z_Te_varied_std-0.20.csv",

    "2z": DATA_ROOT
    / "lsa_reference_grades_2z_Te_varied_std-0.20..csv",
}


# =============================================================================
# Fe/Cu family mapping
# =============================================================================

FE_LABELS = {
    "469",
    "470",
    "471",
    "472",
    "473",
    "SS405",
    "SS465-1",
    "SS467",
}

CU_LABELS = {
    "551",
    "552",
    "553",
    "554",
    "555",
    "556",
}


def material_family(label: str) -> str | None:
    """
    Map a material label to the binary Fe/Cu family.

    Materials outside the scope of the binary rule return None.
    """

    label = str(label).strip()

    if label in FE_LABELS:
        return "Fe"

    if label in CU_LABELS:
        return "Cu"

    return None


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_dataset(
    dataset_name: str,
    csv_path: Path,
):
    """
    Apply the existing fixed Fe/Cu rule to one new synthetic dataset.
    """

    print()
    print("=" * 80)
    print(
        f"EVALUATING NEW SYNTHETIC DATASET: "
        f"{dataset_name}"
    )
    print("=" * 80)

    X, y, wavelengths = load_new_synthetic(
        csv_path
    )

    output_dir = (
        OUTPUT_ROOT
        / dataset_name
        / "rule_based"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_records = []
    line_evidence_records = []

    excluded_records = []

    for sample_index, (
        spectrum,
        material_label,
    ) in enumerate(
        zip(X, y)
    ):

        material_label = str(
            material_label
        ).strip()

        true_family = material_family(
            material_label
        )

        # The current rule is intentionally binary Fe-vs-Cu.
        # Materials such as Al are outside its scope.
        if true_family is None:

            excluded_records.append(
                {
                    "sample_index": sample_index,
                    "material_grade_name": material_label,
                    "reason": (
                        "Outside Fe/Cu binary "
                        "classification scope"
                    ),
                }
            )

            continue

        result = classify_spectrum(
            wavelengths=wavelengths,
            spectrum=spectrum,
        )

        prediction_records.append(
            {
                "dataset": dataset_name,
                "sample_index": sample_index,
                "material_grade_name": material_label,
                "true_family": true_family,
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

        # Store evidence for every Fe line.
        for line_record in result[
            "fe_lines"
        ]:

            line_evidence_records.append(
                {
                    "dataset": dataset_name,
                    "sample_index": sample_index,
                    "material_grade_name": material_label,
                    "true_family": true_family,
                    "element": "Fe",
                    **line_record,
                }
            )

        # Store evidence for every Cu line.
        for line_record in result[
            "cu_lines"
        ]:

            line_evidence_records.append(
                {
                    "dataset": dataset_name,
                    "sample_index": sample_index,
                    "material_grade_name": material_label,
                    "true_family": true_family,
                    "element": "Cu",
                    **line_record,
                }
            )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    (
        results,
        summary,
        per_family,
        normalized_cm,
    ) = evaluate_records(
        prediction_records,
        domain=f"synthetic_{dataset_name}",
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    results.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
    )

    pd.DataFrame(
        line_evidence_records
    ).to_csv(
        output_dir
        / "line_evidence.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "summary_metrics.csv",
        index=False,
    )

    per_family.to_csv(
        output_dir
        / "per_family_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        normalized_cm,
        index=["Fe", "Cu"],
        columns=["Fe", "Cu"],
    ).to_csv(
        output_dir
        / "confusion_normalized.csv"
    )

    pd.DataFrame(
        excluded_records
    ).to_csv(
        output_dir
        / "excluded_samples.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print(
        f"Original spectra : {len(X)}"
    )

    print(
        f"Fe/Cu evaluated  : "
        f"{len(prediction_records)}"
    )

    print(
        f"Excluded         : "
        f"{len(excluded_records)}"
    )

    print()

    print(
        "Excluded labels:"
    )

    if excluded_records:

        excluded_df = pd.DataFrame(
            excluded_records
        )

        print(
            excluded_df[
                "material_grade_name"
            ]
            .value_counts()
            .to_string()
        )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Results written to:"
    )

    print(
        f"  {output_dir}"
    )

    return summary


# =============================================================================
# Main
# =============================================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print(
        "NEW SYNTHETIC DATA — FIXED RULE-BASED Fe/Cu EVALUATION"
    )
    print("=" * 80)

    print()
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
        f"Peak search half-width : "
        f"{LINE_HALF_WINDOW_NM:.2f} nm"
    )

    print(
        f"Background region      : "
        f"{BACKGROUND_INNER_NM:.2f}-"
        f"{BACKGROUND_OUTER_NM:.2f} nm"
    )

    print(
        f"Top lines per element  : "
        f"{TOP_K_LINES}"
    )

    summaries = []

    reference_wavelengths = None

    for dataset_name, csv_path in DATASETS.items():

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Dataset not found:\n{csv_path}"
            )

        # Verify that both generators use a compatible wavelength grid.
        _, _, wavelengths = load_new_synthetic(
            csv_path
        )

        if reference_wavelengths is None:

            reference_wavelengths = wavelengths

        else:

            if not np.array_equal(
                wavelengths,
                reference_wavelengths,
            ):

                raise ValueError(
                    f"{dataset_name}: wavelength grid "
                    f"does not match the first dataset."
                )

        summary = evaluate_dataset(
            dataset_name=dataset_name,
            csv_path=csv_path,
        )

        summaries.append(
            summary
        )

    combined_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    combined_summary.to_csv(
        OUTPUT_ROOT
        / "rule_based_summary_all_datasets.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "COMBINED SUMMARY"
    )
    print("=" * 80)

    print(
        combined_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    print(
        "Combined results:"
    )

    print(
        OUTPUT_ROOT
        / "rule_based_summary_all_datasets.csv"
    )


if __name__ == "__main__":
    main()
