# MEDALS LIBS Rule-Based Classification

This repository contains a knowledge-based LIBS classification workflow developed within the MEDALS project.

The current implementation distinguishes Fe-based and Cu-based material families using selected characteristic Fe I and Cu I emission lines.

## Repository Structure

```text
.
├── data/
│   ├── Medals_Samples_Feb26/
│   ├── SteelGrades/
│   └── SyntheticLIBS/
│       ├── classification_dataset.csv
│       ├── compositions.csv
│       └── New_Syntethic_data/
│           ├── lsa_reference_grades_1z_Te_varied_std-0.20.csv
│           └── lsa_reference_grades_2z_Te_varied_std-0.20..csv
│
├── scripts/
│   ├── rule_based_fe_cu_classifier.py
│   ├── evaluate_new_synthetic_fe_cu.py
│   └── load_new_synthetic.py
│
├── requirements.txt
└── README.md
```

## Method

For each spectrum, the classifier:

1. searches around selected Fe I and Cu I emission lines;
2. estimates the local spectral background;
3. calculates line evidence from the peak-to-background contrast;
4. combines the strongest line evidence for each element; and
5. classifies the spectrum as Fe-based or Cu-based.

For measured spectra, saturated spectra are excluded and a wavelength-domain morphological baseline correction is applied before classification.

The current line sets are:

### Fe I Lines

```text
358.26 nm
372.13 nm
382.18 nm
414.61 nm
418.95 nm
426.42 nm
470.74 nm
561.52 nm
```

### Cu I Lines

```text
510.55 nm
515.32 nm
521.82 nm
570.02 nm
578.21 nm
```

The current implementation searches within ±0.50 nm of each nominal line and uses the strongest three line-evidence values for each element.

## Requirements

The main Python dependencies are:

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Running the Original Synthetic and Measured-Data Experiment

From the repository root, run:

```bash
python scripts/rule_based_fe_cu_classifier.py
```

This evaluates the fixed Fe/Cu rule on:

- the original synthetic dataset;
- the measured `Medals_Samples_Feb26` dataset; and
- the measured `SteelGrades` dataset.

The original synthetic data are loaded from:

```text
data/SyntheticLIBS/classification_dataset.csv
data/SyntheticLIBS/compositions.csv
```

Measured spectra are loaded from:

```text
data/Medals_Samples_Feb26/
data/SteelGrades/
```

Results are written to:

```text
outputs/rule_based_fe_cu/
```

Generated outputs include:

```text
predictions_synthetic.csv
predictions_real.csv
line_evidence_synthetic.csv
line_evidence_real.csv
saturation_report.csv
summary_metrics.csv
per_family_metrics.csv
confusion_synthetic_normalized.csv
confusion_synthetic_normalized.png
confusion_real_normalized.csv
confusion_real_normalized.png
```

## Running the 1z and 2z Synthetic-Data Experiment

The same fixed Fe/Cu rule can also be evaluated on the newer 1z and 2z synthetic datasets.

Run:

```bash
python scripts/evaluate_new_synthetic_fe_cu.py
```

The datasets are loaded from:

```text
data/SyntheticLIBS/New_Syntethic_data/
├── lsa_reference_grades_1z_Te_varied_std-0.20.csv
└── lsa_reference_grades_2z_Te_varied_std-0.20..csv
```

The loader used by this experiment is:

```text
scripts/load_new_synthetic.py
```

Results are written to:

```text
outputs/new_synthetic_evaluation/
```

with separate results for the 1z and 2z datasets and a combined summary file.

## Current Evaluation Results

Using the current fixed rule:

### Original Synthetic Dataset

```text
Spectra evaluated: 2100
Accuracy:          1.0000
Balanced accuracy: 1.0000
Macro F1:          1.0000
```

### Measured Data

```text
Total Fe/Cu spectra:           3126
Saturated spectra excluded:     386
Usable spectra:                2740

Accuracy:                      1.0000
Balanced accuracy:             1.0000
Macro F1:                      1.0000
```

### 1z Synthetic Dataset

```text
Original spectra: 1530
Fe/Cu evaluated:  1428
Excluded:          102

Accuracy:          1.0000
Balanced accuracy: 1.0000
Macro F1:          1.0000
```

The excluded spectra correspond to the `Al_ENAS-42000-1` material, which is outside the current Fe-vs-Cu classification scope.

### 2z Synthetic Dataset

```text
Original spectra: 1220
Fe/Cu evaluated:  1120
Excluded:          100

Accuracy:          1.0000
Balanced accuracy: 1.0000
Macro F1:          1.0000
```

The excluded spectra again correspond to `Al_ENAS-42000-1`.

## Current Scope

The current method is intended as a simple knowledge-based baseline for distinguishing broad Fe-based and Cu-based material families.

The reported results demonstrate strong separation between these two broad material families in the currently available datasets.

They should not be interpreted as demonstrating fine-grained steel-grade classification or complete material identification.

The current implementation is intended to support further development of knowledge-based, physics-informed, and hybrid approaches for LIBS-based material sorting within the MEDALS project.