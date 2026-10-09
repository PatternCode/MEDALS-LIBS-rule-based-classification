# Data

The MEDALS datasets required by the classifier are not included in this repository.

To reproduce the experiment, place the following directories under `data/`:

- `SyntheticLIBS`
- `Medals_Samples_Feb26`
- `SteelGrades`

Expected structure:

data/
├── SyntheticLIBS/
├── Medals_Samples_Feb26/
└── SteelGrades/

Run the classifier from the repository root:

python rule_based_fe_cu_classifier.py
