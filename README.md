# MEDALS LIBS Rule-Based Classification

This repository contains a knowledge-based LIBS classifier developed within the MEDALS project.

The current implementation distinguishes Fe-based and Cu-based materials using selected characteristic Fe I and Cu I emission lines.

## Method

For each spectrum, the classifier:

1. searches around selected Fe I and Cu I wavelengths;
2. estimates the local spectral background;
3. calculates line evidence from peak-to-background contrast;
4. combines the strongest line evidence for each element; and
5. classifies the material as Fe-based or Cu-based.

For measured spectra, saturated spectra are excluded and morphological baseline correction is applied before classification.

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
