# Can Lab Measurements Predict Good Wine?

This project tests whether a wine's laboratory measurements (acidity, sugar,
sulfur dioxide, alcohol and so on) can predict whether expert tasters will rate
it **good**, meaning 7 or higher on a 0–10 scale. It uses the Vinho Verde wine
quality data from Cortez et al. (2009) and models red and white wines
separately. The comparison covers nine models in R: KNN, LDA, QDA,
classification trees, bagging and support vector machines.

📄 **[Read the full report (PDF)](report/report.pdf)**

![Test error by model](figures/test_error_by_model.png)

## Key findings

- **Accuracy is misleading here.** Only 14% of red wines and 22% of white wines
  are rated good, so labeling every wine *not good* already gets a test error of
  0.142 (red) and 0.207 (white). Several standard models barely beat this
  baseline. QDA and the two-predictor SVM match it or do worse.
- **Bagged trees performed best on both wines**, with test error of 0.092 (red)
  and 0.127 (white). Even so, they identify only about **55% of the good
  wines**.
- **Alcohol is by far the most important predictor.** Volatile acidity and
  sulphates follow it for red wine.
- **Standardizing features matters for KNN.** On raw features, sulfur dioxide
  (measured in the hundreds) dominates the distance calculation. Scaling the
  features cut white wine KNN error from 0.195 to 0.156.

| Model | Red error | Red sensitivity | White error | White sensitivity |
|---|---:|---:|---:|---:|
| Baseline (always "not good") | 0.142 | 0% | 0.207 | 0% |
| KNN (raw features) | 0.127 | 51% | 0.195 | 55% |
| KNN (standardized) | 0.117 | 62% | 0.156 | 64% |
| LDA | 0.110 | 41% | 0.192 | 31% |
| QDA | 0.140 | 49% | 0.236 | 69% |
| Tree (unpruned) | 0.110 | 40% | 0.213 | 44% |
| Tree (pruned) | 0.119 | 41% | 0.213 | 44% |
| **Bagged trees** | **0.092** | **54%** | **0.127** | **55%** |
| SVM (alcohol + volatile acidity) | 0.142 | 0% | 0.193 | 23% |
| SVM (all predictors) | 0.096 | 49% | 0.170 | 34% |

*Sensitivity is the share of good wines the model correctly identifies. All
numbers are on a held-out 30% test set.*

## Repository layout

```
├── R/analysis.R          # Full pipeline: load, split, fit, score, plot
├── data/                 # Raw UCI wine quality CSVs (red and white)
├── results/              # Metrics, tuned hyperparameters, variable importance
├── figures/              # Plots used in the report and this README
└── report/
    ├── report.qmd        # Quarto source; numbers are read from results/
    └── report.pdf        # Rendered report
```

## Reproducing the analysis

Requires R (4.x) and these packages:

```r
install.packages(c("class", "MASS", "tree", "randomForest", "e1071"))
```

From the project root, run:

```sh
Rscript R/analysis.R      # about 1–2 minutes; writes results/ and figures/
quarto render             # optional: rebuilds report/report.pdf (needs knitr and rmarkdown)
```

The 70/30 split uses `set.seed(1)`, so results are reproducible. Every figure in
the report is computed from `results/`, which keeps the text and the code in
sync.

## Limitations

- About 15–19% of rows duplicate another wine's measurements, and some
  duplicates land in both the training and test sets. This flatters
  memorization-heavy models such as 1-NN and bagging.
- All results come from a single train/test split. Repeated cross-validation
  would give tighter estimates.
- The binary cutoff discards the ordering in the quality score. Ordinal logistic
  regression is a natural next step.

## Data and citation

The data come from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
and are licensed under CC BY 4.0. Please cite:

> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling
> wine preferences by data mining from physicochemical properties. *Decision
> Support Systems, 47*(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016

## Background

I originally did this analysis as the final project for STAT 7510 and revised it
in 2026. The revision fixed several bugs in the original code, including an
error-rate calculation that used the wrong test-set size and a white wine SVM
that was scored with the red wine model. It also added the majority-class
baseline and sensitivity metrics, and made every result reproducible from a
single script.
