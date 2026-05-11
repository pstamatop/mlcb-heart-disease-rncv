# Machine Learning in Computational Biology

## Heart Disease Classification with Repeated Nested CV

This project builds and evaluates leakage-safe machine learning pipelines for binary heart disease classification on the Cleveland subset (`data/students_dataset.csv`) taken from [MLCB_2026_Assignment_2 repository](https://github.com/MLCB2026/MLCB_2026_Assignment_2). It includes repeated/nested cross-validation, model selection, final pipeline training and SHAP-based interpretation.

### Core Workflow

- exploratory data analysis and data-quality checks
- repeated CV baseline comparison across multiple classifiers
- repeated nested CV with hyperparameter tuning
- feature-selection analysis inside CV
- final full pipeline training and model export
- SHAP interpretation on the final trained model
- Error Analysis

### Project Structure

- `notebooks/`: end-to-end analyses and reporting
- `src/`: reusable CV/model/metric utilities
- `results/`: fold-level outputs, summaries and rankings
- `models/`: saved pipeline artifacts and metadata
- `figures/`: generated plots
- `data/`: local CSV input data

### Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ is recommended.

### Run Order

1. `notebooks/exploratory_data_analysis.ipynb`
2. `notebooks/model_analysis.ipynb`
3. `notebooks/final_model_training_and_shap.ipynb`
4. `notebooks/validation_error_analysis.ipynb`

Notebooks support execution from repo root or from `notebooks/` via internal path bootstrapping.
