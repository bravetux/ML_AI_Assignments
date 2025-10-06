# Assignment 2 — Employee Attrition Prediction (KNN)

This small project demonstrates predicting employee attrition (binary label)
using a K-Nearest Neighbors classifier (scikit-learn).

Contents
- `assignment2.py` — Script that loads a tiny embedded dataset, preprocesses
  features (one-hot encodes `JobRole`, scales numeric features), trains a KNN
  classifier, evaluates on a test split, and prints a sample prediction.
- `requirements.txt` — Lists required Python packages.

How to run (Windows PowerShell)
1. (Optional but recommended) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the script:

```powershell
python assignment2.py
```

Notes
- The dataset in `assignment2.py` is intentionally tiny (10 rows) for teaching
  purposes. Evaluation metrics (perfect accuracy on the small test split) are
  not representative of real-world performance. Use a larger dataset and
  cross-validation for a realistic estimate.
- The script uses a scikit-learn Pipeline so you can easily swap the model or
  add hyperparameter tuning.

Suggested next steps
- Replace the embedded dataset with a CSV loader and add a sample dataset file.
- Add k-fold cross-validation and grid search to tune `n_neighbors`.
- Save the trained pipeline with `joblib` and add a CLI or REST endpoint for
  serving predictions.
