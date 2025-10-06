# Assignment 1 — Multivariate Linear Regression (Advertising -> Sales)

This small project demonstrates training a multivariate linear regression model
that predicts product sales from advertising budgets across three channels:
TV, Radio, and Newspaper.

Contents
- `assignment1.py` — The script that loads the dataset, trains a closed-form
  linear regression model (normal equation), evaluates it, and prints results.
- `requirements.txt` — Python package requirements.

How to run
1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
python -m pip install -r requirements.txt
```

3. Run:

```powershell
python assignment1.py
```

Notes
- The dataset is embedded in `assignment1.py` and is small; this is intended as
  an educational example.
- The model is trained using a closed-form solution (normal equation). For
  larger datasets, prefer iterative solvers (e.g. scikit-learn's LinearRegression
  or SGDRegressor).
