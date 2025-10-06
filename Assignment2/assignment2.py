"""
KNN classification example for employee attrition.

Features: Age, JobRole, MonthlyIncome, JobSatisfaction, YearsAtCompany
Target: Attrition (1 = left, 0 = stayed)

This script requires scikit-learn and pandas. It encodes the categorical
JobRole, scales numeric features, trains a KNN classifier, evaluates it, and
prints metrics and a sample prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_dataset() -> pd.DataFrame:
	data = """
	Age,JobRole,MonthlyIncome,JobSatisfaction,YearsAtCompany,Attrition
	29,Sales Executive,4800,3,4,1
	35,Research Scientist,6000,4,8,0
	40,Laboratory Technician,3400,2,6,0
	28,Sales Executive,4300,3,3,1
	45,Manager,11000,4,15,0
	25,Research Scientist,3500,1,2,1
	50,Manager,12000,4,20,0
	30,Sales Executive,5000,2,5,0
	37,Laboratory Technician,3100,2,9,0
	26,Research Scientist,4500,3,2,1
	"""
	# parse using pandas
	from io import StringIO

	df = pd.read_csv(StringIO(data.strip()))
	# clean whitespace in JobRole
	df['JobRole'] = df['JobRole'].str.strip()
	return df


def build_pipeline() -> Pipeline:
	numeric_features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany']
	categorical_features = ['JobRole']

	numeric_transformer = StandardScaler()
	# use sparse_output for newer scikit-learn versions
	try:
		categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
	except TypeError:
		# fallback for older scikit-learn that expects 'sparse'
		categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

	preprocessor = ColumnTransformer(
		transformers=[
			('num', numeric_transformer, numeric_features),
			('cat', categorical_transformer, categorical_features),
		]
	)

	pipeline = Pipeline(steps=[
		('pre', preprocessor),
		('knn', KNeighborsClassifier(n_neighbors=3))
	])
	return pipeline


def main() -> None:
	df = load_dataset()
	X = df.drop(columns=['Attrition'])
	y = df['Attrition']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	pipeline = build_pipeline()
	pipeline.fit(X_train, y_train)

	y_pred = pipeline.predict(X_test)

	print("Accuracy on test set:", accuracy_score(y_test, y_pred))
	print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
	print("Classification report:\n", classification_report(y_test, y_pred))

	# Example prediction
	sample = pd.DataFrame([{
		'Age': 32,
		'JobRole': 'Research Scientist',
		'MonthlyIncome': 5200,
		'JobSatisfaction': 3,
		'YearsAtCompany': 4
	}])
	pred = pipeline.predict(sample)[0]
	print(f"Sample prediction (1=attrition,0=no): {pred}")


if __name__ == '__main__':
	main()

