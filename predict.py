from joblib import load                         # type: ignore 
from sklearn.datasets import fetch_california_housing          # type: ignore 
from sklearn.model_selection import train_test_split           # type: ignore 

# Load model
model = load('linear_regression.joblib')

# Load and split data
data = fetch_california_housing()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R² score in container: {score:.4f}")




