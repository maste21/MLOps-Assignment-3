from sklearn.datasets import fetch_california_housing # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from joblib import dump                               # type: ignore

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
dump(model, 'linear_regression.joblib')

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R² score: {score:.4f}")

