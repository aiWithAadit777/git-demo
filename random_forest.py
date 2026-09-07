# 1. Import the necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load a built-in sample dataset (Iris flower dataset)
iris = load_iris()
X = iris.data  # Features (sepal length, petal length, etc.)
y = iris.target  # Target labels (flower species)

# 3. Split the data into Training set (80%) and Test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the Random Forest Classifier
# n_estimators=100 means the model will use an ensemble of 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Train the model using the training data
model.fit(X_train, y_train)

# 6. Make predictions on the unseen test data
predictions = model.predict(X_test)

# 7. Evaluate how well the model performed
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
