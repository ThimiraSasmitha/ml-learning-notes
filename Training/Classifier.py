from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load a standard machine learning dataset
print("Loading dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data: 80% for training the model, 20% for testing its performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# 3. Initialize the Machine Learning model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)

# 4. Train (fit) the model using the training data
print("Training model...")
model.fit(X_train, y_train)

# 5. Make predictions on unseen test data
y_pred = model.predict(X_test)

# 6. Evaluate how well the model learned
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
