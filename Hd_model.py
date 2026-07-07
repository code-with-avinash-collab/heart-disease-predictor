import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load Dataset
data = pd.read_csv(r"E:\data1.csv")

# Remove 'source' column (if it exists)
if "source" in data.columns:
    data = data.drop("source", axis=1)

# Check missing values
print("Missing Values:\n")
print(data.isnull().sum())

# Remove rows having missing values
data = data.dropna()

# Encode categorical columns
le = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = le.fit_transform(data[column])

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Print feature names (optional, for checking)
print("\nTraining Features:")
print(X.columns.tolist())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "heart_disease_model.pkl")

print("✅ Model Trained Successfully!")