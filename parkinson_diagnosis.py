
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data_path = "parkinson_diagnosis_data.csv"
df = pd.read_csv(data_path)

# Splitting the data into features and target
X = df[["TremorAmplitude", "VoicePitch", "MotorCoordination", "ReactionTime"]]
y = df["Diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the SVM model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, predictions))

# Saving predictions to a CSV file
X_test["Actual"] = y_test.values
X_test["Predicted"] = predictions
X_test.to_csv("parkinson_predictions.csv", index=False)
print("Predictions saved to parkinson_predictions.csv")
