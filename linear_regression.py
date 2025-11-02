import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Load Dataset
data = pd.read_csv(r"C:\Users\omg\Desktop\ass\student_scores - student_scores.csv")
print("First 5 rows of dataset:")
print(data.head())

X = data['Hours'].values
y = data['Scores'].values

# Normalize X for better convergence
mean_X = np.mean(X)
std_X = np.std(X)
X_norm = (X - mean_X) / std_X

#Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

#SGD Implementation 
m = 0   # slope
b = 0   # intercept
learning_rate = 0.001
epochs = 1000
n = len(X_train)

for epoch in range(epochs):
    for i in range(n):
        xi = X_train[i]
        yi = y_train[i]
        y_pred = m * xi + b
        error = y_pred - yi
        m -= learning_rate * error * xi
        b -= learning_rate * error
    
    # Monitor cost every 100 epochs
    if epoch % 100 == 0:
        cost = np.mean((m * X_train + b - y_train) ** 2)
        print(f"Epoch {epoch}: cost = {cost:.4f}")

print("\nFinal Model Parameters:")
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

#Test Accuracy 
y_test_pred = m * X_test + b
test_cost = np.mean((y_test_pred - y_test) ** 2)
print(f"\nMean Squared Error on Test Set: {test_cost:.4f}")

#User Prediction
user_input = float(input("\nEnter number of study hours to predict score: "))
user_norm = (user_input - mean_X) / std_X
predicted_score = m * user_norm + b
predicted_score = max(0, min(predicted_score, 100))   
print(f"Predicted Score: {predicted_score:.2f}")
