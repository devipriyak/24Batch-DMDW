import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Create data
data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [60, 65, 70, 72, 75, 80, 82, 85, 90, 95],
    'Marks': [45, 50, 55, 60, 65, 70, 75, 80, 88, 92]
}

# 2. Create DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)

# 3. Independent variables (X)
X = df[['Study_Hours', 'Attendance']]

# 4. Dependent variable (y)
y = df['Marks']

print("\nIndependent Variables (X):")
print(X)

print("\nDependent Variable (y):")
print(y)

# 5. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)

# 6. Create Machine Learning model
model = LinearRegression()

# 7. Train the model
model.fit(X_train, y_train)

# 8. Predict using test data
y_pred = model.predict(X_test)

print("\nActual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(y_pred)

# 9. Calculate error
error = mean_absolute_error(y_test, y_pred)

print("\nMean Absolute Error:", error)
