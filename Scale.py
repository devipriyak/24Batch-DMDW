import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create dataset
data = {
    'Age': [20, 25, 30, 35, 40],
    'Salary': [20000, 40000, 60000, 80000, 100000],
    'Marks': [60, 70, 80, 90, 95]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Create scaler
scaler = StandardScaler()

# Apply feature scaling
scaled_data = scaler.fit_transform(df)

# Convert scaled data back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("\nScaled Data:")
print(scaled_df)
