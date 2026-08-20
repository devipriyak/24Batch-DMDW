import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create data
data = {
    'Age': [20, 25, 30, 35, 40],
    'Marks': [50, 60, 70, 80, 90]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Create Min-Max Scaler
scaler = MinMaxScaler()

# Apply scaling
scaled_data = scaler.fit_transform(df)

# Convert into DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("\nMin-Max Scaled Data:")
print(scaled_df)
